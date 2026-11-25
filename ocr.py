"""OCR scanned PDFs with Gemini Flash, one page at a time.

Usage:
    python ocr.py                 # all PDFs in ./pdfs
    python ocr.py --dpi 300       # sharper renders for faint/handwritten scans
    python ocr.py --only "file.pdf"

Each page is saved under output/pages/<pdf name>/page_0001.md as soon as it is
done, so re-running the script skips finished pages and only retries failures.
When every page of a PDF is done, they are joined into output/<pdf name>.md.
"""

import argparse
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from google import genai
from google.genai import types

ROOT = Path(__file__).parent
PDF_DIR = ROOT / "pdfs"
OUT_DIR = ROOT / "output"
PAGES_DIR = OUT_DIR / "pages"

PROMPT = """You are an expert OCR system for Arabic documents.
Transcribe ALL text on this scanned page exactly as written.

Rules:
- Keep the original language. Do not translate, summarize, or correct spelling.
- Preserve structure: headings as Markdown headings, paragraphs as paragraphs,
  lists as lists, and tables as Markdown tables (columns in right-to-left reading order).
- Handwritten text: transcribe it too. Prefix handwritten parts with [خط يد].
- Keep numbers, dates, and reference codes exactly as they appear (Arabic-Indic or Western digits).
- Stamps, signatures, logos: describe briefly in square brackets, e.g. [ختم: وزارة ...] or [توقيع].
- Unreadable words: write [غير واضح].
- If the page is blank, output only: [صفحة فارغة]
Output only the transcription, with no commentary before or after."""


def render_page(pdf_path: Path, page_index: int, dpi: int, rotate: int = 0) -> bytes:
    with pymupdf.open(pdf_path) as doc:
        page = doc[page_index]
        page.set_rotation((page.rotation + rotate) % 360)
        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("jpeg", jpg_quality=90)


# Gemini sometimes blocks or returns nothing for sideways pages (e.g. landscape
# tables); after the first two failures, retry with the page turned.
ROTATIONS = [0, 0, 270, 90, 180]


def ocr_page(client, model, pdf_path: Path, page_index: int, dpi: int, retries: int = 5) -> str:
    config = types.GenerateContentConfig(
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
        # Transcription needs little reasoning; low thinking halves the time per page
        thinking_config=types.ThinkingConfig(thinking_level="low"),
    )
    for attempt in range(1, retries + 1):
        image = render_page(pdf_path, page_index, dpi, ROTATIONS[(attempt - 1) % len(ROTATIONS)])
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=image, mime_type="image/jpeg"), PROMPT],
                config=config,
            )
            text = (resp.text or "").strip()
            if text:
                return text
            raise RuntimeError("empty response")
        except Exception as e:
            if attempt == retries:
                raise
            wait = 2 ** attempt
            print(f"  ! {pdf_path.name} p{page_index + 1}: {e} — retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)


def page_file(pdf_path: Path, page_index: int) -> Path:
    return PAGES_DIR / pdf_path.stem / f"page_{page_index + 1:04d}.md"


def first_page_number(pdf_path: Path) -> int:
    # Files split from one book are named like "Book_part02_p021-040.pdf";
    # use that range so page markers match the original book's page numbers.
    m = re.search(r"_p(\d+)-\d+$", pdf_path.stem)
    return int(m.group(1)) if m else 1


def combine(pdf_path: Path, page_count: int) -> None:
    parts = [f"# {pdf_path.name}\n"]
    start = first_page_number(pdf_path)
    for i in range(page_count):
        parts.append(f"\n<!-- page {start + i} -->\n")
        parts.append(page_file(pdf_path, i).read_text(encoding="utf-8"))
    (OUT_DIR / f"{pdf_path.stem}.md").write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4, help="pages processed in parallel")
    parser.add_argument("--only", help="process just this PDF file name")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    if not os.getenv("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY is not set. Put it in the .env file (see .env.example).")
    model = os.getenv("GEMINI_OCR_MODEL", "gemini-flash-latest")
    client = genai.Client()

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if args.only:
        pdfs = [p for p in pdfs if p.name == args.only]
    if not pdfs:
        sys.exit(f"No PDFs found in {PDF_DIR}")

    # Build the list of pages still to do, across all PDFs
    page_counts, jobs = {}, []
    for pdf in pdfs:
        with pymupdf.open(pdf) as doc:
            page_counts[pdf] = doc.page_count
        for i in range(page_counts[pdf]):
            if not page_file(pdf, i).exists():
                jobs.append((pdf, i))

    total = sum(page_counts.values())
    print(f"Model: {model} | {len(pdfs)} PDFs, {total} pages, {len(jobs)} to do")

    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(ocr_page, client, model, pdf, i, args.dpi): (pdf, i) for pdf, i in jobs}
        for n, fut in enumerate(as_completed(futures), 1):
            pdf, i = futures[fut]
            try:
                text = fut.result()
                out = page_file(pdf, i)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(text, encoding="utf-8")
                print(f"[{n}/{len(jobs)}] {pdf.name} p{i + 1} ✓")
            except Exception as e:
                failed.append((pdf.name, i + 1))
                print(f"[{n}/{len(jobs)}] {pdf.name} p{i + 1} ✗ {e}", file=sys.stderr)

    for pdf, count in page_counts.items():
        if all(page_file(pdf, i).exists() for i in range(count)):
            combine(pdf, count)

    if failed:
        print(f"\n{len(failed)} pages failed — run the script again to retry them:")
        for name, p in failed:
            print(f"  {name} page {p}")
    else:
        print(f"\nDone. Text files are in {OUT_DIR}")


if __name__ == "__main__":
    main()
