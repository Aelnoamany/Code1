"""Find tables and figures on every page with Gemini and save each as its own image.

Usage:
    python crop.py                # all pages of all PDFs in ./pdfs
    python crop.py --pages 34 35  # just these book pages (re-does them)

For each book page, output/crops/page_0034.json lists what was found and the
image files, e.g. output/crops/page_0034_1.jpg. Finished pages are skipped on
re-runs, so the script can be stopped and started again.
"""

import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import pymupdf
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel

from ocr import PDF_DIR, ROTATIONS, first_page_number, render_page

ROOT = Path(__file__).parent
CROPS_DIR = ROOT / "output" / "crops"
DPI = 200
PADDING = 0.025  # extra margin around each box, as a fraction of the page size

PROMPT = """Find every table, figure, diagram, chart, drawing, or photo on this scanned page.
Do NOT include plain text paragraphs, headings, lists, page numbers, or stamps.

For each item return:
- type: "table" or "figure"
- caption: its title exactly as printed (e.g. "جدول (٢-و) ..." or "شكل (٣) ..."). If it has
  no printed title, write a short Arabic description of what it shows.
- box_2d: [ymin, xmin, ymax, xmax] normalized to 0-1000, covering the whole item including
  its title and notes directly attached to it.

If a table fills the whole page, return one box covering it. Return [] if there are none."""


class Item(BaseModel):
    type: Literal["table", "figure"]
    caption: str
    box_2d: list[int]


def page_json(page: int) -> Path:
    return CROPS_DIR / f"page_{page:04d}.json"


def detect(client, model, pdf: Path, index: int, retries: int = 5) -> tuple[list[Item], int]:
    """Return the items found and the rotation of the image they were found on."""
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=list[Item],
        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
        thinking_config=types.ThinkingConfig(thinking_level="low"),
    )
    for attempt in range(1, retries + 1):
        rotate = ROTATIONS[(attempt - 1) % len(ROTATIONS)]
        image = render_page(pdf, index, DPI, rotate)
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=image, mime_type="image/jpeg"), PROMPT],
                config=config,
            )
            if resp.parsed is None:
                raise RuntimeError("empty response")
            return resp.parsed, rotate
        except Exception as e:
            if attempt == retries:
                raise
            wait = 2 ** attempt
            print(f"  ! {pdf.name} p{index + 1}: {e} — retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)


def save_crops(
    page: int, pdf: Path, index: int, items: list[Item], rotate: int, out_dir: Path = CROPS_DIR
) -> list[dict]:
    image = Image.open(io.BytesIO(render_page(pdf, index, DPI, rotate)))
    w, h = image.size
    saved = []
    for n, item in enumerate(items, 1):
        if len(item.box_2d) != 4:
            continue
        ymin, xmin, ymax, xmax = (v / 1000 for v in item.box_2d)
        box = (
            int(max(0, xmin - PADDING) * w),
            int(max(0, ymin - PADDING) * h),
            int(min(1, xmax + PADDING) * w),
            int(min(1, ymax + PADDING) * h),
        )
        if box[2] - box[0] < 50 or box[3] - box[1] < 50:
            continue  # too small to be a real table or figure
        name = f"page_{page:04d}_{n}.jpg"
        image.crop(box).convert("RGB").save(out_dir / name, quality=88)
        saved.append({"file": name, "type": item.type, "caption": item.caption.strip()})
    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pages", type=int, nargs="*", help="book page numbers to (re)do")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    if not os.getenv("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY is not set. Put it in the .env file (see .env.example).")
    model = os.getenv("GEMINI_OCR_MODEL", "gemini-flash-latest")
    client = genai.Client()
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    # Book page number → (PDF, index in that PDF)
    pages = {}
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        start = first_page_number(pdf)
        with pymupdf.open(pdf) as doc:
            for i in range(doc.page_count):
                pages[start + i] = (pdf, i)

    if args.pages:
        todo = [p for p in args.pages if p in pages]
    else:
        todo = [p for p in sorted(pages) if not page_json(p).exists()]
    print(f"Model: {model} | {len(pages)} pages, {len(todo)} to do")

    failed, found = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(detect, client, model, *pages[p]): p for p in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            p = futures[fut]
            try:
                items, rotate = fut.result()
                for old in CROPS_DIR.glob(f"page_{p:04d}_*.jpg"):
                    old.unlink()
                saved = save_crops(p, *pages[p], items, rotate)
                page_json(p).write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")
                found += len(saved)
                summary = ", ".join(f"{s['type']}: {s['caption'][:40]}" for s in saved) or "none"
                print(f"[{n}/{len(todo)}] page {p} ✓ {summary}")
            except Exception as e:
                failed.append(p)
                print(f"[{n}/{len(todo)}] page {p} ✗ {e}", file=sys.stderr)

    print(f"\n{found} tables/figures saved in {CROPS_DIR}")
    if failed:
        print(f"{len(failed)} pages failed — run again to retry: {failed}")


if __name__ == "__main__":
    main()
