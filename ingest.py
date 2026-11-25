"""Turn an uploaded document into text and cropped tables/figures, in the background.

Usage (app.py starts this for every upload):
    python ingest.py library/D1

The folder holds source.pdf and meta.json. Each page's text goes to pages/ and its
tables/figures to crops/ as soon as they are done; meta.json shows the progress.
When every page is done, they are joined into text.md and the status becomes
"ready". Safe to re-run: finished pages are skipped.
"""

import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from google import genai

from crop import detect, save_crops
from ocr import ocr_page

ROOT = Path(__file__).parent
DPI = 200
WORKERS = 8


def update(folder: Path, **changes) -> None:
    """Change fields in meta.json; written via a temp file so the app never reads half a file."""
    path = folder / "meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta.update(changes)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def text_file(folder: Path, page: int) -> Path:
    return folder / "pages" / f"page_{page:04d}.md"


def crops_file(folder: Path, page: int) -> Path:
    return folder / "crops" / f"page_{page:04d}.json"


def finished(folder: Path, page: int) -> bool:
    return text_file(folder, page).exists() and crops_file(folder, page).exists()


def process_page(client, model: str, folder: Path, page: int) -> None:
    pdf = folder / "source.pdf"
    if not text_file(folder, page).exists():
        text_file(folder, page).write_text(ocr_page(client, model, pdf, page - 1, DPI), encoding="utf-8")
    if not crops_file(folder, page).exists():
        items, rotate = detect(client, model, pdf, page - 1)
        saved = save_crops(page, pdf, page - 1, items, rotate, folder / "crops")
        crops_file(folder, page).write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    folder = Path(sys.argv[1])
    load_dotenv(ROOT / ".env")
    model = os.getenv("GEMINI_OCR_MODEL", "gemini-flash-latest")
    client = genai.Client()

    with pymupdf.open(folder / "source.pdf") as doc:
        count = doc.page_count
    (folder / "pages").mkdir(exist_ok=True)
    (folder / "crops").mkdir(exist_ok=True)
    todo = [p for p in range(1, count + 1) if not finished(folder, p)]
    done = count - len(todo)
    update(folder, status="processing", pages=count, done=done, error=None, pid=os.getpid())
    print(f"Model: {model} | {count} pages, {len(todo)} to do", flush=True)

    failed = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(process_page, client, model, folder, p): p for p in todo}
        for fut in as_completed(futures):
            try:
                fut.result()
                done += 1
                update(folder, done=done)
                print(f"page {futures[fut]} ✓", flush=True)
            except Exception as e:
                failed.append(futures[fut])
                print(f"page {futures[fut]} ✗ {e}", flush=True)

    if failed:
        update(folder, status="failed", error=f"{len(failed)} pages failed: {sorted(failed)[:10]}")
        sys.exit(1)

    name = json.loads((folder / "meta.json").read_text(encoding="utf-8"))["name"]
    parts = [f"# {name}\n"]
    for p in range(1, count + 1):
        parts += [f"\n<!-- page {p} -->\n", text_file(folder, p).read_text(encoding="utf-8")]
    (folder / "text.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
    update(folder, status="ready")
    print("Done.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        traceback.print_exc()
        update(Path(sys.argv[1]), status="failed", error=str(e)[:300])
        sys.exit(1)
