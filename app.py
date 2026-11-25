"""Chat with the OCR'd documents using Gemini's long context.

Run:  streamlit run app.py

The fire code (output/*.md) and every uploaded document that is switched on
(library/D*/text.md) are stored in a Gemini cache and read for every question.
"""

import base64
import hmac
import html
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pymupdf
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from markdown_it import MarkdownIt

from ocr import PDF_DIR, first_page_number

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "output"
IMG_DIR = OUT_DIR / "images"
CROPS_DIR = OUT_DIR / "crops"
MAX_SOURCE_PAGES = 8
MAX_OPEN_CROPS = 3  # show the first few tables/figures opened, the rest collapsed
LIBRARY_DIR = ROOT / "library"  # uploaded documents (see ingest.py)
UPLOAD_TYPES = ["pdf", "png", "jpg", "jpeg"]
# Gemini Flash reads up to ~1M tokens; keep room for the conversation
MAX_CONTEXT_TOKENS = 900_000
DETAILED_QUESTION_CHARS = 400  # longer questions usually describe a whole project
CHATS_DIR = ROOT / "chats"  # one JSON file per conversation, files attached to it in a folder
INLINE_LIMIT = 15_000_000  # bigger attachments go through the Gemini Files API
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

load_dotenv(ROOT / ".env")
MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-flash-latest")

INSTRUCTIONS = """You are an expert fire-protection consultant who knows the documents below
by heart. You help engineers, designers, and building owners understand and apply them.
The full text of every document is provided below, inside <document> tags.
Each page starts with a marker such as <!-- page N: cite as [p. N] -->.

## Faithfulness (most important)
- Every requirement, number, unit, limit, and condition must come from the documents,
  exactly as written. Never invent or round values, and never fill gaps from other codes.
- Keep two things clearly apart: what the code REQUIRES (always cited), and your own
  explanation, which may make it easier to understand but never adds requirements.
  Mark general explanations that are not stated in the code as such
  (Arabic: «توضيح:», English: "Note:").
- If the answer is not in the documents, say so plainly, and point to the closest
  related part that does exist.
- If the answer depends on facts the user did not give (occupancy type, building height,
  sprinklers or not, hazard class...), give the answer for each relevant case, or state
  briefly which information decides it.
- If the code's wording allows more than one reading (e.g. how two "cumulative" increases
  combine: added to the base, or multiplied), say so, show each result, and recommend
  confirming with the competent authority. Never silently pick one.

## Project questions: detailed design, not a summary
When the question describes a real project (floors, areas, uses, rooms, number of cars,
building height, location...), the user wants a low-level design they can build from,
not general rules. Then, in addition to the structure below:
- Work through EVERY floor and area the user listed, one by one, in its own subsection
  or table row. Never answer "each floor needs..." when you can compute it per floor.
- Classify each floor/area (occupancy group, hazard class) and say which rule set applies.
- For each, calculate everything the code lets you calculate, showing the formula, the
  code value used with its clause and page, and the result: occupant load (area ÷ m² per person); number of
  exits and stairs; required widths of doors, corridors, stairs and ramps (load × mm per
  person, then the code minimum); travel distance, dead end and common path limits;
  smoke zones (how many, their size); smoke exhaust and make-up air flow per zone and per
  floor; sprinkler hazard class, design density × area, heads (area ÷ coverage per head);
  standpipes (count them) and their flow; pump flow and pressure; tank volume; ramp
  slopes and lengths; fire resistance of every wall, door and special room; the detection,
  alarm and suppression system for each room type the user named.
- Then combine floors into building totals (governing flow, pump, tank, total exhaust).
- State every assumption you make for missing data (ceiling height, number of
  standpipes, hazard class of a room...) once, clearly, and use it consistently.
- Do not shorten or skip floors to save space: a long, complete answer is expected.
  Only fall back to general guidance for an item the code gives no numbers for, and say so.

## How to answer
Reply in the language of the documents your answer is based on, NOT the language of the
question: an English question about an Arabic document gets an Arabic answer (clear Modern
Standard Arabic). If the sources are in different languages, use the main source's language.
Use this structure, with short bold headings written ONLY in the reply language (Arabic
answer → Arabic headings only), and skip any part that does not apply.
A simple question deserves a short answer: do not pad.

1. **الإجابة المختصرة / Short answer** — 1–3 sentences that directly answer the question,
   including the key numbers.
2. **ما ينص عليه الكود / What the code says** — the exact requirements, clause numbers,
   and values. When they come from a table, show only the relevant rows and columns as a
   Markdown table (original Arabic headings). Quote exact wording when it matters.
3. **الشرح / Explanation** — what this means in practice, in plain words. Define
   technical terms (e.g. «حيز الحريق», «مسلك الهروب») using the code's own definitions.
4. **الشروط والاستثناءات / Conditions & exceptions** — footnotes and table notes (ج، د…),
   allowed increases (e.g. with sprinklers), and cases where the rule does not apply.
   Users often miss these, so always look for them.
5. **مثال تطبيقي / Worked example** — only when numbers or a calculation are involved:
   a short realistic example using the code's values, step by step.
6. **بنود ذات صلة / Related requirements** — other clauses or tables in the documents
   the user should also check, with their pages.
7. **ملخص الحسابات / Calculation summary** — whenever the answer calculates anything
   (flows, pressures, volumes, quantities, widths, numbers of exits...), end with ONE
   Markdown table of every final result: item | result with unit | basis or assumption |
   clause | page. Only final figures, no steps, and the same values as in the text above.

## Formatting and citations
- After each requirement or value, give the clause number it comes from, as printed in
  the code, then its page in exactly this form: [p. N] (e.g. [p. 34]), in any reply
  language. One page per brackets. Examples: «البند (٤-٤-٣-١) [p. 132]»,
  «جدول (٦-٢-١) [p. 177]», "clause 4-4-3-1 [p. 132]". Use the most specific number
  printed above the text (sub-clause, item letter, table or table note), the nearest
  one on the same or an earlier page. If the text has no clause or table number, give
  the page only; never invent a clause number.
- Page numbers ALWAYS come from the nearest page marker above the text you use, which
  spells out the exact citation (<!-- page 184: cite as [D3 p. 184] -->); copy it.
  Never use a page number printed in the page text (headers and footers such as
  «٤-٤٢» or «٥٣» are printed numbers: ignore them).
- Documents with an id attribute (e.g. <document name="x.pdf" id="D2">) were uploaded
  by the user. Cite their pages as [D2 p. N] and their images as [fig:D2:5_1].
  The fire code documents have no id: cite them as [p. N] only. Use ONLY the ids listed
  in the document catalog below; never invent one, and never put an id on a fire code
  page.
- Mention tables and figures by their number as printed, e.g. «جدول (٥-١)» or «شكل (٣-٤)».
- Never use LaTeX: no $ or $$ signs and no commands like \\frac, \\times, _{ }, ^{ }.
  Write formulas and calculations as plain text with ×, ÷, ( ) and ^, one step per line,
  e.g. «W = (V ÷ S) × C ÷ (100 − C)» and «٧ كجم + ٧ كجم (رشاشات) = ١٤ كجم».
- In Arabic answers write all numbers with Arabic-Indic digits (٠١٢٣٤٥٦٧٨٩), as the code
  does; in English answers use Western digits. Keep clause numbers as printed.
  Exception: inside [p. N] and [fig:…] tags always use Western digits.
- If documents disagree, show both versions with their sources.
- The text came from OCR; [غير واضح] marks unreadable words and [خط يد] marks handwriting.
  If a value you need is marked unreadable, say so and suggest checking the original page.
- Pages list their tables and figures as <!-- image fig:154_1 (figure): caption -->.
  Whenever your answer uses data from one of these tables or figures, or one directly
  shows what the answer is about, you MUST put its tag exactly as [fig:154_1] right after
  the related text, so the user sees the original image. Tag only the ones you actually
  used (at most 4); never tag one just because it is on a cited page."""


@st.cache_resource
def get_client():
    return genai.Client()


# The model keeps no memory between requests, so the book must be part of every
# one. An explicit cache stores the processed book on Google's side for an hour,
# and each question just points at it: first words in ~2s instead of ~17s.
# Streamlit recreates it after 50 minutes, before Google's copy expires.
@st.cache_resource(ttl=50 * 60)
def get_book_cache(_client, _system_instruction: str, corpus_key: tuple):
    return _client.caches.create(
        model=MODEL,
        config=types.CreateCachedContentConfig(system_instruction=_system_instruction, ttl="3600s"),
    )


# ---- Uploaded documents -------------------------------------------------------
# Each upload gets a folder library/D1, library/D2… with its own page numbers;
# ingest.py reads it in the background. The fire code (output/*.md) has no id.


def doc_folder(doc: str) -> Path:
    return LIBRARY_DIR / doc


def read_meta(doc: str) -> dict | None:
    try:
        return json.loads((doc_folder(doc) / "meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_meta(doc: str, **changes) -> None:
    meta = read_meta(doc) or {}
    meta.update(changes)
    path = doc_folder(doc) / "meta.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def pid_alive(pid: int | None) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, TypeError):
        return False


def library_docs() -> list[dict]:
    """Uploaded documents, oldest first, with whether they are switched on."""
    docs = []
    for path in LIBRARY_DIR.glob("D*/meta.json"):
        meta = read_meta(path.parent.name)
        if not meta:
            continue
        meta["enabled"] = not (path.parent / "disabled").exists()
        if meta["status"] == "processing" and not pid_alive(meta.get("pid")):
            meta["status"] = "stopped"  # e.g. the Mac restarted while it was reading
        docs.append(meta)
    return sorted(docs, key=lambda m: int(m["id"][1:]))


def start_ingest(doc: str) -> None:
    """Run ingest.py detached, so it keeps going whatever the page does."""
    write_meta(doc, status="processing", error=None)
    out = subprocess.run(
        ["/bin/sh", "-c", 'nohup "$0" ingest.py "$1" > "$1/log.txt" 2>&1 & echo $!',
         sys.executable, str(doc_folder(doc))],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    write_meta(doc, pid=int(out.stdout.strip()))


def add_document(name: str, data: bytes) -> str | None:
    """Store an uploaded PDF or image and start reading it. Returns an error, if any."""
    try:
        with pymupdf.open(stream=data, filetype=Path(name).suffix[1:].lower()) as doc:
            pdf = data if doc.is_pdf else doc.convert_to_pdf()
            pages = doc.page_count
    except Exception:
        return f"{name}: not a readable PDF or image"
    LIBRARY_DIR.mkdir(exist_ok=True)
    counter = LIBRARY_DIR / "next_id"  # ids are never reused, so old chats keep pointing right
    n = int(counter.read_text()) if counter.exists() else 1
    counter.write_text(str(n + 1))
    doc = f"D{n}"
    doc_folder(doc).mkdir()
    (doc_folder(doc) / "source.pdf").write_bytes(pdf)
    write_meta(doc, id=doc, name=name, added=time.time(), pages=pages, done=0, status="processing")
    start_ingest(doc)
    return None


def set_enabled(doc: str) -> None:
    flag = doc_folder(doc) / "disabled"
    if st.session_state[f"use-{doc}"]:
        flag.unlink(missing_ok=True)
    else:
        flag.touch()


def delete_document(doc: str) -> None:
    meta = read_meta(doc) or {}
    if meta.get("status") == "processing" and pid_alive(meta.get("pid")):
        os.kill(meta["pid"], signal.SIGTERM)
    shutil.rmtree(doc_folder(doc), ignore_errors=True)


# ---- The text sent to the model -------------------------------------------------


def with_images(text: str, doc: str) -> str:
    """Spell out how to cite each page in its marker (the model otherwise sometimes cites
    the page numbers printed in headers and footers), then list the page's cropped
    tables/figures."""
    prefix = f"{doc}:" if doc else ""

    def add(match: re.Match) -> str:
        page = int(match.group(1))
        cite = f"[{doc} p. {page}]" if doc else f"[p. {page}]"
        lines = [f"<!-- page {page}: cite as {cite} -->"]
        for c in page_crops(doc, page):
            n = c["file"].rsplit("_", 1)[1][:-4]
            lines.append(f"<!-- image fig:{prefix}{page}_{n} ({c['type']}): {c['caption']} -->")
        return "\n".join(lines)

    return re.sub(r"<!-- page (\d+) -->", add, text)


@st.cache_data
def load_corpus(key: tuple, used: tuple) -> str:
    """`key` is only there so Streamlit rebuilds the text when files change."""
    catalog = ["Document catalog:", "- The fire code (FireCode_part…): no id, cite as [p. N]."]
    catalog += [f"- {doc}: {name}, cite as [{doc} p. N]." for doc, name in used]
    docs = ["\n".join(catalog)]
    for f in sorted(OUT_DIR.glob("*.md")):
        text = with_images(f.read_text(encoding="utf-8"), "")
        docs.append(f'<document name="{f.stem}.pdf">\n{text}\n</document>')
    for doc, name in used:
        text = with_images((doc_folder(doc) / "text.md").read_text(encoding="utf-8"), doc)
        docs.append(f'<document name="{html.escape(name)}" id="{doc}">\n{text}\n</document>')
    return "\n\n".join(docs)


@st.cache_data(show_spinner=False)
def corpus_tokens(_corpus: str, key: tuple) -> int:
    return get_client().models.count_tokens(model=MODEL, contents=_corpus).total_tokens


# ---- Pages, tables and figures behind an answer -----------------------------------


@st.cache_data
def page_index() -> dict[int, tuple[str, int]]:
    """Fire code page number → (PDF file name, 0-based page index in that file)."""
    index = {}
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        start = first_page_number(pdf)
        with pymupdf.open(pdf) as doc:
            for i in range(doc.page_count):
                index[start + i] = (pdf.name, i)
    return index


def page_source(doc: str, page: int) -> tuple[Path, int, str] | None:
    """Where a page is: (PDF, 0-based index, name to show), or None if it doesn't exist."""
    if not doc:
        if page not in page_index():
            return None
        name, i = page_index()[page]
        return PDF_DIR / name, i, name
    meta = read_meta(doc)
    if not meta or not 1 <= page <= meta.get("pages", 0):
        return None
    return doc_folder(doc) / "source.pdf", page - 1, f"{doc} · {meta['name']}"


def page_image(doc: str, page: int) -> Path | None:
    """Render a cited page to an image once, then reuse the file."""
    source = page_source(doc, page)
    if not source:
        return None
    folder = doc_folder(doc) / "images" if doc else IMG_DIR
    path = folder / f"page_{page:04d}.jpg"
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        with pymupdf.open(source[0]) as pdf:
            pdf[source[1]].get_pixmap(dpi=150).save(path, jpg_quality=85)
    return path


def crops_dir(doc: str) -> Path:
    return doc_folder(doc) / "crops" if doc else CROPS_DIR


def page_crops(doc: str, page: int) -> list[dict]:
    """Tables and figures cut out of this page by crop.py / ingest.py, if any."""
    path = crops_dir(doc) / f"page_{page:04d}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
# [p. 34], [p. 33, 34], [p. 33-35], for uploads [D2 p. 5], also after a clause
# number inside the same brackets: [البند ٣/١، D2 p. 5]
CITATION = re.compile(r"(?:\b(D\d+)\s*,?\s*)?\bpp?\.\s*([\d\s,،\-–]+)(?=\])")
FIG_TAG = re.compile(r"\[fig:(?:(D\d+):)?(\d+)_(\d+)\]")
NUMBERED = re.compile(r"(جدول|شكل)\s*(?:رقم)?\s*\(([^)]{1,20})\)")


def parse_refs(groups: list[tuple[str, str]]) -> set[tuple[str, int]]:
    """(document id, listed pages) pairs from CITATION → (document id, page) pairs."""
    pages = set()
    for doc, listed in groups:
        for part in re.split(r"[,،]", listed):
            ends = [int(n) for n in re.findall(r"\d+", part)]
            if len(ends) == 2 and 0 < ends[1] - ends[0] <= 5:
                pages.update((doc, p) for p in range(ends[0], ends[1] + 1))
            else:
                pages.update((doc, p) for p in ends)
    return pages


def cited_pages(answer: str) -> list[tuple[str, int]]:
    """(document id, page) for every page the answer cites; "" is the fire code."""
    answer = answer.translate(DIGITS)
    groups = CITATION.findall(answer) + [("", n) for n in re.findall(r"صفحة\s*(\d+)", answer)]
    return [ref for ref in sorted(parse_refs(groups)) if page_source(*ref)][:MAX_SOURCE_PAGES]


# ---- Number check: is each cited value really on the page it cites? ----------------

NUMBER = re.compile(r"(?<![\d.,٫\-/])\d+(?:[.,٫٬]\d+)*(?![\d\-/])")
COMPUTED = re.compile(r"[×÷*+]|\s[-−]\s")  # a line with arithmetic: numbers after "=" are results


def number_keys(token: str) -> set[str]:
    """The values a written number can mean: "٤,٥" is 4.5 in the code, but "٥٠,٠٠٠" may
    be 50000, so a comma is tried both as a decimal point and as a thousands separator."""
    token = token.translate(DIGITS).replace("٫", ",").replace("٬", ",")
    keys = set()
    for text in {token.replace(",", "."), token.replace(",", "")}:
        if text.count(".") <= 1:
            keys.add(f"{float(text):g}")
    return keys


@st.cache_data
def page_numbers(doc: str, mtime: float) -> dict[int, set[str]]:
    """Every number written on each page of a document, as number_keys()."""
    files = [doc_folder(doc) / "text.md"] if doc else sorted(OUT_DIR.glob("*.md"))
    text = "".join(f.read_text(encoding="utf-8") for f in files)
    parts = re.split(r"<!-- page (\d+) -->", text)
    return {
        int(parts[i]): set().union(*(number_keys(t) for t in NUMBER.findall(parts[i + 1])))
        for i in range(1, len(parts), 2)
    }


def numbers_on(doc: str, page: int) -> set[str] | None:
    files = [doc_folder(doc) / "text.md"] if doc else list(OUT_DIR.glob("*.md"))
    if not all(f.exists() for f in files):
        return None
    return page_numbers(doc, max(f.stat().st_mtime for f in files)).get(page)


@st.cache_data(show_spinner=False)
def unverified_numbers(answer: str, question: str) -> list[dict]:
    """Numbers the answer puts right before a citation that are not on the cited pages.
    Results of the answer's own arithmetic and numbers from the question are skipped."""
    given = set().union(*(number_keys(t) for t in NUMBER.findall(question)))
    flagged = []
    for line in plain_math(FIG_TAG.sub("", answer)).splitlines():
        start = 0
        for bracket in re.finditer(r"\[[^\[\]]*\]", line):
            refs = parse_refs(CITATION.findall(bracket.group(0).translate(DIGITS)))
            segment = line[start:bracket.start()]
            start = bracket.end()
            pages = {ref: numbers_on(*ref) for ref in refs}
            pages = {ref: nums for ref, nums in pages.items() if nums is not None}
            if not pages:
                continue  # not a citation, or a page that doesn't exist (shown elsewhere)
            on_pages = set().union(*pages.values())
            segment = re.split(r"[؛;]|\.\s", segment)[-1]  # only the claim this citation ends
            arithmetic = COMPUTED.search(segment)
            results_from = segment.find("=", arithmetic.start()) if arithmetic else -1
            cited = ", ".join(f"{doc + ' ' if doc else ''}p. {p}" for doc, p in sorted(pages))
            for m in NUMBER.finditer(segment):
                keys = number_keys(m.group(0))
                before, after = segment[:m.start()].rstrip(), segment[m.end():]
                if (keys & on_pages or keys & given or 0 <= results_from < m.start()
                        or before.endswith(("≈", "~", "حوالي", "نحو"))  # the answer's own estimate
                        or before.endswith("(") and NUMBER.search(before[-40:])  # a unit conversion
                        # in arithmetic, quantities are the project's; only rates (m²/person,
                        # L/s/m², mm/person...) come from the code
                        or arithmetic and "/" not in re.split(r"[×÷*+=()]", after, maxsplit=1)[0]):
                    continue
                context = re.sub(r"[*#|>]+", " ", segment).strip()
                flagged.append({"number": m.group(0), "cited": cited, "context": context[-140:]})
    return list({(f["number"], f["cited"]): f for f in flagged}.values())


def show_number_check(answer: str, question: str, rtl: bool) -> None:
    flagged = unverified_numbers(answer, question)
    if not flagged:
        return
    label = (f"⚠️ {len(flagged)} رقم غير موجود في الصفحة المذكورة — راجعه" if rtl
             else f"⚠️ {len(flagged)} numbers not found on the page they cite — check them")
    with st.expander(label):
        st.caption("قد يكون الرقم محسوبًا أو محوّلًا من قيمة في الكود، أو خطأً في القراءة الضوئية للصفحة."
                   if rtl else "A number may be derived or converted from a code value, or misread "
                   "by OCR on the page, so this is a list to check, not a list of errors.")
        head = "| الرقم | الصفحة المذكورة | السياق |" if rtl else "| Number | Cited page | Context |"
        rows = [head, "| :---: | :---: | :--- |"]
        for f in flagged:
            cited = f["cited"].replace("p. ", "ص ") if rtl else f["cited"]
            rows.append(f"| **{f['number']}** | {cited} | {f['context']} |")
        st.markdown("\n".join(rows))


def numbered_refs(text: str) -> set[tuple[str, str]]:
    """Table/figure numbers in a text, e.g. 'جدول (٥-١)' → ('جدول', '5-1')."""
    refs = set()
    for kind, num in NUMBERED.findall(text.translate(DIGITS)):
        refs.add((kind, re.sub(r"[^\w-]", "", num.replace("ـ", "")).replace("_", "")))
    return refs


def cited_figures(answer: str, pages: list[tuple[str, int]]) -> list[tuple[str, int, dict]]:
    """Tables/figures to show: ones the model tagged, plus ones on cited pages whose
    number (e.g. جدول (٥-١)) the answer mentions. The model often skips the tags."""
    found = []
    for doc, page, n in dict.fromkeys(FIG_TAG.findall(answer)):
        name = f"page_{int(page):04d}_{n}.jpg"
        found += [(doc, int(page), c) for c in page_crops(doc, int(page)) if c["file"] == name]
    mentioned = numbered_refs(answer)
    for doc, p in pages:
        for c in page_crops(doc, p):
            if (doc, p, c) not in found and numbered_refs(c["caption"]) & mentioned:
                found.append((doc, p, c))
    return found


def as_refs(pages: list) -> list[tuple[str, int]]:
    """Chats saved before uploads existed store fire-code pages as plain numbers."""
    return [("", p) if isinstance(p, int) else tuple(p) for p in pages]


def as_figures(figures: list) -> list[tuple[str, int, dict]]:
    return [("", *f) if len(f) == 2 else tuple(f) for f in figures]


def is_arabic(text: str) -> bool:
    return len(re.findall(r"[؀-ۿ]", text)) > len(re.findall(r"[A-Za-z]", text))


MATH = re.compile(r"\$\$(.+?)\$\$|\$([^$\n]+?)\$", re.S)
TEX_SYMBOLS = {  # longer codes first: \left before \le
    r"\left": "", r"\right": "", r"\times": "×", r"\div": "÷", r"\cdot": "·", r"\geq": "≥", r"\ge": "≥", r"\leq": "≤",
    r"\le": "≤", r"\approx": "≈", r"\neq": "≠", r"\pm": "±", r"^{\circ}": "°", r"^\circ": "°",
    r"\circ": "°", r"\%": "%", r"\,": " ", r"\;": " ",
    "^{2}": "²", "^2": "²", "^{3}": "³", "^3": "³", "^{٢}": "²", "^٢": "²", "^{٣}": "³", "^٣": "³",
}


def plain_math(text: str) -> str:
    """The model sometimes writes formulas in LaTeX despite the instructions. Streamlit
    would show them, but copies and PDFs would not, so turn them into plain text."""
    def convert(m: re.Match) -> str:
        tex = m.group(1) if m.group(1) is not None else m.group(2)
        if m.group(1) is None and not re.search(r"[\\^_=]|^\s*[A-Za-z]\s*$", tex):
            return m.group(0)  # a real dollar amount, e.g. "$40 to $50"
        while True:
            new = re.sub(r"\\d?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1) / (\2)", tex)
            new = re.sub(r"\\sqrt\{([^{}]*)\}", r"√(\1)", new)
            new = re.sub(r"\\(?:text|mathbf|mathrm|operatorname|textbf)\{([^{}]*)\}", r"\1", new)
            if new == tex:
                break
            tex = new
        for code, symbol in TEX_SYMBOLS.items():
            tex = tex.replace(code, symbol)
        tex = re.sub(r"[_^]\{([^{}]*)\}", lambda s: ("_" if s.group(0)[0] == "_" else "^") + s.group(1), tex)
        tex = re.sub(r"\(([A-Za-z0-9.]+)\)(?= /)|(?<=/ )\(([A-Za-z0-9.]+)\)", r"\1\2", tex)
        tex = re.sub(r"\s+", " ", tex.replace("{", "").replace("}", "")).strip()
        return f"\n\n{tex}\n\n" if m.group(1) is not None else tex

    return MATH.sub(convert, text)


def tidy(answer: str, rtl: bool = False) -> str:
    """Swap the model's [fig:…] tags for a small marker when showing an answer;
    in Arabic answers also show [p. 34] as [ص 34] so it reads naturally."""
    answer = plain_math(FIG_TAG.sub(" 🖼️", answer))
    return re.sub(r"\b(D\d+\s*,?\s*)?\bpp?\.\s*(?=[\d٠-٩])", r"\1ص ", answer) if rtl else answer


def message_box(i: int, rtl: bool):
    """A container that lays out a whole message right-to-left (text, lists, tables,
    expander labels) when it is Arabic. The key becomes a CSS class, styled in main()."""
    return st.container(key=f"msg-{i}-{'rtl' if rtl else 'ltr'}")


def show_sources(pages: list, figures: list = (), rtl: bool = False) -> None:
    page_word = "صفحة" if rtl else "page"
    figures = [f for f in as_figures(figures) if (crops_dir(f[0]) / f[2]["file"]).exists()]
    if figures:
        st.caption("📊 الجداول والأشكال — الصور الأصلية" if rtl else "📊 Tables & figures — the original images")
        for n, (doc, p, c) in enumerate(figures):
            icon = "📊" if c["type"] == "table" else "🖼️"
            where = f"{doc} · " if doc else ""
            with st.expander(f"{icon} {c['caption']}  ·  {where}{page_word} {p}", expanded=n < MAX_OPEN_CROPS):
                st.image(str(crops_dir(doc) / c["file"]), width="stretch")
    pages = [ref for ref in as_refs(pages) if page_source(*ref)]  # skips deleted documents
    if not pages:
        return
    st.caption("📎 الصفحات المصدر — اضغط لرؤية الصفحة الأصلية" if rtl else "📎 Source pages — click to see the original scan")
    for doc, p in pages:
        with st.expander(f"📄 {page_word.capitalize()} {p}  ·  {page_source(doc, p)[2]}"):
            st.image(str(page_image(doc, p)), width="stretch")


EXPORT_CSS = """
body { font-family: -apple-system, "SF Arabic", "Geeza Pro", Arial, sans-serif;
       font-size: 11.5pt; line-height: 1.7; color: #111; margin: 0; }
.question { background: #f2f4f7; border-radius: 8px; padding: 10px 14px; margin-bottom: 16px; }
.meta { color: #777; font-size: 9pt; margin-bottom: 4px; }
table { border-collapse: collapse; margin: 8px 0; }
th, td { border: 1px solid #999; padding: 4px 8px; }
th { background: #eee; }
figure { margin: 16px 0; break-inside: avoid; }
figure img { max-width: 100%; max-height: 22cm; border: 1px solid #ddd; }
figcaption { color: #555; font-size: 9.5pt; }
@page { size: A4; margin: 18mm 16mm; }
"""


# Same Markdown rules as Streamlit's chat (GitHub style), so exports look like the screen
MARKDOWN = MarkdownIt("commonmark", {"breaks": True}).enable("table")


def answer_html(answer: str, rtl: bool) -> str:
    """The answer as HTML (tables kept) for copying into Word, email, or a PDF."""
    body = MARKDOWN.render(tidy(FIG_TAG.sub("", answer), rtl))
    return f'<div dir="{"rtl" if rtl else "ltr"}">{body}</div>'


def answer_text(answer: str, rtl: bool) -> str:
    return re.sub(r"\*\*|__", "", tidy(FIG_TAG.sub("", answer), rtl))


@st.cache_data(show_spinner=False, max_entries=50)
def answer_pdf(question: str, answer: str, figures: list, rtl: bool) -> bytes:
    """Print the question, answer, and original table/figure images to PDF with
    headless Chrome, which lays out Arabic and right-to-left tables correctly."""
    page_word = "صفحة" if rtl else "page"
    images = "".join(
        f'<figure><img src="data:image/jpeg;base64,'
        f'{base64.b64encode((crops_dir(d) / c["file"]).read_bytes()).decode()}">'
        f"<figcaption>{html.escape(c['caption'])} · {d + ' · ' if d else ''}{page_word} {p}</figcaption></figure>"
        for d, p, c in as_figures(figures)
        if (crops_dir(d) / c["file"]).exists()
    )
    title = "الكود المصري لحماية المنشآت من أخطار الحريق" if rtl else "Egyptian Fire Code"
    doc = f"""<!doctype html><html dir="{'rtl' if rtl else 'ltr'}"><head><meta charset="utf-8">
<style>{EXPORT_CSS}</style></head><body>
<div class="meta">{title}</div>
<div class="question"><b>{"السؤال" if rtl else "Question"}:</b> {html.escape(question)}</div>
{answer_html(answer, rtl)}{images}</body></html>"""
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "answer.html", Path(tmp) / "answer.pdf"
        src.write_text(doc, encoding="utf-8")
        subprocess.run(
            [str(CHROME), "--headless", "--disable-gpu", "--no-pdf-header-footer",
             f"--user-data-dir={tmp}/profile", f"--print-to-pdf={out}", src.as_uri()],
            capture_output=True, timeout=60, check=True,
        )
        return out.read_bytes()


def copy_button(i: int, answer: str, rtl: bool) -> None:
    """One click copies the answer with its formatting (tables, bold) for Word or
    email; apps that only take plain text get the plain version."""
    label, done = ("📋 نسخ الإجابة", "✓ تم النسخ") if rtl else ("📋 Copy answer", "✓ Copied")
    st.iframe(
        f"""<button id="b">{label}</button><div id="rich" contenteditable></div>
<style>
body {{ margin: 0; }}
#b {{ font: 14px "Source Sans Pro", -apple-system, sans-serif; padding: 6px 12px; cursor: pointer;
      border: 1px solid rgba(49,51,63,.2); border-radius: 8px; background: transparent; color: #31333f; }}
#b:hover {{ border-color: #ff4b4b; color: #ff4b4b; }}
@media (prefers-color-scheme: dark) {{ #b {{ color: #fafafa; border-color: rgba(250,250,250,.2); }} }}
#rich {{ position: fixed; left: -9999px; }}
</style>
<script>
const rich = {json.dumps(answer_html(answer, rtl))}, plain = {json.dumps(answer_text(answer, rtl))};
const b = document.getElementById("b");
function oldCopy() {{  // works where the clipboard API is blocked
  const box = document.getElementById("rich");
  box.innerHTML = rich;
  const range = document.createRange(); range.selectNodeContents(box);
  const sel = getSelection(); sel.removeAllRanges(); sel.addRange(range);
  const ok = document.execCommand("copy"); sel.removeAllRanges();
  return ok;
}}
b.onclick = async () => {{
  let ok = true;
  try {{
    await navigator.clipboard.write([new ClipboardItem({{
      "text/html": new Blob([rich], {{type: "text/html"}}),
      "text/plain": new Blob([plain], {{type: "text/plain"}})}})]);
  }} catch (e) {{ ok = oldCopy(); }}
  b.textContent = ok ? "{done}" : "✗";
  setTimeout(() => b.textContent = "{label}", 2000);
}};
</script>""",
        height=40,
    )


def answer_actions(i: int, question: str, m: dict) -> None:
    """Copy and download buttons under an answer."""
    rtl = m.get("rtl", False)
    copy_col, pdf_col, _ = st.columns([1, 1, 2])
    with copy_col:
        copy_button(i, m["content"], rtl)
    with pdf_col:
        name = f"answer-{i // 2 + 1}"
        if CHROME.exists():
            st.download_button(
                "⬇️ تنزيل PDF" if rtl else "⬇️ Download PDF",
                data=lambda: answer_pdf(question, m["content"], m.get("figures", []), rtl),
                file_name=f"{name}.pdf", mime="application/pdf", key=f"pdf-{i}", on_click="ignore",
            )
        else:  # without Chrome: a web page that can be printed to PDF from any browser
            st.download_button(
                "⬇️ تنزيل" if rtl else "⬇️ Download",
                data=answer_html(m["content"], rtl), file_name=f"{name}.html", mime="text/html",
                key=f"pdf-{i}", on_click="ignore",
            )


def chat_file(chat_id: str) -> Path:
    return CHATS_DIR / f"{chat_id}.json"


def attachments_dir(chat_id: str) -> Path:
    return CHATS_DIR / chat_id


def save_attachments(chat_id: str, files: list) -> list[str]:
    """Keep the files attached to a question next to the chat; return their saved names."""
    folder = attachments_dir(chat_id)
    folder.mkdir(parents=True, exist_ok=True)
    names = []
    for f in files:
        stem, suffix = Path(f.name).stem, Path(f.name).suffix.lower()
        name, n = f"{stem}{suffix}", 1
        while (folder / name).exists():
            n += 1
            name = f"{stem}-{n}{suffix}"
        (folder / name).write_bytes(f.getvalue())
        names.append(name)
    return names


def attachment_parts(chat_id: str, names: list[str]) -> list[types.Part]:
    """The attached files for Gemini, with a note so it reviews them instead of citing them."""
    parts = [types.Part(text=(
        "Attached to this question: " + ", ".join(names) + ". These are the user's own files "
        "(e.g. scanned drawings, reports or letters) to read and review against the documents. "
        "Refer to them by file name; [p. N] citations are only for the documents."
    ))]
    for name in names:
        path = attachments_dir(chat_id) / name
        if not path.exists():
            continue
        mime = "application/pdf" if path.suffix == ".pdf" else f"image/{'png' if path.suffix == '.png' else 'jpeg'}"
        if path.stat().st_size <= INLINE_LIMIT:
            parts.append(types.Part.from_bytes(data=path.read_bytes(), mime_type=mime))
        else:
            uploaded = get_client().files.upload(file=path, config={"mime_type": mime})
            parts.append(types.Part.from_uri(file_uri=uploaded.uri, mime_type=mime))
    return parts


def show_attachments(chat_id: str, names: list[str]) -> None:
    images = [attachments_dir(chat_id) / n for n in names if not n.endswith(".pdf")]
    images = [p for p in images if p.exists()]
    if images:
        st.image([str(p) for p in images], width=160)
    for n in names:
        if n.endswith(".pdf"):
            st.caption(f"📎 {n}")


def list_chats() -> list[dict]:
    """Saved conversations, most recently used first (without their messages)."""
    chats = []
    for f in CHATS_DIR.glob("*.json"):
        try:
            c = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        chats.append({k: c.get(k) for k in ("id", "title", "updated", "archived")})
    return sorted(chats, key=lambda c: c["updated"] or 0, reverse=True)


def save_chat(chat_id: str, messages: list[dict]) -> None:
    path = chat_file(chat_id)
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    CHATS_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps({
        "id": chat_id,
        "title": old.get("title") or (messages[0]["content"] or ", ".join(messages[0].get("files", [])))[:60],
        "created": old.get("created", time.time()),
        "updated": time.time(),
        "archived": old.get("archived", False),
        "messages": messages,
    }, ensure_ascii=False, indent=1), encoding="utf-8")


def open_chat(chat_id: str | None) -> None:
    st.session_state.chat_id = chat_id
    st.session_state.messages = (
        json.loads(chat_file(chat_id).read_text(encoding="utf-8"))["messages"] if chat_id else []
    )


def set_archived(chat_id: str, archived: bool) -> None:
    path = chat_file(chat_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["archived"] = archived
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    if archived and st.session_state.get("chat_id") == chat_id:
        open_chat(None)


def delete_chat(chat_id: str) -> None:
    chat_file(chat_id).unlink(missing_ok=True)
    shutil.rmtree(attachments_dir(chat_id), ignore_errors=True)
    if st.session_state.get("chat_id") == chat_id:
        open_chat(None)


def chat_row(c: dict) -> None:
    """A saved chat in the sidebar: click the title to open it, ⋮ for actions."""
    title_col, menu_col = st.columns([5, 1], vertical_alignment="center")
    current = st.session_state.get("chat_id") == c["id"]
    title_col.button(
        c["title"], key=f"open-{c['id']}", width="stretch", type="primary" if current else "secondary",
        on_click=open_chat, args=(c["id"],),
    )
    with menu_col.popover("", icon=":material/more_vert:"):
        if c["archived"]:
            st.button("Unarchive", key=f"unarch-{c['id']}", icon=":material/unarchive:",
                      on_click=set_archived, args=(c["id"], False))
        else:
            st.button("Archive", key=f"arch-{c['id']}", icon=":material/archive:",
                      on_click=set_archived, args=(c["id"], True))
        st.button("Delete permanently", key=f"del-{c['id']}", icon=":material/delete:",
                  on_click=delete_chat, args=(c["id"],))


def library_state() -> tuple:
    """Changes when a document is added, finishes, fails, is switched or deleted."""
    return tuple((d["id"], d["status"], d["enabled"]) for d in library_docs())


def used_signature() -> tuple:
    """The uploaded documents that go to the model: ready and switched on."""
    return tuple(
        (d["id"], d["name"], (doc_folder(d["id"]) / "text.md").stat().st_mtime)
        for d in library_docs() if d["status"] == "ready" and d["enabled"]
    )


def document_row(d: dict) -> None:
    info_col, use_col, menu_col = st.columns([6, 2, 1], vertical_alignment="center")
    with info_col:
        st.markdown(f"**{d['id']}** · {d['name']}")
        if d["status"] == "ready":
            st.caption(f"✅ {d['pages']} page{'s' if d['pages'] != 1 else ''}")
        elif d["status"] == "processing":
            st.progress(d["done"] / max(d["pages"], 1), text=f"Reading… {d['done']}/{d['pages']} pages")
        else:
            st.caption(f"⚠️ {'Stopped' if d['status'] == 'stopped' else 'Failed'}"
                       f"{': ' + d['error'] if d.get('error') else ''}")
    if d["status"] == "ready":
        use_col.toggle("Use", value=d["enabled"], key=f"use-{d['id']}", on_change=set_enabled,
                       args=(d["id"],), help="Include this document in answers")
    with menu_col.popover("", icon=":material/more_vert:"):
        if d["status"] in ("failed", "stopped"):
            st.button("Retry", key=f"retry-{d['id']}", icon=":material/refresh:",
                      on_click=start_ingest, args=(d["id"],))
        st.button("Delete permanently", key=f"deldoc-{d['id']}", icon=":material/delete:",
                  on_click=delete_document, args=(d["id"],))


def documents_panel(state: tuple) -> None:
    """The document list. While something is being read it refreshes itself every few
    seconds; when the set of documents for the model changes, the whole app reruns."""
    if library_state() != state:
        st.rerun(scope="app")
    st.markdown(f"**📘 Egyptian Fire Code** · {len(page_index())} pages · built in")
    for d in library_docs():
        document_row(d)


def upload_form() -> None:
    with st.form("upload", clear_on_submit=True, border=False):
        files = st.file_uploader(
            "Add PDFs or scanned images", type=UPLOAD_TYPES, accept_multiple_files=True,
            help="Each file is read page by page with Gemini (about 1–2 minutes per 20 pages). "
                 "You can keep chatting meanwhile.",
        )
        if st.form_submit_button("Add", icon=":material/upload:", width="stretch") and files:
            errors = [e for f in files if (e := add_document(f.name, f.getvalue()))]
            for e in errors:
                st.error(e)
            if len(errors) < len(files):
                st.rerun()


def signed_in() -> bool:
    """When APP_PASSWORD is set in .env, ask for it once per browser session."""
    password = os.getenv("APP_PASSWORD")
    if not password or st.session_state.get("signed_in"):
        return True
    with st.form("sign-in", border=False):
        typed = st.text_input("Password / كلمة المرور", type="password")
        if st.form_submit_button("Enter / دخول"):
            if hmac.compare_digest(typed.encode(), password.encode()):
                st.session_state.signed_in = True
                st.rerun()
            st.error("Wrong password / كلمة المرور غير صحيحة")
    return False


def main() -> None:
    st.set_page_config(page_title="Document Chat", page_icon="📄")
    # Arabic messages are fully right-to-left; inside them, each paragraph still picks
    # its own direction so an English sentence or code reads correctly.
    st.markdown(
        """<style>
        [class*="st-key-msg-"][class*="-rtl"] { direction: rtl; text-align: right; }
        [class*="st-key-msg-"][class*="-rtl"] :is(p, li, h1, h2, h3, h4, th, td,
            [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"]) {
            text-align: right !important;
        }
        [class*="st-key-msg-"][class*="-rtl"] :is(ul, ol) { padding-right: 1.5em; padding-left: 0; }
        [class*="st-key-msg-"] :is(p, li, h1, h2, h3, h4, td, th, summary) {
            unicode-bidi: plaintext;
        }
        [data-testid="stChatInputTextArea"] { unicode-bidi: plaintext; }
        /* Saved chats in the sidebar: one line each, Arabic titles read right-to-left */
        [class*="st-key-open-"] button { justify-content: flex-start; }
        [class*="st-key-open-"] button > div { overflow: hidden; }
        [class*="st-key-open-"] button p {
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis; unicode-bidi: plaintext;
        }
        </style>""",
        unsafe_allow_html=True,
    )
    st.title("📄 Document Chat")
    if not signed_in():
        st.stop()

    if not os.getenv("GEMINI_API_KEY"):
        st.error("GEMINI_API_KEY is not set. Put it in the .env file (see .env.example).")
        st.stop()

    texts = sorted(OUT_DIR.glob("*.md"))
    used = used_signature()
    # Changes to the text, the crops, or the chosen uploads rebuild the corpus and the Gemini cache
    corpus_key = (tuple((f.name, f.stat().st_mtime) for f in texts + sorted(CROPS_DIR.glob("*.json"))), used)
    if not texts and not used:
        st.warning("No documents yet. Run `python ocr.py`, or add documents in the sidebar.")
    corpus = load_corpus(corpus_key, tuple((doc, name) for doc, name, _ in used))
    tokens = corpus_tokens(corpus, corpus_key)
    too_big = tokens > MAX_CONTEXT_TOKENS
    client = get_client()

    with st.sidebar:
        st.caption(f"Model: `{MODEL}` · ~{tokens:,} tokens")
        state = library_state()
        busy = any(status == "processing" for _, status, _ in state)
        with st.expander(f"📚 Documents ({1 + len(library_docs())})", expanded=busy):
            st.fragment(documents_panel, run_every=3 if busy else None)(state)
            upload_form()
        if too_big:
            st.error(f"The documents in use are ~{tokens:,} tokens, more than the model can read "
                     f"({MAX_CONTEXT_TOKENS:,}). Switch some off.")
        st.button("New chat", icon=":material/add:", width="stretch", on_click=open_chat, args=(None,))
        chats = list_chats()
        active = [c for c in chats if not c["archived"]]
        archived = [c for c in chats if c["archived"]]
        if active:
            st.caption("Chats")
            for c in active:
                chat_row(c)
        if archived:
            with st.expander(f"Archived ({len(archived)})", icon=":material/inventory_2:"):
                for c in archived:
                    chat_row(c)

    messages = st.session_state.setdefault("messages", [])
    for i, m in enumerate(messages):
        rtl = m.get("rtl", is_arabic(m["content"]))
        with st.chat_message(m["role"]), message_box(i, rtl):
            if m.get("files"):
                show_attachments(st.session_state.chat_id, m["files"])
            st.markdown(tidy(m["content"], rtl))
            if m["role"] == "assistant":
                answer_actions(i, messages[i - 1]["content"], m)
                show_number_check(m["content"], messages[i - 1]["content"], rtl)
            show_sources(m.get("pages", []), m.get("figures", []), rtl)

    sent = st.chat_input(
        "Ask about the documents, or attach scans to review… / اسأل أو أرفق ملفات للمراجعة…",
        accept_file="multiple", file_type=UPLOAD_TYPES, disabled=too_big or not corpus,
    )
    if sent and (sent.text.strip() or sent.files):
        question = sent.text.strip()
        rtl = is_arabic(question)
        new_chat = not st.session_state.get("chat_id")
        if new_chat:
            st.session_state.chat_id = time.strftime("%Y%m%d-%H%M%S")
        chat_id = st.session_state.chat_id
        attached = save_attachments(chat_id, sent.files) if sent.files else []
        messages.append({"role": "user", "content": question, "rtl": rtl, "files": attached})
        with st.chat_message("user"), message_box(len(messages) - 1, rtl):
            show_attachments(chat_id, attached)
            st.markdown(question)

        contents = [
            types.Content(
                role="user" if m["role"] == "user" else "model",
                parts=(attachment_parts(chat_id, m["files"]) if m.get("files") else [])
                + ([types.Part(text=m["content"])] if m["content"] else []),
            )
            for m in messages
        ]
        cache = get_book_cache(client, f"{INSTRUCTIONS}\n\n{corpus}", (corpus_key, INSTRUCTIONS))
        config = types.GenerateContentConfig(
            cached_content=cache.name,
            # Low thinking: same answers on this book in tests, ~8x faster to start.
            # Long questions describe a project and need many calculations, which go
            # wrong at low thinking, so those get more time.
            thinking_config=types.ThinkingConfig(
                thinking_level="high" if attached or len(question) > DETAILED_QUESTION_CHARS else "low"
            ),
        )

        with st.chat_message("assistant"):
            waiting = st.empty()
            waiting.markdown("▌")
            chunks = iter(client.models.generate_content_stream(model=cache.model, contents=contents, config=config))
            # The answer is in the documents' language, not necessarily the question's, so
            # read its first words before choosing the text direction
            answer = ""
            for chunk in chunks:
                answer += chunk.text or ""
                if len(re.findall(r"[؀-ۿA-Za-z]", answer)) >= 30:
                    break
            waiting.empty()
            rtl = is_arabic(answer)
            with message_box(len(messages), rtl):
                # Stream into a placeholder so [fig:…] tags can be tidied as the text arrives
                placeholder = st.empty()
                placeholder.markdown(tidy(answer, rtl) + " ▌")
                for chunk in chunks:
                    if chunk.text:
                        answer += chunk.text
                        placeholder.markdown(tidy(answer, rtl) + " ▌")
                placeholder.markdown(tidy(answer, rtl))
                if usage := chunk.usage_metadata:  # in the server log: how much came from the cache
                    print(f"tokens: {usage.prompt_token_count:,} in ({usage.cached_content_token_count or 0:,} "
                          f"from cache), {usage.candidates_token_count or 0:,} out", flush=True)
                pages = cited_pages(answer)
                figures = cited_figures(answer, pages)
                reply = {"role": "assistant", "content": answer, "pages": pages, "figures": figures, "rtl": rtl}
                answer_actions(len(messages), question, reply)
                show_number_check(answer, question, rtl)
                show_sources(pages, figures, rtl)
        messages.append(reply)
        save_chat(st.session_state.chat_id, messages)
        if new_chat:
            st.rerun()  # show the new chat in the sidebar


main()
