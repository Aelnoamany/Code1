# Arabic PDF → Text → Chatbot (POC)

## Setup (once)
1. Copy `.env.example` to `.env` and paste your Google AI Studio key into it.
2. Put the 20 scanned PDFs in `pdfs/`.

## 1. OCR
```bash
.venv/bin/python ocr.py
```
- One Gemini call per page; results land in `output/pages/...` immediately.
- Safe to stop and re-run: finished pages are skipped, failed ones retried.
- Final text: `output/<pdf name>.md`, with `<!-- page N -->` markers.
- Faint or handwritten scans: `--dpi 300`. Rate-limit errors: `--workers 2`.
- Redo one bad page: delete its `output/pages/<pdf>/page_000N.md` and re-run.

## 1b. Cut out tables and figures
```bash
.venv/bin/python crop.py
```
- Gemini finds every table/figure on each page; each is saved as its own image in
  `output/crops/` with its printed caption (e.g. جدول (٥-١)).
- Redo specific book pages: `.venv/bin/python crop.py --pages 34 35`

## 2. Chat
```bash
.venv/bin/streamlit run app.py
```
Opens in your browser. All documents go into Gemini's context on every question
(~230 pages fits easily), and answers cite fire code pages as `[p. N]`.

**More documents:** in the sidebar, open 📚 Documents, choose PDFs or scanned images
(JPG/PNG), and click Add. Each file is read in the background with Gemini (text plus
cut-out tables and figures, about 1–2 minutes per 20 pages) and joins the answers when
it shows ✅. Answers cite it as [D1 p. 5]. Use the switch to leave a document out of
answers, and ⋮ to retry a failed one or delete it. Files are kept in `library/`.
The model reads at most ~900,000 tokens (roughly 5 books the size of the fire code).

**Attach files to a question:** click 📎 in the question box (or drag files onto it) to
add scanned PDFs or images, such as drawings or a report to review. They go to the model
with that question only (not to the library), are kept in `chats/<chat id>/`, and are
sent again with later questions in the same chat so you can keep asking about them.

Every conversation is saved in `chats/` (one JSON file each) and listed in the
sidebar. Click one to reopen it; its ⋮ menu archives, unarchives, or deletes it.

Under each answer:
- **Copy answer** copies it with formatting (tables, bold) for Word or email.
- **⚠️ numbers not found on the page they cite** (shown only when there are any) lists
  values the answer cites that are not written on the cited page. Usually that means a
  wrong page number or a value that is not in the code; it can also be a number
  converted from a code value, or one the OCR misread, so check those against the page.
- **Download PDF** saves the question, the answer, and the original table and figure
  images. It is printed with Google Chrome, which must be installed.

## Models
Both scripts default to `gemini-flash-latest`. Pin a version in `.env` with
`GEMINI_OCR_MODEL` / `GEMINI_CHAT_MODEL` if you want stable results.
