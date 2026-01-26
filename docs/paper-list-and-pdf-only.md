# Paper List & PDF-Only (Non–ArXiv) Support

## Overview

The batch extraction pipeline uses a **paper list** (CSV or JSON) to decide which papers to process. Each entry can be:

- **ArXiv** – fetch metadata + PDF from ArXiv (current behavior).
- **Local PDF** – use an existing PDF path (e.g. `data/papers/xyz.pdf`).
- **PDF URL** – download PDF from a URL (conference, institutional repo, etc.), then extract.

This allows papers that exist only as PDF (not on ArXiv) to be included in the same pipeline and evaluated against the gold standard.

---

## Paper list format

### JSON (`data/gold_standard/paper_list.json`)

```json
[
  {
    "source": "arxiv",
    "arxiv_id": "2307.09288",
    "title": "Llama 2: Open Foundation and Fine-Tuned Chat Models"
  },
  {
    "source": "local",
    "pdf_path": "data/papers/some_paper.pdf",
    "title": "Improving Language Understanding by Generative Pre-Training"
  },
  {
    "source": "url",
    "pdf_url": "https://example.org/papers/xyz.pdf",
    "title": "BERT: Pre-training of Deep Bidirectional Transformers"
  }
]
```

### CSV (`data/gold_standard/paper_list.csv`)

| source | arxiv_id | pdf_path | pdf_url | title |
|--------|----------|----------|---------|-------|
| arxiv | 2307.09288 | | | Llama 2: Open Foundation and Fine-Tuned Chat Models |
| local | | data/papers/some_paper.pdf | | Improving Language Understanding by Generative Pre-Training |
| url | | | https://example.org/papers/xyz.pdf | BERT: Pre-training... |

**Rules:**

- `source`: one of `arxiv`, `local`, `url`.
- **ArXiv:** set `arxiv_id`; `title` optional (used for verification + gold-standard matching).
- **Local:** set `pdf_path` (relative to project root or absolute); `title` optional but **recommended** for gold-standard matching.
- **URL:** set `pdf_url`; `title` optional but **recommended**.

---

## PDF-only papers (not on ArXiv)

### When to use

- Paper is only available as PDF (no ArXiv).
- PDF from conference/journal page, institutional repo, author homepage, etc.

### Options

| Option | Description | Pipeline behavior |
|--------|-------------|-------------------|
| **Local PDF** | You already have the PDF on disk. | Use `source: local` + `pdf_path`. No download. Parse → extract → save JSON. |
| **PDF URL** | PDF is at a stable URL. | Use `source: url` + `pdf_url`. Download to `data/papers/`, then same as above. |

### Metadata for PDF-only papers

- **No ArXiv** ⇒ no automatic metadata (authors, published, etc.).
- **`title`** in the paper list is used for:
  - Matching to gold-standard rows (same `paper_title`).
  - Logging and output filenames.
- Optional: add `authors`, `year`, `url` in the paper list later if we extend the schema.

### Matching to gold standard

- Gold standard is keyed by **paper title** (e.g. `data/gold_standard/R1364660.json`).
- For **PDF-only** papers, store the **same title** as in the gold standard in the paper list.
- Evaluation can then match extraction output ↔ gold rows by `paper_title`.

---

## Pipeline changes (to implement)

1. **Paper list loader**  
   - Read `data/gold_standard/paper_list.json` (or `.csv`).  
   - Parse into list of `{source, arxiv_id?, pdf_path?, pdf_url?, title?}`.

2. **Resolve PDF path per source**
   - **arxiv:** `PaperFetcher.fetch_paper()` → `metadata["pdf_path"]` (unchanged).
   - **local:** check `Path(pdf_path).exists()` → use as-is.
   - **url:** download to `data/papers/` (reuse logic from `grete_extract_from_url`), return that path.

3. **`process_paper` / batch**
   - Accept a **paper list entry** (not only `arxiv_id`).
   - Branch on `source` → get PDF path (ArXiv / local / URL).
   - Build `paper_metadata`: from ArXiv when `source=arxiv`, else from paper list (`title`, optional `url` for `source=url`).
   - Then: parse PDF → extract → save JSON (and optionally update ORKG). Same as now.

4. **Batch script**
   - Read paper list → for each entry, run extraction (and optionally evaluation).
   - Replace hardcoded `ARXIV_IDS` in `grete_extract_batch.sh` with “read from paper list”.

---

## File locations

| File | Purpose |
|------|--------|
| `data/gold_standard/paper_list.json` | Paper list (primary). Batch script reads this. |
| `data/gold_standard/paper_list.csv` | Alternative; same schema. |
| `data/papers/` | Downloaded PDFs (ArXiv + URL) and optional local PDFs. |
| `data/extracted/` | Extraction JSON outputs. |

---

## Summary

- **ArXiv:** keep current flow; add `arxiv_id` + optional `title` to paper list.
- **PDF-only:** use `source: local` + `pdf_path` or `source: url` + `pdf_url`. Always add `title` when the paper is in the gold standard.
- **Gold-standard matching:** by `paper_title`; for PDF-only, the title in the paper list must match the gold standard.

When you have ArXiv IDs (and titles) plus any PDF-only papers, add them to `paper_list.json` in this format; we can then wire the loader and batch extraction to use it.

---

## Quick reference: PDF-only checklist

1. **Get the PDF** – either save it locally (e.g. under `data/papers/`) or have a stable URL.
2. **Add to paper list** – use `source: local` + `pdf_path` or `source: url` + `pdf_url`.
3. **Set `title`** – use the **exact** gold-standard paper title so evaluation can match.
4. **Run batch extraction** – same as for ArXiv; the pipeline will resolve the PDF and extract.
