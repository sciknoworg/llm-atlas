# Run Extraction + Accuracy Evaluation (PDF URL, e.g. GPT-1)

This workflow runs extraction on a **PDF-only** paper (not on ArXiv), then evaluates against the gold standard to get an accuracy grade (F1, precision, recall).

## Quick commands (GPT-1)

```bash
# 1. Extract (requires KISSKI_API_KEY in .env)
python -m src.pipeline --pdf-url "https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf" --paper-title "Improving Language Understanding by Generative Pre-Training" --no-update

# 2. Evaluate (use the printed saved path from step 1)
python scripts/evaluation/evaluate_extraction.py --prediction "data/extracted/improving_language_understanding_by_generative_pre_training_YYYYMMDD_HHMMSS.json" --gold data/gold_standard/R1364660.json --output data/evaluation_reports/gpt1_report.json
```

## Example: GPT-1 paper (first in gold standard)

The GPT-1 paper *"Improving Language Understanding by Generative Pre-Training"* is not on ArXiv. PDF:

- **URL:** https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf  
- **Gold-standard title:** `Improving Language Understanding by Generative Pre-Training` (must match exactly for evaluation).

### 1. Run extraction (KISSKI pipeline)

From the project root, with `KISSKI_API_KEY` and `ORKG_*` set in `.env` as needed:

```bash
python -m src.pipeline --pdf-url "https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf" --paper-title "Improving Language Understanding by Generative Pre-Training" --no-update
```

- `--no-update`: skip ORKG upload (use only extraction + evaluation).
- The pipeline downloads the PDF, parses it, extracts via KISSKI, and saves JSON to `data/extracted/`.
- The command prints the **saved path**, e.g.  
  `data/extracted/improving_language_understanding_by_generative_pre_training_20260126_123456.json`

### 2. Run accuracy evaluation

Use the printed path as `--prediction`:

```bash
python scripts/evaluation/evaluate_extraction.py --prediction "data/extracted/improving_language_understanding_by_generative_pre_training_20260126_123456.json" --gold data/gold_standard/R1364660.json --output data/evaluation_reports/gpt1_report.json
```

- `--gold`: gold-standard JSON (default: `data/gold_standard/R1364660.json`).
- `--output`: optional; saves the evaluation report as JSON.
- The script filters gold by **paper title** (from the extraction JSON), then compares extracted vs gold and prints **Accuracy**, **Precision**, **Recall**, **F1-Score**, and per-field metrics.

### 3. Interpret the grade

The evaluator reports:

- **Overall metrics:** Accuracy, Precision, Recall, **F1-Score** (main grade).
- **Per-field metrics:** which fields the model got right or wrong.
- **Matched / missing / unmatched models:** how many gold models were matched.

Example summary:

```
OVERALL METRICS (All Fields Combined)
  Accuracy:        xx.xx%
  Precision:       xx.xx%
  Recall:          xx.xx%
  F1-Score:        xx.xx%    <- main grade
```

Rough interpretation: F1 ≥ 80% excellent, ≥ 60% good, ≥ 40% fair, &lt; 40% poor.

**Model matching:** Gold is filtered by paper title. The evaluator matches extracted models to gold by `model_name`: exact match first, then fallback when the gold name is a substring of the prediction (e.g. gold "GPT-1" matches "GPT-1 (117M)" or "GPT-1 117M"). One gold model is matched to at most one prediction.

---

## Paper list (batch use)

The GPT-1 paper is in `data/gold_standard/paper_list.json`:

```json
[
  {
    "source": "url",
    "pdf_url": "https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf",
    "title": "Improving Language Understanding by Generative Pre-Training"
  }
]
```

A future batch script can read this list and run extraction (and evaluation) for each entry. For now, use the two commands above for this single paper.

---

## Checklist

1. **PDF URL** – stable link to the PDF.
2. **Paper title** – **exact** gold-standard title for evaluation matching.
3. **Extract** – `python -m src.pipeline --pdf-url ... --paper-title ... --no-update`.
4. **Evaluate** – `python scripts/evaluation/evaluate_extraction.py --prediction <path> [--output ...]`.
