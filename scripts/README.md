# Scripts

Standalone utilities around the extraction pipeline — corpus scoping, paper
discovery, batch extraction, evaluation and ORKG upload. All are run from the
project root as `python scripts/<name>.py`.

```
scripts/
├── README.md                         
│
│   # Corpus scoping and discovery
├── arxiv_monthly_volume.py            # Count arXiv papers per month per category 
├── discover_papers_llm.py             # Ask a web-capable model which LLM/VLM papers exist, then verify every hit
├── build_papers_list.py               # Build papers list with PDF URLs / arXiv IDs from ORKG and arXiv
├── sync_papers_list_with_gold.py      # Prune papers_list.json down to what's in the gold standard
│
│   # Extraction
├── batch_extract_all_papers.py        # Run extraction over every paper in papers_list.json                                
├── import_extracted_to_model_folders.py  # Sort flat extraction JSONs into <extraction_root>/<model_slug>/
│
│   # Evaluation
├── aggregate_model_evaluation.py      # Score one model's extractions against the gold standard
├── build_results_table.py             # Combine per-model evaluations into one comparison table
│
│   # ORKG
├── append_to_paper.py                 # Append extracted data to a specific ORKG paper ID
├── sandbox_upload.py                  # Upload an extraction JSON to the ORKG sandbox
├── fetch_property_descriptions.py     # Refresh the local ORKG property-description cache
│
│   # Misc
├── list_kisski_models.py              # List available models from the KISSKI /v1/models endpoint
│
├── evaluation/                        # Gold-standard evaluators — see evaluation/README.md
│   ├── README.md
│   ├── convert_gold_standard.py       # ORKG comparison CSV → gold-standard JSON
│   ├── evaluate_extraction.py         # Relaxed evaluator
│   ├── evaluate_extraction_strict.py  # Strict evaluator (thesis metrics)
│   ├── normalize_gold_standard_parameters.py
│   └── verify_gold_standard.py
│
└── data/papers/                       # Local PDF + metadata working copies
```

---

# Discover arXiv Monthly Volume

`arxiv_monthly_volume.py` counts how many arXiv papers were submitted per month
per category, so you can scope how much material the extraction pipeline would
have to process before committing to a run.

## Usage

```bash
python scripts/arxiv_monthly_volume.py
```

Defaults to `cs.AI`, `cs.CL` and `cs.CV`, from `2020-01` to the last complete
month. 

Other common invocations:

```bash
# One category
python scripts/arxiv_monthly_volume.py --categories cs.CL --start 2024-01 --end 2024-12

# Skip the deduplicated union column 
python scripts/arxiv_monthly_volume.py --no-union

```

## Input parameters

| Option | Type | Default | Description |
|---|---|---|---|
| `--start` | `YYYY-MM` | `2020-01` | First month to count. |
| `--end` | `YYYY-MM` | last complete month | Last month to count. |
| `--categories` | one or more strings | `cs.AI cs.CL cs.CV` | arXiv categories to query, space-separated. |
| `--no-union` | flag | off | Skip the deduplicated across-categories count. Only meaningful with 2+ categories. |
| `--output` | path | `data/arxiv_monthly_volume.csv` | Where to write the CSV. |
| `--delay` | float | `3.0` | Seconds between API calls. arXiv asks for at least 3; lower values log a warning. |


## Output

### CSV — `data/arxiv_monthly_volume.csv`

One row per month, one column per category:

| Column | Description |
|---|---|
| `month` | `YYYY-MM`. |
| one per category | Papers in that category that month (e.g. `cs.AI`). |
| `sum_of_categories` | The category columns added together. Counts cross-listed papers more than once. |
| `union` | Distinct papers across all categories, deduplicated by arXiv. Omitted with `--no-union`. |

```csv
month,cs.AI,cs.CL,cs.CV,sum_of_categories,union
2020-01,352,293,781,1426,1341
2020-02,446,362,954,1762,1660
```
