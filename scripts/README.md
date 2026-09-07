# Scripts

Standalone utilities around the extraction pipeline — corpus scoping, paper
discovery, batch extraction, evaluation and ORKG upload.

Scripts can be run using the command:

```bash
python scripts/<name>.py [options]
```

Some scripts need credentials. Where a section lists **Required in `.env`**, add
those variables to the `.env` file at the project root before running it.

## Directory structure

```
scripts/
├── README.md
│
│   # Corpus scoping and discovery
├──  1. arxiv_monthly_volume.py                # count arXiv papers per month per category
├──  2. build_papers_list.py                   # resolve gold-standard titles to PDF URLs / arXiv ids
├──  3. sync_papers_list_with_gold.py          # rewrite papers_list.json to match the gold standard
│
│   # Extraction
├──  4. batch_extract_all_papers.py            # extract every paper in papers_list.json
├──  5. import_extracted_to_model_folders.py   # sort flat extraction JSONs into per-model folders
│
│   # ORKG
├──  6. append_to_paper.py                     # append contributions to an existing ORKG paper
├──  7. sandbox_upload.py                      # upload an extraction JSON as a new ORKG paper
│
│   # Evaluation
├──  8. aggregate_model_evaluation.py          # score one model's extractions against the gold standard
├──  9. build_results_table.py                 # combine per-model scores into one table
│
├── 10. evaluation/                            # gold-standard evaluators — see evaluation/README.md
│
│   # Misc
└── 11. list_kisski_models.py                  # list models available on the KISSKI endpoint
```

---

## Corpus scoping and discovery

### 1. `arxiv_monthly_volume.py`

Counts arXiv papers per month per category, to scope how much material the
extraction pipeline would have to process. 

```bash
python scripts/arxiv_monthly_volume.py
python scripts/arxiv_monthly_volume.py --categories cs.CL --start 2024-01 --end 2024-12
python scripts/arxiv_monthly_volume.py --no-union
python scripts/arxiv_monthly_volume.py --refresh
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--start` | `YYYY-MM` | `2020-01` | First month |
| `--end` | `YYYY-MM` | last complete month | Last month - The current month is always excluded — its count would still be changing daily. |
| `--categories` | one or more strings | `cs.AI cs.CL cs.CV` | arXiv categories, space-separated. |
| `--no-union` | flag | off | Skip the deduplicated across-categories count. Saves one request per month; only meaningful with 2+ categories. |
| `--output` | path | `data/arxiv_monthly_volume.csv` | Where to write the CSV. |
| `--cache` | path | `data/arxiv_monthly_counts.json` | Cache file to read on startup and update during the run. |
| `--refresh` | flag | off | Ignore cached counts and re-fetch every cell. Still rewrites the cache. |
| `--delay` | float | `3.0` | Seconds between API calls. arXiv asks for at least 3; lower values log a warning. |

**Input:** Queries the public arXiv API — no credentials required.

**Outputs — three, and the cache is independent of the CSV:**

1. **CSV** at `--output`. One row per month:

   | Column | Description |
   |---|---|
   | `month` | `YYYY-MM`. |
   | one per category | Papers in that category that month (e.g. `cs.AI`). |
   | `sum_of_categories` | Category columns added together. Counts cross-listed papers more than once. |
   | `union` | Distinct papers across all categories, deduplicated by arXiv. Omitted with `--no-union`. |

   ```csv
   month,cs.AI,cs.CL,cs.CV,sum_of_categories,union
   2020-01,352,293,781,1426,1341
   2020-02,446,362,954,1762,1660
   ```

   Use `union` for "how many papers there are"; `sum_of_categories`
   double-counts anything cross-listed. A cell is left **empty** when that query
   failed after all retries.

2. **Console summary** — per-year totals plus how much double-counting `union`
   removes.

3. **Cache JSON** at `--cache`, of the form
   `{"counts": {"cs.AI|2020-01": 352, ...}}`, one entry per (series, month).
   This is read and written **regardless of the CSV**

### 2. `build_papers_list.py`

Resolves the gold standard's paper titles to PDF URLs and arXiv ids by querying
ORKG and arXiv, producing the list that `batch_extract_all_papers.py` consumes.

```bash
python scripts/build_papers_list.py
python scripts/build_papers_list.py --no-orkg      # arXiv lookups only
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--gold` | path | `data/gold_standard/gold_standard_set.json` | Gold standard to read titles from. |
| `--output` | path | `data/gold_standard/papers_list.json` | Where to write the list. |
| `--no-orkg` | flag | off | Skip the ORKG lookups. |
| `--no-arxiv` | flag | off | Skip the arXiv search. |

**Input:** the gold-standard JSON.

**Required in `.env`:** `ORKG_EMAIL` and `ORKG_PASSWORD` — unless `--no-orkg`
is passed, in which case none are needed.

**Output:** `papers_list.json` — a list of `{paper_title, pdf_url, arxiv_id,
doi, source}` records. Overwrites `--output` if it exists.

### 3. `sync_papers_list_with_gold.py`

Brings `papers_list.json` back in line with the gold standard.

```bash
python scripts/sync_papers_list_with_gold.py
```

**Options:** none.

**Input(s):** `data/gold_standard/gold_standard_set.json` and `data/gold_standard/papers_list.json`.

**Output — `papers_list.json` is overwritten in place.** Three distinct things
happen, all destructive, so copy the file first if you might want it back:

- **Removals.** Entries whose title is not in the gold standard's
  `extraction_data` are dropped — this is how continuation-line junk titles get
  cleaned out. Each removal is logged.
- **Reordering.** Surviving entries keep their existing `pdf_url`, `arxiv_id`,
  `doi` and `source`, but the file is rewritten **sorted alphabetically by
  title**, so any original ordering is lost.
- **Placeholder additions.** Gold-standard titles missing from the list are
  **added** with `pdf_url`, `arxiv_id` and `doi` set to `null` and
  `source: "manual_needed"`. These are not usable for extraction until someone
  fills them in by hand.

The resulting file therefore has exactly one entry per unique gold-standard
title. 

---

## Extraction

### 4. `batch_extract_all_papers.py`

Runs extraction over every paper in the papers list, using the model currently
set in `config/config.yaml`. Extraction only — it does **not** upload to ORKG.

```bash
python scripts/batch_extract_all_papers.py --dry-run
python scripts/batch_extract_all_papers.py --limit 5
python scripts/batch_extract_all_papers.py --skip-existing --start-from 40
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--papers-list` | path | `data/gold_standard/papers_list.json` | Papers to process. |
| `--output-dir` | path | `<pipeline.extraction_output_dir>/<model_slug>/` | Where the per-model copies go. |
| `--dry-run` | flag | off | Show what would be extracted without running. |
| `--skip-existing` | flag | off | Skip papers that already have extraction output. |
| `--start-from` | int | `0` | Start from paper index N, for resuming. |
| `--limit` | int | all | Only process the first N papers, for testing. |

**Input:** the papers list, plus PDFs downloaded to `data/papers/` as needed.

**Required in `.env`:** `KISSKI_API_KEY`.

**Outputs:**

1. One extraction JSON per paper in the configured extraction directory, plus a
   copy under the per-model subdirectory named after the active model — which is
   what `aggregate_model_evaluation.py` then scores.
2. `extraction_summary_<YYYYmmdd_HHMMSS>.json` in `--output-dir`, written on
   every non-dry run. It records the model used, start and end timestamps, and
   per-paper status, source URL, error and output path. The timestamp means
   these **accumulate**: one summary per run, never overwritten, so clear old
   ones out periodically.

### 5. `import_extracted_to_model_folders.py`

Sorts flat extraction JSONs into `<extraction_root>/<model_slug>/` based on the
`model_used` field inside each file. Useful when extractions were produced
before per-model directories existed.

```bash
python scripts/import_extracted_to_model_folders.py
python scripts/import_extracted_to_model_folders.py --move
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--source-dir` | path | `pipeline.extraction_output_dir` from config | Directory of flat JSON files. Top level only — subdirectories are not scanned. |
| `--move` | flag | off | Move instead of copy. **Removes the originals from the source directory.** |

**Input:** extraction JSONs containing a `model_used` field.

**Output:** the same files under `<extraction_root>/<model_slug>/`. Without
`--move` the originals stay put, so the default is safe to re-run.

---

## ORKG

Both upload scripts default to **`sandbox`**. Pass `--host production` to write
to the live system.

### 6. `append_to_paper.py`

Appends contributions to an **existing** ORKG paper instead of creating a new
one — the right tool when a paper is already in the graph and only needs more
model tabs.

```bash
python scripts/append_to_paper.py \
    --file results/extracted/2302.13971.json \
    --paper-id R1568688
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--file` | path | **required** | Extraction JSON whose models become contributions. |
| `--paper-id` | str | **required** | Target ORKG paper id, e.g. `R1568688`. |
| `--host` | str | `sandbox` | ORKG host: `sandbox`, `incubating` or `production`. |

**Input:** one extraction JSON.

**Required in `.env`:** `ORKG_EMAIL` and `ORKG_PASSWORD`.

**Output:** new contributions on the named paper, posted to
`/api/papers/{id}/contributions`. The new contribution ids are logged. Nothing
is written to disk, and the paper itself is not modified beyond gaining tabs.

### 7. `sandbox_upload.py`

Uploads one extraction JSON as a **new** ORKG paper, contributions included.

```bash
python scripts/sandbox_upload.py --file results/extracted/2302.13971.json
python scripts/sandbox_upload.py --file results/extracted/2302.13971.json --host production
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--file` | path | **required** | Extraction JSON to upload |
| `--host` | str | `sandbox` | ORKG host. |

**Input:** one extraction JSON.

**Required in `.env`:** `ORKG_EMAIL` and `ORKG_PASSWORD`.

**Output:** an ORKG paper with one contribution per extracted model; the paper
id and URL are printed. What happens on a re-run **depends on the host**:

- **sandbox / incubating** — a second paper is created. There is no
  deduplication, so use `6. append_to_paper.py` when the paper already exists.
- **production** — the manager first searches for an existing paper with the
  exact same title and, if it finds one, adds only the contribution labels that
  are missing rather than creating a duplicate.

The configured comparison is **not** touched either way: this script builds the
manager without comparison arguments, so the comparison update stays off.

---

## Evaluation

### 8. `aggregate_model_evaluation.py`

Scores one model's extractions against the gold standard across all papers and
reduces them to a single aggregate.

```bash
python scripts/aggregate_model_evaluation.py \
    --model-dir results/extracted/qwen3-6-35b-a3b \
    --model-name qwen3.6-35b-a3b \
    --output results/qwen3-6-35b-a3b_results.json
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--model-dir` | path | **required** | Directory of extraction outputs for this model. |
| `--model-name` | str | **required** | Model name to record in the results table. |
| `--gold` | path | `data/gold_standard/gold_standard_set.json` | Gold standard to score against. |
| `--output` | path | **required** | Where to write the aggregated results JSON. |

**Input:** every extraction JSON in `--model-dir`, plus the gold standard.

**Output:** one JSON at `--output` holding the aggregate scores (overall F1,
BERTScore aggregate, and per-field breakdowns).

### 9. `build_results_table.py`

Combines the aggregated results of several models into one comparison table.

```bash
python scripts/build_results_table.py
python scripts/build_results_table.py --results-dir results/ --output results/table.csv
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--results-dir` | path | `results/` | Directory holding `*_results.json` files. |
| `--output` | path | `results/final_results_table.csv` | Where to write the table. |
| `--format` | str | `csv` | Output format. |

**Input:** the `*_results.json` files written by
`aggregate_model_evaluation.py`. Run that first, once per model.

**Output:** the comparison table, with models grouped into the three categories
(Vision, Think/Reasoning, Instruction Tuned).

### 10. `evaluation/`

Per-paper gold-standard evaluators. All five are documented in their own
[`evaluation/README.md`](evaluation/README.md) — purpose, runnable command,
options, inputs and outputs each — along with the metric definitions:

| Script | Purpose |
|---|---|
| `convert_gold_standard.py` | ORKG comparison CSV → gold-standard JSON |
| `evaluate_extraction.py` | Relaxed evaluator, one paper |
| `evaluate_extraction_strict.py` | Strict evaluator + BERTScore |
| `normalize_gold_standard_parameters.py` | Normalise the gold `parameters` field — rewrites it in place |
| `verify_gold_standard.py` | Read-only sanity check of the gold JSON |


---

## Misc

### 11. `list_kisski_models.py`

Lists the models available on the KISSKI Chat AI endpoint via the
OpenAI-compatible `/v1/models`.

```bash
python scripts/list_kisski_models.py
python scripts/list_kisski_models.py -q -o data/kisski_models_list.txt
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--output`, `-o` | path | — | Also save the model ids to this file. |
| `--quiet`, `-q` | flag | off | Print only ids, one per line, with no header or footer. |

**Input:** none.

**Required in `.env`:** `KISSKI_API_KEY`, `KISSKI_BASE_URL` is optional and defaults to
`https://chat-ai.academiccloud.de/v1`.

**Output:** the model list on stdout, plus the file at `--output` if given. Use
it to check what `kisski.model` and `classifier.model` in `config/config.yaml`
may be set to — a model that has been retired from the endpoint returns
`Model Not Found` at extraction time.
