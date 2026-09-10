# Scripts

Utilities around the extraction pipeline — corpus scoping, paper discovery,
batch extraction, evaluation and ORKG upload — plus `quarterly.py`, the
end-to-end script that runs the catalogue one quarter at a time.

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
│   # Quarterly run
├──  4. quarterly.py                           # discovery -> extraction -> ORKG, one quarter at a time
│
│   # Extraction
├──  5. batch_extract_all_papers.py            # extract every paper in papers_list.json
├──  6. import_extracted_to_model_folders.py   # sort flat extraction JSONs into per-model folders
│
│   # Semantic validation
├──  7. fetch_property_descriptions.py         # refresh the ORKG property-description cache
│
│   # ORKG
├──  8. append_to_paper.py                     # append contributions to an existing ORKG paper
├──  9. sandbox_upload.py                      # upload an extraction JSON as a new ORKG paper
├── 10. reupload_pending.py                    # push already-extracted papers that never reached ORKG
│
│   # Evaluation
├── 11. aggregate_model_evaluation.py          # score one model's extractions against the gold standard
├── 12. build_results_table.py                 # combine per-model scores into one table
│
├── 13. evaluation/                            # gold-standard evaluators — see evaluation/README.md
│
│   # Misc
└── 14. list_kisski_models.py                  # list models available on the KISSKI endpoint
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

## Quarterly run

### 4. `quarterly.py`

The end-to-end script for the catalogue: it walks a quarter's list of papers,
runs the full pipeline over each one (fetch → classify → parse → extract →
merge → validate → map), uploads to ORKG, and records the outcome so
a stopped run resumes where it left off.

Work is organised by quarter — `YYYY-Qn` on the command line, `<year>/Q<n>` on
disk. 

```
data/discovery/
  _index.json              cross-quarter paper index
  processed_papers.csv     the ledger, all quarters
  2020/Q1/
    discovery.json         <- the paper list for this quarter
    extraction/            <- one extraction JSON per paper, written by `run`
```

**Subcommands**

```bash
python scripts/quarterly.py init
python scripts/quarterly.py validate --check-arxiv
python scripts/quarterly.py run --quarter 2024-Q1   # extract, no upload
python scripts/quarterly.py run --all --upload      # extract everything, push to ORKG
python scripts/quarterly.py status
python scripts/quarterly.py recall -v
```

| Command | Does |
|---|---|
| `init` | Create `<year>/Q<n>/extraction/` and an empty `discovery.json` stub for every quarter from 2020-Q1 to now. Existing stubs are never overwritten. |
| `validate` | Report problems without processing anything. Exits `1` if any quarter has problems, so it works as a pre-flight check. |
| `run` | The actual work. Needs `--quarter` or `--all`. |
| `status` | Per quarter: papers listed, extractions on disk, failures counted from the ledger, and the quarter's free-text `notes`. Ends with the unique-paper count in `_index.json`. |
| `recall` | Bucket the gold standard by arXiv **submission** date and compare against what each quarter listed, so a poor draw shows up as low recall rather than just a short list. Dates are cached in `data/gold_standard/papers_submission_dates.json`. |

**Options**

| Option | Command | Type | Default | Description |
|---|---|---|---|---|
| `--until` | `init` | `YYYY-Qn` | current quarter | Last quarter to create. |
| `--quarter` | `validate`, `run` | `YYYY-Qn` | every quarter with a folder | Restrict to one quarter. |
| `--check-arxiv` | `validate` | flag | off | One batched arXiv call per quarter: each id resolves, its title matches the listed one, and it was **submitted** inside the quarter. |
| `--verbose`, `-v` | `validate`, `recall` | flag | off | Also print clean quarters / list the missed gold papers. |
| `--all` | `run` | flag | off | Every quarter that has a folder. |
| `--upload` | `run` | flag | off | Upload to ORKG. Without it, extraction is written to disk only. |
| `--redo` | `run` | flag | off | Re-process papers that already have an extraction file. |
| `--skip` | `run` | ids | none | Never process these arXiv ids, even with `--redo`; their existing results still go into the ledger. |
| `--only` | `run` | ids | none | Process **only** these ids. With `--all`, re-runs scattered papers without naming their quarters. |
| `--dry-run` | `run` | flag | off | List what would run; touch nothing. |
| `--force` | `run` | flag | off | Proceed when the list has unusable entries (they are skipped individually). |
| `--classify` | `run` | flag | off | Domain classifier in **advisory** mode: verdict recorded in the ledger, rejection ignored. |
| `--classify-strict` | `run` | flag | off | Classifier as a **gate** — rejected papers are not extracted. |

Ids in `--skip`/`--only` are normalised the way the runner keys papers, so
`2001.07966`, `2001.07966v3` and `arXiv:2001.07966` all name the same paper.

`validate` catches missing titles, entries with neither `arxiv_id` nor
`pdf_url`, malformed ids, and the same paper listed in two quarters.
`--check-arxiv` adds the two failure modes seen in practice: an id belonging to
a *different* paper, and a paper filed under its **venue's** quarter rather
than its submission (VL-BERT is on arXiv from Aug 2019, at ICLR in Apr 2020).


**Input:** `discovery.json` per quarter, plus PDFs downloaded to `data/papers/`.

**Required in `.env`:** `KISSKI_API_KEY` (plus any extra pool keys); `ORKG_EMAIL`
and `ORKG_PASSWORD` additionally when `--upload` is passed.

**Outputs — three, all written after every paper so an interrupted run loses nothing:**

| Path | Contents |
|---|---|
| `<quarter>/extraction/<arxiv_id>.json` | The full pipeline result, one file per paper. |
| `data/discovery/processed_papers.csv` | The ledger, and the single record of what a run did: one row per paper with its model names, ORKG paper and contribution ids, classifier verdict, and status/error. Rewritten whole via a temp file, so a re-processed paper updates its row instead of gaining a second one. |
| `data/discovery/_index.json` | Which quarter each paper was processed in — what stops a paper listed in two consecutive quarters from being extracted twice. |

Re-running is the normal case: a paper with an extraction file is skipped, but
its ledger row is still rebuilt from the saved result, so the CSV stays
complete rather than covering only the papers that run happened to touch.

---

## Extraction

### 5. `batch_extract_all_papers.py`

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

### 6. `import_extracted_to_model_folders.py`

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

## Semantic validation

The pipeline checks each extracted value against the *meaning* of its property
and drops values that are not an instance of it — `"General RL"` for
`rl_algorithm`, or the trailing half of a split sentence for
`pretraining_corpus`. Three layers run cheapest-first (`src/value_rules.py` →
`src/value_cache.py` → `src/semantic_validator.py`), configured under
`semantic_validation:` in `config/config.yaml`. The property definitions come
from ORKG and are cached on disk so extraction never calls it; this script
refreshes that cache.

### 7. `fetch_property_descriptions.py`

Pulls the `description` of every template predicate from ORKG and writes them
to `data/property_descriptions.json`.

```bash
python scripts/fetch_property_descriptions.py
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--output` | path | `data/property_descriptions.json` | Where to write the cache JSON. |

**Input:** the field→predicate mappings in `src/template_mapper.py`. No
credentials — `GET /api/predicates/{id}` is public.

**Output:** `data/property_descriptions.json`. Entries are keyed by **field
name** (`rl_algorithm`) rather than predicate id, because sandbox and
production use different ids for the same concept (`P203004` vs `P203093`)
while the meaning is identical; each records the description, its source host
and the predicate id. Production is queried first, sandbox as a fallback, since
a few predicates have a usable description on one instance and junk on the
other.

Run it whenever predicates are added or their descriptions edited.

---

## ORKG

Both upload scripts default to **`sandbox`**. Pass `--host production` to write
to the live system.

### 8. `append_to_paper.py`

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

### 9. `sandbox_upload.py`

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
  deduplication, so use `8. append_to_paper.py` when the paper already exists.
- **production** — the manager first searches for an existing paper with the
  exact same title and, if it finds one, adds only the contribution labels that
  are missing rather than creating a duplicate.

The configured comparison is **not** touched either way: this script builds the
manager without comparison arguments, so the comparison update stays off.

### 10. `reupload_pending.py`

Pushes papers that were extracted successfully but never reached ORKG — the
recovery tool for an upload outage. Everything an upload needs was saved at
extraction time, so this replays the stored JSON: no PDF fetching, no
classification, no LLM calls, no KISSKI quota.

```bash
python scripts/reupload_pending.py --dry-run      # show what would upload
python scripts/reupload_pending.py --limit 1      # one paper, to prove it works
python scripts/reupload_pending.py                # the rest
```

**Options**

| Option | Type | Default | Description |
|---|---|---|---|
| `--ledger` | path | `data/discovery/processed_papers.csv` | Ledger to read the work list from. |
| `--limit` | int | all | Only process the first N pending papers. |
| `--arxiv-id` | str | all | Upload only this arXiv id. Repeatable. |
| `--dry-run` | flag | off | List what would be uploaded and whether its extraction file exists. Touches nothing. |
| `--delay` | float | `1.0` | Seconds between uploads. |

**Input:** the quarterly ledger, plus the saved extraction at
`data/discovery/<year>/Q<n>/extraction/<arxiv_id>.json` for each pending row.

**Required in `.env`:** `ORKG_EMAIL` and `ORKG_PASSWORD`.

**Output:** ORKG papers and contributions, with the resulting ids written back
into both the ledger row and the saved extraction JSON — that write-back is
what makes the run resumable. Exits `1` if any upload failed.

---

## Evaluation

### 11. `aggregate_model_evaluation.py`

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

### 12. `build_results_table.py`

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

### 13. `evaluation/`

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

### 14. `list_kisski_models.py`

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
