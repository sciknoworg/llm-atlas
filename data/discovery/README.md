# Quarterly discovery input

One folder per quarter, holding the discovery result and everything produced
from it. How those results were obtained — prompt, service, criteria, run dates
and how repeated runs were reconciled — is in [Provenance](#provenance) below;
the prompt itself is versioned at
[`docs/discovery-prompt.md`](../../docs/discovery-prompt.md).

```
data/discovery/
  _index.json            cross-quarter record of every paper already processed
  processed_papers.csv   the ledger — one row per paper, all quarters
  2020/
    Q1/
      discovery.json     <- the quarter's paper list (see Provenance)
      extraction/        <- one extraction JSON per paper, written by the runner
    Q2/
    Q3/
    Q4/
  2021/
  ...
```

Quarters live under their year, so `ls data/discovery/2022` shows one year at a
time. Commands still take the `YYYY-Qn` form (`--quarter 2022-Q1`); the script
maps that to the folder. A single glob `data/discovery/*/Q[1-4]/discovery.json`
reaches every input file.

## Provenance

`data/discovery/**` is committed as a research dataset, so how each paper list
was assembled is part of the record. Every list here came from the prompt in
[`docs/discovery-prompt.md`](../../docs/discovery-prompt.md)

| | |
|---|---|
| **Discovery prompt** | [`docs/discovery-prompt.md`](../../docs/discovery-prompt.md), version `v1`. Only the date range and the result count on the first `TASK` line change per run. |
| **Service** | ChatGPT's GPT-5.6 Luna with web search / browsing **enabled**. The prompt requires an abstract page to be opened per entry and forbids answering from memory. |
| **Discovery runs** | One query per quarter against the service above. All 27 quarter lists were in place by 2026-08-19, ahead of extraction.
| **Extraction runs** | 2026-08-24 … 2026-09-11, per `processed_at` in `processed_papers.csv`. |
| **Coverage** | 2020-Q1 … 2026-Q2, i.e. arXiv submissions 2020-01-01 … 2026-06-30. 290 papers in the ledger. |
| **Window convention** | A paper belongs to the quarter of its **arXiv v1 submission date**, not its conference or journal date. For papers not on arXiv, the venue's publication date. |

### Inclusion / exclusion criteria

The authoritative wording is in the prompt; this is the summary a reader needs
in order to judge the dataset.

**In** — papers that themselves introduce or release a model: new models and
model families (including a "Technical Report" for a released model), new
versions with their own paper (Llama 3 vs Llama 2, Qwen3 vs Qwen2.5),
vision-language and multimodal models as well as text-only, open-weight and
closed models alike.

**Out** — surveys, position papers and literature reviews; benchmarks, datasets,
evaluation suites and leaderboards; prompting, fine-tuning, quantisation or
agent methods built on somebody else's model; papers that only analyse or probe
existing models; model releases with no paper at all; and any paper whose
submission date falls outside the window, however well known it is and whatever
venue it appeared at during the window.

One object per **paper**, not per model size — a paper covering 8B/70B/405B is a
single entry. Exactly one identifier per entry, `arxiv_id` preferred over
`pdf_url`.

Model releases with no paper are not dropped silently: they go into
`no_paper_found`, so a gap in the catalogue stays visible and attributable.

### How repeated and merged runs were reconciled

The model does not return a stable list, so some quarters were queried more than
once. Four rules were applied, and every application is recorded in that
quarter's free-text `notes` field (shown by `quarterly.py status`):

1. **Union, not replacement.** A second run's results are merged into the
   first's; a paper the re-run dropped is kept if it was independently verified
   as in-window. 
2. **Submission date wins over the query that found it.** A paper returned for
   one quarter but submitted in another is moved to the quarter it belongs to.
3. **One extraction per paper, catalogue-wide.** `_index.json` records which
   quarter each paper was processed in, so a paper the model repeats across
   adjacent quarters is extracted once.

### Auditing this dataset

```bash
# every id resolves, titles match, submissions fall inside their quarter
python scripts/quarterly.py validate --check-arxiv

# per quarter: papers listed / extracted / failed, plus the reconciliation notes
python scripts/quarterly.py status

```

## discovery.json

Paste the reply's `papers` and `no_paper_found` arrays into the quarter's file.
Everything else is optional — the runner fills `quarter` from the folder name if
it is missing.

```json
{
  "quarter": "2020-Q1",
  "papers": [
    {
      "model_name": "GPT-3",
      "title": "Language Models are Few-Shot Learners",
      "arxiv_id": "2005.14165",
      "pdf_url": null
    },
    {
      "model_name": "SomeConfModel",
      "title": "A Model Published at ACL",
      "arxiv_id": null,
      "pdf_url": "https://aclanthology.org/2020.acl-long.1.pdf"
    }
  ],
  "no_paper_found": ["DBRX", "Command R+"]
}
```

An optional top-level `notes` string records anything non-obvious about how the
list was assembled — a merge of two query runs, a paper moved in from another
quarter. It is free text, shown by `status`.

Per paper, exactly one identifier:

| field | meaning |
|---|---|
| `title` | **required** — the exact paper title. Needed for `--pdf-url` and to verify an `arxiv_id` points at the right paper |
| `arxiv_id` | bare id, e.g. `2005.14165`. No `arXiv:` prefix, no `v2` suffix |
| `pdf_url` | direct link ending `.pdf`, for papers not on arXiv |
| `model_name` | the model introduced; used to resolve the paper when the title is wrong |

`no_paper_found` records models with no paper (blog-only releases), so a gap is
visible rather than silently missing.

## Workflow

```bash
# 1. create every quarter folder from 2020-Q1 to now
python scripts/quarterly.py init

# 2. run docs/discovery-prompt.md for the quarter and paste the reply into
data/discovery/<year>/Q<n>/discovery.json

# 3. check the file parses and the identifiers resolve — no extraction, no upload
python scripts/quarterly.py validate --quarter 2020-Q1

# 4. extract (add --upload to also push to ORKG)
python scripts/quarterly.py run --quarter 2020-Q1

# 5. see where everything stands
python scripts/quarterly.py status
```

Runs are resumable: a paper whose extraction file already exists is skipped, so
an interrupted quarter can be re-run and only picks up what is left.

Papers already processed in an earlier quarter are skipped too — ChatGPT tends
to repeat notable papers across adjacent quarters, and `_index.json` catches
that before any PDF is downloaded.
