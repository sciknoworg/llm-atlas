# Quarterly discovery input

One folder per quarter, holding the ChatGPT result and everything produced from it.

```
data/discovery/
  _index.json            cross-quarter record of every paper already processed
  processed_papers.csv   the ledger — one row per paper, all quarters
  2020/
    Q1/
      discovery.json     <- you paste ChatGPT's JSON here
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

## discovery.json

Paste ChatGPT's reply into `papers`. Everything else is optional — the runner
fills `quarter` from the folder name if it is missing.

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

# 2. paste each ChatGPT reply into data/discovery/<quarter>/discovery.json

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
