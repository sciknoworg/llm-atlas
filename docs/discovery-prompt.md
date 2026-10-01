# Paper-discovery prompt for ChatGPT (or any web-search AI)

**Prompt version:** `v1` · **Used for:** every quarter in `data/discovery/`
(2020-Q1 … 2026-Q2) · **Service:** ChatGPT's GPT-5.6 Luna with web search enabled

This file is the discovery dataset's provenance record. Treat it as versioned: if the prompt changes, bump the version above
and record which quarters used which version in
[`data/discovery/README.md`](../data/discovery/README.md#provenance), rather
than editing in place — a quarter's paper list is only reproducible against the
prompt that produced it.

Paste the prompt below into ChatGPT with **web search / browsing enabled**. Only
the date range and the result count on the first `TASK` line are edited per run.
The reply is JSON that feeds straight into a quarter's `discovery.json`.

---

## The prompt

```text
You are helping build a research catalogue of large language models (LLMs) and
vision-language models (VLMs).

TASK
Find research papers published between 2026-01 and 2026-06 that INTRODUCE a new
LLM or VLM. Return the 30 most significant.

DATING — this defines the window, apply it before anything else.
Date each paper by its arXiv submission date (the v1 date shown on the arXiv
abstract page), NOT its conference or journal publication date. Do not include a
paper submitted before the window even if it was presented at a conference
during the window. If a paper is not on arXiv, use the publication date of the
venue. When the date is outside the window, leave the paper out entirely — do
not include it with a note.

USE WEB SEARCH FOR EVERY ENTRY. Do not answer from memory. For each paper, open
the actual abstract page (arXiv, ACL Anthology, OpenReview, the publisher's site)
and copy the title and identifier from that page. If you did not open a page for
an entry, leave it out.

INCLUDE a paper only if the paper itself introduces or releases a model:
  - new models and new model families (e.g. a "Technical Report" for a released model)
  - new versions with their own paper (Llama 3 vs Llama 2, Qwen3 vs Qwen2.5)
  - vision-language / multimodal models, not just text-only
  - open-weight and closed models alike

EXCLUDE:
  - surveys, position papers, literature reviews
  - benchmarks, datasets, evaluation suites and leaderboards
  - prompting, fine-tuning, quantisation or agent methods that use somebody
    else's model rather than releasing one
  - papers that only analyse or probe existing models
  - model releases with NO paper (blog-post-only releases) — put those in
    "no_paper_found" instead of inventing a paper for them
  - papers whose arXiv submission date falls outside the window, however well
    known they are and whatever venue they appeared at during it

COVERAGE — search deliberately across all of these, not just the famous few:
  - US labs: OpenAI, Google/DeepMind, Meta, Anthropic, Microsoft, NVIDIA, Apple, AI2
  - Chinese labs: Alibaba/Qwen, DeepSeek, Zhipu/GLM, Moonshot/Kimi, 01.AI,
    Shanghai AI Lab/InternLM, Tencent, ByteDance, Baidu
  - European labs: Mistral, Aleph Alpha, Stability, EPFL/ETH, academic groups
  - Venues beyond arXiv: NeurIPS, ICML, ICLR, ACL, EMNLP, NAACL, CVPR, ICCV,
    ECCV, TMLR, TACL, and journal publications from any publisher

ONE OBJECT PER PAPER, not per model size. A paper covering 8B/70B/405B is ONE
entry.

IDENTIFIERS — give exactly one, preferring arxiv_id:
  - arxiv_id: the bare arXiv ID, e.g. "2407.21783". No "arXiv:" prefix, no
    version suffix. Only if you opened the arXiv abstract page and the title
    matches. Set pdf_url to null.
  - pdf_url: a DIRECT link to the PDF (ending .pdf) for papers not on arXiv —
    ACL Anthology, OpenReview, CVF, or the publisher. Not a landing page.
    Set arxiv_id to null.
Never output an identifier you have not seen on a page. Omitting a paper is
better than guessing its ID: a wrong ID silently pulls the WRONG paper into the
catalogue, and nothing downstream will catch it.

OUTPUT — valid JSON only, no prose, no markdown fences:

{
  "papers": [
    {
      "model_name": "Llama 3",
      "title": "The Llama 3 Herd of Models",
      "arxiv_id": "2407.21783",
      "pdf_url": null
    }
  ],
  "no_paper_found": ["DBRX", "Command R+"]
}

Field rules:
  model_name  the model the paper introduces
  title       the exact title as printed on the page, copied not recalled
  arxiv_id    string or null
  pdf_url     string or null — exactly one of arxiv_id / pdf_url must be set
  no_paper_found  models you know were released in this period but for which you
                  found no paper. This makes gaps visible instead of silent.
```

---

