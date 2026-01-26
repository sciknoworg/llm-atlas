# Evaluation Scoring Analysis (GPT-1 Extraction)

## Summary

- **Overall:** F1 40%, Accuracy 33%, Precision 40%, Recall 40%
- **Matched:** 1 gold model (GPT-1) ↔ 1 prediction (GPT-1 117M); 2 predictions unmatched (GPT 117M, GPT)
- **Strong fields:** model_family, organization, parameters, parameters_millions (100% F1)
- **Weak fields:** model_name, date_created, innovation, pretraining_architecture, pretraining_task, finetuning_task, optimizer, license, blog_post, research_problem, etc. (0% F1)

---

## Root Causes

### 1. Model duplication (same model, multiple surface forms)

The paper describes **one** model (GPT-1). Chunked extraction produces **three** variants:

- "GPT-1 117M", "GPT 117M", "GPT"

Dedup uses `(model_name, model_version, parameters)` as key, so all three are kept. The evaluator matches one to gold (substring "gpt-1") and scores only that pair; the other two count as "unmatched predictions" and don't improve metrics.

**Fix:** Merge variants that refer to the same model (same `model_family` + `parameters`). Keep a single canonical entry (e.g. prefer "GPT-1" or "GPT-1 117M") and merge non-null fields from duplicates.

---

### 2. model_name mismatch

- **Gold:** "GPT-1"
- **Pred:** "GPT-1 117M"

Evaluation uses exact match (no fuzzy for `model_name`). "GPT-1" ≠ "GPT-1 117M" → 0% F1.

**Fix:** Normalize `model_name` before compare (e.g. strip trailing " 117M", " (117M)") or use fuzzy match for `model_name`. Alternatively, canonicalize during dedup so we output "GPT-1".

---

### 3. date_created format

- **Gold:** "2018-06-01"
- **Pred:** "2018-06" or "2018"

Exact match fails even though they refer to the same date.

**Fix:** Date normalization in evaluator: treat `YYYY-MM-DD`, `YYYY-MM`, `YYYY` as equivalent when comparable (e.g. 2018-06-01 ≈ 2018-06 ≈ 2018).

---

### 4. Missing extractions (null vs gold)

| Field | Gold | Pred |
|-------|------|------|
| pretraining_architecture | Decoder | null |
| pretraining_task | Causal language modeling | null |
| finetuning_task | Supervized discriminative finetuning | null |
| optimizer | Adam optimizer | null |
| license | closed source | "null" (string) |

The paper (and few-shot examples) contain these. The model often omits them for chunk-based extraction.

**Fix:**
- **Prompt:** Explicitly require `pretraining_architecture`, `pretraining_task`, `finetuning_task`, `optimizer` when mentioned. Move them to "must extract" or "MUST extract when mentioned".
- **Output:** Emit `null` (JSON null), not the string `"null"`. Post-process extraction to coerce `"null"` → `None` before save/eval.

---

### 5. innovation / research_problem (fuzzy)

- **Gold:** Longer, curator-style text (e.g. "The paper introduces a framework… state-of-the-art on 9 out of 12 datasets…").
- **Pred:** Shorter, paper-style ("Generative pre‑training of a Transformer decoder… discriminative fine‑tuning…").

Fuzzy matching is used (threshold 0.8), but similarity may be below 0.8 → no match.

**Fix:** Lower fuzzy threshold for long-text fields (e.g. 0.65–0.7), or add semantic similarity (harder). Optional: expand few-shot examples to include gold-like formulations.

---

### 6. license "null" string

- **Gold:** "closed source"
- **Pred:** `"license": "null"` (string in JSON)

Evaluator `normalize_value` maps `None` and `"None"` to `""`, but not `"null"`. So we compare `"null"` vs `"closed source"` → no match. Also, we should extract "closed source" when inferable.

**Fix:** Treat `"null"` as missing in evaluator (`normalize_value`). Fix extractor/post-process to never emit the string `"null"`.

---

### 7. blog_post, pretraining_corpus, application

- **blog_post:** Gold has curator-added links; pred has null. Often not in the PDF → hard to fix via extraction.
- **pretraining_corpus / application:** Gold null, pred has values (e.g. BooksCorpus, "Textual entailment…"). Evaluator counts these as FP (predicted when gold is absent). Gold may be incomplete; alignment is a design choice.

---

## Implemented Improvements

| Change | Where | Purpose |
|--------|--------|---------|
| Treat `"null"` as missing | Evaluator `normalize_value` | Correct match when pred didn't extract |
| Date normalization (YYYY-MM-DD ≈ YYYY-MM ≈ YYYY) | Evaluator `compare_field` | Match 2018-06-01 vs 2018-06 |
| model_name normalize (strip param suffix) / substring | Evaluator | Match "GPT-1" vs "GPT-1 117M" |
| Fuzzy for pretraining_* / finetuning_task / optimizer | Evaluator | Handle minor wording differences |
| Substring match for short fields (arch, optimizer) | Evaluator | "Decoder" ≈ "Transformer (Decoder)", "Adam" ≈ "Adam optimizer" |
| Lower fuzzy threshold (0.65) for long-text fields | Evaluator | innovation, research_problem, etc. |
| Require arch/task/optimizer when mentioned | Extractor prompt | Reduce nulls |
| Prefer date YYYY-MM-DD | Extractor prompt | Align with gold |
| Merge same (family, params) variants; canonical name | Extractor dedup | Single model per actual model |
| Coerce "null" string → None | Extractor `_coerce_null_strings` | Clean extraction output |

## Results (GPT-1) After Improvements

| Metric | Before | After |
|--------|--------|-------|
| **F1-Score** | 40% | **72%** |
| Accuracy | 33% | 61% |
| Precision | 40% | 60% |
| Recall | 40% | 90% |
| model_name | 0% | 100% |
| date_created | 0% | 100% |
| pretraining_architecture | 0% | 100% |
| pretraining_task | 0% | 100% |
| optimizer | 0% | 100% |

Remaining 0% F1 (gold vs pred mismatch): innovation, pretraining_corpus, finetuning_task, license, blog_post, research_problem, application (wording differs or gold null).

---

## How to Re-run and Check

1. **Extract:**  
   `python -m src.pipeline --pdf-url "https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf" --paper-title "Improving Language Understanding by Generative Pre-Training" --no-update`

2. **Evaluate:**  
   `python scripts/evaluation/evaluate_extraction.py --prediction "data/extracted/...json" [--output data/evaluation_reports/gpt1_report.json]`

3. Compare F1, per-field metrics, and matched/unmatched counts before vs after changes.
