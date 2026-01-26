# Extraction Improvement Plan (Best Practices)

## Current State (GPT-1 Extraction)

**Genuine extraction issues:**
1. **License wrong** – Extracted "MIT License", gold is "closed source"
2. **Innovation too brief** – Missing key details like "state-of-the-art on 9 out of 12 datasets"
3. **Extra models extracted** – LSTM, Transformer aux (baseline/ablation models, not main contribution)
4. **pretraining_architecture verbose** – "Transformer (Decoder)" vs gold "Decoder"
5. **finetuning_task wording** – Different phrasing from gold
6. **Blog post missing** – Gold has curator-added URLs; not in PDF (acceptable gap)

**Score-only fixes (don't improve extraction):**
- Date normalization (2018-06 ≈ 2018-06-01)
- model_name strip (GPT-1 (117M) ≈ GPT-1)
- Substring match for arch/optimizer
- Lower fuzzy threshold

---

## Best Practices to Improve Extraction

### 1. **Filter baseline/ablation models**
Papers often describe multiple models: main contribution + baselines/ablations for comparison. Gold standard typically contains **only the main model**.

**Fix:** Post-process to keep only models that:
- Match the paper title or main contribution keywords
- Have the most features/parameters described
- Aren't explicitly labeled as baseline/ablation

**Example:** GPT-1 paper describes:
- GPT-1 (main) ✓ keep
- LSTM baseline ✗ filter out
- Transformer aux ablation ✗ filter out

---

### 2. **Improve few-shot examples to match gold standard style**

Current few-shot examples use paper-style innovation text. Gold standard uses **curator-style** (more detailed, includes results).

**Current extraction:**
> "Generative pre‑training on a large unlabeled corpus followed by discriminative fine‑tuning..."

**Gold standard:**
> "The paper introduces a framework for natural language understanding by first using generative pre-training on a diverse corpus and then fine-tuning for specific tasks. **This approach improved state-of-the-art results on 9 out of 12 datasets**, highlighting the potential of unsupervised learning combined with discriminative tasks."

**Fix:** Add few-shot examples with gold-style innovation text (detailed, includes impact/results).

---

### 3. **Add license inference rules**

Extraction picked "MIT License" (possibly hallucinated or from references). Gold is "closed source" (correct for GPT-1).

**Fix:**
- If no license mentioned in paper → "closed source" (default for pre-2019 models)
- If paper says "we release code/model" → check for license in text
- If arXiv/open-source keywords → "open source"
- Post-2020 + release mentioned → likely open source

---

### 4. **Normalize architecture names in extraction**

Current: "Transformer (Decoder)" → should be just "Decoder" to match gold.

**Fix:** Post-process `pretraining_architecture`:
- "Transformer (Decoder)" → "Decoder"
- "Transformer (Encoder)" → "Encoder"
- Keep "Transformer" if no sub-type specified

---

### 5. **Better chunking strategy**

Current: Fixed-size chunks (8000 chars). Key info might be split.

**Fix options:**
- Semantic chunking (by section: intro, method, results)
- Overlapping chunks
- Full-text extraction for short papers (<8K)

---

### 6. **Create strict evaluator for real accuracy**

Current evaluator has many relaxations (date normalization, model_name tricks, substring match, lower fuzzy).

**Fix:** Create `evaluate_extraction_strict.py` with:
- Exact string match (no fuzzy, no substring)
- Strict date format (YYYY-MM-DD only)
- No "null" → missing coercion
- Use **this** for thesis evaluation metrics

---

## Implementation Priority

| Priority | Task | Impact | Effort |
|----------|------|--------|--------|
| **HIGH** | Filter baseline models | Large (removes false positives) | Low |
| **HIGH** | Improve few-shot examples (gold-style innovation) | Large (better field extraction) | Medium |
| **HIGH** | Create strict evaluator | Large (accurate metrics) | Low |
| **MEDIUM** | License inference rules | Medium (1 field) | Low |
| **MEDIUM** | Normalize architecture names | Medium (1 field) | Low |
| **LOW** | Better chunking | Small (incremental) | High |

---

## Test Protocol

### Phase 1: Baseline (Strict Evaluation)
1. Keep extractor improvements (prompt, dedup, null coercion)
2. Create strict evaluator (no relaxations)
3. Extract GPT-1 paper
4. **Baseline F1** = score with strict eval

### Phase 2: Implement Extraction Improvements
1. Filter baseline models
2. Add gold-style few-shot examples
3. Add license inference + architecture normalization
4. Re-extract GPT-1 paper
5. **Improved F1** = score with strict eval

### Phase 3: Validate on Multiple Papers
1. Test on 3-5 papers from gold standard
2. Average F1 across papers
3. Compare strict vs relaxed evaluator scores
4. Use **strict** scores in thesis

---

## Next Steps (Immediate)

1. **Create strict evaluator** (`scripts/evaluation/evaluate_extraction_strict.py`)
2. **Re-run GPT-1 extraction** with current extractor
3. **Evaluate with strict evaluator** → baseline F1
4. **Implement top 3 priorities** (filter baselines, improve few-shots, license/arch fixes)
5. **Re-extract + re-evaluate** → measure real improvement
6. **Document results** for thesis

---

## Expected Outcomes

**With strict evaluation:**
- Baseline F1: ~45-50% (current extraction, strict grading)
- After improvements: ~60-70% (genuine extraction improvements)

**Remaining gaps:**
- blog_post (not in PDF, curator-added)
- Some wording differences (innovation, research_problem)
- pretraining_corpus / application (gold might be incomplete)

These are acceptable limitations to document in thesis.
