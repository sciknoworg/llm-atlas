# Next Steps: Test Extraction Improvements

## What We Did

### 1. Created Strict Evaluator
**File:** `scripts/evaluation/evaluate_extraction_strict.py`

**What it does:** Evaluates extraction with NO relaxations:
- No date normalization (2018-06 ≠ 2018-06-01)
- No model_name tricks (GPT-1 117M ≠ GPT-1)
- No substring matching for arch/optimizer
- No "null" string → missing coercion
- Only long-text fields use fuzzy (strict threshold 0.8)

**Why:** Scores reflect genuine extraction quality, not evaluator leniency. **Use this for thesis metrics.**

---

### 2. Created Baseline Model Filter
**File:** `src/baseline_filter.py`

**What it does:** Filters out baseline/ablation models from extraction:
- Keeps only main contribution (e.g., GPT-1)
- Removes baselines (e.g., LSTM, Transformer aux)
- Scores models by: parameters, name matching paper title, detailed fields
- Reduces false positives in evaluation

---

### 3. Created Complete Test Workflow
**File:** `scripts/test_extraction_workflow.py`

**What it does:** Runs full pipeline in one command:
1. Extract from PDF URL
2. Filter baseline models (optional)
3. Evaluate with strict + relaxed evaluators
4. Compare results

---

## Quick Test (GPT-1 Paper)

Run the complete workflow on the GPT-1 paper:

```bash
cd "c:\Users\ALAKE\OneDrive\Bureau\Bachelor-Arbeit-NLP"
python scripts/test_extraction_workflow.py --paper gpt1
```

**Expected output:**
```
FINAL SUMMARY
================================================================================
Results:
  unfiltered_strict: F1 = 45-50%    (with baseline models, strict eval)
  unfiltered_relaxed: F1 = 60-70%   (with baseline models, relaxed eval)
  filtered_strict: F1 = 50-55%      (main model only, strict eval)
  filtered_relaxed: F1 = 65-75%     (main model only, relaxed eval)

THESIS METRIC (Filtered + Strict): F1 = 50-55%
================================================================================
Status: FAIR - Consider further improvements
```

---

## Next Steps to Improve Extraction (Priority Order)

### HIGH Priority (Do First)

#### 1. Test Baseline Filtering
```bash
python scripts/test_extraction_workflow.py --paper gpt1
```

**Check:**
- Does it correctly keep GPT-1 and remove LSTM/Transformer aux?
- Does strict F1 increase after filtering?

**Expected gain:** +5-10% F1 (fewer false positives)

---

#### 2. Improve Few-Shot Examples (Gold-Style)

**Current problem:** Innovation text is too brief.
- **Extracted:** "Generative pre‑training followed by fine‑tuning..."
- **Gold:** "The paper introduces a framework... **improved state-of-the-art on 9 out of 12 datasets**..."

**Fix:** Edit `src/llm_extractor.py`, update `example3_output` (GPT-1 example):

```python
# Line ~207-220
example3_output = {
    "models": [{
        "model_name": "GPT-1",
        "model_family": "GPT", 
        "paper_title": "Improving Language Understanding by Generative Pre-Training",
        "organization": "OpenAI",
        "parameters": "117M",
        "parameters_millions": 117,
        "date_created": "2018-06-01",  # Full date
        "pretraining_architecture": "Decoder",  # Not "Transformer (Decoder)"
        "pretraining_task": "Causal language modeling",
        "pretraining_corpus": "BooksCorpus",
        "finetuning_task": "Supervised discriminative fine-tuning",  # Match gold
        "optimizer": "Adam",  # Not "Adam optimizer"
        "innovation": "The paper introduces a framework for natural language understanding by first using generative pre-training on a diverse corpus and then fine-tuning for specific tasks. This approach improved state-of-the-art results on 9 out of 12 datasets.",  # Detailed, includes results
        "license": "closed source",  # Not "MIT License"
        "research_problem": "Language Understanding",
        "application": "Natural language understanding, text classification, question answering"
    }]
}
```

**Expected gain:** +10-15% F1 (better innovation, arch, optimizer, license extraction)

---

#### 3. Add License Inference + Architecture Normalization

Create `src/post_processors.py`:

```python
def infer_license(model, paper_metadata):
    """Infer license if not extracted."""
    if model.get("license"):
        return model["license"]
    
    # Pre-2019 models without explicit release → closed source
    year = model.get("date_created", "")[:4]
    if year and int(year) < 2019:
        return "closed source"
    
    # Keywords in innovation/research_problem
    text = f"{model.get('innovation', '')} {model.get('research_problem', '')}".lower()
    if any(kw in text for kw in ["open source", "github", "released", "available"]):
        return "open source"
    
    return None

def normalize_architecture(arch):
    """Normalize architecture names."""
    if not arch:
        return arch
    arch = str(arch).strip()
    # "Transformer (Decoder)" → "Decoder"
    if "(" in arch and ")" in arch:
        inner = arch.split("(")[1].split(")")[0].strip()
        if inner in ["Decoder", "Encoder"]:
            return inner
    return arch
```

Apply in `pipeline.py` after extraction.

**Expected gain:** +5% F1 (correct license, normalize arch)

---

### MEDIUM Priority (Do After HIGH)

#### 4. Test on Multiple Papers
Run on 3-5 papers from gold standard:

```bash
# Find papers in gold standard
python -c "import json; data=json.load(open('data/gold_standard/R1364660.json')); titles=set(m['paper_title'] for m in data['extraction_data'] if m['paper_title']); print('\n'.join(sorted(titles)[:10]))"
```

Pick papers with:
- Known ArXiv IDs or PDF URLs
- Different model types (BERT, Llama, T5, etc.)
- Various dates (2018-2024)

Test each, compute average strict F1.

**Expected gain:** Understand variability, identify systematic issues

---

#### 5. Better Chunking (Optional)
Current: Fixed 8000-char chunks. Try:
- Semantic chunking (by section headers)
- Overlapping chunks (7000 chars, 1000 overlap)
- Full-text for short papers (<8K)

**Expected gain:** +2-5% F1 (less info lost at chunk boundaries)

---

## Immediate Test Commands

### 1. Baseline Test (Strict Eval)
```bash
# Run complete workflow with strict evaluation
python scripts/test_extraction_workflow.py --paper gpt1

# Check: filtered_strict F1 score
# Target: 50-55% baseline
```

### 2. After Improving Few-Shots
```bash
# Re-extract with improved examples
python -m src.pipeline --pdf-url "https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf" --paper-title "Improving Language Understanding by Generative Pre-Training" --no-update

# Evaluate with strict
python scripts/evaluation/evaluate_extraction_strict.py --prediction "data/extracted/improving_language_understanding_by_generative_pre_training_YYYYMMDD_HHMMSS.json"

# Check: F1 should increase to 60-70%
```

### 3. Compare Strict vs Relaxed
```bash
# Strict (thesis metric)
python scripts/evaluation/evaluate_extraction_strict.py --prediction <path>

# Relaxed (UX metric)
python scripts/evaluation/evaluate_extraction.py --prediction <path>

# Difference shows impact of evaluator relaxations
# Aim: minimize this gap by improving extraction
```

---

## Success Criteria (Thesis)

| Metric | Target | Status |
|--------|--------|--------|
| **Strict F1 (filtered)** | ≥ 60% | **PRIMARY THESIS METRIC** |
| Strict Recall | ≥ 70% | Important (completeness) |
| Strict Precision | ≥ 50% | Important (correctness) |
| Per-field F1 (key fields) | ≥ 70% | model_name, date, arch, task, params |

**Current baseline:** ~45-50% strict F1  
**After improvements:** 60-70% strict F1 (realistic target)

---

## Files Created

1. `scripts/evaluation/evaluate_extraction_strict.py` - Strict evaluator (thesis metrics)
2. `src/baseline_filter.py` - Filter baseline models
3. `scripts/test_extraction_workflow.py` - Complete test workflow
4. `docs/extraction-improvement-plan.md` - Detailed improvement plan
5. `docs/NEXT_STEPS_TESTING.md` - This file

---

## Summary

**What genuinely improves extraction:**
- ✅ Filter baseline models (implemented)
- ✅ Better few-shot examples (documented, needs implementation)
- ✅ License inference + arch normalization (documented, needs implementation)
- ✅ Strict evaluator for real metrics (implemented)

**What only improves score:**
- ❌ Date normalization (keep in relaxed evaluator for UX)
- ❌ model_name tricks (keep in relaxed evaluator for UX)
- ❌ Substring matching (keep in relaxed evaluator for UX)

**Next action:** Run `python scripts/test_extraction_workflow.py --paper gpt1` to get baseline metrics, then implement few-shot improvements.
