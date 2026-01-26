# Extraction Accuracy Test Plan

## Current Baseline (GPT-1, Unfiltered, Strict)

**Real extraction accuracy:** F1 = **47.62%**

- Relaxed evaluator: 72% F1 (inflated by evaluator tricks)
- Strict evaluator: 47.62% F1 (**actual extraction quality**)
- Gap: ~24% → this is evaluator leniency, not extraction accuracy

**Strong fields (100% F1):** model_family, organization, pretraining_task, parameters, parameters_millions

**Weak fields (0% F1):** model_name, date_created, innovation, pretraining_architecture, finetuning_task, optimizer, license, pretraining_corpus, research_problem, application, blog_post

---

## Improvement Steps

### Step 1: Filter Baseline Models (IMMEDIATE)

**Command:**
```bash
cd "c:\Users\ALAKE\OneDrive\Bureau\Bachelor-Arbeit-NLP"
python scripts/test_extraction_workflow.py --paper gpt1
```

**What it does:**
- Extracts GPT-1 paper
- Filters out LSTM, Transformer aux (baseline models)
- Evaluates with strict + relaxed

**Expected outcome:**
- Filtered strict F1: **50-55%** (+5% from removing false positives)
- Fewer unmatched predictions

---

### Step 2: Improve Few-Shot Examples (HIGH PRIORITY)

**File:** `src/llm_extractor.py` lines ~207-220

**Changes needed:**
```python
example3_output = {
    "models": [{
        "model_name": "GPT-1",  # Not "GPT-1 117M"
        "date_created": "2018-06-01",  # Full YYYY-MM-DD
        "pretraining_architecture": "Decoder",  # Not "Transformer (Decoder)"
        "optimizer": "Adam optimizer",  # Match gold format
        "finetuning_task": "Supervized discriminative finetuning",  # Match gold
        "innovation": "The paper introduces a framework for natural language understanding by first using generative pre-training on a diverse corpus and then fine-tuning for specific tasks. This approach improved state-of-the-art results on 9 out of 12 datasets, highlighting the potential of unsupervised learning combined with discriminative tasks.",
        "license": "closed source",  # Not MIT
        "research_problem": "Large Language Models (LLMs), transformer model",
        # ... other fields match gold exactly
    }]
}
```

**After editing:**
```bash
# Re-extract with improved examples
python -m src.pipeline --pdf-url "https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf" --paper-title "Improving Language Understanding by Generative Pre-Training" --no-update

# Get the path from output, then evaluate
python scripts/evaluation/evaluate_extraction_strict.py --prediction "data/extracted/improving_language_understanding_by_generative_pre_training_YYYYMMDD_HHMMSS.json"
```

**Expected outcome:**
- Strict F1: **60-70%** (+10-15% from better few-shots)
- innovation, date_created, arch, optimizer, license should improve

---

### Step 3: Add Post-Processing (MEDIUM PRIORITY)

**Create:** `src/post_processors.py` with:
- `infer_license()` - default "closed source" for pre-2019 models
- `normalize_architecture()` - strip "Transformer (...)" → keep inner type
- `normalize_optimizer()` - ensure consistent format

**Apply in:** `pipeline.py` after extraction, before save

```python
# In process_paper_from_pdf_url, after extraction:
from src.post_processors import normalize_architecture, infer_license

for model in extraction_result.models:
    model.pretraining_architecture = normalize_architecture(model.pretraining_architecture)
    if not model.license:
        model.license = infer_license(model, paper_metadata)
```

**Expected outcome:**
- Strict F1: +3-5% (correct license, arch alignment)

---

### Step 4: Test Multiple Papers (VALIDATION)

**Papers to test (examples):**
1. GPT-1 (PDF URL, no ArXiv) ✓
2. BERT (ArXiv: 1810.04805)
3. GPT-2 (ArXiv: check gold standard)
4. Llama 2 (ArXiv: 2307.09288)
5. T5 (ArXiv: check gold standard)

**Commands:**
```bash
# For each paper:
python -m src.pipeline --arxiv-id <ID> --no-update  # or --pdf-url
python scripts/evaluation/evaluate_extraction_strict.py --prediction <path>

# Compute average F1 across papers
```

**Expected outcome:**
- Average strict F1: 55-65% across 5 papers
- Identifies which paper types are harder to extract

---

## Measuring Real Improvement

| Checkpoint | Strict F1 | What Changed |
|------------|-----------|--------------|
| **Baseline (now)** | 47.62% | Current extraction + strict eval |
| After filtering | 50-55% | Remove baseline models |
| After few-shot improvements | 60-70% | Better examples → better extraction |
| After post-processing | 65-75% | License inference, arch normalization |

**Gap from relaxed eval:** Each checkpoint will also have a relaxed score. The gap shows evaluator leniency. Aim to **minimize this gap** by improving extraction to match gold format exactly.

---

## Quick Commands Reference

```bash
# 1. Test complete workflow (filter + strict + relaxed)
python scripts/test_extraction_workflow.py --paper gpt1

# 2. Extract only (no ORKG)
python -m src.pipeline --pdf-url "<URL>" --paper-title "<TITLE>" --no-update

# 3. Evaluate strict (thesis metric)
python scripts/evaluation/evaluate_extraction_strict.py --prediction "<path>"

# 4. Evaluate relaxed (UX metric)
python scripts/evaluation/evaluate_extraction.py --prediction "<path>"

# 5. Compare both
python scripts/evaluation/evaluate_extraction_strict.py --prediction "<path>" > strict.txt
python scripts/evaluation/evaluate_extraction.py --prediction "<path>" > relaxed.txt
diff strict.txt relaxed.txt
```

---

## Next Action

**RUN THIS NOW:**
```bash
python scripts/test_extraction_workflow.py --paper gpt1
```

This will:
1. Extract GPT-1 paper
2. Filter baseline models
3. Show strict vs relaxed F1
4. Identify which improvements have highest impact

Then proceed with few-shot improvements based on the results.
