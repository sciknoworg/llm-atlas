# Implementation Summary: Model Variant Merging & Semantic Evaluation

## What was implemented

### 1. **Model Variant Merger** (`src/model_variant_merger.py`)

**Purpose:** Merge size variants (e.g., GPT-2 124M, 355M, 774M, 1.5B) into one contribution following gold-standard convention.

**Features:**
- Canonical name extraction (strips "Base", "Large", "124M", "1.5B", etc.)
- Parameter aggregation: comma-separated list, sorted by size
- Maximum parameters_millions computation
- Intelligent field merging (prefer non-null, longest text for narratives)

**Tests:** `scripts/test_model_merger.py`
- GPT-2: 4 variants → 1 contribution ✓
- BERT: 3 variants → 1 contribution ✓
- Distinct models: no unwanted merging ✓

---

### 2. **Pipeline Integration** (`src/pipeline.py`)

**Added:**
- Import `merge_model_variants`
- Step 3.5 (after extraction, before ORKG mapping): merge size variants
- Logging: `Models after merge: N -> M`
- Works for both ArXiv and PDF URL flows

**Result:** All extractions now automatically produce gold-aligned contributions (one per model family with aggregated sizes).

---

### 3. **Semantic Similarity in Evaluation** (`scripts/evaluation/evaluate_extraction_strict.py`)

**Added:**
- Sentence-transformers integration (lazy-loaded `all-MiniLM-L6-v2`)
- `semantic_match()`: computes cosine similarity between embeddings
- Applied to: `innovation`, `pretraining_corpus`, `application`, `research_problem`, `extension`
- CLI flag: `--no-semantic` to disable (falls back to fuzzy matching)

**Result:** Long-text fields judged by meaning (embedding similarity), not just character overlap.

---

### 4. **Set-based Parameters Comparison**

**Added:**
- `compare_parameters_list()`: treats `parameters` as comma-separated set
- Computes precision, recall, F1 over parameter sizes
- Match if F1 ≥ threshold (default 0.8)

**Result:** `parameters: "124M, 355M, 774M, 1.5B"` vs `"124M, 355M, 774M, 1.5B"` → full credit (not just string match).

---

### 5. **Auto-evaluation in Pipeline**

**Added:**
- `--no-evaluate`: skip evaluation after extraction (default: run evaluation)
- `--gold <path>`: gold standard for evaluation (default: R1364660.json)
- `_run_evaluation()`: subprocess strict evaluator after successful extraction

**Result:** One command extracts, merges, and evaluates:
```bash
python -m src.pipeline --pdf-url <url> --paper-title <title> --no-update
```

---

## How GPT-2 evaluation improves

### Before

- **Extracted:** 4 contributions (GPT-2 124M, 355M, 774M, 1.5B)
- **Gold:** 1 contribution (GPT-2 with all sizes)
- **Matched:** 1 of 4 (e.g., GPT-2 124M)
- **Unmatched:** 3 predictions ignored
- **Parameters:** gold `"124M, 355M, 774M, 1.5B"` vs pred `"124M"` → **FAIL**
- **Result:** Underestimates extraction quality

### After

- **Extracted:** 4 contributions
- **Merged:** 1 contribution (GPT-2 with `"124M, 355M, 774M, 1.5B"`, max 1500)
- **Gold:** 1 contribution
- **Matched:** 1 of 1
- **Unmatched:** 0
- **Parameters:** gold vs pred both `"124M, 355M, 774M, 1.5B"` → **MATCH** (F1 = 1.0)
- **Result:** Reflects true extraction quality

---

## Files added/modified

| File | Change |
|------|--------|
| `src/model_variant_merger.py` | **New**: Merge logic, canonical name extraction, parameter aggregation |
| `src/pipeline.py` | **Modified**: Import merger, add Step 3.5 (merge after extraction), add auto-evaluation |
| `scripts/evaluation/evaluate_extraction_strict.py` | **Modified**: Semantic matching, set-based parameters comparison |
| `requirements.txt` | **Modified**: Added `sentence-transformers>=2.2.0` |
| `scripts/test_model_merger.py` | **New**: Unit tests for merge logic |
| `scripts/test_merge_evaluation.py` | **New**: Demo showing evaluation improvement |
| `docs/MODEL_VARIANT_MERGING.md` | **New**: Documentation |

---

## Testing

Run all tests:
```bash
# Unit tests for merger
python scripts/test_model_merger.py

# Evaluation improvement demo
python scripts/test_merge_evaluation.py

# Full pipeline test (if you have KISSKI key)
python -m src.pipeline --pdf-url "..." --paper-title "..." --no-update
```

---

## Next steps

1. **Run extraction on multiple papers** (GPT-1, BERT, GPT-2, etc.) and compare evaluation scores before/after
2. **Verify ORKG upload** creates the right number of contributions (one per merged model)
3. **Finalize gold standard** (review R1364660.json for consistency)
4. **Document evaluation categories** (aggregate vs single-property) for thesis
