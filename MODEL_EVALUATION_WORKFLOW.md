# Model Evaluation Workflow - Complete Guide

## Overview

This workflow tests **10 KISSKI models** (selected by your professor) on **97 papers** from the gold standard, produces evaluation metrics, and builds a final results table.

**Total:** 10 models × 97 papers = 970 extraction runs

---

## The 10 Models (3 Categories)

From your professor's email:

### Vision Models (1)
- `qwen3-vl-30b-a3b-instruct`

### Think / Reasoning Models (3)
- `qwen3-30b-a3b-thinking-2507`
- `qwen3-235b-a22b`
- `deepseek-r1-distill-llama-70b`

### Instruction Tuned Models (6)
- `qwen3-30b-a3b-instruct-2507`
- `mistral-large-3-675b-instruct-2512`
- `meta-llama-3.1-8b-instruct`
- `llama-3.3-70b-instruct`
- `gemma-3-27b-it`
- `apertus-70b-instruct-2509`

---

## Quick Start: Test with 1 model first

**1. Set model in config** (`config/config.yaml`):
```yaml
kisski:
  model: meta-llama-3.1-8b-instruct
```

**2. Test on 3 papers:**
```bash
python scripts/batch_extract_all_papers.py --limit 3
```

**3. Aggregate evaluation:**
```bash
python scripts/aggregate_model_evaluation.py \
    --model-dir data/extracted/meta_llama_3_1_8b_instruct \
    --model-name "meta-llama-3.1-8b-instruct" \
    --output results/meta_llama_3_1_8b_instruct_results.json
```

**4. Check output:**
- Review `results/meta_llama_3_1_8b_instruct_results.json`
- Verify Overall F1 and BERTScore aggregate are present

If this works, proceed to full run!

---

## Full Workflow: All 10 Models

### For each model (repeat 10 times)

**1. Change model in `config/config.yaml`:**
```yaml
kisski:
  model: <model_name>  # e.g. qwen3-30b-a3b-thinking-2507
```

**2. Run extraction (all 97 papers):**
```bash
python scripts/batch_extract_all_papers.py --skip-existing
```
- Time: ~3-8 hours
- Output: `data/extracted/<model_slug>/`
- Logs: `data/logs/`

**3. Aggregate evaluation:**
```bash
python scripts/aggregate_model_evaluation.py \
    --model-dir data/extracted/<model_slug> \
    --model-name "<model_name>" \
    --output results/<model_slug>_results.json
```
- Time: ~10-30 minutes
- Output: One JSON with aggregated metrics for this model

**4. Repeat for next model**

---

## After All 10 Models: Build Results Table

```bash
python scripts/build_results_table.py \
    --results-dir results/ \
    --output results/final_results_table.csv
```

**Output:** `results/final_results_table.csv`

Columns:
- Category (Vision / Think-Reasoning / Instruction Tuned)
- Model name
- Papers Evaluated
- Overall F1, Precision, Recall, Accuracy
- BERTScore Aggregate
- (Optional) Per-field F1 for key fields

Also available in markdown:
```bash
python scripts/build_results_table.py \
    --format markdown \
    --output results/final_results_table.md
```

---

## Scripts Created

| Script | Purpose |
|--------|---------|
| `scripts/build_papers_list.py` | ✅ Build papers list from gold standard (ORKG + ArXiv) |
| `scripts/batch_extract_all_papers.py` | ✅ Extract all 97 papers for current model |
| `scripts/aggregate_model_evaluation.py` | ✅ Evaluate and aggregate for one model |
| `scripts/build_results_table.py` | ✅ Build final table from all models |

---

## Data Files

| File | Description |
|------|-------------|
| `data/gold_standard/papers_list.json` | 97 papers with arxiv_id / pdf_url |
| `data/extracted/<model>/` | Extraction outputs per model |
| `results/<model>_results.json` | Aggregated metrics per model |
| `results/final_results_table.csv` | Final table (all models) |

---

## Time Estimates

| Task | Time |
|------|------|
| Test (1 model, 3 papers) | ~15 minutes |
| Full (1 model, 97 papers) | ~3-8 hours extraction + 30 min evaluation |
| All 10 models (sequential) | ~30-80 hours |
| Build final table | ~1 minute |

**Total sequential:** ~30-80 hours  
**Parallelizable:** If you can run multiple models in parallel (separate terminals/machines), you can reduce to ~3-8 hours per batch.

---

## Testing Steps (Recommended Order)

### Today: Test Phase

1. ✅ Set model: `meta-llama-3.1-8b-instruct` in config
2. ✅ Run: `python scripts/batch_extract_all_papers.py --limit 3`
3. ✅ Check: `data/extracted/meta_llama_3_1_8b_instruct/` has 3 JSONs
4. ✅ Aggregate: `python scripts/aggregate_model_evaluation.py ...`
5. ✅ Check: `results/meta_llama_3_1_8b_instruct_results.json` has metrics

### This Week: First Full Model

1. ✅ Same model (or choose a fast one)
2. ✅ Run: `python scripts/batch_extract_all_papers.py` (all 97)
3. ✅ Aggregate and review results
4. ✅ Fix any issues before running the other 9 models

### Next 1-2 Weeks: All 10 Models

1. For each remaining model: change config → extract → aggregate
2. Build final table
3. Send to professor

---

## Error Handling

**Extraction fails:**
- Check logs in `data/logs/`
- Review `extraction_summary_*.json` in model directory
- Common: PDF parsing errors, API timeouts
- Solution: Use `--skip-existing` to resume

**Evaluation fails:**
- Check paper_title in extraction matches gold
- Verify JSON structure (needs `extraction_data` or `raw_extraction`)
- Check BERTScore installation: `pip install bert-score`

**Rate limiting:**
- KISSKI has limits: 1000/min, 10000/hour
- Your pipeline has 2s delay built in
- If you hit limits, add more delay or spread runs over multiple days

---

## What to Send Your Professor

After all 10 models are done:

1. **The CSV table** (`results/final_results_table.csv`)
2. **Short summary**, e.g.:
   - "I've completed the evaluation for all 10 models across the 97 papers in the gold standard. The results table is attached, with models grouped by the three categories (Vision, Think/Reasoning, Instruction Tuned). Overall F1 ranges from X to Y, with BERTScore aggregate from A to B."
3. **Optional:** Markdown version for nicer formatting

---

## Documentation

- **This guide:** `MODEL_EVALUATION_WORKFLOW.md`
- **Testing details:** `docs/TESTING_GUIDE_MODEL_EVALUATION.md`
- **Evaluation methodology:** `docs/EVALUATION_METHODOLOGY.md`
- **Next steps:** `docs/NEXT_STEPS_KISSKI_EVALUATION.md`

Ready to start testing!
