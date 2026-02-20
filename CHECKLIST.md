# Model Evaluation Checklist

Track your progress testing the 10 KISSKI models.

## ✅ Preparation (Done)

- [x] Got list of 10 models from professor
- [x] Built papers list (97 papers with URLs)
- [x] Created batch extraction script
- [x] Created evaluation aggregation script
- [x] Created results table builder

---

## 📝 Testing Phase (Do First)

- [ ] Set model: `meta-llama-3.1-8b-instruct` in `config/config.yaml`
- [ ] Dry-run: `python scripts/batch_extract_all_papers.py --dry-run --limit 3`
- [ ] Extract 3 papers: `python scripts/batch_extract_all_papers.py --limit 3`
- [ ] Check outputs in `data/extracted/meta_llama_3_1_8b_instruct/`
- [ ] Aggregate: `python scripts/aggregate_model_evaluation.py --model-dir data/extracted/meta_llama_3_1_8b_instruct --model-name "meta-llama-3.1-8b-instruct" --output results/meta_llama_3_1_8b_instruct_results.json`
- [ ] Review `results/meta_llama_3_1_8b_instruct_results.json`

---

## 🚀 Full Run: Model 1 (Complete First)

- [ ] Extract all 97 papers: `python scripts/batch_extract_all_papers.py`
- [ ] Aggregate evaluation (same command as test)
- [ ] Review results and fix any issues

---

## 🔄 Models 2-10 (Repeat for Each)

### Vision (1 model)
- [ ] `qwen3-vl-30b-a3b-instruct`

### Think / Reasoning (3 models)
- [ ] `qwen3-30b-a3b-thinking-2507`
- [ ] `qwen3-235b-a22b`
- [ ] `deepseek-r1-distill-llama-70b`

### Instruction Tuned (6 models)
- [ ] `qwen3-30b-a3b-instruct-2507`
- [ ] `mistral-large-3-675b-instruct-2512`
- [ ] `meta-llama-3.1-8b-instruct` (if not done in testing)
- [ ] `llama-3.3-70b-instruct`
- [ ] `gemma-3-27b-it`
- [ ] `apertus-70b-instruct-2509`

**For each:**
1. Change model in `config/config.yaml`
2. Run `python scripts/batch_extract_all_papers.py --skip-existing`
3. Run aggregation script
4. Check results JSON

---

## 📊 Final Steps

- [ ] Build results table: `python scripts/build_results_table.py --output results/final_results_table.csv`
- [ ] Also markdown: `python scripts/build_results_table.py --format markdown --output results/final_results_table.md`
- [ ] Review table (categories, metrics, completeness)
- [ ] Send to professor

---

## 📌 Notes

**Estimated total time:** 30-80 hours (sequential), or faster if parallelized

**Files to send professor:**
- `results/final_results_table.csv` (or .md)
- Optional: summary of findings

**Backup strategy:**
- Use `--skip-existing` to resume from failures
- Check `extraction_summary_*.json` for progress
- Run overnight or in background (screen/tmux)
