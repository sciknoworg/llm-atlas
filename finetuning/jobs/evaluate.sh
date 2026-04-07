#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# SLURM job: inference + evaluation on test papers
#
# Submit:  RUN_NAME=run_20260328 sbatch finetuning/jobs/evaluate.sh
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=orkg-eval
#SBATCH --partition=kisski
#SBATCH -G A100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH -C inet
#SBATCH --output=finetuning/logs/eval_%j.log
#SBATCH --error=finetuning/logs/eval_%j.err

set -euo pipefail
cd ~/llm-extraction
mkdir -p finetuning/logs

conda activate orkg-ft

RUN_NAME="${RUN_NAME:-default}"
ADAPTER="finetuning/output/$RUN_NAME/final_adapter"
RESULTS="finetuning/results/$RUN_NAME"

echo "=== Step 1: Inference on test papers ==="
python -m finetuning.inference \
    --adapter "$ADAPTER" \
    --all-test \
    --output-dir "$RESULTS"

echo ""
echo "=== Step 2: Evaluation ==="
python -m finetuning.evaluate \
    --results-dir "$RESULTS" \
    --output "$RESULTS/evaluation_report.json"

echo ""
echo "=== Done. Report at $RESULTS/evaluation_report.json ==="
