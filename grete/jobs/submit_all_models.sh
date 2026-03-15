#!/bin/bash
# Submit extraction jobs for all 8 small LMs over all 71 papers.
#
# Usage (run from ~/llm-extraction on Grete):
#   bash grete/jobs/submit_all_models.sh
#
# What it does:
#   - Submits 8 SLURM array jobs (one per model)
#   - Each array job has 71 tasks (one per paper)
#   - Total: 8 × 71 = 568 GPU jobs
#   - Outputs: data/extracted/<arxiv_id>_<timestamp>.json (each JSON has "model_used" field)
#   - Logs:    all_papers_<JOBID>_<TASK>.log / .err

MODELS=(
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct"
    "mistralai/Ministral-3-3B-Instruct-2512"
    "mistralai/Ministral-3-8B-Instruct-2512"
    "mistralai/Ministral-3-3B-Reasoning-2512"
    "mistralai/Ministral-3-8B-Reasoning-2512"
    "Qwen/Qwen3-4B-Instruct-2507"
    "Qwen/Qwen3-4B-Thinking-2507"
)

echo "Submitting extraction jobs for ${#MODELS[@]} models × 71 papers..."
echo ""

for MODEL in "${MODELS[@]}"; do
    JOB_ID=$(sbatch --array=0-70 grete/jobs/grete_extract_all_papers.sh "$MODEL" | awk '{print $4}')
    echo "Submitted job $JOB_ID  ->  $MODEL"
done

echo ""
echo "All jobs submitted."
echo "Monitor with:  squeue -u \$USER"
echo "Logs:          all_papers_<JOBID>_<TASK>.log / .err"
echo "Results:       data/extracted/*.json"
