#!/bin/bash
#SBATCH --job-name=llm-batch
#SBATCH --partition=kisski
#SBATCH -G A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=batch_%A_%a.log
#SBATCH --error=batch_%A_%a.err
#SBATCH -C inet

# Small LM benchmark: run the same paper through all 8 models (one job per model).
# Usage:
#   sbatch grete/jobs/grete_extract_batch.sh [arxiv_id]
#
# Example:
#   sbatch grete/jobs/grete_extract_batch.sh 2302.13971
#
# To process a list of papers with one specific model, set ARXIV_IDS below
# and adjust --array accordingly.

# Paper to process (override via first positional arg)
ARXIV_ID=${1:-"2302.13971"}

# 8 small LMs to benchmark (array index maps to model)
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

MODEL_ID=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Batch Extraction Job on Grete"
echo "=========================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Array Task:   $SLURM_ARRAY_TASK_ID"
echo "ArXiv ID:     $ARXIV_ID"
echo "Model:        $MODEL_ID"
echo "Node:         $SLURMD_NODENAME"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Load miniforge3 module
module load miniforge3

# Initialize and activate conda environment
eval "$(conda shell.bash hook)"
conda activate /mnt/vast-kisski/home/alaa.kefi/u25486/llm-extraction/llm-env

# Navigate to project
cd ~/llm-extraction

# Check GPU
nvidia-smi

# Run extraction with the model for this array task
python grete/extraction/grete_extract_paper.py "$ARXIV_ID" --model "$MODEL_ID"

echo "=========================================="
echo "Completed: $ARXIV_ID with $MODEL_ID"
echo "=========================================="









