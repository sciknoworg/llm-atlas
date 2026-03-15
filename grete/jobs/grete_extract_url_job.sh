#!/bin/bash
#SBATCH --job-name=llm-extract-url
#SBATCH --partition=kisski
#SBATCH -G A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=extract_url_%j.log
#SBATCH --error=extract_url_%j.err
#SBATCH -C inet

# Usage:
#   sbatch grete_extract_url_job.sh <pdf_url> [paper_title] [model_id]
#
# Examples:
#   sbatch grete/jobs/grete_extract_url_job.sh https://arxiv.org/pdf/2302.13971.pdf
#   sbatch grete/jobs/grete_extract_url_job.sh https://arxiv.org/pdf/2302.13971.pdf "LLaMA" meta-llama/Llama-3.2-1B-Instruct

PDF_URL="$1"
PAPER_TITLE="${2:-}"
MODEL_ID="${3:-meta-llama/Meta-Llama-3.1-8B-Instruct}"

echo "=========================================="
echo "LLM Extraction from URL Job on Grete"
echo "URL:   $PDF_URL"
echo "Title: $PAPER_TITLE"
echo "Model: $MODEL_ID"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

module load miniforge3

eval "$(conda shell.bash hook)"
conda activate /mnt/vast-kisski/home/alaa.kefi/u25486/llm-extraction/llm-env

cd /mnt/vast-kisski/home/alaa.kefi/u25486/llm-extraction

nvidia-smi

# Extract from URL with the specified model
python grete/extraction/grete_extract_from_url.py "$PDF_URL" "$PAPER_TITLE" --model "$MODEL_ID"

echo "=========================================="
echo "Job completed: $SLURM_JOB_ID"
echo "=========================================="






