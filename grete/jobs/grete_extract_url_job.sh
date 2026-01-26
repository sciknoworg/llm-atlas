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

echo "=========================================="
echo "LLM Extraction from URL Job on Grete"
echo "Model: Meta-Llama-3.1-8B-Instruct"
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

# Extract from URL: $1 = PDF URL, $2 = optional paper title
# Using Meta-Llama-3.1-8B-Instruct (requires HuggingFace login)
python grete_extract_from_url.py "$1" "$2"

echo "=========================================="
echo "Job completed: $SLURM_JOB_ID"
echo "=========================================="






