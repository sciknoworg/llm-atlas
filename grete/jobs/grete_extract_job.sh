#!/bin/bash
#SBATCH --job-name=llm-extract
#SBATCH --partition=kisski
#SBATCH -G A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=extract_%j.log
#SBATCH --error=extract_%j.err
#SBATCH -C inet

# Usage:
#   sbatch grete_extract_job.sh <arxiv_id> [model_id]
#
# Examples:
#   sbatch grete/jobs/grete_extract_job.sh 2302.13971
#   sbatch grete/jobs/grete_extract_job.sh 2302.13971 meta-llama/Llama-3.2-1B-Instruct

ARXIV_ID=${1:-"2302.13971"}
MODEL_ID=${2:-"meta-llama/Meta-Llama-3.1-8B-Instruct"}

echo "=========================================="
echo "LLM Extraction Job on Grete"
echo "ArXiv ID: $ARXIV_ID"
echo "Model:    $MODEL_ID"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
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

# Run extraction with the specified model
python grete/extraction/grete_extract_paper.py "$ARXIV_ID" --model "$MODEL_ID"

echo "=========================================="
echo "Job completed: $SLURM_JOB_ID"
echo "=========================================="

