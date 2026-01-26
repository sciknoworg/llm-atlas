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

echo "=========================================="
echo "LLM Extraction Job on Grete"
echo "Model: Meta-Llama-3.1-8B-Instruct"
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

# Run extraction (using Meta-Llama-3.1-8B-Instruct)
python grete_extract_paper.py $1

echo "=========================================="
echo "Job completed: $SLURM_JOB_ID"
echo "=========================================="

