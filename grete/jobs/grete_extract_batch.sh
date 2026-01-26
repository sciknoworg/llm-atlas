#!/bin/bash
#SBATCH --job-name=llm-batch
#SBATCH --partition=kisski
#SBATCH -G A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --output=batch_%A_%a.log
#SBATCH --error=batch_%A_%a.err
#SBATCH -C inet

# List of ArXiv IDs to process
ARXIV_IDS=(
    "2302.13971"
    "2307.09288"
    "2401.02385"
    "2403.08295"
    "2404.14219"
)

# Get ArXiv ID for this array task
ARXIV_ID=${ARXIV_IDS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Batch Extraction Job on Grete"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing: $ARXIV_ID"
echo "Node: $SLURMD_NODENAME"
echo "=========================================="

# Load miniforge3 module
module load miniforge3

# Activate conda environment
conda activate ~/llm-extraction/llm-env

# Navigate to project
cd ~/llm-extraction

# Run extraction
python grete_extract_paper.py $ARXIV_ID

echo "=========================================="
echo "Completed: $ARXIV_ID"
echo "=========================================="









