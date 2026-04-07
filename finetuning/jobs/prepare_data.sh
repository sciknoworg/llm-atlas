#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# SLURM job: prepare fine-tuning dataset
#
# Submit:  sbatch finetuning/jobs/prepare_data.sh
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=orkg-prepare
#SBATCH --partition=kisski
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH -C inet
#SBATCH --output=finetuning/logs/prepare_%j.log
#SBATCH --error=finetuning/logs/prepare_%j.err

set -euo pipefail
cd ~/llm-extraction
mkdir -p finetuning/logs

conda activate orkg-ft

echo "=== Preparing fine-tuning dataset ==="
python -m finetuning.prepare_dataset --min-fields 2

echo "=== Dataset statistics ==="
wc -l finetuning/dataset/*.jsonl

echo "=== Done ==="
