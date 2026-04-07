#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# SLURM job: LoRA fine-tuning on KISSKI GPU
#
# Submit:  sbatch finetuning/jobs/train.sh
#
# Default config: Llama-3.2-3B-Instruct, QLoRA, 3 epochs, lr=2e-4, r=16
# Override via environment variables:
#   EPOCHS=2 LR=1e-4 LORA_R=8 sbatch finetuning/jobs/train.sh
# ──────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=orkg-train
#SBATCH --partition=kisski
#SBATCH -G A100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH -C inet
#SBATCH --output=finetuning/logs/train_%j.log
#SBATCH --error=finetuning/logs/train_%j.err

set -euo pipefail
cd ~/llm-extraction
mkdir -p finetuning/logs

conda activate orkg-ft

# Allow overrides through environment
RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
BATCH="${BATCH:-2}"

echo "=== Training configuration ==="
echo "  Run name:   $RUN_NAME"
echo "  Epochs:     $EPOCHS"
echo "  LR:         $LR"
echo "  LoRA r:     $LORA_R"
echo "  LoRA alpha: $LORA_ALPHA"
echo "  Batch size: $BATCH"
echo "  GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

python -m finetuning.train \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --lora-r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA" \
    --batch-size "$BATCH" \
    --run-name "$RUN_NAME"

echo "=== Training complete ==="
echo "Adapter saved to: finetuning/output/$RUN_NAME/final_adapter"
