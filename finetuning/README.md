# Fine-tuning Pipeline for ORKG Information Extraction

LoRA / QLoRA fine-tuning of **Llama-3.2-3B-Instruct** for structured knowledge extraction from LLM research papers, targeting the ORKG template R609825.

## Architecture

```
finetuning/
├── config.py              # Central configuration (paths, ORKG fields, hyperparameters)
├── prepare_dataset.py     # Gold standard + PDFs → instruction/input/output JSONL
├── train.py               # QLoRA fine-tuning with PEFT + SFTTrainer
├── inference.py           # Run extraction on papers using the fine-tuned adapter
├── evaluate.py            # Compare fine-tuned vs baseline (F1, per-field breakdown)
├── dataset/               # Generated JSONL splits (train/val/test)
├── output/                # Saved LoRA adapters and training configs
├── results/               # Extraction JSONs and evaluation reports
└── jobs/                  # SLURM scripts for KISSKI GPU cluster
    ├── setup_env.sh       # One-time conda environment setup
    ├── prepare_data.sh    # Dataset preparation job
    ├── train.sh           # Training job (A100 GPU)
    └── evaluate.sh        # Inference + evaluation job
```

## Quick Start

### 1. Prepare the dataset

```bash
python -m finetuning.prepare_dataset
```

This reads the gold standard (`data/gold_standard/R1364660.json`), parses each paper PDF, chunks the text, and produces train/val/test JSONL files with no data leakage (split by paper).

### 2. Train

```bash
python -m finetuning.train --run-name experiment_1
```

Default: Llama-3.2-3B-Instruct, QLoRA (4-bit), 3 epochs, lr=2e-4, LoRA r=16.

Override any hyperparameter via CLI:

```bash
python -m finetuning.train --epochs 2 --lr 1e-4 --lora-r 8 --run-name lr1e4_r8
```

### 3. Extract test papers

```bash
python -m finetuning.inference \
    --adapter finetuning/output/experiment_1/final_adapter \
    --all-test
```

### 4. Evaluate

```bash
python -m finetuning.evaluate \
    --results-dir finetuning/results/experiment_1/
```

For a side-by-side comparison with the prompt-only baseline:

```bash
python -m finetuning.evaluate \
    --results-dir finetuning/results/experiment_1/ \
    --baseline-dir data/extracted/meta_llama_3_2_3b_instruct/
```

## KISSKI Cluster Workflow

```bash
# 1. One-time setup (interactive session)
bash finetuning/jobs/setup_env.sh

# 2. Prepare data
sbatch finetuning/jobs/prepare_data.sh

# 3. Train
sbatch finetuning/jobs/train.sh

# 4. Evaluate
RUN_NAME=run_20260328 sbatch finetuning/jobs/evaluate.sh
```

## Training Instance Format

Each JSONL record follows the professor's specification:

| Part | Content |
|------|---------|
| `instruction` | Fixed ORKG extraction instruction (18 fields) |
| `input` | One text chunk from a paper |
| `output` | JSON object with ORKG fields (empty string for unsupported fields) |

Example:

```json
{
  "instruction": "Extract structured information about the language model ...",
  "input": "We introduce BERT with 110M, 340M parameters ...",
  "output": "{\"model_name\":\"BERT\",\"model_family\":\"BERT\",\"parameters\":\"110M, 340M\",...}"
}
```

## Hyperparameter Grid

| Parameter | Values to try | Default |
|-----------|--------------|---------|
| Epochs | 2, 3 | 3 |
| Learning rate | 1e-4, 2e-4 | 2e-4 |
| LoRA rank (r) | 8, 16 | 16 |
| LoRA alpha | 16, 32 | 32 |
| LoRA dropout | 0.05 | 0.05 |
| Batch size | 1–4 | 2 |
| Max seq length | 2048–4096 | 4096 |

## Evaluation Metrics

- **Overall F1**: Field-level precision/recall/F1 across all ORKG fields
- **Per-field F1**: Breakdown for each of the 18 ORKG fields
- **JSON validity rate**: Fraction of chunks producing parseable JSON
- **Comparison with baseline**: Side-by-side table against prompt-only extraction
