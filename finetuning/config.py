"""
Central configuration for the ORKG fine-tuning pipeline.

All hyperparameters, paths, and constants live here so that every other
module in the ``finetuning`` package imports a single source of truth.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

GOLD_STANDARD_PATH = PROJECT_ROOT / "data" / "gold_standard" / "R1364660.json"
PAPERS_LIST_PATH = PROJECT_ROOT / "data" / "gold_standard" / "papers_list.json"
PAPERS_DIR = PROJECT_ROOT / "data" / "papers"

DATASET_DIR = PROJECT_ROOT / "finetuning" / "dataset"
DATASET_TRAIN = DATASET_DIR / "train.jsonl"
DATASET_VAL = DATASET_DIR / "val.jsonl"
DATASET_TEST = DATASET_DIR / "test.jsonl"

OUTPUT_DIR = PROJECT_ROOT / "finetuning" / "output"
RESULTS_DIR = PROJECT_ROOT / "finetuning" / "results"

# ---------------------------------------------------------------------------
# ORKG extraction fields (matches gold-standard schema)
# ---------------------------------------------------------------------------
ORKG_FIELDS: List[str] = [
    "model_name",
    "model_family",
    "date_created",
    "organization",
    "pretraining_architecture",
    "pretraining_task",
    "finetuning_task",
    "optimizer",
    "parameters",
    "parameters_millions",
    "hardware_used",
    "blog_post",
    "license",
    "innovation",
    "extension",
    "pretraining_corpus",
    "application",
    "research_problem",
]

# ---------------------------------------------------------------------------
# Fixed instruction (from professor's guide)
# ---------------------------------------------------------------------------
INSTRUCTION = (
    "Extract structured information about the language model described in "
    "the text. Return exactly one valid JSON object using these ORKG fields: "
    "model_name, model_family, date_created, organization, "
    "pretraining_architecture, pretraining_task, finetuning_task, optimizer, "
    "parameters, parameters_millions, hardware_used, blog_post, license, "
    "innovation, extension, pretraining_corpus, application, "
    "research_problem. "
    "Use only information explicitly supported by the text. "
    "If a field is not mentioned, return an empty string. "
    "Do not guess or invent values."
)

# ---------------------------------------------------------------------------
# Chunking (mirrors src/pdf_parser.py defaults)
# ---------------------------------------------------------------------------
MAX_CHUNK_SIZE = 8000
CHUNK_OVERLAP = 500

# ---------------------------------------------------------------------------
# Model & LoRA hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """All knobs for one training run."""

    # Base model
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Training
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 4096

    # Precision & memory
    bf16: bool = True
    gradient_checkpointing: bool = True
    use_4bit: bool = True  # QLoRA

    # Logging & saving
    logging_steps: int = 10
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Dataset split ratios
    val_ratio: float = 0.10
    test_ratio: float = 0.15

    # Reproducibility
    seed: int = 42
