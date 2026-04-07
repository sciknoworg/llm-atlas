"""
LoRA / QLoRA fine-tuning of Llama-3.2-3B-Instruct for ORKG extraction.

Loads the JSONL dataset produced by ``prepare_dataset.py``, formats each
record as a Llama-3 chat conversation, and trains with SFTTrainer from the
``trl`` library using PEFT LoRA adapters.

Usage (local single-GPU or SLURM)
----------------------------------
    python -m finetuning.train
    python -m finetuning.train --epochs 2 --lr 1e-4 --lora-r 8

The final adapter is saved to ``finetuning/output/<run_name>/``.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from finetuning.config import (
    DATASET_TRAIN,
    DATASET_VAL,
    OUTPUT_DIR,
    TrainingConfig,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s"
)
logger = logging.getLogger(__name__)


# ── dataset helpers ──────────────────────────────────────────────────────────


def _load_jsonl(path: Path):
    records = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _format_chat(record: dict, tokenizer) -> str:
    """Convert an instruction/input/output record into a full chat string
    using the tokenizer's chat template (Llama-3 format)."""
    messages = [
        {"role": "system", "content": record["instruction"]},
        {"role": "user", "content": record["input"]},
        {"role": "assistant", "content": record["output"]},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def load_dataset_split(path: Path, tokenizer) -> Dataset:
    """Load a JSONL file and return a HuggingFace Dataset with a ``text``
    column containing the fully formatted chat string."""
    raw = _load_jsonl(path)
    texts = [_format_chat(r, tokenizer) for r in raw]
    return Dataset.from_dict({"text": texts})


# ── model loading ────────────────────────────────────────────────────────────


def load_model_and_tokenizer(cfg: TrainingConfig):
    """Load base model (optionally 4-bit quantized) and tokenizer."""

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if cfg.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    if cfg.use_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.gradient_checkpointing
        )

    return model, tokenizer


def apply_lora(model, cfg: TrainingConfig):
    """Wrap model with LoRA adapters."""
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ── training ─────────────────────────────────────────────────────────────────


def train(cfg: TrainingConfig, run_name: str):
    """Run one full training loop."""

    logger.info("Loading model: %s", cfg.model_name)
    model, tokenizer = load_model_and_tokenizer(cfg)
    model = apply_lora(model, cfg)

    logger.info("Loading datasets …")
    train_ds = load_dataset_split(DATASET_TRAIN, tokenizer)
    val_ds = load_dataset_split(DATASET_VAL, tokenizer)
    logger.info("  train: %d examples, val: %d examples", len(train_ds), len(val_ds))

    output_path = OUTPUT_DIR / run_name
    output_path.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_path),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=cfg.logging_steps,
        eval_strategy=cfg.eval_strategy,
        save_strategy=cfg.save_strategy,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        save_total_limit=2,
        seed=cfg.seed,
        report_to="none",
        max_seq_length=cfg.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    logger.info("Starting training …")
    trainer.train()

    # Save adapter + tokenizer
    final_path = output_path / "final_adapter"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    # Save training config for reproducibility
    config_path = output_path / "training_config.json"
    import dataclasses
    with open(config_path, "w", encoding="utf-8") as fp:
        json.dump(dataclasses.asdict(cfg), fp, indent=2)

    logger.info("Training complete. Adapter saved to %s", final_path)
    return final_path


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="LoRA fine-tuning for ORKG extraction")
    ap.add_argument("--model", type=str, default=None, help="HuggingFace model ID")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--lora-r", type=int, default=None)
    ap.add_argument("--lora-alpha", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-seq-length", type=int, default=None)
    ap.add_argument("--no-4bit", action="store_true", help="Disable QLoRA (use fp16 LoRA)")
    ap.add_argument("--run-name", type=str, default="default",
                     help="Subdirectory name under finetuning/output/")
    args = ap.parse_args()

    cfg = TrainingConfig()
    if args.model:
        cfg.model_name = args.model
    if args.epochs:
        cfg.num_train_epochs = args.epochs
    if args.lr:
        cfg.learning_rate = args.lr
    if args.lora_r:
        cfg.lora_r = args.lora_r
    if args.lora_alpha:
        cfg.lora_alpha = args.lora_alpha
    if args.batch_size:
        cfg.per_device_train_batch_size = args.batch_size
    if args.max_seq_length:
        cfg.max_seq_length = args.max_seq_length
    if args.no_4bit:
        cfg.use_4bit = False

    train(cfg, run_name=args.run_name)


if __name__ == "__main__":
    main()
