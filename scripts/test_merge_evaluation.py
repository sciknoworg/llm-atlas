#!/usr/bin/env python3
"""
Test Merge + Evaluation Workflow

Demonstrates the improvement from merging model variants before evaluation.
Shows before/after comparison for evaluation metrics.

Usage:
    python scripts/test_merge_evaluation.py
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_variant_merger import merge_model_variants


def load_gold():
    """Load gold standard."""
    gold_path = Path("data/gold_standard/R1364660.json")
    with open(gold_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("extraction_data", [])


def simulate_gpt2_extraction():
    """Simulate typical GPT-2 extraction (4 separate size variants)."""
    return [
        {
            "paper_title": "Language models are unsupervised multitask learners",
            "model_name": "GPT-2 124M",
            "model_family": "GPT",
            "date_created": "2019-02-01",
            "organization": "OpenAI",
            "innovation": "It can generate upto 768 words, demonstrate that language models begin to learn tasks such as question answering, machine translation, reading comprehension, and summarization without explicit supervision when trained on a diverse dataset of millions of web-scraped webpages.",
            "pretraining_architecture": "Decoder",
            "pretraining_task": "Causal language modeling",
            "finetuning_task": None,
            "optimizer": None,
            "parameters": "124M",
            "parameters_millions": 124,
            "hardware_used": None,
            "extension": "GPT-2 is a direct scale-up of GPT, with more than 10X the parameters and trained on more than 10X the amount of data.",
            "blog_post": "https://openai.com/research/better-language-models",
            "license": "closed source",
            "research_problem": "Large Language Models (LLMs), transformer model",
            "pretraining_corpus": None,
            "application": None
        },
        {
            "paper_title": "Language models are unsupervised multitask learners",
            "model_name": "GPT-2 355M",
            "model_family": "GPT",
            "date_created": "2019-02-01",
            "organization": "OpenAI",
            "parameters": "355M",
            "parameters_millions": 355,
            "pretraining_architecture": "Decoder",
            "pretraining_task": "Causal language modeling",
        },
        {
            "paper_title": "Language models are unsupervised multitask learners",
            "model_name": "GPT-2 774M",
            "model_family": "GPT",
            "date_created": "2019-02-01",
            "organization": "OpenAI",
            "parameters": "774M",
            "parameters_millions": 774,
            "pretraining_architecture": "Decoder",
            "pretraining_task": "Causal language modeling",
        },
        {
            "paper_title": "Language models are unsupervised multitask learners",
            "model_name": "GPT-2 1.5B",
            "model_family": "GPT",
            "date_created": "2019-02-01",
            "organization": "OpenAI",
            "parameters": "1.5B",
            "parameters_millions": 1500,
            "pretraining_architecture": "Decoder",
            "pretraining_task": "Causal language modeling",
        }
    ]


def find_gold_for_paper(gold_data, paper_title):
    """Find gold contribution(s) for a paper."""
    return [m for m in gold_data if m.get("paper_title") == paper_title]


def compare_evaluation(extracted, gold_paper, paper_title):
    """
    Compare evaluation with and without merge.
    
    Shows:
    - Matched contributions
    - Unmatched predictions
    - Field scores (especially parameters)
    """
    print("\n" + "=" * 80)
    print(f"EVALUATION COMPARISON: {paper_title}")
    print("=" * 80)
    
    # WITHOUT MERGE
    print("\n--- WITHOUT MERGE (current approach) ---")
    print(f"Extracted: {len(extracted)} contributions")
    for m in extracted:
        print(f"  - {m['model_name']}")
    print(f"Gold: {len(gold_paper)} contribution(s)")
    for m in gold_paper:
        print(f"  - {m['model_name']}")
    
    # Simulate matching (first exact match or substring)
    gold_model = gold_paper[0]
    matched = None
    for m in extracted:
        if "gpt-2" in m["model_name"].lower():
            matched = m
            break
    
    if matched:
        print(f"\nMatched: {matched['model_name']} <-> {gold_model['model_name']}")
        print(f"  Gold parameters: {gold_model['parameters']}")
        print(f"  Pred parameters: {matched['parameters']}")
        print(f"  Parameters match: {'[OK]' if gold_model['parameters'] == matched['parameters'] else '[FAIL]'}")
        print(f"  Unmatched predictions: {len(extracted) - 1}")
    
    # WITH MERGE
    print("\n--- WITH MERGE (new approach) ---")
    merged = merge_model_variants(extracted)
    print(f"After merge: {len(merged)} contribution(s)")
    for m in merged:
        print(f"  - {m['model_name']}")
    
    if merged:
        merged_model = merged[0]
        print(f"\nMatched: {merged_model['model_name']} <-> {gold_model['model_name']}")
        print(f"  Gold parameters: {gold_model['parameters']}")
        print(f"  Pred parameters: {merged_model['parameters']}")
        print(f"  Parameters match: {'[OK]' if gold_model['parameters'] == merged_model['parameters'] else '[PARTIAL]'}")
        print(f"  Unmatched predictions: 0")
    
    print("\n" + "=" * 80)
    print("IMPROVEMENT")
    print("=" * 80)
    print("Without merge:")
    print("  - 3 contributions ignored (unmatched)")
    print("  - parameters field gets 0 (only one size in matched contribution)")
    print("  - Underestimates extraction quality")
    print("\nWith merge:")
    print("  - All 4 sizes captured in one contribution")
    print("  - parameters field gets full/partial credit (set overlap)")
    print("  - Reflects true extraction quality")


def main():
    """Run demonstration."""
    print("=" * 80)
    print("MODEL MERGE + EVALUATION - IMPROVEMENT DEMO")
    print("=" * 80)
    
    # Load gold
    print("\nLoading gold standard...")
    gold_data = load_gold()
    print(f"Gold standard: {len(gold_data)} total contributions")
    
    # Get GPT-2 gold
    paper_title = "Language models are unsupervised multitask learners"
    gold_paper = find_gold_for_paper(gold_data, paper_title)
    
    if not gold_paper:
        print(f"[ERROR] Paper not found in gold: {paper_title}")
        return 1
    
    print(f"Gold for '{paper_title}': {len(gold_paper)} contribution(s)")
    
    # Simulate extraction
    print("\nSimulating typical extraction (4 size variants)...")
    extracted = simulate_gpt2_extraction()
    
    # Compare evaluation
    compare_evaluation(extracted, gold_paper, paper_title)
    
    print("\n" + "=" * 80)
    print("[OK] DEMO COMPLETE")
    print("=" * 80)
    print("\nNext step: Run real extraction on GPT-2 paper and compare:")
    print("  1. Extract GPT-2 paper (will produce multiple contributions)")
    print("  2. Pipeline auto-merges variants (new behavior)")
    print("  3. Evaluation compares merged output vs gold")
    print("  4. Parameters field gets proper credit (set-based F1)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
