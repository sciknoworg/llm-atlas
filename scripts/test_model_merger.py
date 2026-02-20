#!/usr/bin/env python3
"""
Test Model Variant Merger

Tests the merge functionality with GPT-2 style examples and shows before/after.

Usage:
    python scripts/test_model_merger.py
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_variant_merger import merge_model_variants


def test_gpt2_variants():
    """Test merging GPT-2 size variants (as typically extracted)."""
    print("=" * 80)
    print("TEST 1: GPT-2 Size Variants")
    print("=" * 80)

    # Simulated extraction: 4 separate contributions for GPT-2 sizes
    extracted = [
        {
            "paper_title": "Language models are unsupervised multitask learners",
            "model_name": "GPT-2 124M",
            "model_family": "GPT",
            "organization": "OpenAI",
            "parameters": "124M",
            "parameters_millions": 124,
            "innovation": "GPT-2 demonstrates unsupervised multitask learning.",
            "pretraining_architecture": "Decoder",
            "date_created": "2019-02-01"
        },
        {
            "paper_title": "Language models are unsupervised multitask learners",
            "model_name": "GPT-2 355M",
            "model_family": "GPT",
            "organization": "OpenAI",
            "parameters": "355M",
            "parameters_millions": 355,
            "pretraining_architecture": "Decoder",
            "date_created": "2019-02-01"
        },
        {
            "paper_title": "Language models are unsupervised multitask learners",
            "model_name": "GPT-2 774M",
            "model_family": "GPT",
            "organization": "OpenAI",
            "parameters": "774M",
            "parameters_millions": 774,
            "pretraining_architecture": "Decoder",
            "date_created": "2019-02-01"
        },
        {
            "paper_title": "Language models are unsupervised multitask learners",
            "model_name": "GPT-2 1.5B",
            "model_family": "GPT",
            "organization": "OpenAI",
            "parameters": "1.5B",
            "parameters_millions": 1500,
            "innovation": "Demonstrates language models can perform many tasks without explicit supervision.",
            "extension": "Direct scale-up of GPT with 10X parameters.",
            "pretraining_architecture": "Decoder",
            "date_created": "2019-02-01"
        }
    ]

    print(f"\nBEFORE MERGE: {len(extracted)} contributions")
    for i, m in enumerate(extracted, 1):
        print(f"  {i}. {m['model_name']} - {m.get('parameters')}")

    # Merge
    merged = merge_model_variants(extracted)

    print(f"\nAFTER MERGE: {len(merged)} contribution(s)")
    for i, m in enumerate(merged, 1):
        print(f"\n  {i}. {m['model_name']}")
        print(f"     parameters: {m.get('parameters')}")
        print(f"     parameters_millions: {m.get('parameters_millions')}")
        print(f"     innovation: {m.get('innovation', '')[:80]}...")

    # Expected result
    expected_params = "124M, 355M, 774M, 1.5B"
    expected_max = 1500

    assert len(merged) == 1, f"Expected 1 merged model, got {len(merged)}"
    assert merged[0]["model_name"] == "GPT-2", f"Expected GPT-2, got {merged[0]['model_name']}"
    assert merged[0]["parameters"] == expected_params, \
        f"Expected params '{expected_params}', got '{merged[0]['parameters']}'"
    assert merged[0]["parameters_millions"] == expected_max, \
        f"Expected max {expected_max}, got {merged[0]['parameters_millions']}"

    print("\n[OK] TEST PASSED: GPT-2 variants merged correctly")
    return merged[0]


def test_bert_variants():
    """Test merging BERT Base/Large variants."""
    print("\n" + "=" * 80)
    print("TEST 2: BERT Base/Large Variants")
    print("=" * 80)

    extracted = [
        {
            "paper_title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "model_name": "BERT",
            "model_family": "BERT",
            "organization": "Google",
            "parameters": "110M",
            "parameters_millions": 110,
            "innovation": "BERT uses masked language modeling for bidirectional pretraining.",
            "pretraining_architecture": "Encoder",
            "date_created": "2018-10-01"
        },
        {
            "paper_title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "model_name": "BERT Base",
            "model_family": "BERT",
            "organization": "Google",
            "parameters": "110M",
            "parameters_millions": 110,
            "pretraining_architecture": "Encoder",
            "date_created": "2018-10-01"
        },
        {
            "paper_title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "model_name": "BERT Large",
            "model_family": "BERT",
            "organization": "Google",
            "parameters": "340M",
            "parameters_millions": 340,
            "pretraining_architecture": "Encoder",
            "date_created": "2018-10-01"
        }
    ]

    print(f"\nBEFORE MERGE: {len(extracted)} contributions")
    for i, m in enumerate(extracted, 1):
        print(f"  {i}. {m['model_name']} - {m.get('parameters')}")

    merged = merge_model_variants(extracted)

    print(f"\nAFTER MERGE: {len(merged)} contribution(s)")
    for i, m in enumerate(merged, 1):
        print(f"\n  {i}. {m['model_name']}")
        print(f"     parameters: {m.get('parameters')}")
        print(f"     parameters_millions: {m.get('parameters_millions')}")

    assert len(merged) == 1, f"Expected 1 merged model, got {len(merged)}"
    assert merged[0]["parameters_millions"] == 340, \
        f"Expected max 340, got {merged[0]['parameters_millions']}"

    print("\n[OK] TEST PASSED: BERT variants merged correctly")
    return merged[0]


def test_bert_concatenated_names():
    """Test BERTBASE (110M) / BERTLARGE (340M) / BERT → single BERT (as in real extraction)."""
    print("\n" + "=" * 80)
    print("TEST 2b: BERT concatenated names (BERTBASE, BERTLARGE)")
    print("=" * 80)

    extracted = [
        {"model_name": "BERTBASE (110M)", "parameters": "110M", "parameters_millions": 110},
        {"model_name": "BERTLARGE (340M)", "parameters": "340M", "parameters_millions": 340},
        {"model_name": "BERT", "parameters": None, "parameters_millions": None},
    ]
    print(f"\nBEFORE MERGE: {len(extracted)} contributions")
    for m in extracted:
        print(f"  - {m['model_name']}")

    merged = merge_model_variants(extracted)
    print(f"\nAFTER MERGE: {len(merged)} contribution(s)")
    for m in merged:
        print(f"  - {m['model_name']} params={m.get('parameters')} max={m.get('parameters_millions')}")

    assert len(merged) == 1, f"Expected 1 merged model, got {len(merged)}"
    assert merged[0]["model_name"] == "BERT", f"Expected model_name BERT, got {merged[0]['model_name']}"
    assert merged[0]["parameters_millions"] == 340, f"Expected max 340, got {merged[0]['parameters_millions']}"
    print("\n[OK] TEST PASSED: BERTBASE/BERTLARGE/BERT merged to one BERT")


def test_no_merge_needed():
    """Test that distinct models are not merged."""
    print("\n" + "=" * 80)
    print("TEST 3: Distinct Models (No Merge)")
    print("=" * 80)

    extracted = [
        {
            "paper_title": "Some paper",
            "model_name": "Model-A",
            "model_family": "FamilyA",
            "parameters": "100M",
            "parameters_millions": 100
        },
        {
            "paper_title": "Some paper",
            "model_name": "Model-B",
            "model_family": "FamilyB",
            "parameters": "200M",
            "parameters_millions": 200
        }
    ]

    print(f"\nBEFORE: {len(extracted)} contributions")
    for m in extracted:
        print(f"  - {m['model_name']}")

    merged = merge_model_variants(extracted)

    print(f"\nAFTER: {len(merged)} contribution(s)")
    for m in merged:
        print(f"  - {m['model_name']}")

    assert len(merged) == 2, f"Expected 2 models (no merge), got {len(merged)}"
    print("\n[OK] TEST PASSED: Distinct models not merged")


def test_llama_version_variants():
    """Test that Llama 3/3.1/3.2 stay separate, but size variants per version merge."""
    print("\n" + "=" * 80)
    print("TEST 4: Llama version variants (preserve versions, merge sizes)")
    print("=" * 80)

    extracted = [
        {"model_name": "Llama 3 8B", "model_family": "LLaMa", "parameters": "8B", "parameters_millions": 8000},
        {"model_name": "Llama 3.1 8B", "model_family": "LLaMa", "parameters": "8B", "parameters_millions": 8000},
        {"model_name": "Llama 3.1 70B", "model_family": "LLaMa", "parameters": "70B", "parameters_millions": 70000},
        {"model_name": "Llama 3.2 11B", "model_family": "LLaMa", "parameters": "11B", "parameters_millions": 11000},
    ]

    print(f"\nBEFORE: {len(extracted)} contributions")
    for m in extracted:
        print(f"  - {m['model_name']} ({m.get('parameters')})")

    merged = merge_model_variants(extracted)

    print(f"\nAFTER: {len(merged)} contribution(s)")
    for m in merged:
        print(f"  - {m['model_name']} ({m.get('parameters')})")

    names = [m["model_name"] for m in merged]
    assert any(n.startswith("Llama 3") and "3.1" not in n and "3.2" not in n for n in names), (
        "Expected Llama 3 to remain as its own version"
    )
    assert any(n == "Llama 3.1" for n in names), "Expected Llama 3.1 to remain as its own version"
    assert any(n.startswith("Llama 3.2") for n in names), "Expected Llama 3.2 to remain as its own version"
    assert len(merged) == 3, f"Expected 3 merged results (3, 3.1, 3.2), got {len(merged)}"

    llama31 = next(m for m in merged if m["model_name"] == "Llama 3.1")
    assert llama31["parameters_millions"] == 70000, (
        f"Expected Llama 3.1 max params 70000, got {llama31['parameters_millions']}"
    )
    assert llama31["parameters"] == "8B, 70B", (
        f"Expected Llama 3.1 parameters '8B, 70B', got {llama31['parameters']}"
    )

    print("\n[OK] TEST PASSED: Llama versions preserved, per-version sizes merged")


def test_xlnet_size_and_corpus_variants():
    """Test XLNet Base/Large and -wikibooks suffixes merge into one XLNet contribution."""
    print("\n" + "=" * 80)
    print("TEST 5: XLNet size + corpus suffix variants (merge to XLNet)")
    print("=" * 80)

    extracted = [
        {"model_name": "XLNet", "parameters": None, "parameters_millions": None},
        {"model_name": "XLNet-Base-wikibooks", "parameters": "110M", "parameters_millions": 110},
        {"model_name": "XLNet-Large-wikibooks", "parameters": "340M", "parameters_millions": 340},
        {"model_name": "XLNet-base", "parameters": "110M", "parameters_millions": 110},
        {"model_name": "XLNet-large", "parameters": "340M", "parameters_millions": 340},
    ]

    print(f"\nBEFORE: {len(extracted)} contributions")
    for m in extracted:
        print(f"  - {m['model_name']} ({m.get('parameters')})")

    merged = merge_model_variants(extracted)

    print(f"\nAFTER: {len(merged)} contribution(s)")
    for m in merged:
        print(f"  - {m['model_name']} ({m.get('parameters')})")

    assert len(merged) == 1, f"Expected 1 merged XLNet contribution, got {len(merged)}"
    assert merged[0]["model_name"] == "XLNet", (
        f"Expected merged name XLNet, got {merged[0]['model_name']}"
    )
    assert merged[0]["parameters"] == "110M, 340M", (
        f"Expected merged parameters '110M, 340M', got {merged[0]['parameters']}"
    )
    assert merged[0]["parameters_millions"] == 340, (
        f"Expected max params 340, got {merged[0]['parameters_millions']}"
    )

    print("\n[OK] TEST PASSED: XLNet corpus/size variants merged to one XLNet")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("MODEL VARIANT MERGER - TEST SUITE")
    print("=" * 80)

    gpt2_merged = test_gpt2_variants()
    bert_merged = test_bert_variants()
    test_bert_concatenated_names()
    test_no_merge_needed()
    test_llama_version_variants()
    test_xlnet_size_and_corpus_variants()

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED")
    print("=" * 80)

    # Show example merged output
    print("\nExample merged contribution (GPT-2):")
    print(json.dumps(gpt2_merged, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
