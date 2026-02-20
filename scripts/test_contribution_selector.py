#!/usr/bin/env python3
"""
Test contribution selector behavior.

Usage:
    python scripts/test_contribution_selector.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_contribution_selector import select_primary_model_contributions


def test_mixed_primary_and_auxiliary():
    """Primary model releases should be kept, auxiliary artifacts filtered."""
    extracted = [
        {"model_name": "Llama 3", "model_family": "Llama", "parameters": "8B, 70B"},
        {"model_name": "Llama 3.1 405B", "model_family": "Llama", "parameters": "405B"},
        {"model_name": "Llama Guard 3", "model_family": "Llama", "parameters": "38B"},
        {"model_name": "Prompt Guard", "model_family": "Prompt Guard", "parameters": "86M"},
        {"model_name": "Speech Adapter", "model_family": "Adapter", "parameters": "100M"},
        {"model_name": "ViT-H/14", "model_family": "ViT", "parameters": "630M"},
        {"model_name": "Llama 3 128K-context", "model_family": "Llama", "parameters": None},
    ]

    selected = select_primary_model_contributions(extracted)
    names = {m["model_name"] for m in selected}

    assert "Llama 3" in names
    assert "Llama 3.1 405B" in names
    assert "Llama Guard 3" not in names
    assert "Prompt Guard" not in names
    assert "Speech Adapter" not in names
    assert "ViT-H/14" not in names
    assert "Llama 3 128K-context" not in names
    assert len(selected) == 2, f"Expected 2 selected models, got {len(selected)}"


def test_conservative_fallback_when_all_look_auxiliary():
    """If selector is unsure (everything looks auxiliary), keep original list."""
    extracted = [
        {"model_name": "Speech Encoder", "model_family": "Speech"},
        {"model_name": "Speech Adapter", "model_family": "Speech"},
    ]

    selected = select_primary_model_contributions(extracted)
    assert len(selected) == len(extracted), "Expected conservative fallback to keep all models"


def main():
    print("=" * 80)
    print("CONTRIBUTION SELECTOR - TEST SUITE")
    print("=" * 80)

    test_mixed_primary_and_auxiliary()
    print("[OK] test_mixed_primary_and_auxiliary")

    test_conservative_fallback_when_all_look_auxiliary()
    print("[OK] test_conservative_fallback_when_all_look_auxiliary")

    print("=" * 80)
    print("ALL TESTS PASSED")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())

