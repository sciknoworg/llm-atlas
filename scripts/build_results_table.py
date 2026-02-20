"""
Build results table from aggregated model evaluations.

Combines evaluation results for all tested models into one table with
the three categories (Vision, Think/Reasoning, Instruction Tuned).

Usage:
    python scripts/build_results_table.py \\
        --results-dir results/ \\
        --output results/final_results_table.csv

Output:
    CSV table with rows = models (grouped by category), columns = metrics
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Setup project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Model categories from professor's email
MODEL_CATEGORIES = {
    "Vision": [
        "qwen3-vl-30b-a3b-instruct"
    ],
    "Think / Reasoning": [
        "qwen3-30b-a3b-thinking-2507",
        "qwen3-235b-a22b",
        "deepseek-r1-distill-llama-70b"
    ],
    "Instruction Tuned": [
        "qwen3-30b-a3b-instruct-2507",
        "mistral-large-3-675b-instruct-2512",
        "meta-llama-3.1-8b-instruct",
        "llama-3.3-70b-instruct",
        "gemma-3-27b-it",
        "apertus-70b-instruct-2509"
    ]
}


def find_category(model_name: str) -> str:
    """Find category for a model."""
    for category, models in MODEL_CATEGORIES.items():
        if model_name in models:
            return category
    return "Unknown"


def load_model_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all aggregated model results."""
    results = []
    
    for file in sorted(results_dir.glob("*_results.json")):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}", file=sys.stderr)
    
    return results


def build_table(model_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build results table with categories."""
    table = []
    
    for result in model_results:
        model_name = result.get("model_name", "unknown")
        category = find_category(model_name)
        
        row = {
            "Category": category,
            "Model": model_name,
            "Papers Evaluated": result["summary"]["successful_evaluations"],
            "Overall F1": result["overall_metrics"].get("f1_score", 0),
            "Overall Precision": result["overall_metrics"].get("precision", 0),
            "Overall Recall": result["overall_metrics"].get("recall", 0),
            "Overall Accuracy": result["overall_metrics"].get("accuracy", 0),
        }
        
        # Add BERTScore if available
        if result.get("bert_score_aggregate") is not None:
            row["BERTScore Aggregate"] = result["bert_score_aggregate"]
        
        # Add per-field F1 for key fields (optional)
        field_metrics = result.get("field_metrics", {})
        for field in ["model_name", "parameters", "innovation", "extension"]:
            if field in field_metrics:
                row[f"{field}_F1"] = field_metrics[field].get("f1_score", 0)
        
        table.append(row)
    
    # Sort by category order, then by model name
    category_order = {"Vision": 0, "Think / Reasoning": 1, "Instruction Tuned": 2, "Unknown": 99}
    table.sort(key=lambda x: (category_order.get(x["Category"], 99), x["Model"]))
    
    return table


def main():
    parser = argparse.ArgumentParser(description="Build results table from model evaluations")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/",
        help="Directory containing aggregated model results (*_results.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/final_results_table.csv",
        help="Output path for results table CSV"
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json", "markdown"],
        default="csv",
        help="Output format"
    )
    
    args = parser.parse_args()
    
    results_dir = PROJECT_ROOT / args.results_dir
    output_path = PROJECT_ROOT / args.output
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}", file=sys.stderr)
        return 1
    
    # Load all model results
    print(f"Loading model results from {results_dir}...")
    model_results = load_model_results(results_dir)
    
    if not model_results:
        print("Error: No model results found", file=sys.stderr)
        return 1
    
    print(f"Loaded {len(model_results)} model results")
    
    # Build table
    table = build_table(model_results)
    
    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.format == "csv":
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            if table:
                writer = csv.DictWriter(f, fieldnames=table[0].keys())
                writer.writeheader()
                writer.writerows(table)
        
    elif args.format == "json":
        output_path = output_path.with_suffix('.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(table, f, indent=2, ensure_ascii=False)
    
    elif args.format == "markdown":
        output_path = output_path.with_suffix('.md')
        with open(output_path, 'w', encoding='utf-8') as f:
            if table:
                # Write header
                keys = list(table[0].keys())
                f.write("| " + " | ".join(keys) + " |\n")
                f.write("| " + " | ".join(["---"] * len(keys)) + " |\n")
                
                # Write rows
                for row in table:
                    values = [str(row.get(k, "")) for k in keys]
                    f.write("| " + " | ".join(values) + " |\n")
    
    print(f"\nSaved results table to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS TABLE SUMMARY")
    print("=" * 70)
    
    categories_count = {}
    for row in table:
        cat = row["Category"]
        categories_count[cat] = categories_count.get(cat, 0) + 1
    
    for cat, count in sorted(categories_count.items()):
        print(f"{cat:20s} {count} models")
    
    print("=" * 70)
    
    # Show top 3 models by Overall F1
    sorted_by_f1 = sorted(table, key=lambda x: x.get("Overall F1", 0), reverse=True)[:3]
    print("\nTop 3 models by Overall F1:")
    for i, row in enumerate(sorted_by_f1, 1):
        print(f"{i}. {row['Model']:<40} F1: {row['Overall F1']:.4f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
