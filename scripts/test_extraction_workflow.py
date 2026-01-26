#!/usr/bin/env python3
"""
Complete Extraction + Evaluation Workflow Test

Runs the full pipeline: extract → filter → evaluate (strict + relaxed)

Usage:
    python scripts/test_extraction_workflow.py --paper gpt1
    python scripts/test_extraction_workflow.py --pdf-url <url> --paper-title <title>
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baseline_filter import filter_baseline_models


KNOWN_PAPERS = {
    "gpt1": {
        "pdf_url": "https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf",
        "title": "Improving Language Understanding by Generative Pre-Training",
        "gold_models": 1
    }
}


def run_command(cmd, description):
    """Run a command and return the result."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[ERROR] Command failed with exit code {result.returncode}")
        print(result.stderr)
        return None
    
    print(result.stdout)
    return result


def extract_paper(pdf_url, paper_title):
    """Extract from PDF URL."""
    cmd = [
        sys.executable, "-m", "src.pipeline",
        "--pdf-url", pdf_url,
        "--paper-title", paper_title,
        "--no-update"
    ]
    
    result = run_command(cmd, "STEP 1: Extract from PDF")
    if not result:
        return None
    
    # Extract saved path from output
    for line in result.stdout.split('\n'):
        if '[OK] Saved to:' in line:
            path = line.split('[OK] Saved to:')[1].strip()
            return path
    
    return None


def filter_extraction(json_path, filter_baselines=True):
    """Filter baseline models from extraction."""
    print(f"\n{'='*80}")
    print("STEP 2: Filter Baseline Models")
    print(f"{'='*80}")
    
    if not filter_baselines:
        print("Skipping baseline filtering (disabled)")
        return json_path
    
    # Load extraction
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    models = data.get("extraction_data", [])
    paper_metadata = data.get("paper_metadata", {})
    
    print(f"Before filtering: {len(models)} models")
    for m in models:
        print(f"  - {m.get('model_name')}")
    
    # Filter
    filtered_models = filter_baseline_models(models, paper_metadata, keep_top_n=1)
    
    print(f"\nAfter filtering: {len(filtered_models)} models")
    for m in filtered_models:
        print(f"  - {m.get('model_name')}")
    
    # Save filtered version
    filtered_path = json_path.replace('.json', '_filtered.json')
    data["extraction_data"] = filtered_models
    data["models_extracted"] = len(filtered_models)
    
    with open(filtered_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved filtered extraction to: {filtered_path}")
    return filtered_path


def evaluate(json_path, gold_path, strict=False):
    """Evaluate extraction."""
    mode = "STRICT" if strict else "RELAXED"
    script = "evaluate_extraction_strict.py" if strict else "evaluate_extraction.py"
    
    cmd = [
        sys.executable, f"scripts/evaluation/{script}",
        "--prediction", json_path,
        "--gold", gold_path
    ]
    
    result = run_command(cmd, f"STEP 3: Evaluate ({mode})")
    
    # Extract F1 score
    if result:
        for line in result.stdout.split('\n'):
            if 'F1-Score:' in line and '%' in line:
                try:
                    f1_str = line.split('F1-Score:')[1].split('%')[0].strip()
                    f1 = float(f1_str)
                    return f1
                except:
                    pass
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Test extraction workflow")
    parser.add_argument("--paper", choices=list(KNOWN_PAPERS.keys()), help="Use known paper config")
    parser.add_argument("--pdf-url", help="PDF URL (if not using --paper)")
    parser.add_argument("--paper-title", help="Paper title (if not using --paper)")
    parser.add_argument("--gold", default="data/gold_standard/R1364660.json", help="Gold standard path")
    parser.add_argument("--no-filter", action="store_true", help="Skip baseline filtering")
    parser.add_argument("--skip-strict", action="store_true", help="Skip strict evaluation")
    
    args = parser.parse_args()
    
    # Get paper config
    if args.paper:
        config = KNOWN_PAPERS[args.paper]
        pdf_url = config["pdf_url"]
        paper_title = config["title"]
    elif args.pdf_url and args.paper_title:
        pdf_url = args.pdf_url
        paper_title = args.paper_title
    else:
        print("Error: Must specify --paper OR (--pdf-url + --paper-title)")
        return 1
    
    print(f"\n{'='*80}")
    print("EXTRACTION WORKFLOW TEST")
    print(f"{'='*80}")
    print(f"Paper: {paper_title}")
    print(f"PDF URL: {pdf_url}")
    print(f"Gold standard: {args.gold}")
    print(f"Filter baselines: {not args.no_filter}")
    print(f"Strict evaluation: {not args.skip_strict}")
    
    # Step 1: Extract
    json_path = extract_paper(pdf_url, paper_title)
    if not json_path:
        print("\n[FAIL] Extraction failed")
        return 1
    
    # Step 2: Filter (optional)
    filtered_path = filter_extraction(json_path, filter_baselines=not args.no_filter)
    
    # Step 3: Evaluate
    results = {}
    
    # Evaluate original (unfiltered)
    print(f"\n{'='*80}")
    print("RESULTS: UNFILTERED")
    print(f"{'='*80}")
    
    if not args.skip_strict:
        f1_strict = evaluate(json_path, args.gold, strict=True)
        if f1_strict is not None:
            results["unfiltered_strict"] = f1_strict
            print(f"Unfiltered (Strict): F1 = {f1_strict:.2f}%")
    
    f1_relaxed = evaluate(json_path, args.gold, strict=False)
    if f1_relaxed is not None:
        results["unfiltered_relaxed"] = f1_relaxed
        print(f"Unfiltered (Relaxed): F1 = {f1_relaxed:.2f}%")
    
    # Evaluate filtered
    if not args.no_filter and filtered_path != json_path:
        print(f"\n{'='*80}")
        print("RESULTS: FILTERED")
        print(f"{'='*80}")
        
        if not args.skip_strict:
            f1_strict = evaluate(filtered_path, args.gold, strict=True)
            if f1_strict is not None:
                results["filtered_strict"] = f1_strict
                print(f"Filtered (Strict): F1 = {f1_strict:.2f}%")
        
        f1_relaxed = evaluate(filtered_path, args.gold, strict=False)
        if f1_relaxed is not None:
            results["filtered_relaxed"] = f1_relaxed
            print(f"Filtered (Relaxed): F1 = {f1_relaxed:.2f}%")
    
    # Summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Extraction saved: {json_path}")
    if filtered_path != json_path:
        print(f"Filtered saved: {filtered_path}")
    print(f"\nResults:")
    for key, value in results.items():
        print(f"  {key}: F1 = {value:.2f}%")
    
    # Recommendation
    if "filtered_strict" in results:
        f1 = results["filtered_strict"]
        print(f"\n{'='*80}")
        print(f"THESIS METRIC (Filtered + Strict): F1 = {f1:.2f}%")
        print(f"{'='*80}")
        if f1 >= 60:
            print("Status: GOOD - Ready for thesis evaluation")
        elif f1 >= 40:
            print("Status: FAIR - Consider further improvements")
        else:
            print("Status: NEEDS IMPROVEMENT - Review extraction issues")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
