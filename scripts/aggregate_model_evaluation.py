"""
Aggregate evaluation results for one KISSKI model across all papers.

Takes all extraction outputs for a model, evaluates each against gold standard,
and produces one aggregated score (Overall F1, BERTScore aggregate, etc.).

Usage:
    python scripts/aggregate_model_evaluation.py \\
        --model-dir data/extracted/meta_llama_3_1_8b_instruct \\
        --model-name "meta-llama-3.1-8b-instruct" \\
        --output results/meta_llama_3_1_8b_instruct_results.json

Output:
    JSON with aggregated metrics for this model over all papers
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

# Setup project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.evaluate_extraction_strict import StrictExtractionEvaluator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def aggregate_evaluations(
    extraction_dir: Path,
    gold_path: Path,
    evaluator: StrictExtractionEvaluator
) -> Dict[str, Any]:
    """
    Evaluate all extractions in directory against gold standard and aggregate.
    
    Returns aggregated metrics across all papers.
    """
    # Find all extraction JSONs
    extraction_files = sorted(extraction_dir.glob("*.json"))
    
    # Filter out summary files
    extraction_files = [f for f in extraction_files if "summary" not in f.name.lower()]
    
    logger.info(f"Found {len(extraction_files)} extraction files in {extraction_dir}")
    
    if not extraction_files:
        logger.error("No extraction files found")
        return None
    
    # Load gold standard
    gold_data_raw = load_json(gold_path)
    gold_data = gold_data_raw.get("extraction_data", gold_data_raw)
    
    # Aggregate results across all papers
    all_field_results = {field: [] for field in evaluator.EVALUATION_FIELDS}
    paper_results = []
    
    for i, extraction_file in enumerate(extraction_files, 1):
        logger.info(f"[{i}/{len(extraction_files)}] Evaluating: {extraction_file.name}")
        
        try:
            # Load extraction
            pred_data_raw = load_json(extraction_file)
            pred_data = (
                pred_data_raw.get("extraction_data") or
                pred_data_raw.get("raw_extraction") or
                pred_data_raw
            )
            
            if not isinstance(pred_data, list):
                logger.warning(f"  Skipping (not a list): {extraction_file.name}")
                continue
            
            # Get paper title for filtering gold
            paper_title = None
            if "paper_metadata" in pred_data_raw:
                paper_title = pred_data_raw["paper_metadata"].get("title")
            elif pred_data and isinstance(pred_data, list) and len(pred_data) > 0:
                paper_title = pred_data[0].get("paper_title")
            
            # Evaluate this paper's extraction
            paper_eval = evaluator.evaluate_dataset(
                gold_data,
                pred_data,
                paper_title=paper_title
            )
            
            # Collect field results
            for field in evaluator.EVALUATION_FIELDS:
                if field in paper_eval["field_metrics"]:
                    # Store per-paper field results for aggregation
                    field_results = paper_eval.get("model_results", [])
                    for model_result in field_results:
                        if "fields" in model_result and field in model_result["fields"]:
                            all_field_results[field].append(model_result["fields"][field])
            
            paper_results.append({
                "paper_title": paper_title,
                "file": extraction_file.name,
                "matched_models": paper_eval["summary"]["matched_models"],
                "overall_f1": paper_eval["overall_metrics"]["f1_score"],
                "bert_score_aggregate": paper_eval.get("bert_score_aggregate")
            })
            
        except Exception as e:
            logger.error(f"  Error evaluating {extraction_file.name}: {e}")
            paper_results.append({
                "file": extraction_file.name,
                "error": str(e)
            })
    
    # Calculate aggregated metrics
    logger.info("Computing aggregated metrics across all papers...")
    
    # Aggregate field metrics
    field_metrics = {}
    for field in evaluator.EVALUATION_FIELDS:
        if all_field_results[field]:
            field_metrics[field] = evaluator.calculate_metrics(all_field_results[field])
    
    # Overall metrics (all fields, all papers)
    all_results_flat = []
    for field_results in all_field_results.values():
        all_results_flat.extend(field_results)
    
    overall_metrics = evaluator.calculate_metrics(all_results_flat) if all_results_flat else {}
    
    # BERTScore aggregation (if available)
    bert_scores_per_paper = [p["bert_score_aggregate"] for p in paper_results 
                             if p.get("bert_score_aggregate") is not None]
    bert_score_aggregate = (
        sum(bert_scores_per_paper) / len(bert_scores_per_paper)
        if bert_scores_per_paper else None
    )
    
    return {
        "summary": {
            "total_papers_evaluated": len(extraction_files),
            "successful_evaluations": len([p for p in paper_results if "error" not in p])
        },
        "overall_metrics": overall_metrics,
        "field_metrics": field_metrics,
        "bert_score_aggregate": bert_score_aggregate,
        "paper_results": paper_results
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate evaluation for one model across all papers")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing extraction outputs for this model"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name (for results table)"
    )
    parser.add_argument(
        "--gold",
        type=str,
        default="data/gold_standard/R1364660.json",
        help="Path to gold standard"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for aggregated results JSON"
    )
    
    args = parser.parse_args()
    
    model_dir = PROJECT_ROOT / args.model_dir
    gold_path = PROJECT_ROOT / args.gold
    output_path = PROJECT_ROOT / args.output
    
    if not model_dir.exists():
        logger.error(f"Model directory not found: {model_dir}")
        return 1
    
    if not gold_path.exists():
        logger.error(f"Gold standard not found: {gold_path}")
        return 1
    
    # Initialize evaluator
    evaluator = StrictExtractionEvaluator(
        fuzzy_threshold=0.8,
        use_semantic=True,
        bert_score_model="roberta-large"
    )
    
    # Aggregate evaluations
    aggregated = aggregate_evaluations(model_dir, gold_path, evaluator)
    
    if not aggregated:
        logger.error("Aggregation failed")
        return 1
    
    # Add model name
    aggregated["model_name"] = args.model_name
    
    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(aggregated, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print("AGGREGATED EVALUATION RESULTS")
    print("=" * 70)
    print(f"Model:              {args.model_name}")
    print(f"Papers evaluated:   {aggregated['summary']['total_papers_evaluated']}")
    print(f"Successful:         {aggregated['summary']['successful_evaluations']}")
    print()
    print("AGGREGATED METRICS:")
    print(f"  Overall F1:              {aggregated['overall_metrics'].get('f1_score', 0):.4f}")
    print(f"  Overall Precision:       {aggregated['overall_metrics'].get('precision', 0):.4f}")
    print(f"  Overall Recall:          {aggregated['overall_metrics'].get('recall', 0):.4f}")
    if aggregated.get("bert_score_aggregate"):
        print(f"  BERTScore Aggregate:     {aggregated['bert_score_aggregate']:.4f}")
    print("=" * 70)
    print(f"\nSaved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
