"""
Strict Evaluation Metrics for LLM Extraction Pipeline

This script uses STRICT matching (no relaxations) to measure real extraction accuracy.
Use this for thesis evaluation metrics.

For relaxed evaluation (better UX), use evaluate_extraction.py instead.

Usage:
    python scripts/evaluation/evaluate_extraction_strict.py \\
        --gold data/gold_standard/R1364660.json \\
        --prediction data/extracted/2401.02385_20251207_223913.json
        
Output:
    Evaluation report with per-field and overall metrics (STRICT)
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrictExtractionEvaluator:
    """Evaluates extraction quality with STRICT matching (no relaxations)."""
    
    # Fields to evaluate from ORKG template R609825
    EVALUATION_FIELDS = [
        "model_name",
        "model_family",
        "date_created",
        "organization",
        "innovation",
        "pretraining_architecture",
        "pretraining_task",
        "pretraining_corpus",
        "finetuning_task",
        "optimizer",
        "parameters",
        "parameters_millions",
        "hardware_used",
        "extension",
        "blog_post",
        "license",
        "research_problem",
        "application"
    ]
    
    # Only long-text fields use fuzzy matching (with strict threshold 0.8)
    FUZZY_FIELDS = ["innovation", "pretraining_corpus", "application", "research_problem"]

    def __init__(self, fuzzy_threshold: float = 0.8):
        """
        Initialize strict evaluator.
        
        Args:
            fuzzy_threshold: Similarity threshold for fuzzy string matching (0-1)
                             Default 0.8 (strict). Use 1.0 for exact-only.
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.results = {}
        
    def normalize_value(self, value: Any) -> str:
        """Normalize a value for comparison. STRICT: only None and empty string are missing."""
        if value is None or value == "":
            return ""
        # STRICT: "null", "none", "n/a" are treated as actual values (not missing)
        return str(value).strip().lower()

    def fuzzy_match(self, str1: str, str2: str) -> float:
        """
        Calculate similarity between two strings using SequenceMatcher.
        
        Returns:
            Similarity score between 0 and 1
        """
        if not str1 and not str2:
            return 1.0  # Both empty = match
        if not str1 or not str2:
            return 0.0  # One empty, one not = no match
            
        return SequenceMatcher(None, str1, str2).ratio()
    
    def compare_field(
        self,
        gold_value: Any,
        pred_value: Any,
        use_fuzzy: bool = False,
        field: Optional[str] = None,
    ) -> Tuple[bool, float]:
        """
        Compare a single field between gold and prediction (STRICT).
        
        Returns:
            (is_match: bool, similarity: float)
        """
        gold_norm = self.normalize_value(gold_value)
        pred_norm = self.normalize_value(pred_value)

        # Exact match
        if gold_norm == pred_norm:
            return True, 1.0
        
        # Fuzzy match (ONLY for long-text fields, strict threshold 0.8)
        if use_fuzzy and field in self.FUZZY_FIELDS and gold_norm and pred_norm:
            similarity = self.fuzzy_match(gold_norm, pred_norm)
            if similarity >= self.fuzzy_threshold:
                return True, similarity
            return False, similarity
        
        return False, 0.0
    
    def evaluate_model(self, gold_model: Dict[str, Any], pred_model: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single model extraction against gold standard.
        
        Returns:
            Dictionary with field-level evaluation results
        """
        field_results = {}
        
        for field in self.EVALUATION_FIELDS:
            gold_value = gold_model.get(field)
            pred_value = pred_model.get(field)
            
            use_fuzzy = field in self.FUZZY_FIELDS
            
            is_match, similarity = self.compare_field(
                gold_value, pred_value, use_fuzzy=use_fuzzy, field=field
            )
            
            field_results[field] = {
                "match": is_match,
                "similarity": similarity,
                "gold": gold_value,
                "predicted": pred_value
            }
        
        return field_results
    
    def calculate_metrics(self, field_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate precision, recall, F-score, and accuracy for a set of field results.
        
        Metrics:
        - Accuracy: (TP + TN) / (TP + TN + FP + FN)
        - Precision: TP / (TP + FP) - Of all predicted values, how many were correct?
        - Recall: TP / (TP + FN) - Of all gold values, how many were found?
        - F-score (F1): 2 * (Precision * Recall) / (Precision + Recall)
        
        Where:
        - TP (True Positive): Field exists in gold, correctly extracted
        - FP (False Positive): Field extracted but doesn't match gold
        - FN (False Negative): Field exists in gold but not extracted or wrong
        - TN (True Negative): Field doesn't exist in gold, not extracted
        """
        tp = 0  # Correct predictions
        fp = 0  # Incorrect predictions (predicted but wrong)
        fn = 0  # Missed (gold exists but not predicted correctly)
        tn = 0  # Correctly identified as absent
        
        for result in field_results:
            gold = result.get("gold")
            pred = result.get("predicted")
            match = result.get("match", False)
            
            gold_exists = gold is not None and str(gold).strip() not in ["", "None", "null"]
            pred_exists = pred is not None and str(pred).strip() not in ["", "None", "null"]
            
            if gold_exists and pred_exists and match:
                tp += 1
            elif gold_exists and pred_exists and not match:
                fp += 1
            elif gold_exists and not pred_exists:
                fn += 1
            elif not gold_exists and not pred_exists:
                tn += 1
            elif not gold_exists and pred_exists:
                fp += 1
        
        # Calculate metrics
        total = tp + fp + fn + tn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "total_fields": total
        }
    
    def filter_gold_by_paper_title(
        self,
        gold_data: List[Dict[str, Any]],
        paper_title: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Filter gold standard data by paper title.
        
        Args:
            gold_data: List of gold-standard models
            paper_title: Paper title to filter by (optional)
            
        Returns:
            Filtered list of gold-standard models
        """
        if not paper_title:
            return gold_data
        
        paper_title_norm = self.normalize_value(paper_title)
        filtered = []
        
        for model in gold_data:
            model_paper_title = self.normalize_value(model.get("paper_title", ""))
            if model_paper_title and model_paper_title == paper_title_norm:
                filtered.append(model)
        
        if len(filtered) == 0:
            logger.warning(f"No models found with paper_title='{paper_title}' in gold standard!")
            logger.warning("Falling back to matching by model_name only (no filtering)")
            return gold_data
        
        logger.info(f"Filtered gold standard: {len(gold_data)} -> {len(filtered)} models (paper: {paper_title})")
        return filtered
    
    def evaluate_dataset(
        self, 
        gold_data: List[Dict[str, Any]], 
        pred_data: List[Dict[str, Any]],
        paper_title: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate entire dataset.
        
        Args:
            gold_data: List of gold-standard models
            pred_data: List of predicted/extracted models
            paper_title: Optional paper title to filter gold standard
            
        Returns:
            Comprehensive evaluation report
        """
        if paper_title:
            gold_data = self.filter_gold_by_paper_title(gold_data, paper_title)
        
        logger.info(f"Evaluating {len(pred_data)} predictions against {len(gold_data)} gold-standard models (STRICT)")
        
        # Match models by model_name: exact match first, then fallback "gold in pred"
        gold_by_name = {self.normalize_value(m.get("model_name")): m for m in gold_data}
        gold_matched_names = set()

        field_level_results = {field: [] for field in self.EVALUATION_FIELDS}
        model_level_results = []
        matched_count = 0
        unmatched_predictions = []

        def find_gold_for_pred(pred_norm: str):
            if pred_norm in gold_by_name and pred_norm not in gold_matched_names:
                return gold_by_name[pred_norm], pred_norm
            for gnorm, g in gold_by_name.items():
                if gnorm in gold_matched_names or not gnorm:
                    continue
                if gnorm in pred_norm:
                    return g, gnorm
            return None, None

        for pred_model in pred_data:
            pred_name = self.normalize_value(pred_model.get("model_name"))
            pair = find_gold_for_pred(pred_name)
            gold_model, matched_gnorm = pair[0], pair[1]

            if gold_model is not None:
                matched_count += 1
                gold_matched_names.add(matched_gnorm)
                model_eval = self.evaluate_model(gold_model, pred_model)
                model_level_results.append({
                    "model_name": pred_model.get("model_name"),
                    "fields": model_eval
                })
                for field in self.EVALUATION_FIELDS:
                    field_level_results[field].append(model_eval[field])
            else:
                unmatched_predictions.append(pred_model.get("model_name"))

        # Calculate per-field metrics
        field_metrics = {}
        for field in self.EVALUATION_FIELDS:
            if field_level_results[field]:
                field_metrics[field] = self.calculate_metrics(field_level_results[field])

        # Calculate overall metrics (across all fields and models)
        all_field_results = []
        for field_results in field_level_results.values():
            all_field_results.extend(field_results)

        overall_metrics = self.calculate_metrics(all_field_results)

        # Missing models: gold not matched to any prediction
        missing_models = [m.get("model_name") for m in gold_data
                         if self.normalize_value(m.get("model_name")) not in gold_matched_names]
        
        return {
            "summary": {
                "total_gold_models": len(gold_data),
                "total_predicted_models": len(pred_data),
                "matched_models": matched_count,
                "unmatched_predictions": len(unmatched_predictions),
                "missing_models": len(missing_models)
            },
            "overall_metrics": overall_metrics,
            "field_metrics": field_metrics,
            "model_results": model_level_results,
            "unmatched_predictions": unmatched_predictions,
            "missing_models": missing_models
        }
    
    def print_report(self, evaluation: Dict[str, Any]) -> None:
        """Print formatted evaluation report."""
        print("\n" + "=" * 80)
        print("STRICT EXTRACTION EVALUATION REPORT")
        print("=" * 80)
        
        # Summary
        summary = evaluation["summary"]
        print(f"\nSummary:")
        print(f"  Gold-standard models:    {summary['total_gold_models']}")
        print(f"  Predicted models:        {summary['total_predicted_models']}")
        print(f"  Matched models:          {summary['matched_models']}")
        print(f"  Unmatched predictions:   {summary['unmatched_predictions']}")
        print(f"  Missing models:          {summary['missing_models']}")
        
        # Overall metrics
        overall = evaluation["overall_metrics"]
        print(f"\n{'=' * 80}")
        print("OVERALL METRICS (All Fields Combined) - STRICT")
        print("=" * 80)
        print(f"  Accuracy:        {overall['accuracy']:.2%}")
        print(f"  Precision:       {overall['precision']:.2%}")
        print(f"  Recall:          {overall['recall']:.2%}")
        print(f"  F1-Score:        {overall['f1_score']:.2%}")
        print(f"\n  True Positives:  {overall['true_positives']}")
        print(f"  False Positives: {overall['false_positives']}")
        print(f"  False Negatives: {overall['false_negatives']}")
        print(f"  True Negatives:  {overall['true_negatives']}")
        
        # Per-field metrics
        print(f"\n{'=' * 80}")
        print("PER-FIELD METRICS")
        print("=" * 80)
        print(f"{'Field':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 80)
        
        field_metrics = evaluation["field_metrics"]
        for field in self.EVALUATION_FIELDS:
            if field in field_metrics:
                metrics = field_metrics[field]
                print(f"{field:<30} {metrics['accuracy']:<12.2%} {metrics['precision']:<12.2%} "
                      f"{metrics['recall']:<12.2%} {metrics['f1_score']:<12.2%}")
        
        # Top performing fields
        print(f"\n{'=' * 80}")
        print("TOP 5 PERFORMING FIELDS (by F1-Score)")
        print("=" * 80)
        sorted_fields = sorted(
            [(field, metrics["f1_score"]) for field, metrics in field_metrics.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        for i, (field, f1) in enumerate(sorted_fields, 1):
            print(f"{i}. {field:<30} F1: {f1:.2%}")
        
        # Bottom performing fields
        print(f"\n{'=' * 80}")
        print("BOTTOM 5 PERFORMING FIELDS (by F1-Score)")
        print("=" * 80)
        sorted_fields_bottom = sorted(
            [(field, metrics["f1_score"]) for field, metrics in field_metrics.items()],
            key=lambda x: x[1]
        )[:5]
        for i, (field, f1) in enumerate(sorted_fields_bottom, 1):
            print(f"{i}. {field:<30} F1: {f1:.2%}")
        
        # Unmatched/missing
        if evaluation["unmatched_predictions"]:
            print(f"\n{'=' * 80}")
            print("UNMATCHED PREDICTIONS (not in gold-standard)")
            print("=" * 80)
            for name in evaluation["unmatched_predictions"]:
                print(f"  - {name}")
        
        if evaluation["missing_models"]:
            print(f"\n{'=' * 80}")
            print("MISSING MODELS (in gold-standard but not predicted)")
            print("=" * 80)
            for name in evaluation["missing_models"][:10]:
                print(f"  - {name}")
            if len(evaluation["missing_models"]) > 10:
                print(f"  ... and {len(evaluation['missing_models']) - 10} more")


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate extraction with STRICT matching (thesis metrics)")
    parser.add_argument(
        "--gold",
        type=str,
        default="data/gold_standard/R1364660.json",
        help="Path to gold-standard JSON file"
    )
    parser.add_argument(
        "--prediction",
        type=str,
        required=True,
        help="Path to extracted/predicted JSON file"
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=0.8,
        help="Similarity threshold for fuzzy matching (0-1). Use 1.0 for exact-only."
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional: Save evaluation report as JSON"
    )
    parser.add_argument(
        "--paper-title",
        type=str,
        help="Optional: Paper title to filter gold standard (auto-detected from prediction JSON if not provided)"
    )
    
    args = parser.parse_args()
    
    # Load data
    project_root = Path(__file__).parent.parent.parent
    gold_path = project_root / args.gold
    pred_path = project_root / args.prediction
    
    if not gold_path.exists():
        logger.error(f"Gold-standard file not found: {gold_path}")
        return
    
    if not pred_path.exists():
        logger.error(f"Prediction file not found: {pred_path}")
        return
    
    logger.info(f"Loading gold-standard from: {gold_path}")
    gold_data_raw = load_json(gold_path)
    gold_data = gold_data_raw.get("extraction_data", gold_data_raw)
    
    logger.info(f"Loading predictions from: {pred_path}")
    pred_data_raw = load_json(pred_path)
    pred_data = (
        pred_data_raw.get("extraction_data") or 
        pred_data_raw.get("raw_extraction") or 
        pred_data_raw
    )
    
    if not isinstance(gold_data, list):
        logger.error("Gold-standard data is not a list")
        return
    
    if not isinstance(pred_data, list):
        logger.error("Prediction data is not a list")
        return
    
    # Extract paper title from prediction JSON (if not provided via CLI)
    paper_title = args.paper_title
    if not paper_title:
        if "paper_metadata" in pred_data_raw and "title" in pred_data_raw["paper_metadata"]:
            paper_title = pred_data_raw["paper_metadata"]["title"]
        elif isinstance(pred_data, list) and len(pred_data) > 0:
            paper_title = pred_data[0].get("paper_title")
        elif "paper_title" in pred_data_raw:
            paper_title = pred_data_raw["paper_title"]
    
    if paper_title:
        logger.info(f"Using paper title for filtering: {paper_title}")
    
    # Evaluate with STRICT matching
    evaluator = StrictExtractionEvaluator(fuzzy_threshold=args.fuzzy_threshold)
    evaluation = evaluator.evaluate_dataset(gold_data, pred_data, paper_title=paper_title)
    
    # Print report
    evaluator.print_report(evaluation)
    
    # Save report if requested
    if args.output:
        output_path = project_root / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation, f, indent=2, ensure_ascii=False)
        logger.info(f"\nEvaluation report saved to: {output_path}")
    
    # Return exit code based on F1 score
    f1 = evaluation["overall_metrics"]["f1_score"]
    if f1 >= 0.8:
        print(f"\n[OK] EXCELLENT: F1-Score {f1:.2%} >= 80%")
        return 0
    elif f1 >= 0.6:
        print(f"\n[OK] GOOD: F1-Score {f1:.2%} >= 60%")
        return 0
    elif f1 >= 0.4:
        print(f"\n[~] FAIR: F1-Score {f1:.2%} >= 40%")
        return 1
    else:
        print(f"\n[FAIL] POOR: F1-Score {f1:.2%} < 40%")
        return 1


if __name__ == "__main__":
    exit(main())
