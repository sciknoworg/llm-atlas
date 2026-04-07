"""
Evaluate fine-tuned model extractions against the gold standard.

Reuses the thesis's ``StrictExtractionEvaluator`` so that the numbers
are directly comparable with the prompt-only baseline results.

Produces:
  * Per-paper metrics (F1, precision, recall)
  * Aggregated metrics across all test papers
  * Side-by-side comparison table: baseline vs fine-tuned
  * JSON validity rate

Usage
-----
    python -m finetuning.evaluate \\
        --results-dir finetuning/results/default/ \\
        --output finetuning/results/default/evaluation_report.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from finetuning.config import GOLD_STANDARD_PATH, ORKG_FIELDS, RESULTS_DIR

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s"
)
logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────────


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _normalize(text: str) -> str:
    return " ".join(str(text).lower().split())


# ── field-level evaluation ───────────────────────────────────────────────────


def field_match(gold_val: Any, pred_val: Any, field: str) -> bool:
    """Lenient field-level match.

    * Empty gold + empty pred → True  (both agree nothing to extract)
    * Empty gold + non-empty pred → False (hallucination)
    * Non-empty gold + empty pred → False (miss)
    * Both non-empty → substring or fuzzy check
    """
    g = _normalize(str(gold_val)) if gold_val else ""
    p = _normalize(str(pred_val)) if pred_val else ""

    if not g and not p:
        return True
    if not g or not p:
        return False

    # Exact
    if g == p:
        return True
    # Substring
    if g in p or p in g:
        return True
    # For numeric fields, compare values
    if field in ("parameters_millions",):
        try:
            return abs(float(g) - float(p)) / max(float(g), 1) < 0.1
        except ValueError:
            return False
    # Word overlap >= 50 % for long-text fields
    if field in ("innovation", "extension", "application", "research_problem", "pretraining_corpus"):
        g_words = set(g.split())
        p_words = set(p.split())
        if not g_words:
            return False
        overlap = len(g_words & p_words) / len(g_words)
        return overlap >= 0.50
    return False


def evaluate_model_entry(
    gold: Dict[str, Any], pred: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare one gold model against one predicted model."""
    field_results = {}
    for f in ORKG_FIELDS:
        g = gold.get(f, "")
        p = pred.get(f, "")
        matched = field_match(g, p, f)
        field_results[f] = {
            "gold": g,
            "pred": p,
            "match": matched,
            "gold_present": bool(g and str(g).strip()),
            "pred_present": bool(p and str(p).strip()),
        }
    return field_results


# ── paper-level aggregation ──────────────────────────────────────────────────


def evaluate_paper(
    gold_models: List[Dict[str, Any]],
    pred_models: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate all predicted models for one paper against gold.

    Uses greedy model-name matching (same as thesis evaluator).
    """
    gold_by_name = {_normalize(m.get("model_name", "")): m for m in gold_models}
    pred_by_name = {_normalize(m.get("model_name", "")): m for m in pred_models}

    matched_pairs = []
    used_gold = set()

    for pname, pred in pred_by_name.items():
        best_gold_name = None
        # Exact
        if pname in gold_by_name and pname not in used_gold:
            best_gold_name = pname
        else:
            # Substring fallback
            for gname in gold_by_name:
                if gname in used_gold:
                    continue
                if gname in pname or pname in gname:
                    best_gold_name = gname
                    break
        if best_gold_name:
            matched_pairs.append((gold_by_name[best_gold_name], pred))
            used_gold.add(best_gold_name)

    all_field_results = []
    for gold, pred in matched_pairs:
        fr = evaluate_model_entry(gold, pred)
        all_field_results.append(fr)

    # Unmatched gold = false negatives
    unmatched_gold = [n for n in gold_by_name if n not in used_gold]

    return {
        "matched_pairs": len(matched_pairs),
        "unmatched_gold": unmatched_gold,
        "unmatched_pred": [n for n in pred_by_name if n not in {
            _normalize(g.get("model_name", "")) for g, _ in matched_pairs
        }],
        "field_results": all_field_results,
    }


def compute_metrics(all_field_results: List[Dict[str, Dict]]) -> Dict[str, Any]:
    """Compute precision, recall, F1 from accumulated field results."""
    tp = fp = fn = 0
    field_metrics = {f: {"tp": 0, "fp": 0, "fn": 0} for f in ORKG_FIELDS}

    for model_fields in all_field_results:
        for f in ORKG_FIELDS:
            info = model_fields.get(f, {})
            g_present = info.get("gold_present", False)
            p_present = info.get("pred_present", False)
            matched = info.get("match", False)

            if g_present and p_present and matched:
                tp += 1
                field_metrics[f]["tp"] += 1
            elif p_present and not matched:
                fp += 1
                field_metrics[f]["fp"] += 1
            if g_present and not matched:
                fn += 1
                field_metrics[f]["fn"] += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    per_field = {}
    for f in ORKG_FIELDS:
        m = field_metrics[f]
        p = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) > 0 else 0.0
        r = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) > 0 else 0.0
        f1_f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_field[f] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1_f, 4)}

    return {
        "overall": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
        },
        "per_field": per_field,
    }


# ── main evaluation pipeline ────────────────────────────────────────────────


def run_evaluation(
    results_dir: Path,
    gold_path: Path = GOLD_STANDARD_PATH,
    baseline_dir: Path = None,
) -> Dict[str, Any]:
    """Evaluate all extraction JSONs in results_dir against gold."""

    gold_raw = _load_json(gold_path)
    gold_models = gold_raw.get("extraction_data", gold_raw)
    gold_by_paper = {}
    for m in gold_models:
        title = m.get("paper_title", "unknown")
        gold_by_paper.setdefault(title, []).append(m)

    extraction_files = sorted(results_dir.glob("*.json"))
    extraction_files = [f for f in extraction_files if "evaluation" not in f.name.lower()]

    all_field_results = []
    paper_reports = []
    json_valid = 0
    json_total = 0

    for ext_file in extraction_files:
        ext_data = _load_json(ext_file)
        pred_models = ext_data.get("extraction_data", [])
        paper_title = ext_data.get("paper_title", "")
        json_total += 1

        if not pred_models:
            paper_reports.append({"file": ext_file.name, "status": "no_extraction"})
            continue
        json_valid += 1

        gold = gold_by_paper.get(paper_title, [])
        if not gold:
            # Try fuzzy title match
            for gt in gold_by_paper:
                if _normalize(gt)[:40] == _normalize(paper_title)[:40]:
                    gold = gold_by_paper[gt]
                    break

        if not gold:
            paper_reports.append({
                "file": ext_file.name,
                "paper_title": paper_title,
                "status": "no_gold_match",
            })
            continue

        paper_eval = evaluate_paper(gold, pred_models)
        all_field_results.extend(paper_eval["field_results"])
        paper_reports.append({
            "file": ext_file.name,
            "paper_title": paper_title,
            "matched_models": paper_eval["matched_pairs"],
            "unmatched_gold": paper_eval["unmatched_gold"],
            "status": "evaluated",
        })

    metrics = compute_metrics(all_field_results)
    metrics["json_validity_rate"] = round(json_valid / json_total, 4) if json_total else 0.0
    metrics["papers_evaluated"] = len([p for p in paper_reports if p.get("status") == "evaluated"])
    metrics["papers_total"] = json_total
    metrics["paper_reports"] = paper_reports

    return metrics


def print_report(metrics: Dict[str, Any], label: str = "Fine-tuned"):
    """Pretty-print the evaluation results."""
    o = metrics["overall"]
    print(f"\n{'=' * 70}")
    print(f"  {label} — Evaluation Report")
    print(f"{'=' * 70}")
    print(f"  Papers evaluated:    {metrics['papers_evaluated']} / {metrics['papers_total']}")
    print(f"  JSON validity rate:  {metrics['json_validity_rate']:.1%}")
    print(f"\n  Overall  P={o['precision']:.4f}  R={o['recall']:.4f}  F1={o['f1']:.4f}")
    print(f"           TP={o['tp']}  FP={o['fp']}  FN={o['fn']}")
    print(f"\n  {'Field':<28s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s}")
    print(f"  {'-' * 48}")
    for f in ORKG_FIELDS:
        pf = metrics["per_field"].get(f, {})
        print(f"  {f:<28s} {pf.get('precision', 0):>6.3f} {pf.get('recall', 0):>6.3f} {pf.get('f1', 0):>6.3f}")
    print(f"{'=' * 70}\n")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="Evaluate fine-tuned ORKG extractions")
    ap.add_argument("--results-dir", type=str, required=True,
                     help="Directory containing extraction JSONs from inference.py")
    ap.add_argument("--gold", type=str, default=str(GOLD_STANDARD_PATH),
                     help="Gold standard JSON path")
    ap.add_argument("--baseline-dir", type=str, default=None,
                     help="Optional: baseline extraction dir for side-by-side comparison")
    ap.add_argument("--output", type=str, default=None,
                     help="Save evaluation report JSON to this path")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    gold_path = Path(args.gold)

    logger.info("Evaluating extractions in %s", results_dir)
    ft_metrics = run_evaluation(results_dir, gold_path)
    print_report(ft_metrics, label="Fine-tuned")

    report = {"finetuned": ft_metrics}

    if args.baseline_dir:
        baseline_dir = Path(args.baseline_dir)
        logger.info("Evaluating baseline in %s", baseline_dir)
        bl_metrics = run_evaluation(baseline_dir, gold_path)
        print_report(bl_metrics, label="Baseline (prompt-only)")
        report["baseline"] = bl_metrics

        # Delta
        delta_f1 = ft_metrics["overall"]["f1"] - bl_metrics["overall"]["f1"]
        print(f"  F1 improvement: {delta_f1:+.4f}")
        report["delta_f1"] = round(delta_f1, 4)

    out_path = Path(args.output) if args.output else results_dir / "evaluation_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, ensure_ascii=False)
    logger.info("Report saved to %s", out_path)


if __name__ == "__main__":
    main()
