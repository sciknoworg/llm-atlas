"""
Baseline Model Filter

Filters out baseline/ablation models from extraction results.
Gold standard typically contains only the main contribution, not comparison baselines.

Usage:
    from src.baseline_filter import filter_baseline_models
    
    filtered = filter_baseline_models(extraction_data, paper_metadata)
"""

import logging
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def filter_baseline_models(
    models: List[Dict[str, Any]],
    paper_metadata: Optional[Dict[str, Any]] = None,
    keep_top_n: int = 1
) -> List[Dict[str, Any]]:
    """
    Filter baseline/ablation models, keeping only main contributions.
    
    Args:
        models: List of extracted models
        paper_metadata: Optional paper metadata (title, etc.)
        keep_top_n: Number of top models to keep (default 1)
    
    Returns:
        Filtered list of models (main contributions only)
    """
    if not models:
        return []
    
    if len(models) == 1:
        return models
    
    logger.info(f"Filtering {len(models)} models (keeping top {keep_top_n})")
    
    # Score each model
    scored_models = []
    for model in models:
        score = _score_model(model, paper_metadata)
        scored_models.append((score, model))
    
    # Sort by score (descending) and keep top N
    scored_models.sort(key=lambda x: x[0], reverse=True)
    filtered = [m for (score, m) in scored_models[:keep_top_n]]
    
    logger.info(f"Kept {len(filtered)} model(s):")
    for model in filtered:
        logger.info(f"  - {model.get('model_name')} (score: {scored_models[0][0]:.2f})")
    
    if len(filtered) < len(models):
        logger.info(f"Filtered out {len(models) - len(filtered)} baseline/ablation models:")
        for score, model in scored_models[keep_top_n:]:
            logger.info(f"  - {model.get('model_name')} (score: {score:.2f})")
    
    return filtered


def _score_model(model: Dict[str, Any], paper_metadata: Optional[Dict[str, Any]]) -> float:
    """
    Score a model (higher = more likely to be main contribution).
    
    Scoring criteria:
    - Has parameters specified (+2)
    - Has most parameters (+1 per parameter tier)
    - Model family matches paper title keywords (+3)
    - Not labeled as baseline/ablation (+5)
    - Has innovation field (+1)
    - Has detailed fields (architecture, task, optimizer) (+0.5 each)
    """
    score = 0.0
    model_name = (model.get("model_name") or "").strip().lower()
    
    # Negative: baseline/ablation keywords in name
    baseline_keywords = ["baseline", "ablation", "lstm", "rnn", "gru", "auxiliary"]
    if any(kw in model_name for kw in baseline_keywords):
        score -= 10  # Strong penalty
        logger.debug(f"Baseline keyword found in '{model.get('model_name')}': -10")
    
    # Positive: has parameters
    params = model.get("parameters")
    params_millions = model.get("parameters_millions")
    if params or params_millions:
        score += 2
        # Bonus for larger models (likely main contribution)
        if params_millions and isinstance(params_millions, (int, float)):
            if params_millions >= 1000:  # 1B+
                score += 3
            elif params_millions >= 100:  # 100M+
                score += 2
            elif params_millions >= 10:   # 10M+
                score += 1
    
    # Positive: model family/name matches paper title
    if paper_metadata and "title" in paper_metadata:
        title = paper_metadata["title"].lower()
        model_family = (model.get("model_family") or "").strip().lower()
        
        # Extract key model names from title (e.g., "GPT", "BERT", "Llama")
        model_keywords = re.findall(r"\b(gpt|bert|llama|t5|bart|transformer|roberta|albert)\b", title)
        
        if model_family and any(kw in model_family for kw in model_keywords):
            score += 3
            logger.debug(f"Model family '{model_family}' matches paper title: +3")
        
        # Also check model name
        if any(kw in model_name for kw in model_keywords):
            score += 2
            logger.debug(f"Model name '{model_name}' matches paper title: +2")
    
    # Positive: has detailed fields
    if model.get("innovation"):
        score += 1
    if model.get("pretraining_architecture"):
        score += 0.5
    if model.get("pretraining_task"):
        score += 0.5
    if model.get("optimizer"):
        score += 0.5
    
    # Positive: has organization
    if model.get("organization"):
        score += 1
    
    # Positive: specific model version/name (not generic)
    # Generic names like "GPT", "Transformer" are less likely to be main contribution
    if model_family := model.get("model_family"):
        if model_name and model_name != model_family.lower():
            score += 1  # Has specific name beyond just family
    
    return score


def is_baseline_model(model: Dict[str, Any]) -> bool:
    """
    Check if a model is likely a baseline/ablation.
    
    Returns:
        True if likely baseline, False if likely main contribution
    """
    model_name = (model.get("model_name") or "").strip().lower()
    
    # Strong indicators of baseline/ablation
    baseline_keywords = [
        "baseline", "ablation", "lstm", "rnn", "gru", 
        "auxiliary", "aux", "comparison", "baseline model"
    ]
    
    if any(kw in model_name for kw in baseline_keywords):
        return True
    
    # Weak indicators (could be main model or baseline)
    # If model has very few fields filled, might be a quick mention
    filled_fields = sum(1 for v in model.values() if v is not None and str(v).strip())
    if filled_fields < 5:  # Less than 5 fields filled
        return True
    
    return False
