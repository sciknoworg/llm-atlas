"""
Model contribution selector.

Filters extracted entries to keep primary, contribution-level models while
removing auxiliary artifacts (e.g., adapters, safety filters, tooling) when a
paper contains both.
"""

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Generic auxiliary terms that are usually not standalone model contributions
# in comparison-style extractions.
_AUXILIARY_KEYWORDS = {
    "adapter",
    "chat",
    "classifier",
    "component",
    "context",
    "detector",
    "embedding",
    "encoder",
    "expert",
    "evaluator",
    "filter",
    "framework",
    "guard",
    "instruct",
    "module",
    "parser",
    "pipeline",
    "prompt guard",
    "resampler",
    "reward model",
    "safety",
    "system-level",
    "tokenizer",
    "tool",
}


def _norm_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _has_auxiliary_keyword(text: str) -> bool:
    if not text:
        return False
    # Use boundary-aware matching for phrases and flexible matching for single
    # tokens to catch concatenations like "LlamaGuard".
    for kw in _AUXILIARY_KEYWORDS:
        if " " in kw or "-" in kw:
            pattern = r"\b" + re.escape(kw) + r"\b"
        else:
            pattern = r"\b\w*" + re.escape(kw) + r"\w*\b"
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True
    return False


def _has_release_signal(model: Dict[str, Any]) -> bool:
    name = _norm_text(model.get("model_name"))
    params = _norm_text(model.get("parameters"))
    model_version = _norm_text(model.get("model_version"))

    # Version-like signal: 3, 3.1, v2, etc.
    has_version_token = bool(re.search(r"\b(v?\d+(?:\.\d+)?)\b", name))
    # Size-like signal: 7B, 405B, 117M, etc.
    has_size_token = bool(re.search(r"\b\d+(?:\.\d+)?\s*[mbt]\b", name + " " + params))

    return has_version_token or has_size_token or bool(model_version)


def _info_richness(model: Dict[str, Any]) -> int:
    fields = [
        "model_name",
        "model_family",
        "organization",
        "innovation",
        "pretraining_task",
        "pretraining_architecture",
        "parameters",
        "license",
    ]
    return sum(1 for f in fields if model.get(f) not in (None, "", "null", "None"))


def _dominant_family(models: List[Dict[str, Any]]) -> Optional[str]:
    families = [_norm_text(m.get("model_family")) for m in models]
    families = [f for f in families if f]
    if not families:
        return None
    family, count = Counter(families).most_common(1)[0]
    # Require at least two occurrences to avoid overfitting to noise.
    return family if count >= 2 else None


def _is_primary_contribution(model: Dict[str, Any], dominant_family: Optional[str]) -> bool:
    name = _norm_text(model.get("model_name"))
    family = _norm_text(model.get("model_family"))
    combined = f"{name} {family}"

    if not name:
        return False

    has_auxiliary = _has_auxiliary_keyword(combined)
    family_mismatch = bool(dominant_family and family and family != dominant_family)
    release_signal = _has_release_signal(model)
    richness = _info_richness(model)

    # Keep entries that look like model releases/contributions, not components.
    return (not has_auxiliary) and (not family_mismatch) and (release_signal or richness >= 4)


def select_primary_model_contributions(
    models: List[Dict[str, Any]],
    paper_metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Keep primary model contributions and remove auxiliary entries.

    The selector is conservative:
    - If no confident primary subset is found, it returns the original list.
    - Filtering is applied only when it can remove a clear auxiliary subset.
    """
    if not models or len(models) <= 1:
        return models

    dominant_family = _dominant_family(models)
    primary = [m for m in models if _is_primary_contribution(m, dominant_family)]

    # Conservative fallback: avoid destructive filtering when uncertain.
    if not primary:
        logger.info("Contribution selector: no confident primary subset found, keeping all models")
        return models

    if len(primary) == len(models):
        logger.info("Contribution selector: all extracted models look like primary contributions")
        return models

    removed = len(models) - len(primary)
    logger.info(
        "Contribution selector: filtered %s auxiliary/non-primary entries (%s -> %s)",
        removed,
        len(models),
        len(primary),
    )
    return primary

