"""
Extraction output normalizer.

Post-processes extracted models for consistent formats (date, organization)
so stored data and evaluation are more reliable. Used after extraction, before merge.
"""

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Canonical organization names: variant -> canonical
ORGANIZATION_ALIASES = {
    "google ai language": "Google",
    "google research": "Google",
    "google deepmind": "Google",
    "meta ai": "Meta",
    "facebook ai": "Meta",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "mistral ai": "Mistral AI",
    "hugging face": "Hugging Face",
    "microsoft research": "Microsoft",
    "nvidia": "NVIDIA",
    "deepmind": "DeepMind",
    "cohere": "Cohere",
}


def normalize_date_created(value: Any) -> Any:
    """
    Normalize date_created to YYYY-MM when possible.
    - "2018" -> "2018-01"
    - "2018-10" -> "2018-10"
    - "2018-10-01" -> "2018-10"
    - Invalid or empty -> return as-is
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return value
    s = str(value).strip()
    # Already YYYY-MM or YYYY-MM-DD
    m = re.match(r"^(\d{4})-(\d{2})(?:-\d{2})?$", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    # Year only
    m = re.match(r"^(\d{4})$", s)
    if m:
        return f"{m.group(1)}-01"
    return value


def normalize_organization(value: Any) -> Any:
    """
    Normalize organization to canonical name when known.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return value
    s = str(value).strip().lower()
    for variant, canonical in ORGANIZATION_ALIASES.items():
        if variant in s or s in variant:
            return canonical
    # Capitalize first letter of each word if all lowercase
    if s == s.lower() and " " in s:
        return value.strip().title()
    return value


def normalize_extraction(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize extracted model dicts for consistent date and organization format.
    Does not mutate input; returns new list of dicts.
    """
    if not models:
        return []
    out = []
    for m in models:
        m2 = dict(m)
        if "date_created" in m2 and m2["date_created"]:
            m2["date_created"] = normalize_date_created(m2["date_created"])
        if "organization" in m2 and m2["organization"]:
            m2["organization"] = normalize_organization(m2["organization"])
        out.append(m2)
    return out
