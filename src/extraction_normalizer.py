"""
Extraction output normalizer.

Post-processes extracted models for consistent formats (date, organization)
so stored data and evaluation are more reliable. Used after extraction, before merge.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

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


def format_published_date_from_metadata(published: Any) -> Optional[str]:
    """
    Format an arXiv (or ISO) published timestamp as YYYY-MM-DD.

    Examples:
        "2023-06-14T18:00:00+00:00" -> "2023-06-14"
        "2023-06-14" -> "2023-06-14"
    """
    if published is None or (isinstance(published, str) and not published.strip()):
        return None

    s = str(published).strip()

    # Fast path: leading YYYY-MM-DD (optionally followed by time / timezone)
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{year:04d}-{month:02d}-{day:02d}"

    try:
        normalized = s.replace("Z", "+00:00")
        if "T" in normalized:
            dt = datetime.fromisoformat(normalized)
        else:
            dt = datetime.strptime(normalized[:10], "%Y-%m-%d")
        return dt.date().isoformat()
    except (ValueError, TypeError):
        logger.debug("Could not parse published date from metadata: %r", published)
        return None


def normalize_date_created(value: Any) -> Any:
    """
    Normalize date_created to a consistent ISO-like string without dropping the day.

    - "2018-10-01" -> "2018-10-01"  (day preserved)
    - "2018-10" -> "2018-10"        (month only when day unknown)
    - "2018" -> "2018-01-01"        (year only -> first of year)
    - Invalid or empty -> return as-is
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return value
    s = str(value).strip()

    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
    if m:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{year:04d}-{month:02d}-{day:02d}"

    m = re.match(r"^(\d{4})-(\d{2})$", s)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            return f"{year:04d}-{month:02d}"

    m = re.match(r"^(\d{4})$", s)
    if m:
        return f"{int(m.group(1)):04d}-01-01"

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
