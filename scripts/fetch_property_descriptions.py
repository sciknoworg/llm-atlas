#!/usr/bin/env python3
"""
Refresh the local ORKG property-description cache.

The semantic validator judges each extracted value against the *meaning* of the
property it was extracted for, and ORKG already stores that meaning: every
predicate carries a ``description`` field, returned by GET /api/predicates/{id}.

This script pulls those descriptions once and writes them to
``data/property_descriptions.json`` so the pipeline never has to call ORKG
during extraction. Rationale for caching rather than fetching at runtime:

  * 47 HTTP calls per pipeline run is pure waste — these descriptions change
    perhaps once a year, when someone edits the template.
  * Extraction would otherwise fail (or silently degrade) whenever ORKG is down.
  * A checked-in cache makes runs reproducible and reviewable in git: if a
    description changes, the diff shows it.

Run it whenever predicates are added or their descriptions edited:

    python scripts/fetch_property_descriptions.py

Descriptions are keyed by *field name* (``rl_algorithm``), not predicate ID,
because sandbox and production use different IDs for the same concept
(P203004 vs P203093) but the meaning is identical. Production is queried first
and sandbox is used as a fallback, which matters more than it sounds: a few
predicates have a usable description on one instance and junk on the other.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.orkg_client import ORKG_HOST_URLS  # noqa: E402
from src.template_mapper import (  # noqa: E402
    _PRODUCTION_FIELD_MAPPING,
    _SANDBOX_FIELD_MAPPING,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "data" / "property_descriptions.json"

# Hosts to query, in priority order. The first host that yields a *usable*
# description wins.
_HOST_ORDER = ("production", "sandbox")

_MAPPING_BY_HOST = {
    "production": _PRODUCTION_FIELD_MAPPING,
    "sandbox": _SANDBOX_FIELD_MAPPING,
}


def is_usable(description: Optional[str]) -> bool:
    """
    Decide whether a description carries meaning a validator can act on.

    Some ORKG predicates store a bare URL instead of prose — P32
    ("research problem") on production is literally
    "https://www.wikidata.org/wiki/Q123578571". That identifies the concept for
    a human following the link, but tells a language model nothing, so we treat
    it as absent and fall back to the next host (or to a local override).
    """
    if not description:
        return False
    text = description.strip()
    if len(text) < 15:
        return False
    if text.startswith(("http://", "https://")) and " " not in text:
        return False
    return True


def fetch_predicate(host: str, predicate_id: str, timeout: int = 20) -> Optional[Dict]:
    """Fetch one predicate's JSON, or None if it doesn't exist / the call fails."""
    url = f"{ORKG_HOST_URLS[host]}/api/predicates/{predicate_id}"
    try:
        response = requests.get(url, headers={"Accept": "application/json"}, timeout=timeout)
    except requests.RequestException as exc:
        logger.warning("  %s/%s: request failed (%s)", host, predicate_id, exc)
        return None
    if response.status_code != 200:
        logger.warning("  %s/%s: HTTP %s", host, predicate_id, response.status_code)
        return None
    try:
        return response.json()
    except ValueError:
        logger.warning("  %s/%s: response was not JSON", host, predicate_id)
        return None


def resolve_field(field: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Find the best available description for one field.

    Returns (description, source_host, predicate_id). All three are None when no
    host had a usable description — the caller reports those so a human can add
    a local override or fix the predicate in ORKG.
    """
    for host in _HOST_ORDER:
        predicate_id = _MAPPING_BY_HOST[host].get(field)
        if not predicate_id:
            continue
        payload = fetch_predicate(host, predicate_id)
        if payload is None:
            continue
        description = (payload.get("description") or "").strip()
        if is_usable(description):
            return description, host, predicate_id
    return None, None, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Where to write the cache JSON"
    )
    args = parser.parse_args()

    # Union of both mappings: a field present on only one instance still gets a
    # description, which is what we want while the two instances drift.
    fields = sorted(set(_PRODUCTION_FIELD_MAPPING) | set(_SANDBOX_FIELD_MAPPING))
    logger.info("Fetching descriptions for %d fields...", len(fields))

    descriptions: Dict[str, Dict[str, str]] = {}
    unusable = []

    for field in fields:
        description, host, predicate_id = resolve_field(field)
        if description is None:
            unusable.append(field)
            continue
        descriptions[field] = {
            "description": description,
            "source_host": host,
            "predicate_id": predicate_id,
        }

    payload = {
        "_comment": (
            "Generated by scripts/fetch_property_descriptions.py — do not edit by hand. "
            "To correct a description, either fix it in ORKG and re-run this script, or "
            "add a local override in src/property_descriptions.py."
        ),
        "_generated_at": datetime.now(timezone.utc).isoformat(),
        "descriptions": descriptions,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    logger.info("Wrote %d descriptions to %s", len(descriptions), args.output)
    if unusable:
        logger.warning(
            "%d field(s) had no usable description on any host — these need a local "
            "override in src/property_descriptions.py: %s",
            len(unusable),
            ", ".join(unusable),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
