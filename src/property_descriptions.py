"""
ORKG property descriptions.

The semantic validator needs to know what each property *means* in order to
judge whether an extracted value actually instantiates it. ORKG already stores
that meaning on every predicate, so this module serves those descriptions from
a checked-in cache (``data/property_descriptions.json``, produced by
``scripts/fetch_property_descriptions.py``), with a small set of local
overrides layered on top.

Lookups are by field name (``rl_algorithm``), never by predicate ID, because
sandbox and production assign different IDs to the same concept.

Precedence, highest first:
    1. _LOCAL_OVERRIDES  — hand-written, for properties whose ORKG description
                           is missing or useless
    2. the generated cache
    3. None              — caller skips semantic validation for that field
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "property_descriptions.json"

# Hand-written descriptions that take precedence over whatever ORKG returns.
#
# Keep this list SHORT and justify every entry. The whole point of reading
# descriptions from ORKG is that the knowledge graph stays the source of truth;
# every override here is a place where our pipeline's behaviour silently
# diverges from what a human reads on the ORKG website. Prefer fixing the
# description in ORKG and re-running the fetch script.
_LOCAL_OVERRIDES: Dict[str, str] = {
    # ORKG P32's description is a bare Wikidata URL on production and the very
    # terse "knowledge gap addressable by research" on sandbox. Neither tells a
    # model what a *good* value looks like for this template, and this is one of
    # the fields where mis-splitting actually happens, so it earns a precise one.
    "research_problem": (
        "The research problem, challenge, or capability gap that the model addresses "
        "(e.g. 'Language Understanding', 'Open agentic intelligence', "
        "'Long-horizon task execution'). A short noun phrase naming the problem, "
        "not a description of the solution or of the model's architecture."
    ),
    # 'license' has no entry in either field mapping (see the note in
    # template_mapper), so the fetch script cannot resolve a predicate for it and
    # the cache has no description. It still appears in extraction output and in
    # _RESOURCE_FIELDS, so give it one here.
    "license": (
        "The license under which the model weights or code are released "
        "(e.g. 'Apache 2.0', 'MIT', 'Llama 3.1 Community License', 'open source', "
        "'closed source'). A license name, not a description of availability."
    ),
}

# Value-shape guidance, appended to the ORKG description when building the
# validator prompt.
#
# ORKG descriptions answer "what does this property mean?" for a human browsing
# the graph. The validator needs a different question answered: "what does one
# valid VALUE look like?" supported_language is the clearest example — its ORKG
# description is a good paragraph about linguistic diversity and global utility,
# and says nothing about the fact that a value must be the name of ONE language.
# GLM-5 duly produced "Chinese, English, Multiple languages".
#
# These are kept separate from _LOCAL_OVERRIDES so the ORKG text stays intact:
# we add to it rather than replace it, and nothing here changes what ORKG shows.
#
# Add an entry only where the shape is genuinely non-obvious from the
# description. This is not the place to restate the description.
_VALUE_GUIDANCE: Dict[str, str] = {
    "supported_language": (
        "Each value must name ONE specific natural language, e.g. 'English', "
        "'German', 'Hindi'. Reject values that count or characterise languages "
        "instead of naming them: 'multiple languages', 'multilingual', "
        "'119 languages', 'various languages', 'cross-lingual'."
    ),
    "application": (
        "Each value must name a concrete task or domain the model is used for, "
        "e.g. 'code generation', 'question answering', 'document understanding'. "
        "Reject placeholders that name no particular use: 'general NLP tasks', "
        "'general knowledge', 'various applications'."
    ),
    "pretraining_corpus": (
        "Each value must name an identifiable dataset or corpus, e.g. "
        "'Common Crawl', 'BookCorpus', 'The Pile', or a described collection "
        "with a stated source. Reject values that only assert breadth: "
        "'diverse corpus of text', 'large-scale web data', 'varied datasets'."
    ),
    "rl_algorithm": (
        "Each value must name a specific algorithm, e.g. 'PPO', 'GRPO', 'DPO', "
        "'REINFORCE'. Reject categories rather than algorithms: 'General RL', "
        "'reinforcement learning', 'policy optimisation'."
    ),
    "license": (
        "Each value must name a license, e.g. 'Apache 2.0', 'MIT', "
        "'Llama 3.1 Community License'. 'open source' and 'closed source' are "
        "acceptable when no license is named. Reject 'custom' on its own — it "
        "identifies nothing."
    ),
    "optimizer": (
        "Each value must name an optimizer, e.g. 'Adam', 'AdamW', 'Muon', "
        "'Adafactor'. Reject descriptions of a training setup rather than the "
        "optimizer itself."
    ),
}

_cache: Optional[Dict[str, str]] = None


def _load_cache() -> Dict[str, str]:
    """
    Read the generated cache from disk, once, and flatten it to field -> text.

    A missing or malformed cache is not fatal: the validator simply has fewer
    descriptions to work with and skips the fields it cannot describe. Failing
    the whole extraction run because a cache file is absent would be a much
    worse outcome than validating a subset of fields.
    """
    global _cache
    if _cache is not None:
        return _cache

    _cache = {}
    if not _CACHE_PATH.exists():
        logger.warning(
            "Property description cache not found at %s — semantic validation will "
            "fall back to local overrides only. Run "
            "scripts/fetch_property_descriptions.py to generate it.",
            _CACHE_PATH,
        )
        return _cache

    try:
        payload = json.loads(_CACHE_PATH.read_text())
        for field, entry in (payload.get("descriptions") or {}).items():
            description = (entry or {}).get("description")
            if description:
                _cache[field] = description.strip()
        logger.debug("Loaded %d property descriptions from %s", len(_cache), _CACHE_PATH)
    except (OSError, ValueError) as exc:
        logger.warning("Could not read property description cache %s: %s", _CACHE_PATH, exc)

    return _cache


def get_description(field_name: str) -> Optional[str]:
    """
    Return the description for a field, or None when we have no usable one.

    Returning None is meaningful: the caller must skip semantic validation for
    that field rather than invent a description, because a made-up description
    would make the validator confidently wrong.
    """
    if field_name in _LOCAL_OVERRIDES:
        return _LOCAL_OVERRIDES[field_name]
    return _load_cache().get(field_name)


def get_value_guidance(field_name: str) -> Optional[str]:
    """Value-shape hint for a field, or None when the description suffices."""
    return _VALUE_GUIDANCE.get(field_name)


def get_validation_text(field_name: str) -> Optional[str]:
    """
    The full definition handed to the validator: the ORKG description, plus the
    value-shape guidance where we have it.

    Returns None when there is no description at all, which tells the caller to
    skip the field rather than judge it against nothing.
    """
    description = get_description(field_name)
    if not description:
        return None
    guidance = get_value_guidance(field_name)
    return f"{description} {guidance}" if guidance else description


def describable_fields(fields) -> Dict[str, str]:
    """Map each field that has a description to that description, skipping the rest."""
    resolved = {}
    for field in fields:
        description = get_description(field)
        if description:
            resolved[field] = description
    return resolved


def reset_cache() -> None:
    """Drop the in-memory cache. Used by tests that patch the cache file."""
    global _cache
    _cache = None
