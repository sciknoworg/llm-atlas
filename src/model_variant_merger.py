"""
Model Variant Merger

Merges non-version variants of the same model into a single contribution,
following the thesis convention:
- different versions remain separate contributions (e.g. Llama 3 vs 3.1)
- size/base-large/context/stage/corpus qualifiers merge into one contribution

Stripping rules applied in _get_canonical_name:
  - Parameter sizes     : "BERT Base / Large", "GPT-2 1.5B"       → base name
  - Context windows     : "Llama 3 8K-context", "128K"            → base name
  - Training stages     : "Llama 3 (pre-trained)"                  → base name
  - Corpus qualifiers   : "XLNet-Base-wikibooks"                   → "XLNet"

Usage:
    from src.model_variant_merger import merge_model_variants

    merged = merge_model_variants(extraction_data, paper_metadata)
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def merge_model_variants(
    models: List[Dict[str, Any]], paper_metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Merge size variants of the same model version into single contributions.

    For papers that introduce one model with multiple sizes (e.g., GPT-2 with
    124M, 355M, 774M, 1.5B), this merges them into one contribution:
    - parameters: comma-separated list of all sizes
    - parameters_millions: maximum size in millions
    - Other fields: merged intelligently (prefer non-null, longest text)

    CRITICAL: Preserves version distinctions (Llama 3, 3.1, 3.2 stay separate).

    Args:
        models: List of extracted models
        paper_metadata: Optional paper metadata

    Returns:
        List of merged model contributions
    """
    if not models:
        return []

    logger.info(f"Merging model variants: {len(models)} extracted model(s)")

    # Group by canonical model name (with version preserved)
    groups = {}
    for model in models:
        canonical = _get_canonical_name(model.get("model_name", ""))
        if canonical not in groups:
            groups[canonical] = []
        groups[canonical].append(model)

    # Validate version diversity (warn if versions might be lost)
    _validate_version_preservation(models, groups)

    # Merge each group
    merged = []
    for canonical_name, group in groups.items():
        if len(group) == 1:
            model = group[0]
            # Always normalise model_name to the canonical form even when there
            # is only one variant.  This ensures that a name like "Llama 3.1 405B"
            # becomes "Llama 3.1" (with "405B" already recorded in `parameters`),
            # matching the gold-standard convention of keeping size out of the name.
            if canonical_name and model.get("model_name", "") != canonical_name:
                model = dict(model)  # shallow copy – avoid mutating caller's data
                model["model_name"] = canonical_name
            merged.append(model)
            logger.info(f"  {canonical_name}: 1 variant (name normalised)")
        else:
            merged_model = _merge_group(group, canonical_name)
            merged.append(merged_model)
            sizes = [m.get("parameters") for m in group if m.get("parameters")]
            logger.info(
                f"  {canonical_name}: merged {len(group)} variants " f"(sizes: {', '.join(sizes)})"
            )

    logger.info(f"Result: {len(merged)} unique model(s) after merging")
    return merged


def _validate_version_preservation(
    original_models: List[Dict[str, Any]], groups: Dict[str, List[Dict[str, Any]]]
) -> None:
    """
    Validate that version diversity is preserved after grouping.
    
    Warns if multiple distinct version numbers were found in extraction but
    collapsed to fewer groups (indicating potential over-merging).
    
    Args:
        original_models: Original extracted models
        groups: Grouped models by canonical name (keys are canonical names)
    """
    # Extract all unique (family, version) pairs from original model names
    version_patterns = set()
    for model in original_models:
        name = model.get("model_name", "")
        version = _extract_version_token(name)
        if version:
            family = (model.get("model_family") or "").strip().lower()
            if family:
                version_patterns.add((family, version))
    
    # If no versions detected in extraction, skip validation
    if not version_patterns:
        return
    
    # Count unique (family, version) pairs in canonical group names
    # The canonical names SHOULD contain version tokens if preservation worked
    group_versions = set()
    for canonical_name in groups.keys():
        # Extract version from canonical name
        version = _extract_version_token(canonical_name)
        if version:
            # Extract family from canonical name - should be the first word before version
            # e.g., "Llama 3.1" → family="Llama", version="3.1"
            family_match = re.match(r'^([A-Za-z][\w-]*)', canonical_name)
            if family_match:
                family = family_match.group(1).lower()
                group_versions.add((family, version))
    
    # Warn if versions were lost during grouping
    if len(version_patterns) > len(group_versions):
        lost_versions = version_patterns - group_versions
        logger.warning(
            f"Version diversity check: {len(version_patterns)} unique versions extracted, "
            f"but only {len(group_versions)} preserved in canonical names"
        )
        if lost_versions:
            logger.warning(f"  Potentially lost versions: {sorted(lost_versions)}")
        logger.warning(
            "  Check that _get_canonical_name() properly preserves version tokens."
        )


def _get_canonical_name(model_name: str) -> str:
    """
    Extract canonical model name by stripping intra-release variant suffixes.

    Policy: keep version tokens (e.g. "3", "3.1", "3.2", "3.3") so that Llama 3,
    Llama 3.1, Llama 3.2, and Llama 3.3 remain separate contributions; strip
    intra-release qualifiers (sizes, layer/depth counts, context windows, training
    stages, dataset/corpus suffixes).

    Version preservation strategy:
    - Detects version patterns like "3", "3.1", "3.2", "v2", "2.0" after model family name
    - Preserves these version tokens while stripping size/variant suffixes
    - Works generically for any model family (Llama, GPT, BERT, etc.)

    Examples:
        "GPT-2 1.5B"               → "GPT-2"        (version 2 preserved)
        "GPT-3.5 Turbo"            → "GPT-3.5"      (version 3.5 preserved)
        "BERT Base"                → "BERT"
        "T5-Large (11B)"           → "T5"
        "Llama 3 8B"               → "Llama 3"      (version 3 preserved)
        "Llama 3.1 8B"             → "Llama 3.1"    (version 3.1 preserved)
        "Llama 3.1 70B"            → "Llama 3.1"    (version 3.1 preserved)
        "Llama 3.2 11B"            → "Llama 3.2"    (version 3.2 preserved)
        "Llama 3.3 70B"            → "Llama 3.3"    (version 3.3 preserved)
        "Llama 3 8K-context"       → "Llama 3"      (context window stripped)
        "Llama 3 (pre-trained)"    → "Llama 3"
        "Transformer-XL 12L"       → "Transformer-XL"
        "XLNet-Base-wikibooks"     → "XLNet"
    """
    if not model_name:
        return ""

    name = model_name.strip()

    # CRITICAL: Extract and preserve version token BEFORE stripping other suffixes
    # This ensures "Llama 3.1 8B" → preserve "3.1" → strip "8B" → result "Llama 3.1"
    version_token = _extract_version_token(name)

    # Strip patterns are applied in order; each pass may expose the next suffix.
    # Run parenthetical-label stripping first so it doesn't block size/layer patterns.
    patterns = [
        # Parenthetical role labels: "(ablation)", "(baseline)", "(checkpoint)", etc.
        # Must come first so e.g. "Transformer-XL 151M (ablation)" exposes "151M" for the next pass.
        r"\s*\((ablation|baseline|checkpoint|variant|experiment|configuration|config|finetuned|fine-tuned|quantized|distilled|pruned|instruct|chat)\)$",  # noqa: E501
        # Parenthetical explicit sizes: "(340M)", "(1.5B)"
        r"\s*\(\d+\.?\d*[MBT]\)$",
        # Parenthetical context window: "(200K context)", "(128K)", etc.
        r"\s*\(\d+[Kk](?:\s+context(?:\s+window)?)?\)$",
        # Trailing corpus/source qualifiers: "-wikibooks", "_bookcorpus", " commoncrawl", etc.
        # Keeps contributions version-centric (e.g. XLNet on WikiBooks merges into XLNet).
        r"[-_\s]+(wikibooks?|wikipedia|bookcorpus|books?corpus|openwebtext|commoncrawl|cc-?news|c4|pile|ptb|wt103|enwik8|text8)$",  # noqa: E501
        # Trailing size words: "BERT Base", "T5-Large", "xlnet_base"
        r"[-_\s]+(Base|Large|XL|XXL|Small|Medium|Tiny|Mini)$",
        # Concatenated size words (no separator): "BERTBASE", "BERTLARGE"
        r"(?<=[a-z0-9])(Base|Large|XL|XXL|Small|Medium|Tiny|Mini)$",
        # Trailing variant qualifiers: "Turbo", "Pro", "Ultra", "Lite"
        r"[-_\s]+(Turbo|Pro|Ultra|Lite|Plus|Premium|Standard|Advanced|Basic)$",
        # Explicit trailing sizes (no parens): "1.5B", "405B", "117M", "151M", "llama-3-8B"
        # BUT preserve version-like patterns (handled by version token preservation above)
        r"[-_\s]+\d+\.?\d*[MBT]$",
        # Layer/depth count suffixes: "12L", "18L", "24L", "96L", "12-layer", "96-layer"
        r"[-_\s]+\d+[-_\s]?[Ll](?:ayer(?:s)?)?$",
        # Context-window variants: "8K", "128K", "8K-context", "128K context"
        r"[-_\s]+\d+[Kk](?:[-_\s]?context(?:\s+window)?)?$",
        # Training-stage variants: "(pre-trained)", "(post-trained)", "(pre trained)"
        r"\s*\((?:pre|post)[-\s]?trained\)$",
        # Cleanup any trailing separators left after previous stripping steps.
        r"[-_\s]+$",
    ]

    for pattern in patterns:
        name = re.sub(pattern, "", name, flags=re.IGNORECASE)

    # If we extracted a version token and it's not already in the name, restore it
    # This handles cases where stripping might have removed the version
    if version_token and version_token not in name:
        name = _restore_version_token(name, version_token)

    return name.strip()


def _extract_version_token(model_name: str) -> Optional[str]:
    """
    Extract version token from model name (e.g., "3", "3.1", "3.2", "v2", "2.0").
    
    This function identifies and extracts version-like patterns that should be
    preserved to differentiate model releases (e.g., Llama 3 vs 3.1 vs 3.2).
    
    Args:
        model_name: Model name string
        
    Returns:
        Version token string (e.g., "3.1") or None if no version found
        
    Examples:
        "Llama 3"         → "3"
        "Llama 3 8B"      → "3"
        "Llama 3.1 70B"   → "3.1"
        "GPT-3.5 Turbo"   → "3.5"
        "GPT-4"           → "4"
        "BERT Base"       → None
    """
    if not model_name:
        return None
    
    # Pattern: model_family + version_number + (space/dash/underscore/end)
    # Matches patterns like:
    # - "Llama 3" → version "3"
    # - "Llama 3 8B" → version "3"
    # - "Llama 3.1 405B" → version "3.1"
    # - "GPT-3.5 Turbo" → version "3.5"
    # - "Claude-2.1" → version "2.1"
    # - "Gemini 1.5 Pro" → version "1.5"
    
    # Look for version pattern: space/dash followed by digits with optional decimal
    # Can be followed by space, dash, end of string, or nothing
    version_patterns = [
        # Pattern 1: "Model 3.1 ..." or "Model 3" (space separator)
        r'(?:^|\s)([A-Za-z][\w-]*?)\s+([vV]?\d+(?:\.\d+)?)(?:\s|[-_]|$)',
        # Pattern 2: "Model-3.1" or "Model_3.1" (dash/underscore separator)
        r'(?:^|\s)([A-Za-z][\w-]*?)[-_]([vV]?\d+(?:\.\d+)?)(?:\s|[-_]|$)',
    ]
    
    for pattern in version_patterns:
        match = re.search(pattern, model_name)
        if match:
            version = match.group(2)
            # Strip optional 'v' or 'V' prefix
            if version.lower().startswith('v'):
                version = version[1:]
            return version
    
    return None


def _restore_version_token(base_name: str, version_token: str) -> str:
    """
    Restore version token to base model name if it was stripped.
    
    Args:
        base_name: Model name after stripping (e.g., "Llama")
        version_token: Version to restore (e.g., "3.1")
        
    Returns:
        Model name with version restored (e.g., "Llama 3.1")
    """
    if not version_token:
        return base_name
    
    # Check if version is already present (case-insensitive)
    if version_token.lower() in base_name.lower():
        return base_name
    
    # Append version with space separator
    return f"{base_name} {version_token}"


def _merge_group(models: List[Dict[str, Any]], canonical_name: str) -> Dict[str, Any]:
    """
    Merge a group of model variants into a single contribution.

    Strategy:
    - model_name: use canonical name
    - parameters: comma-separated list of all sizes (sorted)
    - parameters_millions: maximum across all variants
    - Other fields: prefer non-null; for text, prefer longest or most detailed
    """
    merged = {"model_name": canonical_name}

    # Aggregate parameters
    param_sizes = []
    param_millions_list = []

    for model in models:
        params = model.get("parameters")
        if params:
            # Normalize and collect
            normalized = _normalize_parameter_string(str(params))
            if normalized:
                param_sizes.append(normalized)

        params_m = model.get("parameters_millions")
        if params_m is not None:
            try:
                param_millions_list.append(int(params_m))
            except (ValueError, TypeError):
                pass

    # Deduplicate and sort sizes
    unique_sizes = sorted(set(param_sizes), key=_size_sort_key)
    merged["parameters"] = ", ".join(unique_sizes) if unique_sizes else None
    merged["parameters_millions"] = max(param_millions_list) if param_millions_list else None

    # Merge other fields
    fields_to_merge = [
        "model_family",
        "date_created",
        "organization",
        "innovation",
        "pretraining_architecture",
        "pretraining_task",
        "pretraining_corpus",
        "finetuning_task",
        "optimizer",
        "hardware_used",
        "extension",
        "blog_post",
        "license",
        "research_problem",
        "application",
        "paper_title",
    ]

    for field in fields_to_merge:
        merged[field] = _merge_field(models, field)

    return merged


def _normalize_parameter_string(param_str: str) -> Optional[str]:
    """
    Normalize a parameter size string (e.g., "124M" → "124M", "1.5B" → "1.5B").
    """
    param_str = param_str.strip().upper()
    if not param_str or param_str in ["", "NULL", "NONE", "N/A"]:
        return None

    # Extract numeric + unit (M, B, T)
    match = re.match(r"^(\d+\.?\d*)\s*([MBT])$", param_str)
    if match:
        num, unit = match.groups()
        return f"{num}{unit}"

    # If it's just a number, assume millions
    if param_str.replace(".", "").isdigit():
        return f"{param_str}M"

    return param_str


def _size_sort_key(size_str: str) -> float:
    """
    Return numeric value for sorting (convert M/B/T to comparable numbers).

    Examples:
        "124M" → 124
        "1.5B" → 1500
        "175B" → 175000
    """
    match = re.match(r"(\d+\.?\d*)\s*([MBT])", size_str.upper())
    if not match:
        return 0

    num, unit = match.groups()
    num = float(num)

    multipliers = {"M": 1, "B": 1000, "T": 1_000_000}
    return num * multipliers.get(unit, 1)


def _merge_field(models: List[Dict[str, Any]], field: str) -> Any:
    """
    Merge a field across multiple model variants.

    Strategy:
    - Prefer non-null values
    - For text fields (innovation, extension, etc.): prefer longest/most detailed
    - For short fields (organization, license, etc.): prefer first non-null
    - For lists (blog_post): merge and deduplicate
    """
    values = [m.get(field) for m in models]
    non_null = [v for v in values if v is not None and str(v).strip() not in ["", "null", "None"]]

    if not non_null:
        return None

    # Text fields: prefer longest
    if field in [
        "innovation",
        "extension",
        "pretraining_corpus",
        "application",
        "research_problem",
    ]:
        return max(non_null, key=lambda v: len(str(v)))

    # List/multi-value fields (e.g., blog_post): merge and deduplicate
    if field == "blog_post":
        all_links = []
        for val in non_null:
            links = str(val).split(",")
            all_links.extend(link.strip() for link in links if link.strip())
        unique_links = sorted(set(all_links))
        return ", ".join(unique_links) if unique_links else None

    # Default: first non-null
    return non_null[0]
