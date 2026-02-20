"""
Normalize gold-standard 'parameters' field to GPT-2 style: "124M, 355M, 774M, 1.5B".

Converts any parameter string (e.g. "Base=117M, Large=360M") to comma-separated
ascending list with M/B suffixes. Sets parameters to null when the value is not
a parameter count (e.g. "Same as BART", "Encoder"). Updates parameters_millions
from the normalized string for consistency.

Usage:
    python scripts/evaluation/normalize_gold_standard_parameters.py

Reads/writes: data/gold_standard/R1364660.json
"""

import json
import re
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_parameters_gpt2_style(value: Optional[str]) -> Optional[str]:
    """
    Normalize parameter string to GPT-2 paper style: "124M, 355M, 774M, 1.5B".
    Extracts all Nm, Nb, Nt (case-insensitive); returns comma-separated ascending list.
    Returns None if no parameter counts found (e.g. "Same as BART", "Encoder").
    """
    if value is None or not isinstance(value, str):
        return None
    s = value.strip()
    if not s or s.lower() in ("n/a", "null", "none"):
        return None

    pattern = re.compile(r"(\d+(?:\.\d+)?)\s*([MBT])\b", re.IGNORECASE)
    matches = pattern.findall(s)

    if not matches:
        return None

    millions_list = []
    for num_str, unit in matches:
        num = float(num_str)
        u = unit.upper()
        if u == "M":
            millions_list.append(num)
        elif u == "B":
            millions_list.append(num * 1000)
        else:  # T
            millions_list.append(num * 1_000_000)

    millions_list = sorted(set(millions_list))

    def fmt(m: float) -> str:
        if m >= 1000:
            b = m / 1000
            return f"{int(b)}B" if b == int(b) else f"{b}B"
        return f"{int(round(m))}M"

    return ", ".join(fmt(m) for m in millions_list)


def parameters_millions_from_normalized(normalized: Optional[str]) -> Optional[int]:
    """Max parameter count in millions from normalized string like '124M, 1.5B'."""
    if not normalized:
        return None
    pattern = re.compile(r"(\d+(?:\.\d+)?)\s*([MBT])\b", re.IGNORECASE)
    millions_list = []
    for num_str, unit in pattern.findall(normalized):
        num = float(num_str)
        u = unit.upper()
        if u == "M":
            millions_list.append(num)
        elif u == "B":
            millions_list.append(num * 1000)
        else:
            millions_list.append(num * 1_000_000)
    return int(round(max(millions_list))) if millions_list else None


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    json_path = project_root / "data" / "gold_standard" / "R1364660.json"

    if not json_path.exists():
        logger.error("Gold standard not found: %s", json_path)
        return

    logger.info("Loading gold standard from %s", json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    extraction_data = data.get("extraction_data", [])
    if not extraction_data:
        logger.error("No extraction_data in gold standard")
        return

    changed = 0
    for i, model in enumerate(extraction_data):
        raw = model.get("parameters")
        normalized = normalize_parameters_gpt2_style(raw)
        if normalized != raw:
            changed += 1
            if raw is not None and normalized is not None:
                logger.debug("Model %s: %r -> %r", model.get("model_name"), raw, normalized)
            elif raw is not None and normalized is None:
                logger.debug("Model %s: %r -> null (not a parameter count)", model.get("model_name"), raw)

        model["parameters"] = normalized
        if normalized is not None:
            model["parameters_millions"] = parameters_millions_from_normalized(normalized)
        else:
            model["parameters_millions"] = None

    logger.info("Normalized %d of %d models", changed, len(extraction_data))

    logger.info("Writing gold standard to %s", json_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("Done. Parameters are now in GPT-2 style (e.g. 124M, 355M, 774M, 1.5B).")


if __name__ == "__main__":
    main()
