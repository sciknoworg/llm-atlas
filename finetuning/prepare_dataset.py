"""
Prepare training data for ORKG fine-tuning.

Converts the gold-standard annotations and parsed paper PDFs into
instruction / input / output JSONL files ready for SFTTrainer.

Pipeline
--------
1. Load gold-standard models from ``R1364660.json``.
2. Group gold entries by ``paper_title``.
3. For each paper, locate the PDF in ``data/papers/`` and parse it.
4. Chunk the parsed text using the same strategy as the baseline
   (``PDFParser.chunk_text``).
5. For each (chunk, gold_model) pair, decide which ORKG fields the
   chunk actually supports (simple keyword heuristic) and build the
   target JSON with empty strings for unsupported fields.
6. Emit one JSONL record per instance:
   ``{"instruction": ..., "input": ..., "output": ...}``
7. Stratified split into train / val / test.

Usage
-----
    python -m finetuning.prepare_dataset
    python -m finetuning.prepare_dataset --min-fields 3 --dry-run
"""

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from finetuning.config import (
    CHUNK_OVERLAP,
    DATASET_DIR,
    DATASET_TEST,
    DATASET_TRAIN,
    DATASET_VAL,
    GOLD_STANDARD_PATH,
    INSTRUCTION,
    MAX_CHUNK_SIZE,
    ORKG_FIELDS,
    PAPERS_DIR,
    PAPERS_LIST_PATH,
    TrainingConfig,
)
from src.pdf_parser import PDFParser

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s"
)
logger = logging.getLogger(__name__)

# ── helpers ──────────────────────────────────────────────────────────────────


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace."""
    return " ".join(text.lower().split())


def _field_in_chunk(field_name: str, value: Any, chunk_lower: str) -> bool:
    """Return True when the gold *value* is plausibly mentioned in *chunk_lower*.

    For numeric fields we look for the number itself.  For short
    identifiers we require an exact (case-insensitive) substring match.
    For longer text (innovation, extension, …) we check whether at
    least 40 % of the significant words appear.
    """
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return False

    val_str = str(value).strip()
    if not val_str:
        return False

    val_lower = val_str.lower()

    # Numeric / short identifiers → substring match
    if field_name in (
        "parameters_millions", "parameters", "date_created", "model_name",
        "model_family", "organization", "optimizer", "hardware_used",
        "license", "pretraining_architecture", "blog_post",
    ):
        # For parameters like "110M, 340M" check at least one sub-value
        for part in val_lower.replace(",", " ").split():
            part = part.strip()
            if len(part) >= 2 and part in chunk_lower:
                return True
        return val_lower in chunk_lower

    # Long-text fields → word-overlap heuristic
    stop = {"the", "a", "an", "of", "and", "in", "for", "to", "is", "are",
            "was", "were", "with", "on", "by", "that", "this", "it", "as"}
    words = [w for w in val_lower.split() if len(w) > 2 and w not in stop]
    if not words:
        return False
    hits = sum(1 for w in words if w in chunk_lower)
    return hits / len(words) >= 0.40


def _build_target_json(
    gold_model: Dict[str, Any],
    chunk_text: str,
) -> Tuple[Dict[str, Any], int]:
    """Build the target JSON for one (chunk, model) pair.

    Returns (target_dict, n_fields_with_values).
    """
    chunk_lower = _normalize(chunk_text)
    target: Dict[str, Any] = {}
    n_filled = 0
    for field in ORKG_FIELDS:
        gold_val = gold_model.get(field)
        if _field_in_chunk(field, gold_val, chunk_lower):
            target[field] = gold_val if gold_val is not None else ""
            n_filled += 1
        else:
            target[field] = ""
    return target, n_filled


# ── main pipeline ────────────────────────────────────────────────────────────


def _find_pdf(paper_title: str, arxiv_id: Optional[str]) -> Optional[Path]:
    """Try to locate a downloaded PDF in data/papers/."""
    if arxiv_id:
        clean_id = arxiv_id.replace("/", "_").split("v")[0]
        for pat in (f"{clean_id}*.pdf", f"{arxiv_id.replace('/', '_')}*.pdf"):
            hits = list(PAPERS_DIR.glob(pat))
            if hits:
                return hits[0]
    # Fallback: fuzzy name match
    slug = "".join(c if c.isalnum() else "_" for c in paper_title.lower())[:60]
    for pdf in PAPERS_DIR.glob("*.pdf"):
        if slug[:30] in pdf.stem.lower().replace("-", "_"):
            return pdf
    return None


def build_instances(min_fields: int = 2) -> List[Dict[str, str]]:
    """Return a list of {instruction, input, output} dicts."""

    # 1. Load gold standard
    gold_raw = _load_json(GOLD_STANDARD_PATH)
    gold_models: List[Dict[str, Any]] = gold_raw.get("extraction_data", gold_raw)
    logger.info("Gold standard: %d model entries", len(gold_models))

    # 2. Load papers list for arxiv_id lookup
    papers_list: List[Dict[str, Any]] = []
    if PAPERS_LIST_PATH.exists():
        papers_list = _load_json(PAPERS_LIST_PATH)
    title_to_arxiv = {
        p["paper_title"]: p.get("arxiv_id", "").split("v")[0]
        for p in papers_list if p.get("arxiv_id")
    }

    # 3. Group gold entries by paper_title
    by_paper: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for m in gold_models:
        title = m.get("paper_title", "unknown")
        by_paper[title].append(m)
    logger.info("Papers with gold annotations: %d", len(by_paper))

    # 4. Parse each paper, chunk, and create instances
    parser = PDFParser(method="pdfplumber", extract_tables=True)
    instances: List[Dict[str, str]] = []
    papers_used = 0
    papers_skipped = 0

    for paper_title, models in by_paper.items():
        arxiv_id = title_to_arxiv.get(paper_title)
        pdf_path = _find_pdf(paper_title, arxiv_id)
        if pdf_path is None or not pdf_path.exists():
            logger.debug("PDF not found for: %s", paper_title[:60])
            papers_skipped += 1
            continue

        parsed = parser.parse(pdf_path)
        if parsed is None or not parsed.get("cleaned_text"):
            logger.warning("Parsing failed for: %s", paper_title[:60])
            papers_skipped += 1
            continue

        text = parsed["cleaned_text"]
        chunks = parser.chunk_text(text, max_chunk_size=MAX_CHUNK_SIZE, overlap=CHUNK_OVERLAP)

        for gold_model in models:
            for chunk in chunks:
                target, n_filled = _build_target_json(gold_model, chunk)
                if n_filled < min_fields:
                    continue
                instances.append({
                    "instruction": INSTRUCTION,
                    "input": chunk,
                    "output": json.dumps(target, ensure_ascii=False),
                    "_paper_title": paper_title,
                    "_model_name": gold_model.get("model_name", ""),
                })
        papers_used += 1

    logger.info(
        "Built %d instances from %d papers (%d skipped, no PDF found)",
        len(instances), papers_used, papers_skipped,
    )
    return instances


def split_and_save(
    instances: List[Dict[str, str]],
    cfg: TrainingConfig,
) -> Dict[str, int]:
    """Stratified split by paper and save JSONL files.

    Returns dict with counts per split.
    """
    random.seed(cfg.seed)

    # Group by paper to avoid data leakage
    by_paper: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for inst in instances:
        by_paper[inst["_paper_title"]].append(inst)

    paper_titles = sorted(by_paper.keys())
    random.shuffle(paper_titles)

    n = len(paper_titles)
    n_test = max(1, int(n * cfg.test_ratio))
    n_val = max(1, int(n * cfg.val_ratio))

    test_papers = set(paper_titles[:n_test])
    val_papers = set(paper_titles[n_test : n_test + n_val])
    train_papers = set(paper_titles[n_test + n_val :])

    splits = {"train": [], "val": [], "test": []}
    for title, insts in by_paper.items():
        if title in test_papers:
            splits["test"].extend(insts)
        elif title in val_papers:
            splits["val"].extend(insts)
        else:
            splits["train"].extend(insts)

    # Shuffle train
    random.shuffle(splits["train"])

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    paths = {"train": DATASET_TRAIN, "val": DATASET_VAL, "test": DATASET_TEST}

    for split_name, records in splits.items():
        with open(paths[split_name], "w", encoding="utf-8") as fp:
            for rec in records:
                # Strip internal metadata before saving
                clean = {k: v for k, v in rec.items() if not k.startswith("_")}
                fp.write(json.dumps(clean, ensure_ascii=False) + "\n")
        logger.info("  %-5s  %4d instances  → %s", split_name, len(records), paths[split_name])

    counts = {k: len(v) for k, v in splits.items()}
    logger.info(
        "Split totals — train: %d, val: %d, test: %d (by paper, no leakage)",
        counts["train"], counts["val"], counts["test"],
    )
    return counts


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ORKG fine-tuning dataset from gold standard + parsed PDFs."
    )
    parser.add_argument(
        "--min-fields", type=int, default=2,
        help="Minimum non-empty ORKG fields for a chunk to become a training instance (default: 2)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build instances but do not write files; print statistics only.",
    )
    args = parser.parse_args()

    cfg = TrainingConfig()
    instances = build_instances(min_fields=args.min_fields)

    if not instances:
        logger.error("No instances built. Check that PDFs exist in %s", PAPERS_DIR)
        return 1

    if args.dry_run:
        print(f"\nDry-run: {len(instances)} instances built.")
        print("Example instance (first):")
        print(json.dumps(instances[0], indent=2, ensure_ascii=False)[:2000])
        return 0

    split_and_save(instances, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
