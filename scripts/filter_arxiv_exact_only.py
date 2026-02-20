"""
Keep only papers with source "arxiv_exact" in papers_list and in the gold standard.

- papers_list.json: remove entries where source != "arxiv_exact"
- R1364660.json: remove extraction_data entries whose paper is not in the
  kept papers_list (match by paper_title: exact or starts with "paper_title - ").

Usage:
    python scripts/filter_arxiv_exact_only.py [--dry-run]
"""

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPERS_LIST_PATH = PROJECT_ROOT / "data" / "gold_standard" / "papers_list.json"
GOLD_PATH = PROJECT_ROOT / "data" / "gold_standard" / "R1364660.json"


def main():
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        logger.info("DRY RUN - no files will be modified")

    # Load papers list
    with open(PAPERS_LIST_PATH, "r", encoding="utf-8") as f:
        papers_list = json.load(f)

    kept_papers = [p for p in papers_list if p.get("source") == "arxiv_exact"]
    removed_papers = [p for p in papers_list if p.get("source") != "arxiv_exact"]
    kept_titles = {p.get("paper_title", "").strip() for p in kept_papers if p.get("paper_title")}

    logger.info(
        "papers_list: total=%d, keeping arxiv_exact=%d, removing=%d",
        len(papers_list),
        len(kept_papers),
        len(removed_papers),
    )
    if removed_papers:
        for p in removed_papers[:15]:
            logger.info("  removing: %s (source=%s)", (p.get("paper_title") or "")[:60], p.get("source"))
        if len(removed_papers) > 15:
            logger.info("  ... and %d more", len(removed_papers) - 15)

    # Load gold standard
    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        gold = json.load(f)

    extraction_data = gold.get("extraction_data", [])
    if not extraction_data:
        logger.warning("Gold standard has no extraction_data")
        return 1

    def keep_gold_entry(entry):
        title = (entry.get("paper_title") or "").strip()
        if not title:
            return False
        if title in kept_titles:
            return True
        for pt in kept_titles:
            if title.startswith(pt + " - "):
                return True
        return False

    kept_gold = [e for e in extraction_data if keep_gold_entry(e)]
    removed_gold = [e for e in extraction_data if not keep_gold_entry(e)]
    removed_gold_titles = sorted(set(e.get("paper_title", "") for e in removed_gold))

    logger.info(
        "gold extraction_data: total=%d, keeping=%d, removing=%d",
        len(extraction_data),
        len(kept_gold),
        len(removed_gold),
    )
    for t in removed_gold_titles[:20]:
        logger.info("  removing gold: %s", (t or "")[:70])
    if len(removed_gold_titles) > 20:
        logger.info("  ... and %d more distinct titles", len(removed_gold_titles) - 20)

    if not dry_run:
        # Write filtered papers list
        with open(PAPERS_LIST_PATH, "w", encoding="utf-8") as f:
            json.dump(kept_papers, f, indent=2, ensure_ascii=False)
        logger.info("Wrote %s (%d entries)", PAPERS_LIST_PATH, len(kept_papers))

        # Write filtered gold (preserve top-level keys, update total_models)
        gold["extraction_data"] = kept_gold
        gold["total_models"] = len(kept_gold)
        with open(GOLD_PATH, "w", encoding="utf-8") as f:
            json.dump(gold, f, indent=2, ensure_ascii=False)
        logger.info("Wrote %s (%d models)", GOLD_PATH, len(kept_gold))
    else:
        logger.info("Dry run: would write papers_list=%d, gold extraction_data=%d", len(kept_papers), len(kept_gold))

    return 0


if __name__ == "__main__":
    sys.exit(main())
