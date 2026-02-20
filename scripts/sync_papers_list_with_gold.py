"""
Sync papers_list.json with the cleaned gold standard.

Keeps only papers that appear in R1364660.json extraction_data, preserving
existing pdf_url, arxiv_id, doi, source for each paper. Removes invalid
entries (e.g. continuation-line titles).

Usage:
    python scripts/sync_papers_list_with_gold.py
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLD_PATH = PROJECT_ROOT / "data" / "gold_standard" / "R1364660.json"
PAPERS_LIST_PATH = PROJECT_ROOT / "data" / "gold_standard" / "papers_list.json"


def main():
    # Load gold standard and get valid unique paper titles
    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        gold = json.load(f)
    extraction_data = gold.get("extraction_data", [])
    valid_titles = set()
    for entry in extraction_data:
        t = entry.get("paper_title")
        if t and t.strip():
            valid_titles.add(t.strip())

    logger.info(f"Gold standard has {len(valid_titles)} unique paper titles")

    # Load current papers list
    with open(PAPERS_LIST_PATH, "r", encoding="utf-8") as f:
        papers_list = json.load(f)

    # Build lookup by title from current list (keep first occurrence)
    by_title = {}
    for p in papers_list:
        title = (p.get("paper_title") or "").strip()
        if title and title not in by_title:
            by_title[title] = p

    # Build new list: only valid titles, in sorted order, preserving existing data
    kept = []
    removed = []
    for title in sorted(valid_titles):
        if title in by_title:
            kept.append(by_title[title])
        else:
            kept.append({
                "paper_title": title,
                "pdf_url": None,
                "arxiv_id": None,
                "doi": None,
                "source": "manual_needed"
            })

    for p in papers_list:
        title = (p.get("paper_title") or "").strip()
        if title and title not in valid_titles:
            removed.append(title)

    # Write updated papers list
    with open(PAPERS_LIST_PATH, "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)

    logger.info(f"Papers list: {len(papers_list)} -> {len(kept)} entries")
    if removed:
        logger.info(f"Removed {len(removed)} invalid entries:")
        for t in removed:
            logger.info(f"  - {t[:70]}{'...' if len(t) > 70 else ''}")
    logger.info(f"Saved to {PAPERS_LIST_PATH}")


if __name__ == "__main__":
    main()
