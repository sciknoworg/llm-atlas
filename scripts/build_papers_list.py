"""
Build papers list with PDF URLs/ArXiv IDs from ORKG and ArXiv.

This script:
1. Reads the gold standard (R1364660.json) to get unique papers
2. Fetches paper URLs from ORKG comparison R1364660
3. For missing papers, searches ArXiv by title
4. Outputs a papers list (paper_title, pdf_url, arxiv_id, source)

The output can then be added as a "papers" section to the gold standard.

Usage:
    python scripts/build_papers_list.py --output data/gold_standard/papers_list.json

Requirements:
    - ORKG_EMAIL, ORKG_PASSWORD, ORKG_HOST in .env
    - pip install orkg arxiv
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

# Setup project root and path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_gold_standard(path: Path) -> List[Dict[str, Any]]:
    """Load gold standard and return extraction_data."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("extraction_data", [])


def get_unique_papers(extraction_data: List[Dict[str, Any]]) -> List[str]:
    """Get unique paper titles from gold standard."""
    titles = set()
    for entry in extraction_data:
        title = entry.get("paper_title")
        if title and title.strip():
            titles.add(title.strip())
    return sorted(list(titles))


def fetch_papers_from_orkg(comparison_id: str = "R1364660") -> Dict[str, Dict[str, str]]:
    """
    Fetch paper URLs from ORKG comparison.
    
    Returns dict: {paper_title: {"pdf_url": ..., "arxiv_id": ..., "doi": ...}}
    """
    logger.info(f"Fetching papers from ORKG comparison {comparison_id}...")
    
    try:
        from src.orkg_client import ORKGClient
        
        orkg_client = ORKGClient(
            host=os.getenv("ORKG_HOST", "sandbox"),
            email=os.getenv("ORKG_EMAIL"),
            password=os.getenv("ORKG_PASSWORD")
        )
        
        # Get comparison
        comparison = orkg_client.get_comparison(comparison_id)
        if not comparison:
            logger.warning("Could not fetch comparison from ORKG")
            return {}
        
        # Get contributions
        contributions = orkg_client.get_comparison_contributions(comparison_id)
        if not contributions:
            logger.warning("No contributions found in comparison")
            return {}
        
        logger.info(f"Found {len(contributions)} contributions")
        
        papers_info = {}
        
        # Try to get paper info from each contribution
        for i, contrib in enumerate(contributions):
            # Contributions might have different structures
            # Try to find paper_id or paper reference
            paper_id = None
            
            # Common fields where paper might be referenced
            if isinstance(contrib, dict):
                paper_id = (
                    contrib.get("paper_id") or
                    contrib.get("paper") or
                    contrib.get("paperId")
                )
            
            if paper_id:
                try:
                    paper = orkg_client.get_paper(paper_id)
                    if paper and isinstance(paper, dict):
                        title = paper.get("title") or paper.get("label")
                        if title:
                            papers_info[title] = {
                                "pdf_url": paper.get("url") or paper.get("pdf_url"),
                                "doi": paper.get("doi"),
                                "arxiv_id": None  # Extract from URL if present
                            }
                            
                            # Try to extract arxiv_id from URL or identifiers
                            identifiers = paper.get("identifiers", {})
                            if isinstance(identifiers, dict) and "arxiv" in identifiers:
                                papers_info[title]["arxiv_id"] = identifiers["arxiv"]
                except Exception as e:
                    logger.debug(f"Error fetching paper {paper_id}: {e}")
        
        logger.info(f"Retrieved {len(papers_info)} papers from ORKG")
        return papers_info
        
    except ImportError:
        logger.warning("ORKG client not available (orkg package not installed)")
        return {}
    except Exception as e:
        logger.error(f"Error fetching from ORKG: {e}")
        return {}


def search_arxiv_by_title(title: str) -> Optional[Dict[str, str]]:
    """
    Search ArXiv by paper title and return best match.
    
    Returns: {"arxiv_id": ..., "pdf_url": ..., "title": ...} or None
    """
    try:
        import arxiv
        
        # Clean title for search
        search_title = title.strip()
        
        # Search ArXiv
        search = arxiv.Search(
            query=f'ti:"{search_title}"',
            max_results=3,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = list(search.results())
        if not results:
            # Try without exact title match (just keywords)
            search = arxiv.Search(
                query=search_title,
                max_results=1,
                sort_by=arxiv.SortCriterion.Relevance
            )
            results = list(search.results())
        
        if results:
            paper = results[0]
            arxiv_id = paper.get_short_id()
            
            return {
                "arxiv_id": arxiv_id,
                "pdf_url": paper.pdf_url,
                "title": paper.title,
                "match_score": "exact" if paper.title.lower() == search_title.lower() else "fuzzy"
            }
        
        return None
        
    except ImportError:
        logger.warning("arxiv package not installed")
        return None
    except Exception as e:
        logger.debug(f"Error searching ArXiv for '{title}': {e}")
        return None


def build_papers_list(gold_path: Path, use_orkg: bool = True, use_arxiv: bool = True) -> List[Dict[str, str]]:
    """
    Build complete papers list with URLs.
    
    Returns list of dicts: [{"paper_title": ..., "pdf_url": ..., "arxiv_id": ..., "source": ...}]
    """
    # Load gold standard
    logger.info(f"Loading gold standard from {gold_path}")
    extraction_data = load_gold_standard(gold_path)
    
    # Get unique papers
    unique_papers = get_unique_papers(extraction_data)
    logger.info(f"Found {len(unique_papers)} unique papers in gold standard")
    
    # Fetch from ORKG
    orkg_papers = {}
    if use_orkg:
        orkg_papers = fetch_papers_from_orkg()
    
    # Build final list
    papers_list = []
    
    for title in unique_papers:
        paper_info = {
            "paper_title": title,
            "pdf_url": None,
            "arxiv_id": None,
            "doi": None,
            "source": None
        }
        
        # Check ORKG first
        if title in orkg_papers:
            orkg_info = orkg_papers[title]
            paper_info["pdf_url"] = orkg_info.get("pdf_url")
            paper_info["arxiv_id"] = orkg_info.get("arxiv_id")
            paper_info["doi"] = orkg_info.get("doi")
            paper_info["source"] = "orkg"
            logger.info(f"✓ ORKG: {title[:60]}")
        
        # If still missing, try ArXiv
        if use_arxiv and not paper_info["arxiv_id"]:
            logger.info(f"  Searching ArXiv: {title[:60]}...")
            arxiv_result = search_arxiv_by_title(title)
            if arxiv_result:
                paper_info["arxiv_id"] = arxiv_result["arxiv_id"]
                paper_info["pdf_url"] = arxiv_result["pdf_url"]
                paper_info["source"] = f"arxiv_{arxiv_result['match_score']}"
                logger.info(f"  ✓ ArXiv: {arxiv_result['arxiv_id']} ({arxiv_result['match_score']} match)")
            else:
                paper_info["source"] = "manual_needed"
                logger.warning(f"  ✗ Not found: {title[:60]}")
        
        papers_list.append(paper_info)
    
    return papers_list


def main():
    parser = argparse.ArgumentParser(description="Build papers list with URLs from ORKG and ArXiv")
    parser.add_argument(
        "--gold",
        type=str,
        default="data/gold_standard/R1364660.json",
        help="Path to gold standard JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/gold_standard/papers_list.json",
        help="Output path for papers list"
    )
    parser.add_argument(
        "--no-orkg",
        action="store_true",
        help="Skip ORKG fetching"
    )
    parser.add_argument(
        "--no-arxiv",
        action="store_true",
        help="Skip ArXiv search"
    )
    
    args = parser.parse_args()
    
    gold_path = PROJECT_ROOT / args.gold
    output_path = PROJECT_ROOT / args.output
    
    if not gold_path.exists():
        logger.error(f"Gold standard not found: {gold_path}")
        return 1
    
    # Build papers list
    papers_list = build_papers_list(
        gold_path,
        use_orkg=not args.no_orkg,
        use_arxiv=not args.no_arxiv
    )
    
    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(papers_list, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nSaved papers list to: {output_path}")
    
    # Summary
    found = sum(1 for p in papers_list if p["arxiv_id"] or p["pdf_url"])
    missing = len(papers_list) - found
    
    sources = {}
    for p in papers_list:
        src = p["source"] or "unknown"
        sources[src] = sources.get(src, 0) + 1
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total papers:        {len(papers_list)}")
    print(f"Found URLs/IDs:      {found}")
    print(f"Manual needed:       {missing}")
    print("\nBy source:")
    for src, count in sorted(sources.items()):
        print(f"  {src:20} {count}")
    print("=" * 70)
    print(f"\nReview and edit: {output_path}")
    print("Then add as 'papers' section to gold standard.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
