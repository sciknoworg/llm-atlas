"""
Batch extraction for all papers in papers list.

Runs extraction on every paper from papers_list.json using the current
KISSKI model configured in config/config.yaml.

Usage:
    # 1. Set model in config/config.yaml (kisski.model)
    # 2. Run this script
    python scripts/batch_extract_all_papers.py

    # Dry-run (see what would be extracted without running)
    python scripts/batch_extract_all_papers.py --dry-run

    # Resume from failure
    python scripts/batch_extract_all_papers.py --skip-existing

Output:
    Extraction JSONs saved to data/extracted/<model_name>/
    Log saved to data/logs/batch_extraction_<model>_<timestamp>.log
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Setup project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from src.pipeline import ExtractionPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_papers_list(path: Path) -> List[Dict[str, str]]:
    """Load papers list JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def slugify(text: str) -> str:
    """Create filename-safe slug from text."""
    import re
    slug = text.lower()
    slug = re.sub(r'[^a-z0-9]+', '_', slug)
    slug = slug.strip('_')
    return slug[:100]  # Limit length


def get_model_name_from_config() -> str:
    """Get current KISSKI model from config."""
    import yaml
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get("kisski", {}).get("model", "unknown_model")


def main():
    parser = argparse.ArgumentParser(description="Batch extract all papers using current KISSKI model")
    parser.add_argument(
        "--papers-list",
        type=str,
        default="data/gold_standard/papers_list.json",
        help="Path to papers list JSON"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/extracted/<model_name>/)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be extracted without running"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip papers that already have extraction output"
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Start from paper index N (for resuming)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N papers (for testing)"
    )
    
    args = parser.parse_args()
    
    # Load papers list
    papers_list_path = PROJECT_ROOT / args.papers_list
    if not papers_list_path.exists():
        logger.error(f"Papers list not found: {papers_list_path}")
        return 1
    
    papers = load_papers_list(papers_list_path)
    logger.info(f"Loaded {len(papers)} papers from {papers_list_path}")
    
    # Get current KISSKI model
    current_model = get_model_name_from_config()
    logger.info(f"Current KISSKI model (from config): {current_model}")
    
    # Setup output directory
    if args.output_dir:
        output_dir = PROJECT_ROOT / args.output_dir
    else:
        model_slug = slugify(current_model)
        output_dir = PROJECT_ROOT / "data" / "extracted" / model_slug
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Apply filters
    papers_to_process = papers[args.start_from:]
    if args.limit:
        papers_to_process = papers_to_process[:args.limit]
    
    logger.info(f"Processing {len(papers_to_process)} papers (start: {args.start_from}, limit: {args.limit or 'none'})")
    
    if args.dry_run:
        logger.info("DRY RUN - no extraction will be performed")
        for i, paper in enumerate(papers_to_process, start=args.start_from + 1):
            arxiv_id = paper.get("arxiv_id")
            pdf_url = paper.get("pdf_url")
            title = paper["paper_title"]
            source = paper.get("source", "unknown")
            
            method = "arxiv_id" if arxiv_id else ("pdf_url" if pdf_url else "none")
            print(f"{i:3d}. [{source:15s}] [{method:10s}] {title[:60]}")
        
        logger.info(f"Would process {len(papers_to_process)} papers with model {current_model}")
        return 0
    
    # Initialize pipeline
    logger.info("Initializing extraction pipeline...")
    pipeline = ExtractionPipeline()
    
    # Track results
    results = {
        "model": current_model,
        "total_papers": len(papers_to_process),
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "start_time": datetime.now().isoformat(),
        "papers": []
    }
    
    # Process each paper
    for i, paper in enumerate(papers_to_process, start=args.start_from + 1):
        title = paper["paper_title"]
        arxiv_id = paper.get("arxiv_id")
        pdf_url = paper.get("pdf_url")
        
        logger.info(f"\n[{i}/{len(papers)}] Processing: {title[:80]}")
        
        # Check if output already exists (pipeline names files by arxiv_id or title slug)
        paper_slug = slugify(title)
        if arxiv_id:
            prefix = arxiv_id.replace("/", "_")
        else:
            prefix = paper_slug
        existing_outputs = list(output_dir.glob(f"{prefix}*.json"))
        
        if args.skip_existing and existing_outputs:
            logger.info(f"  Skipping (output exists): {existing_outputs[0].name}")
            results["skipped"] += 1
            continue
        
        # Determine extraction method
        result_entry = {
            "paper_title": title,
            "arxiv_id": arxiv_id,
            "pdf_url": pdf_url,
            "status": None,
            "output_file": None,
            "error": None
        }
        
        try:
            if arxiv_id:
                # Extract by ArXiv ID
                logger.info(f"  Method: ArXiv ID ({arxiv_id})")
                result = pipeline.process_paper(arxiv_id, save_intermediate=True, update_orkg=False)
                
            elif pdf_url:
                # Extract by PDF URL
                logger.info(f"  Method: PDF URL")
                result = pipeline.process_paper_from_pdf_url(
                    pdf_url=pdf_url,
                    paper_title=title,
                    save_intermediate=True,
                    update_orkg=False
                )
            else:
                logger.warning(f"  No arxiv_id or pdf_url for paper: {title}")
                result_entry["status"] = "no_source"
                result_entry["error"] = "No arxiv_id or pdf_url"
                results["failed"] += 1
                results["papers"].append(result_entry)
                continue
            
            if result and result.get("status") == "completed":
                saved_path = result.get("saved_path")
                if saved_path and Path(saved_path).exists():
                    # Copy to model-specific folder so aggregate script finds it
                    dest_file = output_dir / Path(saved_path).name
                    try:
                        shutil.copy2(saved_path, dest_file)
                        result_entry["status"] = "success"
                        result_entry["output_file"] = str(dest_file)
                        results["completed"] += 1
                        logger.info("  [OK] Success: %s", dest_file.name)
                    except Exception as copy_err:
                        result_entry["status"] = "success"
                        result_entry["output_file"] = str(saved_path)
                        results["completed"] += 1
                        logger.info("  [OK] Success (saved to default dir): %s", Path(saved_path).name)
                else:
                    result_entry["status"] = "success"
                    result_entry["output_file"] = str(saved_path) if saved_path else ""
                    results["completed"] += 1
                    logger.info("  [OK] Success: %s", saved_path or "extraction completed")
            else:
                result_entry["status"] = "failed"
                result_entry["error"] = "No result or saved_path"
                results["failed"] += 1
                logger.warning("  [FAIL] No result or saved_path")
        
        except Exception as e:
            result_entry["status"] = "error"
            result_entry["error"] = str(e)
            results["failed"] += 1
            logger.error("  [FAIL] Error: %s", e)
        
        results["papers"].append(result_entry)
        
        # Rate limiting (KISSKI: 2 seconds between requests is already in pipeline)
        # Additional delay for safety
        time.sleep(1)
    
    # Save results summary
    results["end_time"] = datetime.now().isoformat()
    summary_file = output_dir / f"extraction_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print("BATCH EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"Model:      {current_model}")
    print(f"Total:      {results['total_papers']}")
    print(f"Completed:  {results['completed']}")
    print(f"Failed:     {results['failed']}")
    print(f"Skipped:    {results['skipped']}")
    print("=" * 70)
    print(f"\nOutputs saved to: {output_dir}")
    print(f"Summary: {summary_file}")
    
    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
