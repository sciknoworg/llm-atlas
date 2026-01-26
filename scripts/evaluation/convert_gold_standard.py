"""
Convert ORKG Comparison CSV to Gold-Standard JSON Dataset

This script converts an exported ORKG comparison CSV (e.g., R1364660) 
into a JSON gold-standard dataset for evaluating the extraction pipeline.

Usage:
    python scripts/evaluation/convert_gold_standard.py
    
Input:
    data/gold_standard/R1364660.csv (ORKG comparison export)
    
Output:
    data/gold_standard/R1364660.json (Gold-standard dataset)
"""

import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_parameters_millions(value: str) -> int:
    """Convert parameter string to millions (integer)."""
    if not value or value.lower() in ['', 'n/a', 'null', 'none']:
        return None
    
    # Handle ranges like "110-340"
    if '-' in value:
        # Take the maximum value
        parts = value.split('-')
        value = parts[-1].strip()
    
    # Handle comma-separated values like "125M, 350M, 774M"
    if ',' in value:
        # Take the maximum value
        parts = value.split(',')
        value = parts[-1].strip()
    
    # Remove 'M', 'B', and convert
    value = value.upper().replace('M', '').replace('B', '').strip()
    
    try:
        num = float(value)
        return int(num)
    except (ValueError, AttributeError):
        logger.warning(f"Could not parse parameters_millions: {value}")
        return None


def map_csv_row_to_json(row: Dict[str, str]) -> Dict[str, Any]:
    """
    Map a single CSV row to the extraction pipeline's JSON format.
    
    CSV columns → JSON fields mapping:
    - "Title" → paper_title (extract from contribution title)
    - "model family" → model_family
    - "model name" → model_name
    - "date created" → date_created
    - "organization" → organization
    - "innovation" → innovation
    - "pretraining architecture" → pretraining_architecture
    - "pretraining task" → pretraining_task
    - "fine-tuning task" → finetuning_task
    - "optimizer" → optimizer
    - "number of parameters" → parameters
    - "maximum number of parameters (in million)" → parameters_millions
    - "hardware used" → hardware_used
    - "extension" → extension
    - "blog post" → blog_post
    - "license" → license
    - "research problem" → research_problem
    """
    # Extract paper title from contribution title (remove " - Contribution" suffix)
    # After normalization, the column should be "Title" (BOM and quotes removed)
    title_full = row.get("Title", "")
    if not title_full:
        logger.warning(f"Empty Title column for model: {row.get('model name', 'Unknown')}")
    
    # Handle both " - Contribution" and " - Contribution 1", " - Contribution 2", etc.
    if " - Contribution" in title_full:
        paper_title = title_full.split(" - Contribution")[0].strip()
    else:
        paper_title = title_full.strip() if title_full else ""
    
    # Helper function to safely get and strip CSV values
    def safe_get(key: str) -> str:
        value = row.get(key)
        if value is None:
            return None
        stripped = value.strip()
        return stripped if stripped else None
    
    # Map CSV fields to JSON structure
    model_data = {
        "paper_title": paper_title,
        "model_name": safe_get("model name"),
        "model_family": safe_get("model family"),
        "date_created": safe_get("date created"),
        "organization": safe_get("organization"),
        "innovation": safe_get("innovation"),
        "pretraining_architecture": safe_get("pretraining architecture"),
        "pretraining_task": safe_get("pretraining task"),
        "finetuning_task": safe_get("fine-tuning task"),
        "optimizer": safe_get("optimizer"),
        "parameters": safe_get("number of parameters"),
        "parameters_millions": parse_parameters_millions(row.get("maximum number of parameters (in million)", "") or ""),
        "hardware_used": safe_get("hardware used"),
        "extension": safe_get("extension"),
        "blog_post": safe_get("blog post"),
        "license": safe_get("license"),
        "research_problem": safe_get("research problem"),
        "pretraining_corpus": None,  # Not in CSV, set to None
        "application": None  # Not in CSV, set to None
    }
    
    return model_data


def normalize_csv_row(row: Dict[str, str]) -> Dict[str, str]:
    """
    Normalize CSV row by cleaning column names and values.
    Handles BOM characters and quoted column names.
    """
    normalized = {}
    for key, value in row.items():
        if key is None:
            continue
        # Remove BOM and strip quotes from column names
        clean_key = str(key).lstrip('\ufeff').strip('"').strip()
        # Strip quotes from values if present
        if value is None:
            clean_value = None
        else:
            clean_value = str(value).strip('"').strip() if value else value
        normalized[clean_key] = clean_value
    return normalized


def convert_csv_to_json(csv_path: Path, json_path: Path) -> None:
    """
    Convert ORKG comparison CSV to gold-standard JSON dataset.
    
    Args:
        csv_path: Path to input CSV file
        json_path: Path to output JSON file
    """
    logger.info(f"Reading CSV from: {csv_path}")
    
    models = []
    
    with open(csv_path, 'r', encoding='utf-8-sig') as csvfile:  # utf-8-sig handles BOM automatically
        reader = csv.DictReader(csvfile)
        
        for i, row in enumerate(reader, 1):
            # Normalize the row to handle BOM and quoted column names
            normalized_row = normalize_csv_row(row)
            model_data = map_csv_row_to_json(normalized_row)
            models.append(model_data)
            logger.info(f"Processed model {i}: {model_data.get('model_name')}")
    
    # Create gold-standard dataset structure
    gold_standard = {
        "source": "ORKG Comparison R1364660",
        "description": "Gold-standard dataset manually curated from ORKG for LLM extraction evaluation",
        "total_models": len(models),
        "extraction_data": models
    }
    
    # Write JSON output
    logger.info(f"Writing JSON to: {json_path}")
    with open(json_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(gold_standard, jsonfile, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Conversion complete: {len(models)} models converted")
    logger.info(f"✓ Gold-standard dataset saved to: {json_path}")


def main():
    """Main entry point."""
    # Define paths
    project_root = Path(__file__).parent.parent.parent
    csv_path = project_root / "data" / "gold_standard" / "R1364660.csv"
    json_path = project_root / "data" / "gold_standard" / "R1364660.json"
    
    # Validate CSV exists
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        logger.error("Please export the ORKG comparison R1364660 as CSV and place it in data/gold_standard/")
        return
    
    # Create output directory if needed
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert
    try:
        convert_csv_to_json(csv_path, json_path)
        
        # Display sample
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("\n" + "=" * 80)
        print("GOLD-STANDARD DATASET SUMMARY")
        print("=" * 80)
        print(f"Source: {data['source']}")
        print(f"Total models: {data['total_models']}")
        print(f"\nSample (first model):")
        print(json.dumps(data['extraction_data'][0], indent=2, ensure_ascii=False))
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
