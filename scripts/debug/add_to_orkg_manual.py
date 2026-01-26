"""
Manual ORKG Update Script

This script helps you manually add extracted models to ORKG sandbox.
Since the ORKG API for comparisons has changed, this provides a simplified approach.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orkg_client import ORKGClient
from dotenv import load_dotenv
import os

load_dotenv()


def main():
    """Add extracted models to ORKG manually."""
    
    print("=" * 80)
    print("Manual ORKG Update Script")
    print("=" * 80)
    
    # Load extracted data
    extracted_file = "data/extracted/2307.09288_20251130_174636.json"
    
    if not Path(extracted_file).exists():
        print(f"Error: Extracted file not found: {extracted_file}")
        return
    
    with open(extracted_file) as f:
        data = json.load(f)
    
    print(f"\nLoaded data from: {extracted_file}")
    print(f"Models extracted: {data['models_extracted']}")
    print(f"Paper: {data['paper_metadata']['title']}")
    
    # Initialize ORKG client
    print("\nInitializing ORKG client...")
    client = ORKGClient(
        host="sandbox",
        email=os.getenv("ORKG_EMAIL"),
        password=os.getenv("ORKG_PASSWORD")
    )
    
    if not client.ping():
        print("Error: Cannot connect to ORKG")
        return
    
    print("✓ Connected to ORKG sandbox")
    
    # Display extracted models
    print(f"\n{'='*80}")
    print("Extracted Models:")
    print(f"{'='*80}")
    
    for i, model in enumerate(data['extraction_data'][:10], 1):  # Show first 10
        print(f"\n{i}. {model['model_name']}")
        if model.get('parameters'):
            print(f"   Parameters: {model['parameters']}")
        if model.get('architecture'):
            print(f"   Architecture: {model['architecture']}")
        if model.get('organization'):
            print(f"   Organization: {model['organization']}")
    
    if len(data['extraction_data']) > 10:
        print(f"\n... and {len(data['extraction_data']) - 10} more models")
    
    print(f"\n{'='*80}")
    print("Next Steps:")
    print(f"{'='*80}")
    print("\n1. Visit: https://sandbox.orkg.org/comparisons/R2147679")
    print("2. Click 'Edit' or 'Add contribution'")
    print("3. Add the paper: Llama 2: Open Foundation and Fine-Tuned Chat Models")
    print("4. Add model information for each variant (use data above)")
    print("\nAlternatively, export to CSV for easier import:")
    print("  python scripts/export_to_csv.py")
    
    print("\n" + "="*80)
    print("Data is ready for manual addition to ORKG sandbox")
    print("="*80)


if __name__ == "__main__":
    main()

