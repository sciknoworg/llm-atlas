"""
Export extracted models to CSV for easier ORKG import.
"""

import json
import csv
from pathlib import Path


def main():
    """Export models to CSV."""
    
    # Load extracted data
    extracted_file = "data/extracted/2307.09288_20251130_174636.json"
    
    if not Path(extracted_file).exists():
        print(f"Error: File not found: {extracted_file}")
        return
    
    with open(extracted_file) as f:
        data = json.load(f)
    
    # Export to CSV
    output_file = "data/extracted/llama2_models.csv"
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'model_name',
            'model_version',
            'parameters',
            'architecture',
            'training_data',
            'training_compute',
            'context_length',
            'organization',
            'license',
            'model_type'
        ]
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for model in data['extraction_data']:
            row = {}
            for field in fieldnames:
                value = model.get(field)
                if isinstance(value, dict):
                    # Convert dict to string
                    value = "; ".join([f"{k}: {v}" for k, v in value.items()])
                row[field] = value if value else ""
            writer.writerow(row)
    
    print(f"✓ Exported {len(data['extraction_data'])} models to: {output_file}")
    print("\nYou can now:")
    print("1. Open the CSV file in Excel")
    print("2. Review and clean the data")
    print("3. Import to ORKG sandbox")


if __name__ == "__main__":
    main()

