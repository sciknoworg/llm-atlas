"""Quick verification script for gold standard JSON."""
import json
from pathlib import Path

json_path = Path(__file__).parent.parent.parent / "data" / "gold_standard" / "R1364660.json"

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

titles = [item['paper_title'] for item in data['extraction_data']]
unique_titles = set(titles)

print("=" * 80)
print("GOLD STANDARD VERIFICATION")
print("=" * 80)
print(f"Total models: {data['total_models']}")
print(f"Unique papers: {len(unique_titles)}")
print(f"\nInvalid titles check:")
invalid = [t for t in titles if any(x in t for x in ['Compared with T0', 'T0pp is recommended', 'a variety of NLP tasks'])]
print(f"  Invalid titles found: {len(invalid)}")
if invalid:
    for t in invalid:
        print(f"    - {t}")

print(f"\nSample papers (first 10):")
for i, t in enumerate(sorted(unique_titles)[:10], 1):
    print(f"  {i}. {t}")
