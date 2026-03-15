"""
Import flat extraction JSONs into data/extracted/<model_slug>/.

Reads each JSON in data/extracted/ (top-level only), gets model_used,
creates data/extracted/<slug>/ and copies the file there.
Matches the layout used by scripts/batch_extract_all_papers.py.

Usage (from project root):
    python scripts/import_extracted_to_model_folders.py
    python scripts/import_extracted_to_model_folders.py --move   # move instead of copy
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED = PROJECT_ROOT / "data" / "extracted"


def slugify(text: str) -> str:
    """Create filename-safe slug from model name (same as batch_extract_all_papers.py)."""
    slug = (text or "unknown").lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
    return slug[:100] or "unknown"


def main():
    parser = argparse.ArgumentParser(description="Import flat extraction JSONs into model-named subfolders")
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying (removes originals from data/extracted/)",
    )
    args = parser.parse_args()

    if not EXTRACTED.exists():
        print(f"Directory not found: {EXTRACTED}")
        sys.exit(1)

    # Only process JSON files directly in data/extracted/ (not inside subfolders)
    flat_files = [f for f in EXTRACTED.iterdir() if f.is_file() and f.suffix.lower() == ".json"]
    if not flat_files:
        print("No flat JSON files found in data/extracted/")
        return

    print(f"Found {len(flat_files)} JSON file(s) in data/extracted/")
    action = "Moving" if args.move else "Copying"

    for f in flat_files:
        try:
            with open(f, encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception as e:
            print(f"  Skip {f.name}: {e}")
            continue

        model = data.get("model_used") or "unknown"
        slug = slugify(model)
        out_dir = EXTRACTED / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f.name

        if dest.resolve() == f.resolve():
            continue
        if dest.exists() and dest.stat().st_mtime >= f.stat().st_mtime:
            print(f"  Skip (already exists): {slug}/{f.name}")
            continue

        if args.move:
            shutil.move(str(f), str(dest))
        else:
            shutil.copy2(f, dest)
        print(f"  {action.lower():4} {f.name} -> {slug}/")

    print("Done.")


if __name__ == "__main__":
    main()
