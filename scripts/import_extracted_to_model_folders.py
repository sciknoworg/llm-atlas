"""
Import flat extraction JSONs into <extraction_root>/<model_slug>/.

Reads each JSON in the configured pipeline extraction directory (top-level only),
gets model_used, creates <slug>/ and copies the file there.
Default root is pipeline.extraction_output_dir from config/config.yaml.

Usage (from project root):
    python scripts/import_extracted_to_model_folders.py
    python scripts/import_extracted_to_model_folders.py --move   # move instead of copy
    python scripts/import_extracted_to_model_folders.py --source-dir data/extracted
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.path_utils import resolve_project_path

def default_extraction_root() -> Path:
    import yaml

    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    raw = (config.get("pipeline") or {}).get("extraction_output_dir")
    if not raw:
        raise ValueError("Missing pipeline.extraction_output_dir in config/config.yaml")
    return resolve_project_path(str(raw))


def slugify(text: str) -> str:
    """Create filename-safe slug from model name (same as batch_extract_all_papers.py)."""
    slug = (text or "unknown").lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
    return slug[:100] or "unknown"


def main():
    parser = argparse.ArgumentParser(description="Import flat extraction JSONs into model-named subfolders")
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help=(
            "Directory with flat JSON files (relative to project root unless absolute). "
            "Default: pipeline.extraction_output_dir from config/config.yaml."
        ),
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying (removes originals from the source directory)",
    )
    args = parser.parse_args()

    extracted_root = resolve_project_path(args.source_dir) if args.source_dir else default_extraction_root()

    if not extracted_root.exists():
        print(f"Directory not found: {extracted_root}")
        sys.exit(1)

    # Only process JSON files directly under source (not inside subfolders)
    flat_files = [f for f in extracted_root.iterdir() if f.is_file() and f.suffix.lower() == ".json"]
    if not flat_files:
        print(f"No flat JSON files found in {extracted_root}")
        return

    print(f"Found {len(flat_files)} JSON file(s) in {extracted_root}")
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
        out_dir = extracted_root / slug
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
