"""
List available models from KISSKI Chat AI API (programmatic endpoint).

Fetches the full model list via the OpenAI-compatible /v1/models endpoint.
Use this to send the current list to your professor so she can choose which
models to test for the thesis evaluation.

Usage:
    # From project root (recommended)
    python scripts/list_kisski_models.py

    # Save list to file for email/Element
    python scripts/list_kisski_models.py --output data/kisski_models_list.txt

Requirements:
    - KISSKI_API_KEY and KISSKI_BASE_URL in .env (or environment)
    - pip install openai python-dotenv
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on path and load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser(
        description="List available KISSKI Chat AI models via the API (programmatic endpoint)."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Optional: save model IDs to this file (e.g. data/kisski_models_list.txt)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Only print model IDs (one per line), no header/footer",
    )
    args = parser.parse_args()

    api_key = os.getenv("KISSKI_API_KEY")
    base_url = os.getenv("KISSKI_BASE_URL", "https://chat-ai.academiccloud.de/v1")

    if not api_key or api_key == "your_kisski_api_key_here":
        print("ERROR: KISSKI_API_KEY not set. Add it to .env or set the environment variable.", file=sys.stderr)
        print("  .env should contain: KISSKI_API_KEY=<your_key>", file=sys.stderr)
        return 1

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai", file=sys.stderr)
        return 1

    if not args.quiet:
        print("KISSKI Chat AI API – available models (programmatic list)")
        print("Endpoint:", base_url)
        print()

    client = OpenAI(api_key=api_key, base_url=base_url)
    models_response = client.models.list()
    model_ids = [m.id for m in models_response.data]

    # Sort for consistent output
    model_ids.sort()

    for mid in model_ids:
        print(mid)

    if not args.quiet:
        print()
        print(f"Total: {len(model_ids)} models")

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(model_ids) + "\n", encoding="utf-8")
        if not args.quiet:
            print(f"Saved to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
