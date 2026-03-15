#!/bin/bash
# Generate papers_ids.txt from the gold standard papers_list.json.
# Run once on Grete before submitting jobs.
#
# Usage (from ~/llm-extraction on Grete):
#   bash grete/jobs/setup_papers_list.sh

cd ~/llm-extraction

OUTPUT="data/gold_standard/papers_ids.txt"

python3 - <<'EOF'
import json, pathlib

path = pathlib.Path("data/gold_standard/papers_list.json")
if not path.exists():
    raise FileNotFoundError(f"Not found: {path}. Upload data/gold_standard/papers_list.json first.")

data = json.loads(path.read_text())
ids = [p["arxiv_id"] for p in data]
out = pathlib.Path("data/gold_standard/papers_ids.txt")
out.write_text("\n".join(ids) + "\n")
print(f"Written {len(ids)} ArXiv IDs to {out}")
EOF
