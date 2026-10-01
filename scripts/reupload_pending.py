#!/usr/bin/env python3
"""
Re-upload papers that were extracted but never reached ORKG.

The 401 outage left 110 papers with a complete extraction on disk and no ORKG
paper id in the ledger. Everything those uploads need was saved at extraction
time, so this pushes the stored JSON straight to ORKG — no PDF fetching, no
classification, no LLM calls, no KISSKI quota.

Work list comes from the ledger itself (data/discovery/processed_papers.csv):
every row with an empty orkg_paper_id whose saved extraction exists and does not
already record steps.update_orkg == "success". That makes the run resumable — a
paper that uploads successfully gets its id written back to both the ledger and
the extraction file, and is skipped next time.

The ledger's `status` column is deliberately not part of that test; see
pending_rows() for why.

USAGE
-----
    python scripts/reupload_pending.py --dry-run      # show what would upload
    python scripts/reupload_pending.py --limit 1      # one paper, to prove it works
    python scripts/reupload_pending.py                # the rest (sandbox)
    python scripts/reupload_pending.py --host production   # the rest, to the LIVE ORKG
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import ExtractionPipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEDGER = PROJECT_ROOT / "data" / "discovery" / "processed_papers.csv"


def strip_version(arxiv_id: str) -> str:
    return (arxiv_id or "").strip().rstrip().split("v")[0] if "v" in (arxiv_id or "") else (arxiv_id or "").strip()


def extraction_path(quarter: str, arxiv_id: str) -> Path:
    """Saved extraction lives at data/discovery/<year>/<quarter>/extraction/<id>.json."""
    year, q = quarter.split("-")
    return PROJECT_ROOT / "data" / "discovery" / year / q / "extraction" / f"{arxiv_id}.json"


def pending_rows(ledger: Path) -> List[Dict[str, str]]:
    """
    Ledger rows with a finished extraction on disk that never reached ORKG.

    Keyed on the saved extraction rather than the ledger's `status` column: a
    paper that failed once and succeeded on a later run keeps the stale
    `failed` status, so filtering on status alone silently skips papers that
    are extracted and merely un-uploaded. The extraction file's own
    steps.update_orkg is the honest record.
    """
    with ledger.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    pending = []
    for row in rows:
        if (row.get("orkg_paper_id") or "").strip():
            continue  # already in ORKG
        arxiv_id = strip_version(row.get("arxiv_id", ""))
        if not arxiv_id or not row.get("quarter"):
            continue
        path = extraction_path(row["quarter"], arxiv_id)
        if not path.exists():
            continue  # never extracted — needs quarterly.py run, not a re-upload
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (saved.get("steps") or {}).get("update_orkg") == "success":
            continue
        if not saved.get("extraction_data"):
            continue
        pending.append(row)
    return pending


def build_upload_payload(saved: Dict[str, Any]) -> tuple:
    """
    Rebuild exactly the two dicts ExtractionPipeline hands to the manager.

    Mirrors src/pipeline.py: the manager re-maps from the raw models, so
    validated_rows has to travel with them or the upload silently re-runs the
    heuristic splitter and undoes semantic validation.
    """
    metadata = saved.get("paper_metadata") or {}
    published = metadata.get("published") or ""

    extraction_data = {
        "raw_extraction": saved.get("extraction_data"),
        "validated_rows": saved.get("validated_rows"),
        "paper_title": metadata.get("title"),
        "arxiv_id": saved.get("arxiv_id"),
        "paper_url": metadata.get("pdf_url"),
    }

    def part(text: str, start: int, end: int) -> Optional[int]:
        try:
            return int(text[start:end])
        except (ValueError, IndexError):
            return None

    month = part(published, 5, 7)
    orkg_metadata = {
        "title": metadata.get("title"),
        "authors": metadata.get("authors", []),
        "year": part(published, 0, 4),
        "month": month if month and 1 <= month <= 12 else None,
        "url": metadata.get("pdf_url"),
        "doi": metadata.get("doi") if metadata.get("doi") not in (None, "None") else None,
    }
    return extraction_data, orkg_metadata


def write_back(ledger: Path, arxiv_id: str, saved: Dict[str, Any], result: Dict[str, Any]) -> None:
    """
    Rebuild the whole ledger row from the saved extraction, immediately.

    Rewrites every column via quarterly.py's own _ledger_row rather than
    patching the ORKG ids alone. A row reached here because an earlier attempt
    failed, so `status`, `error`, `models_extracted`, `model_names` and
    `processed_at` are all stale too — patching only the ids leaves a row that
    claims failure while carrying a paper id, which is worse than either.

    Written after every paper rather than at the end: a run that dies halfway
    must not re-upload what it already pushed, which would duplicate papers in
    the graph.
    """
    from scripts.quarterly import _ledger_row  # local: keeps import cost off --dry-run

    with ledger.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    for index, row in enumerate(rows):
        if strip_version(row.get("arxiv_id", "")) != arxiv_id:
            continue
        merged = dict(saved)
        merged["orkg_results"] = result
        paper = {"arxiv_id": row.get("arxiv_id", ""), "pdf_url": row.get("pdf_url", "")}
        fresh = _ledger_row(
            row.get("quarter", ""),
            paper,
            row.get("title", ""),
            merged,
            result.get("new_contribution_ids") or result.get("contribution_ids") or [],
            "extracted",
        )
        # Only columns the ledger actually has, so a CSV written by an older
        # version of quarterly.py does not gain stray fields.
        rows[index] = {key: fresh.get(key, row.get(key, "")) for key in fieldnames}
        break

    tmp = ledger.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(ledger)


def update_saved_extraction(path: Path, result: Optional[Dict[str, Any]]) -> None:
    """Keep the stored extraction honest about whether it reached ORKG."""
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("  could not update %s: %s", path.name, exc)
        return
    saved.setdefault("steps", {})["update_orkg"] = "success" if result else "failed"
    saved["orkg_results"] = result or {"status": "failed", "error": "Re-upload failed"}
    try:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        logger.warning("  could not write %s: %s", path.name, exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N pending papers")
    parser.add_argument("--arxiv-id", action="append", default=None,
                        help="upload only this arXiv id (repeatable)")
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be uploaded, touch nothing")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="seconds between uploads (default 1.0)")
    parser.add_argument("--host", default="sandbox",
                        help="ORKG instance: sandbox (default), incubating or production. "
                             "Overrides ORKG_ENDPOINT_URL/ORKG_HOST in .env")
    args = parser.parse_args()

    rows = pending_rows(args.ledger)
    if args.arxiv_id:
        wanted = {strip_version(a) for a in args.arxiv_id}
        rows = [r for r in rows if strip_version(r["arxiv_id"]) in wanted]
    if args.limit:
        rows = rows[: args.limit]

    if not rows:
        logger.info("Nothing pending — every extracted paper has an ORKG id.")
        return 0

    total_models = sum(int(r.get("models_extracted") or 0) for r in rows)
    logger.info("%d paper(s), %d model contribution(s) to upload", len(rows), total_models)

    if args.dry_run:
        for row in rows:
            arxiv_id = strip_version(row["arxiv_id"])
            path = extraction_path(row["quarter"], arxiv_id)
            mark = "ok " if path.exists() else "MISSING"
            logger.info("  [%s] %s %s  %s", mark, row["quarter"], arxiv_id,
                        (row.get("title") or "")[:60])
        missing = sum(1 for r in rows
                      if not extraction_path(r["quarter"], strip_version(r["arxiv_id"])).exists())
        if missing:
            logger.warning("%d paper(s) have no saved extraction and cannot be re-uploaded", missing)
        return 0

    # --host outranks ORKG_ENDPOINT_URL / ORKG_HOST in .env, so the live ORKG is
    # only written to when asked for. ORKG_HOST_ACTIVE keeps the ledger's
    # orkg_url links on the same instance (see quarterly.orkg_paper_url).
    pipeline = ExtractionPipeline(orkg_endpoint_url=args.host)
    os.environ["ORKG_HOST_ACTIVE"] = pipeline.config["orkg"]["host"]
    logger.info("ORKG upload target: %s", pipeline.config["orkg"]["endpoint_url"])

    uploaded = skipped = failed = 0
    for index, row in enumerate(rows, start=1):
        arxiv_id = strip_version(row["arxiv_id"])
        label = f"[{index}/{len(rows)}] {arxiv_id}"
        path = extraction_path(row["quarter"], arxiv_id)

        if not path.exists():
            logger.error("%s -> no saved extraction at %s", label, path)
            skipped += 1
            continue

        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.error("%s -> unreadable extraction: %s", label, exc)
            skipped += 1
            continue

        if not saved.get("extraction_data"):
            logger.error("%s -> extraction JSON has no models", label)
            skipped += 1
            continue

        extraction_data, orkg_metadata = build_upload_payload(saved)
        logger.info("%s -> uploading %d model(s): %s",
                    label, len(extraction_data["raw_extraction"] or []),
                    (orkg_metadata.get("title") or "")[:60])

        try:
            # Going through the property re-checks the access token every time,
            # which is what keeps a long run from dying on an expired header.
            result = pipeline.orkg_manager.process_and_upload(extraction_data, orkg_metadata)
        except Exception as exc:  # noqa: BLE001 - one bad paper must not stop the run
            logger.exception("%s -> crashed: %s", label, exc)
            result = None

        update_saved_extraction(path, result)

        if result and result.get("paper_id"):
            write_back(args.ledger, arxiv_id, saved, result)
            logger.info("%s -> %s (%d contribution(s))", label, result["paper_id"],
                        len(result.get("contribution_ids") or []))
            uploaded += 1
        else:
            logger.error("%s -> upload failed", label)
            failed += 1

        if index < len(rows) and args.delay:
            time.sleep(args.delay)

    logger.info("Done. uploaded=%d failed=%d skipped=%d", uploaded, failed, skipped)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
