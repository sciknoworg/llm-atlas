import argparse
import csv
import json
import logging
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.path_utils import PROJECT_ROOT  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DISCOVERY_ROOT = PROJECT_ROOT / "data" / "discovery"
INDEX_PATH = DISCOVERY_ROOT / "_index.json"
LEDGER_PATH = DISCOVERY_ROOT / "processed_papers.csv"

LEDGER_COLUMNS = [
    "quarter",
    "arxiv_id",
    "pdf_url",
    "title",
    "models_extracted",
    "model_names",
    "orkg_paper_id",
    "contribution_ids",
    "orkg_url",
    "classification",
    "classification_reason",
    "status",
    "error",
    "processed_at",
]
FIRST_QUARTER = (2020, 1)

_QUARTER_RE = re.compile(r"^(\d{4})-Q([1-4])$")


# --------------------------------------------------------------------------- #
# quarters
# --------------------------------------------------------------------------- #

def parse_quarter(text: str) -> Tuple[int, int]:
    match = _QUARTER_RE.match(text.strip().upper())
    if not match:
        raise argparse.ArgumentTypeError(f"expected YYYY-Qn, got {text!r}")
    return int(match.group(1)), int(match.group(2))


def quarter_name(year: int, quarter: int) -> str:
    return f"{year}-Q{quarter}"


def current_quarter(today: Optional[date] = None) -> Tuple[int, int]:
    today = today or date.today()
    return today.year, (today.month - 1) // 3 + 1


def all_quarters(
    start: Tuple[int, int] = FIRST_QUARTER, end: Optional[Tuple[int, int]] = None
) -> List[Tuple[int, int]]:
    """Every quarter from `start` to `end` inclusive (default: the current one)."""
    end = end or current_quarter()
    out, (year, quarter) = [], start
    while (year, quarter) <= end:
        out.append((year, quarter))
        quarter += 1
        if quarter > 4:
            year, quarter = year + 1, 1
    return out


def quarter_dir(name: str) -> Path:
    """Folder for a quarter: data/discovery/2020/Q1 for "2020-Q1"."""
    year, quarter = parse_quarter(name)
    return DISCOVERY_ROOT / str(year) / f"Q{quarter}"


def quarter_from_path(path: Path) -> Optional[str]:
    """
    Recover "2020-Q1" from .../data/discovery/2020/Q1/discovery.json.

    Quarters live one level under their year, so the name is assembled from the
    two enclosing directories rather than read from a single folder name.
    """
    quarter_folder = path.parent if path.suffix else path
    year_folder = quarter_folder.parent
    name = f"{year_folder.name}-{quarter_folder.name}"
    return name if _QUARTER_RE.match(name) else None


def discovery_files() -> List[Path]:
    """Every quarter's discovery.json, in chronological order."""
    return sorted(DISCOVERY_ROOT.glob("*/Q[1-4]/discovery.json"))


# --------------------------------------------------------------------------- #
# discovery.json
# --------------------------------------------------------------------------- #

def paper_key(paper: Dict[str, Any]) -> Optional[str]:
    """
    Stable identity for a paper across quarters.

    arXiv id when present (version suffix stripped so 2005.14165v3 and
    2005.14165 are one paper), otherwise the pdf_url, otherwise the normalized
    title. Titles are the weakest key, which is why they are the last resort.
    """
    arxiv_id = (paper.get("arxiv_id") or "").strip()
    if arxiv_id:
        return re.sub(r"v\d+$", "", arxiv_id)
    pdf_url = (paper.get("pdf_url") or "").strip()
    if pdf_url:
        return pdf_url
    title = (paper.get("title") or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "-", title).strip("-") or None


def load_discovery(name: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Read a quarter's discovery.json. Returns (payload, problems)."""
    path = quarter_dir(name) / "discovery.json"
    if not path.exists():
        return None, [f"missing {path.relative_to(PROJECT_ROOT)}"]

    try:
        payload = json.loads(path.read_text())
    except ValueError as exc:
        return None, [f"invalid JSON: {exc}"]

    problems: List[str] = []
    if not isinstance(payload, dict):
        return None, ["top level must be a JSON object"]

    payload.setdefault("quarter", name)
    if payload["quarter"] != name:
        problems.append(
            f"quarter field is {payload['quarter']!r} but the folder is {name!r}"
        )
    papers = payload.get("papers")
    if not isinstance(papers, list):
        return None, ["'papers' must be a list"]

    seen = set()
    for index, paper in enumerate(papers):
        where = f"papers[{index}]"
        if not isinstance(paper, dict):
            problems.append(f"{where}: not an object")
            continue
        if not (paper.get("title") or "").strip():
            problems.append(f"{where}: missing title")
        arxiv_id, pdf_url = paper.get("arxiv_id"), paper.get("pdf_url")
        if not arxiv_id and not pdf_url:
            problems.append(
                f"{where}: no arxiv_id and no pdf_url — {paper.get('title', '?')[:50]!r}"
            )
        if arxiv_id and pdf_url:
            problems.append(f"{where}: has BOTH arxiv_id and pdf_url; arxiv_id will win")
        if arxiv_id and not re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", str(arxiv_id).strip()):
            problems.append(f"{where}: {arxiv_id!r} is not a bare arXiv id")
        if pdf_url and not str(pdf_url).lower().endswith(".pdf"):
            # Not necessarily wrong: publisher URLs often serve application/pdf
            # without the extension (AAAI's /article/download/... does).
            problems.append(f"{where}: pdf_url has no .pdf suffix (may still be fine) — {pdf_url}")
        key = paper_key(paper)
        if key and key in seen:
            problems.append(f"{where}: duplicate of an earlier entry ({key})")
        if key:
            seen.add(key)

    return payload, problems


# --------------------------------------------------------------------------- #
# cross-quarter index
# --------------------------------------------------------------------------- #

def load_ledger() -> Dict[str, Dict[str, Any]]:
    """
    Existing ledger rows, keyed exactly as writes key them, so a resumed run
    updates a paper's row instead of adding a second one.

    Keying has to go through paper_key(): it strips the version suffix and
    slugifies a title-only entry, and every write path uses it. Reading the raw
    columns instead let a paper listed as "2005.14165v3" load under that key and
    be rewritten under "2005.14165", so save_ledger emitted both rows — and
    load_discovery explicitly accepts versioned ids, so that is valid input.
    """
    if not LEDGER_PATH.exists():
        return {}
    try:
        with LEDGER_PATH.open(newline="", encoding="utf-8") as handle:
            rows: Dict[str, Dict[str, Any]] = {}
            for row in csv.DictReader(handle):
                key = paper_key(row)
                if key:
                    rows[key] = row
            return rows
    except OSError as exc:
        logger.warning("Could not read %s: %s", LEDGER_PATH, exc)
        return {}


def save_ledger(rows: Dict[str, Dict[str, Any]]) -> None:
    """
    Rewrite the ledger, newest quarter last.

    Written whole rather than appended so a re-processed paper updates its row
    instead of gaining a second one, and via a temp file so an interrupted run
    cannot truncate it.
    """
    try:
        LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = LEDGER_PATH.with_suffix(".csv.tmp")
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=LEDGER_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            ordered = sorted(
                rows.values(), key=lambda r: (r.get("quarter") or "", r.get("title") or "")
            )
            for row in ordered:
                writer.writerow(row)
        tmp.replace(LEDGER_PATH)
    except OSError as exc:
        logger.warning("Could not write %s: %s", LEDGER_PATH, exc)


def orkg_paper_url(paper_id: Optional[str]) -> str:
    """Public ORKG link for a paper, on whichever instance the config targets."""
    if not paper_id:
        return ""
    try:
        import yaml

        from src.orkg_client import orkg_frontend_url

        config = yaml.safe_load((PROJECT_ROOT / "config" / "config.yaml").read_text())
        base = orkg_frontend_url(config.get("orkg", {}).get("host", "sandbox"))
    except Exception:  # noqa: BLE001 - a missing link must not break a run
        base = "https://sandbox.orkg.org"
    return f"{base}/paper/{paper_id}"


def load_index() -> Dict[str, Any]:
    if not INDEX_PATH.exists():
        return {}
    try:
        return json.loads(INDEX_PATH.read_text()).get("papers", {})
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring unreadable %s: %s", INDEX_PATH, exc)
        return {}


def save_index(papers: Dict[str, Any]) -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = INDEX_PATH.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(
            {"updated_at": datetime.now().isoformat(), "papers": papers},
            indent=1,
            sort_keys=True,
        )
        + "\n"
    )
    tmp.replace(INDEX_PATH)


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #

def cmd_init(args) -> int:
    end = args.until or current_quarter()
    created = 0
    for year, quarter in all_quarters(end=end):
        name = quarter_name(year, quarter)
        folder = quarter_dir(name)
        (folder / "extraction").mkdir(parents=True, exist_ok=True)
        stub = folder / "discovery.json"
        if stub.exists():
            continue
        stub.write_text(
            json.dumps(
                {
                    "quarter": name,
                    "papers": [],
                    "no_paper_found": [],
                },
                indent=2,
            )
            + "\n"
        )
        created += 1
    total = len(all_quarters(end=end))
    print(f"{total} quarter folder(s) under {DISCOVERY_ROOT.relative_to(PROJECT_ROOT)}")
    print(f"created {created} new discovery.json stub(s)")
    return 0


def quarter_window(name: str) -> Tuple[str, str]:
    """First and last day of a quarter, as ISO dates."""
    year, quarter = parse_quarter(name)
    first_month = (quarter - 1) * 3 + 1
    last_month = first_month + 2
    import calendar

    last_day = calendar.monthrange(year, last_month)[1]
    return f"{year}-{first_month:02d}-01", f"{year}-{last_month:02d}-{last_day:02d}"


def check_against_arxiv(name: str, papers: List[Dict[str, Any]]) -> List[str]:
    """
    Confirm each arxiv_id exists, its title matches, and it was submitted in the
    quarter. One batched request, so this costs a single API call per quarter.

    Catches the two failure modes seen in practice: an identifier that belongs to
    a different paper, and a paper listed under the quarter its VENUE fell in
    rather than the quarter it was submitted (VL-BERT is on arXiv from Aug 2019
    but appeared at ICLR in Apr 2020).
    """
    import urllib.parse
    import urllib.request
    import xml.etree.ElementTree as ET
    from difflib import SequenceMatcher

    atom = "{http://www.w3.org/2005/Atom}"
    ids = [str(p["arxiv_id"]).strip() for p in papers if p.get("arxiv_id")]
    if not ids:
        return []

    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"id_list": ",".join(ids), "max_results": len(ids)}
    )
    try:
        root = ET.fromstring(urllib.request.urlopen(url, timeout=90).read())
    except Exception as exc:  # noqa: BLE001 - a check, not a gate
        return [f"could not reach arXiv to verify ids: {exc}"]

    found = {}
    for entry in root.findall(atom + "entry"):
        aid = re.sub(r"v\d+$", "", (entry.findtext(atom + "id") or "").rsplit("/", 1)[-1])
        found[aid] = {
            "title": " ".join((entry.findtext(atom + "title") or "").split()),
            "published": (entry.findtext(atom + "published") or "")[:10],
        }

    def norm(text: str) -> str:
        return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())

    start, end = quarter_window(name)
    notes = []
    for paper in papers:
        aid = paper.get("arxiv_id")
        if not aid:
            continue
        aid = re.sub(r"v\d+$", "", str(aid).strip())
        hit = found.get(aid)
        label = paper.get("model_name") or paper.get("title", "?")[:30]
        if not hit:
            notes.append(f"{label}: arXiv has no {aid}")
            continue
        listed_title, real_title = norm(paper.get("title", "")), norm(hit["title"])
        ratio = SequenceMatcher(None, listed_title, real_title).ratio()
        # Papers are often cited without the "ModelName: " prefix that the arXiv
        # title carries ("Video Instruction Tuning With Synthetic Data" vs
        # "LLaVA-Video: Video Instruction Tuning With Synthetic Data"). One title
        # containing the other is the same paper, not a mismatch.
        contained = bool(listed_title) and (
            listed_title in real_title or real_title in listed_title
        )
        if ratio < 0.9 and not contained:
            notes.append(
                f"{label}: {aid} is {hit['title'][:52]!r}, not the listed title"
            )
        if not (start <= hit["published"] <= end):
            notes.append(
                f"{label}: {aid} submitted {hit['published']}, outside {name} ({start}..{end})"
            )
    return notes


def listed_elsewhere(current: str) -> Dict[str, str]:
    """
    Map paper key -> the OTHER quarter that already lists it.

    A generative model asked about consecutive quarters repeats notable papers:
    GPT-3 and Oscar were returned for both 2020-Q2 and 2020-Q3. The runner would
    catch this eventually via _index.json, but only after the earlier quarter had
    been processed. Reading the sibling discovery.json files surfaces it while
    the lists are still being collected.
    """
    elsewhere: Dict[str, str] = {}
    for folder in discovery_files():
        name = quarter_from_path(folder)
        if name is None or name == current:
            continue
        try:
            payload = json.loads(folder.read_text())
        except (OSError, ValueError):
            continue
        for paper in payload.get("papers") or []:
            if not isinstance(paper, dict):
                continue
            key = paper_key(paper)
            if key:
                elsewhere.setdefault(key, name)
    return elsewhere


def cmd_validate(args) -> int:
    names = _target_quarters(args)
    bad = 0
    for name in names:
        payload, problems = load_discovery(name)
        count = len(payload.get("papers", [])) if payload else 0
        if payload:
            elsewhere = listed_elsewhere(name)
            for paper in payload.get("papers") or []:
                if not isinstance(paper, dict):
                    continue
                key = paper_key(paper)
                if key and key in elsewhere:
                    problems.append(
                        f"{paper.get('model_name') or key}: also listed in "
                        f"{elsewhere[key]} — will only be processed once"
                    )
        if payload and args.check_arxiv:
            problems = problems + check_against_arxiv(name, payload.get("papers") or [])
        if problems:
            bad += 1
            print(f"\n{name}: {count} paper(s), {len(problems)} problem(s)")
            for problem in problems:
                print(f"   - {problem}")
        elif args.verbose or count:
            print(f"{name}: {count} paper(s), ok")
    print(f"\n{len(names)} quarter(s) checked, {bad} with problems")
    return 1 if bad else 0


def failures_by_quarter() -> Dict[str, int]:
    """
    Failed papers per quarter, counted from the ledger.

    The ledger is the only durable record of a run, so a failure stays counted
    until a later run overwrites that paper's row with a success.
    """
    failures: Dict[str, int] = {}
    for row in load_ledger().values():
        if (row.get("status") or "").strip().lower() == "failed":
            quarter = row.get("quarter") or ""
            failures[quarter] = failures.get(quarter, 0) + 1
    return failures


def cmd_status(args) -> int:
    index = load_index()
    failures = failures_by_quarter()
    print(f"{'quarter':10} {'listed':>7} {'done':>6} {'failed':>7}  notes")
    print("-" * 62)
    totals = [0, 0, 0]
    for year, quarter in all_quarters():
        name = quarter_name(year, quarter)
        folder = quarter_dir(name)
        if not folder.exists():
            continue
        payload, _ = load_discovery(name)
        listed = len(payload.get("papers", [])) if payload else 0
        done = len(list((folder / "extraction").glob("*.json"))) if folder.exists() else 0
        failed = failures.get(name, 0)
        engine = (payload or {}).get("notes") or ""
        totals = [totals[0] + listed, totals[1] + done, totals[2] + failed]
        print(f"{name:10} {listed:>7} {done:>6} {failed:>7}  {engine[:40]}")
    print("-" * 62)
    print(f"{'total':10} {totals[0]:>7} {totals[1]:>6} {totals[2]:>7}")
    print(f"\nunique papers in index: {len(index)}")
    return 0


GOLD_LIST = PROJECT_ROOT / "data" / "gold_standard" / "papers_list.json"
GOLD_DATES = PROJECT_ROOT / "data" / "gold_standard" / "papers_submission_dates.json"


def gold_by_quarter() -> Dict[str, List[Tuple[str, str]]]:
    """
    Gold-standard papers bucketed by the quarter they were SUBMITTED in.

    Submission dates come from arXiv and are cached, because they never change
    and the lookup is ~2 batched calls. Bucketing by submission date matches the
    DATING rule the discovery prompt applies, so the two are directly comparable.
    """
    import urllib.parse
    import urllib.request
    import xml.etree.ElementTree as ET

    dates: Dict[str, Dict[str, str]] = {}
    if GOLD_DATES.exists():
        try:
            dates = json.loads(GOLD_DATES.read_text())
        except (OSError, ValueError):
            dates = {}

    gold = json.loads(GOLD_LIST.read_text())
    wanted = [re.sub(r"v\d+$", "", p["arxiv_id"]) for p in gold if p.get("arxiv_id")]
    missing = [a for a in wanted if a not in dates]

    if missing:
        logger.info("Fetching submission dates for %d gold paper(s)...", len(missing))
        atom = "{http://www.w3.org/2005/Atom}"
        for start in range(0, len(missing), 50):
            batch = missing[start : start + 50]
            url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
                {"id_list": ",".join(batch), "max_results": len(batch)}
            )
            try:
                root = ET.fromstring(urllib.request.urlopen(url, timeout=120).read())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not fetch gold dates: %s", exc)
                break
            for entry in root.findall(atom + "entry"):
                aid = re.sub(
                    r"v\d+$", "", (entry.findtext(atom + "id") or "").rsplit("/", 1)[-1]
                )
                dates[aid] = {
                    "date": (entry.findtext(atom + "published") or "")[:10],
                    "title": " ".join((entry.findtext(atom + "title") or "").split()),
                }
            time.sleep(4)
        GOLD_DATES.write_text(json.dumps(dates, indent=1, sort_keys=True) + "\n")

    buckets: Dict[str, List[Tuple[str, str]]] = {}
    for aid, info in dates.items():
        date = info.get("date") or ""
        if len(date) < 7:
            continue
        year, month = int(date[:4]), int(date[5:7])
        buckets.setdefault(quarter_name(year, (month - 1) // 3 + 1), []).append(
            (aid, info.get("title", ""))
        )
    return buckets


def cmd_recall(args) -> int:
    """
    Compare what discovery listed against the gold standard, per quarter.

    Answers the question "which quarters are worth re-running": a quarter with
    several gold papers missing got a poor draw, whereas one with none missing
    is probably fine however few papers it listed.
    """
    buckets = gold_by_quarter()
    listed_ids: Dict[str, set] = {}
    for path in discovery_files():
        name = quarter_from_path(path)
        if name is None:
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if payload.get("papers"):
            listed_ids[name] = {
                re.sub(r"v\d+$", "", str(p.get("arxiv_id") or ""))
                for p in payload["papers"]
                if p.get("arxiv_id")
            }

    print(f"{'quarter':10} {'listed':>7} {'gold':>5} {'found':>6} {'recall':>7}")
    print("-" * 40)
    total_gold = total_found = total_listed = 0
    worst: List[Tuple[int, str]] = []
    for name in sorted(listed_ids):
        gold_here = buckets.get(name, [])
        found = sum(1 for aid, _ in gold_here if aid in listed_ids[name])
        total_gold += len(gold_here)
        total_found += found
        total_listed += len(listed_ids[name])
        rate = f"{100 * found / len(gold_here):.0f}%" if gold_here else "-"
        print(f"{name:10} {len(listed_ids[name]):>7} {len(gold_here):>5} {found:>6} {rate:>7}")
        if gold_here and found < len(gold_here):
            worst.append((len(gold_here) - found, name))
    print("-" * 40)
    overall = f"{100 * total_found / total_gold:.0f}%" if total_gold else "-"
    print(f"{'total':10} {total_listed:>7} {total_gold:>5} {total_found:>6} {overall:>7}")

    if worst:
        print("\nquarters to consider re-running (most gold papers missed first):")
        for gap, name in sorted(worst, reverse=True):
            print(f"   {name}: {gap} gold paper(s) missing")
            if args.verbose:
                for aid, title in sorted(buckets.get(name, [])):
                    if aid not in listed_ids[name]:
                        print(f"        {aid}  {title[:62]}")
    return 0


def cmd_run(args) -> int:
    from src.pipeline import ExtractionPipeline

    names = _target_quarters(args)

    # One pipeline for the whole run so the value-validation, resource and
    # property-description caches are shared across every paper.
    pipeline = ExtractionPipeline()

    # The classifier exists to filter a firehose of arXiv categories down to
    # LLM/VLM papers. In this workflow discovery has ALREADY made that decision,
    # so it is off unless asked for.
    #
    # --classify runs it in ADVISORY mode: the verdict is recorded in the ledger
    # but a rejection no longer discards the paper. Gating would, and the
    # classifier is not reliable enough for that here — it rejected "Language
    # Models are Few-Shot Learners" (GPT-3) as OUT_OF_DOMAIN. Use
    # --classify-strict for the old gating behaviour.
    if not (args.classify or args.classify_strict):
        pipeline.paper_classifier = None
        logger.info("Classifier disabled — discovery already selected these papers")
    elif args.classify_strict:
        logger.warning(
            "Classifier in STRICT mode — papers it rejects will NOT be extracted"
        )
    else:
        pipeline.classification_advisory = True
        logger.info("Classifier in advisory mode — verdicts recorded, nothing discarded")

    index = load_index()
    ledger = load_ledger()

    for name in names:
        payload, problems = load_discovery(name)
        if payload is None:
            logger.error("%s: %s", name, "; ".join(problems))
            continue
        papers = payload.get("papers") or []
        if not papers:
            logger.info("%s: nothing listed, skipping", name)
            continue
        blocking = [p for p in problems if "missing title" in p or "no arxiv_id" in p]
        if blocking and not args.force:
            logger.error(
                "%s: %d unusable entr(ies) — fix them or pass --force to skip them:\n   %s",
                name,
                len(blocking),
                "\n   ".join(blocking),
            )
            continue

        _run_quarter(pipeline, name, payload, index, ledger, args)

    return 0


def _ledger_row(quarter, paper, title, result, contribution_ids, status, error=""):
    """Build one ledger row. Shared so every outcome records the same columns."""
    models = result.get("extraction_data") or []
    orkg = result.get("orkg_results") or {}
    classification = result.get("classification") or {}
    return {
        "quarter": quarter,
        "arxiv_id": paper.get("arxiv_id") or "",
        "pdf_url": paper.get("pdf_url") or "",
        "title": title,
        "models_extracted": len(models),
        "model_names": "; ".join(
            str(m.get("model_name")) for m in models if isinstance(m, dict)
        ),
        "orkg_paper_id": orkg.get("paper_id") or "",
        # Bare ids, semicolon-separated — one per contribution tab. A tab lives
        # at /papers/<paper>/<contribution>, so the id is what is needed.
        "contribution_ids": "; ".join(str(c) for c in (contribution_ids or []) if c),
        "orkg_url": orkg_paper_url(orkg.get("paper_id")),
        "classification": (
            "" if not classification
            else ("accepted" if classification.get("is_valid") else "rejected")
        ),
        "classification_reason": classification.get("reason", ""),
        "status": status,
        "error": error,
        "processed_at": datetime.now().isoformat(timespec="seconds"),
    }


def _our_contribution_ids(client, paper_id, model_names):
    """
    Find the contributions WE added to an existing ORKG paper.

    A paper can already carry other people's contributions — ImageBERT's paper
    has "Dataset", "Method" and "Modality" from an earlier curator — so the
    paper's full contribution list is not ours. Matching on the model labels we
    extracted picks out only the tabs this project created.
    """
    if not paper_id or not model_names:
        return []
    wanted = {str(n).strip().lower() for n in model_names if n}
    try:
        paper = client.get_paper(paper_id) or {}
    except Exception as exc:  # noqa: BLE001 - a backfill must not break a run
        logger.debug("Could not read paper %s for contribution ids: %s", paper_id, exc)
        return []
    return [
        c.get("id")
        for c in (paper.get("contributions") or [])
        if isinstance(c, dict)
        and c.get("id")
        and str(c.get("label", "")).strip().lower() in wanted
    ]


def _upload_client(pipeline):
    """The ORKG client, only if the pipeline already has one (never forces a connect)."""
    return getattr(pipeline, "_orkg_client", None)


def _record_completed(ledger, quarter, paper, title, extraction_file, client=None):
    """
    Put an already-extracted paper into the ledger.

    Reads the saved extraction rather than re-running anything, so a resumed run
    still ends up with a complete CSV instead of one covering only the papers it
    happened to touch.

    The contribution ids are the one thing the saved file may not know: when a
    paper is appended to an existing ORKG entry, the ids are assigned after that
    file was written. So when a client is available they are resolved from ORKG
    by matching our model labels — the paper's own contribution list also holds
    other people's tabs.
    """
    key = paper_key(paper)
    try:
        result = json.loads(extraction_file.read_text())
    except (OSError, ValueError) as exc:
        logger.warning("Could not read %s for the ledger: %s", extraction_file.name, exc)
        return

    orkg = result.get("orkg_results") or {}
    paper_id = orkg.get("paper_id")
    contribution_ids = orkg.get("new_contribution_ids") or []

    if not contribution_ids and paper_id and client is not None:
        model_names = [
            m.get("model_name")
            for m in (result.get("extraction_data") or [])
            if isinstance(m, dict)
        ]
        contribution_ids = _our_contribution_ids(client, paper_id, model_names)
        if contribution_ids:
            logger.info(
                "Backfilled contribution id(s) for %s: %s",
                paper_id, ", ".join(contribution_ids),
            )

    ledger[key] = _ledger_row(
        quarter, paper, title, result, contribution_ids, "extracted"
    )


def _run_quarter(pipeline, name: str, payload: Dict, index: Dict, ledger: Dict, args) -> None:
    folder = quarter_dir(name)
    extraction_dir = folder / "extraction"
    extraction_dir.mkdir(parents=True, exist_ok=True)
    papers = payload["papers"]

    logger.info("=" * 70)
    logger.info("%s — %d paper(s) listed", name, len(papers))
    logger.info("=" * 70)

    # Run totals for the closing log line only. The durable record of what
    # happened is the ledger; nothing here is written to disk.
    counts = {"extracted": 0, "skipped": 0, "failed": 0}

    # Guards against the same paper appearing twice in one quarter's list. The
    # cross-quarter index cannot catch that: it is only written after a paper is
    # processed, and both copies carry the current quarter.
    seen_this_run: set = set()

    # Same normalisation as paper_key so "2001.07966v3", "arXiv:2001.07966" and
    # "2001.07966" all name the same paper.
    skip_keys = {
        re.sub(r"v\d+$", "", str(s).strip().replace("arXiv:", "").replace("arxiv:", ""))
        for s in (getattr(args, "skip", None) or [])
    }
    if skip_keys:
        logger.info(
            "Excluding %d paper(s) via --skip: %s",
            len(skip_keys), ", ".join(sorted(skip_keys)),
        )

    # Mirror image of --skip: process ONLY these papers and pass over the rest.
    # Lets a handful of papers be re-run by id without naming their quarters,
    # which matters for recovery work where the failures are scattered.
    only_keys = {
        re.sub(r"v\d+$", "", str(s).strip().replace("arXiv:", "").replace("arxiv:", ""))
        for s in (getattr(args, "only", None) or [])
    }

    for position, paper in enumerate(papers, 1):
        key = paper_key(paper)

        # Filtered out before any bookkeeping: a paper the caller did not ask
        # for should leave no trace in the ledger.
        if only_keys and key not in only_keys:
            continue

        title = (paper.get("title") or "").strip()
        label = f"[{position}/{len(papers)}] {title[:56]}"

        # A key can be derived from the title alone, but the PIPELINE needs a
        # real identifier. Without one there is nothing to fetch, so check for
        # it explicitly rather than relying on the key being present.
        has_identifier = bool(
            (paper.get("arxiv_id") or "").strip() or (paper.get("pdf_url") or "").strip()
        )
        if not title or not has_identifier:
            missing = "title" if not title else "arxiv_id/pdf_url"
            counts["failed"] += 1
            logger.error("%s -> unusable entry (missing %s)", label, missing)
            # An entry with neither a title nor an identifier cannot be keyed,
            # so the log above is the only record it can have.
            if key:
                ledger[key] = _ledger_row(
                    name, paper, title, {}, [], "failed", f"missing {missing}"
                )
                save_ledger(ledger)
            continue

        # Explicitly excluded by the caller. Checked before --redo so that
        # "redo everything except X" is expressible; the paper's existing
        # results still go into the ledger so the CSV stays complete.
        if key in skip_keys:
            counts["skipped"] += 1
            logger.info("%s -> skipped (--skip)", label)
            target_existing = extraction_dir / f"{re.sub(r'[^A-Za-z0-9._-]', '_', key)}.json"
            if target_existing.exists():
                _record_completed(
                    ledger, name, paper, title, target_existing,
                    client=_upload_client(pipeline) if args.upload else None,
                )
                save_ledger(ledger)
            continue

        if key in seen_this_run:
            counts["skipped"] += 1
            logger.info("%s -> duplicate within %s", label, name)
            continue
        seen_this_run.add(key)

        target = extraction_dir / f"{re.sub(r'[^A-Za-z0-9._-]', '_', key)}.json"

        if target.exists() and not args.redo:
            counts["skipped"] += 1
            logger.info("%s -> already done", label)
            # A skipped paper is still part of the collection, so it belongs in
            # the ledger. Rebuild its row from the saved extraction rather than
            # leaving whatever an earlier run happened to write — that is how a
            # paper processed before a column existed gets backfilled.
            _record_completed(
                ledger, name, paper, title, target,
                client=_upload_client(pipeline) if args.upload else None,
            )
            save_ledger(ledger)
            continue

        seen_in = index.get(key, {}).get("quarter")
        if seen_in and seen_in != name and not args.redo:
            counts["skipped"] += 1
            logger.info("%s -> duplicate of %s", label, seen_in)
            continue

        if args.dry_run:
            logger.info("%s -> would process", label)
            continue

        logger.info("%s -> processing", label)
        try:
            if paper.get("arxiv_id"):
                result = pipeline.process_paper(
                    re.sub(r"v\d+$", "", str(paper["arxiv_id"]).strip()),
                    save_intermediate=False,
                    update_orkg=args.upload,
                )
            else:
                result = pipeline.process_paper_from_pdf_url(
                    paper["pdf_url"],
                    title,
                    save_intermediate=False,
                    update_orkg=args.upload,
                )
        except Exception as exc:  # noqa: BLE001 - one bad paper must not stop the quarter
            counts["failed"] += 1
            logger.exception("%s -> crashed", label)
            ledger[key] = _ledger_row(
                name, paper, title, {}, [], "failed", f"{type(exc).__name__}: {exc}"
            )
            save_ledger(ledger)
            continue

        if result.get("status") != "completed":
            error = result.get("error", "pipeline did not complete")
            counts["failed"] += 1
            logger.error("%s -> failed: %s", label, error)
            ledger[key] = _ledger_row(name, paper, title, result, [], "failed", error)
            save_ledger(ledger)
        else:
            target.write_text(json.dumps(result, indent=1, default=str) + "\n")
            models = result.get("extraction_data") or []
            orkg = result.get("orkg_results") or {}
            counts["extracted"] += 1
            ledger[key] = _ledger_row(
                name, paper, title, result,
                orkg.get("new_contribution_ids"), "extracted",
            )
            save_ledger(ledger)
            index[key] = {
                "quarter": name,
                "title": title,
                "processed_at": datetime.now().isoformat(),
                "orkg_paper_id": orkg.get("paper_id"),
            }
            save_index(index)
            logger.info("%s -> %d model(s)%s", label, len(models),
                        f", ORKG {orkg.get('paper_id')}" if orkg.get("paper_id") else "")

    logger.info(
        "%s done — %d extracted, %d skipped, %d failed",
        name, counts["extracted"], counts["skipped"], counts["failed"],
    )


def _target_quarters(args) -> List[str]:
    if getattr(args, "quarter", None):
        return [quarter_name(*args.quarter)]
    return [
        quarter_name(y, q)
        for y, q in all_quarters()
        if quarter_dir(quarter_name(y, q)).exists()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="create quarter folders and stub files")
    p_init.add_argument("--until", type=parse_quarter, default=None, metavar="YYYY-Qn")
    p_init.set_defaults(func=cmd_init)

    p_val = sub.add_parser("validate", help="check discovery.json without processing")
    p_val.add_argument("--quarter", type=parse_quarter, metavar="YYYY-Qn")
    p_val.add_argument("-v", "--verbose", action="store_true")
    p_val.add_argument("--check-arxiv", action="store_true",
                       help="verify ids resolve, titles match, and dates fall in the quarter")
    p_val.set_defaults(func=cmd_validate)

    p_run = sub.add_parser("run", help="extract the papers of one or all quarters")
    p_run.add_argument("--quarter", type=parse_quarter, metavar="YYYY-Qn")
    p_run.add_argument("--all", action="store_true", help="every quarter with a folder")
    p_run.add_argument("--upload", action="store_true", help="also upload to ORKG")
    p_run.add_argument("--redo", action="store_true", help="re-process papers already done")
    p_run.add_argument("--skip", nargs="*", default=[], metavar="ID",
                       help="never process these papers (arXiv ids), even with --redo; "
                            "their existing results are still recorded in the ledger")
    p_run.add_argument("--only", nargs="*", default=[], metavar="ID",
                       help="process ONLY these papers (arXiv ids) and pass over the rest; "
                            "combine with --all to re-run scattered papers without "
                            "naming their quarters")
    p_run.add_argument("--dry-run", action="store_true", help="list what would run")
    p_run.add_argument("--force", action="store_true", help="proceed despite bad entries")
    p_run.add_argument("--classify", action="store_true",
                       help="run the domain classifier and record its verdict, without "
                            "letting a rejection discard the paper")
    p_run.add_argument("--classify-strict", action="store_true",
                       help="run the classifier as a GATE: rejected papers are not extracted")
    p_run.set_defaults(func=cmd_run)

    p_recall = sub.add_parser("recall",
                              help="compare discovery against the gold standard per quarter")
    p_recall.add_argument("-v", "--verbose", action="store_true",
                          help="list the missing gold papers")
    p_recall.set_defaults(func=cmd_recall)

    p_status = sub.add_parser("status", help="progress across all quarters")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    if args.command == "run" and not args.quarter and not args.all:
        parser.error("run needs --quarter YYYY-Qn or --all")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
