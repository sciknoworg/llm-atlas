#!/usr/bin/env python3
"""
Count arXiv papers per month per category, to scope how much material the
extraction pipeline would have to process.

Default scope: cs.AI, cs.CL and cs.CV, from 2020-01 to the last complete month.

USAGE
-----
    python scripts/arxiv_monthly_volume.py
    
"""

import argparse
import calendar
import csv
import json
import logging
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

API_URL = "https://export.arxiv.org/api/query"
OPENSEARCH_NS = {"opensearch": "http://a9.com/-/spec/opensearch/1.1/"}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = PROJECT_ROOT / "data" / "arxiv_monthly_counts.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "arxiv_monthly_volume.csv"

DEFAULT_CATEGORIES = ["cs.AI", "cs.CL", "cs.CV"]
UNION_KEY = "union"

# arXiv's stated minimum interval between API calls.
DEFAULT_DELAY = 3.0


def month_range(start: Tuple[int, int], end: Tuple[int, int]) -> Iterator[Tuple[int, int]]:
    """Yield (year, month) from start to end inclusive."""
    year, month = start
    while (year, month) <= end:
        yield year, month
        month += 1
        if month > 12:
            year, month = year + 1, 1


def month_window(year: int, month: int) -> str:
    """
    Build the arXiv submittedDate range covering one whole calendar month.

    The API wants YYYYMMDDHHMM on both ends and treats the range as inclusive,
    so the window runs from 00:00 on the 1st to 23:59 on the true last day —
    calendar.monthrange supplies that, which is what makes February and leap
    years correct without special-casing.
    """
    last_day = calendar.monthrange(year, month)[1]
    return f"submittedDate:[{year}{month:02d}010000 TO {year}{month:02d}{last_day}2359]"


def series_cache_name(name: str, categories: List[str]) -> str:
    """
    Cache name for one series.

    A plain category caches under its own name, but the union is only meaningful
    together with the categories it spans: `union` alone would let a
    `--categories cs.AI cs.CL` run read the three-category union written by an
    earlier default run, reporting more distinct papers than the two categories
    contain. Naming the set keeps every scope in its own namespace.
    """
    if name != UNION_KEY:
        return name
    return f"{UNION_KEY}({'+'.join(sorted(categories))})"


def build_query(categories: List[str], year: int, month: int) -> str:
    """
    Compose the search_query for one cell of the table.

    A single category yields `cat:cs.AI AND submittedDate:[...]`; several yield
    a parenthesised OR, which arXiv deduplicates so a cross-listed paper is
    counted once.
    """
    if len(categories) == 1:
        category_clause = f"cat:{categories[0]}"
    else:
        category_clause = "(" + " OR ".join(f"cat:{c}" for c in categories) + ")"
    return f"{category_clause} AND {month_window(year, month)}"


class Throttle:
    """
    Adaptive pacing between arXiv calls.

    arXiv documents a 3-second minimum, but a sustained run of a few hundred
    requests gets HTTP 429 anyway — an observed full run started being
    throttled after ~18 requests and eventually failed outright. So the delay
    is not fixed: every 429 widens it and a long clean streak narrows it back
    toward the floor. This keeps a long run alive without hand-tuning --delay.
    """

    def __init__(self, base: float, ceiling: float = 30.0):
        self.floor = base
        self.current = base
        self.ceiling = ceiling
        self._clean_streak = 0

    def wait(self) -> None:
        time.sleep(self.current)

    def penalise(self, retry_after: Optional[float] = None) -> None:
        self._clean_streak = 0
        widened = max(self.current * 1.5, (retry_after or 0) + 1.0)
        self.current = min(widened, self.ceiling)
        logger.warning("  throttled by arXiv — inter-request delay now %.1fs", self.current)

    def reward(self) -> None:
        self._clean_streak += 1
        if self._clean_streak >= 20 and self.current > self.floor:
            self.current = max(self.floor, self.current * 0.8)
            self._clean_streak = 0
            logger.info("  20 clean requests — easing delay to %.1fs", self.current)


def _retry_after_seconds(response: Optional[requests.Response]) -> Optional[float]:
    """Read a Retry-After header if the server sent one."""
    if response is None:
        return None
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fetch_total(
    query: str,
    session: requests.Session,
    throttle: Optional["Throttle"] = None,
    retries: int = 6,
    timeout: int = 60,
) -> Optional[int]:
    """
    Return the number of papers matching `query`, or None if the call failed.

    Reads <opensearch:totalResults> and never touches the entries, so the
    response size does not depend on how many papers matched.

    Note max_results=1 rather than 0: arXiv answers max_results=0 with an
    HTTP 500.

    Rate-limit responses (429/503) are retried with exponential backoff, honour
    Retry-After when present, and permanently widen the caller's throttle so the
    rest of the run goes slower instead of hammering into the same wall.
    """
    params = {"search_query": query, "start": 0, "max_results": 1}

    for attempt in range(1, retries + 1):
        response = None
        try:
            response = session.get(API_URL, params=params, timeout=timeout)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            node = root.find("opensearch:totalResults", OPENSEARCH_NS)
            if node is None or node.text is None:
                raise ValueError("response contained no totalResults element")
            if throttle:
                throttle.reward()
            return int(node.text)
        except (requests.RequestException, ET.ParseError, ValueError) as exc:
            status = getattr(response, "status_code", None)
            rate_limited = status in (429, 503)
            if rate_limited and throttle:
                throttle.penalise(_retry_after_seconds(response))

            if attempt == retries:
                logger.error("  giving up after %d attempts: %s", retries, exc)
                return None

            wait = _retry_after_seconds(response) or DEFAULT_DELAY * (2 ** (attempt - 1))
            wait = min(wait, 120.0)
            logger.warning(
                "  attempt %d/%d failed (%s) — retrying in %.0fs", attempt, retries, exc, wait
            )
            time.sleep(wait)
    return None


class CountCache:
    """
    Disk cache of {category|YYYY-MM: count}.

    Worth having because a full run takes ~16 minutes of deliberately throttled
    requests: without it, every re-run or interrupted run pays that again. Past
    months never change, so cached values stay valid indefinitely.
    """

    def __init__(self, path: Path, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self.data: Dict[str, int] = {}
        if enabled and path.exists():
            try:
                self.data = json.loads(path.read_text()).get("counts", {})
                # Entries written before the union key named its category set
                # cannot be attributed to a scope, so they are dropped rather
                # than risk serving one query's union to another. They are
                # removed from the file on the next save; the affected cells
                # are simply re-fetched once.
                legacy = [k for k in self.data if k.startswith(f"{UNION_KEY}|")]
                for stale in legacy:
                    del self.data[stale]
                if legacy:
                    logger.warning(
                        "Discarded %d un-scoped '%s|' cache entries — these will be re-fetched",
                        len(legacy), UNION_KEY,
                    )
                logger.info("Loaded %d cached counts from %s", len(self.data), path)
            except (OSError, ValueError) as exc:
                logger.warning("Ignoring unreadable cache %s: %s", path, exc)

    @staticmethod
    def key(category: str, year: int, month: int) -> str:
        return f"{category}|{year}-{month:02d}"

    def get(self, category: str, year: int, month: int) -> Optional[int]:
        return self.data.get(self.key(category, year, month)) if self.enabled else None

    def put(self, category: str, year: int, month: int, count: int) -> None:
        self.data[self.key(category, year, month)] = count

    def save(self) -> None:
        """
        Persist the cache.

        Written via a temp file and atomic rename: a run interrupted mid-write
        would otherwise leave truncated JSON that the next run discards, losing
        everything. Learned the hard way — an earlier version only saved at the
        very end, and a killed run threw away 224 successfully fetched counts
        (~11 minutes of throttled requests). save() is now called after every
        completed month.
        """
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps({"counts": self.data}, indent=1, sort_keys=True) + "\n")
            tmp.replace(self.path)
        except OSError as exc:
            logger.warning("Could not save cache to %s: %s", self.path, exc)


def last_complete_month(today: Optional[date] = None) -> Tuple[int, int]:
    """
    The most recent month that has fully elapsed.

    The current month is excluded because its count would be a partial figure
    that changes every day — misleading in a table of monthly totals.
    """
    today = today or date.today()
    return (today.year - 1, 12) if today.month == 1 else (today.year, today.month - 1)


def parse_month(text: str) -> Tuple[int, int]:
    try:
        year, month = text.split("-")
        year, month = int(year), int(month)
        if not 1 <= month <= 12:
            raise ValueError
        return year, month
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError(f"expected YYYY-MM, got {text!r}")


def collect(
    categories: List[str],
    start: Tuple[int, int],
    end: Tuple[int, int],
    include_union: bool,
    cache: CountCache,
    delay: float,
) -> List[Dict]:
    """Query every (series, month) cell and return one row per month."""
    series = list(categories) + ([UNION_KEY] if include_union and len(categories) > 1 else [])
    months = list(month_range(start, end))

    # Cache names, not display names: the union's entry is scoped to the exact
    # category set so a narrower run cannot read a wider run's answer.
    cache_names = {s: series_cache_name(s, categories) for s in series}

    todo = sum(
        1 for y, m in months for s in series if cache.get(cache_names[s], y, m) is None
    )
    logger.info(
        "%d month(s) x %d series = %d cell(s); %d need fetching (~%.0f min at %.1fs/request)",
        len(months),
        len(series),
        len(months) * len(series),
        todo,
        todo * delay / 60,
        delay,
    )

    rows: List[Dict] = []
    session = requests.Session()
    session.headers.update({"User-Agent": "llm-atlas-volume-scoping (arXiv API client)"})
    throttle = Throttle(delay)
    fetched = 0
    failed = 0

    try:
        for year, month in months:
            row: Dict = {"month": f"{year}-{month:02d}"}
            month_had_fetch = False

            for name in series:
                cached = cache.get(cache_names[name], year, month)
                if cached is not None:
                    row[name] = cached
                    continue

                if fetched:
                    throttle.wait()  # only between real calls, never after a cache hit
                query = build_query(categories if name == UNION_KEY else [name], year, month)
                count = fetch_total(query, session, throttle=throttle)
                fetched += 1
                month_had_fetch = True

                if count is None:
                    logger.error("  %s %s-%02d: FAILED", name, year, month)
                    row[name] = None
                    failed += 1
                    continue

                cache.put(cache_names[name], year, month, count)
                row[name] = count
                logger.info("  %s %s-%02d: %d", name, year, month, count)

            # A total is only meaningful when every category answered. Summing
            # the survivors would emit a subtotal that reads as a complete
            # figure, which is worse than an empty cell.
            counts = [row.get(c) for c in categories]
            row["sum_of_categories"] = (
                sum(counts) if all(c is not None for c in counts) else None
            )
            rows.append(row)

            # Persist after every month that cost real requests, so an
            # interruption loses at most one month rather than the whole run.
            if month_had_fetch:
                cache.save()
    except KeyboardInterrupt:
        # Re-raised so main() can skip writing the CSV: silently publishing the
        # rows gathered so far would replace a complete file with a truncated
        # one while still exiting successfully. The cache is saved either way,
        # so a re-run resumes rather than restarting.
        cache.save()
        logger.warning("Interrupted — cached counts up to this point are saved")
        raise
    finally:
        cache.save()

    if failed:
        logger.warning(
            "%d cell(s) failed and were NOT cached — re-run to retry just those", failed
        )
    return rows


def write_csv(rows: List[Dict], categories: List[str], include_union: bool, path: Path) -> None:
    columns = ["month"] + categories + ["sum_of_categories"]
    if include_union and len(categories) > 1:
        columns.append(UNION_KEY)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d rows to %s", len(rows), path)


def print_summary(rows: List[Dict], categories: List[str], include_union: bool) -> None:
    """Per-year totals — the month-by-month detail lives in the CSV."""
    if not rows:
        return
    series = list(categories) + ([UNION_KEY] if include_union and len(categories) > 1 else [])

    by_year: Dict[str, Dict[str, int]] = {}
    # A failed cell makes its year's total a subtotal. Tracking which series
    # are short lets the year be flagged rather than silently under-reported.
    incomplete: Dict[str, set] = {}
    collected: Dict[str, Dict[str, int]] = {}
    for row in rows:
        year = row["month"][:4]
        bucket = by_year.setdefault(year, {s: 0 for s in series})
        incomplete.setdefault(year, set())
        seen = collected.setdefault(year, {s: 0 for s in series})
        for s in series:
            if row.get(s) is None:
                incomplete[year].add(s)
            else:
                bucket[s] += row[s]
                seen[s] += 1

    width = max(len(s) for s in series) + 2
    print("\nPapers per year")
    print("year  " + "".join(f"{s:>{width}}" for s in series) + f"{'sum':>{width}}")
    print("-" * (6 + width * (len(series) + 1)))
    for year in sorted(by_year):
        bucket = by_year[year]
        missing = incomplete[year]
        total = (
            sum(bucket[c] for c in categories)
            if not (missing & set(categories))
            else None
        )
        def cell(s: str) -> str:
            if s not in missing:
                return format(bucket[s], ",")
            # Nothing came back for this series all year: "~0" would read as
            # "about zero papers" rather than "no data".
            return "n/a" if collected[year][s] == 0 else "~" + format(bucket[s], ",")

        cells = "".join(f"{cell(s):>{width}}" for s in series)
        total_cell = f"{format(total, ','):>{width}}" if total is not None else f"{'n/a':>{width}}"
        print(f"{year}  " + cells + total_cell)

    if any(incomplete.values()):
        print(
            "\n~ marks a partial figure: at least one month in that series failed "
            "and is missing from the total. Re-run to retry just those cells."
        )

    if include_union and len(categories) > 1:
        if any(incomplete.values()):
            print(
                "\nCross-listing overlap not reported: some cells failed, so the "
                "summed and distinct totals are not comparable."
            )
            return
        grand_sum = sum(sum(b[c] for c in categories) for b in by_year.values())
        grand_union = sum(b[UNION_KEY] for b in by_year.values())
        print(
            f"\nSummed categories {grand_sum:,} vs {grand_union:,} distinct papers — "
            f"{grand_sum - grand_union:,} counted more than once "
            f"({100 * (grand_sum - grand_union) / grand_sum:.1f}% cross-listed). "
            f"Use the union column for 'how many papers'."
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--start", type=parse_month, default=(2020, 1), metavar="YYYY-MM")
    parser.add_argument(
        "--end", type=parse_month, default=None, metavar="YYYY-MM",
        help="default: last complete month",
    )
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument(
        "--no-union", action="store_true",
        help="skip the deduplicated across-categories count (saves 1 request per month)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--refresh", action="store_true", help="ignore cached counts")
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_DELAY,
        help=f"seconds between API calls (default {DEFAULT_DELAY}; arXiv asks for >= 3)",
    )
    args = parser.parse_args()

    end = args.end or last_complete_month()
    if args.start > end:
        parser.error(f"--start {args.start} is after --end {end}")
    if args.delay < 3.0:
        logger.warning("delay %.1fs is below arXiv's requested 3s minimum", args.delay)

    include_union = not args.no_union
    logger.info(
        "Categories: %s | %s..%s | union=%s",
        ", ".join(args.categories),
        f"{args.start[0]}-{args.start[1]:02d}",
        f"{end[0]}-{end[1]:02d}",
        include_union,
    )

    cache = CountCache(args.cache, enabled=not args.refresh)
    try:
        rows = collect(args.categories, args.start, end, include_union, cache, args.delay)
    except KeyboardInterrupt:
        # Deliberately leaves --output untouched. Overwriting a complete CSV
        # with the prefix gathered before the interrupt, and exiting 0, would
        # look like a successful run that quietly lost most of its rows.
        logger.warning(
            "Interrupted before completion — %s was left unchanged. "
            "Re-run to resume from the cache.", args.output,
        )
        return 130

    if not rows:
        logger.error("No data collected")
        return 1

    write_csv(rows, args.categories, include_union, args.output)
    print_summary(rows, args.categories, include_union)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
