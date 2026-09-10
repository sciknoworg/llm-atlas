"""
Layer 2 of value validation: a persistent verdict cache.

Extracted values repeat heavily across papers — "Adam", "English", "Apache 2.0",
"BPE", "Transformer". Measured on this project's own extraction corpus (750
result files, 2,223 models): 33,760 value instances but only 7,009 distinct
(field, value) pairs, i.e. **79% of values are repeats**. Validating them again
on every run would be almost entirely wasted work.

The cache persists to disk so the saving carries across runs, not just within
one batch. At steady state (last 100 papers of that corpus) a paper introduces
about 10 previously-unseen values, so the LLM layer sees roughly ten short
strings per paper instead of several hundred.

Keys are (field_name, casefolded value). Field is part of the key because the
same string can be valid for one property and wrong for another — "Transformer"
is a fine pretraining_architecture but a poor model_family.
"""

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "value_validation_cache.json"
)

# Bump when the validation prompt or rules change in a way that invalidates
# previously stored verdicts. Entries written under an older version are ignored
# on load, so a prompt change does not silently keep stale judgements alive.
#
# 2: value-shape guidance (_VALUE_GUIDANCE) is now appended to the property
#    description, so the model is judging against different instructions than
#    it was for version-1 verdicts.
CACHE_VERSION = 2


class ValueCache:
    """Disk-backed store of per-(field, value) validation verdicts."""

    def __init__(self, path: Optional[Path] = None, enabled: bool = True):
        self.path = Path(path) if path else DEFAULT_CACHE_PATH
        self.enabled = enabled
        self._entries: Dict[str, Dict] = {}
        self._dirty = False
        # Guards _entries. The pipeline is single-threaded today, but a cache
        # that corrupts itself under concurrency is a nasty thing to debug later.
        self._lock = threading.Lock()
        if self.enabled:
            self._load()

    # ASCII unit separator: cannot occur in a field name or an extracted value,
    # so "field<US>value" is unambiguous and still readable in the JSON file.
    _KEY_SEP = "\x1f"

    @classmethod
    def _key(cls, field_name: str, value: str) -> str:
        return f"{field_name}{cls._KEY_SEP}{str(value).strip().casefold()}"

    def _load(self) -> None:
        if not self.path.exists():
            logger.debug("No value cache at %s (starting empty)", self.path)
            return
        try:
            payload = json.loads(self.path.read_text())
        except (OSError, ValueError) as exc:
            # A corrupt cache must never break a pipeline run — it is an
            # optimisation, not a source of truth. Start empty and move on.
            logger.warning("Could not read value cache %s: %s — starting empty", self.path, exc)
            return

        if payload.get("version") != CACHE_VERSION:
            logger.info(
                "Value cache at %s was written by version %s (current %s) — ignoring it",
                self.path,
                payload.get("version"),
                CACHE_VERSION,
            )
            return

        self._entries = payload.get("entries") or {}
        logger.info("Loaded %d cached value verdicts from %s", len(self._entries), self.path)

    def get(self, field_name: str, value: str) -> Optional[Tuple[bool, Optional[str]]]:
        """Return (keep, reason) for a previously judged value, or None if unseen."""
        if not self.enabled:
            return None
        with self._lock:
            entry = self._entries.get(self._key(field_name, value))
        if entry is None:
            return None
        return bool(entry.get("keep")), entry.get("reason")

    def put(self, field_name: str, value: str, keep: bool, reason: Optional[str] = None) -> None:
        """Record a verdict. Call save() to persist."""
        if not self.enabled:
            return
        with self._lock:
            self._entries[self._key(field_name, value)] = {
                "field": field_name,
                "value": str(value).strip(),
                "keep": bool(keep),
                "reason": reason,
            }
            self._dirty = True

    def save(self) -> None:
        """
        Write the cache to disk if anything changed.

        Writes via a temporary file and an atomic rename so an interrupted run
        cannot leave a half-written JSON file behind — that would be silently
        discarded on the next load, throwing away every verdict.
        """
        if not self.enabled or not self._dirty:
            return
        with self._lock:
            payload = {
                "version": CACHE_VERSION,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "entries": self._entries,
            }
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(self.path.suffix + ".tmp")
                tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
                tmp.replace(self.path)
                self._dirty = False
                logger.debug("Saved %d value verdicts to %s", len(self._entries), self.path)
            except OSError as exc:
                logger.warning("Could not save value cache to %s: %s", self.path, exc)

    def __len__(self) -> int:
        return len(self._entries)
