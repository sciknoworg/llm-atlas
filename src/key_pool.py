"""
Rotating pool of API keys.

A single KISSKI key has per-minute/hour/day quotas. A 290-paper batch run with
chunked extraction plus one classifier call and one semantic-validation call per
paper will exhaust one, and the run then spends its time sleeping through
exponential backoff. With more than one key available, a rate-limited key can be
parked and the next one used immediately.

Two rules make this behave:

* A key that returns 429 is put in COOLDOWN rather than discarded. Quotas
  refill, so a key exhausted at minute 3 is usable again later in a run that
  lasts hours.
* Rotation is not free retries. The caller keeps its own attempt budget; this
  class only answers "is there another key I could use right now?".
"""

import logging
import os
import time
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# How long a rate-limited key is left alone before being tried again. KISSKI
# publishes per-minute, per-hour and per-day limits; five minutes clears a
# per-minute exhaustion without writing off a key for an hourly one.
DEFAULT_COOLDOWN_SECONDS = 300.0


class KeyPool:
    """An ordered set of API keys with per-key cooldown after rate limiting."""

    def __init__(
        self, keys: Sequence[str], cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS
    ):
        # Order preserved, duplicates dropped: the same key listed twice would
        # otherwise look like a fallback and silently retry the exhausted quota.
        seen: Dict[str, None] = {}
        for key in keys:
            cleaned = (key or "").strip()
            if cleaned:
                seen.setdefault(cleaned, None)
        self._keys: List[str] = list(seen)
        if not self._keys:
            raise ValueError("KeyPool needs at least one non-empty key")

        self._cooldown = cooldown_seconds
        self._index = 0
        self._blocked_until: Dict[str, float] = {}

    def __len__(self) -> int:
        return len(self._keys)

    @property
    def current(self) -> str:
        return self._keys[self._index]

    def _available(self, key: str, now: float) -> bool:
        return self._blocked_until.get(key, 0.0) <= now

    def penalise_current(self, now: Optional[float] = None) -> None:
        """Park the key in use after it returned 429."""
        now = time.time() if now is None else now
        key = self.current
        self._blocked_until[key] = now + self._cooldown
        logger.warning(
            "API key %s rate limited — cooling down for %.0fs (%d key(s) in pool)",
            self.describe(key),
            self._cooldown,
            len(self._keys),
        )

    def rotate(self, now: Optional[float] = None) -> Optional[str]:
        """
        Switch to the next key that is not cooling down.

        Returns the new key, or None when every key is cooling down — which is
        the caller's signal to fall back to waiting rather than rotating.
        """
        now = time.time() if now is None else now
        for step in range(1, len(self._keys) + 1):
            candidate_index = (self._index + step) % len(self._keys)
            candidate = self._keys[candidate_index]
            if self._available(candidate, now):
                self._index = candidate_index
                logger.info("Switched to API key %s", self.describe(candidate))
                return candidate
        return None

    def seconds_until_any_available(self, now: Optional[float] = None) -> float:
        """How long until the earliest-expiring cooldown lifts. 0 if one is free."""
        now = time.time() if now is None else now
        if any(self._available(k, now) for k in self._keys):
            return 0.0
        return max(0.0, min(self._blocked_until.values()) - now)

    @staticmethod
    def describe(key: str) -> str:
        """Identify a key in logs without printing it."""
        return f"...{key[-4:]}" if len(key) >= 4 else "<short>"

    def status(self, now: Optional[float] = None) -> str:
        now = time.time() if now is None else now
        parts = []
        for index, key in enumerate(self._keys):
            state = "active" if index == self._index else "idle"
            if not self._available(key, now):
                state = f"cooling {self._blocked_until[key] - now:.0f}s"
            parts.append(f"{self.describe(key)}:{state}")
        return ", ".join(parts)


def load_keys_from_env(
    primary_var: str = "KISSKI_API_KEY", list_var: str = "KISSKI_API_KEYS"
) -> List[str]:
    """
    Collect every configured key, in priority order.

    Three forms are accepted so the existing single-key setup keeps working:

        KISSKI_API_KEY=abc                 the original variable, used first
        KISSKI_API_KEY_2=def               numbered extras, _2 upwards
        KISSKI_API_KEYS=abc,def,ghi        comma-separated list

    Order matters: the primary key is tried first, so behaviour with one key
    configured is identical to before.
    """
    keys: List[str] = []

    primary = os.getenv(primary_var)
    if primary and primary.strip():
        keys.append(primary.strip())

    # Numbered extras, stopping at the first gap so a stray _7 is not mistaken
    # for the whole sequence.
    index = 2
    while True:
        value = os.getenv(f"{primary_var}_{index}")
        if not value or not value.strip():
            break
        keys.append(value.strip())
        index += 1

    listed = os.getenv(list_var)
    if listed:
        keys.extend(part.strip() for part in listed.split(",") if part.strip())

    # De-duplicate, preserving priority order.
    seen: Dict[str, None] = {}
    for key in keys:
        seen.setdefault(key, None)
    return list(seen)
