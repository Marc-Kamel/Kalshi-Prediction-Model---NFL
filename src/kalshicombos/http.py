"""Small HTTP helper: polite pacing, retries with backoff, honours Retry-After.
Read-only GET requests only — this project never trades."""
from __future__ import annotations

import random
import time
from typing import Any

import requests

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "kalshicombos/0.1 (read-only research tool)"})

MIN_INTERVAL = 0.35   # seconds between requests (Kalshi/Odds API rate limits)
MAX_WAIT = 60.0
_last_call = 0.0


def _pace() -> None:
    global _last_call
    wait = MIN_INTERVAL - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()


def _backoff(attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            return min(MAX_WAIT, float(retry_after))
        except ValueError:
            pass
    return min(MAX_WAIT, 2.0 ** attempt + random.uniform(0, 1.0))


def get_json(url: str, params: dict | None = None, retries: int = 8, timeout: int = 20) -> tuple[Any, dict]:
    last: Exception | None = None
    for attempt in range(retries):
        _pace()
        try:
            r = _SESSION.get(url, params=params, timeout=timeout)
        except requests.RequestException as e:  # network blip
            last = e
            time.sleep(_backoff(attempt, None))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            last = RuntimeError(f"{r.status_code} from {url}")
            time.sleep(_backoff(attempt, r.headers.get("Retry-After")))
            continue
        r.raise_for_status()
        return r.json(), dict(r.headers)
    raise RuntimeError(f"GET failed after {retries} tries: {url}: {last}")
