"""Small HTTP helper with retries. Read-only GET requests only — this project never trades."""
from __future__ import annotations

import time
from typing import Any

import requests

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "kalshicombos/0.1 (read-only research tool)"})


def get_json(url: str, params: dict | None = None, retries: int = 4, timeout: int = 20) -> tuple[Any, dict]:
    last = None
    for attempt in range(retries):
        try:
            r = _SESSION.get(url, params=params, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                last = RuntimeError(f"{r.status_code} from {url}")
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json(), dict(r.headers)
        except requests.RequestException as e:  # network blips
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url}: {last}")
