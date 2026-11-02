"""The Odds API (v4) client: sportsbook game lines and player props."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from . import teams
from .http import get_json

BASE = "https://api.the-odds-api.com/v4"


@dataclass
class Quote:
    book: str
    market: str          # Odds API market key, e.g. player_reception_yds
    name: str            # Over / Under / Yes / No / team name
    description: str     # player name for props
    point: float | None
    price: float         # American odds


class OddsClient:
    def __init__(self, cfg: dict, getter: Callable | None = None):
        self.cfg = cfg
        key = os.environ.get(cfg.get("api_key_env", "ODDS_API_KEY"), "")
        self.key = key
        self.sport = cfg.get("sport", "americanfootball_nfl")
        self.credits_remaining: str | None = None
        self._get = getter or self._http

    def _http(self, path: str, params: dict):
        if not self.key:
            raise RuntimeError("ODDS_API_KEY is not set. Get a key at the-odds-api.com and add it as a secret.")
        data, headers = get_json(BASE + path, dict(params, apiKey=self.key))
        self.credits_remaining = headers.get("x-requests-remaining", self.credits_remaining)
        return data

    def events(self) -> list[dict]:
        """Upcoming games (this endpoint costs no credits)."""
        return self._get(f"/sports/{self.sport}/events", {"dateFormat": "iso"})

    def event_quotes(self, event_id: str) -> list[Quote]:
        params = {
            "markets": ",".join(self.cfg["markets"]),
            "oddsFormat": "american",
            "dateFormat": "iso",
        }
        if self.cfg.get("bookmakers"):
            params["bookmakers"] = ",".join(self.cfg["bookmakers"])
        else:
            params["regions"] = self.cfg.get("regions", "us")
        data = self._get(f"/sports/{self.sport}/events/{event_id}/odds", params)
        out: list[Quote] = []
        for b in data.get("bookmakers", []):
            for mk in b.get("markets", []):
                for o in mk.get("outcomes", []):
                    if o.get("price") is None:
                        continue
                    out.append(Quote(book=b["key"], market=mk["key"], name=o.get("name", ""),
                                     description=o.get("description", "") or "",
                                     point=o.get("point"), price=float(o["price"])))
        return out


def match_event(events: list[dict], away: str, home: str) -> dict | None:
    for e in events:
        if teams.from_name(e.get("home_team", "")) == home and teams.from_name(e.get("away_team", "")) == away:
            return e
    return None
