"""Kalshi public market data (no API key needed for reading prices)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from . import teams
from .http import get_json

Getter = Callable[[str, dict], dict]


@dataclass
class KalshiMarket:
    ticker: str
    game: str            # e.g. "CHISEA" (away+home)
    date: str            # e.g. "26NOV02"
    away: str
    home: str
    kind: str            # moneyline | spread | total | receptions | rec_yds | rush_yds | pass_yds | td
    title: str
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    team: str | None = None
    player: str | None = None
    threshold: float | None = None   # YES means stat > threshold ("30+" -> 29.5)
    extra: dict = field(default_factory=dict)

    @property
    def yes_mid(self) -> float:
        return (self.yes_bid + self.yes_ask) / 2.0

    def label(self, side: str) -> str:
        t = self.title.rstrip("?")
        return t if side == "yes" else f"No · {t}"


def _price(m: dict, key: str) -> float | None:
    v = m.get(f"{key}_dollars")
    if v not in (None, ""):
        return float(v)
    v = m.get(key)
    return None if v is None else float(v) / 100.0


def _event_parts(event_ticker: str) -> tuple[str, str, str, str] | None:
    suffix = event_ticker.split("-", 1)[1]
    mt = re.match(r"(\d{2}[A-Z]{3}\d{2})([A-Z]+)$", suffix)
    if not mt:
        return None
    date, matchup = mt.groups()
    sp = teams.split_matchup(matchup)
    if not sp:
        return None
    return date, matchup, sp[0], sp[1]


def _team_prefix(segment: str, away: str, home: str) -> str | None:
    seg = segment.upper()
    for t in sorted({away, home}, key=len, reverse=True):
        for alias in [t] + [a for a, c in teams.ALIASES.items() if c == t]:
            if seg.startswith(alias):
                return t
    return None


def parse_market(m: dict, kind: str) -> KalshiMarket | None:
    parts = _event_parts(m["event_ticker"])
    if not parts:
        return None
    date, matchup, away, home = parts
    yb, ya = _price(m, "yes_bid"), _price(m, "yes_ask")
    if yb is None or ya is None or ya <= 0:
        return None
    nb = _price(m, "no_bid")
    na = _price(m, "no_ask")
    nb = 1.0 - ya if nb is None else nb
    na = 1.0 - yb if na is None else na
    km = KalshiMarket(ticker=m["ticker"], game=matchup, date=date, away=away, home=home, kind=kind,
                      title=m.get("title", ""), yes_bid=yb, yes_ask=ya, no_bid=nb, no_ask=na)
    tail = m["ticker"].split("-")
    if kind == "moneyline":
        km.team = teams.code(tail[-1])
    elif kind == "spread":
        km.team = teams.code(re.sub(r"\d+$", "", tail[-1]))
        km.threshold = float(m["floor_strike"])
    elif kind == "total":
        km.threshold = float(m["floor_strike"])
    else:
        sub = m.get("yes_sub_title") or m.get("title", "")
        name = sub.split(":")[0].strip()
        if "D/ST" in name:
            return None  # books don't price defence/special-teams TDs
        km.player = name
        km.team = _team_prefix(tail[2] if len(tail) > 2 else "", away, home)
        km.threshold = float(m["floor_strike"])
    if km.team is not None and km.team not in (away, home):
        return None
    return km


class KalshiClient:
    def __init__(self, base_url: str, getter: Getter | None = None):
        self.base = base_url.rstrip("/")
        self._get = getter or (lambda path, params: get_json(self.base + path, params)[0])

    def _paged(self, path: str, params: dict, key: str) -> list[dict]:
        out, cursor = [], None
        while True:
            p = dict(params, limit=params.get("limit", 1000))
            if cursor:
                p["cursor"] = cursor
            data = self._get(path, p)
            out.extend(data.get(key) or [])
            cursor = data.get("cursor")
            if not cursor:
                return out

    def open_games(self, game_series: str) -> list[str]:
        """Event suffixes like '26NOV02CHISEA' for every open NFL game."""
        evs = self._paged("/events", {"series_ticker": game_series, "status": "open", "limit": 200}, "events")
        return [e["event_ticker"].split("-", 1)[1] for e in evs]

    def markets(self, series_by_kind: dict[str, str], game_suffix: str) -> list[KalshiMarket]:
        out: list[KalshiMarket] = []
        for kind, series in series_by_kind.items():
            raw = self._paged("/markets", {"event_ticker": f"{series}-{game_suffix}"}, "markets")
            for m in raw:
                if m.get("status") not in (None, "active", "open"):
                    continue
                km = parse_market(m, kind)
                if km:
                    out.append(km)
        return out
