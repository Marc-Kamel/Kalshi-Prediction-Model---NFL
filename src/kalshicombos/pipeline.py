"""End-to-end run: Kalshi prices + sportsbook odds -> +EV legs -> simulated combos."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import yaml

from .combos import Combo, build_for_game
from .edges import Leg, evaluate, passes, suspicious
from .fair import BookLines, FairPricer
from .kalshi import KalshiClient
from .odds import OddsClient, match_event


@dataclass
class GameResult:
    game: str
    kickoff: str
    legs: list[Leg] = field(default_factory=list)       # every priced leg (both sides)
    qualifying: list[Leg] = field(default_factory=list)
    combos: list[Combo] = field(default_factory=list)
    suspicious: list[Leg] = field(default_factory=list)  # edges too big to trust
    raw_quotes: list[dict] = field(default_factory=list)  # saved so a run can be audited/replayed
    note: str = ""


@dataclass
class RunResult:
    run_id: str
    games: list[GameResult]
    credits_remaining: str | None
    config: dict


def load_config(path: str | os.PathLike = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def run(cfg: dict, kalshi_getter: Callable | None = None, odds_getter: Callable | None = None,
        now: datetime | None = None, only: list[str] | None = None) -> RunResult:
    now = now or datetime.now(timezone.utc)
    kc = KalshiClient(cfg["kalshi"]["base_url"], kalshi_getter)
    oc = OddsClient(cfg["odds"], odds_getter)
    series = cfg["kalshi"]["series"]
    only = [g.upper() for g in (only or cfg["games"].get("only") or [])]
    horizon = now + timedelta(hours=float(cfg["games"]["hours_ahead"]))

    events = oc.events()
    results: list[GameResult] = []
    for suffix in kc.open_games(series["moneyline"]):
        matchup = suffix[7:]
        if only and matchup not in only:
            continue
        markets = kc.markets(series, suffix)
        if not markets:
            continue
        away, home = markets[0].away, markets[0].home
        ev = match_event(events, away, home)
        if ev is None:
            continue
        kickoff = _parse_time(ev["commence_time"])
        if kickoff <= now:
            continue  # started: pre-game prices only
        if not only and kickoff > horizon:
            continue
        gr = GameResult(game=matchup, kickoff=ev["commence_time"])
        quotes = oc.event_quotes(ev["id"])
        gr.raw_quotes = [q.__dict__ for q in quotes]
        if not quotes:
            gr.note = "no sportsbook prices yet"
            results.append(gr)
            continue
        pricer = FairPricer(BookLines(quotes, cfg["odds"]["one_sided_margin"]), cfg["model"], cfg["odds"])
        gr.legs = evaluate(markets, pricer, cfg["filters"]["stake_usd"], cfg["kalshi"]["taker_fee_rate"])
        gr.qualifying = sorted([l for l in gr.legs if passes(l, cfg["filters"])], key=lambda l: l.ev, reverse=True)
        gr.suspicious = sorted([l for l in gr.legs if suspicious(l, cfg["filters"])], key=lambda l: l.edge, reverse=True)
        gr.combos = build_for_game(gr.qualifying, cfg["combos"])
        if len(gr.qualifying) < cfg["combos"]["min_legs"]:
            gr.note = f"only {len(gr.qualifying)} leg(s) passed the filters, so no combo"
        results.append(gr)

    return RunResult(run_id=now.strftime("%Y%m%dT%H%MZ"), games=results,
                     credits_remaining=oc.credits_remaining, config=cfg)


# ---------- offline fixtures (tests / replaying a saved snapshot) ----------

def fixture_getters(folder: str | Path) -> tuple[Callable, Callable]:
    folder = Path(folder)

    def kalshi_get(path: str, params: dict) -> dict:
        if path == "/events":
            return json.loads((folder / "kalshi" / f"events_{params['series_ticker']}.json").read_text())
        f = folder / "kalshi" / f"{params['event_ticker']}.json"
        return json.loads(f.read_text()) if f.exists() else {"markets": []}

    def odds_get(path: str, params: dict):
        if path.endswith("/events"):
            return json.loads((folder / "odds" / "events.json").read_text())
        event_id = path.split("/events/")[1].split("/")[0]
        return json.loads((folder / "odds" / f"{event_id}.json").read_text())

    return kalshi_get, odds_get
