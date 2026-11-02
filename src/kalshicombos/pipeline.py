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
from .edges import Leg, evaluate, passes, passes_combo, suspicious
from .fair import BookLines, FairPricer
from .kalshi import KalshiClient
from .odds import OddsClient, match_event


@dataclass
class GameResult:
    game: str
    kickoff: str
    legs: list[Leg] = field(default_factory=list)       # every priced leg (both sides)
    qualifying: list[Leg] = field(default_factory=list)   # single bets: beat ask + fee
    combo_legs: list[Leg] = field(default_factory=list)   # combo legs: fair >= Kalshi middle price
    combos: list[Combo] = field(default_factory=list)
    suspicious: list[Leg] = field(default_factory=list)  # edges too big to trust
    raw_quotes: list[dict] = field(default_factory=list)  # saved so a run can be audited/replayed
    markets: list = field(default_factory=list)            # Kalshi markets seen (saved for replay)
    note: str = ""
    combo_rules: dict | None = None                        # Kalshi's combo collections for this game (raw, saved)


@dataclass
class RunResult:
    run_id: str
    games: list[GameResult]
    credits_remaining: str | None
    config: dict
    notes: list[str] = field(default_factory=list)   # why requested games were not priced


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
    notes: list[str] = []
    seen: set[str] = set()
    for suffix in kc.open_games(cfg["kalshi"].get("discovery_series", "KXNFLGAME")):
        matchup = suffix[7:]
        if only and matchup not in only:
            continue
        seen.add(matchup)
        markets = kc.markets(series, suffix)
        if not markets:
            if only:
                notes.append(f"{matchup}: Kalshi has no open player markets for this game yet.")
            continue
        away, home = markets[0].away, markets[0].home
        ev = match_event(events, away, home)
        if ev is None:
            if only:
                notes.append(f"{matchup}: sportsbooks aren't listing this game (finished, or not posted yet).")
            continue
        kickoff = _parse_time(ev["commence_time"])
        if kickoff <= now:
            if only:
                notes.append(f"{matchup}: already kicked off ({ev['commence_time']}); only pre-game prices are used.")
            continue  # started: pre-game prices only
        if not only and kickoff > horizon:
            continue
        gr = GameResult(game=matchup, kickoff=ev["commence_time"], markets=markets)
        try:   # Kalshi's live combo rules (eligible events, Yes-only, per-event caps, size limits)
            gr.combo_rules = {"collections": kc.combo_collections(f"{cfg['kalshi'].get('discovery_series', 'KXNFLGAME')}-{suffix}")}
        except Exception as e:  # rules unavailable: the logical checks still apply
            gr.combo_rules = {"collections": [], "error": f"{type(e).__name__}: {e}"}
        quotes = oc.event_quotes(ev["id"])
        gr.raw_quotes = [q.__dict__ for q in quotes]
        try:
            results.append(_price_game(gr, quotes, cfg))
        except Exception as e:  # one bad game must not sink the run; raw data is still saved
            gr.note = f"skipped: pricing error ({type(e).__name__}: {e}). Raw quotes saved for a replay."
            results.append(gr)

    for g in only:
        if g not in seen:
            notes.append(f"{g}: not an open game on Kalshi. It has already been played, isn't listed yet, "
                         f"or the code is wrong (use away+home, e.g. CHISEA).")
    return RunResult(run_id=now.strftime("%Y%m%dT%H%MZ"), games=results,
                     credits_remaining=oc.credits_remaining, config=cfg, notes=notes)


_SHAPES_CACHE: dict = {}


def _shapes(cfg: dict):
    from .history import ShapeModel
    m = cfg["model"]
    path = m.get("history_file")
    if not path:
        return None
    key = (path, str(m.get("shape_alpha")), str(m.get("shape_zero_scale")))
    if key not in _SHAPES_CACHE:
        _SHAPES_CACHE[key] = ShapeModel.load(path, m.get("shape_alpha"), m.get("shape_zero_scale"))
    return _SHAPES_CACHE[key]


def _price_game(gr: GameResult, quotes: list, cfg: dict) -> GameResult:
    if not quotes:
        gr.note = "no sportsbook prices yet"
        return gr
    lines = BookLines(quotes, cfg["odds"]["one_sided_margin"], cfg["odds"].get("one_sided_margin_props"),
                      bool(cfg["odds"].get("allow_one_sided_props", False)))
    pricer = FairPricer(lines, cfg["model"], cfg["odds"], _shapes(cfg))
    gr.legs = evaluate(gr.markets, pricer, cfg["filters"]["stake_usd"], cfg["kalshi"]["taker_fee_rate"])
    gr.qualifying = sorted([l for l in gr.legs if passes(l, cfg["filters"])], key=lambda l: l.ev, reverse=True)
    gr.suspicious = sorted([l for l in gr.legs if suspicious(l, cfg["filters"])], key=lambda l: l.edge, reverse=True)
    gr.combo_legs = sorted([l for l in gr.legs if passes_combo(l, cfg["filters"])],
                           key=lambda l: l.edge_mid, reverse=True)
    gr.combos = build_for_game(gr.combo_legs, cfg["combos"], _shapes(cfg), rules_for(gr))
    gr.note = ""
    if len(gr.combo_legs) < cfg["combos"]["min_legs"]:
        gr.note = f"only {len(gr.combo_legs)} leg(s) passed the combo filters, so no combo"
    elif not gr.combos:
        gr.note = ("legs passed but none can be combined (they're rungs of the same player stat on the "
                   "same side, which say the same thing); treat the best one as a single bet")
    return gr


def rules_for(gr: GameResult):
    """Kalshi's combo rules for this game, or None when they weren't fetched."""
    from .buildable import ComboRules
    from .kalshi import event_of
    cols = (gr.combo_rules or {}).get("collections") or []
    if not cols:
        return None
    return ComboRules.from_collections(cols, {event_of(m.ticker) for m in gr.markets})


def replay(run_dir: str | Path, cfg: dict) -> RunResult:
    """Re-price a saved run (Kalshi prices + book quotes) with the current config. No API calls, no credits."""
    import pandas as pd
    from .kalshi import KalshiMarket
    from .odds import Quote
    run_dir = Path(run_dir)
    km = pd.read_csv(run_dir / "kalshi_markets.csv", keep_default_na=False)
    bq = pd.read_csv(run_dir / "book_quotes.csv.gz", keep_default_na=False)
    games = []
    for game, rows in km.groupby("game", sort=False):
        markets = []
        for r in rows.to_dict("records"):
            kickoff = r.pop("kickoff")
            for k in ("team", "player"):
                r[k] = r[k] or None
            r["threshold"] = None if r["threshold"] == "" else float(r["threshold"])
            markets.append(KalshiMarket(**{k: v for k, v in r.items() if k in KalshiMarket.__dataclass_fields__}))
        qs = [Quote(book=r["book"], market=r["market"], name=r["name"], description=r["description"],
                    point=None if r["point"] == "" else float(r["point"]), price=float(r["price"]))
              for r in bq[bq.game == game].to_dict("records")]
        gr = GameResult(game=game, kickoff=kickoff, markets=markets, raw_quotes=[q.__dict__ for q in qs])
        rf = run_dir / "combo_rules.json"
        if rf.exists():
            gr.combo_rules = json.loads(rf.read_text()).get(game)
        games.append(_price_game(gr, qs, cfg))
    return RunResult(run_id=run_dir.name + "-replay", games=games, credits_remaining=None, config=cfg)


# ---------- offline fixtures (tests / replaying a saved snapshot) ----------

def fixture_getters(folder: str | Path) -> tuple[Callable, Callable]:
    folder = Path(folder)

    def kalshi_get(path: str, params: dict) -> dict:
        if path == "/events":
            return json.loads((folder / "kalshi" / f"events_{params['series_ticker']}.json").read_text())
        if path == "/multivariate_event_collections":
            f = folder / "kalshi" / "combo_collections.json"
            return json.loads(f.read_text()) if f.exists() else {"multivariate_contracts": []}
        f = folder / "kalshi" / f"{params['event_ticker']}.json"
        return json.loads(f.read_text()) if f.exists() else {"markets": []}

    def odds_get(path: str, params: dict):
        if path.endswith("/events"):
            return json.loads((folder / "odds" / "events.json").read_text())
        event_id = path.split("/events/")[1].split("/")[0]
        return json.loads((folder / "odds" / f"{event_id}.json").read_text())

    return kalshi_get, odds_get
