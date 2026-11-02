"""Friend mode: one big same-game combo built the way the friend's agent built its 34-leg PHI@WAS combo.

His recipe (reverse-engineered from his logs, re-implemented here; no code copied):
  1. One game, run close to kickoff.
  2. Every Kalshi player leg whose sportsbook fair chance is above 55%, is +EV after fees at
     the ask, and is priced by 5+ books. Alternate (Over-only) lines are allowed.
  3. One line per player per stat (best EV). No same-stat ranges.
  4. ALL of them in one combo. Fair chance = the legs multiplied together (independence).
  5. Request a quote; accept if it pays at least fair + 3%.

What we add (he had none of this): a reality check on every combo.
  - The same combo priced with measured correlations.
  - Our strict model's price (two-sided lines only), and Kalshi's own leg prices.
  - The accept-if number defaults to the STRICT price: on PHI@WAS his alternate-line fair prices
    ran ~5 pts above Kalshi on likely outcomes, which made a ~1-in-1,200 combo look like 1-in-287.
Suggest-only: nothing is ever bought.
"""
from __future__ import annotations

import copy
import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import pipeline
from .combos import simulate_hits
from .edges import Leg, passes
from .fair import BookLines, FairPricer
from .odds import Quote


def friend_cfg(cfg: dict) -> dict:
    """The run config with the friend's leg rules switched on."""
    fm = cfg.get("friend_mode", {})
    f = copy.deepcopy(cfg)
    extra = [m for m in fm.get("extra_markets", []) if m not in f["odds"]["markets"]]
    f["odds"]["markets"] = list(f["odds"]["markets"]) + extra
    f["odds"]["allow_one_sided_props"] = bool(fm.get("allow_one_sided_props", True))
    f["odds"]["min_books"] = int(fm.get("min_books", 5))
    f["filters"] = dict(f["filters"],
                        min_kalshi_prob=float(fm.get("min_kalshi_prob", 0.0)),
                        min_fair_prob=float(fm.get("min_fair_prob", 0.55)),
                        min_edge=float(fm.get("min_edge", 0.0)),
                        one_sided_buffer=0.0, shape_buffer=[], max_edge=float(fm.get("max_edge", 1.0)))
    return f


@dataclass
class FriendCombo:
    game: str
    legs: list[Leg]
    strict_p: list[float | None]          # strict model P(this side) per leg, None if it can't price it
    his_indep: float                      # friend-style fair: legs multiplied
    his_corr: float                       # friend-style leg prices, measured correlations
    strict_corr: float                    # strict prices (Kalshi where strict can't price), correlations
    kalshi_corr: float                    # Kalshi's own leg prices, correlations
    accept_margin: float
    basis: str                            # "strict" or "friend"
    notes: list[str] = field(default_factory=list)

    def payout(self, p: float) -> float:
        return 1.0 / p if p > 0 else float("inf")

    @property
    def accept_x(self) -> float:
        p = self.strict_corr if self.basis == "strict" else self.his_indep
        return self.payout(p) * (1.0 + self.accept_margin)

    @property
    def friend_accept_x(self) -> float:
        return self.payout(self.his_indep) * (1.0 + self.accept_margin)

    @property
    def n_strict_plus_ev(self) -> int:
        return sum(1 for l, p in zip(self.legs, self.strict_p) if p is not None and p - l.all_in > 0)


def _with_probs(legs: list[Leg], probs: list[float]) -> list[Leg]:
    return [dataclasses.replace(l, fair_prob=float(p)) for l, p in zip(legs, probs)]


def combo_probability(legs: list[Leg], probs: list[float], cfg: dict, shapes, sims: int, chunk: int = 250_000) -> float:
    """P(every leg wins) under measured correlations; simulated in chunks so 30+ legs stay cheap."""
    legs = _with_probs(legs, probs)
    wins = total = 0
    i = 0
    while total < sims:
        n = min(chunk, sims - total)
        c = dict(cfg, sims=n, seed=int(cfg.get("seed", 7)) + i)
        wins += int(simulate_hits(legs, c, shapes).all(axis=1).sum())
        total += n
        i += 1
    return wins / total


def select_legs(gr: pipeline.GameResult, fcfg: dict) -> list[Leg]:
    """His filter: passes his gates, best EV per player-stat, sorted by EV."""
    ok = [l for l in gr.legs if passes(l, fcfg["filters"]) and l.market.player]
    best: dict[tuple, Leg] = {}
    for l in sorted(ok, key=lambda l: l.ev, reverse=True):
        best.setdefault((l.market.player, l.market.kind), l)
    legs = sorted(best.values(), key=lambda l: l.ev, reverse=True)
    cap = int(fcfg.get("friend_mode", {}).get("max_legs", 0))
    return legs[:cap] if cap > 0 else legs


def build(gr: pipeline.GameResult, cfg: dict, basis: str | None = None) -> FriendCombo | None:
    fcfg = friend_cfg(cfg)
    fm = cfg.get("friend_mode", {})
    legs = select_legs(gr, fcfg)
    if len(legs) < 2:
        return None
    quotes = [Quote(**q) for q in gr.raw_quotes]
    strict = FairPricer(BookLines(quotes, cfg["odds"]["one_sided_margin"], cfg["odds"].get("one_sided_margin_props"),
                                  False), cfg["model"], cfg["odds"], pipeline._shapes(cfg))
    strict_p: list[float | None] = []
    for l in legs:
        f = strict.price(l.market)
        strict_p.append(None if f is None else (f.p if l.side == "yes" else 1.0 - f.p))
    shapes = pipeline._shapes(cfg)
    sims = int(fm.get("sims", 2_000_000))
    ccfg = cfg["combos"]
    his = [l.fair_prob for l in legs]
    kal = [l.kalshi_prob for l in legs]
    strict_or_kal = [s if s is not None else k for s, k in zip(strict_p, kal)]
    fc = FriendCombo(
        game=gr.game, legs=legs, strict_p=strict_p,
        his_indep=float(np.prod(his)),
        his_corr=combo_probability(legs, his, ccfg, shapes, sims),
        strict_corr=combo_probability(legs, strict_or_kal, ccfg, shapes, sims),
        kalshi_corr=combo_probability(legs, kal, ccfg, shapes, sims),
        accept_margin=float(ccfg["accept_margin"]),
        basis=basis or fm.get("accept_basis", "strict"),
    )
    n_one = sum(1 for l in legs if l.one_sided)
    if n_one:
        fc.notes.append(f"{n_one} of {len(legs)} legs are priced from Over/Yes-only lines (vig guessed); "
                        f"those ran ~4–5 pts too high vs Kalshi on CHI@SEA and PHI@WAS.")
    if fc.strict_corr * fc.payout(fc.his_indep) < 0.7:
        fc.notes.append("Our strict model says this combo is much less likely than the friend-style price implies.")
    return fc


# ---------- output ----------

def _x(p: float) -> str:
    return f"{1 / p:,.0f}x" if p > 0 else "—"


def _p(p: float) -> str:
    return f"{p * 100:.3f}%" if p < 0.01 else f"{p * 100:.1f}%"


def to_markdown(run_id: str, combos: list[FriendCombo], skipped: list[str]) -> str:
    out = [f"# Friend mode — run {run_id}", "",
           "His recipe: every player leg with fair > 55%, +EV after fees, 5+ books, one line per player-stat, "
           "alternate lines allowed, all in one combo. Suggestions only; nothing is bought.", ""]
    for s in skipped:
        out += [f"_{s}_", ""]
    for c in combos:
        out += [f"## {c.game} — {len(c.legs)}-leg combo", "",
                "| How it's priced | Chance | Fair payout |", "|---|---|---|",
                f"| Friend-style (legs multiplied, like his model) | {_p(c.his_indep)} | {_x(c.his_indep)} |",
                f"| Friend-style legs, measured correlations | {_p(c.his_corr)} | {_x(c.his_corr)} |",
                f"| Our strict model, measured correlations | {_p(c.strict_corr)} | {_x(c.strict_corr)} |",
                f"| Kalshi's own leg prices, measured correlations | {_p(c.kalshi_corr)} | {_x(c.kalshi_corr)} |", "",
                f"**Accept only if Kalshi's quote pays ≥ {c.accept_x:,.0f}x** "
                f"({'our strict model' if c.basis == 'strict' else 'friend-style price'} + {c.accept_margin * 100:.0f}%). "
                f"His model would accept ≥ {c.friend_accept_x:,.0f}x.", ""]
        out += [f"- {n}" for n in c.notes]
        out += ["", "| Leg | Kalshi | Ask + fee | Friend fair | Strict fair | Strict edge |", "|---|---|---|---|---|---|"]
        for l, s in zip(c.legs, c.strict_p):
            sf = "—" if s is None else f"{s * 100:.1f}%"
            se = "—" if s is None else f"{(s - l.all_in) * 100:+.1f}¢"
            tag = " ¹" if l.one_sided else ""
            out.append(f"| {l.label}{tag} | {l.kalshi_prob * 100:.1f}% | {l.all_in * 100:.1f}¢ | "
                       f"{l.fair_prob * 100:.1f}% | {sf} | {se} |")
        out += ["", f"¹ priced from an Over/Yes-only line. Strict model: {c.n_strict_plus_ev} of {len(c.legs)} legs +EV "
                    f"after fees; '—' = it can't price that leg (two-sided line too far away), Kalshi's price is used.", ""]
    return "\n".join(out)


def write(run_id: str, combos: list[FriendCombo], skipped: list[str], out_dir: str | Path) -> Path:
    base = Path(out_dir)
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in combos:
        rows.append({"game": c.game, "n_legs": len(c.legs),
                     "friend_fair_prob": c.his_indep, "friend_corr_prob": c.his_corr,
                     "strict_corr_prob": c.strict_corr, "kalshi_corr_prob": c.kalshi_corr,
                     "accept_if_payout_x_at_least": round(c.accept_x, 1),
                     "friend_accept_x": round(c.friend_accept_x, 1),
                     "legs": " + ".join(l.label for l in c.legs),
                     "tickers": ",".join(l.market.ticker for l in c.legs),
                     "sides": ",".join(l.side for l in c.legs)})
    pd.DataFrame(rows).to_csv(run_dir / "friend_combo.csv", index=False)
    md = to_markdown(run_id, combos, skipped)
    (run_dir / "friend_report.md").write_text(md)
    (base / "latest_friend.md").write_text(md)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as fh:
            fh.write(md + "\n")
    return run_dir


def run(cfg: dict, games: list[str] | None = None, replay_dir: str | None = None,
        kalshi_getter=None, odds_getter=None, now=None, basis: str | None = None):
    fcfg = friend_cfg(cfg)
    res = (pipeline.replay(replay_dir, fcfg) if replay_dir
           else pipeline.run(fcfg, kalshi_getter, odds_getter, now=now, only=games))
    combos, skipped = [], []
    for gr in res.games:
        fc = build(gr, cfg, basis)
        if fc is None:
            skipped.append(f"{gr.game}: fewer than 2 legs passed his rules. {gr.note}".strip())
        else:
            combos.append(fc)
    return res, combos, skipped
