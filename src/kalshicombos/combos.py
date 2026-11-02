"""Monte Carlo same-game combo builder.

Legs in one game are correlated (a QB's passing yards and his WR's receiving yards rise
together; a "No TD" leg wins more often when the game is low scoring). Each leg gets a latent
performance score built from shared factors:

    team offense (OFF_T)   team pass volume (PASS_T)   player's own game (PLAYER_p)

plus independent noise. A YES leg wins when its score clears the cut that reproduces the
leg's fair probability, so every leg keeps its exact fair chance and only the joint
behaviour comes from the factors. Loadings live in config.yaml.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .edges import Leg


@dataclass
class Combo:
    legs: list[Leg]
    fair_prob: float          # correlated simulation
    indep_prob: float         # product of fair leg probabilities (no correlation)
    kalshi_prob: float        # product of Kalshi leg probabilities
    leg_ask_payout: float     # payout multiple if priced at the product of leg asks + fees
    fair_payout: float
    min_accept_payout: float  # accept a Kalshi quote only if it pays at least this multiple

    @property
    def ev_at_leg_asks(self) -> float:
        return self.fair_prob * self.leg_ask_payout - 1.0

    def as_row(self) -> dict:
        return {
            "game": self.legs[0].market.game, "n_legs": len(self.legs),
            "legs": " + ".join(l.label for l in self.legs),
            "fair_prob": round(self.fair_prob, 4), "indep_prob": round(self.indep_prob, 4),
            "kalshi_prob": round(self.kalshi_prob, 4),
            "fair_payout_x": round(self.fair_payout, 2),
            "accept_if_payout_x_at_least": round(self.min_accept_payout, 2),
            "max_price_per_1_payout": round(1.0 / self.min_accept_payout, 4),
            "leg_ask_payout_x": round(self.leg_ask_payout, 2),
            "ev_pct_at_leg_asks": round(self.ev_at_leg_asks * 100, 2),
            "tickers": ",".join(l.market.ticker for l in self.legs),
            "sides": ",".join(l.side for l in self.legs),
        }


def _group(leg: Leg) -> tuple:
    """Legs in the same group can't go in one combo (same stat for the same player, or both
    moneyline/spread on one game, which are near-duplicates)."""
    m = leg.market
    if m.kind in ("moneyline", "spread"):
        return (m.game, "side")
    if m.kind == "total":
        return (m.game, "total")
    return (m.game, m.kind, m.player)


def _loadings(leg: Leg, idx: dict[str, int], cfg: dict) -> np.ndarray:
    m = leg.market
    w = np.zeros(len(idx))
    if m.kind in ("moneyline", "spread"):
        g = cfg["game_loading"]
        opp = m.home if m.team == m.away else m.away
        w[idx[f"OFF_{m.team}"]] = g
        w[idx[f"OFF_{opp}"]] = -g
    elif m.kind == "total":
        t = cfg["total_loading"]
        w[idx[f"OFF_{m.away}"]] = t
        w[idx[f"OFF_{m.home}"]] = t
    else:
        off, pas, own = cfg["loadings"][m.kind]
        if m.team:
            w[idx[f"OFF_{m.team}"]] = off
            w[idx[f"PASS_{m.team}"]] = pas
        w[idx[f"PL_{m.player}"]] = own
    ss = float(np.sum(w ** 2))
    if ss >= 0.98:  # keep room for leg-specific noise
        w *= np.sqrt(0.98 / ss)
    return w


def simulate_hits(legs: list[Leg], cfg: dict) -> np.ndarray:
    """Boolean matrix (sims x legs): did each leg win in each simulated game."""
    m0 = legs[0].market
    names = [f"OFF_{m0.away}", f"OFF_{m0.home}", f"PASS_{m0.away}", f"PASS_{m0.home}"]
    names += sorted({f"PL_{l.market.player}" for l in legs if l.market.player})
    idx = {n: i for i, n in enumerate(names)}
    rng = np.random.default_rng(cfg.get("seed", 7))
    n = int(cfg["sims"])
    F = rng.standard_normal((n, len(names)))
    hits = np.empty((n, len(legs)), dtype=bool)
    for j, leg in enumerate(legs):
        w = _loadings(leg, idx, cfg)
        resid = np.sqrt(max(1e-6, 1.0 - float(np.sum(w ** 2))))
        x = F @ w + resid * rng.standard_normal(n)
        cut = stats.norm.ppf(1.0 - leg.p_yes)
        hits[:, j] = (x > cut) if leg.side == "yes" else (x <= cut)
    return hits


def build_for_game(candidates: list[Leg], cfg: dict) -> list[Combo]:
    legs = sorted(candidates, key=lambda l: l.ev, reverse=True)
    # one leg per group (the best-EV one), then cap the pool
    seen, pool = set(), []
    for l in legs:
        g = _group(l)
        if g not in seen:
            seen.add(g)
            pool.append(l)
    pool = pool[: int(cfg["candidates_per_game"])]
    if len(pool) < int(cfg["min_legs"]):
        return []

    hits = simulate_hits(pool, cfg)
    combos: list[Combo] = []
    for k in range(int(cfg["min_legs"]), min(int(cfg["max_legs"]), len(pool)) + 1):
        for ids in itertools.combinations(range(len(pool)), k):
            chosen = [pool[i] for i in ids]
            p = float(hits[:, list(ids)].all(axis=1).mean())
            if p <= 0:
                continue
            ask_cost = float(np.prod([l.all_in for l in chosen]))
            combos.append(Combo(
                legs=chosen, fair_prob=p,
                indep_prob=float(np.prod([l.fair_prob for l in chosen])),
                kalshi_prob=float(np.prod([l.kalshi_prob for l in chosen])),
                leg_ask_payout=1.0 / ask_cost, fair_payout=1.0 / p,
                min_accept_payout=(1.0 / p) * (1.0 + float(cfg["accept_margin"])),
            ))

    combos.sort(key=lambda c: (c.ev_at_leg_asks, len(c.legs)), reverse=True)
    picked: list[Combo] = []
    for c in combos:
        ids = {l.market.ticker + l.side for l in c.legs}
        if all(len(ids & {l.market.ticker + l.side for l in p.legs}) <= int(cfg["max_shared_legs"]) for p in picked):
            picked.append(c)
        if len(picked) >= int(cfg["combos_per_game"]):
            break
    return picked
