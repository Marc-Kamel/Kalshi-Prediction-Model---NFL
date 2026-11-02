"""Monte Carlo same-game combo builder (player markets).

Legs in one game are correlated (a QB's passing yards and his WR's receiving yards rise
together; a "No TD" leg wins more often when the offence stalls). Each player stat gets one
latent performance score built from shared factors:

    team offense (OFF_T)   team pass volume (PASS_T)   player's own game (PL_p)

plus noise. Every rung of the same player stat reads the SAME score, so "30+ rushing" and
"No 80+ rushing" behave exactly like the 30–79 range they describe. A leg wins when the
score clears the cut that reproduces its fair probability, so each leg keeps its exact fair
chance; only the joint behaviour comes from the factors (loadings in config.yaml).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .buildable import can_add
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
    sim_hits: int = 0         # simulated games the combo won (precision of fair_prob)
    kalshi_corr: float = 0.0  # chance implied by Kalshi's own leg prices, with correlations

    @property
    def ev_at_leg_asks(self) -> float:
        return self.fair_prob * self.leg_ask_payout - 1.0

    @property
    def n_ranges(self) -> int:
        return sum(1 for a in self.legs for b in self.legs
                   if a.side == "yes" and b.side == "no" and _same_stat(a, b))

    def as_row(self) -> dict:
        return {
            "game": self.legs[0].market.game, "n_legs": len(self.legs), "ranges": self.n_ranges,
            "legs": " + ".join(l.label for l in self.legs),
            "fair_prob": round(self.fair_prob, 6), "indep_prob": round(self.indep_prob, 6),
            "kalshi_prob": round(self.kalshi_prob, 6),
            "fair_payout_x": round(self.fair_payout, 2),
            "accept_if_payout_x_at_least": round(self.min_accept_payout, 2),
            "max_price_per_1_payout": round(1.0 / self.min_accept_payout, 6),
            "leg_ask_payout_x": round(self.leg_ask_payout, 2),
            "kalshi_implied_payout_x": round(1 / self.kalshi_corr, 2) if self.kalshi_corr > 0 else None,
            "ev_pct_at_leg_asks": round(self.ev_at_leg_asks * 100, 2),
            "sim_hits": self.sim_hits,
            "tickers": ",".join(l.market.ticker for l in self.legs),
            "sides": ",".join(l.side for l in self.legs),
        }


def _same_stat(a: Leg, b: Leg) -> bool:
    ma, mb = a.market, b.market
    return ma.game == mb.game and ma.kind == mb.kind and ma.player == mb.player and ma.player is not None


def compatible(a: Leg, b: Leg, allow_ranges: bool = True) -> bool:
    """Can these two legs sit in one combo?"""
    if a.market.ticker == b.market.ticker:
        return False
    if a.market.player is None or b.market.player is None:   # game lines: one per kind
        return a.market.kind != b.market.kind or a.market.game != b.market.game
    if not _same_stat(a, b):
        return True                       # different stat or player: fine
    if not allow_ranges or a.side == b.side:
        return False                      # two Yes rungs (or two No rungs) are redundant
    yes, no = (a, b) if a.side == "yes" else (b, a)
    return yes.market.threshold < no.market.threshold   # Yes 30+ & No 80+ = 30–79; else impossible


def _loadings(m, idx: dict[str, int], cfg: dict) -> np.ndarray:
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
    if ss >= 0.98:  # keep room for stat-specific noise
        w *= np.sqrt(0.98 / ss)
    return w


DEFAULT_RHO = {"same_player": 0.5, "teammate": 0.0, "opponent": 0.0}
FALLBACK_POS = {"pass_yds": "QB", "rush_yds": "RB", "rr_yds": "RB"}


def _pos_for(shapes, player: str, kind: str) -> str:
    from .history import STAT_POS
    pos = shapes.position(player) if shapes is not None else ""
    allowed = STAT_POS.get(kind, ())
    if pos in allowed:
        return pos
    if pos == "FB" and "RB" in allowed:
        return "RB"
    return FALLBACK_POS.get(kind, "WR")


def _nearest_corr(C: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh((C + C.T) / 2)
    C2 = V @ np.diag(np.clip(w, 1e-6, None)) @ V.T
    d = np.sqrt(np.diag(C2))
    return C2 / np.outer(d, d)


def historical_corr(keys: list[tuple], shapes) -> np.ndarray:
    """Correlation matrix between player-stat latents, measured from past games."""
    k = len(keys)
    C = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            (ka, pa, ta), (kb, pb, tb) = keys[i], keys[j]
            rel = "same_player" if pa == pb else ("teammate" if ta and ta == tb else "opponent")
            r = shapes.rho(ka, _pos_for(shapes, pa, ka), kb, _pos_for(shapes, pb, kb), rel)
            C[i, j] = C[j, i] = DEFAULT_RHO[rel] if r is None else r
    return _nearest_corr(C)


def simulate_hits(legs: list[Leg], cfg: dict, shapes=None) -> np.ndarray:
    """Boolean matrix (sims x legs): did each leg win in each simulated game.
    Uses correlations measured from history when available (Gaussian copula), else the
    factor model with the loadings in config.yaml."""
    if shapes is not None and getattr(shapes, "corr", None) and all(l.market.player for l in legs):
        return _simulate_copula(legs, cfg, shapes)
    return _simulate_factor(legs, cfg)


def _simulate_copula(legs: list[Leg], cfg: dict, shapes) -> np.ndarray:
    keys, index = [], {}
    for l in legs:
        k = (l.market.kind, l.market.player, l.market.team)
        if k not in index:
            index[k] = len(keys)
            keys.append(k)
    L = np.linalg.cholesky(historical_corr(keys, shapes))
    rng = np.random.default_rng(cfg.get("seed", 7))
    n = int(cfg["sims"])
    Z = rng.standard_normal((n, len(keys))) @ L.T
    hits = np.empty((n, len(legs)), dtype=bool)
    for j, leg in enumerate(legs):
        x = Z[:, index[(leg.market.kind, leg.market.player, leg.market.team)]]
        cut = stats.norm.ppf(1.0 - leg.p_yes)
        hits[:, j] = (x > cut) if leg.side == "yes" else (x <= cut)
    return hits


def _simulate_factor(legs: list[Leg], cfg: dict) -> np.ndarray:
    m0 = legs[0].market
    names = [f"OFF_{m0.away}", f"OFF_{m0.home}", f"PASS_{m0.away}", f"PASS_{m0.home}"]
    names += sorted({f"PL_{l.market.player}" for l in legs if l.market.player})
    idx = {n: i for i, n in enumerate(names)}
    rng = np.random.default_rng(cfg.get("seed", 7))
    n = int(cfg["sims"])
    F = rng.standard_normal((n, len(names)))
    latent: dict[tuple, np.ndarray] = {}
    hits = np.empty((n, len(legs)), dtype=bool)
    for j, leg in enumerate(legs):
        m = leg.market
        key = (m.kind, m.player) if m.player else (m.kind, m.ticker)
        if key not in latent:  # one score per player stat, shared by all its rungs
            w = _loadings(m, idx, cfg)
            resid = np.sqrt(max(1e-6, 1.0 - float(np.sum(w ** 2))))
            latent[key] = F @ w + resid * rng.standard_normal(n)
        x = latent[key]
        cut = stats.norm.ppf(1.0 - leg.p_yes)
        hits[:, j] = (x > cut) if leg.side == "yes" else (x <= cut)
    return hits


def _make(chosen: list[Leg], mask: np.ndarray, cfg: dict) -> Combo:
    h = int(mask.sum())
    p = h / mask.size
    ask_cost = float(np.prod([l.all_in for l in chosen]))
    return Combo(legs=list(chosen), fair_prob=p,
                 indep_prob=float(np.prod([l.fair_prob for l in chosen])),
                 kalshi_prob=float(np.prod([l.kalshi_prob for l in chosen])),
                 leg_ask_payout=1.0 / ask_cost, fair_payout=1.0 / p,
                 min_accept_payout=(1.0 / p) * (1.0 + float(cfg["accept_margin"])), sim_hits=h)


def _greedy_path(pool: list[Leg], hits: np.ndarray, cfg: dict) -> list[tuple[int, np.ndarray]]:
    """Add legs one at a time, each time the leg that keeps the combo most +EV (highest chance
    of still winning relative to Kalshi's price for it). Returns every step, so a k-leg combo is the first
    k steps. Stops at max_legs, when nothing compatible is left, or when the combo would win in
    fewer than min_sim_hits simulated games (its chance would be too noisy to trust)."""
    allow = bool(cfg.get("allow_ranges", True))
    min_hits = int(cfg.get("min_sim_hits", 300))
    limit = int(cfg.get("max_legs", 40))
    mask = np.ones(hits.shape[0], dtype=bool)
    chosen: list[int] = []
    path: list[tuple[int, np.ndarray]] = []
    while len(chosen) < limit:
        base = max(1, int(mask.sum()))
        best, best_score, best_mask = None, -1.0, None
        for j in range(len(pool)):
            if j in chosen or not all(compatible(pool[j], pool[i], allow) for i in chosen):
                continue
            if not can_add([pool[i] for i in chosen], pool[j]):   # Kalshi refuses implied/impossible legs
                continue
            m = mask & hits[:, j]
            c = int(m.sum())
            if c < min_hits:
                continue
            score = (c / base) / pool[j].kalshi_prob   # conditional win rate per Kalshi price
            if score > best_score:
                best, best_score, best_mask = j, score, m
        if best is None:
            break
        chosen.append(best)
        mask = best_mask
        path.append((best, mask))
    return path


def pick_rungs(legs: list[Leg], rule: str = "highest_payout", per_side: bool = True) -> list[Leg]:
    """When several lines of the same player stat qualify, keep one.
    highest_payout: the least likely qualifying line (biggest payout), e.g. No 70+ at 61% over No 80+ at 87%.
    best_ev: the line with the best EV.
    per_side=True keeps one Yes and one No per player stat (so ranges stay possible)."""
    if rule not in ("highest_payout", "best_ev"):
        raise ValueError(f"unknown rung_choice: {rule}")
    best: dict[tuple, Leg] = {}
    for l in legs:
        m = l.market
        if not m.player:
            best[(m.ticker, l.side)] = l
            continue
        key = (m.game, m.player, m.kind) + ((l.side,) if per_side else ())
        cur = best.get(key)
        if cur is None:
            best[key] = l
        elif rule == "best_ev" and l.ev > cur.ev:
            best[key] = l
        elif rule == "highest_payout" and (l.kalshi_prob, -l.ev) < (cur.kalshi_prob, -cur.ev):
            best[key] = l
    return list(best.values())


def build_for_game(candidates: list[Leg], cfg: dict, shapes=None) -> list[Combo]:
    candidates = pick_rungs(candidates, cfg.get("rung_choice", "highest_payout"), per_side=True)
    pool = sorted(candidates, key=lambda l: l.ev, reverse=True)[: int(cfg["candidates_per_game"])]
    if len(pool) < int(cfg["min_legs"]):
        return []
    hits = simulate_hits(pool, cfg, shapes)
    path = _greedy_path(pool, hits, cfg)
    sizes = cfg.get("sizes") or [cfg.get("max_legs", 4)]
    out: list[Combo] = []
    done: set[int] = set()
    for s in sizes:
        k = len(path) if int(s) == 0 else int(s)
        if k < int(cfg["min_legs"]) or k > len(path) or k in done:
            continue
        done.add(k)
        out.append(_make([pool[i] for i, _ in path[:k]], path[k - 1][1], cfg))
    if out:  # what Kalshi's own leg prices imply for each combo (same correlations)
        import dataclasses
        kal = [dataclasses.replace(l, fair_prob=l.kalshi_prob) for l in pool]
        kh = simulate_hits(kal, cfg, shapes)
        idx = {id(l): i for i, l in enumerate(pool)}
        for cb in out:
            cols = [idx[id(l)] for l in cb.legs]
            cb.kalshi_corr = float(kh[:, cols].all(axis=1).mean())
    return sorted(out, key=lambda c: len(c.legs))
