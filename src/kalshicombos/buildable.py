"""Can Kalshi build this combo?

Kalshi refuses a combo when one leg is already guaranteed by the others ("D'Andre Swift fewer than
30 receiving yards and fewer than 80 rushing yards already guarantee NO 115+ rushing and receiving
yards combined") or when the legs can't all be true at once. Pairwise checks miss this, because the
link runs across related stats of one player:

    rush + receiving = rush+rec yards,   a catch is needed for any receiving yards.

Each player's legs become integer bounds on those stats; bounds are pushed through the identities
until they stop moving. A leg is redundant when the other legs' bounds already decide it; the set is
contradictory when a bound is empty. Stats are treated as never negative, which is the conservative
choice (it can only flag more conflicts, not fewer).
"""
from __future__ import annotations

import math
from typing import Iterable

from .edges import Leg

LINKED = ("rush_yds", "rec_yds", "rr_yds", "receptions")
# Stats where Kalshi allows only ONE line per player in a combo (no ranges). Kalshi: "you can only
# choose one reception line per player". Add other stats here if Kalshi says the same about them.
SINGLE_LINE_KINDS = {"receptions"}
INF = float("inf")


def _rung(leg: Leg) -> tuple[str, int]:
    """('min', n) for Yes on "n+" (stat >= n), ('max', n-1) for No (stat <= n-1)."""
    n = math.floor(float(leg.market.threshold)) + 1      # threshold 29.5 -> "30+"
    return ("min", n) if leg.side == "yes" else ("max", n - 1)


def _bounds(legs: Iterable[Leg]) -> dict[str, list[float]] | None:
    """Tightest bounds the legs imply, or None if they contradict each other."""
    b = {k: [0.0, INF] for k in LINKED}
    for l in legs:
        kind = l.market.kind
        if kind not in b or l.market.threshold is None:
            continue
        which, n = _rung(l)
        if which == "min":
            b[kind][0] = max(b[kind][0], n)
        else:
            b[kind][1] = min(b[kind][1], n)
    rs, rc, rr, cat = b["rush_yds"], b["rec_yds"], b["rr_yds"], b["receptions"]
    for _ in range(6):
        before = [tuple(v) for v in b.values()]
        rr[0] = max(rr[0], rs[0] + rc[0]); rr[1] = min(rr[1], rs[1] + rc[1])
        rs[0] = max(rs[0], rr[0] - rc[1]); rs[1] = min(rs[1], rr[1] - rc[0])
        rc[0] = max(rc[0], rr[0] - rs[1]); rc[1] = min(rc[1], rr[1] - rs[0])
        if rc[0] >= 1:
            cat[0] = max(cat[0], 1)          # receiving yards need a catch
        if cat[1] <= 0:
            rc[1] = min(rc[1], 0)            # no catch, no receiving yards
        if any(lo > hi for lo, hi in b.values()):
            return None
        if [tuple(v) for v in b.values()] == before:
            break
    return b


def _decided(leg: Leg, b: dict[str, list[float]]) -> bool:
    """Do these bounds already settle this leg (either way)?"""
    kind = leg.market.kind
    if kind not in b or leg.market.threshold is None:
        return False
    n = math.floor(float(leg.market.threshold)) + 1
    lo, hi = b[kind]
    return lo >= n or hi <= n - 1


def problems(legs: list[Leg]) -> list[str]:
    """Why Kalshi would refuse these legs as one combo (empty list = buildable)."""
    out: list[str] = []
    seen: set[str] = set()
    for l in legs:
        if l.market.ticker in seen:
            out.append(f"same market twice: {l.market.ticker}")
        seen.add(l.market.ticker)
    for i, a in enumerate(legs):
        for b in legs[i + 1:]:
            if _one_line_clash(a, b):
                out.append(f"{a.market.player}: Kalshi allows only one {a.market.kind} line per player")
    by_player: dict[tuple, list[Leg]] = {}
    for l in legs:
        if l.market.player and l.market.kind in LINKED:
            by_player.setdefault((l.market.game, l.market.player), []).append(l)
    for (_, player), group in by_player.items():
        if _bounds(group) is None:
            out.append(f"{player}: legs can't all be true together")
            continue
        for i, l in enumerate(group):
            rest = group[:i] + group[i + 1:]
            if rest and _decided(l, _bounds(rest)):
                out.append(f"{player}: '{l.label}' is already guaranteed (or ruled out) by the other legs")
    return out


def _one_line_clash(a: Leg, b: Leg) -> bool:
    ma, mb = a.market, b.market
    return (ma.kind in SINGLE_LINE_KINDS and ma.kind == mb.kind and ma.player is not None
            and ma.player == mb.player and ma.game == mb.game)


def can_add(chosen: list[Leg], cand: Leg) -> bool:
    """True if adding cand keeps the combo buildable on Kalshi (chosen is assumed buildable)."""
    if any(c.market.ticker == cand.market.ticker or _one_line_clash(c, cand) for c in chosen):
        return False
    m = cand.market
    if not m.player or m.kind not in LINKED:
        return True
    group = [c for c in chosen if c.market.player == m.player and c.market.game == m.game
             and c.market.kind in LINKED]
    if not group:
        return True
    full = group + [cand]
    if _bounds(full) is None:
        return False
    for i, l in enumerate(full):
        if _decided(l, _bounds(full[:i] + full[i + 1:])):
            return False
    return True


def prune(legs: list[Leg]) -> list[Leg]:
    """Keep legs in the given order (best first), dropping any that would break buildability."""
    kept: list[Leg] = []
    for l in legs:
        if can_add(kept, l):
            kept.append(l)
    return kept
