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
from dataclasses import dataclass, field
from typing import Iterable

from .edges import Leg
from .kalshi import event_of


@dataclass
class ComboRules:
    """Kalshi's published rules for one game's combos (from /multivariate_event_collections).
    Per event (e.g. KXNFLREC-26NOV02CHISEA = every receptions market in the game): may it be used at
    all, Yes side only?, max markets from it. Per combo: min/max legs. None = no limit."""
    collection: str = ""
    eligible: set[str] | None = None            # None = unknown (don't restrict)
    yes_only: set[str] = field(default_factory=set)
    event_max: dict[str, int] = field(default_factory=dict)
    size_min: int = 2
    size_max: int | None = None
    one_per_event: bool = False

    @classmethod
    def from_collections(cls, cols: list[dict], events: set[str]) -> "ComboRules | None":
        """Pick the collection that covers the most of this game's events (ties: the loosest)."""
        best, best_key = None, None
        for c in cols:
            assoc = c.get("associated_events") or [{"ticker": t} for t in c.get("associated_event_tickers", [])]
            tick = {a.get("ticker") for a in assoc}
            yes_only = tick if c.get("is_all_yes") else {a["ticker"] for a in assoc if a.get("is_yes_only")}
            key = (len(events & tick), -len(yes_only & events), -(c.get("size_min") or 0))
            if best_key is None or key > best_key:
                best_key = key
                best = cls(collection=c.get("collection_ticker", ""), eligible=tick, yes_only=yes_only,
                           event_max={a["ticker"]: int(a["size_max"]) for a in assoc if a.get("size_max")},
                           size_min=int(c.get("size_min") or 2),
                           size_max=int(c["size_max"]) if c.get("size_max") else None,   # 0 = no limit
                           one_per_event=bool(c.get("is_single_market_per_event")))
        return best

    def leg_ok(self, leg: Leg) -> bool:
        ev = event_of(leg.market.ticker)
        if self.eligible is not None and ev not in self.eligible:
            return False
        return not (leg.side == "no" and ev in self.yes_only)

    def why_not(self, leg: Leg) -> str:
        ev = event_of(leg.market.ticker)
        if self.eligible is not None and ev not in self.eligible:
            return f"{ev} isn't offered for combos"
        if leg.side == "no" and ev in self.yes_only:
            return f"{ev} allows Yes legs only"
        return ""

    def event_cap(self, ev: str) -> int | None:
        return 1 if self.one_per_event else self.event_max.get(ev)

LINKED = ("rush_yds", "rec_yds", "rr_yds", "receptions")
# One line per player per stat in a combo (no ranges), for every stat. Kalshi said so for receptions
# ("you can only choose one reception line per player"); the user chose the same for all stats.
SINGLE_LINE_KINDS = None   # None = every player stat; or a set like {"receptions"} to limit it
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


def problems(legs: list[Leg], rules: ComboRules | None = None) -> list[str]:
    """Why Kalshi would refuse these legs as one combo (empty list = buildable)."""
    out: list[str] = []
    if rules is not None:
        out += [w for w in (rules.why_not(l) for l in legs) if w]
        if len(legs) < rules.size_min:
            out.append(f"needs at least {rules.size_min} legs")
        if rules.size_max and len(legs) > rules.size_max:
            out.append(f"at most {rules.size_max} legs allowed")
        per_ev: dict[str, int] = {}
        for l in legs:
            ev = event_of(l.market.ticker)
            per_ev[ev] = per_ev.get(ev, 0) + 1
        for ev, n in per_ev.items():
            cap = rules.event_cap(ev)
            if cap is not None and n > cap:
                out.append(f"{ev}: at most {cap} market(s) from this event")
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
    return ((SINGLE_LINE_KINDS is None or ma.kind in SINGLE_LINE_KINDS) and ma.kind == mb.kind and ma.player is not None
            and ma.player == mb.player and ma.game == mb.game)


def can_add(chosen: list[Leg], cand: Leg, rules: ComboRules | None = None) -> bool:
    """True if adding cand keeps the combo buildable on Kalshi (chosen is assumed buildable)."""
    if any(c.market.ticker == cand.market.ticker or _one_line_clash(c, cand) for c in chosen):
        return False
    if rules is not None:
        if not rules.leg_ok(cand):
            return False
        if rules.size_max and len(chosen) + 1 > rules.size_max:
            return False
        ev = event_of(cand.market.ticker)
        cap = rules.event_cap(ev)
        if cap is not None and sum(event_of(c.market.ticker) == ev for c in chosen) + 1 > cap:
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


def prune(legs: list[Leg], rules: ComboRules | None = None) -> list[Leg]:
    """Keep legs in the given order (best first), dropping any that would break buildability."""
    kept: list[Leg] = []
    for l in legs:
        if can_add(kept, l, rules):
            kept.append(l)
    return kept
