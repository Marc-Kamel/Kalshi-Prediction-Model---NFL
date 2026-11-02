"""Turn sportsbook quotes into a fair (no-vig) probability for any Kalshi market."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from scipy import stats
from scipy.optimize import brentq

from . import teams
from .kalshi import KalshiMarket
from .names import norm
from .odds import Quote
from .odds_math import american_to_prob, devig_one_sided, devig_power

PROP_MARKETS = {
    "player_receptions": "receptions",
    "player_reception_yds": "rec_yds",
    "player_rush_yds": "rush_yds",
    "player_pass_yds": "pass_yds",
    "player_anytime_td": "td",
}


def _stat_of(market_key: str) -> str | None:
    return PROP_MARKETS.get(market_key.replace("_alternate", ""))


@dataclass
class Fair:
    p: float                  # fair probability that the Kalshi market resolves YES
    n_books: int
    by_book: dict[str, float] = field(default_factory=dict)


class BookLines:
    """Per-book, vig-free probabilities built from raw quotes for one game."""

    def __init__(self, quotes: list[Quote], one_sided_margin: float):
        self.props: dict[tuple[str, str], dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
        self.ml: dict[str, dict[str, float]] = defaultdict(dict)                    # book -> team -> p
        self.spreads: dict[str, list[tuple[str, float, float]]] = defaultdict(list)  # book -> (team, point, p_cover)
        self.totals: dict[str, list[tuple[float, float]]] = defaultdict(list)        # book -> (point, p_over)
        self._build(quotes, one_sided_margin)

    def _build(self, quotes: list[Quote], m1: float) -> None:
        groups: dict[tuple, dict[str, float]] = defaultdict(dict)
        for q in quotes:
            try:
                p = american_to_prob(q.price)
            except ValueError:
                continue
            if q.market == "h2h":
                groups[(q.book, "h2h")][q.name] = p
            elif q.market in ("spreads", "alternate_spreads") and q.point is not None:
                groups[(q.book, "spreads", abs(float(q.point)))][f"{q.name}|{q.point}"] = p
            elif q.market in ("totals", "alternate_totals") and q.point is not None:
                groups[(q.book, "totals", float(q.point))][q.name.lower()] = p
            else:
                stat = _stat_of(q.market)
                if stat is None or not q.description:
                    continue
                pt = 0.5 if stat == "td" else q.point
                if pt is None:
                    continue
                groups[(q.book, stat, norm(q.description), float(pt))][q.name.lower()] = p

        for key, sides in groups.items():
            book, kind = key[0], key[1]
            if kind == "h2h" and len(sides) == 2:
                (ta, pa), (tb, pb) = sides.items()
                fa, fb = devig_power(pa, pb)
                for t, f in ((ta, fa), (tb, fb)):
                    c = teams.from_name(t)
                    if c:
                        self.ml[book][c] = f
            elif kind == "spreads" and len(sides) == 2:
                (ka, pa), (kb, pb) = sides.items()
                fa, fb = devig_power(pa, pb)
                for k, f in ((ka, fa), (kb, fb)):
                    name, pt = k.rsplit("|", 1)
                    c = teams.from_name(name)
                    if c:
                        self.spreads[book].append((c, float(pt), f))
            elif kind == "totals" and {"over", "under"} <= sides.keys():
                f_over, _ = devig_power(sides["over"], sides["under"])
                self.totals[book].append((key[2], f_over))
            elif kind in ("receptions", "rec_yds", "rush_yds", "pass_yds", "td"):
                _, stat, player, pt = key
                yes = sides.get("over", sides.get("yes"))
                no = sides.get("under", sides.get("no"))
                if yes is None:
                    continue
                f = devig_power(yes, no)[0] if no is not None else devig_one_sided(yes, m1)
                self.props[(stat, player)][book].append((pt, f))


# ---------- distributions used to move a book's line to a Kalshi rung ----------

def _gamma_sf_from_line(line: float, p_over: float, target: float, cv: float) -> float:
    k = 1.0 / cv ** 2

    def f(mu: float) -> float:
        return stats.gamma.sf(line, a=k, scale=mu / k) - p_over

    mu = brentq(f, 1e-3, 5000.0)
    return float(stats.gamma.sf(target, a=k, scale=mu / k))


def _count_sf(x: float, mu: float, disp: float) -> float:
    n = math.floor(x)  # P(X > x) == P(X >= n+1)
    if disp <= 1.0001:
        return float(stats.poisson.sf(n, mu))
    r = mu / (disp - 1.0)
    return float(stats.nbinom.sf(n, r, 1.0 / disp))


def _count_sf_from_line(line: float, p_over: float, target: float, disp: float) -> float:
    mu = brentq(lambda m: _count_sf(line, m, disp) - p_over, 1e-4, 60.0)
    return _count_sf(target, mu, disp)


def _td_from_anytime(p1: float, target: float) -> float:
    if target < 1:
        return p1
    lam = -math.log(max(1e-9, 1.0 - p1))
    k = math.floor(target)  # P(X >= k+1)
    return float(stats.poisson.sf(k, lam))


# ---------- public ----------

class FairPricer:
    def __init__(self, lines: BookLines, model_cfg: dict, odds_cfg: dict):
        self.lines = lines
        self.m = model_cfg
        self.weights = odds_cfg.get("book_weights", {}) or {}
        self.min_books = int(odds_cfg.get("min_books", 2))

    def _can_extrapolate(self, kind: str, line: float, target: float) -> bool:
        """Moving a book's line to a far-away Kalshi rung relies on the distribution's tail,
        which is unreliable — especially for low-volume players (a backup RB's receiving
        yards, a QB's rushing) whose games are mostly small with the odd big play."""
        rule = (self.m.get("extrapolation") or {}).get(kind)
        if not rule:
            return True
        min_line = float(rule.get("min_line", 0))
        if min_line > 0 and line < min_line:
            return False
        band = max(float(rule.get("abs", 0)), float(rule.get("rel", 0)) * line)
        return abs(target - line) <= band + 1e-9

    def _blend(self, by_book: dict[str, float]) -> Fair | None:
        by_book = {b: p for b, p in by_book.items() if p is not None and 0 < p < 1}
        if len(by_book) < self.min_books:
            return None
        w = {b: float(self.weights.get(b, 1.0)) for b in by_book}
        p = sum(by_book[b] * w[b] for b in by_book) / sum(w.values())
        return Fair(p=p, n_books=len(by_book), by_book=by_book)

    def price(self, km: KalshiMarket) -> Fair | None:
        try:
            return self._price(km)
        except (ValueError, RuntimeError):  # root-finding failed on a degenerate quote
            return None

    def _price(self, km: KalshiMarket) -> Fair | None:
        L = self.lines
        if km.kind == "moneyline":
            return self._blend({b: t.get(km.team) for b, t in L.ml.items()})
        if km.kind == "spread":
            sd = self.m["margin_sd"]
            out = {}
            for b, rows in L.spreads.items():
                mine = [(pt, p) for t, pt, p in rows if t == km.team]
                if not mine:
                    continue
                pt, p = min(mine, key=lambda r: abs(-r[0] - km.threshold))  # line nearest the rung
                if not self._can_extrapolate("spread", -pt, km.threshold):
                    continue
                mu = -pt + sd * stats.norm.ppf(p)          # expected margin for km.team
                out[b] = float(stats.norm.sf(km.threshold, loc=mu, scale=sd))
            return self._blend(out)
        if km.kind == "total":
            sd = self.m["total_sd"]
            out = {}
            for b, rows in L.totals.items():
                pt, p = min(rows, key=lambda r: abs(r[0] - km.threshold))
                if not self._can_extrapolate("total", pt, km.threshold):
                    continue
                mu = pt + sd * stats.norm.ppf(p)
                out[b] = float(stats.norm.sf(km.threshold, loc=mu, scale=sd))
            return self._blend(out)

        books = L.props.get((km.kind, norm(km.player or "")))
        if not books:
            return None
        out = {}
        for b, rows in books.items():
            if km.kind == "td":
                p1 = next((p for pt, p in rows if pt == 0.5), None)
                out[b] = None if p1 is None else _td_from_anytime(p1, km.threshold)
                continue
            line, p = min(rows, key=lambda r: abs(r[0] - km.threshold))
            if abs(line - km.threshold) < 1e-9:
                out[b] = p
            elif not self._can_extrapolate(km.kind, line, km.threshold):
                continue  # rung too far from this book's line (or low-volume player): don't guess
            elif km.kind == "receptions":
                out[b] = _count_sf_from_line(line, p, km.threshold, self.m["receptions_dispersion"])
            else:
                out[b] = _gamma_sf_from_line(line, p, km.threshold, self.m["cv"][km.kind])
        return self._blend(out)
