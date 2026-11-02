"""Compare Kalshi prices (plus fees) with fair probabilities, on both YES and NO."""
from __future__ import annotations

from dataclasses import dataclass

from .fair import Fair, FairPricer
from .kalshi import KalshiMarket
from .odds_math import kalshi_fee_per_contract


@dataclass
class Leg:
    market: KalshiMarket
    side: str            # "yes" | "no"
    ask: float           # price you pay for this side
    fee: float           # Kalshi taker fee per contract
    kalshi_prob: float   # Kalshi's own probability for this side (mid)
    fair_prob: float     # sportsbook no-vig probability for this side
    n_books: int
    one_sided: bool = False  # fair price relied on a guessed vig (Yes-only quotes)
    dist: float = 0.0        # distance of the rung from the books' lines (log scale)

    @property
    def all_in(self) -> float:
        return self.ask + self.fee

    @property
    def edge(self) -> float:          # probability points
        return self.fair_prob - self.all_in

    @property
    def ev(self) -> float:            # profit per $1 staked
        return self.fair_prob / self.all_in - 1.0

    @property
    def label(self) -> str:
        return self.market.label(self.side)

    @property
    def p_yes(self) -> float:         # fair P(YES) regardless of side, for the simulator
        return self.fair_prob if self.side == "yes" else 1.0 - self.fair_prob

    def as_row(self) -> dict:
        m = self.market
        return {
            "game": m.game, "leg": self.label, "side": self.side, "kind": m.kind,
            "player": m.player or "", "team": m.team or "", "ticker": m.ticker,
            "kalshi_prob": round(self.kalshi_prob, 4), "ask": round(self.ask, 4), "fee": round(self.fee, 4),
            "all_in": round(self.all_in, 4), "fair_prob": round(self.fair_prob, 4),
            "edge_pts": round(self.edge * 100, 2), "ev_pct": round(self.ev * 100, 2), "books": self.n_books,
            "one_sided": self.one_sided, "dist_from_line": round(self.dist, 3),
        }


def evaluate(markets: list[KalshiMarket], pricer: FairPricer, stake: float, fee_rate: float) -> list[Leg]:
    legs: list[Leg] = []
    for km in markets:
        fair: Fair | None = pricer.price(km)
        if fair is None:
            continue
        for side in ("yes", "no"):
            if side == "yes":
                ask, mid, fp = km.yes_ask, km.yes_mid, fair.p
            else:
                ask, mid, fp = km.no_ask, 1.0 - km.yes_mid, 1.0 - fair.p
            if not (0.0 < ask < 1.0):
                continue
            legs.append(Leg(km, side, ask, kalshi_fee_per_contract(ask, stake, fee_rate), mid, fp, fair.n_books,
                            fair.one_sided, fair.dist))
    return legs


def shape_buffer(dist: float, table: list | None) -> float:
    """Extra edge required for rungs far from the books' lines, where the shape is less certain."""
    for upto, extra in table or []:
        if dist <= upto + 1e-12:
            return float(extra)
    return float(table[-1][1]) if table else 0.0


def required_edge(leg: Leg, f: dict) -> float:
    return (f["min_edge"] + (f.get("one_sided_buffer", 0.0) if leg.one_sided else 0.0)
            + shape_buffer(leg.dist, f.get("shape_buffer")))


def passes(leg: Leg, f: dict) -> bool:
    need = required_edge(leg, f)
    return (leg.kalshi_prob >= f["min_kalshi_prob"] and leg.fair_prob >= f["min_fair_prob"]
            and need < leg.edge <= f.get("max_edge", 1.0))


def suspicious(leg: Leg, f: dict) -> bool:
    return leg.kalshi_prob >= f["min_kalshi_prob"] and leg.edge > f.get("max_edge", 1.0)
