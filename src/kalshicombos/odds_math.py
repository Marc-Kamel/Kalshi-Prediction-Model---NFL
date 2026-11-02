"""Odds conversion, vig removal and Kalshi fee maths."""
from __future__ import annotations

import math

from scipy.optimize import brentq


def american_to_prob(price: float) -> float:
    price = float(price)
    if price >= 100:
        return 100.0 / (price + 100.0)
    if price <= -100:
        return -price / (-price + 100.0)
    raise ValueError(f"not an American price: {price}")


def devig_power(p_a: float, p_b: float) -> tuple[float, float]:
    """Power method: find k with p_a**k + p_b**k == 1. Handles favourite/longshot bias
    better than simple normalisation."""
    total = p_a + p_b
    if total <= 1.0 or max(p_a, p_b) >= 1.0:  # no vig (or degenerate) — just normalise
        return p_a / total, p_b / total
    f = lambda k: p_a ** k + p_b ** k - 1.0
    hi = 50.0
    while f(hi) > 0 and hi < 1e5:  # very lopsided pairs need a bigger exponent
        hi *= 4
    if f(hi) > 0:
        return p_a / total, p_b / total
    k = brentq(f, 1.0, hi)
    return p_a ** k, p_b ** k


def devig_one_sided(p: float, margin: float) -> float:
    """Yes-only market (e.g. anytime TD with no No price): strip an assumed margin."""
    return min(p / (1.0 + margin), 0.999)


def kalshi_fee_per_contract(price: float, stake_usd: float, rate: float = 0.07) -> float:
    """Kalshi taker fee is ceil(rate * C * P * (1 - P)) dollars-to-the-cent per order.
    Returned per contract, at the number of contracts `stake_usd` buys."""
    if price <= 0 or price >= 1:
        return 0.0
    contracts = max(1, math.floor(stake_usd / price))
    fee_cents = math.ceil(round(rate * contracts * price * (1 - price) * 100, 6))
    return fee_cents / 100.0 / contracts
