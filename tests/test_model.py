from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from kalshicombos import pipeline, teams
from kalshicombos.combos import simulate_hits
from kalshicombos.edges import Leg
from kalshicombos.fair import _count_sf_from_line, _gamma_sf_from_line, _td_from_anytime
from kalshicombos.kalshi import parse_market
from kalshicombos.names import norm
from kalshicombos.odds_math import american_to_prob, devig_power, kalshi_fee_per_contract

FIX = Path(__file__).parent / "fixtures" / "chisea"
CFG = pipeline.load_config(Path(__file__).parents[1] / "config.yaml")
NOW = datetime(2026, 11, 2, 17, 0, tzinfo=timezone.utc)


def test_american_and_devig():
    assert american_to_prob(-110) == pytest.approx(0.5238, abs=1e-4)
    assert american_to_prob(150) == pytest.approx(0.4)
    a, b = devig_power(american_to_prob(-110), american_to_prob(-110))
    assert a == pytest.approx(0.5) and a + b == pytest.approx(1.0)
    fav, dog = devig_power(american_to_prob(-300), american_to_prob(250))
    assert fav + dog == pytest.approx(1.0) and fav > 0.7


def test_kalshi_fee_rounds_up_per_order():
    # 10 contracts at 50c: 0.07*10*0.25 = $0.175 -> 18c -> 1.8c each
    assert kalshi_fee_per_contract(0.50, 5.0) == pytest.approx(0.018)
    assert kalshi_fee_per_contract(0.99, 10) > 0  # never free for a taker


def test_team_split_and_names():
    assert teams.split_matchup("CHISEA") == ("CHI", "SEA")
    assert teams.split_matchup("NYJKC") == ("NYJ", "KC")
    assert teams.split_matchup("CLENO") == ("CLE", "NO")
    assert norm("A.J. Barner") == norm("AJ Barner")
    assert norm("Luther Burden III") == norm("Luther Burden")


def test_parse_real_kalshi_shapes():
    m = {"ticker": "KXNFLRECYDS-26NOV02CHISEA-CHIDSWIFT4-30", "event_ticker": "KXNFLRECYDS-26NOV02CHISEA",
         "title": "D'Andre Swift: 30+ receiving yards", "yes_sub_title": "D'Andre Swift: 30+", "floor_strike": 29.5,
         "yes_bid_dollars": "0.2800", "yes_ask_dollars": "0.3100", "no_bid_dollars": "0.6900", "no_ask_dollars": "0.7200"}
    k = parse_market(m, "rec_yds")
    assert (k.player, k.team, k.threshold, k.away, k.home) == ("D'Andre Swift", "CHI", 29.5, "CHI", "SEA")
    s = parse_market({"ticker": "KXNFLSPREAD-26NOV02CHISEA-SEA11", "event_ticker": "KXNFLSPREAD-26NOV02CHISEA",
                      "title": "Seattle wins by over 10.5 points?", "floor_strike": 10.5,
                      "yes_bid_dollars": "0.25", "yes_ask_dollars": "0.26"}, "spread")
    assert (s.team, s.threshold, s.no_ask) == ("SEA", 10.5, pytest.approx(0.75))
    dst = parse_market({"ticker": "KXNFLTD-26NOV02CHISEA-SEASEADST-1", "event_ticker": "KXNFLTD-26NOV02CHISEA",
                        "title": "SEA Seahawks D/ST: 1+ touchdowns", "yes_sub_title": "SEA Seahawks D/ST: 1+",
                        "floor_strike": 0.5, "yes_bid_dollars": "0.14", "yes_ask_dollars": "0.15"}, "td")
    assert dst is None


def test_rung_pricing_is_monotone_and_exact_at_line():
    p_at = _gamma_sf_from_line(64.5, 0.5, 64.5, 0.8)
    assert p_at == pytest.approx(0.5, abs=1e-6)
    rungs = [_gamma_sf_from_line(64.5, 0.5, t - 0.5, 0.8) for t in (40, 60, 80, 100)]
    assert all(a > b for a, b in zip(rungs, rungs[1:]))
    assert _count_sf_from_line(4.5, 0.5, 4.5, 1.15) == pytest.approx(0.5, abs=1e-6)
    assert _td_from_anytime(0.5, 0.5) == 0.5
    assert _td_from_anytime(0.5, 1.5) == pytest.approx(1 - 0.5 * (1 + np.log(2)), abs=1e-6)


def _leg(km_kwargs, side, p_yes):
    from kalshicombos.kalshi import KalshiMarket
    km = KalshiMarket(ticker=km_kwargs.get("ticker", "T"), game="CHISEA", date="26NOV02", away="CHI", home="SEA",
                      title="t", yes_bid=0.5, yes_ask=0.51, no_bid=0.49, no_ask=0.5, **{k: v for k, v in km_kwargs.items() if k != "ticker"})
    fair = p_yes if side == "yes" else 1 - p_yes
    return Leg(km, side, 0.5, 0.01, fair, fair, 3)


def test_simulation_keeps_each_legs_fair_chance_and_correlates():
    cfg = dict(CFG["combos"], sims=200000)
    qb = _leg({"ticker": "A", "kind": "pass_yds", "team": "SEA", "player": "Sam Darnold", "threshold": 224.5}, "yes", 0.7)
    wr = _leg({"ticker": "B", "kind": "rec_yds", "team": "SEA", "player": "JSN", "threshold": 79.5}, "yes", 0.65)
    no_td = _leg({"ticker": "C", "kind": "td", "team": "SEA", "player": "JSN2", "threshold": 0.5}, "no", 0.45)
    hits = simulate_hits([qb, wr, no_td], cfg)
    assert hits[:, 0].mean() == pytest.approx(0.70, abs=0.005)
    assert hits[:, 1].mean() == pytest.approx(0.65, abs=0.005)
    assert hits[:, 2].mean() == pytest.approx(0.55, abs=0.005)
    assert (hits[:, 0] & hits[:, 1]).mean() > 0.70 * 0.65 + 0.01        # QB + WR overs go together
    assert (hits[:, 0] & hits[:, 2]).mean() < 0.70 * 0.55                 # "No TD" fights the QB over


def test_end_to_end_fixture(tmp_path):
    kg, og = pipeline.fixture_getters(FIX)
    res = pipeline.run(CFG, kg, og, now=NOW)
    assert len(res.games) == 1
    g = res.games[0]
    assert g.game == "CHISEA" and g.legs
    for l in g.qualifying:
        assert l.kalshi_prob >= 0.60 and l.fair_prob >= 0.60 and l.edge > 0
    assert not any("D/ST" in l.label for l in g.legs)
    for c in g.combos:
        assert 2 <= len(c.legs) <= CFG["combos"]["max_legs"]
        assert c.min_accept_payout > c.fair_payout
        keys = [(l.market.kind, l.market.player) for l in c.legs if l.market.player]
        assert len(keys) == len(set(keys))  # never two rungs of the same player stat
    from kalshicombos import report
    out = report.write(res, tmp_path)
    assert (out / "combos.csv").exists() and (tmp_path / "latest.md").read_text().startswith("# Kalshi NFL combos")


def test_started_games_are_skipped():
    kg, og = pipeline.fixture_getters(FIX)
    late = datetime(2026, 11, 3, 2, 0, tzinfo=timezone.utc)
    assert pipeline.run(CFG, kg, og, now=late).games == []
