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


def test_no_far_extrapolation_for_low_volume_players():
    from kalshicombos.fair import BookLines, FairPricer
    from kalshicombos.kalshi import KalshiMarket
    from kalshicombos.odds import Quote
    qs = []
    for b in ("pinnacle", "fanduel"):
        qs += [Quote(b, "player_reception_yds", "Over", "Roschon Johnson", 6.5, -115),
               Quote(b, "player_reception_yds", "Under", "Roschon Johnson", 6.5, -105),
               Quote(b, "player_reception_yds", "Over", "Jaxon Smith-Njigba", 89.5, -110),
               Quote(b, "player_reception_yds", "Under", "Jaxon Smith-Njigba", 89.5, -110)]
    fp = FairPricer(BookLines(qs, 0.07), CFG["model"], CFG["odds"])

    def km(player, thr):
        return KalshiMarket(ticker="T", game="CHISEA", date="26NOV02", away="CHI", home="SEA", kind="rec_yds",
                            title="t", yes_bid=.2, yes_ask=.21, no_bid=.79, no_ask=.8, team="CHI", player=player, threshold=thr)
    assert fp.price(km("Roschon Johnson", 14.5)) is None        # low-volume: exact line only
    assert fp.price(km("Roschon Johnson", 6.5)) is not None      # exact line is fine
    assert fp.price(km("Jaxon Smith-Njigba", 99.5)) is not None   # within 25% of 89.5
    assert fp.price(km("Jaxon Smith-Njigba", 149.5)) is None      # too far out on the tail


def test_huge_edges_are_flagged_not_suggested():
    from kalshicombos.edges import passes, suspicious
    leg = _leg({"kind": "rush_yds", "team": "SEA", "player": "Sam Darnold", "threshold": 9.5}, "no", 0.12)
    leg.kalshi_prob = 0.675
    assert leg.edge > CFG["filters"]["max_edge"]
    assert not passes(leg, CFG["filters"]) and suspicious(leg, CFG["filters"])


def test_end_to_end_fixture(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)  # keep test output out of the Actions summary
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


def test_total_ladder_uses_realistic_spread():
    from kalshicombos.fair import BookLines, FairPricer
    from kalshicombos.kalshi import KalshiMarket
    from kalshicombos.odds import Quote
    qs = []
    for b in ("pinnacle", "fanduel"):
        qs += [Quote(b, "totals", "Over", "", 46.5, -110), Quote(b, "totals", "Under", "", 46.5, -110)]
    fp = FairPricer(BookLines(qs, 0.07), CFG["model"], CFG["odds"])

    def tot(x):
        return KalshiMarket(ticker="T", game="CHISEA", date="26NOV02", away="CHI", home="SEA", kind="total",
                            title="t", yes_bid=.5, yes_ask=.51, no_bid=.49, no_ask=.5, threshold=x)
    assert fp.price(tot(48.5)).p == pytest.approx(0.443, abs=0.01)   # near the line: priced
    assert fp.price(tot(58.5)) is None                               # 12 points away: not guessed


def test_replay_reproduces_a_saved_run(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    from kalshicombos import report
    kg, og = pipeline.fixture_getters(FIX)
    res = pipeline.run(CFG, kg, og, now=NOW)
    out = report.write(res, tmp_path)
    again = pipeline.replay(out, CFG)
    a = sorted((l.market.ticker, l.side, round(l.fair_prob, 6)) for l in res.games[0].legs)
    b = sorted((l.market.ticker, l.side, round(l.fair_prob, 6)) for l in again.games[0].legs)
    assert a == b


def test_ranges_are_compatible_and_contradictions_are_not():
    from kalshicombos.combos import compatible
    y30 = _leg({"ticker": "Y30", "kind": "rush_yds", "team": "CHI", "player": "Swift", "threshold": 29.5}, "yes", 0.8)
    y50 = _leg({"ticker": "Y50", "kind": "rush_yds", "team": "CHI", "player": "Swift", "threshold": 49.5}, "yes", 0.6)
    n80 = _leg({"ticker": "N80", "kind": "rush_yds", "team": "CHI", "player": "Swift", "threshold": 79.5}, "no", 0.25)
    n20 = _leg({"ticker": "N20", "kind": "rush_yds", "team": "CHI", "player": "Swift", "threshold": 19.5}, "no", 0.9)
    rec = _leg({"ticker": "R", "kind": "rec_yds", "team": "CHI", "player": "Swift", "threshold": 14.5}, "yes", 0.6)
    assert compatible(y30, n80)            # 30–79 yards: a range
    assert not compatible(y30, y50)        # two Yes rungs: redundant
    assert not compatible(y30, n20)        # 30+ and under 20: impossible
    assert compatible(y30, rec)            # different stat, same player: fine
    assert not compatible(y30, n80, allow_ranges=False)


def test_range_probability_is_exact():
    cfg = dict(CFG["combos"], sims=300000)
    y30 = _leg({"ticker": "Y30", "kind": "rush_yds", "team": "CHI", "player": "Swift", "threshold": 29.5}, "yes", 0.80)
    n80 = _leg({"ticker": "N80", "kind": "rush_yds", "team": "CHI", "player": "Swift", "threshold": 79.5}, "no", 0.25)
    hits = simulate_hits([y30, n80], cfg)
    # P(30 <= X < 80) = P(30+) - P(80+) = 0.80 - 0.25 because both rungs read the same latent
    assert (hits[:, 0] & hits[:, 1]).mean() == pytest.approx(0.55, abs=0.004)


def test_over_only_lines_use_the_books_own_margin():
    from kalshicombos.fair import BookLines
    from kalshicombos.odds import Quote
    qs = [Quote("bk", "player_reception_yds", "Over", "X", 49.5, -115),
          Quote("bk", "player_reception_yds", "Under", "X", 49.5, -105),        # ~4.3% overround
          Quote("bk", "player_reception_yds_alternate", "Over", "X", 19.5, -500)]
    assert 19.5 not in dict(BookLines(qs, 0.07, 0.06).props[("rec_yds", "x")]["bk"])  # ignored by default
    bl = BookLines(qs, 0.07, 0.06, allow_one_sided_props=True)
    p_alt = dict(bl.props[("rec_yds", "x")]["bk"])[19.5]
    assert 0.78 < p_alt < american_to_prob(-500)   # if allowed: vig removed, but not crushed
    assert ("rec_yds", "x", "bk", 19.5) in bl.one_sided_rows


def test_yes_only_td_legs_need_extra_edge():
    from kalshicombos.edges import passes
    leg = _leg({"kind": "td", "team": "SEA", "player": "X", "threshold": 0.5}, "no", 0.15)
    leg.kalshi_prob, leg.ask, leg.fee = 0.84, 0.84, 0.01        # 0.87 fair vs 0.85 all-in: +2c
    leg.fair_prob = 0.87
    assert passes(leg, CFG["filters"])
    leg.one_sided = True                                        # vig was guessed: needs +3c more
    assert not passes(leg, CFG["filters"])


def test_player_only_and_big_combos(monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    kg, og = pipeline.fixture_getters(FIX)
    g = pipeline.run(CFG, kg, og, now=NOW).games[0]
    assert all(l.market.player for l in g.legs)                     # no game lines
    assert any(l.market.kind == "rr_yds" for l in g.legs)           # rush+rec yards priced
    sizes = [len(c.legs) for c in g.combos]
    assert sizes == sorted(sizes) and all(s >= 2 for s in sizes)
    for c in g.combos:
        from kalshicombos.combos import compatible
        assert all(compatible(a, b) for i, a in enumerate(c.legs) for b in c.legs[i + 1:])
        assert c.sim_hits >= CFG["combos"]["min_sim_hits"]


def test_devig_survives_garbage_pairs():
    a, b = devig_power(0.995, 0.99)          # both sides "favourites": must not crash
    assert a + b == pytest.approx(1.0)
    a, b = devig_power(0.9999, 0.02)         # extremely lopsided alt line
    assert 0.97 < a < 1.0 and a + b == pytest.approx(1.0, abs=1e-6)
    from kalshicombos.fair import BookLines
    from kalshicombos.odds import Quote
    bl = BookLines([Quote("bk", "player_receptions", "Over", "X", 1.5, -2000),
                    Quote("bk", "player_receptions", "Under", "X", 1.5, -1500)], 0.07)
    assert ("receptions", "x") not in bl.props   # skipped, not crashed


def test_one_bad_game_does_not_sink_the_run(monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    import kalshicombos.pipeline as pl
    def boom(*a, **k):
        raise ValueError("bad data")
    monkeypatch.setattr(pl, "_price_game", boom)
    kg, og = pipeline.fixture_getters(FIX)
    res = pl.run(CFG, kg, og, now=NOW)
    assert res.games and "pricing error" in res.games[0].note and res.games[0].raw_quotes


def test_yes_only_longshot_td_is_not_crushed():
    from kalshicombos.fair import BookLines
    from kalshicombos.odds import Quote
    bl = BookLines([Quote("dk", "player_anytime_td", "Yes", "Eric Saubert", None, 1100)], 0.07)
    p = dict(bl.props[("td", "eric saubert")]["dk"])[0.5]
    assert p == pytest.approx((100 / 1200) / 1.07, abs=1e-6)   # ~7.8%, not ~2%


def test_history_correlations_drive_the_copula():
    from kalshicombos.history import ShapeModel
    from kalshicombos.combos import historical_corr
    sm = ShapeModel({"correlations": {
        "teammate|pass_yds@QB|rec_yds@WR": {"rho": 0.22, "n": 9000},
        "same_player|rec_yds@WR|receptions@WR": {"rho": 0.77, "n": 9000}},
        "positions": {"sam darnold": "QB", "jaxon smith njigba": "WR"}})
    keys = [("pass_yds", "Sam Darnold", "SEA"), ("rec_yds", "Jaxon Smith-Njigba", "SEA"),
            ("receptions", "Jaxon Smith-Njigba", "SEA"), ("rec_yds", "Rome Odunze", "CHI")]
    C = historical_corr(keys, sm)
    assert C[0, 1] == pytest.approx(0.22, abs=0.02) and C[1, 2] == pytest.approx(0.77, abs=0.02)
    assert abs(C[1, 3]) < 1e-6                        # unmeasured opponent pair -> 0
    assert np.all(np.linalg.eigvalsh(C) > 0)          # valid correlation matrix
    cfg = dict(CFG["combos"], sims=200000)
    qb = _leg({"ticker": "A", "kind": "pass_yds", "team": "SEA", "player": "Sam Darnold", "threshold": 224.5}, "yes", 0.6)
    wr = _leg({"ticker": "B", "kind": "rec_yds", "team": "SEA", "player": "Jaxon Smith-Njigba", "threshold": 79.5}, "yes", 0.6)
    from kalshicombos.combos import simulate_hits
    h = simulate_hits([qb, wr], cfg, sm)
    assert h[:, 0].mean() == pytest.approx(0.6, abs=0.005)
    assert 0.36 + 0.02 < (h[:, 0] & h[:, 1]).mean() < 0.36 + 0.08   # modest positive link


def test_grading_settles_bets_combos_and_calibration(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    import json
    from kalshicombos import report
    from kalshicombos.grade import grade
    kg, og = pipeline.fixture_getters(FIX)
    res = pipeline.run(CFG, kg, og, now=NOW)
    report.write(res, tmp_path)
    # Settle: every YES market resolves "yes" if fair_prob >= 0.5 (deterministic fake results)
    fair = {l.market.ticker: l.p_yes for l in res.games[0].legs}

    def settled_get(path, params):
        data = kg(path, params)
        for m in data.get("markets", []):
            m["result"] = "yes" if fair.get(m["ticker"], 0) >= 0.5 else "no"
        return data
    out = grade(tmp_path, CFG["kalshi"]["base_url"], settled_get)
    s = (tmp_path / "grades" / "summary.md").read_text()
    assert out["legs_calibrated"] > 0 and "Brier score" in s
    assert out["brier_model"] < out["brier_kalshi"] + 0.5
    if res.games[0].qualifying:
        assert out["bets"] == len({(l.market.ticker, l.side) for l in res.games[0].qualifying})


def test_friend_mode_recipe(monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    from kalshicombos import friend
    cfg = dict(CFG, friend_mode=dict(CFG["friend_mode"], min_books=2, sims=200000))
    kg, og = pipeline.fixture_getters(FIX)
    fcfg = friend.friend_cfg(cfg)
    assert fcfg["odds"]["allow_one_sided_props"] and fcfg["filters"]["min_fair_prob"] == 0.55
    assert any(m.endswith("_alternate") for m in fcfg["odds"]["markets"])
    assert CFG["odds"]["allow_one_sided_props"] is False            # main config untouched
    res, combos, skipped = friend.run(cfg, ["CHISEA"], kalshi_getter=kg, odds_getter=og, now=NOW)
    assert combos or skipped
    for c in combos:
        keys = [(l.market.player, l.market.kind) for l in c.legs]
        assert len(keys) == len(set(keys))                           # one line per player-stat
        assert all(l.fair_prob > 0.55 and l.edge > 0 for l in c.legs)
        assert c.his_indep == pytest.approx(float(np.prod([l.fair_prob for l in c.legs])))
        assert c.accept_x >= 1 / c.strict_corr                       # strict basis by default
        md = friend.to_markdown(res.run_id, combos, skipped)
        assert "Accept only if" in md and "Kalshi's own leg prices" in md


def test_chunked_combo_probability_matches_single_run():
    from kalshicombos.friend import combo_probability
    from kalshicombos.combos import simulate_hits
    legs = [_leg({"ticker": f"T{i}", "kind": "rec_yds", "team": "SEA", "player": f"P{i}", "threshold": 9.5}, "yes", 0.8)
            for i in range(6)]
    cfg = dict(CFG["combos"], sims=400000)
    one = simulate_hits(legs, cfg).all(axis=1).mean()
    chunked = combo_probability(legs, [0.8] * 6, cfg, None, 400000, chunk=100000)
    assert chunked == pytest.approx(one, abs=0.01)


def test_finished_or_unknown_games_are_explained(monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    from kalshicombos import friend, report
    kg, og = pipeline.fixture_getters(FIX)
    res = pipeline.run(CFG, kg, og, now=NOW, only=["PHIWAS"])
    assert not res.games and any("PHIWAS" in n and "not an open game" in n for n in res.notes)
    assert "PHIWAS" in report.to_markdown(res)
    late = datetime(2026, 11, 3, 2, 0, tzinfo=timezone.utc)
    res2 = pipeline.run(CFG, kg, og, now=late, only=["CHISEA"])
    assert any("already kicked off" in n for n in res2.notes)
    _, combos, skipped = friend.run(dict(CFG, friend_mode=dict(CFG["friend_mode"], sims=100000)), ["PHIWAS"],
                                    kalshi_getter=kg, odds_getter=og, now=NOW)
    assert not combos and any("PHIWAS" in s for s in skipped)


def test_friend_run_keeps_strict_report_clean(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    from kalshicombos import friend, report
    kg, og = pipeline.fixture_getters(FIX)
    res, combos, skipped = friend.run(dict(CFG, friend_mode=dict(CFG["friend_mode"], min_books=2, sims=100000)),
                                      ["CHISEA"], kalshi_getter=kg, odds_getter=og, now=NOW)
    d = report.write_raw(res, tmp_path)
    friend.write(res.run_id, combos, skipped, tmp_path)
    assert (d / "book_quotes.csv.gz").exists() and (d / "kalshi_markets.csv").exists()
    assert not (d / "legs.csv").exists() and not (tmp_path / "latest.md").exists()
    if combos:
        assert "." in friend._mult(combos[0].accept_x)   # e.g. 3.28x, never a bare "3x"


def test_pick_the_biggest_payout_rung_that_still_qualifies():
    from kalshicombos.combos import pick_rungs
    from kalshicombos.edges import passes
    def no_leg(thr, kalshi, fair):
        l = _leg({"ticker": f"R{thr}", "kind": "rush_yds", "team": "SEA", "player": "RB", "threshold": thr - 0.5}, "no", 1 - fair)
        l.kalshi_prob, l.ask, l.fee = kalshi, kalshi + 0.01, 0.01
        return l
    legs = [no_leg(60, 0.40, 0.43), no_leg(70, 0.61, 0.645), no_leg(80, 0.87, 0.94)]   # 80+ has the best EV
    ok = [l for l in legs if passes(l, CFG["filters"])]            # 60+ fails the 60% rule
    assert [l.market.ticker for l in ok] == ["R70", "R80"]
    assert pick_rungs(ok, "highest_payout")[0].market.ticker == "R70"
    assert pick_rungs(ok, "best_ev")[0].market.ticker == "R80"


def test_combo_legs_use_mid_price_and_single_bets_stay_strict():
    from kalshicombos.edges import passes, passes_combo
    leg = _leg({"kind": "rec_yds", "team": "SEA", "player": "X", "threshold": 19.5}, "yes", 0.75)
    leg.kalshi_prob, leg.ask, leg.fee, leg.fair_prob = 0.74, 0.745, 0.012, 0.75   # +1 pt vs mid, -0.7c vs ask+fee
    assert passes_combo(leg, CFG["filters"]) and not passes(leg, CFG["filters"])
    td = _leg({"kind": "td", "team": "SEA", "player": "Y", "threshold": 0.5}, "no", 0.17)
    td.kalshi_prob, td.fair_prob, td.one_sided = 0.82, 0.83, True                  # Yes-only TD: needs +3 pts
    assert not passes_combo(td, CFG["filters"])


def _sw(kind, th, side, ticker):
    return _leg({"ticker": ticker, "kind": kind, "player": "D'Andre Swift", "threshold": th}, side, 0.5)


def test_kalshi_rejects_implied_legs_like_swift():
    """Real Kalshi rejection (CHI@SEA): No 30+ rec + No 80+ rush already guarantee No 115+ rush+rec."""
    from kalshicombos.buildable import can_add, problems, prune
    rec = _sw("rec_yds", 29.5, "no", "REC30")
    rush = _sw("rush_yds", 79.5, "no", "RSH80")
    rr = _sw("rr_yds", 114.5, "no", "RR115")
    assert problems([rec, rush, rr])                 # the exact combo Kalshi refused
    assert not can_add([rec, rush], rr)
    assert can_add([rec], rr) and can_add([rush], rr)   # any two of them are fine
    assert len(prune([rec, rush, rr])) == 2
    # No 30+ rec + No 80+ rush cap him at 108, so No 110+ rr is implied too, but No 100+ rr is not
    assert not can_add([rec, rush], _sw("rr_yds", 109.5, "no", "RR110"))
    assert can_add([rec, rush], _sw("rr_yds", 99.5, "no", "RR100"))
    # Yes 50+ rush + Yes 30+ rec guarantee Yes 70+ rr (and rule out No 60+ rr)
    y1, y2 = _sw("rush_yds", 49.5, "yes", "R50"), _sw("rec_yds", 29.5, "yes", "C30")
    assert not can_add([y1, y2], _sw("rr_yds", 69.5, "yes", "RR70"))
    assert not can_add([y1, y2], _sw("rr_yds", 59.5, "no", "RR60N"))
    assert can_add([y1, y2], _sw("rr_yds", 99.5, "yes", "RR100Y"))
    # Yes 100+ rr + No 30+ rec force rush >= 71, so Yes 60+ rush is implied
    assert not can_add([_sw("rr_yds", 99.5, "yes", "A"), _sw("rec_yds", 29.5, "no", "B")],
                       _sw("rush_yds", 59.5, "yes", "C"))
    # receiving yards need a catch; no catch means no receiving yards
    assert not can_add([_sw("rec_yds", 9.5, "yes", "D")], _sw("receptions", 0.5, "yes", "E"))
    assert not can_add([_sw("receptions", 0.5, "no", "F")], _sw("rec_yds", 9.5, "no", "G"))
    # Kalshi: only one reception line per player (so no receptions ranges)
    assert not can_add([_sw("receptions", 1.5, "yes", "L")], _sw("receptions", 4.5, "no", "M"))
    assert problems([_sw("receptions", 1.5, "yes", "L"), _sw("receptions", 4.5, "no", "M")])
    # one receiving-yards line per player too
    assert not can_add([_sw("rec_yds", 14.5, "yes", "N")], _sw("rec_yds", 59.5, "no", "O"))
    # one line per player for every stat: no rushing / passing / rush+rec ranges either
    assert not can_add([_sw("rush_yds", 29.5, "yes", "H")], _sw("rush_yds", 79.5, "no", "I"))
    assert not can_add([_sw("rr_yds", 49.5, "yes", "H2")], _sw("rr_yds", 119.5, "no", "I2"))
    assert can_add([_sw("rush_yds", 29.5, "yes", "H")], _sw("rec_yds", 9.5, "yes", "I3"))  # other stats fine
    assert not can_add([_sw("rush_yds", 79.5, "yes", "J")], _sw("rush_yds", 29.5, "no", "K"))


def test_suggested_combos_are_buildable_on_kalshi(monkeypatch):
    from kalshicombos.buildable import problems
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    kg, og = pipeline.fixture_getters(FIX)
    res = pipeline.run(CFG, kg, og, now=NOW)
    for g in res.games:
        rules = pipeline.rules_for(g)
        assert rules is not None and rules.collection == "KXMVECROSSCATEGORY-TEST"
        for c in g.combos:
            assert problems(c.legs, rules) == [], problems(c.legs, rules)


def test_live_kalshi_combo_rules_are_enforced():
    from kalshicombos.buildable import ComboRules, can_add, problems
    a = _leg({"ticker": "KXNFLREC-26NOV02CHISEA-CHIDSWIFT4-3", "kind": "receptions", "player": "A", "threshold": 2.5}, "no", 0.4)
    b = _leg({"ticker": "KXNFLRSHYDS-26NOV02CHISEA-CHIDSWIFT4-50", "kind": "rush_yds", "player": "A", "threshold": 49.5}, "yes", 0.6)
    c = _leg({"ticker": "KXNFLTD-26NOV02CHISEA-CHIDSWIFT4-1", "kind": "td", "player": "A", "threshold": 0.5}, "no", 0.4)
    cols = [{"collection_ticker": "X", "size_min": 2, "size_max": 2, "is_all_yes": False,
             "associated_events": [{"ticker": "KXNFLREC-26NOV02CHISEA", "is_yes_only": True},
                                   {"ticker": "KXNFLRSHYDS-26NOV02CHISEA", "is_yes_only": False, "size_max": 1}]}]
    r = ComboRules.from_collections(cols, {"KXNFLREC-26NOV02CHISEA", "KXNFLRSHYDS-26NOV02CHISEA"})
    assert not r.leg_ok(a)                      # No side on a Yes-only event
    assert not r.leg_ok(c)                      # TD event isn't in the collection
    assert r.leg_ok(b)
    b2 = _leg({"ticker": "KXNFLRSHYDS-26NOV02CHISEA-SEAKWALKER9-30", "kind": "rush_yds", "player": "B", "threshold": 29.5}, "yes", 0.6)
    assert not can_add([b], b2, r)              # event allows 1 market
    assert problems([b, b2], r)
    # TD: one line per player, like receptions
    t1 = _leg({"ticker": "TD1", "kind": "td", "player": "A", "threshold": 0.5}, "yes", 0.6)
    t2 = _leg({"ticker": "TD2", "kind": "td", "player": "A", "threshold": 1.5}, "no", 0.6)
    assert not can_add([t1], t2)
