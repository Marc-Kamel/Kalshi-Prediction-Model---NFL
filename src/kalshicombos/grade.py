"""Tuesday grading: settle every logged leg and combo with Kalshi's own results.

Answers the only question that matters: when the model said a leg was +EV, was it?
- Bets: every leg that passed the filters (first time it passed = entry). Flat 1-contract P&L
  at the all-in price, and closing-line value (Kalshi's last pre-kickoff price vs what you paid).
- Calibration: for every priced leg, did things the model called 80% happen ~80% of the time?
  Brier score of the model's fair price vs Kalshi's own price (lower is better).
- Combos: did suggested combos hit about as often as their fair chance said?
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .kalshi import KalshiClient


def _run_time(name: str) -> datetime | None:
    try:
        return datetime.strptime(name[:14], "%Y%m%dT%H%MZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def load_runs(reports: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    legs, combos = [], []
    for d in sorted(p for p in reports.iterdir() if p.is_dir() and _run_time(p.name)):
        t = _run_time(d.name)
        kick = {}
        if (d / "kalshi_markets.csv").exists():
            km = pd.read_csv(d / "kalshi_markets.csv", keep_default_na=False)
            kick = dict(zip(km.ticker, km.kickoff))
        if (d / "legs.csv").exists() and (d / "legs.csv").stat().st_size > 1:
            lg = pd.read_csv(d / "legs.csv", keep_default_na=False)
            if len(lg):
                lg["run"], lg["run_time"] = d.name, t
                lg["kickoff"] = lg.ticker.map(kick).fillna("")
                legs.append(lg)
        if (d / "combos.csv").exists() and (d / "combos.csv").stat().st_size > 1:
            cb = pd.read_csv(d / "combos.csv", keep_default_na=False)
            if len(cb):
                cb["run"], cb["run_time"] = d.name, t
                combos.append(cb)
    L = pd.concat(legs, ignore_index=True) if legs else pd.DataFrame()
    C = pd.concat(combos, ignore_index=True) if combos else pd.DataFrame()
    return L, C


def fetch_results(tickers: list[str], client: KalshiClient) -> dict[str, str]:
    """ticker -> 'yes' | 'no' for settled markets (others omitted)."""
    events = sorted({"-".join(t.split("-")[:2]) for t in tickers})
    out: dict[str, str] = {}
    for ev in events:
        for m in client._paged("/markets", {"event_ticker": ev}, "markets"):
            r = (m.get("result") or "").lower()
            if r in ("yes", "no"):
                out[m["ticker"]] = r
    return out


def _pre_kickoff(L: pd.DataFrame) -> pd.DataFrame:
    k = pd.to_datetime(L.kickoff.replace("", None), utc=True, errors="coerce")
    rt = pd.to_datetime(L.run_time, utc=True)
    return L[k.isna() | (rt < k)]


def grade(reports_dir: str | Path, base_url: str, getter: Callable | None = None) -> dict:
    reports = Path(reports_dir)
    L, C = load_runs(reports)
    out_dir = reports / "grades"
    out_dir.mkdir(parents=True, exist_ok=True)
    if L.empty:
        (out_dir / "summary.md").write_text("# Grades\n\nNo logged runs yet.\n")
        return {}
    L = _pre_kickoff(L)
    results = fetch_results(sorted(L.ticker.unique()), KalshiClient(base_url, getter))
    L = L[L.ticker.isin(results)].copy()
    if L.empty:
        (out_dir / "summary.md").write_text("# Grades\n\nNothing has settled yet.\n")
        return {}
    L["won"] = [(results[t] == s) for t, s in zip(L.ticker, L.side)]
    L["passes"] = L.passes.astype(str).str.lower().eq("true")

    # Bets: first pre-kickoff snapshot where the leg passed; close = last pre-kickoff snapshot
    L = L.sort_values("run_time")
    bets = L[L.passes].groupby(["ticker", "side"], as_index=False).first()
    close = L.groupby(["ticker", "side"], as_index=False).last()[["ticker", "side", "kalshi_prob"]]
    bets = bets.merge(close.rename(columns={"kalshi_prob": "close_prob"}), on=["ticker", "side"], how="left")
    bets["pnl"] = bets.won.astype(float) - bets.all_in.astype(float)
    bets["clv_pts"] = (bets.close_prob.astype(float) - bets.all_in.astype(float)) * 100

    # Calibration: last pre-kickoff snapshot of every priced leg, YES side only (avoid double count)
    cal = L[L.side == "yes"].groupby("ticker", as_index=False).last()
    cal["fair_prob"] = cal.fair_prob.astype(float)
    cal["kalshi_prob"] = cal.kalshi_prob.astype(float)
    y = cal.won.astype(float)
    brier_fair = float(np.mean((cal.fair_prob - y) ** 2))
    brier_kalshi = float(np.mean((cal.kalshi_prob - y) ** 2))
    cal["bucket"] = pd.cut(cal.fair_prob, [0, .1, .2, .4, .6, .8, .9, 1.0])
    calib = cal.groupby("bucket", observed=True).agg(n=("won", "size"), model=("fair_prob", "mean"),
                                                     kalshi=("kalshi_prob", "mean"), actual=("won", "mean"))

    # Combos: settled only if every leg settled
    combo_rows = []
    for _, c in C.iterrows() if not C.empty else []:
        tk, sd = str(c.tickers).split(","), str(c.sides).split(",")
        if not all(t in results for t in tk):
            continue
        combo_rows.append({"run": c.run, "n_legs": int(c.n_legs), "fair_prob": float(c.fair_prob),
                           "accept_x": float(c.accept_if_payout_x_at_least),
                           "won": all(results[t] == s for t, s in zip(tk, sd))})
    CB = pd.DataFrame(combo_rows)

    bets.to_csv(out_dir / "bets.csv", index=False)
    cal.drop(columns=["bucket"]).to_csv(out_dir / "calibration_legs.csv", index=False)
    if not CB.empty:
        CB.to_csv(out_dir / "combos.csv", index=False)

    md = ["# Grades", "", f"Graded {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC from {L.run.nunique()} run(s).", ""]
    md += ["## Legs the model called +EV", ""]
    if bets.empty:
        md += ["None settled yet.", ""]
    else:
        n = len(bets)
        md += [f"- Bets: **{n}**, won {int(bets.won.sum())} ({bets.won.mean() * 100:.1f}%), "
               f"model expected {bets.fair_prob.astype(float).mean() * 100:.1f}%",
               f"- Flat 1-contract P&L: **{bets.pnl.sum():+.2f}** dollars on {bets.all_in.astype(float).sum():.2f} staked "
               f"(ROI {bets.pnl.sum() / bets.all_in.astype(float).sum() * 100:+.1f}%)",
               f"- Closing-line value: average **{bets.clv_pts.mean():+.2f} pts** "
               f"(positive = Kalshi moved toward the model before kickoff; the best early sign of a real edge)",
               "", "_Under a few hundred bets, P&L is mostly luck; CLV and calibration say more._", ""]
    md += ["## Calibration (every priced leg)", "",
           f"Brier score — model: **{brier_fair:.4f}**, Kalshi: **{brier_kalshi:.4f}** (lower is better; "
           f"{len(cal)} legs)", "",
           "| Model said | Legs | Model avg | Kalshi avg | Actually happened |", "|---|---|---|---|---|"]
    for b, r in calib.iterrows():
        md.append(f"| {b} | {int(r.n)} | {r.model * 100:.1f}% | {r.kalshi * 100:.1f}% | {r.actual * 100:.1f}% |")
    md += ["", "## Suggested combos", ""]
    if CB.empty:
        md += ["None settled yet."]
    else:
        md += [f"- Settled combos: {len(CB)}, hit {int(CB.won.sum())}, model expected {CB.fair_prob.sum():.2f} hits",
               "- By size: " + ", ".join(f"{k} legs: {int(g.won.sum())}/{len(g)} (exp {g.fair_prob.sum():.2f})"
                                         for k, g in CB.groupby("n_legs"))]
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")
    return {"bets": len(bets), "legs_calibrated": len(cal), "combos": len(CB),
            "brier_model": brier_fair, "brier_kalshi": brier_kalshi}
