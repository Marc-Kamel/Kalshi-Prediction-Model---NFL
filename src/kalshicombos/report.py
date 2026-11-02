"""Write CSVs and a readable Markdown report."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from .pipeline import RunResult


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def to_markdown(res: RunResult) -> str:
    f = res.config["filters"]
    c = res.config["combos"]
    out = [f"# Kalshi NFL combos — run {res.run_id}", "",
           f"Leg filter: Kalshi ≥ {_pct(f['min_kalshi_prob'])}, sportsbook fair ≥ {_pct(f['min_fair_prob'])}, "
           f"and fair beats Kalshi ask + fee. Suggestions only; nothing is bought.", ""]
    if res.credits_remaining is not None:
        out += [f"Odds API credits left: {res.credits_remaining}", ""]
    for n in res.notes:
        out += [f"**{n}**", ""]
    if not res.games and not res.notes:
        out += ["No games in the window (or no sportsbook prices yet)."]
    for g in res.games:
        kick = datetime.fromisoformat(g.kickoff.replace("Z", "+00:00")).strftime("%a %b %d %H:%M UTC")
        out += [f"## {g.game} — {kick}", ""]
        if g.note:
            out += [f"_{g.note}_", ""]
        if g.combos:
            out += ["### Suggested combos", "",
                    "| Legs | Picks | Fair chance | Fair payout | **Accept if Kalshi pays ≥** | EV at leg asks |",
                    "|---|---|---|---|---|---|"]
            for cb in g.combos:
                legs = "<br>".join(l.label for l in cb.legs)
                rng = f" ({cb.n_ranges} range{'s' if cb.n_ranges != 1 else ''})" if cb.n_ranges else ""
                fc = f"{cb.fair_prob * 100:.2f}%" if cb.fair_prob < 0.01 else _pct(cb.fair_prob)
                if cb.sim_hits < 1000:
                    fc += f" (±{100 / cb.sim_hits ** 0.5:.0f}%)"
                out.append(f"| {len(cb.legs)}{rng} | {legs} | {fc} | {cb.fair_payout:.2f}x | "
                           f"**{cb.min_accept_payout:.2f}x** | {cb.ev_at_leg_asks * 100:+.1f}% |")
            out += ["", f"Each bigger combo adds legs to the smaller one. Build it in Kalshi and request a quote; "
                        f"take it only if the payout multiple is at or above the bold number "
                        f"(fair + {c['accept_margin'] * 100:.0f}%). A range = Yes on a low rung + No on a higher "
                        f"rung of the same player stat.", ""]
        if g.qualifying:
            out += ["### Legs that passed", "",
                    "| Leg | Kalshi | Ask + fee | Fair | Edge | Books |", "|---|---|---|---|---|---|"]
            for l in g.qualifying:
                out.append(f"| {l.label} | {_pct(l.kalshi_prob)} | {l.all_in * 100:.1f}¢ | {_pct(l.fair_prob)} | "
                           f"{l.edge * 100:+.1f}¢ | {l.n_books} |")
            out.append("")
        if g.suspicious:
            out += [f"Skipped as likely data errors (edge > {f.get('max_edge', 1) * 100:.0f}¢; check the line before trusting):", ""]
            out += [f"- {l.label}: fair {_pct(l.fair_prob)} vs Kalshi {_pct(l.kalshi_prob)}" for l in g.suspicious[:8]]
            out.append("")
        if not g.qualifying:
            near = sorted([l for l in g.legs if l.kalshi_prob >= f["min_kalshi_prob"]], key=lambda l: l.edge, reverse=True)[:5]
            if near:
                out += ["Closest misses (60%+ legs, best edge first):", ""]
                out += [f"- {l.label}: fair {_pct(l.fair_prob)} vs {l.all_in * 100:.1f}¢ all-in ({l.edge * 100:+.1f}¢)" for l in near]
                out.append("")
    out += ["---", "Fair = sportsbook prices with the vig removed (power method), weighted toward sharp books. "
                   "Combo chances come from a correlated simulation; the correlation settings are estimates."]
    return "\n".join(out)


def write_raw(res: RunResult, out_dir: str | os.PathLike | None = None) -> Path:
    """Only the raw prices (Kalshi markets + book quotes) so a run can be replayed. Used by friend
    mode, which must not overwrite the strict report or add its looser picks to grading."""
    base = Path(out_dir or res.config["output"]["dir"])
    run_dir = base / res.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    km = [{k: v for k, v in m.__dict__.items() if k != "extra"} | {"kickoff": g.kickoff}
          for g in res.games for m in g.markets]
    if km:
        pd.DataFrame(km).to_csv(run_dir / "kalshi_markets.csv", index=False)
    raw = [q | {"game": g.game} for g in res.games for q in g.raw_quotes]
    if raw:
        pd.DataFrame(raw).to_csv(run_dir / "book_quotes.csv.gz", index=False)
    return run_dir


def write(res: RunResult, out_dir: str | os.PathLike | None = None) -> Path:
    base = Path(out_dir or res.config["output"]["dir"])
    run_dir = base / res.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    all_legs = [l.as_row() | {"passes": l in g.qualifying} for g in res.games for l in g.legs]
    pd.DataFrame(all_legs).to_csv(run_dir / "legs.csv", index=False)
    pd.DataFrame([c.as_row() for g in res.games for c in g.combos]).to_csv(run_dir / "combos.csv", index=False)
    km = [{k: v for k, v in m.__dict__.items() if k != "extra"} | {"kickoff": g.kickoff}
          for g in res.games for m in g.markets]
    if km:
        pd.DataFrame(km).to_csv(run_dir / "kalshi_markets.csv", index=False)
    raw = [q | {"game": g.game} for g in res.games for q in g.raw_quotes]
    if raw:
        pd.DataFrame(raw).to_csv(run_dir / "book_quotes.csv.gz", index=False)
    md = to_markdown(res)
    (run_dir / "report.md").write_text(md)
    (base / "latest.md").write_text(md)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as fh:
            fh.write(md + "\n")
    return run_dir
