"""Command line: `kalshicombos run [--game CHISEA] [--config config.yaml] [--fixtures DIR]`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import pipeline, report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kalshicombos", description="Suggest +EV Kalshi NFL combos (never trades).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="price legs and build combos")
    r.add_argument("--config", default="config.yaml")
    r.add_argument("--game", action="append", help="matchup code like CHISEA (repeatable); ignores the time window")
    r.add_argument("--fixtures", help="replay saved Kalshi/odds JSON from this folder instead of calling the APIs")
    r.add_argument("--out", help="output folder (default from config)")
    rp = sub.add_parser("replay", help="re-price a saved run with the current config (no API calls)")
    rp.add_argument("run_dir")
    rp.add_argument("--config", default="config.yaml")
    rp.add_argument("--out", help="output folder (default from config)")
    g = sub.add_parser("grade", help="settle logged legs/combos with Kalshi results; write reports/grades")
    g.add_argument("--config", default="config.yaml")
    g.add_argument("--reports", help="reports folder (default from config)")
    fh = sub.add_parser("fit-history", help="refit shapes/correlations from nflverse game logs")
    fh.add_argument("--config", default="config.yaml")
    fh.add_argument("--years", default="", help="comma list, default: last 3 seasons + current")
    fh.add_argument("--local", help="folder with already-downloaded nflverse CSVs")
    fr = sub.add_parser("friend", help="one big combo built the friend's way, with a reality check")
    fr.add_argument("--config", default="config.yaml")
    fr.add_argument("--game", action="append", help="matchup code like PHIWAS (repeatable)")
    fr.add_argument("--replay", help="use a saved run folder instead of calling the APIs")
    fr.add_argument("--min-fair", type=float, help="override his 55%% fair-chance gate")
    fr.add_argument("--basis", choices=["strict", "friend"], help="what the accept-if number is based on")
    fr.add_argument("--out", help="output folder (default from config)")
    a = ap.parse_args(argv)

    cfg = pipeline.load_config(a.config)
    if a.cmd == "friend":
        from . import friend
        if a.min_fair is not None:
            cfg.setdefault("friend_mode", {})["min_fair_prob"] = a.min_fair
        res, combos, skipped = friend.run(cfg, a.game, a.replay, basis=a.basis)
        if not a.replay:
            report.write(res, a.out)   # also keep the normal logs (quotes/markets) for replay & grading
        folder = friend.write(res.run_id, combos, skipped, a.out or cfg["output"]["dir"])
        print(friend.to_markdown(res.run_id, combos, skipped))
        print(f"\nSaved to {folder}", file=sys.stderr)
        return 0
    if a.cmd == "grade":
        from . import grade as grading
        out = grading.grade(a.reports or cfg["output"]["dir"], cfg["kalshi"]["base_url"])
        print(out)
        return 0
    if a.cmd == "fit-history":
        from datetime import date
        from . import history
        years = [int(y) for y in a.years.split(",") if y] or list(range(date.today().year - 3, date.today().year + 1))
        path = Path(cfg["model"].get("history_file", "data/history_model.json"))
        m = history.fit_all(years, path.parent, a.local)
        print(f"fitted {len(m['shapes'])} shapes, {len(m['correlations'])} correlations from {years} -> {path}")
        return 0
    if a.cmd == "replay":
        res = pipeline.replay(a.run_dir, cfg)
    else:
        kg = og = None
        if a.fixtures:
            kg, og = pipeline.fixture_getters(a.fixtures)
        res = pipeline.run(cfg, kg, og, only=a.game)
    folder = report.write(res, a.out)
    print(report.to_markdown(res))
    print(f"\nSaved to {folder}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
