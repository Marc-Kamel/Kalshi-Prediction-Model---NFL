"""Command line: `kalshicombos run [--game CHISEA] [--config config.yaml] [--fixtures DIR]`."""
from __future__ import annotations

import argparse
import sys

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
    a = ap.parse_args(argv)

    cfg = pipeline.load_config(a.config)
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
