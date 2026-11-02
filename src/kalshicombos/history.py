"""Learn outcome shapes and leg correlations from historical NFL game logs (nflverse).

Why: a book posts one line per player stat (say Over 34.5 receiving yards). To price Kalshi's
other rungs (10+, 60+, 100+) we need the *shape* of that player's possible games: how often a
backup gets zero catches, how often a WR3 breaks one long play. A textbook curve (gamma) gets
those tails wrong. Here the shape comes from real games.

Method
- For every player-game, "expected" = the player's exponentially-weighted average of previous
  games (half-life 4 games, at least 3 prior games). ratio = actual / expected.
- Ratios are grouped by stat, position and size of expected (a 15-yard tight end and a 90-yard
  WR1 have very different shapes). Each group stores its ratio quantiles.
- To price a rung: scale the group's ratio distribution so it matches the book's devigged
  P(over line), then read off P(over rung).
- Correlations: each player-stat game is turned into a normal score (rank within its group);
  average correlations are measured per relationship (same player, teammates, opponents) and
  stat/position pair. The simulator uses them as a Gaussian copula.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .names import norm

NFLVERSE_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{year}.csv"

STAT_COLS = {
    "rec_yds": "receiving_yards",
    "receptions": "receptions",
    "rush_yds": "rushing_yards",
    "pass_yds": "passing_yards",
    "rr_yds": None,          # rushing + receiving
    "td": None,              # rushing + receiving TDs
}
STAT_POS = {
    "rec_yds": ("WR", "TE", "RB"), "receptions": ("WR", "TE", "RB"),
    "rush_yds": ("RB", "QB", "WR"), "rr_yds": ("RB", "WR", "TE"),
    "pass_yds": ("QB",), "td": ("RB", "WR", "TE", "QB"),
}
TIERS = {  # buckets of "expected" (EWMA) per stat
    "rec_yds": [0, 15, 30, 45, 60, 80, 1e9],
    "receptions": [0, 1.5, 3, 4.5, 6, 1e9],
    "rush_yds": [0, 10, 25, 45, 65, 1e9],
    "rr_yds": [0, 25, 45, 65, 90, 1e9],
    "pass_yds": [0, 190, 230, 270, 1e9],
}
N_Q = 201
HALF_LIFE = 4
MIN_PRIOR = 3
MIN_GROUP = 150


def load_logs(years: list[int], folder: str | Path | None = None) -> pd.DataFrame:
    """Read nflverse weekly player stats from `folder` (if given) or download them."""
    import requests
    frames = []
    for y in years:
        f = Path(folder) / f"stats_player_week_{y}.csv" if folder else None
        if f and f.exists():
            frames.append(pd.read_csv(f, low_memory=False))
            continue
        r = requests.get(NFLVERSE_URL.format(year=y), timeout=60)
        if r.status_code == 404:
            continue
        r.raise_for_status()
        frames.append(pd.read_csv(io.StringIO(r.text), low_memory=False))
    d = pd.concat(frames, ignore_index=True)
    d = d[d.season_type.isin(["REG", "POST"])].copy()
    d["rr_yds"] = d["rushing_yards"].fillna(0) + d["receiving_yards"].fillna(0)
    d["td"] = d["rushing_tds"].fillna(0) + d["receiving_tds"].fillna(0)
    for s, c in STAT_COLS.items():
        if c:
            d[s] = d[c].fillna(0)
    d["pos"] = d["position"].fillna("")
    d["name"] = d["player_display_name"].fillna(d["player_name"]).fillna("").astype(str).map(norm)
    d = d.sort_values(["player_id", "season", "week"])
    return d


def add_expected(d: pd.DataFrame) -> pd.DataFrame:
    for s in TIERS:
        g = d.groupby("player_id")[s]
        d[f"exp_{s}"] = g.transform(lambda x: x.ewm(halflife=HALF_LIFE).mean().shift(1))
        d[f"n_{s}"] = g.cumcount()
    return d


def _tier(stat: str, value: float) -> int:
    edges = TIERS[stat]
    return int(np.clip(np.searchsorted(edges, value, side="right") - 1, 0, len(edges) - 2))


def fit_shapes(d: pd.DataFrame) -> dict:
    out: dict = {}
    qs = np.linspace(0, 1, N_Q)
    for s in TIERS:
        for pos in STAT_POS[s] + ("ALL",):
            sub = d[(d[f"n_{s}"] >= MIN_PRIOR) & (d[f"exp_{s}"] > 0)]
            if pos != "ALL":
                sub = sub[sub.pos == pos]
            else:
                sub = sub[sub.pos.isin(STAT_POS[s])]
            if sub.empty:
                continue
            ratio = (sub[s] / sub[f"exp_{s}"]).to_numpy()
            tiers = np.array([_tier(s, v) for v in sub[f"exp_{s}"].to_numpy()])
            entry = {"tiers": TIERS[s][:-1] + [None], "quantiles": [], "n": []}
            for t in range(len(TIERS[s]) - 1):
                r = ratio[tiers == t]
                if len(r) < MIN_GROUP:  # borrow from neighbouring tiers
                    r = ratio[np.abs(tiers - t) <= 1]
                entry["quantiles"].append(np.round(np.quantile(r, qs), 5).tolist() if len(r) else None)
                entry["n"].append(int(len(r)))
            out[f"{s}:{pos}"] = entry
    return out


def _normal_scores(d: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """One row per player-game-stat with a normal score of (actual / expected) within its tier."""
    rows = []
    for s in list(TIERS) + ["td"]:
        if s == "td":
            sub = d[d.pos.isin(STAT_POS["td"])].copy()
            sub = sub[(sub.groupby("player_id").cumcount() >= MIN_PRIOR)]
            val = sub["td"].to_numpy() + rng.uniform(0, 1e-3, len(sub))   # random tie-break
            grp = sub.pos.to_numpy()
        else:
            sub = d[(d[f"n_{s}"] >= MIN_PRIOR) & (d[f"exp_{s}"] > 0) & d.pos.isin(STAT_POS[s])].copy()
            val = (sub[s] / sub[f"exp_{s}"]).to_numpy() + rng.uniform(0, 1e-6, len(sub))
            grp = np.array([f"{p}{_tier(s, e)}" for p, e in zip(sub.pos, sub[f"exp_{s}"])])
        sub = sub[["game_id", "team", "player_id", "pos"]].copy()
        sub["stat"], sub["val"], sub["grp"] = s, val, grp
        sub["z"] = sub.groupby("grp")["val"].rank(pct=True).clip(1e-4, 1 - 1e-4).map(stats.norm.ppf)
        rows.append(sub[["game_id", "team", "player_id", "pos", "stat", "z"]])
    return pd.concat(rows, ignore_index=True)


def _role(stat: str, pos: str) -> str:
    return f"{stat}@{pos}"


def fit_correlations(d: pd.DataFrame, seed: int = 1, min_pairs: int = 200) -> dict:
    """Average correlation of normal scores per (relationship, role pair)."""
    rng = np.random.default_rng(seed)
    z = _normal_scores(d, rng)
    z["role"] = [_role(s, p) for s, p in zip(z.stat, z.pos)]
    acc: dict[tuple, list] = {}
    for gid, g in z.groupby("game_id"):
        if len(g) > 120:
            continue
        a = g.to_dict("records")
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                x, y = a[i], a[j]
                if x["player_id"] == y["player_id"]:
                    if x["stat"] == y["stat"]:
                        continue
                    rel = "same_player"
                elif x["team"] == y["team"]:
                    rel = "teammate"
                else:
                    rel = "opponent"
                r1, r2 = sorted([x["role"], y["role"]])
                acc.setdefault((rel, r1, r2), []).append((x["z"], y["z"]) if r1 == x["role"] else (y["z"], x["z"]))
    out = {}
    for (rel, r1, r2), pairs in acc.items():
        if len(pairs) < min_pairs:
            continue
        p = np.array(pairs)
        out[f"{rel}|{r1}|{r2}"] = {"rho": round(float(np.corrcoef(p[:, 0], p[:, 1])[0, 1]), 4), "n": len(p)}
    return out


def fit_positions(d: pd.DataFrame) -> dict:
    last = d.sort_values(["season", "week"]).groupby("name").tail(1)
    return {r["name"]: r["pos"] for _, r in last.iterrows() if r["pos"]}


def fit_all(years: list[int], out_dir: str | Path, folder: str | Path | None = None) -> dict:
    d = add_expected(load_logs(years, folder))
    model = {
        "years": years, "half_life_games": HALF_LIFE,
        "shapes": fit_shapes(d),
        "correlations": fit_correlations(d),
        "positions": fit_positions(d),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "history_model.json").write_text(json.dumps(model))
    return model


# ---------- used at pricing time ----------

class ShapeModel:
    def __init__(self, model: dict, alpha: dict | None = None, zero_scale: dict | None = None):
        # alpha < 1 narrows the non-zero part of the shape (ratio -> ratio**alpha). Needed because
        # "expected" in history is a recent average, noisier than a sportsbook line, which makes
        # raw historical ratios too spread out. zero_scale shrinks the share of zero games likewise.
        self.alpha = alpha or {}
        self.zero_scale = zero_scale or {}
        self.shapes = model.get("shapes", {})
        self.positions = model.get("positions", {})
        self.corr = model.get("correlations", {})

    @classmethod
    def load(cls, path: str | Path, alpha: dict | None = None, zero_scale: dict | None = None) -> "ShapeModel | None":
        p = Path(path)
        return cls(json.loads(p.read_text()), alpha, zero_scale) if p.exists() else None

    def position(self, player: str | None) -> str:
        return self.positions.get(norm(player or ""), "")

    def _quantiles(self, stat: str, pos: str, line: float) -> np.ndarray | None:
        entry = self.shapes.get(f"{stat}:{pos}") or self.shapes.get(f"{stat}:ALL")
        if not entry:
            return None
        edges = TIERS[stat]
        t = int(np.clip(np.searchsorted(edges, line, side="right") - 1, 0, len(edges) - 2))
        q = entry["quantiles"][t]
        if q is None:
            return None
        q = np.asarray(q, dtype=float)
        a = float(self.alpha.get(stat, 1.0))
        z = float(self.zero_scale.get(stat, 1.0))
        pos = q > 1e-9
        q = np.where(pos, q, 0.0)   # negative yards count as zero
        if a != 1.0 and pos.any():
            q[pos] = q[pos] ** a
            q[pos] = q[pos] / float(np.median(q[pos])) * float(np.median(np.asarray(entry["quantiles"][t])[pos]))
        if z != 1.0:
            zero_share = float(np.mean(~pos))
            new_share = zero_share * z
            levels = np.linspace(0, 1, len(q))
            src = np.where(levels < new_share, 0.0,
                           np.interp(zero_share + (levels - new_share) * (1 - zero_share) / max(1e-9, 1 - new_share),
                                     levels, q))
            q = src
        return q

    def sf_from_line(self, stat: str, player: str | None, line: float, p_over: float, target: float) -> float | None:
        """P(stat > target) given the book's devigged P(stat > line), using the historical shape."""
        q = self._quantiles(stat, self.position(player), line)
        if q is None or not (0 < p_over < 1):
            return None
        levels = np.linspace(0, 1, len(q))
        r_line = float(np.interp(1 - p_over, levels, q))      # ratio at which P(over) = p_over
        if r_line <= 1e-9:
            return None                                       # line sits inside the "zero" mass
        scale = line / r_line                                 # player's expected value implied by the book
        r_t = target / scale
        if r_t <= q[0]:
            return 1.0 - 1e-6
        if r_t >= q[-1]:
            return 1e-6
        # P(ratio > r_t): fraction of quantile levels above r_t (handles flat zero mass correctly)
        cdf = float(np.interp(r_t, q, levels, left=0.0, right=1.0))
        if r_t < q[0] + 1e-12:
            cdf = 0.0
        return float(np.clip(1.0 - cdf, 1e-6, 1 - 1e-6))

    def rho(self, a_stat: str, a_pos: str, b_stat: str, b_pos: str, rel: str) -> float | None:
        r1, r2 = sorted([_role(a_stat, a_pos), _role(b_stat, b_pos)])
        v = self.corr.get(f"{rel}|{r1}|{r2}")
        return None if v is None else float(v["rho"])
