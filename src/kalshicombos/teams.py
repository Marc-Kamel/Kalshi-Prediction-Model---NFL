"""NFL team codes as Kalshi writes them, mapped to the full names sportsbooks use."""
from __future__ import annotations

TEAMS: dict[str, str] = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAC": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LV": "Las Vegas Raiders", "LAC": "Los Angeles Chargers",
    "LAR": "Los Angeles Rams", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SF": "San Francisco 49ers", "SEA": "Seattle Seahawks", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}
ALIASES = {"JAX": "JAC", "LA": "LAR", "WSH": "WAS"}
NAME_TO_CODE = {v.lower(): k for k, v in TEAMS.items()}


def code(c: str) -> str:
    c = c.upper()
    return ALIASES.get(c, c)


def split_matchup(s: str) -> tuple[str, str] | None:
    """'CHISEA' -> ('CHI', 'SEA'); 'NYJKC' -> ('NYJ', 'KC'). Kalshi lists away team first."""
    s = s.upper()
    for i in range(2, len(s) - 1):
        a, h = code(s[:i]), code(s[i:])
        if a in TEAMS and h in TEAMS:
            return a, h
    return None


def from_name(name: str) -> str | None:
    return NAME_TO_CODE.get(name.lower().strip())
