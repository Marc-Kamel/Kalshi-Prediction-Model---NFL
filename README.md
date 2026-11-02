# kalshi-combos

Finds Kalshi NFL legs that are **likely (60%+) and priced better than the sportsbooks say they should be**, then builds same-game combos from them with a correlated simulation. It only suggests; it never logs in to Kalshi or places a bet.

Every run writes `reports/latest.md`. It's also shown on the run's page under **Actions**.

## What a leg has to pass
1. **Kalshi's own price** for that side (yes or no) is at least 60%.
2. **The sportsbook fair price** is also at least 60%. Fair price means the books' odds with the vig removed.
3. **Fair beats what you'd actually pay**: Kalshi's ask plus Kalshi's taker fee. The fee is `ceil(0.07 × contracts × P × (1−P))` per order.

You can change all three in `config.yaml` under `filters`.

## How the fair price is made
- **Books:** prices come from Pinnacle, FanDuel, DraftKings, BetMGM, Caesars and others, through [The Odds API](https://the-odds-api.com).
- **Vig removal:** each Over/Under or Yes/No pair has its vig removed with the power method. Anytime-TD markets that only show a Yes price get an assumed margin instead.
- **Kalshi rungs:** a book's line, say Over 64.5 yards, is turned into a full distribution so every Kalshi rung ("40+", "60+", "80+") gets a price.
  - Yards use a gamma distribution.
  - Receptions use a negative binomial.
  - 2+ TDs use a Poisson from the anytime-TD price.
  - Spreads and totals use a normal distribution.
- **Averaging:** books are averaged with sharp books weighted higher (Pinnacle ×3). A price needs at least 2 books.

## How combos are built
- **Pool:** for each game, the model takes the best qualifying legs, one per player-stat, and never both moneyline and spread together.
- **Simulation:** it simulates the game 200,000 times with shared factors (team offense, team pass volume, each player's own game).
  - Correlated legs move together: a QB's yards with his WR's yards, or a "No TD" leg with a low-scoring game.
  - Every leg still hits at exactly its fair chance.
- **Output:** for each combo you get:
  - **Fair chance**, from the simulation, with correlation included.
  - **Fair payout** = 1 ÷ fair chance.
  - **Accept if Kalshi pays ≥**: fair payout plus a 3% cushion. **This is the number that matters.** Kalshi prices combos by quote (RFQ). Build the combo in the app, look at the payout it offers, and take it only if the payout is at or above this number.
  - **EV at leg asks** is only a rough reference. Market makers price correlation themselves.

## Setup (once)
1. **Get an Odds API key** at the-odds-api.com. The free tier gives 500 credits a month.
2. **Add the key to GitHub:** repo → Settings → Secrets and variables → Actions → New repository secret. Name it `ODDS_API_KEY`.
3. **Run it:** Actions → **combos** → Run workflow. You can type a game like `CHISEA`, or leave it blank for every game in the next 36 hours.

It also runs on its own before Thursday, Sunday and Monday games (see `.github/workflows/combos.yml`).

### Credits
Each game costs about 1 credit per market listed in `config.yaml` (8 by default). A full week on the default schedule uses roughly 150–200 credits, which is about 3 weeks of the free tier. To stretch it, remove markets, delete scheduled runs, or use the $30/month tier.

## Run locally
```
pip install -e ".[dev]"
pytest -q
export ODDS_API_KEY=...        # PowerShell: $env:ODDS_API_KEY="..."
kalshicombos run               # games in the next 36h
kalshicombos run --game CHISEA
kalshicombos run --fixtures tests/fixtures/chisea --game CHISEA   # offline demo (synthetic book odds)
```

## Files
- `reports/latest.md` is the most recent report.
- `reports/<run>/legs.csv` holds every priced leg, both sides, with edge and EV.
- `reports/<run>/combos.csv` holds the suggested combos with tickers, sides and the accept-if number.

## Limits worth knowing
- **Correlation settings** (`combos.loadings`) are reasoned starting values, not fitted to data yet. Treat combo chances as estimates. The single-leg fair prices are the solid part.
- **Spreads:** normal distributions ignore NFL key numbers (3, 7), so spread rungs are approximate.
- **Prices move:** fair prices are only as fresh as the run. Re-check right before kickoff, especially after injury news.
- **No guarantees:** a +EV leg still loses often. This is not financial advice.
