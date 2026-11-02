# kalshi-combos

Finds Kalshi NFL **player** legs that are **likely (60%+) and priced better than the sportsbooks say they should be**, then builds same-game combos from them, from 3 legs up to 30+, with a correlated simulation. It only suggests; it never logs in to Kalshi or places a bet.

Markets covered: receptions, receiving yards, rushing yards, rushing + receiving yards, passing yards, and touchdowns. Both Yes and No sides are checked. Game lines (moneyline, spread, total) are left out.

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
- **Pool:** for each game, every qualifying player leg, up to 60.
- **Ranges:** the same player stat can appear twice, as Yes on a low rung plus No on a higher one. For example, "Swift 30+ rushing" with "No Swift 80+ rushing" means 30 to 79 yards. Two Yes rungs (redundant) or contradictory pairs are never combined.
- **Sizes:** one combo per size in `combos.sizes` (default 3, 5, 8, 12, 20, and "as many as fit"). The builder adds legs one at a time, each time picking the leg that keeps the combo most +EV, so each bigger combo is the smaller one plus more legs.
- **Simulation:** it simulates the game 200,000 times with shared factors (team offense, team pass volume, each player's own game).
  - Correlated legs move together: a QB's yards with his WR's yards, or a "No TD" leg with a low-scoring game.
  - Every leg still hits at exactly its fair chance, and different lines of the same player stat share one simulated number, so ranges are priced exactly.
  - A combo stops growing once it wins in fewer than 300 simulated games, so very long combos still have a trustworthy chance. Below 1,000 wins the report shows the margin of error.
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
Each game costs 1 credit per market in `config.yaml`, which is 11 by default: 6 main player markets plus 5 alternate-line markets. The alternate lines are what give exact prices for low and high rungs like "10+ receiving yards". A full week on the default schedule is about 230 credits, so the free 500 lasts about 2 weeks. To stretch it, delete some scheduled runs, or remove alternate markets (cheaper, but fewer rungs get priced). The $30/month tier (20,000 credits) covers every run with plenty to spare.

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
