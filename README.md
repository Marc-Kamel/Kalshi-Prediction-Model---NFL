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
- **Two-sided lines only for yards and receptions.** A book's vig can only be measured when it posts both Over and Under. Alternate lines are Over-only. On CHI@SEA (Nov 2) no single vig guess made them agree with Kalshi, and every guess produced a different set of fake edges, so they're off by default (`allow_one_sided_props`).
- **Anytime TD** is usually Yes-only, so its vig is removed proportionally with an assumed 7% margin, and those legs must clear an extra 3¢ (`one_sided_buffer`).
- **Vig removal:** each Over/Under or Yes/No pair has its vig removed with the power method. Anytime-TD markets that only show a Yes price get an assumed margin instead.
- **Kalshi rungs:** a book's line, say Over 64.5 yards, is turned into a full distribution so every Kalshi rung ("40+", "60+", "80+") gets a price.
  - Yards use a gamma distribution.
  - Receptions use a negative binomial.
  - 2+ TDs use a Poisson from the anytime-TD price.
  - Spreads and totals use a normal distribution.
- **Averaging:** books are averaged with sharp books weighted higher (Pinnacle ×3). A price needs at least 2 books.

## History: shapes and correlations (`data/history_model.json`)
Built from nflverse game logs (2023 to now) by `kalshicombos fit-history`, which runs automatically every Tuesday.
- **Shapes:** for each stat, position and usage level, the spread of real outcomes around a player's recent average. This includes zero games (low-volume WRs have 0 yards in about half their games) and big games. It's used to price Kalshi rungs away from the book's line.
  - Two shrink settings per stat (`model.shape_alpha`, `shape_zero_scale`) correct for recent averages being noisier than book lines.
  - On CHI@SEA they matched Kalshi's whole ladder to within ±0.6 pts in every distance zone.
- **Edge buffer:** further from the book's line, a leg needs more edge (`filters.shape_buffer`, up to +3¢), matching the model's measured error at that distance.
- **Correlations:** measured between player-stat pairs (same player, teammates, opponents, by position). For example, QB passing vs his WR's receiving yards is 0.22, two WRs on the same team about 0, and a WR's receptions vs his yards 0.77. The combo simulator uses these in place of guessed loadings.

## Friend mode (`kalshicombos friend --game PHIWAS`)
Rebuilds the recipe behind a friend's 34-leg PHI@WAS combo that hit at about 519x, written from scratch here.
- **His rules:** every player leg with sportsbook fair above 55%, +EV after fees at the ask, 5+ books, alternate (Over-only) lines allowed, one line per player-stat.
- **One combo:** every qualifying leg goes into a single combo.
- **Reality check:** the report prices that combo four ways:
  - his way (legs multiplied)
  - his leg prices with measured correlations
  - our strict model
  - Kalshi's own leg prices
- **Accept-if:** the number defaults to the strict price (`friend_mode.accept_basis`).

Why the reality check: on his saved PHI@WAS data, his alternate-line prices ran about 5 pts above Kalshi on likely outcomes. His model rated the combo 1-in-287; Kalshi's own legs, with correlations, implied about 1-in-1,200, and he took 556x. Re-run with this recipe, the same data gives an 8-leg combo (about 15x), 7 of whose 8 legs our strict model also rates +EV.

Cost: about 11 credits per game (alternate markets included). Run it from Actions → combos → command `friend` with a game code.

## Grading (`reports/grades/summary.md`, every Tuesday)
Settles every logged leg and combo with Kalshi's own results:
- **Legs the model called +EV:** win rate vs expected, flat-stake P&L, and **closing-line value** (did Kalshi move toward the model before kickoff, the best early sign of a real edge).
- **Calibration:** of all priced legs, did "80%" things happen about 80% of the time? Also the Brier score of the model vs Kalshi's own prices.
- **Combos:** hits vs the number expected from their fair chances.

To run it manually: Actions → combos → Run workflow → command `grade` (or `fit-history`).

## Combo legs vs single bets
- **Single bets** must beat Kalshi's **ask + fee** (you pay both when you buy one leg).
- **Combo legs** only need the books' fair chance to be **at least Kalshi's middle price**, because a combo is bought with one quote, not leg by leg. The Yes-only TD cushion still applies (`filters.combo_cushions`).
- On CHI@SEA this lifts combo legs from 2 to 74 and allows combos of up to 25 legs.
- Each combo shows its **fair payout**, the **accept-if** number (fair + 3%), and what **Kalshi's own leg prices imply**. Market makers usually quote below Kalshi's number, so a quote between accept-if and that number is the window to look for.

## How combos are built
- **Pool:** for each game, every qualifying player leg, up to 60.
- **Ranges:** the same player stat can appear twice, as Yes on a low rung plus No on a higher one. For example, "Swift 30+ rushing" with "No Swift 80+ rushing" means 30 to 79 yards. Two Yes rungs (redundant) or contradictory pairs are never combined.
- **Buildable on Kalshi:** Kalshi refuses a combo when one leg is already guaranteed by the others, even across different stats. For example, "No Swift 30+ receiving" plus "No Swift 80+ rushing" caps him at 108 combined yards, so "No Swift 115+ rush+rec" adds nothing and Kalshi rejects it. `buildable.py` turns each player's rushing, receiving, rush+rec and receptions legs into bounds (rush + receiving = rush+rec; receiving yards need a catch) and skips any leg the others already settle, or that can't hit together with them. This applies to both the regular combos and friend mode.
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
Each game costs 1 credit per market in `config.yaml`, which is 6 by default. A full week on the default schedule is about 125 credits, so the free 500 lasts about a month.

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
