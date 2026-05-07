# Calibrated Gaussian Value-Trading — Results

## Strategy idea

Build a per-city calibrated Gaussian PDF over daily-high temperature, centered
on the day's NWS forecast (with optional bias correction) and an empirically
fit standard deviation. For every Kalshi & Polymarket bracket, compute
`fair_value = sum(pdf[d] for d in bracket_degrees)`. If the best ask is enough
below fair value AND the ladder has depth, lift YES. Exit at a target P&L
(fraction of the predicted edge) or on a time stop.

The hypothesis: the original bot used a fixed σ=2°F across all five cities,
but residual variability differs city-to-city. A per-city σ should price
brackets more accurately and harvest edge.

## Calibration

The data dictionary describes 400 historical settlements (Dec 2025 – May 2026)
but the `forecasts` table only covers 2026-05-05 / 06 / 07. So we cannot
compute true `forecast − actual` residuals over history.

Fallback: per-city σ is fit as the residual std-dev of actual highs around a
centered 7-day rolling mean (a "naïve persistence/seasonal" baseline), then
shrunk by 0.5× to approximate real NWS skill (NWS day-1 MAE is roughly half
of persistence MAE in CONUS), with a floor of 1.5 and cap of 4.0. Bias is
the mean of those residuals (effectively zero for all cities).

Calibration outputs (`calibration.json`):

| City          | σ (calibrated) | σ (persistence) | bias  | n   |
|---------------|---------------:|----------------:|------:|----:|
| Miami         |          2.23  |           4.46  | +0.06 |  93 |
| LA            |          2.59  |           5.18  | +0.25 |  50 |
| Austin        |          3.43  |           6.87  | −0.15 |  60 |
| San Francisco |          2.58  |           5.16  | +0.03 |  60 |
| Seattle       |          2.21  |           4.42  | +0.04 | 133 |

## Iterations

Backtest run on 2026-05-06 snapshots (340 per city/venue), entry by lifting
the ask ladder, exit via `simulate_exit` with target = `target_frac × edge`.
Fee = $0.005/contract per side. Trade size = 100 contracts.

| version | n_trades | win_rate | total_pnl ($) | avg/trade ($) | sharpe | notes |
|---------|---------:|---------:|--------------:|--------------:|-------:|-------|
| v1 | 5,689 | 2.9% |  −14,005 | −2.46 | −0.74 | naïve, 5c edge, no filters |
| v2 |    99 | 4.0% |    −535  | −5.41 | −1.76 | exclude tails; ask∈[0.10,0.90]; 60min stop |
| v3 |   138 | 3.6% |    −712  | −5.16 | −1.74 | + σ=1.5; fv≥0.20 |
| v4 |   160 | 3.1% |  −1,156  | −7.22 | −1.84 | central-bracket-only |

All four versions lose money. There is no positive-PnL configuration in the
parameter space we explored. Sharpe is negative for every iteration.

## Failure modes (the honest part)

1. **Tail-bracket flood (v1).** 88% of v1 losers had ask ≤ $0.05. A Gaussian
   centered on Miami's 89°F NWS with σ=2.23 puts ~14% mass on "92° or above"
   (which spans 92..97 due to TAIL_SPREAD=5). The market priced that tail at
   1¢. The market was right almost every time. v1 fired 4,191 "buy a penny,
   collect zero" trades that all decayed to no-liquidity exits.

2. **The market is more confident than our PDF.** Even after excluding tails
   and forcing σ=1.5 (v3), the strategy still loses. The brackets where our
   PDF says "fv − ask > threshold" are exactly the brackets the market
   already priced down because it has information we don't (intraday
   observations, updated NWS issuances, etc.).

3. **Static-forecast staleness.** We use the FIRST NWS forecast of the day
   for the entire trading session. For 2026-05-06, the NWS forecast was off
   by ≥2°F in 4/5 cities (LA forecast 66 → settled 68-69; SF forecast 66 →
   settled 67-68; Seattle forecast 70 → settled ≤68; Austin forecast 84 →
   settled ≤85). Miami was the only "on-target" city (89 → 88-89). On v4
   (central-bracket-only) we made 0 Miami trades because Miami's central
   bracket was already efficiently priced and never offered a 5c edge.

4. **Adverse selection.** This is the structural problem: a static-PDF
   value-trader can only fire when the market disagrees with the PDF, and
   the market disagrees most loudly precisely when it knows something we
   don't. Negative expected value by construction.

5. **Calibration data is the wrong shape.** σ fit from
   `actual − rolling_mean(actual)` measures weather variability, not
   forecast skill. With actual NWS forecasts attached to historical
   settlements we could compute true forecast residuals, but the database
   doesn't have them.

## Recommended next steps

- **Refresh the forecast PDF intraday.** Pulling the NWS PDF every 15-30
  min and re-pricing against it would erase most of the staleness loss.
  For the snapshot data we have, this isn't possible (only one
  forecast/day at the start), but a live bot should be doing this.
- **Use the venues' own implied PDFs as the prior.** Pool Kalshi+Polymarket
  PMFs into a "consensus" and look for cross-venue dispersion (this is what
  strategy_2_cross_venue_convergence is for). Don't try to beat the market
  with a static Gaussian.
- **Backtest on multiple settled days.** One trading day across five cities
  is a sample size of 5 settlement outcomes. Even if the strategy worked,
  results would be statistically noisy. Need ≥30 settled days to draw
  conclusions.
- **Capture historical NWS forecasts going forward** so a future calibration
  can compute true `forecast − actual` residuals (and a meaningful bias).
- If you must persist a value-trader on these markets, consider trading
  only the central bracket within a short window after each NWS issuance
  (when our PDF is freshest) and require a much higher edge bar (e.g.
  ≥10c) plus narrow [0.20, 0.80] ask band.

## Conclusion

Calibrated Gaussian value-trading does not work on this data. v1 lost
~$14k, every refinement just lost less. The strategy's structural problem
(adverse selection vs. an informed market with a stale forecast) is
unfixable within this template. Negative result; don't deploy.
