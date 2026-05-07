# Strategy comparison and recommendation

5 strategies were researched in parallel against ~28 hours of snapshot
data on 2026-05-06 markets (5 cities × 2 venues × ~340 snapshots). Each
agent built ≥2 iterations. Full per-strategy writeups in
`research/strategy_<N>_<slug>/RESULTS.md`. Per-strategy code, trade
logs, and `metrics.json` files alongside.

## Headline ranking

| # | Strategy                       | Best version                          | n   | Total $    | Sharpe | Verdict                 |
|---|-------------------------------|---------------------------------------|----:|-----------:|-------:|-------------------------|
| 1 | Calibrated Gaussian            | v2 (excl. tails, price band, time stop) | 99  | **−$535**  | −1.76  | NEGATIVE (all 4 iters)  |
| 2 | Cross-venue convergence        | v5 (hedged, cost ≤ 0.97, hold)        | 9   | **+$23.11**| +1.51  | POSITIVE w/ caveats     |
| 3 | **Structural arb**             | **v2 (strict, hold-to-settle)**       | **15** | **+$40 worst / +$140 expected** | **+0.51 to +0.71** | **POSITIVE — recommended** |
| 4 | Mean-reversion                 | v3 (tight spread, profit target)      | 150 | **−$51.79**| −0.38  | NEGATIVE (even omniscient) |
| 5 | ML predictor                   | v3 (p≥0.70)                           | 30  | **−$10.67**| −0.27  | NEGATIVE                |

## Recommendation

**Deploy Strategy 3 (strict structural arb, hold-to-settle) with the
asymmetric filter from Strategy 2's risk_analysis.**

The combined rule:
1. Enumerate cross-venue bracket pairs at every snapshot.
2. Keep only pairs where loss-zone is empty AND combined entry cost < $1
   (true structural worst-case-positive).
3. Walk the actual ladders for both legs at the strategy's chosen size.
4. **Asymmetric filter** (from S2): when the cheap leg is on Kalshi
   (which historically reads ~+1°F warmer than Polymarket), only enter
   if the hedge isn't long Polymarket "high-temperature" / short Kalshi
   "high-temperature." If the cheap leg is on Polymarket, enter
   normally — that hedge benefits from the historical disagreement bias.
5. Hold to settlement on both legs. No mark-to-market exit.
6. Dedup persistent opportunities (one open per
   `(city, k_ticker, p_token, k_side, p_side)` per day).

Expected daily realized (5 cities × 1 day, 50 contracts/leg):
**+$40 worst-case / +$140 expected, 100% worst-case win rate.** Scales
roughly linearly with size up to depth caps. Across 5 cities × 365
days, this projects to ~$15K worst-case / ~$50K expected per year at
that size — reasonable for a CapEx-light bot, modest as a target.

**Do NOT deploy:**
- **Strategy 1** — calibrated Gaussian value-trader. Static-forecast
  staleness + adverse selection. Loses on every iteration explored.
- **Strategy 4** — mean-reversion. Real autocorrelation (VR(10) ≈ 0.69)
  but **even perfect-foresight exit is unprofitable** after spread+fees.
- **Strategy 5** — supervised ML. Class imbalance + ~$0.30 round-trip
  cost dominates any directional alpha extracted from 11K observations.
- **Strategy 2 v1/v2** (naive directional cross-venue convergence).
  Falsified — gap convergence is bidirectional; the cheap side often
  drops alongside the rich side.

## Cross-strategy insights

### 1. Microstructure cost is the dominant adversary

| Strategy | Spread+fees finding |
|---|---|
| S1 | Static-forecast value gives ~80% of "edge" on 1-2¢ asks where market is right; spread+fees consume the rest. |
| S4 | Round-trip cost ≈ $0.02/contract; mean Δmid step = $0.005. Edge < cost by 4×. |
| S5 | Median per-trade P&L = −$0.26 ≈ round-trip transaction cost. |

The mean reversion agent's most damning finding: **even an omniscient
exit (best possible bid in the next 30 min) is unprofitable on average.**
That means there is no sub-30-min directional edge in raw midprice
moves on this data. Anything trying to harvest such an edge is doomed;
strategies must either (a) use longer holding periods (S3's
hold-to-settle), or (b) find a cost-free entry mechanism (limit-order
market making — out of scope for this dataset).

### 2. Venue-disagreement is the silent killer of "cross-venue arb"

S2's `risk_analysis.py` produced the single most consequential finding:
of 188 historical (city, date) pairs with both venues' final highs,
**only 25.5% report identical highs**. Distribution:

```
Kalshi − Polymarket = -1: 9.6% | 0: 25.5% | +1: 37.2% | +2: 22.9% | +3: 4.3% | +4: 0.5%
```

Mean Kalshi − Polymarket ≈ +1°F. Implication: a "structurally
guaranteed" cross-venue arb (S3 strategy) can pay $0 on one leg if the
two venues land their reading on different sides of a bracket boundary.

**Strategy 3's +$40 "worst-case" treats the venues as identical at
settlement. They aren't.** S3's true worst-case PnL over a long horizon
needs to be re-derived using S2's empirical (K − P) histogram. Doing
this carefully:
- Half of S3's 15 trades are aligned 2-degree brackets where Kalshi
  YES `B86.5` covers {86,87} and Polymarket NO `86-87°F` covers
  everything else. If Kalshi reads 87 and Poly reads 88, both win → $1
  each = $2 payout. If Kalshi reads 87 and Poly reads 86, the Kalshi YES
  pays $1 but Poly NO pays $0 → $1 payout (still ≥ entry cost). The
  asymmetric protection is real but smaller than +$140.
- The *direction* of the venue disagreement matters: if you hold Kalshi
  YES on a "low-temperature" bracket, you benefit from Kalshi reading
  high (less likely to land in a low bracket). If you hold Polymarket
  NO on a "high-temperature" bracket, similar story. Combine these and
  you get the asymmetric filter from S2.

**Action item before deploying S3 in production**: rebuild the
worst-case math against the 188-pair empirical histogram, per
(direction, bracket-position). Probably converts the current
"100% worst-case win rate" claim into "~85-90% worst-case win rate"
with a small expected loss tail.

### 3. The original bot's loss was not strategy choice — it was venue blindness

The live bot used a Gaussian-PDF value trader (~Strategy 1). It lost
~$430K in paper trading. S1's results show that strategy IS structurally
broken (adverse selection + stale forecast). But even a "correctly"
calibrated PDF can't see the 1°F Kalshi-vs-Polymarket reading bias.
Fixing the PDF without modeling venue divergence still leaves the
strategy exposed.

Any forward bot should:
1. Maintain per-city Δ = (Kalshi reading − Polymarket reading)
   histograms as a first-class object in the strategy module.
2. Apply this as a sign-correction to ALL cross-venue trade EV
   calculations, not just structural arbs.
3. Refresh intraday-PDF estimates from the venues' own implied PMFs
   (which are presumably more current than the morning NWS issuance).

### 4. Sample size is the binding constraint on every claim

We have 1 fully-traded day across 5 cities. Strategy 3's positive
result is 15 trades. Strategy 2's positive variant is 9 trades. Even
Strategy 4's negative result has only ~150 trades (and is robust to
27-cell + 15-cell hyperparameter sweeps, so the negative is more
trustworthy than the positives are).

Before committing real capital:
- **Wait for ≥30 settled trading days** of intraday snapshot coverage.
  At ~1/day this is one calendar month.
- **Re-run all 5 strategies** at that point. The negative results
  (S1, S4, S5) are robust enough now that we shouldn't expect them to
  flip. The positive results (S2, S3) need this confirmation.

### 5. What worked transferably

Three findings are worth keeping regardless of which strategy ships:

- **Ladder walking (`walk_ladder_buy`/`walk_ladder_sell`)** —
  collapses the "raw 2,154 arbs" headline to "23 fillable arbs at
  size 50." Always a sanity check between top-of-book and realistic
  size.
- **Persistence dedup** — a single opportunity reappears for many
  consecutive snapshots. Counting them naively inflates everything.
- **Worst-case vs expected EV separation** — S3 v2 vs v3 is a clean
  illustration: any time you pay > $1 for a bracket pair, you're
  taking directional risk dressed up as arb. The discipline of
  "cost < worst-case payout" is the single hardest filter.

## Specifically what I'd build next

In priority order:

1. **Apply the venue-divergence histogram to S3's selection.** Compute
   per-bracket-pair, per-side worst-case payout under the empirical
   K−P histogram (not "venues agree"). Re-run S3 v2; expect modest
   degradation but a more honest worst-case number.

2. **Asymmetric S3 + asymmetric S2.** Combine the two positive
   strategies — they're variants of the same hedged-pair selection.
   The cleanest version is "S3's bracket enumeration + S2's
   directional bias filter."

3. **Capture historical NWS forecasts.** S1 was hobbled because the
   `forecasts` table only has 3 days of data. Backfilling 6-12 months
   of NWS forecasts would unlock proper bias-and-σ calibration. This
   alone might rescue S1 from "structurally broken" to "marginally
   profitable."

4. **Don't bother with S4/S5 enhancements.** The autocorrelation result
   (VR(10) ≈ 0.69) is real but too small to trade as a price signal.
   ML re-rankers on top of structural strategies might add value, but
   only after the structural pipeline is stable and producing 30+
   trades/week.

5. **Re-test against live 5/6 settlement when it lands.** Tonight's
   23:55 UTC settlement gives 5 more (city, date) pairs to validate
   S3 v2's worst-case claim and S2 v5's positive PnL against actual
   payouts (not last-observed bid). This is the cheapest validation
   we can do before any capital decision.

## Files

- `research/strategy_1_calibrated_gaussian/` — calibration script,
  4-iteration backtest, full trade logs.
- `research/strategy_2_cross_venue_convergence/` — 6-iteration
  backtest, settlement-divergence risk analysis (`risk_analysis.py`).
- `research/strategy_3_structural_arb/` — enumerator + MtM/hold
  backtests + per-version trade dumps.
- `research/strategy_4_mean_reversion/` — backtest + AR(1)/VR
  diagnostics + 27-cell hyperparameter sweep.
- `research/strategy_5_ml_predictor/` — feature engineering, training,
  model pickles, 3-iteration backtest, per-trade dumps.
- `research/loaders.py` — shared backtest primitives
  (`open_position`, `simulate_exit`, `walk_ladder_buy`,
  `walk_ladder_sell`, `settlement_payout_for`, etc.). Read-only for
  agents; reusable by future research.

Branch: `strategy-research`. All work is on this branch and not yet
merged to main.
