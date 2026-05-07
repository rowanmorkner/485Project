# Round-1 Post-Mortem: Strategies 1, 4, 5

Three strategies from round 1 lost money in every iteration we tried. They are
archived under `research/archive/<strategy_dir>/` with only `RESULTS.md`
preserved; all source, metrics, and trade logs were removed from version
control. This document consolidates *why* they failed, what they share, and
what we should remember from each going into round 2.

## Why each failed

### Strategy 1 — Calibrated Gaussian value-trading
A static per-city Gaussian PDF over daily-high temperature, centered on the
day's first NWS forecast and used to value-trade brackets. Every
configuration lost: v1 fired 5,689 trades for **−$14,005** (win rate 2.9%,
Sharpe −0.74); the most filtered iteration (v4, central-bracket-only) still
lost **−$1,156** at Sharpe −1.84. The mechanism is structural adverse
selection. The strategy can only fire when the market disagrees with the
PDF, and the market disagrees most loudly precisely when it has information
the static forecast doesn't (intraday observations, NWS re-issuances, etc.).
Compounding this: on 2026-05-06 the first NWS forecast was off by ≥2°F in
4/5 cities, so the PDF's center was wrong from the open. v1 also drowned
in tail-bracket noise — 88% of v1 losers had ask ≤ $0.05, where a
Gaussian assigns ~14% mass to brackets the market correctly priced at 1¢.

### Strategy 4 — Intraday bracket-midprice mean reversion
Per-bracket z-score reversion: lift the contrarian side when midprice spikes
beyond k·σ of its rolling history, exit on z-revert / target / time stop.
Every config lost. The least-bad iteration (v3: depth 50, spread ≤ 2¢,
target $0.02, 120-min stop) still came in at **−$51.79 over 150 trades**
(Sharpe −0.381). The autocorrelation **is** real — median AR(1) = −0.066,
VR(10) = 0.69 (85% of brackets below 1) — but the tradeable magnitude is
swamped by the round-trip cost: median full bid-ask spread is $0.010 plus
$0.010 round-trip fees, against a mean |Δmid| step of $0.005. Even with
**perfect-foresight exit timing** over 978 v2-filtered events, mean
gross P&L was **−$0.004/contract** and only 26% of signals were profitable
gross. Spike-makers also tend to be informed — the z exit fires on a midprice
revert, but the bid hasn't followed, so the realized "target" exit booked
median **−$0.006/contract** (a cosmetic-only win).

### Strategy 5 — ML bracket-move predictor
GBM classifiers predicting UP/DOWN over the next K=6 snapshots, trading on
high-confidence signals. v1 (threshold 0.20) lost **−$108.50 over 282 trades**;
v3 (threshold 0.70, the cherry-picked winning bin) lost **−$10.67 over 30
trades**. The headline metrics looked fine — GBM-UP had 90.9% test accuracy —
but the class distribution was 80.5% SIDEWAYS, so a "majority baseline" beats
the model on raw accuracy. Effective signal precision was 14% (DOWN) to 40%
(UP); at ~$0.30 round-trip cost per 10-contract trade and median per-trade
P&L of −$0.26, the model couldn't clear the cost barrier. Top GBM feature
importance was `mid_t` itself (0.354), meaning the model mostly learned
"brackets near $0.50 move more than brackets near $0.02" — a level proxy,
not a directional forecast. Effective sample size after temporal split and
NaN drop was only 7,900 train / 3,400 test rows with ~10% positive rate, so
<1,100 positive examples to learn directional structure from.

## Common failure modes across all 3

1. **Microstructure cost dominates the signal.** All three strategies cross
   the spread on entry and again on exit. Round-trip fee ($0.010) plus
   median spread ($0.010) is ~$0.020/contract before any directional
   movement is required. The signals we built (Gaussian fair-value gap;
   z-score reversion; ML directional probability) all produced expected
   moves on the order of cents per contract — too small to clear cost
   reliably.

2. **Static / stale prior vs. an informed market (adverse selection).**
   S1 used a once-a-day NWS forecast; S4 used a per-bracket rolling history
   of its own midprice; S5 trained on the same intraday book without any
   exogenous fundamental anchor. All three are the *uninformed* counterparty
   when their signal disagrees with the live market — the disagreement is
   typically real new information, not noise to fade.

3. **Spike / disagreement = information, not error.** S1's "cheap" tails
   were correctly priced cheap by an informed market. S4's worst losses
   came from spikes that turned out to be the *start* of a real persistent
   move (e.g. the SF Kalshi 67.5° example: 0.41 → 0.79 over 60 min as the
   intraday observation firmed up the bracket). S5's one positive
   confidence bin (0.70–0.80) collapsed when slightly more trades fired.

4. **No-liquidity exit punishment.** Brackets stale or die intraday;
   when our exit logic can't find a bid, the position books the full entry
   cost as loss. S5 v1 had 42/282 trades exit `no_liquidity`. S4's
   time-stop exits in v2 summed to −$104.24 (vs +$3.19 on z-target exits).

5. **One-day backtest = ~5 settlement outcomes.** All three were tested
   on 2026-05-06 only (the only day with both forecast and orderbook
   coverage in the snapshot data). Statistically a sample of 5 settled
   cities — even if the strategy worked, results would be noisy.

## What we are NOT doing in round 2 because of these results

- **No static NWS-forecast value-trading.** The first-of-day NWS PDF is
  too stale to beat an informed orderbook. If we use a forecast prior at
  all, it must be refreshed against intraday observations, and we should
  not lift asks based on it. (S1 closed this lane.)
- **No spread-crossing intraday momentum/reversion signals.** The
  AR(1) / VR(10) reversion is real but smaller than the round-trip cost.
  Any future use of this finding has to earn the spread (passive limit
  posting), not pay it. (S4 closed this lane.)
- **No pure ML on bracket midprice / book features without structural
  anchoring.** 11K rows of intraday prints with 80% sideways labels do not
  contain enough directional signal to clear cost. ML can re-rank a
  structural signal (Strategy 2 / 3) but cannot stand alone. (S5 closed
  this lane.)
- **No more single-day backtests as the only evidence.** Round 2 results
  on 2026-05-06 alone are not sufficient to ship; we should at minimum
  cross-check against any other partial-day data and clearly mark the
  small-sample caveat.
- **No "cherry-picked confidence bin" iteration.** S5 v3 reproduced the
  classic small-sample mirage. If a sub-bin is the only thing that worked
  in v1, that's a one-shot lucky outcome, not a strategy.

## What's worth keeping from each

### From S1 (`research/archive/strategy_1_calibrated_gaussian/RESULTS.md`)
- The **per-city σ calibration table** (Miami 2.23, LA 2.59, Austin 3.43,
  SF 2.58, Seattle 2.21) is a reasonable order-of-magnitude prior for the
  width of the daily-high distribution; useful as a sanity check on
  any future Gaussian PDF that anchors a structural model.
- The **forecast-error audit on 2026-05-06** ("NWS off ≥2°F in 4/5 cities")
  is a concrete data point about how much the static forecast moves intraday.
  Round-2 strategies that use the NWS forecast should size their edge bar
  accordingly.
- The **adverse-selection framing** ("a static-PDF value-trader can only fire
  when the market disagrees, and the market disagrees most loudly when it
  knows something we don't") generalizes — apply it as a sanity check to
  any strategy that buys when a price is below "fair".

### From S4 (`research/archive/strategy_4_mean_reversion/RESULTS.md`)
- The **AR(1) / VR(q) diagnostic** (median AR(1) = −0.066, VR(10) = 0.69,
  85% of brackets below 1) is a real, generic finding about the bracket
  midprice process. It rules in passive-MM strategies (which can earn the
  reversion by collecting the spread) and rules out spread-crossing ones.
  Worth re-running on round-2 data and worth citing in any limit-order
  market-making proposal (cf. `research/strategy_10_limit_order_mm/`).
- The **perfect-foresight ceiling analysis** (mean gross −$0.004/contract,
  26% gross-profitable rate over 978 events) is a useful template: before
  building a directional intraday strategy, compute the upper bound under
  oracle exit and check it clears costs. If the ceiling is negative, stop.
- The **exit-reason decomposition** (z-revert events booked median
  −$0.006/contract — i.e. a "cosmetic win") warns that win-rate metrics
  on simulated mid-mark exits are misleading; always re-mark exits against
  the actual contra-side bid.

### From S5 (`research/archive/strategy_5_ml_predictor/RESULTS.md`)
- The **feature list** (`mid_t`, `spread`, `bid_size`, `ask_size`,
  `bid_depth`, `ask_depth`, `dist_peak`, `hour_utc`, `mins_since_open`,
  `has_two_sided`, `mid_dt3/6/12`, `cross_diff`) is reusable as a starting
  schema for any future ML-as-filter work; the `cross_diff` feature in
  particular (this bracket's mid − overlapping-bracket mid on the other
  venue) is exactly the structural quantity Strategy 2 trades, and ranked
  #2 in GBM importance — worth folding into S2/S3 enhancements.
- The **temporal-split + NaN-drop discipline** (first 70% / last 30% by
  snapshot per (city, venue, bracket); drop rather than impute) is the
  right default for any future supervised work on this data.
- The **majority-baseline check** (80.5% sideways → "always sideways" beats
  every classifier we trained on raw accuracy) is the kind of sanity check
  worth re-running on any future class-imbalanced label.
- The reframing recommendation — **"ML score as a filter, not a primary
  signal"** layered on top of S2 cross-venue or S3 structural — is a
  concrete round-2 candidate if the structural strategies need a
  noise-suppression layer.
