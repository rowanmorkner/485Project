# Strategy 5: ML Bracket-Move Predictor

## Strategy idea

Train supervised classifiers on per-bracket per-timestamp features to
predict whether a bracket's midprice will go UP, SIDEWAYS, or DOWN over
the next K=6 snapshots (~30 minutes). When the model says UP with
confidence above a threshold, lift the YES ask; when it says DOWN, take
NO synthetically. Exit on a target (~$0.02 per contract net of fees), a
30-min time stop, or end of window.

## Key implementation choices

- **Features (14)**: `mid_t`, `spread`, `bid_size`, `ask_size`,
  `bid_depth` (total qty across yes-bid ladder), `ask_depth`,
  `dist_peak` (|bracket center − NWS forecast peak|), `hour_utc`,
  `mins_since_open`, `has_two_sided`, `mid_dt3/dt6/dt12` (mid change
  over 3/6/12 prior snapshots), and `cross_diff` (this bracket's mid −
  best-overlap bracket on the other venue at the same timestamp).
- **Label**: ε = $0.015 around delta. 3-class for the LogReg baseline,
  binary UP and binary DOWN for the GBMs that drive trading.
- **Split**: temporal — first 70% snapshots of each (city, venue,
  bracket) are train, last 30% are test. No random shuffle, so future
  cannot leak into past.
- **NaN handling**: drop any feature row missing a feature or label
  (e.g. early in the day before lag windows fill, or when the
  cross-venue counterpart has no quote). 16,050 of 27,346 rows are
  dropped this way, leaving 11,296 usable rows.
- **Models**: scikit-learn `LogisticRegression(class_weight=balanced)`
  for 3-class (interpretation only), `GradientBoostingClassifier(200
  estimators, depth 3, lr=0.05)` for the binary UP and binary DOWN
  classifiers used by the trader.
- **Trade simulator**: `loaders.open_position` to lift the right ask
  ladder, `loaders.simulate_exit` with `target=$0.02`, `time_stop=30
  min`, `fee=$0.005/contract`, size=10. Duplicate-open guard:
  block reopening the same `(city, venue, bracket, side)` for 6
  snapshots (~30 min).

## Backtest results (2026-05-06, mark-to-market)

| version | threshold | liq filter | n_trades | win_rate | total $ | avg $/trade | sharpe | max DD |
|---------|-----------|------------|----------|----------|---------|-------------|--------|--------|
| v1      | 0.20      | none       | 282      | 19.9%    | -108.50 | -0.385      | -0.587 | -108.6 |
| v2      | 0.35      | spread/size/mid | 9   | 22.2%    | -3.55   | -0.394      | -0.560 |  -3.74 |
| v3      | 0.70      | none       | 30       | 30.0%    | -10.67  | -0.356      | -0.268 |  -10.7 |

All three iterations are unambiguously money-losing.

### Classifier headline numbers (test set, K=6)

- 3-class label distribution on test: SIDEWAYS 80.5%, UP 8.3%, DOWN 11.2%.
- Majority baseline (always predict SIDEWAYS): **80.5% test accuracy**.
- LogReg 3-class (balanced): 62.4% test accuracy. The "balanced"
  reweighting trades headline accuracy for class-1/-1 recall, which is
  what we actually need — but produces lots of false directional calls.
- GBM binary UP: 90.9% test accuracy, log loss 0.241. Looks great until
  you read the confusion: 52 TP / 78 FP / 232 FN. Precision when
  predicting UP = 40%; recall = 18%.
- GBM binary DOWN: 83.7% test accuracy, log loss 0.321. 35 TP / 211 FP /
  347 FN. Precision when predicting DOWN = 14%; recall = 9%.

### Top GBM-UP feature importances

```
mid_t        0.354
cross_diff   0.087
mid_dt6      0.080
ask_size     0.074
mid_dt12     0.070
mid_dt3      0.066
dist_peak    0.058
spread       0.054
bid_size     0.041
mins_since_open 0.040
ask_depth    0.038
hour_utc     0.018
bid_depth    0.018
has_two_sided 0.000
```

`mid_t` dominating is suspicious: it suggests the model has learned that
"a bracket priced near 0.50 is more likely to move than one priced near
0.02" — true, but a feature of the price level rather than a forecast
of direction. `cross_diff` and short lag-features matter modestly.

### v1 by signal-confidence bin (post-hoc inspection)

| p_signal bin | n  | avg pnl $ | win rate |
|--------------|----|-----------|----------|
| 0.20–0.30    | 90 | -0.328    | 16.7%    |
| 0.30–0.40    | 58 | -0.376    | 29.3%    |
| 0.40–0.50    | 48 | -0.691    | 14.6%    |
| 0.50–0.60    | 49 | -0.451    | 14.3%    |
| 0.60–0.70    | 17 | -0.289    |  0.0%    |
| 0.70–0.80    | 10 | +0.379    | 60.0%    |
| 0.80–1.00    | 10 | -0.071    | 40.0%    |

The single positive bin (0.70–0.80) was the motivation for v3. But once
the duplicate-suppression window allowed slightly more entries (30
trades vs the 20 in raw bin counts), the cherry-picked "good" bin
collapsed to -$10.67. This is exactly the small-sample mirage you'd
expect.

## Failure modes

1. **Class imbalance with no real signal.** 80% of 30-min windows are
   sideways. The directional move is roughly a random thin-tailed event
   with respect to the features I built. Even an oracle "balanced"
   classifier produces too many false positives to clear bid-ask spread
   plus fees.

2. **Spread + fees dominate.** Round-trip cost per contract is ~$0.005 ×
   2 = $0.01 in fees plus the $0.01–$0.05 bid-ask spread. With size=10
   that's $0.20–$0.60 in transaction cost per trade. Median pnl is
   −$0.26 per trade — almost exactly the round-trip cost. Even when the
   prediction is directionally correct, the bracket has to move enough
   to clear that hurdle and the model isn't picking those moves.

3. **`mid_t` as a confidence proxy, not a direction.** The dominant
   feature is the price level, not anything about momentum or
   forecast-relative cheapness. The model is mostly saying "this market
   is alive, so something might happen" — which is not actionable.

4. **Cross-venue feature is shallow.** `cross_diff` is just a midpoint
   difference between brackets with overlapping degrees, and brackets
   often don't align cleanly between venues (Polymarket has 11 brackets
   per city, Kalshi has 6). When brackets do align, the difference is
   noisy due to venue spreads and idiosyncratic illiquidity.

5. **Lookahead-free, but data-poor.** ~7,900 train / ~3,400 test rows
   after dropping NaNs. Within-bracket autocorrelation makes the
   effective sample size much smaller. With binary labels at ~10%
   positive rate, we have <1,100 positive examples to learn from.

6. **No-liquidity exits punish the strategy.** v1 has 42/282 trades
   exiting `no_liquidity` (the bracket dies before any bid is hit), each
   booking the full entry cost as loss.

## Recommended next steps

- **Don't ship this.** Pure ML on 11K rows of intraday bracket prints
  doesn't have the sample size to beat a $0.30 round-trip cost.
- **Re-frame as a filter, not a primary signal.** The ML score might
  still be a useful re-ranker on top of a structural strategy (e.g.
  Strategy 1's calibrated-Gaussian or Strategy 2's cross-venue arb):
  open a position only if the structural signal AND the ML model agree.
- **Use richer features.** Realized hourly weather observations
  (current temperature vs forecast peak), time of day relative to
  expected daily-high time, and prior-day calibration all carry signal
  the bracket book doesn't.
- **Predict cross-venue convergence, not within-venue drift.** The
  question "will Kalshi-vs-Polymarket spread close" probably has more
  signal than "will this bracket midprice drift up" because settlement
  forces convergence to within a few cents.
- **Try a simpler rule: trade against extreme `cross_diff`.** That's a
  hand-coded version of what the GBM's third-most-important feature is
  trying to express, with no ML overhead.
- **Calibrate before thresholding.** Sklearn GBM probabilities are
  uncalibrated. Wrap with `CalibratedClassifierCV` (Platt or isotonic)
  and revisit the bin analysis above; today's "p>=0.7" is not a
  defensible 70% probability, just a high score.

## Honest summary

The classifier numerically "beats" random (62% > 33% on 3-class) but
loses to the dumb "always predict sideways" majority baseline (80%).
The directional-only GBMs look strong on accuracy because they predict
NOT-UP or NOT-DOWN almost always; their precision on the actual signals
is 14–40%, not enough to clear the ~$0.30/trade cost barrier. All three
backtest iterations lost money. **Negative result: simple ML on this
data does not produce a tradeable signal.**
