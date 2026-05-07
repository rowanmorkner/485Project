# Strategy 4: Mean reversion on intraday bracket midprices

## Strategy idea

Bracket midprices on a single (city, date, venue, bracket) ought to
oscillate around a slowly-moving "true" probability during the trading
day.  When the midprice spikes — say, on a single thin trade or because
one venue temporarily diverges from the consensus — it should tend to
revert.  Trade those spikes contrarian: if the midprice jumps up, buy
NO; if it jumps down, buy YES; exit when the price has reverted, hits a
profit target, or a time stop fires.

**Headline result: the strategy loses money on every configuration we
tried.**  The autocorrelation of midprice changes is genuinely negative
(median AR(1) = -0.066, 79% of brackets negative), but the *tradeable*
signal — what bid you can sell into after lifting the ask — is
swamped by the round-trip cost of crossing the spread plus fees on
these illiquid markets.  See "Failure modes" below for the mechanism.

## Key implementation choices

- **Per-bracket time series.**  For each (city, date='2026-05-06', venue,
  bracket_id) we compute midprice = (best_yes_bid + best_yes_ask)/2 at
  every snapshot where both sides are quoted.  Snapshots without a
  midprice are skipped (we do not impute).
- **Rolling stats with strict no-lookahead.**  At time t the rolling
  mean and stddev use ONLY observations t-N..t-1.  The current midprice
  is appended to history *after* the signal check, never before.
- **Signal.**  z = (mid_t − rolling_mean) / rolling_std.  Buy NO if
  z > k, buy YES if z < -k.  No trade if rolling_std is zero (constant
  history: typical of stale brackets).
- **Entry simulation.**  `open_position` lifts the appropriate ask
  ladder via `walk_ladder_buy` — for `no_long` this lifts the inverted
  yes-bid ladder.  Position size is fixed at 10 contracts so depth
  filters bind consistently.
- **Exit simulation.**  Custom in-script `_simulate_exit` that walks
  forward and exits on whichever fires first: a profit target evaluated
  from a real walk-the-bid simulation; a z-revert (|z|<exit_z, with z
  computed against the *entry-time* mean and std — keeping the
  reversion target fixed); the time stop; or end-of-window fallback.
- **Filters introduced in v2.**  min_depth = 50 contracts on both the
  entry-side and the exit-side of the book; max_spread cap; skip the
  last 60 minutes of the trading window where venues stale; skip
  midprices outside [0.05, 0.95] where reversion is structurally
  bounded.
- **One open position per bracket at a time.**  Tracked via an
  `open_brackets` set; freed when the position's exit_ts equals the
  current snapshot.

### What I discarded

- **Updating the rolling mean during the hold.**  The bracket's true
  level can drift while we hold; if I let mean and sd update, the
  z-revert exit triggers earlier (because the mean catches up to the
  spike) and my "win rate" inflates without my P&L improving.  I keep
  the entry-time mean/sd fixed for the exit rule.
- **k > 3.0 thresholds.**  Sweeping k up to 4.0 reduced trade count but
  did not improve P&L per trade — extreme spikes are *more* likely to
  be real information than noise.
- **A momentum variant (sanity check).**  I flipped the signal
  (`yes_long` if z > k) to confirm the data isn't trending into our
  spreads in some other direction.  Momentum loses too (Sharpe ≈ -0.5).
  Conclusion: the issue is structural cost, not direction.

## Backtest results

All trades on 2026-05-06 across {Miami, LA, Austin, San Francisco,
Seattle}.  Fee: $0.005/contract per side ($0.01 round-trip per
contract).  Size 10 contracts/trade.  Mark-to-market only (no
hold-to-settle, since the strategy intends intraday reversion).

| version | description                                                    | n   | win    | total $ | avg $/tr | Sharpe | dd $    |
|---------|----------------------------------------------------------------|-----|--------|---------|----------|--------|---------|
| v1      | bare baseline. N=20, k=2.0, exit_z=0.5, no filters             | 311 | 19.0%  | −133.89 | −0.4305  | −0.548 | −133.89 |
| v2      | + min_depth=50, max_spread≤5c, no opens in final 60 min        | 271 | 22.9%  | −101.34 | −0.3740  | −0.499 | −102.85 |
| v3      | + tight spread (≤2c) + profit-target $0.02 + 120-min time stop | 150 | 35.3%  |  −51.79 | −0.3453  | −0.381 |  −54.44 |

Best (N, k, exit_z) configurations from a 27-cell sweep under v2 filters:

| N  | k   | exit_z | n   | win   | total $  | Sharpe |
|----|-----|--------|-----|-------|----------|--------|
| 30 | 2.5 | 0.0    | 158 | 17.1% |  −71.43  | −0.524 |
| 30 | 2.5 | 0.5    | 166 | 20.5% |  −73.18  | −0.531 |
| 30 | 2.5 | 1.0    | 174 | 20.7% |  −74.91  | −0.548 |
| 20 | 2.5 | 0.0    | 197 | 19.3% |  −82.07  | −0.531 |

**Every configuration in every sweep loses money.**  v3 is "less bad"
but still has total P&L = −$51.79 across 150 trades.

## Diagnostic: is the underlying mean reversion real?

Yes, in the midprice signal.  Across 71 (venue, bracket) series with
≥20 observations on 2026-05-06:

| stat   | median | mean   | frac &lt; 0 / &lt; 1 |
|--------|--------|--------|--------------------|
| AR(1)  | −0.066 | −0.105 | 79% negative       |
| VR(2)  |  0.921 |  0.892 | 79% &lt; 1         |
| VR(5)  |  0.779 |  0.769 | 87% &lt; 1         |
| VR(10) |  0.686 |  0.694 | 85% &lt; 1         |

So the variance ratio at lag 10 is 0.69 of what a random-walk would
predict — there is real mean reversion in the midprice over a 50-min
horizon.  See `diagnostics.json` and `diagnostics.py`.  **The
autocorrelation finding is a real research result, but it is too small
to be tradeable on these markets.**

## Failure modes

**1. Spread + fee &gt; reversion magnitude.**  The median full bid-ask
spread on a quoted bracket is $0.010, and round-trip fees are $0.010 —
so a strategy that lifts the ask and hits the bid pays $0.020/contract
in fixed costs before the position even has to revert.  The mean
absolute Δmid step is $0.005; the median is $0.000.  The reversion has
to integrate over many steps to overcome the cost, and it usually
doesn't get there before something else moves the level entirely.

**2. Spikes carry information.**  Inspecting the worst-loss trades, the
"spike" was the start of a real persistent move, not noise.  Example:
SF Kalshi B67.5° went 0.41 → 0.46 → 0.45 → 0.47 (z=2.75 here, we sold
NO at $0.565), then continued 0.53 → 0.60 → 0.68 → 0.79 over the next
60 minutes.  This is the temperature outcome firming up — by mid-day a
~70°F observation in San Francisco makes the 67-68°F bracket
increasingly likely.  Mean reversion fights this real information
arrival.

**3. Asymmetric move at entry.**  When the midprice spikes, often only
one side of the book moves first (a single buyer lifts the ask, leaving
the bid stale).  We then enter against the move (lift the *new*, higher
ask) but our exit fill goes into the unmoved bid — we pay the spike,
sell pre-spike, double-pay the spread.

**4. Adverse exit reasons.**  In v2 (271 trades): 180 hit time-stop
(sum P&L −$104.24), 85 hit "target" (z-revert, sum P&L +$3.19).  Even
when the z exit fires, the median per-contract P&L on a "target" exit
is **−$0.006** — the bid did not actually catch up to the midprice
revert.  The z-revert event is mostly a cosmetic win.

**5. Perfect-foresight ceiling.**  Across 978 v2-filtered signal events,
even taking the **best possible** sell price within the next 30 min:

- mean gross P&L per contract = −$0.004
- median gross = −$0.010
- 26% of signals are profitable gross; 24% after $0.01 round-trip fee

So under perfect exit timing the strategy still loses on average.
Tightening to spread ≤ 2c and lookahead 60 min is the only
configuration where the perfect-foresight ceiling is positive
(+$0.018/contract, 46% win rate) — but the *realistic* version of that
config (v3) still loses, because the strategy can't pick the optimal
exit minute.

## Recommended next steps

- **Drop the spread-crossing assumption.**  This strategy could only
  work as a *limit-order* market maker — earning the spread by passively
  posting the contrarian side and hoping to get filled by the spike-makers,
  not paying it.  That's a different research problem (queue priority,
  adverse-selection-aware fade), and the snapshot data we have is too
  coarse (~5 min) to model fill probability convincingly.
- **Use a longer-horizon, cross-bracket prior** instead of one
  bracket's own history.  A bracket's "fair" mid is constrained by
  its neighbours summing to 1; deviations from the venue's own
  consistent set might be more tradeable than deviations from a single
  bracket's rolling mean.  (See strategy_3_structural_arb for a related
  attempt.)
- **Don't trade the cities/brackets/days where mean reversion is
  weakest.**  Seattle had the smallest fraction of negative-AR(1)
  brackets (64% vs ~85% elsewhere) but it's not enough — even the most
  reverting bracket cohort here doesn't beat the spread.
- **Honest takeaway: the autocorrelation result (VR(10) ≈ 0.69) is
  real and worth keeping in mind for other strategies, but on its own
  it does not yield a profitable directional trade after spread + fees
  on this data.**

## Files

- `backtest.py` — strategy implementation and the full backtest run.
- `diagnostics.py` — autocorrelation / variance-ratio analysis.
- `metrics.json` — canonical metrics dict (v1, v2, v3 + sweeps).
- `diagnostics.json` — per-(venue, bracket) AR(1) and VR(q) values.
- `trades_v{1,2,3}.csv` — every simulated trade with z-score, prices,
  exit reason, P&L.
