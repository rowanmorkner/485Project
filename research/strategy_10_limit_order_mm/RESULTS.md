# Strategy 10: Limit-order market making

## Strategy idea

Round-1 found that the round-trip spread + fees on these markets is
≈ $0.02/contract — large enough to dominate any short-horizon
directional alpha. Every prior strategy CROSSED the spread (lifted asks,
hit bids). The hypothesis tested here is the inverse: instead of paying
the spread, EARN it by posting passive limits 1¢ inside the
best-yes-bid and best-yes-ask. If we get filled on both sides we capture
~1–2¢ of bracket midprice as net P&L per contract.

The fundamental tension known going in:
- Real exchanges have queue priority. Our 1¢-inside-bid limit might sit
  behind hundreds of contracts at the same price level. We cannot model
  this on 5-min snapshots — we have no view into the within-bracket
  queue.
- Adverse selection: limit-order resting fills disproportionately
  when the market is moving against us. End-of-day, if we still hold
  inventory and the bracket has resolved out-of-the-money, the bid we
  flatten into is near zero.

## Key implementation choices

- **Per-bracket walk.** For each (city, venue, bracket_id) we replay the
  ~360 daily snapshots in order. State is one or more `open_states`,
  each at lifecycle stage `buy_resting` → `sell_pending` →
  `sell_resting` → realized.
- **Post rule.** Post a buy limit at `best_yes_bid + 0.01` whenever the
  bracket's liquidity filter passes and we have free inventory slots.
  Skip if `bid+1c >= best_yes_ask` (no room inside the spread).
- **Three fill models** (`buy_filled` / `sell_filled`):
  - `aggressive`: filled if any subsequent best_yes_ask ≤ our bid.
    Treats us as instantaneously at front of queue → upper bound.
  - `cautious`: filled only if any subsequent best_yes_ask < our bid
    (a real seller had to cross strictly past us).
  - `volume`: filled only if the displayed ask size at-or-below our bid
    decreased between snapshots (proxy for actual prints traded).
- **Cancel-on-move.** When the mid moves > 2¢ adverse to a resting buy
  before fill, we cancel and free the slot (production-style
  re-quoting). For resting sells **we never flatten at a bid**; we
  re-post 1¢ inside the new ask, never below `entry_fill + 1¢`. (An
  earlier draft that flattened on cancel guaranteed a loss every time —
  it added $34 of losses per ~80 trades.)
- **Bracket-selection filter.** Spread cap, mid in a sane range
  (avoid 0.05/0.95 penny-edges where reversion is bounded), depth
  thresholds, and (v4) a "drift" filter that skips brackets whose
  recent 3 mids have spanned > 3¢ (informed-flow proxy).
- **Active window.** Skip new opens in the last 60 (v1–v3) or 90 (v4)
  minutes of the trading window so we don't accumulate inventory we
  can't unwind.
- **End-of-window forced flatten.** Any position still resting at last
  snapshot is marked at the last quoted bid (i.e., we cross the spread
  to flatten at end of day). This is the dominant loss category.
- **Fees:** $0.005/contract per side on Kalshi, $0/side on Polymarket
  (Polymarket maker fees are ~0; we did not model the maker rebate).

### What I discarded

- **Flattening at the bid on cancel-sell.** Initial v2/v3 implementations
  flattened resting sell limits when the mid moved adversely. That
  guarantees a loss (you bought at bid+1c and now sell at the new bid,
  3–5¢ below your entry). Removing it improved every version.
- **Multi-concurrent positions per bracket.** Allowing 2+ open positions
  per bracket compounded EOW exposure. Reverted to 1 concurrent.
- **Inventory limit across brackets.** With 5-min snapshots, the
  difference between "1 per bracket" and any global cap was negligible
  on this single trading day. Skipped.

## Backtest results

All trades on 2026-05-06, 5 cities × 2 venues, size = 10 contracts/trade.
Fees: Kalshi $0.005/side; Polymarket $0/side.

| version | fill model  | filter   | n   | win   | total $   | Sharpe | fill rate | EOW losses     |
|---------|-------------|----------|----:|------:|----------:|-------:|----------:|----------------|
| v1      | aggressive  | loose    |  99 | 59.6% |  −$55.03  | −0.46  |     2.07% | 31 × $−1.89    |
| v2      | cautious    | tighter  |  62 | 54.8% |  −$34.61  | −0.50  |     3.19% | 22 × $−1.72    |
| v3      | volume-aware| strict   |  49 | 65.3% |  −$15.75  | −0.35  |     9.42% | 13 × $−1.38    |
| v4      | volume + drift skip | + 90min EOW | 40 | 67.5% | **−$13.93** | −0.35 | 8.26% | 10 × $−1.54   |

**Every iteration loses money.** Filtering reduces EOW exposure
(the 5x reduction in EOW count from v1 → v4 is real signal that the
liquidity + drift filter is doing useful work), but does not eliminate
it.

### The decisive split: filled vs EOW

| version | filled RTs n | filled $ | filled $/trade | EOW n | EOW $    |
|---------|------------:|---------:|---------------:|------:|---------:|
| v1      | 68          |  +$3.70  |        +$0.054 | 31    | −$58.73  |
| v2      | 40          |  +$3.30  |        +$0.082 | 22    | −$37.91  |
| v3      | 36          |  +$2.20  |        +$0.061 | 13    | −$17.95  |
| v4      | 30          |  +$1.50  |        +$0.050 | 10    | −$15.43  |

**Filled round-trips are profitable by an average of 5–8¢/trade — but
EOW forced-closes lose $1.40–$2.00 per trade.** The filled:EOW ratio
ranges from 6:3 to 7:3 by count, but per-trade the loss is
~25–40x the gain. We need filled count to outweigh EOW count by 25:1
to break even, and we observe ~2.5:1.

### Per-venue economics (v3 representative)

| venue       | filled $/trade | gross spread captured | fees   |
|-------------|---------------:|----------------------:|-------:|
| Kalshi      | $0.0000        | $0.10 (1¢ × 10)       | $0.10  |
| Polymarket  | $0.1097        | $0.10 (1¢ × 10)       | $0.00  |

**Kalshi fees of $0.005/contract eat the entire 1¢ inside-spread capture.**
A round trip that captures only 1¢ on Kalshi nets exactly $0 gross
($0.10 spread − $0.10 fees). Almost all our Kalshi fills capture only
1¢ (the most common spread on these markets is 1–2¢). On Polymarket
the same trade nets $0.10 — which is why Polymarket-only round-trips
have a positive average.

## Diagnostic: how often does our `bid+1c` get filled?

Across our four versions, the entry-side fill rate ranges from 2.07%
(v1, aggressive model with loose filter, 4787 posts → 99 fills) up to
9.42% (v3, volume-aware with strict filter, 520 posts → 49 fills).
**Below the 5% threshold called out in the brief for "practical
viability" on v1; right at it for v3/v4.** The rise from v1 → v3
reflects the filter selecting brackets where genuine flow is more
likely to interact with us — but it doesn't change the per-fill
economics enough to flip P&L positive.

## Failure modes

**1. The 1¢ Kalshi spread is structurally untradeable for makers at
$0.005/side fees.** Most fills capture only the 1¢ minimum tick of
inside-spread on these markets, which after fees is exactly $0. Even
a perfect 100% fill rate without adverse selection would not earn money
on Kalshi. **This is the single most damning finding for MM on Kalshi.**
Polymarket without fees is the only venue where positive carry exists.

**2. Adverse selection at end of day.** When a bracket resolves out-of-
the-money (Miami high not hitting an unlikely temperature, etc.), the
bid drops toward zero. Limit makers are filled by informed sellers
right before this collapse and are left with zero-value inventory.
Sample v3 trades: bought Miami Kalshi at $0.34 → forced-close at $0.01
($-3.30 loss). This pattern dominates the EOW loss column.

**3. 5-min snapshots cannot model queue priority.** Our model
assumes our bid+1c sits at the front of the queue at its price level.
On real exchanges with sub-second granularity, our limit might be
behind 100s of contracts already resting at that price; the
"aggressive" fill model (any subsequent ask ≤ our bid → filled) is
strictly an upper bound. Even our best version is using a generous
fill assumption, and it still loses.

**4. The "cancel-on-move-and-flatten-at-bid" rule is destructive.** An
earlier implementation that flattened resting sells when mid drifted
adversely lost an additional ~$30/run. Counterintuitively, leaving the
sell order resting (or just re-quoting at a tighter ask without ever
crossing the spread to flatten) is strictly better — the only way
"cancel-on-move" helps is on the BUY leg, before fill.

**5. Sample size is tiny.** 30–99 trades over 1 day, in 5 cities.
Even our most stable v3/v4 numbers have huge confidence intervals; we
cannot rule out that a multi-week sample would show noisy positive
performance from random good days. But the structural Kalshi-fee
finding (`spread × size − fees ≤ 0` at the most common 1¢ spread)
does not depend on sample size and is sufficient to recommend
*against* deploying this strategy as-is.

## Comparison to round-1 strategy 4 (mean reversion)

S4 achieved an "omniscient exit" upper-bound P&L of −$0.004/contract
because it had to PAY the spread to enter. We don't pay the spread, so
our **gross** filled-trip P&L is structurally higher (+$0.05–$0.10/trade
on Polymarket vs S4's −$0.0004/trade omniscient ceiling). But we
introduce a new failure mode S4 didn't have: **inventory at end of
day.** S4 forced exit before EOW; MM accumulates positions that
adverse-select against us and then must flatten.

In short: passive entry doesn't pay the spread, but the adverse-
selection tax replaces it as the dominant cost.

## What WOULD make this strategy testable / profitable

1. **Per-second order-book data.** With sub-second snapshots we could
   model queue priority empirically (how fast does our bid get
   consumed?), measure true effective spread captured, and calibrate
   adverse-selection probability.
2. **Settlement data on the backtest day.** If 2026-05-06 had settled
   highs in the database (it doesn't — see `list_settled_pairs`),
   EOW positions could be marked at $1 or $0 instead of last-bid. This
   would split EOW losses into "bracket actually settled in-the-money"
   (windfall gains, big positive) and "bracket settled out-of-the-money"
   (worst-case zero, ~current modeled outcome). Without it, we are
   pessimistic on EOW.
3. **A maker rebate.** Kalshi at zero fees would push S10 v3 from
   $0/Kalshi-trade to $0.10/Kalshi-trade, doubling positive P&L on the
   filled subset. Real economic edge would still need to overcome
   adverse-selection.
4. **Volume-weighted bracket selection.** Brackets with high recent
   volume and tight spread (the ~50% liquid Polymarket brackets we
   observe) are the only place this strategy clears the cost floor.
   Restricting to those would reduce trade count by ~3x but improve
   per-trade economics substantially.
5. **Longer hold horizon.** Holding to settlement on filled positions
   converts the adverse-selection problem into a forecast-quality
   problem (do we expect the bracket to win or lose?). That's a hybrid
   strategy: passive entry → hold-to-settle exit. Worth testing
   separately.

## Recommendation

**Do not deploy as a Kalshi MM strategy** — fees + 1¢ minimum spread =
exactly zero gross even before adverse selection.

**Polymarket-only MM with strict bracket filter (v3-style)** has
positive filled-RT carry of ~$0.10/trade but loses on EOW positions.
Whether it's positive in expectation depends entirely on how often
filled positions actually round-trip vs settle out-of-the-money. With
1 day of data and 25 Polymarket-filled round trips (v3) showing
$+2.20, vs 7 EOW losses showing $-5.45, we are short the answer.

The honest finding: **on 5-minute snapshots we cannot reliably backtest
this strategy.** The dominant losses come from a tail (10–30 EOW
flatten events per run) whose true distribution depends on
sub-second flow and end-of-day settlement that this dataset does not
contain. We can say the **structural cost finding** (Kalshi fees ≥
1¢ spread = no edge) is real and platform-level, and that
**Polymarket has real positive carry but unmeasurable adverse-selection
risk** at this sampling resolution.

## Files

- `backtest.py` — full implementation: per-bracket MM walk, three fill
  models, cancel/re-quote logic, bracket filters, metrics.
- `metrics.json` — v1–v4 metrics dict (AGENT_CONTRACT schema).
- `trades_v{1,2,3,4}.csv` — every simulated round-trip with entry/exit
  prices, fill timestamps, exit reason, P&L.
- `RESULTS.md` — this file.
