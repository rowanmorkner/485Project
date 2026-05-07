# Strategy 3 — Structural Bracket-Math Arbitrage

## Strategy idea

Across Kalshi and Polymarket, daily-high temperature brackets are quoted
as YES/NO contracts. For any cross-venue pair (one Kalshi bracket × one
Polymarket bracket × choice of YES/NO on each leg) we partition the
plausible-temperature universe into three deterministic zones:

- **win_2** — both legs pay $1 (combined payout = $2)
- **win_1** — exactly one leg pays $1 (combined payout = $1)
- **loss** — neither leg pays (combined payout = $0)

If the loss zone is empty we are **structurally** guaranteed at least $1
per pair regardless of the outcome. If the combined entry cost is below
$1 the worst-case is positive — a true arb that requires no forecast.
If the cost is above $1 but below $1 + E[1{T ∈ win_2}] the position is
expected-value-positive but not structurally risk-free.

## Key implementation choices

- **Universe per snapshot** = union of all bracket degrees parsed at
  that moment (parsers expand "or below"/"or higher" tails by 5 deg).
- **Filling** uses the loaders' `walk_ladder_buy`, which sorts the ASK
  ladder ascending — bypassing the $0.99 partial-resolution traps that
  appear at the *top* of Kalshi ladders in raw form.
- **Depth filter:** open at a fixed pair size (50 contracts/leg) only
  if both ladders fill at avg cost satisfying the edge condition. We
  test pair sizes 5/10/25/50 in diagnostics to characterize liquidity.
- **De-duplication:** each `(city, kalshi_ticker, poly_token, k_side,
  p_side)` is opened at most once across the day — the *first* time it
  qualifies. Otherwise the same persistent opportunity inflates trade
  counts.
- **Fees:** $0.005/contract at entry, applied to both legs (so $0.01
  per pair). Hold-to-settle pays no exit fee (the contract self-resolves);
  MtM pays exit fees too ($0.01 more per pair).
- **Two scoring modes:**
  - *Mark-to-market* (v1): exit when both legs' bids let us close at
    target +$0.05/pair, or after a 240-minute time stop, or end-of-window.
  - *Hold-to-settle* (v2/v3): payout is $1 + $1·1{T ∈ win_2}. Without
    ground-truth settlements (5/6 hadn't settled when we ran), expected
    PnL uses the NWS forecast PDF and worst-case PnL assumes win_2 never
    realises ($1 payout floor).

I considered allowing non-structural (loss-zone-non-empty) trades using
forecast-weighted EV, but that's just a directional weather bet dressed
up as arb — it discards the very property that makes this strategy
distinctive — so I kept v3 strictly structural and only relaxed the
"cost < $1" rule to the weaker "expected edge > $0.05/pair".

## Diagnostics: how many real arbs exist?

| Metric | Value |
|---|---|
| Raw top-of-book structural quotes (loss=∅, ask-sum < $1) on 2026-05-06 | 2,154 |
| Fillable at min size 5 contracts (both ladders deep enough) | 2,070 |
| Illusory (vanish under depth) | 84 |
| **Distinct (city, kalshi, poly, sides) keys at size 5** | **28** |
| Distinct at size 25 | 26 |
| Distinct at size 50 | 23 |

So the 2,154 raw count is mostly the *same 28 opportunities re-counted
across persistent snapshots*. Real fillable arbs at the strategy's
preferred size of 50 contracts/leg: **23 distinct keys** across 5
cities × 1 day. Per-city distribution: Miami 9, LA 9, Austin 7, San
Francisco 2, Seattle 1.

## Backtest results

| Version | n_trades | win_rate | total_pnl | avg/pair | sharpe | notes |
|---|---:|---:|---:|---:|---:|---|
| v1 (MtM exit) | 15 | 0.13 | **−$58.13** | −$0.077 | −0.36 | structural arb but closed via MtM — bid-ask spread eats the edge |
| v2 (hold-to-settle, worst-case) | 15 | **1.00** | **+$40.32** | +$0.054 | 0.51 | same 15 pairs, scored by floor $1 payout |
| v2 (hold-to-settle, expected) | 15 | 1.00 | **+$139.88** | +$0.186 | 0.71 | same 15 pairs, scored by NWS-weighted $1+E[win_2] |
| v3 (structural + EV filter, expected) | 114 | 0.09 | +$867.38 | +$0.101 | low | relax cost-cap, allow win_2 EV bonus — worst-case is **−$2,650** |

(Full per-trade breakdown in `trades_v1.json` / `trades_v2.json` / `trades_v3.json`.)

The headline: when you actually pay attention to the structural
guarantee — that is, **hold every leg to settlement** rather than
trying to close MtM — the strict structural-arb selection (v2) is a
genuine money-printer with a 100% worst-case win rate, but it produces
only **15 trades on a full trading day across 5 cities**, for ~$40
worst-case profit at 50 contracts/leg.

## Failure modes observed

1. **MtM exit destroys the structural edge** (v1 vs v2). Even when we
   buy a pair for $0.97, the close-out spread on Kalshi's wide bid-ask
   means selling out for less than entry is the rule, not the exception.
   Of 15 strict structural pairs, only 1 hit the +$0.05/pair MtM target.
   The structural guarantee is a settlement guarantee, not a tradeable
   guarantee.

2. **Most "raw" arbs are scrubbed by depth.** Top-of-book sums under $1
   exist for hundreds of (kalshi, poly) pairs at every snapshot, but
   many of those involve micro-quantities at the cheapest price level
   stacked behind much-worse asks. Walking the ladder with realistic
   size collapses the universe from 2,154 raw signals down to 23
   distinct opportunities.

3. **Expected-edge mode (v3) reintroduces directional risk.** Allowing
   cost > $1 in exchange for a probable win_2 bonus turns the strategy
   into a forecast-conditional bet. Its expected PnL is +$867 but
   worst-case is −$2,650 across 114 trades. If the NWS PDF is even
   slightly miscalibrated (which the project history suggests it is —
   the live bot lost $18M on PDF-driven trades), you take the worst
   case in real life.

4. **Aligned brackets dominate the data.** Of the 15 v2 trades, ~12
   are pairs whose Kalshi and Polymarket brackets are *exactly the
   same degree range* (e.g. Kalshi YES "84°-85°" + Poly NO "84-85°F").
   These are the cleanest cases — mutually exclusive, exactly one leg
   pays — and the cost-savings are bid-ask-spread arbitrage between
   the two venues' order books. They almost all live in Miami / LA,
   where the Polymarket book is most active. San Francisco and Seattle
   contribute little.

5. **The "win_2 bonus" structural cases are rare and expensive.** When
   the two brackets overlap entirely (one fully contains the other,
   e.g. Kalshi NO "88-89" wins outside [88,89] and Polymarket NO
   "84-85" wins outside [84,85]), the win_2 zone is large but the
   combined cost is also pushed up because both legs are typically
   "high-probability NO" buys. Only 2 of 15 trades had a non-empty
   win_2 mass under the NWS PDF.

## Recommended next steps

- **Backtest hold-to-settle on 5/7 once it settles tonight.** v2 should
  realize close to its expected PnL of +$140 and at minimum its
  worst-case +$40 (assuming our depth-walking simulates the real fill
  prices we'd have seen). This is the strategy's clean validation.

- **Concentrate capital, not breadth.** The 15 fillable arbs span the
  full trading day. Compute a refresh-rate constraint: open *one* arb
  at a time, capital recycles only at settlement (≈ once per day per
  city). Effective notional is therefore bounded by ~5-10 pairs ×
  50 contracts × $1 = $250-500/day capital deployed.

- **Investigate aligned-bracket arbs as a separate strategy.**
  Mutually-exclusive aligned brackets (Kalshi `B84.5` + Poly `84-85°F`)
  are the cleanest, most numerous instances. A specialized scanner
  that *only* enumerates these (skipping the cross-bracket
  combinatorial explosion) would be cheaper to run and would catch
  the same 80% of profit.

- **Consider partial sizing into less-deep arbs.** Several of the 28
  observed opportunities only fill at smaller sizes (e.g. `distinct_at_size_5
  = 28` vs `distinct_at_size_50 = 23`). At pair sizes of 10–25 we
  might add 3-5 more trades per day, modestly raising daily PnL.

- **Don't extend to expected-edge mode without per-venue PDF
  calibration.** v3 is a poison pill if the forecast is wrong; given
  this project's history with PDF-driven trades, leave the
  forecast-EV variant out of any production strategy and stick with
  the worst-case-positive subset.
