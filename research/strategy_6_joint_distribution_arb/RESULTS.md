# Strategy 6 — Joint-Distribution Structural Arbitrage

## Strategy idea

Round-1 winner S3 (`strategy_3_structural_arb`) found 15 cross-venue
"structural" arbs on 2026-05-06 with a quoted "100% worst-case win rate"
(+$40 / +$140 expected, 50 contracts/leg). But its worst-case math
implicitly assumed Kalshi and Polymarket both settle to the *same*
integer-degree high. The empirical evidence says otherwise: of 188
historical (city, date) pairs where both venues settled, only **25.5%
report identical highs**. Kalshi reads ~+1 °F warmer on average.

S6 keeps S3's enumeration / ladder-walking machinery but replaces the
"venues agree" assumption with the empirical joint distribution
P(K_high, P_high) provided by `joint_kp_distribution()`. For each
candidate pair I compute the true expected payout and the 5th-percentile
payout under that joint distribution, and use them as filter and score.

## Key implementation choices

- **Reused from S3** (imported / paralleled, not modified): bracket-set
  parsing (`_parse_kalshi_range`, `_parse_polymarket_range`), per-side
  ladder construction (NO ladder = `(1 - p, q)` on YES bid book),
  `walk_ladder_buy` for fills, the `(K_bracket × side_K × P_bracket × side_P)`
  enumeration loop, and the `(city, K_ticker, P_token, sides)`
  dedup-key. Pair size 50, MIN_PAIR_SIZE 5, fee $0.005/contract entry-
  only at settlement.
- **New scoring**: for each pair I build
  `f(k, p) = 1{k ∈ K_win} + 1{p ∈ P_win}` and compute
  `expected_payout_under_joint_kp(f, joint_kp)` and
  `quantile_payout_under_joint_kp(f, joint_kp, q=0.05)`. Joint distribution
  is per-city (`venue_divergence_histogram(city=...)` × `load_forecast_pdf`).
- **Hold-to-settle scoring** because that was the only S3 mode that
  worked (mark-to-market exit destroys the structural edge — S3 v1
  confirmed −$58).
- **Settlement check**: `load_settlement(city, '2026-05-06')` returned
  `None` for all 5 cities at the time this ran, so `settled_pnl_dollars`
  is 0 / null in `metrics.json`. The framework falls through cleanly to
  joint-distribution scoring; the moment 5/6 settles, rerunning would
  populate the `settled_pnl` field on each trade automatically.

## Approximation honesty

`joint_kp_distribution(forecast_pdf, city)` factors as
`P(K = k) · P(Δ = k − p | K = k)` with the empirical Δ histogram treated
as **K-independent**. Per-city Δ samples are 30–90, which can't refute
that. The forecast PDF (NWS) plays the role of P(K_high). Round-1 history
shows the live bot lost $18 M on PDF-driven trades, so v2 / v3 / v4 / v5
inherit any PDF miscalibration.

## Backtest results (5 cities, 2026-05-06, hold-to-settle)

| Version | Filter | n_trades | win_rate | scored PnL | E[PnL\|joint] | q05 PnL | S3-worst PnL | sharpe | max_dd |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 | S3 replication: structural ∧ cost<$1, score under K=P | 15 | 1.00 | **+$140** (S3 score) | +$126 | **−$410** | +$40 | 0.68 | $0 |
| v2 | E[payout\|joint] − cost > $0.05 | 607 | 1.00 | **+$4,408** | +$4,408 | **−$11,345** | −$18,045 | 1.15 | $0 |
| v3 | q05[payout\|joint] > cost (CVaR) | 130 | 0.68 | +$791 | +$791 | **+$410** | −$6,390 | 0.64 | −$17 |
| v4 | v3 ∧ asymmetric directional | 111 | 0.68 | +$609 | +$609 | **+$366** | −$5,334 | 0.71 | −$18 |
| v5 | v3 ∧ one trade per (K_ticker, K_side), one per (P_token, P_side) | **14** | 0.71 | **+$38** | +$38 | **+$45** | −$905 | 0.46 | −$2 |

Headline metrics chosen for `metrics.json` `total_pnl_dollars`:
- v1: S3-style expected payout (forecast PDF, K=P) — for direct
  comparability with S3 v2's published +$140
- v2/v3/v4/v5: joint-K-P expected payout

Per-city counts (v3 / v5): Miami 12 / 1, LA 38 / 3, Austin 27 / 3, San Francisco 35 / 4, Seattle 18 / 3.
Notice S3 found arbs almost only in Miami / LA / Austin; v3 finds them in
SF / Seattle too because the CVaR filter rewards the cross-bracket
"two-cheap-NOs-on-rare-brackets" structure that exists everywhere.

## The single sharpest finding

**Zero of S3 v2's 15 trades pass the CVaR filter.** Every one of S3's
"100% worst-case win rate" trades has q05_payout ≤ entry cost when scored
under the joint K-P distribution. In words: in 5%+ of plausible
(K_high, P_high) outcomes, *both legs lose*, and S3 silently took that
risk by assuming the venues report the same integer high.

Worked example (Miami `86°-87°` Kalshi YES + Polymarket NO, S3 v2 trade
#1): S3 scored worst-case = $1/pair guaranteed, expected = $1.18/pair.
Under the joint distribution: E = $0.92/pair, **P(payout = $0) = 15.7%**.
Failure mode is K=88, P=87 (probability 7.4%; both legs lose because K
is *above* the bracket while P is *inside* it). With a +$0.20/pair entry
margin S3 hoped for, this trade is genuinely positive-EV but
0% q05-payout.

When all 15 S3 v2 trades are re-scored under the joint distribution,
their 5th-percentile total PnL is **−$410** (vs. S3's quoted +$40). v1
is therefore the most important row in the table: the "winner" of round
1 was a forecast-conditional bet, not an arb.

## What v3 / v5 actually buys

Inspecting the v5 trade list (14 trades): most have side combo
`(no_long, no_long)` — buying NO on a low-probability bracket on each
venue. Top trade is LA `K=70°-71° NO + P=70-71°F NO`: cost $1.495,
E[payout] = $1.94, q05_payout = $2.00 (joint chance the actual high is
NEITHER inside K's nor P's bracket is essentially 100%, since LA on 5/6
is forecast far above 71°). It pays $2.00 in the 5%-tail world, so
q05 PnL = +$24.76 / pair × 50 contracts.

This is structurally a different beast from S3's aligned-bracket pairs:
v3 picks out brackets that the forecast says are far off the mode, so
both NOs are near-certain $1 contracts and buying them at $0.75 + $0.75
is a synthetic arb across venues. The CVaR filter happens to reward this
because the joint distribution puts ≥ 95% mass outside both brackets.

The asymmetric directional filter (v4) costs us 19 trades and ~$45 in
expected PnL by dropping `(yes_long, yes_long)` pairs where the Kalshi
bracket is *below* the Polymarket bracket — these are the cases where
Kalshi's +1 °F bias works *against* the hedge. Net: marginally improved
sharpe (0.71 vs 0.64), slightly lower edge.

## Failure modes

1. **Joint distribution is approximate.** Δ histogram has only 188
   samples, ~30-90 per city. Treating Δ as K-independent is a
   modeling choice, not a fact. If hot days have systematically different
   Δ behavior than cool days, our q05 estimates are off.
2. **Forecast PDF is a known weakness.** The same NWS PDF that lost the
   live bot $18 M is the foundation of P(K). v2 in particular is naked
   PDF risk: 607 forecast-conditional bets with no worst-case floor and
   a −$11,345 q05 PnL.
3. **v3's 130 trades double-count Kalshi positions.** With 22 unique
   K-tickers across 130 v3 trades, the same Kalshi leg appears in ~6
   pairs on average. v5 fixes this by enforcing one trade per K/P
   position, dropping the count to 14 and the joint-q05 PnL to +$45 —
   probably the most realistic capital-deployable number.
4. **The "two cheap NOs" pattern is a directional bet on the forecast
   being roughly right.** If the NWS PDF is wildly off (e.g. LA actually
   hits 70 °F instead of the forecasted 80s) v5's biggest trade pays
   $0 instead of $2. The CVaR filter at 5% protects against
   *frequent* small mis-pricings but not against an unlikely catastrophic
   PDF failure.
5. **No 5/6 settlement at runtime.** All `settled_pnl` fields are null.
   The cleanest validation will come from rerunning after settlements
   land in `data/bot.db`.

## Recommended next steps

- **Re-run after 5/6 settles.** v5's 14 trades have a clean expected /
  q05 spread (+$38 / +$45). When settlement is known, each trade gets
  a deterministic realized payout (0/1/2 per pair) — this is the gold
  standard validation. The framework already computes this when both
  venues' highs are present in `settlements`.
- **Production-grade rule = v5.** v3 over-counts capital usage; v2 is
  PDF-naked; v1 is the very thing we're fixing. v5 is conservative,
  capital-realistic, and structurally-honest under venue divergence.
- **Tighten the joint distribution.** With more (city, date) pairs,
  re-fit Δ conditional on K bucket (e.g. cool/mild/hot days).
- **Layer S2's hedge framing on top of v5.** S2's risk_analysis showed
  the asymmetric Kalshi-warmer effect explicitly. v4 only used a coarse
  bracket-center comparison; a per-pair expected-Δ-at-bracket-center
  filter could squeeze more edge out without bleeding trade count.
- **Compute `settled_pnl_dollars` on a multi-day rerun.** The
  framework already pulls `load_settlement(city, date)` per trade —
  point it at every backtestable (city, date) where both venues
  settled and you'll get a real out-of-sample win rate for the CVaR
  filter, not just a forecast-conditional one.
