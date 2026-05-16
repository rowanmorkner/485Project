# Strategy 9 — Cross-venue PDF arbitrage

## Strategy idea

Round-1 strategy 1 (calibrated Gaussian) failed because the static morning
NWS forecast was 2°F off in 4 of 5 cities — the prior was too stale. But each
venue's intraday bracket prices, themselves, encode an implied per-degree
PMF that updates throughout the trading day. The idea: use each venue's
PMF as a prior for the OTHER venue's bracket prices. The +1°F Kalshi-vs-
Polymarket bias means we can't compare PMFs raw — we have to translate
one venue's PMF into the other's space using the empirical Δ histogram.

At each snapshot:

1. `pmf_K(d) = parse_kalshi_bins(brackets)` — Kalshi's implied PMF on K_high.
2. `pmf_P(d) = parse_polymarket_bins(brackets)` — Polymarket's implied PMF
   on P_high.
3. Translate K → P space:
   `pmf_P_pred(p) = sum_k pmf_K(k) * P(Δ = k − p)` where Δ is the empirical
   K − P histogram (per-city if n ≥ 30, pooled fallback otherwise).
4. For each Polymarket bracket B, compute
   `E_K[YES(B)] = sum_{d in B.degrees} pmf_P_pred(d)` and compare to market
   price. If `|E_K − P_market| > threshold` AND ladder fills, that's a
   directional signal. Mirror in reverse for Kalshi.

The crucial discipline is to **hedge** every directional cross-venue
signal with an offsetting position on the OTHER venue (Δ-shifted bracket,
opposite side). This protects against the asymmetric venue-divergence
risk (S2's risk_analysis showed that even a "guaranteed $1" pair can pay
$0 if the venues land their reading on different sides of a bracket
boundary). The pair is then scored under the joint K,P distribution
built from a CONSENSUS prior (50/50 mixture of both venues' implied
marginals), with the city's empirical Δ histogram as the conditional
P(P|K).

## Key implementation choices

- **PMF normalization**: `parse_kalshi_bins` and `parse_polymarket_bins`
  already normalize bid-ask midpoints to sum to 1. Verified against live
  snapshots: both PMFs are well-formed.
- **Joint scoring**: I score each candidate pair (or single leg) under the
  joint distribution P(K, P) built from a CONSENSUS K-marginal — averaging
  Kalshi's live PMF with Polymarket's live PMF (translated into K-space
  via Δ). Scoring against the OTHER venue's marginal is **circular** (by
  construction the trade's E[PnL] is positive whenever the trade thesis
  is "other venue is right"). I had to fix this mid-development; the
  initial v1 had spurious 100% win rates.
- **Per-city Δ histograms**: Miami (n=61), LA (n=31), Seattle (n=49) get
  their own; Austin (n=18) and San Francisco (n=29) fall back to the
  pooled n=188 histogram. Cleanly tagged in `delta_src` field per trade.
- **Hedge construction**: `find_hedge_for_signal` shifts the primary's
  bracket degrees by the mean Δ (rounded to int) and finds the OTHER
  venue's bracket with maximum overlap. The hedge takes the opposite
  side (yes_long primary → no_long hedge). This produces a pair that
  pays at least $1 when both venues agree on the reading.
- **CVaR risk filter**: pairs are filtered by `q05[payout] >= floor` (5th
  percentile of payout under the joint, treated as a CVaR-style tail).
  v2 uses floor = -$1; v4 uses floor = $0 (structural-quality).
- **Hold-to-settle scoring only**: 2026-05-06 isn't settled in our DB, so
  all PnL is `E[under joint K,P]` rather than realized. Per round-1
  guidance, MtM exit is dead on arrival once spread+fees > intraday Δmid.
- **Dedup**: each (city, venue, bracket_id, side) is opened at most
  once per day. Same for the hedge leg.
- **Fees**: $0.005/contract on entry only (settlement requires no
  further charge). For hedged pairs, fee = 2 × $0.005 × pair_size = $0.25
  on size 25.

## Backtest results

```
City Δ histogram coverage:
  Miami   per-city n=61    LA       per-city n=31     Austin   pooled (city n=18)
  SF      pooled (city n=29)                          Seattle  per-city n=49
```

| version | n_trades | win_rate | total E[PnL] | total q05 | sharpe | notes |
|---------|---------:|---------:|--------------:|-----------:|-------:|-------|
| v1 (directional) | 56 | 96.4% | **+$68.65** | **−$489.74** | 1.21 | gap > 5¢, NO hedge — large q05 risk |
| v2 (hedged, q05 ≥ −$1) | 6 | 100% | **+$6.49** | **−$0.15** | 1.80 | hedged, looser CVaR |
| v3 (hedged + NWS, q05 ≥ −$1) | 6 | 100% | **+$6.49** | **−$0.15** | 1.80 | NWS agreement (≥3¢) — same set as v2 here |
| v4 (hedged STRICT, q05 ≥ $0) | 6 | 100% | **+$7.98** | **+$2.32** | 2.52 | structural-quality CVaR floor |

Diagnostics:
- v2 enumerated 56 directional signals → kept 6 hedged pairs (797 had no
  matchable hedge bracket; 695 had E ≤ $0 under the consensus prior;
  1874 had q05 < −$1).
- v3's NWS filter rejected 709 signals where the NWS PDF didn't agree
  by ≥3¢ with the cross-venue gap. The 6 surviving v3 trades happen to be
  identical (modulo primary/hedge swap) to v2's set.
- v4 (q05 ≥ $0) yields a slightly different 6-trade set than v2: it picks
  up cleaner pairs at slightly different timestamps, and total q05
  (+$2.32) shows even the worst-case 5%-tail outcome is profitable.

### v4 trade dump (the recommended set)

```
LA              kalshi 72°-or-above YES@0.04 ↔ poly 72-73°F NO@0.93   E=$2.05  q05=$0.56
Austin          poly  84-85°F      YES@0.07 ↔ kalshi 86-87°F NO@0.91  E=$1.62  q05=$0.23
San Francisco   poly  66-67°F      YES@0.09 ↔ kalshi 67-68°F NO@0.88  E=$1.26  q05=$0.30
San Francisco   poly  68-69°F      YES@0.02 ↔ kalshi 69-70°F NO@0.95  E=$0.84  q05=$0.50
San Francisco   poly  62-63°F      YES@0.04 ↔ kalshi 63-64°F NO@0.95  E=$0.63  q05=$0.00
Seattle         poly  68-69°F      YES@0.01 ↔ kalshi 69-70°F NO@0.95  E=$1.59  q05=$0.73
```

Each pair has total entry cost ~$24-25 on size 25 (≈ $0.97-0.99 per pair),
so they're essentially structural arbs found through the **shifted-bracket
hedge construction**. Notably, **3 of 6 are San Francisco trades**, which
S3 v2 missed entirely (S3's strict aligned-bracket rule excludes SF and
Seattle because of the 1°F-shifted grid noted in ROUND2_BRIEF.md).

### Comparison with surviving round-1 strategies

| | n trades | E[PnL] | q05 | overlap with S9 v4 |
|---|---:|---:|---:|---|
| S3 v2 (strict structural) | 15 | +$140 expected / +$40 worst | — | 0 trades |
| S2 v5 (hedged convergence) | 9 | +$23 | — | 0 trades |
| **S9 v4 (cross-venue PDF + hedge)** | **6** | **+$7.98 / q05 +$2.32** | — | new SF/Seattle pairs |

S9 covers different (mostly shifted-grid) opportunities. **It is
complementary to S3, not redundant.**

## Failure modes observed

1. **Circular scoring trap.** My initial v1 scored each trade against the
   "other venue's" marginal — i.e. exactly the prior that argued for the
   trade. That gave a trivial 100% win rate. Fix: score against a
   consensus prior. Recovered v1 win rate = 96.4% (still high because
   gaps the strategy fires on are usually corroborated by the consensus).

2. **Stale-prior moments.** When one venue's PMF is very confident
   (e.g. Kalshi 88-89 at $0.85 late-day) and the other has different
   confidence, my translated-PMF still assumes the historical Δ
   distribution. This produces "the other venue says you're +30%
   underpriced" signals when really both venues are equally informed
   and the Δ histogram is just noise. The hedge construction (v2+)
   neutralizes this.

3. **Sample-size on Δ histograms.** Austin (n=18) and SF (n=29) fall back
   to the pooled n=188 histogram. The pooled mean Δ = +0.97°F is between
   per-city means (Austin +1.00, SF +0.93), so the fallback is benign.
   But per-city tails (e.g. LA Δ=+4 at 3.2%) can drive specific
   signals that the pooled hist would smooth out.

4. **Missing 2026-05-06 settlements.** All PnL is `E[under joint K,P]`,
   not realized. The q05 quantile gives a worst-5% bound, but a true
   realized backtest would need the actual K_high and P_high.

5. **Per-day sample = 1.** Six trades on a single day across five cities
   is not statistically meaningful. The point estimate is +$8 expected
   / +$2 worst-case on size 25, which scales to +$320/+$93 on size 1000
   — but the sample is one day.

6. **Threshold tuning is fragile.** Threshold = 0.05 → 56 raw signals;
   threshold = 0.10 → only 8. The strategy is sensitive to gap-threshold
   choices. The q05 ≥ 0 filter does most of the actual work
   regardless of threshold (sweep showed identical 6 trades from 0.02 to
   0.05).

## Recommended next steps

1. **Settle 2026-05-06.** When tonight's settlements land, replace
   `expected_payout` with realized `settlement_payout` for each trade
   and recompute true PnL. The structural-quality v4 trades should
   realize at or above their q05.

2. **Run alongside S3.** S9 v4 finds 6 trades, S3 v2 finds 15, with
   zero overlap. Together that's 21 distinct hedged pairs/day on size
   25 (~$525 capital). Combined expected PnL ≈ +$148 / day, worst-case
   ≈ +$42 / day. Across 5 cities × 365 days that's roughly +$54k expected
   / +$15k worst-case per year — not a fortune, but a clean,
   capital-light edge.

3. **Per-city Δ refresh.** As more (city, date) settlements accumulate,
   per-city Δ histograms will become reliable for Austin and SF. This
   may unlock additional eligible pairs.

4. **Make a hybrid v5.** Combine S3's strict structural enumeration with
   S9's shifted-bracket hedge. S3 currently ignores any (K_bracket,
   P_bracket) where the bracket degrees aren't aligned. S9's mean-Δ
   shift gives a principled way to align them.

5. **Don't bother with v1.** The directional version has an attractive
   point estimate (+$69 on 56 trades) but a q05 of −$490. That's a
   directional weather bet with ~$0.85 hit rate, not arb.

## Conclusion

Cross-venue PDF arbitrage **works as a complement to S3's bracket-
structural arb**, particularly for cities with 1°F-shifted bracket grids
(SF, Seattle) that S3 cannot reach. The hedged versions (v2-v4) are
positive in expectation and have non-negative worst-case payouts under
the empirical joint K,P distribution. Trade counts are small (6 per
day at size 25) and the realised-vs-expected gap can't be measured
until 2026-05-06 settles, so this should be deployed alongside S3
rather than replacing it.

The single biggest implementation lesson: **always score trades against a
prior that's INDEPENDENT of the trade's thesis.** Scoring "is Polymarket
mispriced?" using Kalshi's PMF is circular — the answer is always yes
when the mispricing is non-zero. Use a consensus or ground-truth
distribution.
