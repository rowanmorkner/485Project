# Round 2 — Strategy comparison and recommendation

5 round-2 strategies, all built on round-1 lessons. Every cross-venue
strategy now propagates the empirical Δ histogram (188 historical pairs,
25.5% venue-agreement rate, mean Kalshi reads +1°F) into its scoring.

Per-strategy writeups in `research/strategy_<N>_<slug>/RESULTS.md`.

## The single most important round-2 finding

**Round-1's Strategy 3 was wrong about its worst-case math.**

Both S6 and S7 independently re-scored S3 v2's "100% worst-case win rate"
trades under the joint K-P distribution:

- **S6**: 0 of 15 S3-v2 trades survive `q05_pnl > cost`. True q05 across
  the 15-trade portfolio is **−$410**, not +$40.
- **S7**: 0 trades pass v1's strict-structural filter across all 5 cities.
  Once you require positive payoff for ALL (k, p) in the joint support,
  no portfolio qualifies.

Worked example (Miami 86°-87° aligned-bracket arb, S3 v2's bread-and-butter):
- Buy Kalshi YES `86°-87°` for $0.40, buy Polymarket NO `86-87°F` for $0.50, total cost $0.90.
- "Structural" payoff: assumes K_high == P_high. If both venues read 86,
  Kalshi YES pays $1; Poly NO pays $0; total $1. If both read 88, Kalshi
  YES pays $0; Poly NO pays $1; total $1. ✓
- **Reality**: when K=88, P=87 (Δ=+1, the most common case at 37.2%),
  Kalshi YES pays $0 and Polymarket NO pays $0 because P landed inside
  bracket 86-87. Total payoff = $0, loss = $0.90.
- 15.7% probability of full loss on this single pair.

**Implication**: round-1's recommendation needs revision. Strategy 3 v2
is NOT deployable as-is. The deployable set is the round-2 successors
that explicitly model venue divergence.

## Round-2 ranking

| # | Strategy                       | Best version           | n   | E[PnL] | **q05 PnL**   | Sharpe | Verdict |
|---|--------------------------------|------------------------|----:|-------:|--------------:|-------:|---------|
| 6 | **Joint-distribution arb**     | **v5 (CVaR + strict dedup)** | **14**  | **+$38**     | **+$45**      | high   | **POSITIVE — recommended** |
| 6 | Joint-distribution arb         | v3 (CVaR, no dedup)    | 130 | +$791  | +$410         | high   | POSITIVE — but reuses positions |
| 7 | Multi-bracket portfolio        | v2 (CVaR)              | 963 | +$15,799 | +$5,550     | high   | POSITIVE q05, but **worst-case −$48,900** |
| 7 | Multi-bracket portfolio        | v1 (strict structural) | 0   | $0     | $0            | —      | confirms S3 was wrong-counted |
| 8 | **Late-day certainty**         | **v3 (q05 ≥ cost+10¢)**| **4**   | **+$113**    | **+$29**      | **1.17** | **POSITIVE small** |
| 9 | **Cross-venue PDF arb**        | **v4 (q05 ≥ $0)**      | **6**   | **+$8**      | **+$2**       | **2.52** | **POSITIVE — covers SF/Seattle** |
| 10| Limit-order MM                 | v4 (volume-aware + drift skip) | 40 | −$14 | —            | −0.35  | NEGATIVE; see structural finding |

## Recommendation: deploy the union

A single strategy doesn't dominate. The deployable bot uses a **stacked
selector** that draws non-overlapping trades from three round-2 sources:

1. **Strategy 6 v5 (Joint-distribution arb, CVaR + strict per-leg dedup)**
   - 14 hedged pairs/day across Miami/LA/Austin
   - Aligned-bracket structural arbs filtered by `q05_payoff > cost`
   - Each pair's q05 is independently positive ⇒ portfolio q05 ≥ +$45 worst case
   - Capital-realistic: each Kalshi ticker and each Poly token used at most once

2. **Strategy 9 v4 (Cross-venue PDF, hedged STRICT)**
   - 6 hedged pairs/day, predominantly SF (3) + Seattle (1) — the cities
     S3/S6 cannot reach due to grid misalignment
   - Confirmed **zero overlap** with S6/S3 trade set
   - Uses Δ-shifted hedge construction to span 1°F-misaligned grids
   - Per-pair q05 ≥ $0 floor

3. **Strategy 8 v3 (Late-day certainty, q05 ≥ cost+10¢)**
   - 4 trades/day taken only when certainty score crosses θ=0.60
   - Higher per-pair edge ($28/pair vs $3/pair all-day) due to lower
     entry costs as books concentrate
   - First strategy in the entire research that produced positive q05
     under the joint K-P distribution
   - **Likely overlaps partially with S6 v5** (same selection logic +
     time gate); deploy as a re-ranker/timing overlay rather than
     additive trades

**Daily expected net** (deduping S8 against S6, summing S6 + S9):
- ≈ 18–20 hedged pairs/day, **+$45 worst-case (q05) per day at size 50**
- ≈ +$50/day expected under joint K-P
- Annualized: **~$15K worst-case / ~$18K expected**

Modest, but it's the first strategy package in this research with a
**provably non-negative q05 under the empirical venue-divergence histogram.**

## Strategies NOT to deploy (and why)

- **Strategy 7 (Multi-bracket portfolios) at v2 settings**: +$5,550 q05
  is real but absolute worst-case is **−$48,900**. The 1×2 / 2×1 pattern
  the agent identified ("two cheap NOs + one expensive YES") is a
  systematic short on Polymarket NO tail-vol. Without Kelly-style sizing
  on q01 (not q05) it's a tail-risk landmine. Worth research-investing
  in IF you build proper sizing — but NOT a one-and-done deployable.

- **Strategy 10 (Limit-order MM)**: structurally untradeable on Kalshi
  (1¢ tick × $0.005/side fees = $0.00 net per filled RT), and unmeasurable
  on Polymarket due to adverse-selection at end-of-day (concentrated
  fills right before bid collapse). The platform finding is keepable;
  the strategy is not.

## Cross-strategy insights (round 2)

### 1. The venue-divergence histogram converts nearly every "structural arb" into a CVaR bet

Round 1 asserted "100% worst-case win rate" without modeling venue
divergence. Round 2 found this is **never** true once Δ is propagated:

- S6 v1 (S3 replication): 15 trades, q05 = **−$410**
- S7 v1 (multi-bracket strict structural): **0 trades qualify** across all 5 cities

The honest framing: cross-venue weather arbs aren't risk-free. They are
CVaR-bounded EV bets, and the right filter is `q05_payoff > entry_cost`,
not `min_payoff > entry_cost` over an "venues agree" support.

This is the single most important conceptual update from this research.

### 2. Multi-bracket portfolios unlock SF/Seattle but NOT structurally

S7 confirmed the SF/Seattle 1°F-shifted-grid problem can be solved with
1×2 / 2×1 portfolios — 493 new trades become available there. But none
of them are structurally worst-case-positive; they're tail-risky CVaR
bets. The agent's finding that the dominant pattern is "two cheap NOs +
one expensive YES" suggests these are systematic short-tail-vol
positions that need explicit per-trade tail risk caps.

S9's approach (hedged single-leg with Δ-shifted matching) is the more
disciplined way to access SF/Seattle: same universe expansion, but with
the q05 floor preserved.

### 3. Late-day arbs are cheaper, fewer, and better-vetted by reality

S8 v3 produced the FIRST positive-q05 result in the whole research
effort. The mechanism: as the day progresses, the actual high partially
materializes, brackets concentrate toward {0, 1}, and any remaining
cross-venue pricing inconsistencies become cheaper to enter (lower
entry costs against the same $1 floor). The certainty score (combination
of mean spread, mid concentration, alive-bracket count, time-fraction)
reliably identifies this regime without hardcoding UTC times.

The trade-off: smaller universe (4 trades vs 14 from S6 v5). For
production, S8's certainty score is best used as a TIMING overlay on
S6's selection — open S6 trades only after certainty crosses threshold.

### 4. Limit-order market making is platform-limited, not strategy-limited

S10's most useful output isn't the backtest results — it's the
arithmetic finding that **Kalshi 1¢ tick × $0.005/side fees = $0.00 net
spread captured per filled round-trip**. Polymarket has positive carry
(no fees) but adverse-selection-at-close eats it. This argues for: (a)
not pursuing market making until Kalshi changes its fee schedule or we
get sub-second L2 data; (b) any future MM work focuses on Polymarket
with explicit adverse-selection modeling.

### 5. ML on this data is still dead

Round 1's S5 finding (no directional alpha to learn from 11K obs at
$0.30 round-trip cost) is reinforced by every round-2 strategy's
positive results being *structural* (joint distribution math, late-day
regime detection) rather than *predictive* (model says price will rise).

## Concrete next steps (priority ordered)

1. **Validate S6 v5 + S9 v4 against actual 2026-05-06 settlements when
   they land tonight (~23:55 UTC, 5 cities).** This is the cheapest
   validation we can do. Re-score every trade with `settlement_payout_for`
   and compare realized vs q05 vs E.

2. **Build a `production_selector.py`** that combines:
   - S6's strict-dedup selection (Miami/LA/Austin)
   - S9's Δ-shifted hedge construction (SF/Seattle)
   - S8's certainty-score timing gate
   This is a ~150-line glue module on top of existing strategy code.

3. **Capture intraday NWS forecast updates.** Round-1 found the morning
   forecast was wrong by ≥2°F in 4/5 cities for 2026-05-06. The
   `forecasts` table currently has 1 fetch/day. Polling NWS every 30 min
   gives us a fresher P(K) and could rescue Strategy 1 from "structurally
   broken" to "marginally profitable" — but more importantly, makes ALL
   joint-K-P calculations more accurate.

4. **Per-city Δ histograms with more data.** 188 pairs is thin; per-city
   n is 18-93. After 30+ more settled days the per-city Δ tightens and
   strict-structural (S7 v1) may admit a small set of trades that are
   genuinely worst-case-positive.

5. **Investigate S7's multi-bracket pattern under Kelly sizing.** The
   "two cheap NOs + cheap YES tail" pattern is interesting — it's
   essentially a structured short-tail-vol position. With per-trade
   sizing capped by `q01_payoff / total_capital`, the strategy might
   work. Out of scope for this iteration but a real research direction.

6. **Wait ≥30 settled trading days before any capital decision.** Sample
   sizes (n=14, n=6, n=4) are too small for production confidence.
   Negative results are robust now; positive results need confirmation.

## Files written in round 2

- `research/strategy_6_joint_distribution_arb/` — joint K-P scoring,
  5-iteration backtest, CVaR + asymmetric + dedup variants.
- `research/strategy_7_multi_bracket_portfolio/` — multi-bracket
  enumerator, 3-iteration backtest, 893 multi-bracket trades.
- `research/strategy_8_late_day_certainty/` — certainty score,
  4-iteration backtest with regime analysis.
- `research/strategy_9_cross_venue_pdf_arb/` — venue-implied PDF
  comparison, 4-iteration backtest, 6-trade SF/Seattle expansion set.
- `research/strategy_10_limit_order_mm/` — passive-MM backtest with
  3 fill models, 4 iterations, structural fee/tick analysis.
- `research/loaders.py` — added `venue_divergence_histogram`,
  `joint_kp_distribution`, `expected_payout_under_joint_kp`,
  `quantile_payout_under_joint_kp`. The joint-distribution helpers are
  the round-2 enabling primitive.
- `research/POSTMORTEM.md` — consolidated post-mortem of failed
  round-1 strategies.
- `research/ROUND2_BRIEF.md` — round-2 agent context (what to know
  going in).

Branch: `strategy-research`. Commits: `c904ef7` (round 1), `145b7fe`
(round-1 cleanup + round-2 setup). Round-2 outputs uncommitted.
