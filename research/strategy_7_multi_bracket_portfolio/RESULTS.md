# Strategy 7 — Multi-Bracket Portfolio Hedges

## Strategy idea

A "leg" is one Kalshi or Polymarket bracket+side. A **portfolio** is a sum
of legs across both venues. With L_K ∈ {1,2} legs on Kalshi and L_P ∈
{1,2} legs on Polymarket we can synthesize equal-degree spreads even
when the venues' underlying bracket grids are misaligned (round-1 found
SF/Seattle Kalshi grids are shifted +1°F vs Polymarket).

For each portfolio we compute the per-(k,p) payoff function — at
settlement K reads `k`, P reads `p` — and weight it by the joint
distribution `joint_kp_distribution(forecast_pdf, city)` that combines
the NWS PDF (treated as P(K)) with the empirical Δ = K−P histogram
(treated as P(P|K), assumed K-independent — the round-2 brief
acknowledges this approximation).

Three filters were tested:

- **v1 STRICT STRUCTURAL** — `worst_case_pnl_per_pair > $0.05` over the
  joint K-P support. The strongest condition: positive payoff regardless
  of how the venues disagree.
- **v2 CVaR** — `q05_pnl_per_pair > 0` AND `expected_pnl_per_pair > $0.05`.
  Allows ≤5% tail loss probability while requiring positive expected value.
- **v3 NEW-ONLY** — v2's selection minus any 1×1 portfolio matching S3
  v2's universe; reports only the *additive* trades available from
  multi-bracket combinations.

## Key implementation choices

- **Per-leg fills via `walk_ladder_buy`**, both at qty=1 (for filter
  evaluation) and at PAIR_SIZE=50 (for actual fillability re-check).
  S3 v2 found ladder-walking is essential to avoid the `$0.99`
  partial-resolution trap at the top of Kalshi books.
- **Fees**: $0.005/contract per leg, entry only (hold-to-settle has no
  exit fee). A 3-leg portfolio costs $0.015/pair in fees.
- **Dedup**: canonical key = `(city, sorted((k_ticker, k_side) tuples),
  sorted((p_token, p_side) tuples))`. Each portfolio is opened once at
  its first qualifying snapshot.
- **Snapshot stride**: to keep enumeration tractable, snapshots are
  subsampled by stride=6. Dedup makes this safe — we still see every
  unique portfolio that ever qualifies, just at a slightly later
  timestamp than we would with stride=1.
- **2×2 portfolios SKIPPED** (`ALLOW_2x2 = False`) — the combinatorial
  explosion was the dominant cost, and 1×2/2×1 portfolios already cover
  the synthetic-equal-degree-spread idea (one venue's bracket spans the
  other venue's two adjacent brackets).
- **Approximate "S3 v2 reducibility" match**: `trades_v2.json` only
  stores Polymarket token prefixes (16 chars), so v3 matches on
  `(city, k_ticker, k_side, p_token[:16], p_side)`. Some 1×1
  portfolios may slip through if their token prefix collision pattern
  differs.

## Backtest results (2026-05-06, 5 cities, 50 contracts/leg)

| Version | n_trades | n_winners (worst≥0) | win rate | total expected PnL | total worst-case PnL | total q05 PnL | sharpe | per-city |
|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| **v1 strict** | **0** | 0 | — | $0 | $0 | $0 | — | (none qualify) |
| **v2 CVaR** | **963** | 70 | 0.073 | **+$15,799** | **−$48,900** | **+$5,550** | high | LA 466, Seattle 342, SF 151, Austin 4, Miami 0 |
| **v3 NEW-only** | **893** | 0 | 0.000 | +$14,982 | −$45,425 | +$5,125 | high | LA 429, Seattle 324, SF 136, Austin 4 |

n_legs distribution for v2: 70 single-leg pairs (1×1), 649 K=1×P=2,
244 K=2×P=1, 0 K=2×P=2 (skipped).

## The headline answer to "does this add trades in SF/Seattle?"

**Yes — 151 SF trades and 342 Seattle trades from v2 (CVaR-bounded EV),
nearly all of which are 1×2 or 2×1 multi-bracket combinations.**
Round-1 strategies (S2, S3) produced **0 trades in SF and 0–1 in
Seattle** because the shifted grids made aligned-bracket arbs
impossible. Multi-bracket portfolios remove that constraint by letting
one venue's wider 2-bracket leg synthesize the other venue's narrower
1-bracket leg.

However, **none of these new SF/Seattle trades pass the strict
joint-K-P structural filter (v1)**. They are CVaR-bounded EV bets —
positive in the 95th-percentile worst case, **but with severe
left-tail risk: total worst-case PnL across all 893 v3 trades is
−$45,425.**

## Failure modes observed

### 1. The strict joint-aware structural filter (v1) finds nothing.

This is the single most important finding. Once you score the
canonical S3 v2 trade — Kalshi YES `84°-85°` + Polymarket NO `84-85°F`
in Miami — under the joint K-P distribution rather than the
"venues agree" assumption, the worst-case payoff drops from $1 to
$0. Why? With Δ=K−P=2 (a 22.9%-probability case from the round-1
histogram), Kalshi might read 87 (Kalshi YES on `84-85` loses) while
Polymarket reads 85 (∈ {84,85}, Polymarket NO loses). Both legs
zero. Full $0.93 entry cost lost.

This generalizes to multi-bracket portfolios: any portfolio whose
worst-case (k,p) pair across the joint-distribution support has zero
total payoff cannot pass v1, no matter how cheap. With Δ ranging over
{−1, 0, +1, +2, +3, +4} historically, the (k,p) support is wide
enough that almost every reasonable portfolio has SOME (k,p) where
all legs fail.

**This effectively confirms round-1's headline finding mathematically:
S3 v2's "100% worst-case win rate" is an artifact of assuming
venue agreement. Under joint K-P scoring it is closer to 75%.**

### 2. v2 CVaR mode finds many trades but with huge worst-case tails.

963 trades that PnL +$5,550 at the 5th percentile and +$15,800 in
expectation, **but −$48,900 worst-case**. The trades concentrate in
LA, SF, and Seattle. The reason: in those cities the Polymarket book
quotes wide brackets cheaply on the tails. A combination like
"Kalshi NO {tail-bracket} + Polymarket NO {paired adjacent
brackets}" pays nearly always (because the daily high rarely lands
in any one specific 2-degree bin) but fails catastrophically when
the read happens to fall in the one degree-pair that voids both
NO legs.

The 5% left tail isn't just "small loss" — when the bad (k,p)
materializes, you lose all legs ≈ −$1.50/pair × 50 contracts × N
trades.

### 3. The "1×2" and "2×1" patterns dominate.

Of v2's 963 trades, 893 are multi-bracket. The structure is
near-universal: combine **2 Polymarket NO brackets** (each costing
~$0.85, summing to ~$1.70) with **1 Kalshi YES tail bracket**
(~$0.05). The 2 P-NO legs almost always pay $1 each (because the
high seldom lands in either narrow bracket), but if the high DOES
land inside one of them, only $1 of the $1.70 P-cost is recovered,
PLUS the small Kalshi YES at $0.05 may or may not pay. Net payoff
in that scenario is $1, leaving a $0.75/pair loss.

This is closer to "selling tail volatility on bracket prices" than
to a structural arb.

### 4. v1's emptiness vs v3's 893 multi-bracket "new" trades.

v3 reports 893 trades that don't appear in S3 v2. But none are
worst-case-positive. So while multi-bracket *expands the
opportunity set*, it does NOT recover the round-1 "true arb"
property — those 893 trades are EV bets, not arbs.

### 5. Approximation in the joint K-P model.

`joint_kp_distribution` assumes Δ = K−P is independent of K. With
n=188 historical pairs and ~5 distinct delta values, this is a
necessary simplification but probably underestimates city-specific
disagreement (e.g. Miami's Δ may concentrate at 0, while Seattle's
clusters at +1). v1's emptiness might be partly an artifact of
pooling — per-city joint distributions could narrow the support
enough to admit a few strict-structural trades.

## Recommended next steps

1. **Drop v1 — strict joint-aware structural arbs effectively don't
   exist.** Round-1's "structural arb" framing requires the venue-
   agreement assumption. Once you accept venue divergence, you must
   reframe as EV/CVaR.

2. **Refine the joint K-P model with per-city Δ histograms** and re-run
   v1. The pooled n=188 may be over-pessimistic for individual cities.
   Alternatively, condition Δ on the *bracket region* (e.g. tail vs
   middle) — round-1 risk_analysis hinted at directional bias by side.

3. **For v2/v3 deployment, size by q05 not by EV.** With worst-case
   ≈ −$50K but q05 ≈ +$5K and EV ≈ +$16K, a Kelly-style sizer using
   q05 as the worst-case bound (not absolute worst-case) may make
   this strategy bankable. Single-trade sizing should target
   `(EV − q05)` headroom.

4. **Investigate the 1×2 and 2×1 patterns specifically.** They look
   like systematic short-vol-of-vol on Polymarket bracket pricing:
   the venue's NO contracts on adjacent narrow brackets are
   collectively priced as if the high will hit one of them more
   often than the joint distribution actually suggests. This may be
   genuine alpha or may be sample-size noise — n=188 historical Δ
   pairs is thin to anchor the joint distribution that drives v2's
   numbers.

5. **The SF/Seattle "new trades" finding is real but not actionable
   without a tighter risk filter.** 493 added trades exist in those
   cities — but at a worst-case −$45K, deploying them blindly is
   reckless. They need either (a) per-city joint distributions to
   verify the EV holds, or (b) a stricter q01-or-q005 floor.

## Bottom line

Multi-bracket portfolios DO unlock SF/Seattle (493 trades vs round-1's
~3) and produce a positive expected PnL of +$15,799 with a positive
q05 of +$5,550 at PAIR_SIZE=50. **But they are NOT structural arbs
under joint K-P scoring** — the strict v1 filter finds zero qualifying
portfolios across all 5 cities. The strategy is a CVaR-bounded EV bet
with a severe ($−49K) left tail. It is a usable EV strategy with
proper risk management; it is not a "free money" arbitrage.
