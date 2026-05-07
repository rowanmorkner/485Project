# Round 2 brief — what round 1 taught us

You are a round-2 strategy agent. Read this BEFORE your strategy-specific
prompt to understand what's already been ruled out and what to build on.

Round 1 dispatched 5 strategies in parallel. Full results in
`research/COMPARISON.md` and per-strategy `RESULTS.md`. Round-2 priorities
are derived directly from round-1 findings.

## Surviving strategies (build on these)

- **`research/strategy_3_structural_arb/`** — POSITIVE. Strict structural
  arb (combined cost < $1, empty loss zone), hold-to-settle, 100%
  worst-case win rate. **15 trades, +$40 worst / +$140 expected**, but
  worst-case calculation assumed venues agree at settlement (they don't).
  Read `strategy.py` — the enumeration logic and ladder-walking are the
  reference implementation for cross-venue trade selection.

- **`research/strategy_2_cross_venue_convergence/`** — MIXED. Naive
  directional convergence (v1, v2) failed. Hedged-pair hold-to-settle
  (v5) showed +$23 at n=9 but exposed to the same venue-divergence risk.
  Read `risk_analysis.py` for the empirical (K−P) histogram and the
  asymmetric-filter recommendation.

## The single most important round-1 finding

**Of 188 historical (city, date) pairs with both venues' final highs,
only 25.5% report identical highs. Kalshi reads ~+1°F warmer than
Polymarket on average.**

```
K - P delta:  -1 → 9.6%
              +0 → 25.5%   ← only this case = "venues agree"
              +1 → 37.2%
              +2 → 22.9%
              +3 → 4.3%
              +4 → 0.5%
```

Per-city means (from `venue_divergence_histogram`):
- Miami +0.84°F, LA +1.06°F, Austin +1.00°F, SF +0.93°F, Seattle +0.76°F

**Implication**: any cross-venue trade that "guarantees" $1/pair at
settlement is implicitly assuming venues agree. They don't, 75% of the
time. A structural arb pair where one leg pays only when Kalshi reads
inside bracket B_K and the other pays only when Polymarket does NOT
read inside B_P can pay $0 if K_high lands in B_K but P_high does not
land where you expect.

## New loader primitives (round 2 — read-only)

`research/loaders.py` now provides:

```python
venue_divergence_histogram(city=None)
  # → {delta: prob}, e.g. {-1: 0.096, 0: 0.255, 1: 0.372, 2: 0.229, 3: 0.043, 4: 0.005}

joint_kp_distribution(forecast_pdf_kalshi, city=None)
  # → {(K_high, P_high): prob} approximating P(K, P) using:
  #   P(K=k) = forecast_pdf[k]    (NWS targets the same source as Kalshi)
  #   P(P=p|K=k) = empirical Δ histogram, treated as K-independent
  # Documented approximation; agents should acknowledge in writeups.

expected_payout_under_joint_kp(pair_payout_fn, joint_kp)
  # E[payout] over the joint distribution

quantile_payout_under_joint_kp(pair_payout_fn, joint_kp, q=0.05)
  # Lower q-tail payout. Use this for a CVaR-style risk filter:
  # only enter pairs where q05_payout > entry_cost.
```

`pair_payout_fn(k, p)` is YOUR function: given Kalshi reads `k` and
Polymarket reads `p`, what does YOUR pair pay (sum across legs)?

## Other round-1 lessons to remember

1. **Microstructure cost dominates.** Round-trip spread + fees ≈ $0.02/
   contract. Mean per-step Δmid is $0.005. Anything sub-30-min is dead
   on arrival unless you're MAKING the spread, not crossing it.

2. **Even omniscient mean-reversion exit is unprofitable.** S4 confirmed
   the perfect-foresight ceiling is negative. Don't pretend short-horizon
   directional moves are tradeable.

3. **Persistence dedup matters enormously.** Raw enumeration sees the
   same opportunity 100s of times. S3 went from 2,154 raw signals → 23
   distinct fillable. Always dedup before reporting trade counts.

4. **Ladder walking reveals truth.** Top-of-book quotes lie about depth.
   Use `walk_ladder_buy`/`walk_ladder_sell` for any size > 1.

5. **Cost > payout-floor = directional bet, not arb.** S3 v3 showed
   that loosening "cost < $1" reintroduces forecast risk and a $2,650
   worst-case loss. Stay disciplined.

6. **Hold-to-settle is the only strategy that worked.** MtM exit gives
   back the spread you crossed to enter. Anything depending on
   intraday convergence underperforms.

7. **SF/Seattle have 1°F-shifted bracket grids** between Kalshi and
   Polymarket — no aligned bracket pairs exist there. Strategies must
   either build synthetic alignments OR accept that they'll only work
   in Miami/LA/Austin.

## Required output (same as round 1)

Per `research/AGENT_CONTRACT.md`:
- `metrics.json` (standard schema)
- `RESULTS.md` (1–3 page narrative)
- code in your subdirectory
- ≥2 iterations

Constraints unchanged: read-only on `research/loaders.py`, `data/bot.db`,
and other strategies' folders. Use venv `/home/rowan/githuball/485Project/.venv/bin/python`.
