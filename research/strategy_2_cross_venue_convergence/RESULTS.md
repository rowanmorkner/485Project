# Strategy 2 — Cross-Venue Convergence / Relative-Value

## Strategy idea

Kalshi and Polymarket sometimes disagree on the implied probability of
overlapping `(city, date, bracket)` outcomes. The naive convergence
hypothesis is that, during the trading day, those gaps shrink as both
venues incorporate new information about the day's high temperature.
Buy the cheap side, plan to sell when the gap closes. No forecast model
is required — only relative pricing across venues.

This document presents 6 iterations: a directional baseline (v1, v2),
proper hedged relative-value pairs (v3), and several structural-arb
variants (v4–v6) with hold-to-end-of-window exits. We also include a
settlement-risk analysis (`risk_analysis.py`) that quantifies the
exposure to venue disagreement on the actual reported daily high.

## Key implementation choices and discards

1. **Bracket alignment.** Kalshi quotes 2-degree brackets (`84° to 85°`,
   `86° to 87°`...) plus tail brackets. Polymarket also uses 2-degree
   brackets (`84-85°F`...) plus tails. For Miami, LA, and Austin the
   two venues' grids ALIGN exactly. For San Francisco and Seattle the
   Kalshi grid is shifted by 1 degree relative to Polymarket
   (Kalshi `63-64`, Poly `62-63`), so no aligned bracket pairs exist.
   This is why all v3–v6 trades are concentrated in Miami/LA/Austin.

2. **Signal definition.**
   - **v1 (strict alignment)**: gap = K_mid − P_mid for identical-degree
     brackets. Trade triggers when |gap| ≥ 5¢. Buy YES on whichever venue
     has the lower mid. UNHEDGED single-leg directional bet.
   - **v2 (per-degree PDF)**: builds a per-degree implied probability
     (using mids spread uniformly across each bracket's degrees) for each
     venue, then for each bracket computes (rival_pdf summed over this
     bracket's degrees − own_mid). This handles partial overlaps
     (e.g. Kalshi `83° or below` vs Polymarket `82-83°F`). UNHEDGED.
   - **v3–v6 (hedged pair)**: long YES on the cheap side, long NO on the
     rich side (NO = 1 − YES_bid synthetic). On aligned 2-degree
     brackets this is a structural arb provided both venues settle on
     the same daily high.

3. **Exit rules tried.**
   - Mid-gap convergence (gap shrinks to half of entry): kept (v1, v2).
   - Proceeds-target (round-trip bid-side proceeds clear ≥2¢/pair
     post-fee): used in v3.
   - Hold to end-of-window (last available snapshot bid, with last-mid
     fallback if a leg's book disappeared): v4, v5.
   - Strict no-mid-fallback variant: v6.

4. **Cost cap.** For hedged pairs we require the at-entry sum of both
   leg ASKS to be ≤ $1.00 (v3, v4) or ≤ $0.97 (v5, v6). Without this
   cap many "trades" had per-pair entry > $1, meaning we paid more than
   the maximum possible payout — guaranteeing a loss.

5. **De-dup with cooldown.** Each `(city, cheap_v, cheap_id, rich_v,
   rich_id)` is opened once per backtest. We added a `cooldown_minutes`
   parameter to allow re-entry after gap reopens but it is unused here.

6. **Fees.** $0.005 per contract per side ($0.02 per hedged pair
   round-trip, applied 4× = entry + exit on each leg).

## Backtest results (2026-05-06, 5 cities × ~340 snapshots/venue, 25 contracts/leg)

| Version | n_trades | win_rate | total $ | avg $/trade | sharpe | exit reasons | comment |
|---:|---:|---:|---:|---:|---:|---|---|
| v1 | 9 | 11% | -10.67 | -1.19 | -1.10 | end:3, target:4, stop:2 | UNHEDGED directional. Convergence happens, but bidirectionally — both legs move toward each other, so a directional buy on the cheap side captures only half of the gap closure (or none, if the rich side does most of the moving). |
| v2 | 24 | 17% | -11.97 | -0.50 | -0.48 | end:4, target:8, stop:12 | UNHEDGED PDF gap — more signals, slightly less per-trade pain, but still loses. Same pathology as v1. |
| v3 | 9 | 33% | -11.99 | -1.33 | -0.63 | target:2, stop:7 | HEDGED with proceeds-target exit (2¢ profit). Almost no proceeds-target hits — bid-side proceeds rarely cover entry cost during the trading day. Most exits are time stops at unfavourable bid. |
| v4 | 9 | 89% | +20.33 | +2.26 | +1.16 | end-of-window:9 | HEDGED hold-to-EOW; mid-fallback ON. Most trades close at last snapshot's bids. Late-day spreads tighten. |
| v5 | 9 | 100% | +23.11 | +2.57 | +1.51 | end-of-window:9 | HEDGED hold-to-EOW with strict 0.97 cost cap. Each pair was entered at ≤ $0.97/contract and closed at ≥ $0.99 average. |
| v6 | 9 | 100% | +23.11 | +2.57 | +1.51 | end-of-window:9 | Identical to v5 — none of v5's trades depended on mid-fallback (every leg had observable bids at end-of-window). |

Notes on n=9:
- The 9 trades come from one open per (city, aligned-bracket pair) over
  Miami/LA/Austin only. There are ~1300 aligned-bracket-snapshot pairs
  per city, but our once-per-pair de-dup collapses each to a single
  entry at the first qualifying timestamp.
- SF/Seattle yield 0 trades because their Kalshi vs Polymarket grids
  differ by 1 degree (no bracket-set equality).
- This is a SMALL SAMPLE. Sharpe and win-rate numbers are not
  statistically robust at n=9.

## Failure modes observed

### 1. Directional convergence is not free money (v1, v2).
Even when a 5¢ gap "converges", BOTH legs typically move — the rich side
drops AND the cheap side drops, with the rich side moving more. Our
unhedged long on the cheap side then loses on price even as the gap
closes. v2's "target" exits (8 of 24) had lost money on average. This
falsifies the simple version of the convergence hypothesis: mid-gap
convergence and price improvement on the cheap side are *not* the same
event.

### 2. Spreads + fees frequently exceed the entry edge.
A 5¢ mid gap typically corresponds to 1-2¢ ask-spread on each leg, and
$0.02 round-trip fees per pair. So you need a 5¢ gap to close to ≤ 1¢
just to break even on a hedged proceeds-target exit. Empirically that
rarely happens *during* the trading window — late-day price collapse to
{0, 1} is what tightens proceeds, not mid-day convergence (v3 hit
proceeds-target only 2/9).

### 3. SETTLEMENT RISK is the real cost (`risk_analysis.py`).
This is the most important honesty check. The hedged structural arb
implicitly assumes both venues will settle on the same daily high. The
historical settlements table (188 pairs with both venues) shows:

| K_high − P_high | frequency |
|---:|---:|
| −1 | 9.6% |
| 0 | 25.5% |
| +1 | 37.2% |
| +2 | 22.9% |
| +3 | 4.3% |
| +4 | 0.5% |

**Only 25.5% of historical (city, date) pairs had Kalshi and Polymarket
report identical daily highs.** Kalshi systematically reads ~+1°F
higher than Polymarket. This breaks the "guaranteed $1 payout per
hedged pair" math: when Kalshi reports the high inside our bracket but
Polymarket doesn't (or vice versa), the pair pays $0, not $1.

The risk analysis simulates each v5 trade against the historical (K, P)
diff distribution and finds:
- Trades that LONG Kalshi (cheap) + SHORT Polymarket (rich) consistently
  win — mean ~+$15/trade. The directional bias of the historical
  disagreement (K > P) helps this side.
- Trades that LONG Polymarket (cheap) + SHORT Kalshi (rich) consistently
  LOSE — mean ~−$11/trade with a 51% chance of −$10+ loss. Because
  Kalshi reads higher, when we bet "Kalshi is overpriced and Poly is
  cheap" we are betting against the historical disagreement direction.
- 4/9 v5 trades are of the second (bad) type. Risk-adjusted, the
  ostensible $23 v5 profit may be roughly halved by settlement risk.

### 4. End-of-window p&l is mark-to-market on closing snapshots.
The 5/6 markets had not settled at the time of analysis. v5/v6's
positive p&l comes from the last observable bid being meaningfully
higher than entry asks because spreads tighten as the day progresses.
But the last observable bid is *not* the same thing as the actual
settlement payout. If we held to actual settlement on a venue-
disagreement day, the (Polymarket-long + Kalshi-short) trades on
brackets where K_high>bracket>P_high would pay $0.

### 5. Tiny sample size.
9 unique aligned-bracket pairs across 3 cities. SF/Seattle don't
contribute. Half the cities' data are unusable for this strategy.

## Recommended next steps

1. **Asymmetric strategy**: only take hedged trades that LONG the
   *historically lower-reading* venue (Polymarket) and SHORT the
   *historically higher-reading* venue (Kalshi) when the cheap side is
   Polymarket. In our notation: only take pairs where the cheap side
   is Polymarket. Skip pairs where the cheap side is Kalshi (and the
   short would be Polymarket → exposed to disagreement). This single
   filter cuts the 4 historically-bad v5 trades.

2. **Wait for settlement to score honestly**: 5/6 settlement lands later
   today. Re-score v4–v6 against actual settlements rather than
   last-observed mids. The structural risk simulation predicts ~50% of
   "polymarket-long, kalshi-short" trades will lose ~$10/contract pair.

3. **Bracket-pair construction for SF/Seattle**: the misaligned grids
   prevent direct alignment. With multi-bracket combinations on each
   side (e.g., Polymarket's 2 brackets `64-65 + 66-67` vs Kalshi's 2
   brackets `65-66 + 67-68`) covers degrees `64-68` overlapping
   `65-68`, you can construct synthetic equal-degree spreads. The
   payoff structure is more complex but the entry universe expands by
   ~70%.

4. **Use a directional pair-trade weighted by venue-disagreement bias**:
   instead of a structural arb, treat the K_mid − P_mid spread as
   informative, but apply the historical K-P>0 bias as a sign-correction.
   Concretely: if K_mid is 5¢ higher than P_mid AND Kalshi historically
   reads ~+1F higher, then the "true" probability of bracket B
   containing the high should be biased toward Kalshi's view rather
   than treated as an averaging exercise.

5. **Convergence at end-of-day has structure**: the only "convergence"
   that meaningfully closes the bid-ask gap is the day's final price
   collapse to {0, 1}. A simple "wait until 22:00 UTC, evaluate which
   bracket the consensus expects, and lift the cheap side ask vs the
   rich side bid" trade may dominate any continuous-time strategy.

## Bottom line

The simple cross-venue convergence hypothesis FAILS as a directional
trade (v1, v2 lose money — gap convergence is bidirectional, not
favouring the cheap side). The HEDGED structural-arb formulation
(v4–v6) shows positive p&l on the 2026-05-06 mark-to-market window
(with last-bid exits), but the historical settlement disagreement rate
of 74% is a serious risk that mark-to-market scoring hides. The
strategy is workable only with the asymmetric filter (long Polymarket,
short Kalshi when Polymarket is cheap) and with explicit treatment of
venue-disagreement risk. Sample size n=9 is too small to claim
production viability.
