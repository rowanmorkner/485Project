"""
Honesty check: hedged-pair settlement risk analysis.

For an aligned-bracket pair where we LONG YES on cheap venue and LONG NO
on rich venue:

  cheap = Polymarket, rich = Kalshi (gap = K_mid - P_mid > 0)
  long leg pays $1 iff P_high ∈ bracket
  short leg pays $1 iff K_high ∉ bracket

  4 scenarios:
    K_high ∈ B, P_high ∈ B  -> $1 (both legs say "in"; YES wins, NO loses)
    K_high ∉ B, P_high ∉ B  -> $1 (both say "out"; NO wins, YES loses)
    K_high ∈ B, P_high ∉ B  -> $0 (rich-side was right we got 0 from both)
    K_high ∉ B, P_high ∈ B  -> $2 (cheap-side was right -- bonus)

So the failure mode is when the rich-side venue's reading lands in the
bracket but the cheap-side's doesn't -- exactly the reverse of what the
mid-prices implied.

Using the 188 historical pairs with both venues, we estimate per-bracket
disagreement rates conditional on bracket center.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.loaders import list_settled_pairs  # noqa: E402


def main():
  pairs = list_settled_pairs(both_venues=True)

  # Empirical |K-P| disagreement
  diffs = Counter([k - p for c, d, k, p in pairs])
  total = sum(diffs.values())
  print(f"Historical K-P signed diff distribution (n={total}):")
  for d in sorted(diffs):
    print(f"  K-P={d:+d}: {diffs[d]} ({100*diffs[d]/total:.1f}%)")
  print()
  print(f"Pr(K=P) = {diffs.get(0, 0) / total:.2%}  (we get $1 cleanly)")
  print(f"Pr(|K-P|>=1) = {1 - diffs.get(0, 0) / total:.2%}")
  print()

  # For each trade in v5, compute hedged-pair payoff under each historical
  # (K_high - P_high) diff scenario, anchoring "true high" to bracket interior.
  trades = json.load(open(Path(__file__).resolve().parent / "trades.json"))["v5"]

  size = 25
  fee = 0.005

  print("Per-trade hedged-pair settlement-risk simulation:")
  print(f"  (anchoring K_high to bracket center; varying P_high by historical diff)")
  print()
  for t in trades:
    degs = set(t["degrees"])
    d0, d1 = sorted(degs)
    parts = t["venue"].split("+")
    long_v = parts[0].replace("_long", "")
    short_v = parts[1].replace("_short", "")
    # Compute outcome for each (K, P) historical pair where we anchor
    # K_high inside the bracket (since we entered when the rich venue's
    # mid suggested K is in this bracket -- after all, the rich venue
    # priced it high).
    entry_cost = t["entry_price"] * size
    fees = 4 * fee * size

    # Scenario: assume K_high == d0 (interior of bracket).
    # For each historical pair's (K-P) diff, P_high = K_high - (K-P).
    payoff_buckets = Counter()
    pnls = []
    for c2, d2, k2, p2 in pairs:
      kp_diff = k2 - p2
      # Two anchorings of "true K_high" to the bracket: d0 and d1 (avg)
      for k_anchor in [d0, d1]:
        K = k_anchor
        P = K - kp_diff
        # Long leg ("yes_long") pays $1 if long_v's reading ∈ bracket
        long_reading = K if long_v == "kalshi" else P
        short_reading = K if short_v == "kalshi" else P
        long_pay = 1.0 if long_reading in degs else 0.0
        short_pay = 1.0 if short_reading not in degs else 0.0
        total_pay = (long_pay + short_pay) * size
        pnl = total_pay - entry_cost - fees
        pnls.append(pnl)
        payoff_buckets[round(long_pay + short_pay, 1)] += 1

    avg_pnl = sum(pnls) / len(pnls)
    win = sum(1 for p in pnls if p > 0) / len(pnls)
    bad = sum(1 for p in pnls if p < -10) / len(pnls)
    print(f"  {t['city']:14s} {long_v:11s} long, {short_v:11s} short bracket={list(degs)}")
    print(f"    entry_cost=${entry_cost:.2f}  fees=${fees:.2f}")
    print(f"    payoff dist: {dict(payoff_buckets)}")
    print(f"    avg p&l ${avg_pnl:+.2f}, win rate {win:.1%}, bad-loss rate (<-$10) {bad:.1%}")
    print()


if __name__ == "__main__":
  main()
