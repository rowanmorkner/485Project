"""
Console pretty-printing for HedgedPair selections.
"""

from contracts import HedgedPair


def display_hedged_pairs(pairs: list[HedgedPair]) -> None:
  """Print accepted hedged pairs in a flat tabular format."""
  if not pairs:
    print("  No hedged pairs found above threshold.")
    return

  print()
  print(f"  {'City':<14} {'Date':<11} {'K leg':<24} {'P leg':<22} "
        f"{'Cost':>7} {'E[pay]':>7} {'q05':>6}")
  print("  " + "-" * 99)
  for h in pairs:
    k_leg = f"{h.kalshi_side.upper()} {h.kalshi_label}"
    p_leg = f"{h.poly_side.upper()} {h.poly_label}"
    print(
      f"  {h.city:<14} {h.date:<11} {k_leg:<24.24} {p_leg:<22.22} "
      f"{h.cost_per_pair:>7.4f} {h.expected_payoff:>7.4f} {h.q05_payoff:>6.4f}"
    )
  print()
