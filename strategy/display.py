"""
Console pretty-printing for distribution comparisons.

Kept separate from arbitrage detection so the analytical layer never
depends on stdout side effects.
"""

from strategy.distributions import build_cdf, cdf_at


# Per-degree spread above which we mark a row as "diverging"
HIGHLIGHT_THRESHOLD = 0.05


def display_comparison(
  kalshi_dist: dict[int, float],
  poly_dist: dict[int, float],
  forecast_dist: dict[int, float],
):
  """
  Print side-by-side PMF and CDF tables for the three distributions.
  Rows where any pair of sources differs by >= HIGHLIGHT_THRESHOLD are flagged.
  """
  k_cdf = build_cdf(kalshi_dist)
  p_cdf = build_cdf(poly_dist)
  f_cdf = build_cdf(forecast_dist)

  # Union of all degrees present in any source
  all_degrees = sorted(
    set(kalshi_dist.keys()) | set(poly_dist.keys()) | set(forecast_dist.keys())
  )

  if not all_degrees:
    print("  No data to display.")
    return

  # ── PMF table ──
  print("\n" + "=" * 78)
  print("  PROBABILITY MASS FUNCTION (per-degree)")
  print("=" * 78)
  print(f"  {'Deg':>4}  {'Kalshi':>8}  {'Poly':>8}  {'NWS':>8}  {'Spread':>8}  {'Flag':>5}")
  print(f"  {'-' * 72}")

  for deg in all_degrees:
    k = kalshi_dist.get(deg, 0.0)
    p = poly_dist.get(deg, 0.0)
    f = forecast_dist.get(deg, 0.0)
    spread = max(k, p, f) - min(k, p, f)
    flag = " ***" if spread >= HIGHLIGHT_THRESHOLD else ""
    print(
      f"  {deg:>4}°F  {k:>8.4f}  {p:>8.4f}  {f:>8.4f}  {spread:>8.4f}  {flag}"
    )

  # ── CDF table ──
  print("\n" + "=" * 78)
  print("  CUMULATIVE DISTRIBUTION FUNCTION")
  print("=" * 78)
  print(f"  {'Deg':>4}  {'K CDF':>8}  {'P CDF':>8}  {'F CDF':>8}  {'Spread':>8}  {'Flag':>5}")
  print(f"  {'-' * 72}")

  for deg in all_degrees:
    k_c = cdf_at(k_cdf, deg)
    p_c = cdf_at(p_cdf, deg)
    f_c = cdf_at(f_cdf, deg)
    spread = max(k_c, p_c, f_c) - min(k_c, p_c, f_c)
    flag = " ***" if spread >= HIGHLIGHT_THRESHOLD else ""
    print(
      f"  {deg:>4}°F  {k_c:>8.4f}  {p_c:>8.4f}  {f_c:>8.4f}  {spread:>8.4f}  {flag}"
    )

  print()
