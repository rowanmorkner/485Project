"""
Arbitrage Analysis Module — Cross-platform temperature distribution comparison.

Compares implied probability distributions from Kalshi, Polymarket, and NWS
weather forecasts to find pricing discrepancies on Miami daily high temperature
markets.

The core challenge: Kalshi and Polymarket use DIFFERENT temperature bins that
don't align (offset by 1 degree). We reconstruct a common per-degree probability
distribution from each platform's bin prices using a piecewise-uniform assumption,
then compare them to identify arbitrage opportunities.

Kalshi bins (typical):  <=76, 77-78, 79-80, 81-82, 83-84, >=85
Polymarket bins (typical): <=73, 74-75, 76-77, 78-79, 80-81, 82-83,
                           84-85, 86-87, 88-89, 90-91, >=92
"""

import re
import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# How many degrees beyond the boundary to spread tail-bin probability
TAIL_SPREAD = 5


# ── Kalshi bin parsing ────────────────────────────────────────────────────

def parse_kalshi_bins(brackets: list[dict]) -> dict[int, float]:
  """
  Convert Kalshi bracket market data into a per-degree probability distribution.

  Each bracket has a 'subtitle' like "77° to 78°", "76° or below", or
  "85° or above", plus 'best_yes_bid' and 'best_yes_ask' pricing fields.

  We compute a midpoint implied probability from bid/ask, then spread it
  uniformly across the integer degrees inside each bin (piecewise-uniform
  assumption).

  Args:
    brackets: List of Kalshi bracket dicts. Expected keys:
      - subtitle: str — temperature range description
      - best_yes_bid: float | None — best bid price for yes
      - best_yes_ask: float | None — best ask price for yes

  Returns:
    Dict mapping integer degree (°F) to implied probability.
  """
  distribution: dict[int, float] = {}

  for bracket in brackets:
    subtitle = bracket.get("subtitle", "")
    bid = bracket.get("best_yes_bid")
    ask = bracket.get("best_yes_ask")

    # Compute midpoint price as the implied probability
    implied_prob = _midpoint_price(bid, ask)
    if implied_prob is None or implied_prob <= 0:
      continue

    # Parse the temperature range from the subtitle
    degrees = _parse_kalshi_range(subtitle)
    if not degrees:
      logger.warning("Could not parse Kalshi subtitle: '%s'", subtitle)
      continue

    # Spread probability uniformly across the degrees in this bin
    prob_per_degree = implied_prob / len(degrees)
    for deg in degrees:
      distribution[deg] = distribution.get(deg, 0.0) + prob_per_degree

  # Normalize so probabilities sum to 1.0
  return _normalize(distribution)


def _parse_kalshi_range(subtitle: str) -> list[int]:
  """
  Extract the list of integer degrees covered by a Kalshi bracket subtitle.

  Handles formats like:
    "77° to 78°"    -> [77, 78]
    "76° or below"  -> [71, 72, 73, 74, 75, 76]  (boundary - TAIL_SPREAD .. boundary)
    "85° or above"  -> [85, 86, 87, 88, 89, 90]  (boundary .. boundary + TAIL_SPREAD)

  Returns:
    List of integer degree values, or empty list on parse failure.
  """
  # Pattern: "X° to Y°" or "X° - Y°"
  range_match = re.search(r"(\d+)\s*°?\s*(?:to|-)\s*(\d+)\s*°?", subtitle)
  if range_match:
    lo = int(range_match.group(1))
    hi = int(range_match.group(2))
    return list(range(lo, hi + 1))

  # Pattern: "X° or below" / "X° or less" / "Under X°" / "≤X°"
  below_match = re.search(r"(\d+)\s*°?\s*(?:or\s+(?:below|less|lower)|and\s+below)", subtitle, re.IGNORECASE)
  if not below_match:
    below_match = re.search(r"(?:under|below|≤|<=)\s*(\d+)\s*°?", subtitle, re.IGNORECASE)
  if below_match:
    boundary = int(below_match.group(1))
    return list(range(boundary - TAIL_SPREAD, boundary + 1))

  # Pattern: "X° or above" / "X° or more" / "Over X°" / "≥X°"
  above_match = re.search(r"(\d+)\s*°?\s*(?:or\s+(?:above|more|higher)|and\s+above)", subtitle, re.IGNORECASE)
  if not above_match:
    above_match = re.search(r"(?:over|above|≥|>=)\s*(\d+)\s*°?", subtitle, re.IGNORECASE)
  if above_match:
    boundary = int(above_match.group(1))
    return list(range(boundary, boundary + TAIL_SPREAD + 1))

  return []


# ── Polymarket bin parsing ────────────────────────────────────────────────

def parse_polymarket_bins(brackets: list[dict]) -> dict[int, float]:
  """
  Convert Polymarket bracket market data into a per-degree probability distribution.

  Each bracket has a 'question' like "78-79°F" and a 'yes_price' that serves
  as the implied probability for that bin.

  Args:
    brackets: List of Polymarket bracket dicts. Expected keys:
      - question: str — temperature range description (e.g. "78-79°F")
      - yes_price: float | None — implied probability

  Returns:
    Dict mapping integer degree (°F) to implied probability.
  """
  distribution: dict[int, float] = {}

  for bracket in brackets:
    question = bracket.get("question", "")
    yes_price = bracket.get("yes_price")

    if yes_price is None or yes_price <= 0:
      continue

    # Parse the temperature range from the question string
    degrees = _parse_polymarket_range(question)
    if not degrees:
      logger.warning("Could not parse Polymarket question: '%s'", question)
      continue

    # Spread probability uniformly across the degrees in this bin
    prob_per_degree = yes_price / len(degrees)
    for deg in degrees:
      distribution[deg] = distribution.get(deg, 0.0) + prob_per_degree

  # Normalize so probabilities sum to 1.0
  return _normalize(distribution)


def _parse_polymarket_range(question: str) -> list[int]:
  """
  Extract the list of integer degrees covered by a Polymarket bracket question.

  Handles formats like:
    "78-79°F"               -> [78, 79]
    "73°F or less"          -> [68, 69, 70, 71, 72, 73]
    "92°F or more"          -> [92, 93, 94, 95, 96, 97]
    "Less than 74°F"        -> [68, 69, 70, 71, 72, 73]
    "Greater than 91°F"     -> [92, 93, 94, 95, 96, 97]

  Returns:
    List of integer degree values, or empty list on parse failure.
  """
  # Strip trailing °F / F for cleaner matching
  cleaned = question.strip()

  # Pattern: "X-Y°F" or "X-Y" (range bins)
  range_match = re.search(r"(\d+)\s*-\s*(\d+)", cleaned)
  if range_match:
    lo = int(range_match.group(1))
    hi = int(range_match.group(2))
    return list(range(lo, hi + 1))

  # Pattern: "X°F or less" / "X or less" / "Less than X°F" / "Under X"
  below_match = re.search(
    r"(\d+)\s*°?\s*F?\s*(?:or\s+(?:less|below|lower))",
    cleaned, re.IGNORECASE
  )
  if not below_match:
    below_match = re.search(
      r"(?:less\s+than|under|below|≤|<=)\s*(\d+)\s*°?\s*F?",
      cleaned, re.IGNORECASE
    )
  if below_match:
    boundary = int(below_match.group(1))
    return list(range(boundary - TAIL_SPREAD, boundary + 1))

  # Pattern: "X°F or more" / "Greater than X°F" / "Over X"
  above_match = re.search(
    r"(\d+)\s*°?\s*F?\s*(?:or\s+(?:more|above|higher|greater))",
    cleaned, re.IGNORECASE
  )
  if not above_match:
    above_match = re.search(
      r"(?:more\s+than|greater\s+than|over|above|≥|>=)\s*(\d+)\s*°?\s*F?",
      cleaned, re.IGNORECASE
    )
  if above_match:
    boundary = int(above_match.group(1))
    return list(range(boundary, boundary + TAIL_SPREAD + 1))

  return []


# ── NWS forecast → distribution ──────────────────────────────────────────

def forecast_to_distribution(
  forecast_high: float,
  std_dev: float = 2.0,
) -> dict[int, float]:
  """
  Convert an NWS point forecast into a per-degree probability distribution.

  Models the forecast uncertainty as a normal (Gaussian) distribution centered
  on the forecast high, where each integer degree gets the probability mass
  between degree - 0.5 and degree + 0.5.

  Args:
    forecast_high: NWS forecast high temperature in °F.
    std_dev: Standard deviation of forecast uncertainty in °F.
             Default 2.0°F, which covers ±4° at ~95% confidence — a
             reasonable approximation for 24-48hr NWS forecasts.

  Returns:
    Dict mapping integer degree (°F) to probability, normalized to sum to 1.0.
  """
  distribution: dict[int, float] = {}

  # Cover a generous range around the forecast (±4 sigma)
  lo = int(math.floor(forecast_high - 4 * std_dev))
  hi = int(math.ceil(forecast_high + 4 * std_dev))

  for deg in range(lo, hi + 1):
    # Probability that temp falls in [deg - 0.5, deg + 0.5)
    # using the normal CDF: P = Phi((deg+0.5 - mu)/sigma) - Phi((deg-0.5 - mu)/sigma)
    p = _normal_cdf((deg + 0.5 - forecast_high) / std_dev) - \
        _normal_cdf((deg - 0.5 - forecast_high) / std_dev)
    if p > 1e-8:  # skip negligible tails
      distribution[deg] = p

  return _normalize(distribution)


def _normal_cdf(x: float) -> float:
  """
  Standard normal CDF using the math.erf function.
  Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
  """
  return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ── CDF builder ──────────────────────────────────────────────────────────

def build_cdf(distribution: dict[int, float]) -> dict[int, float]:
  """
  Convert a per-degree probability distribution (PMF) into a CDF.

  Args:
    distribution: Dict mapping integer degree → probability.

  Returns:
    Dict mapping integer degree → P(temp <= degree), i.e. cumulative
    probability up to and including that degree.
  """
  cdf: dict[int, float] = {}
  cumulative = 0.0

  for deg in sorted(distribution.keys()):
    cumulative += distribution[deg]
    cdf[deg] = cumulative

  return cdf


# ── Discrepancy finder ───────────────────────────────────────────────────

def find_discrepancies(
  kalshi_dist: dict[int, float],
  poly_dist: dict[int, float],
  forecast_dist: dict[int, float],
  min_edge: float = 0.05,
) -> list[dict]:
  """
  Compare all three distributions at each integer degree and flag disagreements.

  For every degree present in at least one distribution, reports the implied
  probability from each source and computes the maximum spread (difference
  between the highest and lowest probability among the three).

  Args:
    kalshi_dist: Per-degree distribution from Kalshi bins.
    poly_dist: Per-degree distribution from Polymarket bins.
    forecast_dist: Per-degree distribution from NWS forecast.
    min_edge: Minimum probability spread to flag as a discrepancy.

  Returns:
    List of dicts, one per flagged degree, sorted by spread (largest first):
      {
        "degree": int,
        "kalshi_prob": float,
        "poly_prob": float,
        "forecast_prob": float,
        "max_spread": float,
        "sources_disagreeing": list[str],
      }
  """
  # Union of all degrees present in any distribution
  all_degrees = sorted(
    set(kalshi_dist.keys()) | set(poly_dist.keys()) | set(forecast_dist.keys())
  )

  discrepancies = []
  for deg in all_degrees:
    k_prob = kalshi_dist.get(deg, 0.0)
    p_prob = poly_dist.get(deg, 0.0)
    f_prob = forecast_dist.get(deg, 0.0)

    probs = [k_prob, p_prob, f_prob]
    max_spread = max(probs) - min(probs)

    if max_spread >= min_edge:
      # Identify which pairs disagree
      sources = []
      if abs(k_prob - p_prob) >= min_edge:
        sources.append("kalshi-vs-poly")
      if abs(k_prob - f_prob) >= min_edge:
        sources.append("kalshi-vs-forecast")
      if abs(p_prob - f_prob) >= min_edge:
        sources.append("poly-vs-forecast")

      discrepancies.append({
        "degree": deg,
        "kalshi_prob": round(k_prob, 4),
        "poly_prob": round(p_prob, 4),
        "forecast_prob": round(f_prob, 4),
        "max_spread": round(max_spread, 4),
        "sources_disagreeing": sources,
      })

  # Sort by largest spread first (most interesting opportunities)
  discrepancies.sort(key=lambda d: d["max_spread"], reverse=True)
  return discrepancies


# ── Arbitrage opportunity finder ─────────────────────────────────────────

def find_arbitrage_opportunities(
  kalshi_dist: dict[int, float],
  poly_dist: dict[int, float],
  forecast_dist: dict[int, float],
  kalshi_fee: float = 0.0,
  poly_fee: float = 0.0,
  min_edge: float = 0.03,
  same_station: bool = True,
) -> list[dict]:
  """
  Find actionable arbitrage opportunities across Kalshi and Polymarket.

  For each degree where both platforms assign probability, compares the
  implied prices. If one platform's implied probability is significantly
  cheaper than the other (or than the NWS forecast "fair value"), there's
  an opportunity to buy cheap on one side and sell expensive on the other.

  When the platforms resolve against different weather stations (same_station=False),
  cross-platform trades carry "basis risk" — the stations may record different
  temperatures. These are flagged separately from clean same-station arbs.

  Args:
    kalshi_dist: Per-degree distribution from Kalshi.
    poly_dist: Per-degree distribution from Polymarket.
    forecast_dist: Per-degree distribution from NWS forecast (used as fair value).
    kalshi_fee: Kalshi transaction fee as a fraction (e.g., 0.07 for 7%).
    poly_fee: Polymarket transaction fee as a fraction (e.g., 0.02 for 2%).
    min_edge: Minimum expected edge after fees to flag as an opportunity.
    same_station: Whether both platforms resolve against the same weather station.
      If False, cross-platform trades are flagged as "basis risk" and require
      a higher edge threshold to compensate for station divergence.

  Returns:
    List of opportunity dicts, sorted by edge (largest first).
  """
  # When stations differ, cross-platform arbs need a larger edge to
  # compensate for the basis risk (stations can differ by 1-3°F)
  BASIS_RISK_PENALTY = 0.05

  # Degrees where both platforms have pricing
  shared_degrees = sorted(
    set(kalshi_dist.keys()) & set(poly_dist.keys())
  )

  opportunities = []
  for deg in shared_degrees:
    k_price = kalshi_dist[deg]
    p_price = poly_dist[deg]
    f_fair = forecast_dist.get(deg, 0.0)

    # Cross-platform arbitrage: buy cheap, sell expensive
    gross_edge = abs(k_price - p_price)

    # Determine direction
    if k_price < p_price:
      buy_platform, sell_platform = "kalshi", "polymarket"
      buy_price, sell_price = k_price, p_price
      buy_fee, sell_fee = kalshi_fee, poly_fee
    else:
      buy_platform, sell_platform = "polymarket", "kalshi"
      buy_price, sell_price = p_price, k_price
      buy_fee, sell_fee = poly_fee, kalshi_fee

    # Net edge after fees on both sides
    cost_with_fee = buy_price * (1.0 + buy_fee)
    proceeds_after_fee = sell_price * (1.0 - sell_fee)
    net_edge = proceeds_after_fee - cost_with_fee

    # Apply basis risk penalty for different-station cross-platform trades
    effective_min_edge = min_edge
    if not same_station:
      effective_min_edge = min_edge + BASIS_RISK_PENALTY

    if net_edge >= effective_min_edge:
      # Classify the trade type
      if same_station:
        trade_type = "CLEAN ARB"
        risk_note = "Same station — true arbitrage"
      else:
        trade_type = "BASIS RISK"
        risk_note = "Different stations — edge may be absorbed by station divergence"

      opportunities.append({
        "degree": deg,
        "action": f"Buy {deg}°F on {buy_platform}, sell on {sell_platform}",
        "buy_platform": buy_platform,
        "sell_platform": sell_platform,
        "buy_price": round(buy_price, 4),
        "sell_price": round(sell_price, 4),
        "forecast_fair": round(f_fair, 4),
        "gross_edge": round(gross_edge, 4),
        "net_edge": round(net_edge, 4),
        "trade_type": trade_type,
        "risk_note": risk_note,
      })

    # Forecast-based value trades: market mispriced vs NWS fair value
    if f_fair > 0:
      for platform, price, fee in [
        ("kalshi", k_price, kalshi_fee),
        ("polymarket", p_price, poly_fee),
      ]:
        # Underpriced vs forecast: buy opportunity
        if f_fair > price:
          value_edge = (f_fair - price) - price * fee
          if value_edge >= min_edge:
            opportunities.append({
              "degree": deg,
              "action": f"Buy {deg}°F on {platform} (underpriced vs forecast)",
              "buy_platform": platform,
              "sell_platform": "forecast-implied",
              "buy_price": round(price, 4),
              "sell_price": round(f_fair, 4),
              "forecast_fair": round(f_fair, 4),
              "gross_edge": round(f_fair - price, 4),
              "net_edge": round(value_edge, 4),
              "trade_type": "FORECAST",
              "risk_note": "Edge depends on NWS forecast accuracy",
            })

  # Sort by net edge, largest first
  opportunities.sort(key=lambda o: o["net_edge"], reverse=True)
  return opportunities


# ── Display / comparison table ───────────────────────────────────────────

def display_comparison(
  kalshi_dist: dict[int, float],
  poly_dist: dict[int, float],
  forecast_dist: dict[int, float],
):
  """
  Pretty-print a side-by-side comparison of all three distributions.

  Shows both the per-degree PMF and the cumulative CDF for each source,
  with visual markers (***) on rows where distributions diverge significantly.

  Args:
    kalshi_dist: Per-degree distribution from Kalshi.
    poly_dist: Per-degree distribution from Polymarket.
    forecast_dist: Per-degree distribution from NWS forecast.
  """
  # Build CDFs for all three
  k_cdf = build_cdf(kalshi_dist)
  p_cdf = build_cdf(poly_dist)
  f_cdf = build_cdf(forecast_dist)

  # Union of all degrees
  all_degrees = sorted(
    set(kalshi_dist.keys()) | set(poly_dist.keys()) | set(forecast_dist.keys())
  )

  if not all_degrees:
    print("  No data to display.")
    return

  # Divergence threshold for highlighting
  highlight_threshold = 0.05

  # ── PMF table ──
  print("\n" + "=" * 78)
  print("  PROBABILITY MASS FUNCTION (per-degree)")
  print("=" * 78)
  header = f"  {'Deg':>4}  {'Kalshi':>8}  {'Poly':>8}  {'NWS':>8}  {'Spread':>8}  {'Flag':>5}"
  print(header)
  print(f"  {'-' * 72}")

  for deg in all_degrees:
    k = kalshi_dist.get(deg, 0.0)
    p = poly_dist.get(deg, 0.0)
    f = forecast_dist.get(deg, 0.0)
    spread = max(k, p, f) - min(k, p, f)
    flag = " ***" if spread >= highlight_threshold else ""
    print(
      f"  {deg:>4}°F"
      f"  {k:>8.4f}"
      f"  {p:>8.4f}"
      f"  {f:>8.4f}"
      f"  {spread:>8.4f}"
      f"  {flag}"
    )

  # ── CDF table ──
  print("\n" + "=" * 78)
  print("  CUMULATIVE DISTRIBUTION FUNCTION")
  print("=" * 78)
  header = f"  {'Deg':>4}  {'K CDF':>8}  {'P CDF':>8}  {'F CDF':>8}  {'Spread':>8}  {'Flag':>5}"
  print(header)
  print(f"  {'-' * 72}")

  # Helper to look up CDF with proper carry-forward: if a degree is above
  # a distribution's max, the CDF should be 1.0 (not 0.0)
  def _cdf_at(cdf: dict[int, float], deg: int) -> float:
    if deg in cdf:
      return cdf[deg]
    if not cdf:
      return 0.0
    max_deg = max(cdf.keys())
    min_deg = min(cdf.keys())
    if deg > max_deg:
      return cdf[max_deg]  # carry forward (should be ~1.0)
    if deg < min_deg:
      return 0.0
    # Between entries — find the closest lower key
    lower = max(k for k in cdf if k <= deg)
    return cdf[lower]

  for deg in all_degrees:
    k_c = _cdf_at(k_cdf, deg)
    p_c = _cdf_at(p_cdf, deg)
    f_c = _cdf_at(f_cdf, deg)
    spread = max(k_c, p_c, f_c) - min(k_c, p_c, f_c)
    flag = " ***" if spread >= highlight_threshold else ""
    print(
      f"  {deg:>4}°F"
      f"  {k_c:>8.4f}"
      f"  {p_c:>8.4f}"
      f"  {f_c:>8.4f}"
      f"  {spread:>8.4f}"
      f"  {flag}"
    )

  print()


# ── Utility helpers ──────────────────────────────────────────────────────

def _midpoint_price(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
  """
  Compute the midpoint implied probability from bid and ask prices.
  Falls back to whichever side is available if only one is present.
  """
  if bid is not None and ask is not None:
    return (bid + ask) / 2.0
  if bid is not None:
    return bid
  if ask is not None:
    return ask
  return None


def _normalize(distribution: dict[int, float]) -> dict[int, float]:
  """
  Normalize a distribution so all probabilities sum to 1.0.
  Returns a new dict (does not mutate the input).
  """
  total = sum(distribution.values())
  if total <= 0:
    return distribution
  return {deg: prob / total for deg, prob in distribution.items()}


# ── Main: demo with example data ─────────────────────────────────────────

if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)

  # Example Kalshi brackets (simulated from real Miami high temp event)
  # Bins: <=76, 77-78, 79-80, 81-82, 83-84, >=85
  example_kalshi_brackets = [
    {"ticker": "KXHIGHMIA-26APR02-T76", "subtitle": "76° or below",
     "best_yes_bid": 0.05, "best_yes_ask": 0.07},
    {"ticker": "KXHIGHMIA-26APR02-T77", "subtitle": "77° to 78°",
     "best_yes_bid": 0.10, "best_yes_ask": 0.13},
    {"ticker": "KXHIGHMIA-26APR02-T79", "subtitle": "79° to 80°",
     "best_yes_bid": 0.22, "best_yes_ask": 0.25},
    {"ticker": "KXHIGHMIA-26APR02-T81", "subtitle": "81° to 82°",
     "best_yes_bid": 0.30, "best_yes_ask": 0.33},
    {"ticker": "KXHIGHMIA-26APR02-T83", "subtitle": "83° to 84°",
     "best_yes_bid": 0.15, "best_yes_ask": 0.18},
    {"ticker": "KXHIGHMIA-26APR02-T85", "subtitle": "85° or above",
     "best_yes_bid": 0.08, "best_yes_ask": 0.10},
  ]

  # Example Polymarket brackets (simulated from real Miami high temp event)
  # Bins: <=73, 74-75, 76-77, 78-79, 80-81, 82-83, 84-85, 86-87, 88-89, 90-91, >=92
  example_poly_brackets = [
    {"question": "73°F or less", "condition_id": "0xaaa1", "yes_price": 0.02},
    {"question": "74-75°F",      "condition_id": "0xaaa2", "yes_price": 0.04},
    {"question": "76-77°F",      "condition_id": "0xaaa3", "yes_price": 0.10},
    {"question": "78-79°F",      "condition_id": "0xaaa4", "yes_price": 0.18},
    {"question": "80-81°F",      "condition_id": "0xaaa5", "yes_price": 0.28},
    {"question": "82-83°F",      "condition_id": "0xaaa6", "yes_price": 0.22},
    {"question": "84-85°F",      "condition_id": "0xaaa7", "yes_price": 0.10},
    {"question": "86-87°F",      "condition_id": "0xaaa8", "yes_price": 0.04},
    {"question": "88-89°F",      "condition_id": "0xaaa9", "yes_price": 0.01},
    {"question": "90-91°F",      "condition_id": "0xaaaa", "yes_price": 0.005},
    {"question": "92°F or more", "condition_id": "0xaaab", "yes_price": 0.005},
  ]

  # NWS forecast: high of 81°F with 2°F uncertainty
  nws_forecast_high = 81.0
  nws_std_dev = 2.0

  # ── Build distributions ──
  print("\n" + "=" * 78)
  print("  WEATHER ARBITRAGE ANALYSIS — Miami Daily High Temperature")
  print("=" * 78)

  kalshi_dist = parse_kalshi_bins(example_kalshi_brackets)
  poly_dist = parse_polymarket_bins(example_poly_brackets)
  forecast_dist = forecast_to_distribution(nws_forecast_high, nws_std_dev)

  # ── Display side-by-side comparison ──
  display_comparison(kalshi_dist, poly_dist, forecast_dist)

  # ── Find discrepancies ──
  print("=" * 78)
  print("  DISCREPANCIES (min_edge = 0.05)")
  print("=" * 78)

  discrepancies = find_discrepancies(kalshi_dist, poly_dist, forecast_dist, min_edge=0.05)
  if discrepancies:
    for d in discrepancies:
      print(
        f"  {d['degree']:>3}°F  |  "
        f"K={d['kalshi_prob']:.4f}  P={d['poly_prob']:.4f}  "
        f"F={d['forecast_prob']:.4f}  |  "
        f"spread={d['max_spread']:.4f}  "
        f"({', '.join(d['sources_disagreeing'])})"
      )
  else:
    print("  No discrepancies found above threshold.")

  # ── Find arbitrage opportunities ──
  print("\n" + "=" * 78)
  print("  ARBITRAGE OPPORTUNITIES (min_edge = 0.03, fees = 0%)")
  print("=" * 78)

  opportunities = find_arbitrage_opportunities(
    kalshi_dist, poly_dist, forecast_dist,
    kalshi_fee=0.0, poly_fee=0.0, min_edge=0.03,
  )
  if opportunities:
    for opp in opportunities:
      print(
        f"  {opp['action']}\n"
        f"    Buy @ {opp['buy_price']:.4f} on {opp['buy_platform']}, "
        f"sell @ {opp['sell_price']:.4f} on {opp['sell_platform']}\n"
        f"    Forecast fair value: {opp['forecast_fair']:.4f}, "
        f"net edge: {opp['net_edge']:.4f}\n"
      )
  else:
    print("  No arbitrage opportunities found above threshold.")

  print("=" * 78 + "\n")
