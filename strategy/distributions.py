"""
Probability distribution math: PDFs and normal-from-forecast.

Pure functions over dict[int, float] distributions keyed by integer °F.
"""

import math


def _normal_cdf(x: float) -> float:
  """Standard normal CDF: Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))."""
  return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normalize(distribution: dict[int, float]) -> dict[int, float]:
  """Rescale so probabilities sum to 1.0; returns a new dict."""
  total = sum(distribution.values())
  if total <= 0:
    return distribution
  return {deg: prob / total for deg, prob in distribution.items()}


def forecast_to_distribution(
  forecast_high: float,
  std_dev: float = 2.0,
) -> dict[int, float]:
  """
  Convert a single point forecast (NWS forecast high) into a per-degree
  PMF by treating uncertainty as a normal distribution.

  Each integer degree gets the mass between (deg - 0.5) and (deg + 0.5)
  under N(forecast_high, std_dev). Default std_dev=2.0°F is a reasonable
  24-48hr NWS proxy (~95% within ±4°F).
  """
  distribution: dict[int, float] = {}
  lo = int(math.floor(forecast_high - 4 * std_dev))
  hi = int(math.ceil(forecast_high + 4 * std_dev))

  for deg in range(lo, hi + 1):
    p = _normal_cdf((deg + 0.5 - forecast_high) / std_dev) - \
        _normal_cdf((deg - 0.5 - forecast_high) / std_dev)
    if p > 1e-8:
      distribution[deg] = p

  return normalize(distribution)
