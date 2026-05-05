"""
Probability distribution math: PDFs, CDFs, normalization, normal-from-forecast.

Pure functions over dict[int, float] distributions keyed by integer °F.
This module has no dependencies on other strategy modules.
"""

import json
import math
from pathlib import Path

# Optional per-city forecast calibration (written by bin/calibrate_forecast.py).
# When absent, forecast_to_distribution falls back to the caller-supplied
# std_dev (default 2.0°F).
_CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "data" / "forecast_calibration.json"
_cal_cache: dict[str, dict] = {}
_cal_mtime: float = 0.0


def _load_calibration() -> dict[str, dict]:
  """Mtime-aware read of forecast_calibration.json."""
  global _cal_cache, _cal_mtime
  if not _CALIBRATION_PATH.exists():
    return {}
  m = _CALIBRATION_PATH.stat().st_mtime
  if m == _cal_mtime and _cal_cache:
    return _cal_cache
  _cal_cache = json.loads(_CALIBRATION_PATH.read_text())
  _cal_mtime = m
  return _cal_cache


def _normal_cdf(x: float) -> float:
  """Standard normal CDF: Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))."""
  return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normalize(distribution: dict[int, float]) -> dict[int, float]:
  """
  Rescale a distribution so probabilities sum to 1.0.
  Returns a new dict; does not mutate the input.
  """
  total = sum(distribution.values())
  if total <= 0:
    return distribution
  return {deg: prob / total for deg, prob in distribution.items()}


def build_cdf(distribution: dict[int, float]) -> dict[int, float]:
  """
  Convert a per-degree PMF into a CDF: deg -> P(temp <= deg).
  """
  cdf: dict[int, float] = {}
  cumulative = 0.0
  for deg in sorted(distribution.keys()):
    cumulative += distribution[deg]
    cdf[deg] = cumulative
  return cdf


def forecast_to_distribution(
  forecast_high: float,
  std_dev: float = 2.0,
  city: str | None = None,
) -> dict[int, float]:
  """
  Convert a single point forecast (e.g. NWS forecast high) into a per-degree
  PMF by treating uncertainty as a normal distribution.

  Each integer degree gets the mass between (deg - 0.5) and (deg + 0.5)
  under a Gaussian centered on `forecast_high` with standard deviation `std_dev`.

  Args:
    forecast_high: Forecast high temperature in °F.
    std_dev: Forecast uncertainty in °F. Default 2.0°F is a reasonable
      24-48hr NWS proxy (~95% within ±4°F).
    city: Optional city name. If provided AND data/forecast_calibration.json
      has a "publish": true entry for this city, the empirical std_dev and
      bias from past observations override the caller's defaults. Lets the
      bot self-tune as it accumulates settled forecasts.
  """
  # Apply per-city calibration if available
  if city:
    cal = _load_calibration().get(city, {})
    if cal.get("publish"):
      std_dev = cal.get("empirical_std_dev", std_dev) or std_dev
      forecast_high = forecast_high + cal.get("bias", 0.0)

  distribution: dict[int, float] = {}
  # Cover ±4 sigma around the forecast
  lo = int(math.floor(forecast_high - 4 * std_dev))
  hi = int(math.ceil(forecast_high + 4 * std_dev))

  for deg in range(lo, hi + 1):
    p = _normal_cdf((deg + 0.5 - forecast_high) / std_dev) - \
        _normal_cdf((deg - 0.5 - forecast_high) / std_dev)
    if p > 1e-8:  # drop negligible tails
      distribution[deg] = p

  return normalize(distribution)


def cdf_at(cdf: dict[int, float], deg: int) -> float:
  """
  Look up CDF at a degree, with proper carry-forward: above the max key
  returns 1.0 (well, the max value), below the min key returns 0.0,
  in-between gaps fill from the closest lower key.
  """
  if deg in cdf:
    return cdf[deg]
  if not cdf:
    return 0.0
  max_deg = max(cdf.keys())
  min_deg = min(cdf.keys())
  if deg > max_deg:
    return cdf[max_deg]
  if deg < min_deg:
    return 0.0
  lower = max(k for k in cdf if k <= deg)
  return cdf[lower]
