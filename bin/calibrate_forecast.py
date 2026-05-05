"""
Daily cron job: compare logged forecasts against the venue-reported actual
highs and write per-city empirical std_dev + bias to data/forecast_calibration.json.

This auto-tunes the forecast uncertainty currently hardcoded to 2.0°F in
strategy/distributions.py:forecast_to_distribution. Once a city has enough
settled forecasts, the strategy will use the calibrated value instead.

For each (city, date) where we logged BOTH a forecast and have a settlement:
  1. Compute the forecast's mean (E[temp] from PDF)
  2. Treat the venue's settled high as ground truth (we use Kalshi's reading
     when available, otherwise Polymarket's, since they agree most of the time)
  3. residual = actual - forecast_mean
  4. Aggregate per city: mean (bias) and stdev (calibration)
  5. Coverage check: did ±1σ and ±2σ from forecast cover the actual?

Run:
  python -m bin.calibrate_forecast
"""

import argparse
import json
import logging
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from persistence import db


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

OUTPUT_PATH = PROJECT_ROOT / "data" / "forecast_calibration.json"

# Below this sample size, we won't trust the empirical std_dev — better to
# fall back to the hardcoded default in forecast_to_distribution.
MIN_N_FOR_PUBLISH = 30


def pdf_mean(pdf: dict[int, float]) -> float | None:
  """E[X] over an integer-keyed PMF. Returns None on empty input."""
  total = sum(pdf.values())
  if total <= 0:
    return None
  return sum(int(d) * p for d, p in pdf.items()) / total


def actual_high(row) -> int | None:
  """Pick the best ground-truth high: Kalshi first (NWS CLI), else Polymarket."""
  if row["kalshi_high_f"] is not None:
    return int(row["kalshi_high_f"])
  if row["polymarket_high_f"] is not None:
    return int(row["polymarket_high_f"])
  return None


def compute_calibration() -> dict[str, dict]:
  """For each city, return calibration stats over all matched forecasts."""
  with db.connect() as conn:
    rows = conn.execute(
      """
      SELECT f.city AS city, f.date AS date, f.std_dev AS forecast_std_dev,
             f.pdf_json AS pdf_json,
             s.kalshi_high_f, s.polymarket_high_f
      FROM forecasts f
      JOIN settlements s ON s.city = f.city AND s.date = f.date
      """
    ).fetchall()

  log.info("Found %d (forecast, settlement) pairs", len(rows))

  per_city: dict[str, list[dict]] = defaultdict(list)
  for r in rows:
    actual = actual_high(r)
    if actual is None:
      continue
    try:
      pdf = {int(k): float(v) for k, v in json.loads(r["pdf_json"]).items()}
    except (TypeError, ValueError):
      continue
    mean = pdf_mean(pdf)
    if mean is None:
      continue
    per_city[r["city"]].append({
      "actual": actual,
      "forecast_mean": mean,
      "residual": actual - mean,
      "forecast_std_dev": r["forecast_std_dev"],
    })

  out: dict[str, dict] = {}
  for city, samples in per_city.items():
    n = len(samples)
    residuals = [s["residual"] for s in samples]
    bias = sum(residuals) / n
    if n > 1:
      var = sum((x - bias) ** 2 for x in residuals) / (n - 1)
      empirical_std = math.sqrt(var)
    else:
      empirical_std = 0.0
    # Coverage: did the forecast PDF (treating it as Gaussian around mean)
    # cover the actual within ±1σ and ±2σ?
    cov_1, cov_2 = 0, 0
    for s in samples:
      sd = s["forecast_std_dev"] or empirical_std or 1.0
      z = abs(s["residual"]) / sd if sd > 0 else 0.0
      if z <= 1.0:
        cov_1 += 1
      if z <= 2.0:
        cov_2 += 1
    out[city] = {
      "n": n,
      "bias": round(bias, 3),
      "empirical_std_dev": round(empirical_std, 3),
      "coverage_1sigma": round(cov_1 / n, 4),
      "coverage_2sigma": round(cov_2 / n, 4),
      "publish": n >= MIN_N_FOR_PUBLISH,
    }
  return out


def main():
  global MIN_N_FOR_PUBLISH
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--min-n", type=int, default=MIN_N_FOR_PUBLISH,
                      help="Minimum sample size before publishing the empirical std_dev.")
  args = parser.parse_args()
  MIN_N_FOR_PUBLISH = args.min_n

  cal = compute_calibration()
  if not cal:
    log.warning("No (forecast, settlement) pairs found — output not written.")
    return

  OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
  tmp = OUTPUT_PATH.with_suffix(".json.tmp")
  tmp.write_text(json.dumps(cal, indent=2))
  os.replace(tmp, OUTPUT_PATH)
  log.info("Wrote %s with %d cities", OUTPUT_PATH, len(cal))

  for city, s in cal.items():
    flag = "✓" if s["publish"] else "(too few)"
    log.info("  %s: n=%d bias=%+.2f σ=%.2f cov1σ=%.0f%% cov2σ=%.0f%%  %s",
             city, s["n"], s["bias"], s["empirical_std_dev"],
             100*s["coverage_1sigma"], 100*s["coverage_2sigma"], flag)


if __name__ == "__main__":
  main()
