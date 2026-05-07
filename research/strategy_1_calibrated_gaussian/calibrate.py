"""
Per-city Gaussian calibration from historical settlements.

Original bot used a fixed sigma=2 F across all cities. We fit per-city
(sigma, bias) from the 400 historical settlements (Dec 2025 - May 2026).

Important caveat: The settlements table has actual realized highs but
no per-day NWS forecast (the forecasts table only covers 2026-05-05/06/07).
So we cannot compute true "forecast - actual" residuals.

Fallback we use: fit sigma as the residual std-dev of actual highs around
a centered 7-day rolling-mean per city. This reflects how much daily highs
vary around a smooth seasonal trend, which approximates the skill of a
"persistence + seasonal" forecast. It's a pessimistic upper bound on real
NWS sigma -- a real human/model forecast would do better than persistence.

Bias is set to 0 (no historical forecasts to estimate it from).

Output: data/forecast_calibration.json equivalent dict, but written into
strategy_1_calibrated_gaussian/calibration.json so we don't pollute the
project root data/. Per-city: {empirical_std_dev, bias, n_samples,
fallback_method}.
"""
from __future__ import annotations

import json
import sqlite3
import statistics
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "bot.db"
OUT_PATH = Path(__file__).resolve().parent / "calibration.json"

CITIES = ["Miami", "LA", "Austin", "San Francisco", "Seattle"]
ROLLING_WINDOW_DAYS = 7  # centered window for "naive seasonal" baseline


def load_city_highs(city: str) -> list[tuple[str, int]]:
  """Return [(date_iso, high_f)] for `city`, using kalshi_high_f if set
  else polymarket_high_f. Sorted by date ascending."""
  with sqlite3.connect(str(DB_PATH)) as c:
    c.row_factory = sqlite3.Row
    rows = c.execute(
      """SELECT date,
                COALESCE(kalshi_high_f, polymarket_high_f) AS h
           FROM settlements
          WHERE city=?
            AND (kalshi_high_f IS NOT NULL OR polymarket_high_f IS NOT NULL)
          ORDER BY date""",
      (city,),
    ).fetchall()
  return [(r["date"], int(r["h"])) for r in rows]


def centered_rolling_mean(highs: list[tuple[str, int]], window: int) -> list[tuple[str, int, float]]:
  """For each (date, high), compute a centered rolling mean over a +/-window/2
  day window of the *other* settlements. We index by date (not row position)
  so gaps don't bias the smoother. Returns [(date, high, baseline)]."""
  date_to_h = {d: h for d, h in highs}
  half = window // 2
  out = []
  for d, h in highs:
    target = date.fromisoformat(d)
    nearby = []
    for off in range(-half, half + 1):
      if off == 0:
        continue  # exclude self to keep the residual non-trivially defined
      probe = (target + timedelta(days=off)).isoformat()
      if probe in date_to_h:
        nearby.append(date_to_h[probe])
    if len(nearby) >= 2:
      baseline = statistics.fmean(nearby)
      out.append((d, h, baseline))
  return out


def fit_city(city: str) -> dict:
  highs = load_city_highs(city)
  if len(highs) < 10:
    return {
      "empirical_std_dev": 2.0,
      "bias": 0.0,
      "n_samples": len(highs),
      "fallback_method": "default_sigma_2",
      "raw_std_dev": None,
      "persistence_sigma": None,
      "warning": "fewer than 10 settlements; defaulting to sigma=2",
    }
  # Residual std around centered rolling mean (= persistence/seasonal baseline)
  triples = centered_rolling_mean(highs, ROLLING_WINDOW_DAYS)
  if len(triples) < 5:
    vals = [h for _, h in highs]
    return {
      "empirical_std_dev": statistics.pstdev(vals),
      "bias": 0.0,
      "n_samples": len(vals),
      "fallback_method": "raw_std_no_smoothing",
      "raw_std_dev": statistics.pstdev(vals),
      "persistence_sigma": None,
    }
  residuals = [h - baseline for _, h, baseline in triples]
  persistence_sigma = statistics.stdev(residuals) if len(residuals) > 1 else 2.0
  bias = statistics.fmean(residuals)

  # The persistence baseline measures day-to-day variability, NOT real NWS
  # day-of forecast skill. NWS day-of forecasts dramatically beat persistence.
  # Empirical rule of thumb: NWS day-1 MAE ~= 2-3 F for daily highs in CONUS,
  # vs persistence MAE ~= 4-6 F. So we shrink: nws_sigma ~= persistence * 0.5.
  # Floor at 1.5 (don't go tighter than what we'd believe a real NWS PDF is)
  # and cap at 4.0 (don't go absurdly wide if the city is volatile).
  shrunk = max(1.5, min(4.0, persistence_sigma * 0.5))

  return {
    "empirical_std_dev": round(shrunk, 4),
    "bias": round(bias, 4),
    "n_samples": len(residuals),
    "fallback_method": f"persistence_residual_x_0.5_window{ROLLING_WINDOW_DAYS}",
    "raw_std_dev": round(statistics.pstdev([h for _, h in highs]), 4),
    "persistence_sigma": round(persistence_sigma, 4),
    "median_abs_residual": round(statistics.median([abs(r) for r in residuals]), 4),
  }


def main() -> None:
  cal = {}
  for city in CITIES:
    cal[city] = fit_city(city)
    print(f"{city}: {cal[city]}")
  OUT_PATH.write_text(json.dumps(cal, indent=2))
  print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
  main()
