"""
Daily cron job: rebuild the per-city Δ histogram from real Polymarket-vs-
Kalshi settlements stored in the database, and atomically replace
data/risk_histograms.json. The strategy code (strategy/risk.py) picks
this up on its next mtime check — no bot restart needed.

Replaces the static cli_vs_asos_divergence.json (built from an ASOS proxy)
with empirical signal from the actual venues. The longer the bot runs,
the more accurate this gets.

Run:
  python -m bin.refresh_risk_model [--decay 0.99] [--lookback-days 365]
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from persistence import db


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

OUTPUT_PATH = PROJECT_ROOT / "data" / "risk_histograms.json"


def build_histogram(decay: float, lookback_days: int) -> dict[str, dict]:
  """
  Read settlements with both venues populated; return per-city histograms
  in the same shape as backtests/cli_vs_asos_divergence.json.

  Δ = kalshi_high_f - polymarket_high_f (positive = Kalshi runs warmer).

  decay: per-day exponential weight. 1.0 = no decay (all observations
    weighted equally). 0.99 = today's observation has weight 1, an
    observation from 100 days ago has weight 0.99^100 ≈ 0.37.
  """
  cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()

  with db.connect() as conn:
    rows = conn.execute(
      """
      SELECT city, date, kalshi_high_f, polymarket_high_f
      FROM settlements
      WHERE kalshi_high_f IS NOT NULL
        AND polymarket_high_f IS NOT NULL
        AND date >= ?
      """,
      (cutoff,),
    ).fetchall()

  log.info("Found %d settlement pairs with both venues populated (since %s)",
           len(rows), cutoff)

  # Per-city weighted histogram of Δ
  per_city_hist: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
  per_city_n: dict[str, int] = defaultdict(int)
  today = date.today()

  for r in rows:
    # Each "*_high_f" stored value is the LOW edge of the 2°F-wide winning
    # bracket (see bin/log_settlements.reading_or_none for why). Compute Δ
    # as the distance between the two bracket RANGES — ranges that overlap
    # contribute Δ=0 (consistent readings, just different bracket alignment).
    # Only disjoint ranges contribute non-zero Δ (real basis risk).
    k_lo = int(r["kalshi_high_f"]);     k_hi = k_lo + 1
    p_lo = int(r["polymarket_high_f"]); p_hi = p_lo + 1
    if k_lo <= p_hi and p_lo <= k_hi:
      delta = 0  # ranges overlap
    elif k_lo > p_hi:
      delta = k_lo - p_hi  # Kalshi reads strictly higher
    else:
      delta = -(p_lo - k_hi)  # Polymarket reads strictly higher

    try:
      d = datetime.strptime(r["date"], "%Y-%m-%d").date()
    except ValueError:
      continue
    age = max((today - d).days, 0)
    weight = decay ** age if decay < 1.0 else 1.0
    per_city_hist[r["city"]][delta] += weight
    per_city_n[r["city"]] += 1

  # Convert to the same shape as cli_vs_asos_divergence.json so strategy/risk.py
  # can read it without changes. Only the "histogram" field is consumed today;
  # the tail-probability fields are populated for compatibility with reports.
  out: dict[str, dict] = {}
  for city, hist in per_city_hist.items():
    n = per_city_n[city]
    if n == 0:
      continue
    abs_weights = [abs(k) * v for k, v in hist.items()]
    total_w = sum(hist.values()) or 1.0
    p_ge1 = sum(v for k, v in hist.items() if abs(k) >= 1) / total_w
    p_ge2 = sum(v for k, v in hist.items() if abs(k) >= 2) / total_w
    p_ge3 = sum(v for k, v in hist.items() if abs(k) >= 3) / total_w
    median_abs = _weighted_median([(abs(k), v) for k, v in hist.items()])
    out[city] = {
      "n_days": n,
      "median_abs": median_abs,
      "mean_abs": round(sum(abs_weights) / total_w, 3),
      "p_diff_ge_1": round(p_ge1, 4),
      "p_diff_ge_2": round(p_ge2, 4),
      "p_diff_ge_3": round(p_ge3, 4),
      "histogram": {str(k): round(v, 4) for k, v in sorted(hist.items())},
    }
  return out


def _weighted_median(pairs: list[tuple[int, float]]) -> float:
  """Weighted median over (value, weight) pairs."""
  if not pairs:
    return 0.0
  s = sorted(pairs, key=lambda x: x[0])
  total = sum(w for _, w in s)
  half = total / 2.0
  acc = 0.0
  for v, w in s:
    acc += w
    if acc >= half:
      return float(v)
  return float(s[-1][0])


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--decay", type=float, default=1.0,
                      help="Per-day exponential weight (1.0 = no decay).")
  parser.add_argument("--lookback-days", type=int, default=365,
                      help="Ignore settlements older than this.")
  args = parser.parse_args()

  hist = build_histogram(decay=args.decay, lookback_days=args.lookback_days)
  if not hist:
    log.warning("No settlement data found — output not written.")
    return

  # Atomic write: write to .tmp, then rename. Avoids partial reads from the
  # bot's mtime-check thread.
  OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
  tmp = OUTPUT_PATH.with_suffix(".json.tmp")
  tmp.write_text(json.dumps(hist, indent=2))
  os.replace(tmp, OUTPUT_PATH)
  log.info("Wrote %s with %d cities", OUTPUT_PATH, len(hist))

  # Pretty summary
  for city, s in hist.items():
    log.info("  %s: n=%d  P(|Δ|≥1)=%.1f%%  P(|Δ|≥2)=%.1f%%",
             city, s["n_days"],
             100*s["p_diff_ge_1"], 100*s["p_diff_ge_2"])


if __name__ == "__main__":
  main()
