"""
Diagnostics: are bracket midprices mean-reverting at the timescales we have?

We compute:
  - Per-bracket time series of midprice = (best_yes_bid + best_yes_ask)/2
    (None when either side is unquoted).
  - Lag-1 autocorrelation of price changes.  AR(1) on Δp of <0 indicates
    mean reversion (a positive change tends to be followed by a negative
    change).  AR(1) > 0 indicates trending; AR(1) ≈ 0 indicates random walk.
  - Variance ratio statistic VR(q) = Var(p_t - p_{t-q}) / (q * Var(p_t - p_{t-1})).
    VR<1 implies mean reversion at horizon q; VR>1 trending; VR=1 RW.

Run-only.  Writes a JSON summary used by the writeup.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

# Add project root so we can import research.* and strategy.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from research.loaders import iter_snapshots  # noqa: E402

CITIES = ["Miami", "LA", "Austin", "San Francisco", "Seattle"]
DATE = "2026-05-06"
VENUES = ("kalshi", "polymarket")


def _midprice(b: dict) -> float | None:
  """Return (bid+ask)/2 if both are quoted, else None."""
  bid = b.get("best_yes_bid")
  ask = b.get("best_yes_ask")
  if bid is None or ask is None:
    return None
  return 0.5 * (float(bid) + float(ask))


def _bracket_id(b: dict, venue: str) -> str | None:
  return b.get("ticker") if venue == "kalshi" else b.get("token_id")


def collect_series(city: str, date: str) -> dict[tuple[str, str], list[tuple[str, float]]]:
  """Return {(venue, bracket_id): [(ts, mid), ...]} dropping snapshots with no mid."""
  series: dict[tuple[str, str], list[tuple[str, float]]] = {}
  for ts, venues in iter_snapshots(city, date):
    for venue, brackets in venues.items():
      for b in brackets:
        bid = _bracket_id(b, venue)
        if bid is None:
          continue
        m = _midprice(b)
        if m is None:
          continue
        series.setdefault((venue, bid), []).append((ts, m))
  return series


def lag1_autocorr(diffs: list[float]) -> float | None:
  """Pearson autocorrelation of diffs vs diffs shifted by 1."""
  if len(diffs) < 5:
    return None
  m = statistics.fmean(diffs)
  a = [d - m for d in diffs]
  num = sum(a[i] * a[i + 1] for i in range(len(a) - 1))
  den = sum(x * x for x in a)
  if den == 0:
    return None
  return num / den


def variance_ratio(prices: list[float], q: int) -> float | None:
  """VR(q) = Var(p_t - p_{t-q}) / (q * Var(p_t - p_{t-1}))."""
  if len(prices) < q + 5:
    return None
  d1 = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
  dq = [prices[i + q] - prices[i] for i in range(len(prices) - q)]
  if len(d1) < 2 or len(dq) < 2:
    return None
  v1 = statistics.pvariance(d1)
  vq = statistics.pvariance(dq)
  if v1 == 0:
    return None
  return vq / (q * v1)


def diagnostics_for_series(prices: list[float]) -> dict:
  if len(prices) < 5:
    return {"n": len(prices), "ar1": None, "vr2": None, "vr5": None, "sd_diff": None}
  diffs = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
  return {
    "n": len(prices),
    "ar1": lag1_autocorr(diffs),
    "vr2": variance_ratio(prices, 2),
    "vr5": variance_ratio(prices, 5),
    "vr10": variance_ratio(prices, 10),
    "sd_diff": statistics.pstdev(diffs) if len(diffs) >= 2 else None,
    "mean": statistics.fmean(prices),
    "sd": statistics.pstdev(prices),
  }


def summarize(stats_per_bracket: list[dict]) -> dict:
  def _agg(key: str) -> dict:
    vals = [s[key] for s in stats_per_bracket if s.get(key) is not None]
    if not vals:
      return {"n": 0}
    return {
      "n": len(vals),
      "mean": statistics.fmean(vals),
      "median": statistics.median(vals),
      "p10": sorted(vals)[len(vals) // 10] if len(vals) >= 10 else None,
      "p90": sorted(vals)[(len(vals) * 9) // 10] if len(vals) >= 10 else None,
      "frac_negative": sum(1 for v in vals if v < 0) / len(vals),
      "frac_below_1": sum(1 for v in vals if v < 1) / len(vals),
    }
  return {k: _agg(k) for k in ("ar1", "vr2", "vr5", "vr10")}


def main():
  out_dir = Path(__file__).resolve().parent
  big_summary: dict = {"cities": {}, "all_brackets": [], "all_summary": None}

  for city in CITIES:
    series = collect_series(city, DATE)
    rows = []
    for (venue, bid), pts in series.items():
      if len(pts) < 20:
        continue
      prices = [p for _, p in pts]
      d = diagnostics_for_series(prices)
      d.update({"venue": venue, "bracket_id": bid, "n_obs": len(pts), "city": city})
      rows.append(d)
      big_summary["all_brackets"].append(d)
    big_summary["cities"][city] = {
      "n_brackets": len(rows),
      "summary": summarize(rows),
    }

  big_summary["all_summary"] = summarize(big_summary["all_brackets"])

  # Strip ar1=None etc from per-bracket rows but keep summary intact
  with open(out_dir / "diagnostics.json", "w") as f:
    json.dump(big_summary, f, indent=2, default=str)

  # Print readable summary
  print("Per-city summary (autocorrelation of midprice changes):")
  for city, d in big_summary["cities"].items():
    s = d["summary"]
    ar1 = s.get("ar1", {})
    vr5 = s.get("vr5", {})
    print(
      f"  {city:14s}  n={d['n_brackets']:3d}  "
      f"AR1 median={ar1.get('median'):.3f}  frac_neg={ar1.get('frac_negative'):.2f}  "
      f"VR5 median={vr5.get('median'):.3f}  frac_<1={vr5.get('frac_below_1'):.2f}"
    )
  s = big_summary["all_summary"]
  print()
  print("Overall (n =", s["ar1"]["n"], "brackets):")
  for k in ("ar1", "vr2", "vr5", "vr10"):
    v = s[k]
    print(
      f"  {k}: median={v['median']:.3f}  mean={v['mean']:.3f}  "
      f"frac_neg={v['frac_negative']:.2f}  frac_<1={v['frac_below_1']:.2f}"
    )


if __name__ == "__main__":
  main()
