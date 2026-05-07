"""
Build per-bracket per-timestamp feature rows from snapshot data.

For each (city, venue, bracket) and each timestamp t, we compute:
  - mid_t            current bracket midprice
  - spread           best_ask - best_bid
  - bid_size, ask_size
  - bid_depth, ask_depth   total qty across the ladder
  - dist_from_peak   |bracket_center - forecast_peak_degree|
  - hour_utc, mins_since_open
  - mid_dt3, mid_dt6, mid_dt12   mid_t - mid_{t-K}
  - cross_diff       this bracket's mid - mid of best-overlap bracket
                     on the OTHER venue at same timestamp (NaN if missing)
  - mid_t_plus_K     LABEL TARGET — mid at t+6 snapshots (the "future"
                     label, only used for label generation, not features)

Outputs a parquet/csv to disk.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.loaders import iter_snapshots, load_forecast_pdf
from strategy.parsers import _parse_kalshi_range, _parse_polymarket_range


CITIES = ["Miami", "LA", "Austin", "San Francisco", "Seattle"]
DATE = "2026-05-06"
LABEL_HORIZON = 6  # K snapshots ahead (~30 min at 5-min cadence)
RECENT_LAGS = (3, 6, 12)


def bracket_id_and_degrees(b: dict, venue: str):
  if venue == "kalshi":
    return b.get("ticker", ""), _parse_kalshi_range(b.get("subtitle", ""))
  return b.get("token_id", ""), _parse_polymarket_range(b.get("question", ""))


def bracket_center(degs: list[int]) -> float:
  if not degs:
    return float("nan")
  return (degs[0] + degs[-1]) / 2.0


def bracket_label(b: dict, venue: str) -> str:
  return b.get("subtitle", "") if venue == "kalshi" else b.get("question", "")


def midprice(b: dict):
  bid = b.get("best_yes_bid")
  ask = b.get("best_yes_ask")
  if bid is None and ask is None:
    return None
  if bid is None:
    return ask
  if ask is None:
    return bid
  return (bid + ask) / 2.0


def total_depth(ladder):
  if not ladder:
    return 0.0
  return float(sum(q for _, q in ladder if q))


def forecast_peak(pdf):
  if not pdf:
    return None
  return max(pdf.items(), key=lambda kv: kv[1])[0]


def parse_ts(ts: str) -> datetime:
  return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def best_overlap_bracket(my_degs: list[int], other_brackets, other_venue):
  """Find bracket on other venue with most overlapping degrees with mine."""
  if not my_degs or not other_brackets:
    return None
  my_set = set(my_degs)
  best = None
  best_overlap = 0
  for ob in other_brackets:
    _, odegs = bracket_id_and_degrees(ob, other_venue)
    if not odegs:
      continue
    ov = len(my_set & set(odegs))
    if ov > best_overlap:
      best_overlap = ov
      best = ob
  return best


def build_for_city(city: str, date: str):
  pdf = load_forecast_pdf(city, date)
  peak = forecast_peak(pdf)
  snaps = list(iter_snapshots(city, date))
  if not snaps:
    return []

  # Sort snapshots by timestamp (already ordered by loader, but be safe)
  snaps.sort(key=lambda x: x[0])
  open_dt = parse_ts(snaps[0][0])

  # Per-bracket time series of mid for fast lag lookup.
  # Key = (venue, bracket_id) -> list of (idx, ts, mid, bracket_dict)
  per_bracket = {}
  for idx, (ts, venues) in enumerate(snaps):
    for venue in ("kalshi", "polymarket"):
      brks = venues.get(venue, [])
      for b in brks:
        bid, degs = bracket_id_and_degrees(b, venue)
        if not bid or not degs:
          continue
        m = midprice(b)
        if m is None:
          continue
        per_bracket.setdefault((venue, bid), []).append((idx, ts, m, b, degs))

  rows = []
  for (venue, bid), series in per_bracket.items():
    # Index series by snapshot idx for lag lookup
    by_idx = {entry[0]: entry for entry in series}
    other_venue = "polymarket" if venue == "kalshi" else "kalshi"
    for entry in series:
      idx, ts, mid_t, b, degs = entry
      future = by_idx.get(idx + LABEL_HORIZON)
      if future is None:
        continue  # cannot label
      mid_future = future[2]

      best_bid = b.get("best_yes_bid")
      best_ask = b.get("best_yes_ask")
      spread = (best_ask - best_bid) if (best_bid is not None and best_ask is not None) else float("nan")
      bid_sz = float(b.get("best_yes_bid_size") or 0.0)
      ask_sz = float(b.get("best_yes_ask_size") or 0.0)
      bdepth = total_depth(b.get("yes_bid_ladder"))
      adepth = total_depth(b.get("yes_ask_ladder"))

      center = bracket_center(degs)
      dist_peak = abs(center - peak) if peak is not None else float("nan")

      cur_dt = parse_ts(ts)
      hour_utc = cur_dt.hour + cur_dt.minute / 60.0
      mins_since_open = (cur_dt - open_dt).total_seconds() / 60.0

      lag_feats = {}
      for K in RECENT_LAGS:
        prev = by_idx.get(idx - K)
        lag_feats[f"mid_dt{K}"] = (mid_t - prev[2]) if prev else float("nan")

      # Cross-venue diff
      other_brks = snaps[idx][1].get(other_venue, [])
      ob = best_overlap_bracket(degs, other_brks, other_venue)
      if ob is not None:
        om = midprice(ob)
        cross_diff = (mid_t - om) if om is not None else float("nan")
      else:
        cross_diff = float("nan")

      # Liquidity feature: 1 if both bid and ask exist
      has_two_sided = 1.0 if (best_bid is not None and best_ask is not None) else 0.0

      rows.append({
        "city": city,
        "date": date,
        "venue": venue,
        "bracket_id": bid,
        "bracket_label": bracket_label(b, venue),
        "deg_low": degs[0],
        "deg_high": degs[-1],
        "ts": ts,
        "snap_idx": idx,
        "mid_t": mid_t,
        "best_bid": best_bid if best_bid is not None else float("nan"),
        "best_ask": best_ask if best_ask is not None else float("nan"),
        "spread": spread,
        "bid_size": bid_sz,
        "ask_size": ask_sz,
        "bid_depth": bdepth,
        "ask_depth": adepth,
        "dist_peak": dist_peak,
        "hour_utc": hour_utc,
        "mins_since_open": mins_since_open,
        "has_two_sided": has_two_sided,
        "mid_dt3": lag_feats["mid_dt3"],
        "mid_dt6": lag_feats["mid_dt6"],
        "mid_dt12": lag_feats["mid_dt12"],
        "cross_diff": cross_diff,
        "mid_future": mid_future,
        "mid_delta": mid_future - mid_t,
      })
  return rows


def main():
  out = Path(__file__).parent / "features.csv"
  total = 0
  fields = None
  with open(out, "w", newline="") as fh:
    writer = None
    for city in CITIES:
      rows = build_for_city(city, DATE)
      total += len(rows)
      print(f"  {city}: {len(rows)} rows")
      if writer is None and rows:
        fields = list(rows[0].keys())
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
      for r in rows:
        writer.writerow(r)
  print(f"Wrote {total} rows -> {out}")


if __name__ == "__main__":
  main()
