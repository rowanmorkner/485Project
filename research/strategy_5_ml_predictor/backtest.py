"""
Backtest the ML bracket-move predictor.

For each (city, date, venue, bracket), at each TEST snapshot t (last 30%
of timestamps), compute the feature vector and ask the trained model:
  - P(UP) — buy YES if above threshold
  - P(DOWN) — buy NO synthetically if above threshold

Open the position via open_position() and exit via simulate_exit() with a
target = predicted move size and a 30-min time stop. Apply $0.005/contract
fees.

Two iterations:
  v1: GBM models, threshold 0.20 for UP/DOWN, no liquidity filtering.
  v2: GBM models, higher threshold + liquidity filter (spread <= 0.05,
      bid_size & ask_size > 50, has_two_sided=1) and skip mid_t outside
      [0.05, 0.95] (extreme prices).

We avoid double-counting: for each (bracket_id, side), only open one
position per session and don't reopen until the previous has exited.
"""
from __future__ import annotations

import json
import math
import pickle
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.loaders import (
  iter_snapshots, open_position, simulate_exit,
  find_kalshi_bracket, find_polymarket_bracket,
)
from strategy.parsers import _parse_kalshi_range, _parse_polymarket_range


HERE = Path(__file__).parent
MODELS_DIR = HERE / "models"
CITIES = ["Miami", "LA", "Austin", "San Francisco", "Seattle"]
DATE = "2026-05-06"

FEATURE_COLS = [
  "mid_t", "spread", "bid_size", "ask_size", "bid_depth", "ask_depth",
  "dist_peak", "hour_utc", "mins_since_open", "has_two_sided",
  "mid_dt3", "mid_dt6", "mid_dt12", "cross_diff",
]

FEE_PER_CONTRACT = 0.005
SIZE = 10
TIME_STOP_MIN = 30


def load_models():
  with open(MODELS_DIR / "v2_gbm_up.pkl", "rb") as fh:
    up = pickle.load(fh)
  with open(MODELS_DIR / "v2_gbm_down.pkl", "rb") as fh:
    down = pickle.load(fh)
  return up, down


def load_features():
  return pd.read_csv(HERE / "features.csv")


def temporal_test_indices(df: pd.DataFrame, train_frac: float = 0.7):
  """Return set of (city, venue, bracket_id, snap_idx) that are in TEST."""
  test_keys = set()
  for (city, venue, bid), grp in df.groupby(["city", "venue", "bracket_id"]):
    grp = grp.sort_values("snap_idx")
    n = len(grp)
    if n < 10:
      continue
    cut = int(n * train_frac)
    for _, row in grp.iloc[cut:].iterrows():
      test_keys.add((city, venue, bid, int(row["snap_idx"])))
  return test_keys


def trade_metrics(trades: list[dict]) -> dict:
  pnls = [t["pnl"] for t in trades]
  if not pnls:
    return {
      "n_trades": 0, "n_winners": 0, "win_rate": None,
      "total_pnl_dollars": 0.0, "avg_pnl_per_trade": None,
      "median_pnl_per_trade": None, "sharpe_per_trade": None,
      "max_drawdown_dollars": 0.0, "exit_reason_counts": {},
    }
  n = len(pnls)
  winners = [p for p in pnls if p > 0]
  total = sum(pnls)
  avg = total / n
  med = statistics.median(pnls)
  if n > 1:
    sd = statistics.pstdev(pnls)
    sharpe = avg / sd if sd > 0 else None
  else:
    sharpe = None

  # Equity curve drawdown (in trade order)
  cum = 0.0
  peak = 0.0
  max_dd = 0.0
  for p in pnls:
    cum += p
    peak = max(peak, cum)
    dd = cum - peak
    if dd < max_dd:
      max_dd = dd

  reasons = defaultdict(int)
  for t in trades:
    reasons[t["exit_reason"]] += 1

  return {
    "n_trades": n,
    "n_winners": len(winners),
    "win_rate": len(winners) / n,
    "total_pnl_dollars": total,
    "avg_pnl_per_trade": avg,
    "median_pnl_per_trade": med,
    "sharpe_per_trade": sharpe,
    "max_drawdown_dollars": max_dd,
    "exit_reason_counts": dict(reasons),
  }


def build_lookup(df: pd.DataFrame):
  """Map (city, venue, bracket_id, snap_idx) -> feature row dict."""
  lookup = {}
  for _, row in df.iterrows():
    key = (row["city"], row["venue"], row["bracket_id"], int(row["snap_idx"]))
    lookup[key] = row
  return lookup


def liquidity_ok(row) -> bool:
  if row["has_two_sided"] != 1.0:
    return False
  if row["spread"] > 0.05:
    return False
  if row["bid_size"] < 50 or row["ask_size"] < 50:
    return False
  if row["mid_t"] < 0.05 or row["mid_t"] > 0.95:
    return False
  return True


def run_backtest(version: str, prob_threshold: float, use_liquidity: bool):
  print(f"\n=== Running backtest: {version} threshold={prob_threshold} liq={use_liquidity} ===")
  up_pkg, down_pkg = load_models()
  up_model = up_pkg["model"]
  down_model = down_pkg["model"]

  feats_df = load_features()
  feats_df = feats_df.dropna(subset=FEATURE_COLS + ["mid_delta"]).reset_index(drop=True)
  test_keys = temporal_test_indices(feats_df)
  lookup = build_lookup(feats_df)

  trades = []
  signals_emitted = 0
  open_attempts = 0
  open_failures = 0

  # Track open positions by (city, venue, bracket_id, side) to avoid double-count
  open_until_idx = {}

  for city in CITIES:
    snaps = list(iter_snapshots(city, DATE))
    snaps.sort(key=lambda x: x[0])

    for entry_idx, (ts, venues) in enumerate(snaps):
      for venue in ("kalshi", "polymarket"):
        for b in venues.get(venue, []):
          if venue == "kalshi":
            bid = b.get("ticker", "")
            degs = _parse_kalshi_range(b.get("subtitle", ""))
          else:
            bid = b.get("token_id", "")
            degs = _parse_polymarket_range(b.get("question", ""))
          if not bid or not degs:
            continue

          key = (city, venue, bid, entry_idx)
          if key not in test_keys:
            continue
          row = lookup.get(key)
          if row is None:
            continue
          if use_liquidity and not liquidity_ok(row):
            continue

          x = np.array([[row[c] for c in FEATURE_COLS]], dtype=float)
          p_up = up_model.predict_proba(x)[0, 1]
          p_down = down_model.predict_proba(x)[0, 1]

          # Pick the stronger directional signal
          if p_up >= prob_threshold and p_up >= p_down:
            side = "yes_long"
            p_signal = p_up
          elif p_down >= prob_threshold and p_down > p_up:
            side = "no_long"
            p_signal = p_down
          else:
            continue

          # Avoid duplicate open
          dkey = (city, venue, bid, side)
          if dkey in open_until_idx and open_until_idx[dkey] > entry_idx:
            continue

          signals_emitted += 1
          open_attempts += 1
          pos = open_position(city, DATE, venue, b, side=side, size=SIZE, entry_ts=ts)
          if pos is None:
            open_failures += 1
            continue
          pos.bracket_degrees = degs

          # Future snapshots for this venue only
          future = [(t, vv.get(venue, [])) for t, vv in snaps[entry_idx + 1:]]

          # Target: predicted move size of EPS (1.5 cents) per contract,
          # but the predicted-direction probability is the bet — use a
          # modest target equal to ~2 cents minus fees doubled = 1 cent net.
          target = 0.02

          exit_res = simulate_exit(
            pos, future,
            target_pnl_per_contract=target,
            time_stop_minutes=TIME_STOP_MIN,
            fee_per_contract=FEE_PER_CONTRACT,
          )

          # Block reopens until ~time_stop expires (assume 5-min snapshots ~6 idx)
          open_until_idx[dkey] = entry_idx + 6

          trades.append({
            "city": city,
            "venue": venue,
            "bracket_id": bid,
            "bracket_label": pos.bracket_label,
            "side": side,
            "entry_ts": ts,
            "entry_idx": entry_idx,
            "entry_price": pos.entry_price,
            "exit_ts": exit_res.exit_ts,
            "exit_reason": exit_res.exit_reason,
            "exit_price": exit_res.exit_price,
            "pnl": exit_res.realized_pnl,
            "p_up": float(p_up),
            "p_down": float(p_down),
            "p_signal": float(p_signal),
          })

  metrics = trade_metrics(trades)
  metrics["version"] = version
  metrics["signals_emitted"] = signals_emitted
  metrics["open_attempts"] = open_attempts
  metrics["open_failures"] = open_failures
  metrics["prob_threshold"] = prob_threshold
  metrics["use_liquidity_filter"] = use_liquidity

  return metrics, trades


def main():
  results = {"strategy_name": "ml_bracket_move_predictor", "iterations": []}

  # v1: low threshold, no filtering
  m1, t1 = run_backtest("v1", prob_threshold=0.20, use_liquidity=False)
  m1["notes"] = "v1 GBM, threshold=0.20, no filters. Tests whether even noisy ML signal turns a profit on its own."
  m1["settled_n"] = 0
  m1["settled_pnl_dollars"] = None
  print(json.dumps({k: v for k, v in m1.items() if k != "exit_reason_counts"}, indent=2, default=str))
  results["iterations"].append(m1)
  with open(HERE / "trades_v1.json", "w") as fh:
    json.dump(t1, fh, indent=2, default=str)

  # v2: higher threshold + liquidity filter
  m2, t2 = run_backtest("v2", prob_threshold=0.35, use_liquidity=True)
  m2["notes"] = "v2 GBM, threshold=0.35 + liquidity filter (spread<=0.05, bid_size & ask_size>50, mid in [0.05,0.95])."
  m2["settled_n"] = 0
  m2["settled_pnl_dollars"] = None
  print(json.dumps({k: v for k, v in m2.items() if k != "exit_reason_counts"}, indent=2, default=str))
  results["iterations"].append(m2)
  with open(HERE / "trades_v2.json", "w") as fh:
    json.dump(t2, fh, indent=2, default=str)

  # v3: very high threshold (cherry-picked from v1 calibration), no liq filter.
  # Inspecting v1 by p_signal bins, only p in [0.7, 0.8) was profitable in
  # this backtest. v3 isolates that bin to test the "high-confidence" hypothesis
  # — but warning: this is partly fitting-to-test and the sample is small.
  m3, t3 = run_backtest("v3", prob_threshold=0.70, use_liquidity=False)
  m3["notes"] = "v3 GBM, threshold=0.70, no liquidity filter. High-confidence-only filter, picked post-hoc from v1 bin analysis (small-sample, partly overfit)."
  m3["settled_n"] = 0
  m3["settled_pnl_dollars"] = None
  print(json.dumps({k: v for k, v in m3.items() if k != "exit_reason_counts"}, indent=2, default=str))
  results["iterations"].append(m3)
  with open(HERE / "trades_v3.json", "w") as fh:
    json.dump(t3, fh, indent=2, default=str)

  with open(HERE / "metrics.json", "w") as fh:
    json.dump(results, fh, indent=2, default=str)

  print("\nWrote metrics.json")


if __name__ == "__main__":
  main()
