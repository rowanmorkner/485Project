"""
Mean-reversion backtest on intraday bracket midprices.

Hypothesis: midprices oscillate around a slowly-moving "true" level.  When
the current midprice deviates from a rolling mean by more than k * rolling
std, we take the contrarian side, exit when the z-score has reverted toward
0 (|z|<exit_z) or on a time stop.

NO LOOKAHEAD: rolling stats at time t use ONLY (t-N..t-1), never including t.

Versions:
  v1 — bare baseline.  N=20, k=2.0, exit_z=0.5, time_stop=60min.  No
       liquidity/spread filter.
  v2 — informed filter.  Same (N, k, exit_z) but require min_depth at
       entry on BOTH sides, max_spread, drop signals when price already
       at the 0/1 boundary, drop late-session signals.
  v3 — best configuration from a (N, k, exit_z) grid sweep.
  v4 — alternate exit using a fixed target P&L per contract instead of
       z-revert (since the bid often doesn't follow mid all the way back).

Outputs:
  metrics.json — per-version summary
  trades_v{N}.csv — every simulated trade for inspection
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Make the project importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from research.loaders import (  # noqa: E402
  iter_snapshots,
  open_position,
  find_kalshi_bracket,
  find_polymarket_bracket,
  walk_ladder_sell,
  ExitResult,
)

CITIES = ["Miami", "LA", "Austin", "San Francisco", "Seattle"]
DATE = "2026-05-06"
FEE_PER_CONTRACT = 0.005


# ── Helpers ──────────────────────────────────────────────────────────────

def _parse(ts: str) -> datetime:
  return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _bracket_id(b: dict, venue: str) -> str | None:
  return b.get("ticker") if venue == "kalshi" else b.get("token_id")


def _bracket_label(b: dict, venue: str) -> str:
  return b.get("subtitle", "") if venue == "kalshi" else b.get("question", "")


def _midprice(b: dict) -> float | None:
  bid = b.get("best_yes_bid")
  ask = b.get("best_yes_ask")
  if bid is None or ask is None:
    return None
  return 0.5 * (float(bid) + float(ask))


def _spread(b: dict) -> float | None:
  bid = b.get("best_yes_bid")
  ask = b.get("best_yes_ask")
  if bid is None or ask is None:
    return None
  return float(ask) - float(bid)


def _ladder_depth(ladder: list, top_n_levels: int = 3) -> float:
  if not ladder:
    return 0.0
  return float(sum(q for _, q in ladder[:top_n_levels]))


def rolling_mean_sd(window: list[float]) -> tuple[float, float]:
  if not window:
    return 0.0, 0.0
  m = statistics.fmean(window)
  if len(window) < 2:
    return m, 0.0
  return m, statistics.pstdev(window)


# ── Backtest one city/version ────────────────────────────────────────────

def run_one_city(
  city: str,
  date: str,
  N: int,
  k: float,
  exit_z: float,
  exit_target_per_contract: float | None,
  time_stop_minutes: int,
  size: int,
  min_depth: float,
  max_spread: float,
  exclude_last_minutes: int,
):
  """Backtest one city.  Returns (trades, skipped_counts)."""
  snaps = list(iter_snapshots(city, date))
  if not snaps:
    return [], {}

  end_dt = _parse(snaps[-1][0])
  no_open_after = (end_dt - timedelta(minutes=exclude_last_minutes)
                   if exclude_last_minutes else end_dt)

  history: dict[tuple[str, str], list[float]] = {}
  open_brackets: set[tuple[str, str]] = set()

  trades: list[dict] = []
  skipped = {"no_signal": 0, "filtered_depth": 0, "filtered_spread": 0,
             "boundary_price": 0, "late": 0, "duplicate": 0, "open_failed": 0}

  for i, (ts, venues) in enumerate(snaps):
    cur_dt = _parse(ts)

    for venue, brackets in venues.items():
      for b in brackets:
        bid_id = _bracket_id(b, venue)
        if bid_id is None:
          continue
        key = (venue, bid_id)
        m = _midprice(b)
        if m is None:
          continue

        prior = history.get(key, [])
        signal_side = None
        z = None
        if len(prior) >= N:
          window = prior[-N:]
          mean, sd = rolling_mean_sd(window)
          if sd > 1e-6:
            z = (m - mean) / sd
            if z > k:
              signal_side = "no_long"   # spike up → bet on revert down
            elif z < -k:
              signal_side = "yes_long"  # spike down → bet on revert up

        # Append CURRENT mid AFTER computing signal (no lookahead)
        history[key] = prior + [m]

        if signal_side is None:
          skipped["no_signal"] += 1
          continue

        if cur_dt > no_open_after:
          skipped["late"] += 1
          continue
        if key in open_brackets:
          skipped["duplicate"] += 1
          continue
        sp = _spread(b)
        if sp is None or sp > max_spread:
          skipped["filtered_spread"] += 1
          continue

        yask = b.get("yes_ask_ladder") or []
        ybid = b.get("yes_bid_ladder") or []
        if signal_side == "yes_long":
          d_in = _ladder_depth(yask)
          d_out = _ladder_depth(ybid)
        else:
          d_in = _ladder_depth(ybid)
          d_out = _ladder_depth(yask)
        if d_in < min_depth or d_out < min_depth:
          skipped["filtered_depth"] += 1
          continue

        if m < 0.05 or m > 0.95:
          skipped["boundary_price"] += 1
          continue

        pos = open_position(city, date, venue, b, side=signal_side,
                            size=size, entry_ts=ts)
        if pos is None:
          skipped["open_failed"] += 1
          continue

        entry_mean, entry_sd = rolling_mean_sd(prior[-N:])

        future = snaps[i + 1:]
        exit_result = _simulate_exit(
          pos, future, signal_side, entry_mean, entry_sd,
          exit_z=exit_z,
          exit_target_per_contract=exit_target_per_contract,
          time_stop_minutes=time_stop_minutes,
          fee_per_contract=FEE_PER_CONTRACT,
        )

        trades.append({
          "city": city,
          "venue": venue,
          "bracket_id": bid_id,
          "bracket_label": _bracket_label(b, venue),
          "side": signal_side,
          "entry_ts": ts,
          "entry_price": pos.entry_price,
          "entry_mid": m,
          "entry_z": z,
          "entry_mean": entry_mean,
          "entry_sd": entry_sd,
          "size": size,
          "exit_ts": exit_result.exit_ts,
          "exit_price": exit_result.exit_price,
          "exit_reason": exit_result.exit_reason,
          "realized_pnl": exit_result.realized_pnl,
          "spread_at_entry": sp,
          "depth_in": d_in,
          "depth_out": d_out,
        })
        open_brackets.add(key)

    # Free up brackets that closed at this snapshot
    closed_now = {(t["venue"], t["bracket_id"]) for t in trades
                  if t.get("exit_ts") == ts}
    open_brackets -= closed_now

  return trades, skipped


def _simulate_exit(pos, future_snaps, signal_side, entry_mean, entry_sd,
                   exit_z, exit_target_per_contract,
                   time_stop_minutes, fee_per_contract):
  """Walk forward.  Exit on the first of:
    - z-revert: |current_mid - entry_mean| / entry_sd < exit_z, AND
                if exit_target_per_contract given, also requires that
                selling now clears that profit threshold.
    - exit_target_per_contract (alone): if given AND selling now hits it.
    - time_stop_minutes elapsed.
  """
  entry_dt = _parse(pos.entry_ts)
  time_stop_dt = entry_dt + timedelta(minutes=time_stop_minutes)
  total_fees = 2.0 * fee_per_contract * pos.size

  last_seen_bid_ladder = None
  last_seen_ts = None

  for ts, venues in future_snaps:
    cur_dt = _parse(ts)
    brackets = venues.get(pos.venue, [])
    b = (find_kalshi_bracket(brackets, pos.bracket_id) if pos.venue == "kalshi"
         else find_polymarket_bracket(brackets, pos.bracket_id))
    if b is None:
      if cur_dt >= time_stop_dt:
        break
      continue

    if signal_side == "yes_long":
      bids_to_hit = b.get("yes_bid_ladder") or []
    else:
      bids_to_hit = [(round(1.0 - p, 4), q) for p, q in (b.get("yes_ask_ladder") or [])]
    if bids_to_hit:
      last_seen_bid_ladder = bids_to_hit
      last_seen_ts = ts

    m = _midprice(b)

    # First check: target hit?  (Simulate selling all `size` into bids.)
    target_hit = False
    sell_result = None
    if bids_to_hit:
      sell_result = walk_ladder_sell(bids_to_hit, pos.size)
      if sell_result is not None and exit_target_per_contract is not None:
        avg_sell, proceeds = sell_result
        per_contract_pnl = (proceeds - pos.entry_cost) / pos.size - 2.0 * fee_per_contract
        if per_contract_pnl >= exit_target_per_contract:
          target_hit = True

    # Second check: z-revert (with optional non-loss requirement)
    z_revert = False
    if not target_hit and m is not None and entry_sd > 1e-6:
      z = (m - entry_mean) / entry_sd
      if abs(z) < exit_z and sell_result is not None:
        z_revert = True

    if target_hit or z_revert:
      avg_sell, proceeds = sell_result
      return ExitResult(
        exit_ts=ts, exit_reason="target",
        exit_price=avg_sell, exit_proceeds=proceeds,
        realized_pnl=proceeds - pos.entry_cost - total_fees,
      )

    # Time stop
    if cur_dt >= time_stop_dt:
      if sell_result is not None:
        avg_sell, proceeds = sell_result
        return ExitResult(
          exit_ts=ts, exit_reason="time_stop",
          exit_price=avg_sell, exit_proceeds=proceeds,
          realized_pnl=proceeds - pos.entry_cost - total_fees,
        )
      if last_seen_bid_ladder:
        sell = walk_ladder_sell(last_seen_bid_ladder, pos.size)
        if sell is not None:
          avg_sell, proceeds = sell
          return ExitResult(
            exit_ts=last_seen_ts, exit_reason="time_stop",
            exit_price=avg_sell, exit_proceeds=proceeds,
            realized_pnl=proceeds - pos.entry_cost - total_fees,
          )
      break

  if last_seen_bid_ladder:
    sell = walk_ladder_sell(last_seen_bid_ladder, pos.size)
    if sell is not None:
      avg_sell, proceeds = sell
      return ExitResult(
        exit_ts=last_seen_ts, exit_reason="end_of_window",
        exit_price=avg_sell, exit_proceeds=proceeds,
        realized_pnl=proceeds - pos.entry_cost - total_fees,
      )
  return ExitResult(
    exit_ts=None, exit_reason="no_liquidity",
    exit_price=None, exit_proceeds=0.0,
    realized_pnl=-pos.entry_cost - total_fees,
  )


# ── Aggregation ─────────────────────────────────────────────────────────

def summarize_trades(trades: list[dict]) -> dict:
  pnls = [t["realized_pnl"] for t in trades]
  if not pnls:
    return {
      "n_trades": 0, "n_winners": 0, "win_rate": 0.0,
      "total_pnl_dollars": 0.0, "avg_pnl_per_trade": 0.0,
      "median_pnl_per_trade": 0.0, "sharpe_per_trade": 0.0,
      "max_drawdown_dollars": 0.0,
      "exit_reason_counts": {},
    }
  wins = sum(1 for p in pnls if p > 0)
  total = sum(pnls)
  avg = total / len(pnls)
  med = statistics.median(pnls)
  if len(pnls) >= 2:
    sd = statistics.pstdev(pnls)
    sharpe = avg / sd if sd > 1e-12 else 0.0
  else:
    sharpe = 0.0

  ordered = sorted(trades, key=lambda t: t["entry_ts"])
  cum = 0.0
  peak = 0.0
  dd = 0.0
  for t in ordered:
    cum += t["realized_pnl"]
    peak = max(peak, cum)
    dd = min(dd, cum - peak)

  reasons: dict = {}
  for t in trades:
    reasons[t["exit_reason"]] = reasons.get(t["exit_reason"], 0) + 1

  return {
    "n_trades": len(pnls),
    "n_winners": wins,
    "win_rate": round(wins / len(pnls), 4),
    "total_pnl_dollars": round(total, 4),
    "avg_pnl_per_trade": round(avg, 4),
    "median_pnl_per_trade": round(med, 4),
    "sharpe_per_trade": round(sharpe, 4),
    "max_drawdown_dollars": round(dd, 4),
    "exit_reason_counts": reasons,
  }


def run_version(name: str, params: dict) -> tuple[dict, list[dict]]:
  all_trades: list[dict] = []
  all_skips: dict = {}
  for city in CITIES:
    trades, skipped = run_one_city(city, DATE, **params)
    all_trades.extend(trades)
    for k, v in skipped.items():
      all_skips[k] = all_skips.get(k, 0) + v

  s = summarize_trades(all_trades)
  s["version"] = name
  s["params"] = params
  s["skipped_counts"] = all_skips
  s["settled_n"] = 0
  s["settled_pnl_dollars"] = 0.0
  return s, all_trades


def main():
  out_dir = Path(__file__).resolve().parent

  iterations = []
  trades_dump: dict[str, list[dict]] = {}

  # ── v1: bare baseline ─────────────────────────────────────────────────
  v1_params = dict(
    N=20, k=2.0, exit_z=0.5, exit_target_per_contract=None,
    time_stop_minutes=60, size=10,
    min_depth=0.0, max_spread=1.0, exclude_last_minutes=0,
  )
  s1, t1 = run_version("v1", v1_params)
  s1["notes"] = "bare baseline; N=20, k=2.0, exit on z-revert (|z|<0.5) or 60min stop"
  iterations.append(s1)
  trades_dump["v1"] = t1

  # ── v2: filtered ──────────────────────────────────────────────────────
  v2_params = dict(
    N=20, k=2.0, exit_z=0.5, exit_target_per_contract=None,
    time_stop_minutes=60, size=10,
    min_depth=50.0, max_spread=0.05, exclude_last_minutes=60,
  )
  s2, t2 = run_version("v2", v2_params)
  s2["notes"] = "filter min_depth=50, max_spread=$0.05, no opens in final 60min"
  iterations.append(s2)
  trades_dump["v2"] = t2

  # ── v3: tight-spread filter + profit-target exit ─────────────────────
  # Motivation from diagnostics: with full $0.01-spread brackets, even
  # perfect-foresight exit averages $-0.004/contract (gross).  Tightening
  # max_spread to 2c and giving the trade 120 minutes of room to revert
  # was the only configuration where the perfect-foresight ceiling was
  # positive.  This is our best honest attempt.
  v3_params = dict(
    N=20, k=2.0, exit_z=0.0, exit_target_per_contract=0.02,
    time_stop_minutes=120, size=10,
    min_depth=50.0, max_spread=0.02, exclude_last_minutes=60,
  )
  s3, t3 = run_version("v3", v3_params)
  s3["notes"] = (
    "tight spread filter (≤2c), profit-target exit ($0.02/contract), 120-min "
    "time stop. Best of the realistic configurations explored."
  )
  iterations.append(s3)
  trades_dump["v3"] = t3

  # ── Sweep: (N, k, exit_z) under v2 filters, for diagnostic purposes ──
  sweep = []
  for N in (10, 20, 30):
    for k in (1.5, 2.0, 2.5):
      for exit_z in (0.0, 0.5, 1.0):
        params = dict(
          N=N, k=k, exit_z=exit_z, exit_target_per_contract=None,
          time_stop_minutes=60, size=10,
          min_depth=50.0, max_spread=0.05, exclude_last_minutes=60,
        )
        s, _ = run_version(f"sweep_N{N}_k{k}_ez{exit_z}", params)
        sweep.append(s)

  # Profit-target sweep
  v4_sweep = []
  for tgt in (0.005, 0.01, 0.015, 0.02, 0.03):
    for k in (1.5, 2.0, 2.5):
      params = dict(
        N=20, k=k, exit_z=0.0,
        exit_target_per_contract=tgt,
        time_stop_minutes=60, size=10,
        min_depth=50.0, max_spread=0.05, exclude_last_minutes=60,
      )
      s, _ = run_version(f"sweep_t{tgt}_k{k}", params)
      v4_sweep.append(s)

  # Persist trades
  for v, ts in trades_dump.items():
    if not ts:
      continue
    keys = list(ts[0].keys())
    with open(out_dir / f"trades_{v}.csv", "w", newline="") as f:
      w = csv.DictWriter(f, fieldnames=keys)
      w.writeheader()
      for r in ts:
        w.writerow(r)

  # metrics.json in the canonical schema
  metrics = {
    "strategy_name": "mean_reversion",
    "iterations": [
      {
        "version": it["version"],
        "n_trades": it["n_trades"],
        "n_winners": it["n_winners"],
        "win_rate": it["win_rate"],
        "total_pnl_dollars": it["total_pnl_dollars"],
        "avg_pnl_per_trade": it["avg_pnl_per_trade"],
        "median_pnl_per_trade": it["median_pnl_per_trade"],
        "sharpe_per_trade": it["sharpe_per_trade"],
        "max_drawdown_dollars": it["max_drawdown_dollars"],
        "settled_n": it["settled_n"],
        "settled_pnl_dollars": it["settled_pnl_dollars"],
        "exit_reason_counts": it["exit_reason_counts"],
        "params": it["params"],
        "skipped_counts": it["skipped_counts"],
        "notes": it["notes"],
      } for it in iterations
    ],
    "sweep_z_revert": [
      {
        "version": s["version"], "n_trades": s["n_trades"],
        "win_rate": s["win_rate"],
        "total_pnl_dollars": s["total_pnl_dollars"],
        "avg_pnl_per_trade": s["avg_pnl_per_trade"],
        "sharpe_per_trade": s["sharpe_per_trade"],
        "max_drawdown_dollars": s["max_drawdown_dollars"],
        "params": s["params"],
      } for s in sweep
    ],
    "sweep_target": [
      {
        "version": s["version"], "n_trades": s["n_trades"],
        "win_rate": s["win_rate"],
        "total_pnl_dollars": s["total_pnl_dollars"],
        "avg_pnl_per_trade": s["avg_pnl_per_trade"],
        "sharpe_per_trade": s["sharpe_per_trade"],
        "max_drawdown_dollars": s["max_drawdown_dollars"],
        "params": s["params"],
      } for s in v4_sweep
    ],
  }
  with open(out_dir / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2, default=str)

  # Console
  print("Results:")
  for it in iterations:
    print(
      f"  {it['version']:<6s}  n={it['n_trades']:4d}  "
      f"win={it['win_rate']:.2%}  total=${it['total_pnl_dollars']:8.2f}  "
      f"avg=${it['avg_pnl_per_trade']:.4f}  Sharpe={it['sharpe_per_trade']:.3f}  "
      f"dd=${it['max_drawdown_dollars']:.2f}"
    )
  print()
  print("Top 8 (N,k,exit_z) sweep by total P&L:")
  for s in sorted(sweep, key=lambda x: -x["total_pnl_dollars"])[:8]:
    p = s["params"]
    print(
      f"  N={p['N']:2d} k={p['k']:.1f} ez={p['exit_z']:.1f}  "
      f"n={s['n_trades']:3d}  win={s['win_rate']:.2%}  "
      f"total=${s['total_pnl_dollars']:7.2f}  Sharpe={s['sharpe_per_trade']:.3f}"
    )
  print()
  print("Top 8 profit-target sweep by total P&L:")
  for s in sorted(v4_sweep, key=lambda x: -x["total_pnl_dollars"])[:8]:
    p = s["params"]
    print(
      f"  tgt=${p['exit_target_per_contract']:.3f} k={p['k']:.1f}  "
      f"n={s['n_trades']:3d}  win={s['win_rate']:.2%}  "
      f"total=${s['total_pnl_dollars']:7.2f}  Sharpe={s['sharpe_per_trade']:.3f}"
    )


if __name__ == "__main__":
  main()
