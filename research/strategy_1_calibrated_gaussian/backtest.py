"""
Calibrated Forecast Gaussian Value-Trading backtest on 2026-05-06 snapshots.

For each (city, ts) on 2026-05-06:
  - Build a calibrated Gaussian PDF centered on (NWS forecast + bias)
    with std_dev = calibration.json[city].empirical_std_dev.
  - For each Kalshi & Polymarket bracket, fair_value = sum(pdf[d] for d in degrees).
  - If fair_value - best_ask > THRESHOLD AND ladder depth >= MIN_SIZE
    AND best_ask in [MIN_ASK, MAX_ASK], buy YES.
  - Exit via simulate_exit with target_pnl = TARGET_FRAC * predicted_edge.
  - One open position per (city, venue, bracket_id) at a time.

Iterations:
  v1: naive threshold 0.05, min ask depth 50, no price band, hold-to-EOD.
  v2: tighter threshold + ask price band [0.05, 0.95] + min depth 100
      + smaller target (50% of predicted edge) + 60-min time stop.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# Make project root importable
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from research.loaders import (  # noqa: E402
  iter_snapshots, load_forecast_pdf, list_settled_pairs,
  open_position, simulate_exit, settlement_payout_for,
)
from strategy.parsers import _parse_kalshi_range, _parse_polymarket_range  # noqa: E402

CITIES = ["Miami", "LA", "Austin", "San Francisco", "Seattle"]
DATE = "2026-05-06"
FEE = 0.005

CAL_PATH = Path(__file__).resolve().parent / "calibration.json"


# ── Distribution helpers ─────────────────────────────────────────────────

def _normal_cdf(x: float) -> float:
  return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def gaussian_pdf(center: float, sigma: float) -> dict[int, float]:
  """Discretized Gaussian over integer degrees, +/- 4 sigma. Normalized."""
  lo = int(math.floor(center - 4 * sigma))
  hi = int(math.ceil(center + 4 * sigma))
  pdf = {}
  for d in range(lo, hi + 1):
    p = (_normal_cdf((d + 0.5 - center) / sigma)
         - _normal_cdf((d - 0.5 - center) / sigma))
    if p > 1e-9:
      pdf[d] = p
  total = sum(pdf.values())
  return {d: p / total for d, p in pdf.items()}


def forecast_mean_from_pdf(pdf: dict[int, float]) -> float:
  total = sum(pdf.values())
  if total == 0:
    return float("nan")
  return sum(d * p for d, p in pdf.items()) / total


def fair_value(pdf: dict[int, float], degrees: list[int]) -> float:
  return sum(pdf.get(d, 0.0) for d in degrees)


# ── Bracket parsing ──────────────────────────────────────────────────────

def parse_bracket_degrees(bracket: dict, venue: str) -> list[int]:
  if venue == "kalshi":
    return _parse_kalshi_range(bracket.get("subtitle", ""))
  return _parse_polymarket_range(bracket.get("question", ""))


def best_ask_and_depth(bracket: dict) -> tuple[Optional[float], int]:
  ask = bracket.get("best_yes_ask")
  asks = bracket.get("yes_ask_ladder") or []
  # Total size at the best (lowest) ask
  if not asks:
    return ask, 0
  best = min((float(p), float(q)) for p, q in asks if q and q > 0)
  best_price = best[0]
  total_at_best = sum(float(q) for p, q in asks if abs(float(p) - best_price) < 1e-9)
  return best_price, int(total_at_best)


# ── Backtest config ──────────────────────────────────────────────────────

@dataclass
class StrategyConfig:
  version: str
  edge_threshold: float        # min (fair_value - best_ask) to enter
  min_depth_at_best: int       # ladder size at best ask required
  size: int                    # contracts per trade
  target_frac_of_edge: float   # exit target = frac * edge
  time_stop_minutes: Optional[int]
  ask_min: float               # don't trade asks below this (partial-resolution noise)
  ask_max: float               # don't trade asks above this
  exclude_tail_brackets: bool = False
  sigma_override: Optional[float] = None  # if set, ignore per-city calibration sigma
  fair_value_min: float = 0.0  # only trade brackets where our PDF says fv >= this
  central_bracket_only: bool = False  # only trade the bracket containing the forecast mean
  notes: str = ""


# ── Per-trade run ────────────────────────────────────────────────────────

def run_strategy(cfg: StrategyConfig, calibration: dict) -> dict:
  """Run cfg over all 5 cities for DATE. Returns metrics dict."""
  trades: list[dict] = []
  exit_reasons: dict[str, int] = {}
  per_city_pnl: dict[str, float] = {}

  for city in CITIES:
    cal = calibration.get(city, {})
    sigma = cfg.sigma_override if cfg.sigma_override is not None else cal.get("empirical_std_dev", 2.0)
    bias = cal.get("bias", 0.0)

    nws_pdf = load_forecast_pdf(city, DATE)
    if not nws_pdf:
      continue
    nws_mean = forecast_mean_from_pdf(nws_pdf)
    if math.isnan(nws_mean):
      continue
    # Calibrated PDF: re-center on (forecast_mean + bias) with our sigma
    pdf = gaussian_pdf(nws_mean + bias, sigma)

    snaps = list(iter_snapshots(city, DATE))
    open_keys: set[tuple[str, str]] = set()  # (venue, bracket_id) currently open
    city_pnl = 0.0

    for entry_idx, (ts, venues) in enumerate(snaps):
      for venue in ("kalshi", "polymarket"):
        for b in venues.get(venue, []):
          degrees = parse_bracket_degrees(b, venue)
          if not degrees:
            continue
          if cfg.exclude_tail_brackets:
            label = b.get("subtitle", "") if venue == "kalshi" else b.get("question", "")
            if any(kw in label.lower() for kw in (" or below", " or above", " or higher", " or less")):
              continue
          ask, depth = best_ask_and_depth(b)
          if ask is None or ask <= 0:
            continue
          if not (cfg.ask_min <= ask <= cfg.ask_max):
            continue
          if depth < cfg.min_depth_at_best:
            continue
          fv = fair_value(pdf, degrees)
          if fv < cfg.fair_value_min:
            continue
          if cfg.central_bracket_only:
            # Only trade if the (rounded) NWS forecast mean falls in this bracket
            if int(round(nws_mean + bias)) not in degrees:
              continue
          edge = fv - ask
          if edge < cfg.edge_threshold:
            continue
          # Already open?
          bracket_id = (b.get("ticker") if venue == "kalshi"
                        else b.get("token_id"))
          key = (venue, bracket_id or "")
          if key in open_keys:
            continue

          # Open
          pos = open_position(city, DATE, venue, b, side="yes_long",
                              size=cfg.size, entry_ts=ts)
          if pos is None:
            continue
          pos.bracket_degrees = degrees
          open_keys.add(key)

          # Future snapshots for this venue
          future = [(t, v.get(venue, [])) for t, v in snaps[entry_idx + 1:]]

          target = cfg.target_frac_of_edge * edge  # $/contract
          exit_res = simulate_exit(
            pos, future,
            target_pnl_per_contract=target,
            time_stop_minutes=cfg.time_stop_minutes,
            fee_per_contract=FEE,
          )
          # Once exited, free the slot
          open_keys.discard(key)

          trades.append({
            "city": city, "venue": venue, "bracket_id": bracket_id,
            "label": pos.bracket_label, "degrees": degrees,
            "entry_ts": ts, "exit_ts": exit_res.exit_ts,
            "entry_price": pos.entry_price,
            "exit_price": exit_res.exit_price,
            "size": cfg.size, "edge_at_entry": edge,
            "fair_value": fv, "ask_at_entry": ask,
            "exit_reason": exit_res.exit_reason,
            "realized_pnl": exit_res.realized_pnl,
          })
          exit_reasons[exit_res.exit_reason] = exit_reasons.get(exit_res.exit_reason, 0) + 1
          city_pnl += exit_res.realized_pnl

    per_city_pnl[city] = round(city_pnl, 4)

  # Compute metrics
  pnls = [t["realized_pnl"] for t in trades]
  n = len(trades)
  n_winners = sum(1 for p in pnls if p > 0)
  total_pnl = sum(pnls)
  if n > 0:
    avg_pnl = total_pnl / n
    median_pnl = statistics.median(pnls)
    if n > 1:
      sd = statistics.stdev(pnls)
      sharpe = avg_pnl / sd if sd > 0 else None
    else:
      sharpe = None
    # Approximate drawdown over the chronological sequence
    sorted_trades = sorted(trades, key=lambda t: t["entry_ts"])
    cum = 0.0
    peak = 0.0
    dd = 0.0
    for t in sorted_trades:
      cum += t["realized_pnl"]
      peak = max(peak, cum)
      dd = min(dd, cum - peak)
  else:
    avg_pnl = median_pnl = sharpe = None
    dd = 0.0

  return {
    "version": cfg.version,
    "n_trades": n,
    "n_winners": n_winners,
    "win_rate": round(n_winners / n, 4) if n else None,
    "total_pnl_dollars": round(total_pnl, 4),
    "avg_pnl_per_trade": round(avg_pnl, 4) if avg_pnl is not None else None,
    "median_pnl_per_trade": round(median_pnl, 4) if median_pnl is not None else None,
    "sharpe_per_trade": round(sharpe, 4) if sharpe is not None else None,
    "max_drawdown_dollars": round(dd, 4),
    "settled_n": 0,
    "settled_pnl_dollars": 0.0,
    "exit_reason_counts": exit_reasons,
    "per_city_pnl": per_city_pnl,
    "config": {k: v for k, v in asdict(cfg).items()},
    "notes": cfg.notes,
    "_trades": trades,  # kept for diagnostics; stripped before metrics.json
  }


def main() -> None:
  calibration = json.loads(CAL_PATH.read_text())

  # v1 — naive baseline using calibrated per-city sigma + bias. No price filter,
  # no tail exclusion. We expect this to be a disaster because our wide sigma
  # makes us over-value tail brackets.
  v1 = StrategyConfig(
    version="v1",
    edge_threshold=0.05,
    min_depth_at_best=50,
    size=100,
    target_frac_of_edge=0.5,
    time_stop_minutes=None,
    ask_min=0.0,
    ask_max=1.0,
    notes="naive: 5c edge, 50-depth, no filters, hold-to-EOD",
  )
  # v2 — informed by v1's blow-up: the tail brackets at ask < 0.05 caused
  # most losses (88% of losers had ask <= 5c). Add price band, tail exclusion,
  # bigger edge threshold and a time stop so dead positions don't decay all day.
  v2 = StrategyConfig(
    version="v2",
    edge_threshold=0.07,
    min_depth_at_best=100,
    size=100,
    target_frac_of_edge=0.4,
    time_stop_minutes=60,
    ask_min=0.10,
    ask_max=0.90,
    exclude_tail_brackets=True,
    notes=("v1 fix: tail brackets excluded, ask in [0.10,0.90], 60-min stop, "
           "depth>=100, edge>=7c"),
  )
  # v3 — informed by v2: still losing because the calibrated sigma (2.2-3.4)
  # is too WIDE -- the market is more confident than our PDF says. Override
  # sigma to 1.5 globally and require fair_value >= 0.20 (high-conviction
  # brackets only).
  v3 = StrategyConfig(
    version="v3",
    edge_threshold=0.07,
    min_depth_at_best=100,
    size=100,
    target_frac_of_edge=0.4,
    time_stop_minutes=60,
    ask_min=0.10,
    ask_max=0.90,
    exclude_tail_brackets=True,
    sigma_override=1.5,
    fair_value_min=0.20,
    notes=("sigma=1.5, fv>=0.20 (only buy brackets we think are 20%+ likely)"),
  )
  # v4 — last attempt: trade ONLY the bracket the forecast mean lives in
  # ("central bracket only"). If our forecast is right and that bracket is
  # priced below fair value, this is the cleanest possible trade.
  v4 = StrategyConfig(
    version="v4",
    edge_threshold=0.05,
    min_depth_at_best=50,
    size=100,
    target_frac_of_edge=0.4,
    time_stop_minutes=60,
    ask_min=0.10,
    ask_max=0.90,
    exclude_tail_brackets=True,
    sigma_override=1.5,
    central_bracket_only=True,
    notes="central-bracket-only: trade only the bracket containing the forecast mean",
  )

  iteration_notes = {
    "v1": ("v1 baseline: blew up. 88% of losers had ask<=5c (tail-bracket "
           "flood); calibrated sigma over-weights tails relative to market."),
    "v2": ("v2: tail brackets excluded + price band + time stop -> 99 trades "
           "but still losing. Avg loss per trade larger than v1 because each "
           "loser holds longer before time stop."),
    "v3": ("v3: tighter sigma=1.5 + fv>=0.20 (high-conviction brackets only). "
           "Still loses; the brackets where market disagrees with us most "
           "are the brackets where the market is right."),
    "v4": ("v4: central-bracket-only. 0 Miami trades (well-priced); biggest "
           "losses on LA/SF where NWS forecast was off by 2F. Confirms the "
           "static-forecast staleness failure mode."),
  }

  results = []
  for cfg in (v1, v2, v3, v4):
    r = run_strategy(cfg, calibration)
    r["notes"] = iteration_notes.get(r["version"], r["notes"])
    print(f"\n=== {r['version']} ===")
    print(f"  n_trades={r['n_trades']} winners={r['n_winners']} "
          f"win_rate={r['win_rate']} total_pnl=${r['total_pnl_dollars']}")
    print(f"  avg/trade=${r['avg_pnl_per_trade']} median=${r['median_pnl_per_trade']} "
          f"sharpe={r['sharpe_per_trade']} max_dd=${r['max_drawdown_dollars']}")
    print(f"  exits: {r['exit_reason_counts']}")
    print(f"  per-city pnl: {r['per_city_pnl']}")
    results.append(r)

  # Strip _trades for metrics.json; keep separately
  metrics = {
    "strategy_name": "calibrated_gaussian",
    "iterations": [{k: v for k, v in r.items() if k != "_trades"} for r in results],
  }
  out = Path(__file__).resolve().parent / "metrics.json"
  out.write_text(json.dumps(metrics, indent=2))
  print(f"\nWrote {out}")

  # Persist trades for inspection
  trades_path = Path(__file__).resolve().parent / "trades.json"
  trades_path.write_text(json.dumps(
    {r["version"]: r["_trades"] for r in results}, indent=2, default=str))
  print(f"Wrote {trades_path}")


if __name__ == "__main__":
  main()
