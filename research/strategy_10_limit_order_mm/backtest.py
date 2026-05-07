"""
Strategy 10: Limit-order market making (passive spread capture).

Round 1 finding: round-trip spread + fees ≈ $0.02/contract dwarfs
short-horizon directional alpha. Every prior strategy CROSSED the spread.
This strategy POSTS limits inside the spread to EARN it.

Mechanics
---------
At snapshot t, we post a YES bid at (best_yes_bid + 0.01) on a chosen
bracket. We then walk forward snapshot-by-snapshot and check whether
our resting bid would have been filled, using one of three fill models:

  - aggressive: filled if any subsequent best_yes_ask <= our_bid
                (treats our presence in the queue as instantaneous and
                priority-blind; upper-bound).
  - cautious:   filled only if any subsequent best_yes_ask < our_bid
                (a real seller had to cross strictly past us).
  - volume:     filled only if total ask-size at-or-below our_bid
                drops snapshot-to-snapshot (proxy for actual prints).

After fill, we post a YES ask at (best_yes_ask - 0.01) and apply the
same fill-model logic on the bid ladder. P&L is (sell - buy) * size
minus fees.

Versions
--------
v1 — aggressive fill, no cancel-on-move, all liquid brackets.
v2 — cautious fill, cancel-on-move (mid moves > 2¢ against us → cancel),
     same liquid-bracket filter.
v3 — volume-aware fill, cancel-on-move, plus inventory limit (≤ K
     concurrent positions per (city, venue, bracket)) and stricter
     bracket-selection filter (mid in [0.10, 0.90], spread ≤ 3¢, depth ≥ 30
     each side, skip last 60 min of trading window).

Caveats acknowledged
--------------------
- 5-min snapshots cannot model queue priority. Our 1¢-inside bid is
  assumed to sit at the FRONT of its price level, which is optimistic.
- Real fills happen at sub-second granularity; one snapshot's quote can
  hide many crossing prints. Fill rates here are upper-bound proxies.
- We use a simplified fee schedule: $0.005/side on Kalshi; $0/side on
  Polymarket (verifying maker rebates is out of scope, but we sweep
  both).
"""
from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.loaders import iter_snapshots  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
CITIES = ["Miami", "LA", "Austin", "San Francisco", "Seattle"]
DATE = "2026-05-06"

# Fees (per contract per side)
KALSHI_FEE = 0.005
POLY_FEE = 0.0  # Polymarket maker is roughly $0; sweep with KALSHI_FEE for sensitivity


# ── Helpers ──────────────────────────────────────────────────────────────

def best_quotes(b: dict) -> tuple[Optional[float], Optional[float],
                                  Optional[float], Optional[float]]:
  """Return (yes_bid, yes_ask, yes_bid_size, yes_ask_size). Use the
  ladder's actual best level when available — the cached `best_yes_*`
  field can be stale or null on Polymarket."""
  bid_ladder = b.get("yes_bid_ladder") or []
  ask_ladder = b.get("yes_ask_ladder") or []
  yes_bid = None
  yes_bid_sz = 0.0
  if bid_ladder:
    levels = sorted(((float(p), float(q)) for p, q in bid_ladder if q and q > 0),
                    key=lambda pq: -pq[0])
    if levels:
      yes_bid, yes_bid_sz = levels[0]
  yes_ask = None
  yes_ask_sz = 0.0
  if ask_ladder:
    levels = sorted(((float(p), float(q)) for p, q in ask_ladder if q and q > 0))
    if levels:
      yes_ask, yes_ask_sz = levels[0]
  return yes_bid, yes_ask, yes_bid_sz, yes_ask_sz


def total_size_at_or_below(ladder: list, price: float, eps: float = 1e-9) -> float:
  """Total displayed size at ask prices <= `price` (cents-tolerant)."""
  if not ladder:
    return 0.0
  return sum(float(q) for p, q in ladder if q and float(q) > 0 and float(p) <= price + eps)


def total_size_at_or_above(ladder: list, price: float, eps: float = 1e-9) -> float:
  if not ladder:
    return 0.0
  return sum(float(q) for p, q in ladder if q and float(q) > 0 and float(p) >= price - eps)


def bracket_id(b: dict, venue: str) -> str:
  return b.get("ticker", "") if venue == "kalshi" else b.get("token_id", "")


def label(b: dict, venue: str) -> str:
  return b.get("subtitle", "") if venue == "kalshi" else b.get("question", "")


def parse_ts(ts: str) -> datetime:
  return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def fee_for_venue(venue: str) -> float:
  return KALSHI_FEE if venue == "kalshi" else POLY_FEE


# ── Bracket-selection filter ─────────────────────────────────────────────

def is_liquid_bracket(b: dict, *, version: str,
                       recent_mids: Optional[list[float]] = None) -> bool:
  yes_bid, yes_ask, bid_sz, ask_sz = best_quotes(b)
  if yes_bid is None or yes_ask is None:
    return False
  spread = yes_ask - yes_bid
  if spread <= 0:
    return False
  mid = 0.5 * (yes_bid + yes_ask)
  if version == "v1":
    # Loose: spread <= 5¢, mid not pinned to penny edges
    if spread > 0.05:
      return False
    if mid < 0.05 or mid > 0.95:
      return False
    if bid_sz < 5 or ask_sz < 5:
      return False
    return True
  if version == "v2":
    if spread > 0.04:
      return False
    if mid < 0.08 or mid > 0.92:
      return False
    if bid_sz < 10 or ask_sz < 10:
      return False
    return True
  if version == "v3":
    if spread > 0.03:
      return False
    if mid < 0.10 or mid > 0.90:
      return False
    if bid_sz < 30 or ask_sz < 30:
      return False
    return True
  # v4: v3 strict + drift filter — don't post on brackets where mid is
  # moving rapidly (proxy for "informed flow incoming"). We require that
  # the recent N mids are within `drift_cap` cents of the current mid.
  if version == "v4":
    if spread > 0.03:
      return False
    if mid < 0.10 or mid > 0.90:
      return False
    if bid_sz < 30 or ask_sz < 30:
      return False
    if recent_mids is not None and len(recent_mids) >= 3:
      # range over last 3 snapshots ≤ 3¢
      window = recent_mids[-3:]
      if max(window) - min(window) > 0.03 + 1e-9:
        return False
    return True
  return False


# ── Trade record ─────────────────────────────────────────────────────────

@dataclass
class MMTrade:
  city: str
  venue: str
  bracket_id: str
  bracket_label: str
  size: int
  entry_post_ts: str
  entry_post_price: float       # bid we posted
  entry_fill_ts: Optional[str]
  entry_fill_price: Optional[float]
  entry_canceled: bool
  exit_post_ts: Optional[str]
  exit_post_price: Optional[float]   # ask we posted
  exit_fill_ts: Optional[str]
  exit_fill_price: Optional[float]
  exit_reason: str              # 'filled' | 'canceled_pre_entry' | 'canceled_post_entry' | 'unfilled_eow' | 'no_exit_eow'
  pnl: float


# ── Fill models ──────────────────────────────────────────────────────────

def buy_filled(future_b: dict, my_bid: float, prev_b: Optional[dict],
               model: str) -> bool:
  """Did the buy limit at price `my_bid` get filled given the future
  snapshot? `prev_b` is the previous snapshot's bracket state."""
  yes_bid, yes_ask, _, _ = best_quotes(future_b)
  ask_ladder = future_b.get("yes_ask_ladder") or []
  if model == "aggressive":
    if yes_ask is not None and yes_ask <= my_bid + 1e-9:
      return True
    return False
  if model == "cautious":
    if yes_ask is not None and yes_ask < my_bid - 1e-9:
      return True
    return False
  # volume model: did displayed ask size at-or-below my_bid SHRINK
  # (and ask price-line came down to/below my_bid, i.e. someone hit it)?
  if prev_b is None:
    return False
  prev_size = total_size_at_or_below(prev_b.get("yes_ask_ladder") or [], my_bid)
  cur_size = total_size_at_or_below(ask_ladder, my_bid)
  # Only count it as a fill if someone actually traded into our level:
  # ask side at-or-below us decreased AND there was at least some size
  # there in the prior snapshot (otherwise nothing to take). We're a
  # bit lenient: also accept ask price stepping down to <= our bid as a
  # signal of a real seller (pulls cautious-model logic in too).
  if prev_size > 0 and cur_size < prev_size - 1e-9:
    return True
  if yes_ask is not None and yes_ask < my_bid - 1e-9:
    return True
  return False


def sell_filled(future_b: dict, my_ask: float, prev_b: Optional[dict],
                model: str) -> bool:
  """Did the sell limit at `my_ask` fill?"""
  yes_bid, yes_ask, _, _ = best_quotes(future_b)
  bid_ladder = future_b.get("yes_bid_ladder") or []
  if model == "aggressive":
    if yes_bid is not None and yes_bid >= my_ask - 1e-9:
      return True
    return False
  if model == "cautious":
    if yes_bid is not None and yes_bid > my_ask + 1e-9:
      return True
    return False
  # volume
  if prev_b is None:
    return False
  prev_size = total_size_at_or_above(prev_b.get("yes_bid_ladder") or [], my_ask)
  cur_size = total_size_at_or_above(bid_ladder, my_ask)
  if prev_size > 0 and cur_size < prev_size - 1e-9:
    return True
  if yes_bid is not None and yes_bid > my_ask + 1e-9:
    return True
  return False


# ── Cancel-on-move logic ─────────────────────────────────────────────────

def cancel_buy(cur_b: dict, entry_mid: float, threshold: float) -> bool:
  yes_bid, yes_ask, _, _ = best_quotes(cur_b)
  if yes_bid is None or yes_ask is None:
    return True  # liquidity vanished — pull the order
  cur_mid = 0.5 * (yes_bid + yes_ask)
  # We are LONG-ing, so adverse = mid drops. Cancel if mid moved down by
  # more than `threshold` cents.
  if entry_mid - cur_mid > threshold + 1e-9:
    return True
  return False


def cancel_sell(cur_b: dict, fill_mid: float, threshold: float) -> bool:
  yes_bid, yes_ask, _, _ = best_quotes(cur_b)
  if yes_bid is None or yes_ask is None:
    return True
  cur_mid = 0.5 * (yes_bid + yes_ask)
  # We are LONG and selling. Adverse = mid drops below the level we filled at.
  if fill_mid - cur_mid > threshold + 1e-9:
    return True
  return False


# ── Per-bracket walk ─────────────────────────────────────────────────────

def run_bracket_mm(
  city: str, venue: str, b_id: str, b_label: str,
  snaps_by_ts: list[tuple[str, dict]],   # (ts, bracket) for THIS bracket only, ordered
  *,
  version: str,
  fill_model: str,
  cancel_threshold_buy: Optional[float],   # in dollars; None disables
  cancel_threshold_sell: Optional[float],
  size: int,
  fee_per_side: float,
  max_concurrent: int = 1,
  active_window_end: Optional[datetime] = None,  # skip new opens after this
) -> list[MMTrade]:
  """Walk through one bracket's snapshot timeline and run the MM loop.
  At any time we hold up to `max_concurrent` open positions. On each
  snapshot:
    - If we have a live BUY limit, decide fill / cancel.
    - If we have a filled position with a live SELL limit, decide fill / cancel.
    - If we have no live order and capacity remains, post a new BUY.
  """
  trades: list[MMTrade] = []
  # State: list of dicts representing positions in lifecycle stages
  #   stage = 'buy_resting' | 'sell_resting'
  # Each position has: post_price, post_ts, fill_price, fill_ts, fill_mid
  open_states: list[dict] = []

  prev_b: Optional[dict] = None
  mid_history: list[float] = []

  for idx, (ts, b) in enumerate(snaps_by_ts):
    cur_dt = parse_ts(ts)
    yb_now, ya_now, _, _ = best_quotes(b)
    if yb_now is not None and ya_now is not None:
      mid_history.append(0.5 * (yb_now + ya_now))
    # Process states in order; mutations happen below.
    next_open: list[dict] = []
    for st in open_states:
      if st["stage"] == "buy_resting":
        # Check fill first
        if buy_filled(b, st["post_price"], prev_b, fill_model):
          st["fill_ts"] = ts
          st["fill_price"] = st["post_price"]
          yes_bid, yes_ask, _, _ = best_quotes(b)
          st["fill_mid"] = 0.5 * (yes_bid + yes_ask) if (yes_bid and yes_ask) else st["post_price"]
          # Post an exit at best_ask - 1c (using current snapshot)
          if yes_ask is not None and yes_bid is not None:
            exit_post = yes_ask - 0.01
            # Don't sell below buy + 2c floor or at <= buy
            if exit_post <= st["post_price"] + 0.005:
              # Post at buy + 2c; if spread is too tight, skip selling and book a flat exit later
              exit_post = st["post_price"] + 0.02
            st["exit_post_price"] = round(exit_post, 4)
            st["exit_post_ts"] = ts
            st["stage"] = "sell_resting"
            next_open.append(st)
          else:
            # No quotes — we hold, will try to exit next snapshot
            st["stage"] = "sell_pending"
            next_open.append(st)
          continue
        # Cancel?
        if cancel_threshold_buy is not None and cancel_buy(b, st["entry_mid"], cancel_threshold_buy):
          # Trade is canceled, never filled; record as a no-op (no trade created)
          # But we DO consume the slot? We free it.
          continue
        # Else stays resting
        next_open.append(st)
      elif st["stage"] == "sell_pending":
        # Try to post on this snapshot
        yes_bid, yes_ask, _, _ = best_quotes(b)
        if yes_ask is not None and yes_bid is not None:
          exit_post = yes_ask - 0.01
          if exit_post <= st["fill_price"] + 0.005:
            exit_post = st["fill_price"] + 0.02
          st["exit_post_price"] = round(exit_post, 4)
          st["exit_post_ts"] = ts
          st["stage"] = "sell_resting"
        next_open.append(st)
      elif st["stage"] == "sell_resting":
        # Fill check on the sell
        if sell_filled(b, st["exit_post_price"], prev_b, fill_model):
          st["exit_fill_ts"] = ts
          st["exit_fill_price"] = st["exit_post_price"]
          st["exit_reason"] = "filled"
          # Realize trade
          gross = (st["exit_fill_price"] - st["fill_price"]) * size
          fees = 2.0 * fee_per_side * size
          trades.append(MMTrade(
            city=city, venue=venue, bracket_id=b_id, bracket_label=b_label,
            size=size,
            entry_post_ts=st["post_ts"], entry_post_price=st["post_price"],
            entry_fill_ts=st["fill_ts"], entry_fill_price=st["fill_price"],
            entry_canceled=False,
            exit_post_ts=st["exit_post_ts"], exit_post_price=st["exit_post_price"],
            exit_fill_ts=st["exit_fill_ts"], exit_fill_price=st["exit_fill_price"],
            exit_reason="filled",
            pnl=gross - fees,
          ))
          continue
        # Re-quote logic (don't flatten — flattening guarantees a loss
        # since we'd be hitting a bid below our buy). Just slide our ask
        # to be 1c inside the new ask whenever it changes.
        if cancel_threshold_sell is not None and cancel_sell(b, st["fill_mid"], cancel_threshold_sell):
          yes_bid, yes_ask, _, _ = best_quotes(b)
          if yes_ask is not None and yes_bid is not None:
            new_post = yes_ask - 0.01
            # Floor: never sell below entry + 1c (for fee + edge)
            min_acceptable = st["fill_price"] + 0.01
            if new_post < min_acceptable:
              new_post = min_acceptable
            st["exit_post_price"] = round(new_post, 4)
            st["exit_post_ts"] = ts
            st["fill_mid"] = 0.5 * (yes_bid + yes_ask)
          # else: liquidity vanished, keep resting
          next_open.append(st)
          continue
        # On every snapshot, also try to TIGHTEN our ask if a better
        # ask is now resting (we want to be 1c inside the best ask).
        yes_bid, yes_ask, _, _ = best_quotes(b)
        if yes_ask is not None and yes_bid is not None:
          target = yes_ask - 0.01
          # Don't move down past our minimum-acceptable
          min_acceptable = st["fill_price"] + 0.01
          if target < min_acceptable:
            target = min_acceptable
          # Only move ask DOWN (improves chance of fill) when it's safe
          if target < st["exit_post_price"] - 1e-9:
            st["exit_post_price"] = round(target, 4)
            st["exit_post_ts"] = ts
        next_open.append(st)
      else:
        next_open.append(st)

    open_states = next_open

    # Maybe post a new BUY
    if (active_window_end is None or cur_dt < active_window_end) and \
       len(open_states) < max_concurrent and \
       is_liquid_bracket(b, version=version, recent_mids=mid_history):
      yes_bid, yes_ask, _, _ = best_quotes(b)
      if yes_bid is not None and yes_ask is not None:
        post_price = round(yes_bid + 0.01, 4)
        # Don't post equal-or-above ask
        if post_price < yes_ask - 1e-9:
          mid = 0.5 * (yes_bid + yes_ask)
          open_states.append({
            "stage": "buy_resting",
            "post_price": post_price,
            "post_ts": ts,
            "entry_mid": mid,
            "fill_ts": None,
            "fill_price": None,
            "fill_mid": None,
            "exit_post_price": None,
            "exit_post_ts": None,
            "exit_fill_ts": None,
            "exit_fill_price": None,
            "exit_reason": None,
          })

    prev_b = b

  # End of the trading window: close out anything still open
  # Find the last snapshot with a usable bid (walk backward).
  last_bid = None
  last_bid_ts = None
  for ts_b, b_b in reversed(snaps_by_ts):
    yb, _, _, _ = best_quotes(b_b)
    if yb is not None:
      last_bid = yb
      last_bid_ts = ts_b
      break

  for st in open_states:
    if st["stage"] == "buy_resting":
      # never filled — record as canceled-pre-entry, no trade
      continue
    # We have a filled position. Mark it out at last seen bid (i.e.,
    # cross the spread to flatten at end of the trading window).
    if last_bid is None:
      # No usable bid was ever quoted on this bracket. We *would* have
      # had to either (a) hold to settle ($1 or $0) or (b) accept a
      # zero-value flatten. We record this as unfilled_eow and conserve
      # the buy cost — a worst-case mark to zero.
      gross = -st["fill_price"] * size
      fees = 2.0 * fee_per_side * size
      trades.append(MMTrade(
        city=city, venue=venue, bracket_id=b_id, bracket_label=b_label,
        size=size,
        entry_post_ts=st["post_ts"], entry_post_price=st["post_price"],
        entry_fill_ts=st["fill_ts"], entry_fill_price=st["fill_price"],
        entry_canceled=False,
        exit_post_ts=st["exit_post_ts"], exit_post_price=st["exit_post_price"],
        exit_fill_ts=None, exit_fill_price=None,
        exit_reason="unfilled_eow_no_bid",
        pnl=gross - fees,
      ))
      continue
    gross = (last_bid - st["fill_price"]) * size
    fees = 2.0 * fee_per_side * size
    trades.append(MMTrade(
      city=city, venue=venue, bracket_id=b_id, bracket_label=b_label,
      size=size,
      entry_post_ts=st["post_ts"], entry_post_price=st["post_price"],
      entry_fill_ts=st["fill_ts"], entry_fill_price=st["fill_price"],
      entry_canceled=False,
      exit_post_ts=st["exit_post_ts"], exit_post_price=st["exit_post_price"],
      exit_fill_ts=last_bid_ts, exit_fill_price=last_bid,
      exit_reason="forced_close_eow",
      pnl=gross - fees,
    ))
  return trades


# ── Driver ───────────────────────────────────────────────────────────────

def reorganize_by_bracket(
  city: str, date: str,
) -> dict[tuple[str, str], list[tuple[str, dict]]]:
  """Build {(venue, bracket_id): [(ts, bracket_dict), ...]} sorted by ts."""
  out: dict[tuple[str, str], list[tuple[str, dict]]] = {}
  for ts, venues in iter_snapshots(city, date):
    for venue, brackets in venues.items():
      for b in brackets:
        bid = bracket_id(b, venue)
        if not bid:
          continue
        out.setdefault((venue, bid), []).append((ts, b))
  for k in out:
    out[k].sort(key=lambda r: r[0])
  return out


def cutoff_time(any_ts: str, hours_before_end: int = 1) -> datetime:
  """Construct a cutoff: `hours_before_end` hours before end-of-stream.
  We compute end from the data, but since we don't know it here, we fall
  back to 1 hour before max ts of the bracket — caller can pass a global
  cutoff."""
  return parse_ts(any_ts)


def trading_window_end(snaps_by_ts: list[tuple[str, dict]],
                       skip_minutes_at_end: int = 60) -> datetime:
  if not snaps_by_ts:
    return datetime.max.replace(tzinfo=None)
  end_dt = parse_ts(snaps_by_ts[-1][0])
  return end_dt - timedelta(minutes=skip_minutes_at_end)


def run_version(version: str, *, fill_model: str,
                cancel_threshold_buy: Optional[float],
                cancel_threshold_sell: Optional[float],
                max_concurrent: int = 1,
                size: int = 10,
                skip_end_minutes: int = 60) -> tuple[list[MMTrade], dict]:
  trades: list[MMTrade] = []
  posts_total = 0
  posts_filled = 0
  per_bracket_stats = []
  for city in CITIES:
    by_bracket = reorganize_by_bracket(city, DATE)
    for (venue, b_id), snaps in by_bracket.items():
      if len(snaps) < 5:
        continue
      b_label = label(snaps[0][1], venue)
      window_end = trading_window_end(snaps, skip_minutes_at_end=skip_end_minutes)
      bracket_trades = run_bracket_mm(
        city=city, venue=venue, b_id=b_id, b_label=b_label,
        snaps_by_ts=snaps,
        version=version,
        fill_model=fill_model,
        cancel_threshold_buy=cancel_threshold_buy,
        cancel_threshold_sell=cancel_threshold_sell,
        size=size,
        fee_per_side=fee_for_venue(venue),
        max_concurrent=max_concurrent,
        active_window_end=window_end,
      )
      trades.extend(bracket_trades)
      # Count posts/fills per bracket. We cheat-count here from the
      # trade objects; canceled-pre-fill posts aren't recorded as trades,
      # so we'd undercount the denominator. Instead, do a simple replay
      # to count posts: every snapshot where filter passes is a "post
      # opportunity" (capped at 1 outstanding). We'll just track it
      # alongside the inner walk in v_meta below.
      posts_filled += sum(1 for t in bracket_trades if t.entry_fill_ts is not None)
  # We also want a posts_total — replay quickly to count.
  posts_total = count_posts(version=version, size=size, max_concurrent=max_concurrent,
                             skip_end_minutes=skip_end_minutes)
  return trades, {"posts_total": posts_total, "posts_filled_entry": posts_filled}


def count_posts(*, version: str, size: int, max_concurrent: int,
                skip_end_minutes: int) -> int:
  """Lightweight pass: how many BUY limits would the strategy have posted
  if we never got filled (i.e., the maximum opportunity count)?
  We approximate by counting snapshots where the bracket-liquidity filter
  passes and we have capacity (always — we never fill in this counter)."""
  posts = 0
  for city in CITIES:
    by_bracket = reorganize_by_bracket(city, DATE)
    for (venue, b_id), snaps in by_bracket.items():
      if len(snaps) < 5:
        continue
      window_end = trading_window_end(snaps, skip_minutes_at_end=skip_end_minutes)
      # Cap concurrent posts at max_concurrent — under "never fill" we'd
      # always have max_concurrent outstanding once we ramp up. So count
      # = (#snapshots filter passes) - (max_concurrent-1) saturating to 0.
      count = 0
      mid_hist: list[float] = []
      for ts, b in snaps:
        yb, ya, _, _ = best_quotes(b)
        if yb is not None and ya is not None:
          mid_hist.append(0.5 * (yb + ya))
        if parse_ts(ts) >= window_end:
          continue
        if is_liquid_bracket(b, version=version, recent_mids=mid_hist):
          count += 1
      # Each bracket can have at most 1 post per snapshot in our model.
      posts += count
  return posts


# ── Metrics ──────────────────────────────────────────────────────────────

def compute_metrics(trades: list[MMTrade], extra: dict) -> dict:
  n = len(trades)
  if n == 0:
    return {
      "n_trades": 0, "n_winners": 0, "win_rate": 0.0, "total_pnl_dollars": 0.0,
      "avg_pnl_per_trade": 0.0, "median_pnl_per_trade": 0.0, "sharpe_per_trade": 0.0,
      "max_drawdown_dollars": 0.0,
      "settled_n": 0, "settled_pnl_dollars": 0.0,
      "exit_reason_counts": {},
      **extra,
    }
  pnls = [t.pnl for t in trades]
  pnls_sorted = sorted(pnls)
  median = pnls_sorted[n // 2] if n % 2 == 1 else 0.5 * (pnls_sorted[n // 2 - 1] + pnls_sorted[n // 2])
  total = sum(pnls)
  avg = total / n
  if n > 1:
    var = sum((p - avg) ** 2 for p in pnls) / (n - 1)
    sd = math.sqrt(var)
    sharpe = avg / sd if sd > 0 else 0.0
  else:
    sharpe = 0.0
  # Max drawdown on cumulative pnl curve (in trade order)
  cum = 0.0
  peak = 0.0
  dd = 0.0
  for p in pnls:
    cum += p
    if cum > peak:
      peak = cum
    if peak - cum > dd:
      dd = peak - cum
  reasons: dict[str, int] = {}
  for t in trades:
    reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
  winners = sum(1 for p in pnls if p > 0)
  posts_total = extra.get("posts_total", 0)
  fill_rate = (extra.get("posts_filled_entry", 0) / posts_total) if posts_total else 0.0
  # Subset: filled round-trips only (true spread-capture economics)
  filled_pnls = [t.pnl for t in trades if t.exit_reason == "filled"]
  eow_pnls = [t.pnl for t in trades if t.exit_reason in ("forced_close_eow", "unfilled_eow_no_bid")]
  filled_avg = sum(filled_pnls) / len(filled_pnls) if filled_pnls else 0.0
  eow_avg = sum(eow_pnls) / len(eow_pnls) if eow_pnls else 0.0
  return {
    "n_trades": n,
    "n_winners": winners,
    "win_rate": round(winners / n, 4),
    "total_pnl_dollars": round(total, 4),
    "avg_pnl_per_trade": round(avg, 6),
    "median_pnl_per_trade": round(median, 6),
    "sharpe_per_trade": round(sharpe, 4),
    "max_drawdown_dollars": round(-dd, 4),
    "settled_n": 0,
    "settled_pnl_dollars": 0.0,
    "exit_reason_counts": reasons,
    "posts_total": posts_total,
    "posts_filled_entry": extra.get("posts_filled_entry", 0),
    "entry_fill_rate": round(fill_rate, 4),
    # Diagnostic split between completed round trips (= what spread
    # capture earns when it works) and end-of-window forced flattens
    # (= adverse selection tax).
    "n_filled_round_trips": len(filled_pnls),
    "filled_round_trips_pnl": round(sum(filled_pnls), 4),
    "filled_round_trips_avg_pnl": round(filled_avg, 6),
    "n_eow_forced_close": len(eow_pnls),
    "eow_forced_close_pnl": round(sum(eow_pnls), 4),
    "eow_forced_close_avg_pnl": round(eow_avg, 6),
  }


def write_trades_csv(trades: list[MMTrade], path: Path) -> None:
  fieldnames = [
    "city", "venue", "bracket_id", "bracket_label", "size",
    "entry_post_ts", "entry_post_price",
    "entry_fill_ts", "entry_fill_price",
    "entry_canceled",
    "exit_post_ts", "exit_post_price",
    "exit_fill_ts", "exit_fill_price",
    "exit_reason", "pnl",
  ]
  with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for t in trades:
      w.writerow(asdict(t))


# ── Main ─────────────────────────────────────────────────────────────────

def main():
  iterations = []

  print("=" * 70)
  print("v1: aggressive fill, no cancel, loose liquidity filter")
  print("=" * 70)
  v1_trades, v1_extra = run_version(
    "v1", fill_model="aggressive",
    cancel_threshold_buy=None, cancel_threshold_sell=None,
    max_concurrent=1, size=10, skip_end_minutes=60,
  )
  v1_m = compute_metrics(v1_trades, v1_extra)
  v1_m["version"] = "v1"
  v1_m["notes"] = ("Aggressive fill model (best_yes_ask <= my_bid). No "
                   "cancel-on-move. Loose liquidity filter. Upper bound: "
                   "this overstates fills because real queue priority "
                   "would block us behind existing orders.")
  iterations.append(v1_m)
  write_trades_csv(v1_trades, OUT_DIR / "trades_v1.csv")
  print(f"  trades={v1_m['n_trades']} winners={v1_m['n_winners']} "
        f"total=${v1_m['total_pnl_dollars']:.2f} sharpe={v1_m['sharpe_per_trade']:.3f} "
        f"fill_rate={v1_m['entry_fill_rate']:.3%} posts={v1_m['posts_total']}")

  print()
  print("=" * 70)
  print("v2: cautious fill + cancel-on-move (2c) + tighter filter")
  print("=" * 70)
  v2_trades, v2_extra = run_version(
    "v2", fill_model="cautious",
    cancel_threshold_buy=0.02, cancel_threshold_sell=0.02,
    max_concurrent=1, size=10, skip_end_minutes=60,
  )
  v2_m = compute_metrics(v2_trades, v2_extra)
  v2_m["version"] = "v2"
  v2_m["notes"] = ("Cautious fill model (best_yes_ask STRICTLY < my_bid). "
                   "Cancel-on-move at 2c adverse mid drift. Tighter liquidity "
                   "filter (spread <= 4c, mid 0.08-0.92, depth 10).")
  iterations.append(v2_m)
  write_trades_csv(v2_trades, OUT_DIR / "trades_v2.csv")
  print(f"  trades={v2_m['n_trades']} winners={v2_m['n_winners']} "
        f"total=${v2_m['total_pnl_dollars']:.2f} sharpe={v2_m['sharpe_per_trade']:.3f} "
        f"fill_rate={v2_m['entry_fill_rate']:.3%} posts={v2_m['posts_total']}")

  print()
  print("=" * 70)
  print("v3: volume-aware fill + cancel-on-move + strict filter + 1 concurrent")
  print("=" * 70)
  v3_trades, v3_extra = run_version(
    "v3", fill_model="volume",
    cancel_threshold_buy=0.02, cancel_threshold_sell=0.02,
    max_concurrent=1, size=10, skip_end_minutes=60,
  )
  v3_m = compute_metrics(v3_trades, v3_extra)
  v3_m["version"] = "v3"
  v3_m["notes"] = ("Volume-aware fill model: at-or-below-my-bid ask size "
                   "must shrink snap-to-snap. Cancel-on-move at 2c (cancel "
                   "buy; re-quote sell, no flatten). Strict filter "
                   "(spread <= 3c, mid 0.10-0.90, depth 30). 1 concurrent "
                   "per bracket.")
  iterations.append(v3_m)
  write_trades_csv(v3_trades, OUT_DIR / "trades_v3.csv")
  print(f"  trades={v3_m['n_trades']} winners={v3_m['n_winners']} "
        f"total=${v3_m['total_pnl_dollars']:.2f} sharpe={v3_m['sharpe_per_trade']:.3f} "
        f"fill_rate={v3_m['entry_fill_rate']:.3%} posts={v3_m['posts_total']}")

  print()
  print("=" * 70)
  print("v4: v3 + drift filter (skip brackets where mid is moving) + 90min EOW skip")
  print("=" * 70)
  v4_trades, v4_extra = run_version(
    "v4", fill_model="volume",
    cancel_threshold_buy=0.015, cancel_threshold_sell=0.02,
    max_concurrent=1, size=10, skip_end_minutes=90,
  )
  v4_m = compute_metrics(v4_trades, v4_extra)
  v4_m["version"] = "v4"
  v4_m["notes"] = ("v3 + drift filter: skip brackets where the last 3 mids "
                   "spanned > 3c (informed flow proxy). Tighter buy-cancel "
                   "at 1.5c. Skip the last 90 min of the trading window.")
  iterations.append(v4_m)
  write_trades_csv(v4_trades, OUT_DIR / "trades_v4.csv")
  print(f"  trades={v4_m['n_trades']} winners={v4_m['n_winners']} "
        f"total=${v4_m['total_pnl_dollars']:.2f} sharpe={v4_m['sharpe_per_trade']:.3f} "
        f"fill_rate={v4_m['entry_fill_rate']:.3%} posts={v4_m['posts_total']}")

  metrics = {
    "strategy_name": "limit_order_market_making",
    "iterations": iterations,
  }
  with open(OUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
  print()
  print(f"Wrote metrics.json + trades_v{{1,2,3}}.csv to {OUT_DIR}")


if __name__ == "__main__":
  main()
