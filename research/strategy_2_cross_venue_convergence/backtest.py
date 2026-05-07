"""
Cross-venue convergence / relative-value backtest.

Hypothesis: Kalshi and Polymarket sometimes disagree on the implied
probability of overlapping (city, date, bracket) outcomes. When the
disagreement is large enough to overcome bid-ask spread + fees, buy the
*cheap* side (lift its ask) and plan to sell when the gap converges to
half its entry size or smaller -- mark-to-market exit.

We never short. To bet "Polymarket says it's underpriced relative to
Kalshi" we BUY YES on Polymarket. To bet "Kalshi says it's underpriced",
we BUY YES on Kalshi. We do NOT buy NO synthetically; the snapshot data
only has YES books and the synthetic NO ladder transformation tends to
produce thin liquidity. Instead, when Polymarket's probability is HIGHER
than Kalshi's for a bracket, we treat that as "Kalshi YES is cheap" and
buy YES on Kalshi (entering on its ask). Conversely if Kalshi is higher,
we buy YES on Polymarket.

We then simulate exit by mark-to-market hits to bid:
- TARGET: gap (now) <= entry_gap / 2 -- i.e., the venues converged
  enough that selling the YES we just bought into its current bid clears
  a profit threshold.
- TIME STOP: 2 hours.
- END OF WINDOW: sell at last observable bid.

v1: bracket-vs-bracket on EXACTLY identical degree coverage only.
v2: per-degree implied PDF comparison; also computes signal on partial
    overlap brackets and applies a depth filter.
v3 (optional): tune gap threshold, exit target.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.loaders import (  # noqa: E402
  iter_snapshots,
  open_position,
  walk_ladder_buy,
  walk_ladder_sell,
  find_kalshi_bracket,
  find_polymarket_bracket,
)
from strategy.parsers import _parse_kalshi_range, _parse_polymarket_range  # noqa: E402

CITIES = ["Miami", "LA", "Austin", "San Francisco", "Seattle"]
DATE = "2026-05-06"
FEE_PER_CONTRACT = 0.005


# ── Helpers ─────────────────────────────────────────────────────────────

def _parse_dt(ts: str) -> datetime:
  return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _midprice(b: dict) -> Optional[float]:
  bid = b.get("best_yes_bid")
  ask = b.get("best_yes_ask")
  if bid is not None and ask is not None:
    return (bid + ask) / 2.0
  if bid is not None:
    return float(bid)
  if ask is not None:
    return float(ask)
  return None


def _degrees(b: dict, venue: str) -> list[int]:
  if venue == "kalshi":
    return _parse_kalshi_range(b.get("subtitle", ""))
  return _parse_polymarket_range(b.get("question", ""))


def _bracket_id(b: dict, venue: str) -> str:
  return b.get("ticker", "") if venue == "kalshi" else (b.get("token_id", "") or "")


# ── Bracket alignment ───────────────────────────────────────────────────

def find_aligned_brackets(
  k_bracks: list[dict], p_bracks: list[dict]
) -> list[tuple[dict, dict, list[int]]]:
  """Return list of (kalshi_bracket, poly_bracket, shared_degrees) for
  brackets whose degree coverage is IDENTICAL (used by v1)."""
  pairs = []
  for kb in k_bracks:
    kdegs = _degrees(kb, "kalshi")
    if not kdegs:
      continue
    kset = set(kdegs)
    for pb in p_bracks:
      pdegs = _degrees(pb, "polymarket")
      if not pdegs:
        continue
      if set(pdegs) == kset:
        pairs.append((kb, pb, sorted(kset)))
        break
  return pairs


def implied_pdf(brackets: list[dict], venue: str) -> dict[int, float]:
  """Build a per-degree implied probability map using each bracket's mid
  spread uniformly across covered degrees. Does NOT normalize -- raw
  implied prob per degree (so we can sum overlaps)."""
  pdf: dict[int, float] = {}
  for b in brackets:
    mid = _midprice(b)
    if mid is None or mid <= 0:
      continue
    degs = _degrees(b, venue)
    if not degs:
      continue
    per = mid / len(degs)
    for d in degs:
      pdf[d] = pdf.get(d, 0.0) + per
  return pdf


# ── Strategy core ──────────────────────────────────────────────────────

@dataclass
class TradeRecord:
  city: str
  entry_ts: str
  exit_ts: Optional[str]
  exit_reason: str
  venue: str
  bracket_id: str
  bracket_label: str
  degrees: list[int]
  size: int
  entry_price: float
  exit_price: Optional[float]
  entry_gap: float
  rival_venue: str
  rival_price: float
  pnl: float


def build_signals_v1(
  city: str, ts: str, venues: dict[str, list[dict]],
  gap_threshold: float, size: int,
) -> list[dict]:
  """Aligned brackets only. Returns list of signals.
  Each signal: which venue+bracket to BUY (lift ask)."""
  signals = []
  if "kalshi" not in venues or "polymarket" not in venues:
    return signals
  for kb, pb, degs in find_aligned_brackets(venues["kalshi"], venues["polymarket"]):
    k_mid = _midprice(kb)
    p_mid = _midprice(pb)
    if k_mid is None or p_mid is None:
      continue
    gap = k_mid - p_mid  # positive => Kalshi richer => Poly YES is cheap relative
    if abs(gap) < gap_threshold:
      continue
    if gap > 0:
      # Polymarket YES looks cheap -> buy YES on Polymarket
      buy_b = pb
      buy_venue = "polymarket"
      buy_ask = pb.get("best_yes_ask")
      rival_price = k_mid
    else:
      buy_b = kb
      buy_venue = "kalshi"
      buy_ask = kb.get("best_yes_ask")
      rival_price = p_mid
    if buy_ask is None:
      continue
    signals.append({
      "venue": buy_venue,
      "bracket": buy_b,
      "degrees": degs,
      "size": size,
      "entry_gap": abs(gap),
      "rival_venue": "kalshi" if buy_venue == "polymarket" else "polymarket",
      "rival_price": rival_price,
      "buy_price_estimate": buy_ask,
    })
  return signals


def build_signals_v2(
  city: str, ts: str, venues: dict[str, list[dict]],
  gap_threshold: float, size: int, depth_min: float = 50.0,
) -> list[dict]:
  """Per-degree implied PDF comparison; trades each bracket whose summed
  degree-prob differs by gap_threshold or more, requires bid_size on the
  cheap-side venue >= depth_min so we can EXIT later."""
  signals = []
  if "kalshi" not in venues or "polymarket" not in venues:
    return signals
  k_pdf = implied_pdf(venues["kalshi"], "kalshi")
  p_pdf = implied_pdf(venues["polymarket"], "polymarket")
  if not k_pdf or not p_pdf:
    return signals

  # Iterate each bracket on each venue. For that bracket's degrees,
  # compare own-implied prob (mid summed over degrees) vs rival-implied
  # prob (sum of rival's PDF over the same degrees). If |gap| >= thresh,
  # buy the bracket whose own-implied is LOWER.
  for venue, my_brackets, rival_pdf in (
    ("kalshi", venues["kalshi"], p_pdf),
    ("polymarket", venues["polymarket"], k_pdf),
  ):
    rival = "polymarket" if venue == "kalshi" else "kalshi"
    for b in my_brackets:
      degs = _degrees(b, venue)
      if not degs:
        continue
      mid = _midprice(b)
      ask = b.get("best_yes_ask")
      bid = b.get("best_yes_bid")
      if mid is None or ask is None:
        continue
      own_prob = mid
      rival_prob = sum(rival_pdf.get(d, 0.0) for d in degs)
      if rival_prob <= 0:
        continue
      gap = rival_prob - own_prob  # positive => rival thinks higher; my YES looks cheap
      if gap < gap_threshold:
        continue
      # Depth filter: we need own bid liquidity to sell into later.
      bid_sz = float(b.get("best_yes_bid_size") or 0.0)
      ask_sz = float(b.get("best_yes_ask_size") or 0.0)
      if ask_sz < size:
        continue
      if bid_sz < depth_min:
        # Optional filter; keep but tag
        pass
      signals.append({
        "venue": venue,
        "bracket": b,
        "degrees": degs,
        "size": size,
        "entry_gap": gap,
        "rival_venue": rival,
        "rival_price": rival_prob,
        "buy_price_estimate": ask,
      })
  # De-duplicate: a (venue, bracket_id) appears once
  seen = set()
  uniq = []
  for s in signals:
    key = (s["venue"], _bracket_id(s["bracket"], s["venue"]))
    if key in seen:
      continue
    seen.add(key)
    uniq.append(s)
  return uniq


# ── Exit simulation ────────────────────────────────────────────────────

def simulate_convergence_exit(
  position, future_snaps: list[tuple[str, dict]],
  entry_gap: float, rival_venue: str,
  exit_gap_factor: float = 0.5,
  time_stop_minutes: int = 120,
  fee: float = FEE_PER_CONTRACT,
):
  """Walk forward; close when the cross-venue gap shrinks to
  entry_gap * exit_gap_factor or less. Otherwise time-stop or end-of-
  window. Returns (exit_ts, exit_reason, exit_price, exit_proceeds, pnl)."""
  total_fees = 2 * fee * position.size
  entry_dt = _parse_dt(position.entry_ts)
  stop_dt = entry_dt + timedelta(minutes=time_stop_minutes)

  last_bids = None
  last_ts = None

  for ts, venues in future_snaps:
    own = (find_kalshi_bracket(venues.get("kalshi", []), position.bracket_id)
           if position.venue == "kalshi"
           else find_polymarket_bracket(venues.get("polymarket", []), position.bracket_id))
    if own is None:
      continue
    own_bids = own.get("yes_bid_ladder") or []
    if own_bids:
      last_bids = own_bids
      last_ts = ts

    # Compute current cross-venue gap (using mids).
    rival_brackets = venues.get(rival_venue, [])
    own_mid = _midprice(own)
    if not rival_brackets or own_mid is None:
      cur_dt = _parse_dt(ts)
      if cur_dt >= stop_dt and last_bids:
        break
      continue

    # Find rival bracket(s) covering this bracket's degrees and sum their
    # implied per-degree prob over the position's degree set.
    rival_pdf = implied_pdf(rival_brackets, rival_venue)
    rival_prob = sum(rival_pdf.get(d, 0.0) for d in position.bracket_degrees)
    cur_gap = rival_prob - own_mid  # was positive at entry; convergence => smaller

    if cur_gap <= entry_gap * exit_gap_factor and own_bids:
      # Convergence target hit -> sell into bid
      sell = walk_ladder_sell(own_bids, position.size)
      if sell is not None:
        avg_sell, proceeds = sell
        pnl = proceeds - position.entry_cost - total_fees
        return (ts, "target", avg_sell, proceeds, pnl)

    cur_dt = _parse_dt(ts)
    if cur_dt >= stop_dt:
      if last_bids:
        sell = walk_ladder_sell(last_bids, position.size)
        if sell is not None:
          avg_sell, proceeds = sell
          pnl = proceeds - position.entry_cost - total_fees
          return (last_ts, "time_stop", avg_sell, proceeds, pnl)
      break

  if last_bids:
    sell = walk_ladder_sell(last_bids, position.size)
    if sell is not None:
      avg_sell, proceeds = sell
      pnl = proceeds - position.entry_cost - total_fees
      return (last_ts, "end_of_window", avg_sell, proceeds, pnl)

  return (None, "no_liquidity", None, 0.0, -position.entry_cost - total_fees)


# ── Top-level backtest ────────────────────────────────────────────────

def run_backtest(version: str, signal_fn, gap_threshold: float, size: int,
                 exit_gap_factor: float = 0.5, time_stop_minutes: int = 120,
                 depth_min: float = 0.0,
                 ) -> dict:
  trades: list[TradeRecord] = []
  open_keys: set = set()  # (city, venue, bracket_id, side) already open in-session
  for city in CITIES:
    snaps = list(iter_snapshots(city, DATE))
    open_in_city = set()
    for i, (ts, venues) in enumerate(snaps):
      if "kalshi" not in venues or "polymarket" not in venues:
        continue
      if signal_fn is build_signals_v1:
        sigs = signal_fn(city, ts, venues, gap_threshold, size)
      else:
        sigs = signal_fn(city, ts, venues, gap_threshold, size, depth_min=depth_min)
      for sig in sigs:
        bid_sz = float(sig["bracket"].get("best_yes_bid_size") or 0.0)
        if depth_min and bid_sz < depth_min and version != "v1":
          continue
        bid_or_id = _bracket_id(sig["bracket"], sig["venue"])
        key = (city, sig["venue"], bid_or_id)
        if key in open_in_city:
          continue
        pos = open_position(city, DATE, sig["venue"], sig["bracket"],
                            side="yes_long", size=sig["size"], entry_ts=ts)
        if pos is None:
          continue
        pos.bracket_degrees = sig["degrees"]
        open_in_city.add(key)

        future = snaps[i + 1:]
        exit_ts, exit_reason, exit_price, exit_proceeds, pnl = simulate_convergence_exit(
          pos, future,
          entry_gap=sig["entry_gap"], rival_venue=sig["rival_venue"],
          exit_gap_factor=exit_gap_factor, time_stop_minutes=time_stop_minutes,
        )

        trades.append(TradeRecord(
          city=city, entry_ts=ts, exit_ts=exit_ts, exit_reason=exit_reason,
          venue=sig["venue"], bracket_id=bid_or_id,
          bracket_label=pos.bracket_label, degrees=sig["degrees"],
          size=sig["size"], entry_price=pos.entry_price,
          exit_price=exit_price, entry_gap=sig["entry_gap"],
          rival_venue=sig["rival_venue"], rival_price=sig["rival_price"],
          pnl=pnl,
        ))

  return summarize(version, trades)


def summarize(version: str, trades: list[TradeRecord]) -> dict:
  if not trades:
    return {
      "version": version, "n_trades": 0, "n_winners": 0, "win_rate": None,
      "total_pnl_dollars": 0.0, "avg_pnl_per_trade": None,
      "median_pnl_per_trade": None, "sharpe_per_trade": None,
      "max_drawdown_dollars": 0.0, "settled_n": 0, "settled_pnl_dollars": 0.0,
      "exit_reason_counts": {}, "notes": "no trades"
    }
  pnls = [t.pnl for t in trades]
  winners = sum(1 for p in pnls if p > 0)
  total = sum(pnls)
  avg = total / len(pnls)
  med = statistics.median(pnls)
  std = statistics.pstdev(pnls) if len(pnls) > 1 else 0.0
  sharpe = (avg / std) if std > 0 else None
  # Drawdown on cumulative pnl, ordered by entry_ts
  ordered = sorted(trades, key=lambda t: t.entry_ts)
  cum = 0.0
  peak = 0.0
  mdd = 0.0
  for t in ordered:
    cum += t.pnl
    peak = max(peak, cum)
    mdd = min(mdd, cum - peak)
  reason_counts: dict[str, int] = {}
  for t in trades:
    reason_counts[t.exit_reason] = reason_counts.get(t.exit_reason, 0) + 1
  return {
    "version": version,
    "n_trades": len(trades),
    "n_winners": winners,
    "win_rate": winners / len(trades),
    "total_pnl_dollars": round(total, 4),
    "avg_pnl_per_trade": round(avg, 6),
    "median_pnl_per_trade": round(med, 6),
    "sharpe_per_trade": round(sharpe, 4) if sharpe is not None else None,
    "max_drawdown_dollars": round(mdd, 4),
    "settled_n": 0,
    "settled_pnl_dollars": 0.0,
    "exit_reason_counts": reason_counts,
    "notes": "",
    "_trades": [asdict(t) for t in trades],
  }


def run_hedged_backtest(version: str, gap_threshold: float, size: int,
                         exit_gap_factor: float = 0.5, time_stop_minutes: int = 240,
                         require_alignment: bool = True,
                         hold_to_endwindow: bool = False,
                         exit_on_proceeds_target: Optional[float] = None,
                         max_entry_cost_per_pair: float = 1.05,
                         use_mid_fallback: bool = True,
                         cooldown_minutes: int = 0,
                         ) -> dict:
  """v4 hedged: for each aligned-bracket pair where mid gap >= threshold,
  buy YES on cheap venue and buy NO on rich venue (synth via 1 - best_yes_bid).
  This locks in gap minus all-in spread cost.
  Exit: hold both legs to end of window OR when gap converges.
  """
  trades: list[TradeRecord] = []
  for city in CITIES:
    snaps = list(iter_snapshots(city, DATE))
    open_in_city: dict = {}  # key -> last_open_ts
    for i, (ts, venues) in enumerate(snaps):
      if "kalshi" not in venues or "polymarket" not in venues:
        continue
      if not require_alignment:
        # not implemented for now
        continue
      for kb, pb, degs in find_aligned_brackets(venues["kalshi"], venues["polymarket"]):
        k_mid = _midprice(kb)
        p_mid = _midprice(pb)
        if k_mid is None or p_mid is None:
          continue
        gap = k_mid - p_mid
        if abs(gap) < gap_threshold:
          continue
        # Determine cheap / rich:
        if gap > 0:
          # Kalshi richer -> short Kalshi YES (buy NO on Kalshi); long Polymarket YES
          rich_b, rich_v = kb, "kalshi"
          cheap_b, cheap_v = pb, "polymarket"
        else:
          rich_b, rich_v = pb, "polymarket"
          cheap_b, cheap_v = kb, "kalshi"
        cheap_id = _bracket_id(cheap_b, cheap_v)
        rich_id = _bracket_id(rich_b, rich_v)
        key = (city, cheap_v, cheap_id, rich_v, rich_id)
        if key in open_in_city:
          last_ts = open_in_city[key]
          if cooldown_minutes <= 0:
            continue
          elapsed = (_parse_dt(ts) - _parse_dt(last_ts)).total_seconds() / 60.0
          if elapsed < cooldown_minutes:
            continue
        # Open BOTH legs at this snapshot:
        long_pos = open_position(city, DATE, cheap_v, cheap_b,
                                 side="yes_long", size=size, entry_ts=ts)
        short_pos = open_position(city, DATE, rich_v, rich_b,
                                  side="no_long", size=size, entry_ts=ts)
        if long_pos is None or short_pos is None:
          continue
        # Cost cap: don't pay more than max_entry_cost_per_pair per contract pair.
        per_pair_cost = long_pos.entry_price + short_pos.entry_price
        if per_pair_cost > max_entry_cost_per_pair:
          continue
        long_pos.bracket_degrees = degs
        short_pos.bracket_degrees = degs
        open_in_city[key] = ts

        future = snaps[i + 1:]
        # Total entry cost (both legs) plus 4 fee-events (2 contracts each leg, 2 sides)
        total_entry = long_pos.entry_cost + short_pos.entry_cost
        total_fees = 4 * FEE_PER_CONTRACT * size

        # Exit: same logic, but we need to close both legs.
        long_bid_ladder_last = None
        short_bid_ladder_last = None  # NO bid ladder = (1-yes_ask, qty)
        long_last_ts = None
        short_last_ts = None
        long_last_mid = None
        short_last_mid = None
        entry_dt = _parse_dt(ts)
        stop_dt = entry_dt + timedelta(minutes=time_stop_minutes)

        exit_ts_used = None
        exit_reason = "no_liquidity"
        long_proceeds = 0.0
        short_proceeds = 0.0

        target_factor = exit_gap_factor

        for fts, fv in future:
          long_b = (find_kalshi_bracket(fv.get("kalshi", []), long_pos.bracket_id)
                    if long_pos.venue == "kalshi"
                    else find_polymarket_bracket(fv.get("polymarket", []), long_pos.bracket_id))
          short_b = (find_kalshi_bracket(fv.get("kalshi", []), short_pos.bracket_id)
                     if short_pos.venue == "kalshi"
                     else find_polymarket_bracket(fv.get("polymarket", []), short_pos.bracket_id))
          if long_b is not None:
            lbids = long_b.get("yes_bid_ladder") or []
            if lbids:
              long_bid_ladder_last = lbids
              long_last_ts = fts
            l_m = _midprice(long_b)
            if l_m is not None:
              long_last_mid = l_m
          if short_b is not None:
            # NO bid ladder = transform yes_ask_ladder to (1-p, q)
            sbids = [(round(1.0 - p, 4), q) for p, q in (short_b.get("yes_ask_ladder") or [])]
            if sbids:
              short_bid_ladder_last = sbids
              short_last_ts = fts
            s_m = _midprice(short_b)  # short = NO; NO mid = 1 - YES_mid
            if s_m is not None:
              short_last_mid = 1.0 - s_m

          # PROCEEDS-based exit target: combined bid-side proceeds covers
          # entry cost + fees + minimum profit target.
          if (exit_on_proceeds_target is not None and not hold_to_endwindow
              and long_bid_ladder_last and short_bid_ladder_last):
            l_sell = walk_ladder_sell(long_bid_ladder_last, size)
            s_sell = walk_ladder_sell(short_bid_ladder_last, size)
            if l_sell and s_sell:
              cand_proceeds = l_sell[1] + s_sell[1]
              cand_pnl = cand_proceeds - total_entry - total_fees
              if cand_pnl >= exit_on_proceeds_target * size:
                long_proceeds = l_sell[1]
                short_proceeds = s_sell[1]
                exit_ts_used = fts
                exit_reason = "target"
                break

          # Mid-gap-based convergence target (legacy)
          if (exit_on_proceeds_target is None and long_b is not None
              and short_b is not None and not hold_to_endwindow):
            l_mid = _midprice(long_b)
            s_mid = _midprice(short_b)
            if l_mid is not None and s_mid is not None:
              # gap definition consistent with entry: rich - cheap (mids)
              cur_gap = s_mid - l_mid  # rich(short) - cheap(long)
              # Convergence: |cur_gap| <= |entry_gap| * factor
              if abs(cur_gap) <= abs(gap) * target_factor:
                if long_bid_ladder_last and short_bid_ladder_last:
                  l_sell = walk_ladder_sell(long_bid_ladder_last, size)
                  s_sell = walk_ladder_sell(short_bid_ladder_last, size)
                  if l_sell and s_sell:
                    long_proceeds = l_sell[1]
                    short_proceeds = s_sell[1]
                    exit_ts_used = fts
                    exit_reason = "target"
                    break

          # Time stop
          cur_dt = _parse_dt(fts)
          if cur_dt >= stop_dt and not hold_to_endwindow:
            if long_bid_ladder_last and short_bid_ladder_last:
              l_sell = walk_ladder_sell(long_bid_ladder_last, size)
              s_sell = walk_ladder_sell(short_bid_ladder_last, size)
              if l_sell and s_sell:
                long_proceeds = l_sell[1]
                short_proceeds = s_sell[1]
                exit_ts_used = fts
                exit_reason = "time_stop"
            break

        if exit_reason == "no_liquidity":
          # End of window: sell both at last seen bids. If a leg has no
          # bids and use_mid_fallback is True, use last seen mid as a
          # proxy for "hold to settle" expected value.
          if long_bid_ladder_last:
            l_sell = walk_ladder_sell(long_bid_ladder_last, size)
            if l_sell:
              long_proceeds = l_sell[1]
          elif use_mid_fallback:
            long_proceeds = (long_last_mid or 0.0) * size
          if short_bid_ladder_last:
            s_sell = walk_ladder_sell(short_bid_ladder_last, size)
            if s_sell:
              short_proceeds = s_sell[1]
          elif use_mid_fallback:
            short_proceeds = (short_last_mid or 0.0) * size
          exit_ts_used = long_last_ts or short_last_ts
          if long_proceeds > 0 or short_proceeds > 0 or use_mid_fallback:
            exit_reason = "end_of_window"

        pnl = long_proceeds + short_proceeds - total_entry - total_fees

        # We log this as ONE trade record summarizing the hedged pair
        trades.append(TradeRecord(
          city=city, entry_ts=ts, exit_ts=exit_ts_used, exit_reason=exit_reason,
          venue=f"{cheap_v}_long+{rich_v}_short",
          bracket_id=f"{cheap_id}|{rich_id}",
          bracket_label=str(degs),
          degrees=degs,
          size=size,
          entry_price=long_pos.entry_price + short_pos.entry_price,
          exit_price=(long_proceeds + short_proceeds) / max(size, 1),
          entry_gap=abs(gap),
          rival_venue=rich_v,
          rival_price=p_mid if rich_v == "polymarket" else k_mid,
          pnl=pnl,
        ))

  return summarize(version, trades)


def main():
  out_dir = Path(__file__).resolve().parent

  # v1 — UNHEDGED baseline (directional bet on cheap-side YES); 5c threshold
  # PURPOSE: tests the original assumption that gaps converge in a way
  # that benefits a directional buy on the cheap side.
  print("Running v1: UNHEDGED directional, gap>=5c, gap-halves exit")
  v1 = run_backtest("v1", build_signals_v1, gap_threshold=0.05, size=25,
                    exit_gap_factor=0.5, time_stop_minutes=120)
  v1["notes"] = ("UNHEDGED directional buy on cheap-side YES (single leg); "
                 "identical-bracket alignment only; gap >=5c; exit when gap halves "
                 "or 2h time stop; size=25; fee=0.005/contract. "
                 "BASELINE: convergence is bidirectional, so directional bet exposed.")

  # v2 — UNHEDGED per-degree PDF (more signals incl SF/Seattle partials) + depth
  # PURPOSE: tests whether widening the signal universe + depth filter helps.
  print("Running v2: UNHEDGED per-degree PDF gap, gap>=5c, depth_min=50")
  v2 = run_backtest("v2", build_signals_v2, gap_threshold=0.05, size=25,
                    exit_gap_factor=0.5, time_stop_minutes=120, depth_min=50)
  v2["notes"] = ("UNHEDGED; per-degree PDF probability gap (sums rival PDF over "
                 "this bracket's degrees -- handles partial overlap brackets like "
                 "Kalshi tails vs Polymarket 2-deg bins). depth_min=50 contracts on best bid.")

  # v3 — HEDGED relative-value (long cheap YES + short rich YES via NO);
  # proceeds-target exit (close when bid-side proceeds cover entry + fees + 2c profit).
  # PURPOSE: the proper relative-value formulation; exit only when book actually
  # offers a profitable close.
  print("Running v3: HEDGED pair, proceeds-target >=2c/pair profit, gap>=5c")
  v3 = run_hedged_backtest("v3", gap_threshold=0.05, size=25,
                            exit_on_proceeds_target=0.02,
                            time_stop_minutes=240, max_entry_cost_per_pair=1.0,
                            use_mid_fallback=False)
  v3["notes"] = ("HEDGED pair: long YES cheap-side + long NO rich-side. "
                  "Exit when combined bid-side close-proceeds clear 2c/pair profit "
                  "post-fee, OR 4h time stop, OR end-of-window. NO mid-fallback "
                  "(if a leg goes no-bid, that leg's exit value = 0).")

  # v4 — HEDGED hold-to-end-of-window (mid-fallback enabled).
  # PURPOSE: tests pure structural arb hypothesis (same venue = same settlement).
  # Mid-fallback simulates "if book dies, take last observable mid as the
  # market's expected settlement value" (proxy for hold-to-settle).
  print("Running v4: HEDGED hold-to-EOW, gap>=5c, mid-fallback ON")
  v4 = run_hedged_backtest("v4", gap_threshold=0.05, size=25,
                            hold_to_endwindow=True, max_entry_cost_per_pair=1.0,
                            use_mid_fallback=True)
  v4["notes"] = ("HEDGED hold-to-end-of-window. Cost cap 1.0/pair. Mid-fallback: "
                  "if a leg's bid book dies before EOW, value = last observable mid * size. "
                  "Approximates hold-to-settle EV but assumes mid is unbiased and venues "
                  "agree on settlement (historically only ~26% of pairs agreed).")

  # v5 — HEDGED hold-to-EOW with strict cost cap 0.97 (filter for clear arb)
  # PURPOSE: only enter when sum of entry asks <= 0.97; that's a 3c+ structural
  # edge BEFORE we incur any exit slippage. Should be the most robust version.
  print("Running v5: HEDGED hold-to-EOW, cost_cap<=0.97, gap>=5c")
  v5 = run_hedged_backtest("v5", gap_threshold=0.05, size=25,
                            hold_to_endwindow=True, max_entry_cost_per_pair=0.97,
                            use_mid_fallback=True)
  v5["notes"] = ("HEDGED hold-to-EOW with strict 0.97 cost cap. Entry-cost cap means "
                  "we only enter when there is at-entry edge >= 3c per pair PRE-exit. "
                  "Same caveats: assumes both venues settle on same daily high.")

  # v6 — HEDGED hold-to-EOW, NO mid-fallback (conservative -- tests how
  # vulnerable the strategy is to one leg's book dying).
  print("Running v6: HEDGED hold-to-EOW, cost_cap<=0.97, NO mid-fallback")
  v6 = run_hedged_backtest("v6", gap_threshold=0.05, size=25,
                            hold_to_endwindow=True, max_entry_cost_per_pair=0.97,
                            use_mid_fallback=False)
  v6["notes"] = ("Same as v5 but mid-fallback OFF. If a leg has no bid book at "
                  "exit, that leg yields $0 -- pessimistic. Shows the impact of "
                  "the mid-fallback assumption (i.e., how much of v5's pnl depends on it).")

  metrics = {
    "strategy_name": "cross_venue_convergence",
    "iterations": [
      {k: v for k, v in row.items() if not k.startswith("_")}
      for row in (v1, v2, v3, v4, v5, v6)
    ],
  }
  with (out_dir / "metrics.json").open("w") as f:
    json.dump(metrics, f, indent=2)

  details = {"v1": v1["_trades"], "v2": v2["_trades"], "v3": v3["_trades"],
             "v4": v4["_trades"], "v5": v5["_trades"], "v6": v6["_trades"]}
  with (out_dir / "trades.json").open("w") as f:
    json.dump(details, f, indent=2)

  print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
  main()
