"""
Shared data loaders for strategy backtests.

Read-only access to data/bot.db. All functions return plain Python types
(no ORM, no model classes) so each strategy can shape data its own way.

Usage:
    from research.loaders import (
        list_settled_pairs, load_settlement, load_forecast_pdf,
        iter_snapshots, BracketRow,
    )

    for city, date, k_high, p_high in list_settled_pairs(both_venues=True):
        pdf = load_forecast_pdf(city, date)
        for ts, kalshi_brackets, poly_brackets in iter_snapshots(city, date):
            ...
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "bot.db"


def _conn() -> sqlite3.Connection:
  c = sqlite3.connect(str(DB_PATH))
  c.row_factory = sqlite3.Row
  return c


# ── Settlements ──────────────────────────────────────────────────────────

def list_settled_pairs(both_venues: bool = False) -> list[tuple[str, str, Optional[int], Optional[int]]]:
  """List (city, date, kalshi_high, poly_high). If both_venues, only rows
  where neither final is NULL."""
  q = "SELECT city, date, kalshi_high_f, polymarket_high_f FROM settlements"
  if both_venues:
    q += " WHERE kalshi_high_f IS NOT NULL AND polymarket_high_f IS NOT NULL"
  q += " ORDER BY city, date"
  with _conn() as c:
    return [(r["city"], r["date"], r["kalshi_high_f"], r["polymarket_high_f"])
            for r in c.execute(q)]


def load_settlement(city: str, date: str) -> Optional[tuple[Optional[int], Optional[int]]]:
  """Return (kalshi_high, poly_high) for the (city, date), or None if no row."""
  with _conn() as c:
    r = c.execute(
      "SELECT kalshi_high_f, polymarket_high_f FROM settlements WHERE city=? AND date=?",
      (city, date),
    ).fetchone()
  return (r["kalshi_high_f"], r["polymarket_high_f"]) if r else None


# ── Forecasts ────────────────────────────────────────────────────────────

def load_forecast_pdf(city: str, date: str, source: str = "nws_normal") -> dict[int, float]:
  """Return the first-fetched forecast PDF for (city, date) as {deg: prob}.
  Returns {} if no forecast exists."""
  with _conn() as c:
    r = c.execute(
      """SELECT pdf_json FROM forecasts
           WHERE city=? AND date=? AND source=?
           ORDER BY fetched_at_utc ASC LIMIT 1""",
      (city, date, source),
    ).fetchone()
  if not r:
    return {}
  return {int(k): float(v) for k, v in json.loads(r["pdf_json"]).items()}


def load_forecast_meta(city: str, date: str, source: str = "nws_normal") -> dict:
  """Return {std_dev, fetched_at_utc} for the first forecast row, or {}."""
  with _conn() as c:
    r = c.execute(
      """SELECT std_dev, fetched_at_utc FROM forecasts
           WHERE city=? AND date=? AND source=?
           ORDER BY fetched_at_utc ASC LIMIT 1""",
      (city, date, source),
    ).fetchone()
  return dict(r) if r else {}


# ── Snapshots ────────────────────────────────────────────────────────────

def iter_snapshots(
  city: str, date: str, venues: tuple[str, ...] = ("kalshi", "polymarket"),
) -> Iterator[tuple[str, dict[str, list]]]:
  """Yield (timestamp_utc, {venue: brackets_list}) — venue-keyed dict so a
  single timestamp gives you both venues' bracket states. Snapshots are
  yielded by SHARED timestamp (joined on fetched_at_utc); if one venue
  is missing for that timestamp, the dict has that venue absent.
  """
  with _conn() as c:
    rows = c.execute(
      """SELECT fetched_at_utc, venue, brackets_json FROM snapshots
           WHERE city=? AND date=? AND venue IN (%s)
           ORDER BY fetched_at_utc""" % ",".join("?" * len(venues)),
      (city, date, *venues),
    ).fetchall()
  by_ts: dict[str, dict[str, list]] = {}
  for r in rows:
    by_ts.setdefault(r["fetched_at_utc"], {})[r["venue"]] = json.loads(r["brackets_json"])
  for ts in sorted(by_ts.keys()):
    yield ts, by_ts[ts]


def list_dates_with_snapshots() -> list[tuple[str, str]]:
  """List all distinct (city, date) pairs that have at least one snapshot."""
  with _conn() as c:
    return [(r["city"], r["date"]) for r in c.execute(
      "SELECT DISTINCT city, date FROM snapshots ORDER BY city, date")]


# ── Trade scoring ────────────────────────────────────────────────────────

@dataclass
class Trade:
  """A simulated trade. P&L scored by score_trade() once the actual high
  is known. `side` is the venue side ('yes'/'no'); `price` is the buy
  price you paid; `size` in contracts; `degrees` are the integer degrees
  the bracket covers (so we can check whether actual_high lies inside)."""
  city: str
  date: str
  venue: str           # "kalshi" | "polymarket"
  bracket_label: str
  degrees: list[int]
  side: str            # "yes" | "no"
  price: float         # 0..1
  size: int


def score_trade(trade: Trade, kalshi_high: Optional[int], poly_high: Optional[int]) -> Optional[float]:
  """Return realized P&L in dollars, or None if outcome unavailable.

  Uses the venue's own reading: a Kalshi trade settles vs kalshi_high;
  a Polymarket trade settles vs poly_high. If the venue has no recorded
  high, returns None (so the caller can drop that trade)."""
  actual = kalshi_high if trade.venue == "kalshi" else poly_high
  if actual is None:
    return None
  in_bracket = actual in trade.degrees
  yes_wins = (trade.side == "yes" and in_bracket) or (trade.side == "no" and not in_bracket)
  payout_per_contract = 1.0 if yes_wins else 0.0
  return (payout_per_contract - trade.price) * trade.size


# ── Convenience: settled (city, date) pairs that ALSO have snapshots ─────

def list_backtestable_pairs(venue: str = "kalshi") -> list[tuple[str, str, int]]:
  """Return (city, date, actual_high) for pairs that have BOTH a settled
  outcome AND at least one snapshot. `venue` selects which final to use.
  This is the set you can run a closed-loop backtest on."""
  col = "kalshi_high_f" if venue == "kalshi" else "polymarket_high_f"
  with _conn() as c:
    rows = c.execute(
      f"""SELECT s.city, s.date, t.{col} AS h
            FROM (SELECT DISTINCT city, date FROM snapshots) s
            JOIN settlements t USING(city, date)
           WHERE t.{col} IS NOT NULL
           ORDER BY s.city, s.date"""
    ).fetchall()
  return [(r["city"], r["date"], r["h"]) for r in rows]


# ── Bracket lookup helpers (find the same bracket across snapshots) ──────

def find_kalshi_bracket(brackets: list[dict], ticker: str) -> Optional[dict]:
  """Find a Kalshi bracket by ticker (the stable identifier across snapshots)."""
  for b in brackets:
    if b.get("ticker") == ticker:
      return b
  return None


def find_polymarket_bracket(brackets: list[dict], token_id: str) -> Optional[dict]:
  """Find a Polymarket bracket by token_id (stable across snapshots)."""
  for b in brackets:
    if b.get("token_id") == token_id:
      return b
  return None


# ── Ladder-walking fills ─────────────────────────────────────────────────

def walk_ladder_buy(ladder_asks: list, size: int) -> Optional[tuple[float, float]]:
  """Walk an ASK ladder to fill a BUY for `size` contracts.

  ladder_asks is a list of [price, qty] entries, ordered best (lowest)
  ask first in our captured data. Returns (avg_price, total_cost) or
  None if the ladder can't fill the requested size.
  """
  if not ladder_asks or size <= 0:
    return None
  remaining = float(size)
  total_cost = 0.0
  # Defensive: re-sort ascending in case stored order is unexpected
  levels = sorted([(float(p), float(q)) for p, q in ladder_asks if q and q > 0])
  for price, qty in levels:
    take = min(qty, remaining)
    total_cost += take * price
    remaining -= take
    if remaining <= 1e-9:
      return (total_cost / size, total_cost)
  return None  # not enough depth


def walk_ladder_sell(ladder_bids: list, size: int) -> Optional[tuple[float, float]]:
  """Walk a BID ladder to fill a SELL for `size` contracts.

  ladder_bids is a list of [price, qty] entries. We sort descending so we
  hit the best (highest) bid first. Returns (avg_price, total_proceeds)
  or None if depth insufficient.
  """
  if not ladder_bids or size <= 0:
    return None
  remaining = float(size)
  total = 0.0
  levels = sorted(
    [(float(p), float(q)) for p, q in ladder_bids if q and q > 0],
    key=lambda pq: -pq[0],
  )
  for price, qty in levels:
    take = min(qty, remaining)
    total += take * price
    remaining -= take
    if remaining <= 1e-9:
      return (total / size, total)
  return None


def max_size_at_or_better(ladder_asks: list, max_price: float) -> int:
  """Largest BUY size that fills at avg price <= max_price using `ladder_asks`."""
  if not ladder_asks:
    return 0
  levels = sorted([(float(p), float(q)) for p, q in ladder_asks if q and q > 0])
  filled = 0.0
  cost = 0.0
  for price, qty in levels:
    if price > max_price:
      break
    filled += qty
    cost += qty * price
    if filled > 0 and cost / filled > max_price:
      # walking further would push avg above max_price
      break
  return int(filled)


# ── Position simulation: enter at snapshot t, exit on a rule ─────────────

@dataclass
class Position:
  """A simulated position. side='yes_long' means we bought YES contracts;
  'no_long' means we bought NO contracts (= sold YES at 1-price).
  size is integer contracts. entry_price is the avg fill price paid.
  """
  city: str
  date: str
  venue: str
  bracket_id: str           # ticker (kalshi) or token_id (polymarket)
  bracket_label: str
  bracket_degrees: list[int]
  side: str                 # 'yes_long' | 'no_long'
  size: int
  entry_ts: str
  entry_price: float        # avg fill price paid (in YES-equivalent units)
  entry_cost: float         # total $ paid


@dataclass
class ExitResult:
  exit_ts: Optional[str]
  exit_reason: str          # 'target' | 'time_stop' | 'hold_to_settle' | 'no_liquidity'
  exit_price: Optional[float]   # avg sell fill or settle payout
  exit_proceeds: float
  realized_pnl: float       # exit_proceeds - position.entry_cost - fees


def _bracket_id(b: dict, venue: str) -> str:
  return b.get("ticker", "") if venue == "kalshi" else b.get("token_id", "")


def _ask_ladder(b: dict, side: str) -> list:
  """Return the ladder you'd LIFT to BUY this side."""
  if side == "yes_long":
    return b.get("yes_ask_ladder") or []
  # no_long ↔ buying NO ↔ selling YES into bids ... but since both venues
  # quote a YES book only in our snapshots, buying NO == selling YES at
  # 1-(yes ask), priced 1-yes_bid. Caller should use yes_bid_ladder
  # transformed to a NO ask ladder via [(1-p, q) for ...]
  return [(round(1.0 - p, 4), q) for p, q in (b.get("yes_bid_ladder") or [])]


def _bid_ladder(b: dict, side: str) -> list:
  """Return the ladder you'd HIT to SELL this side (close the position)."""
  if side == "yes_long":
    return b.get("yes_bid_ladder") or []
  return [(round(1.0 - p, 4), q) for p, q in (b.get("yes_ask_ladder") or [])]


def open_position(
  city: str, date: str, venue: str, bracket: dict, side: str,
  size: int, entry_ts: str,
) -> Optional[Position]:
  """Open a long position by lifting the appropriate ask ladder. Returns
  None if depth insufficient."""
  asks = _ask_ladder(bracket, side)
  fill = walk_ladder_buy(asks, size)
  if fill is None:
    return None
  avg_price, total_cost = fill
  if venue == "kalshi":
    bracket_id = bracket.get("ticker", "")
    label = bracket.get("subtitle", "")
  else:
    bracket_id = bracket.get("token_id", "")
    label = bracket.get("question", "")
  # degrees parsing — use existing parsers when called
  return Position(
    city=city, date=date, venue=venue,
    bracket_id=bracket_id, bracket_label=label, bracket_degrees=[],
    side=side, size=size, entry_ts=entry_ts,
    entry_price=avg_price, entry_cost=total_cost,
  )


def simulate_exit(
  position: Position,
  future_snapshots: list[tuple[str, dict]],   # [(ts, brackets_for_position.venue), ...] AFTER entry
  target_pnl_per_contract: Optional[float] = None,
  time_stop_minutes: Optional[int] = None,
  fee_per_contract: float = 0.0,
  settlement_payout: Optional[float] = None,  # if hold-to-settle, $ paid per contract
) -> ExitResult:
  """Simulate exiting `position`. Walks `future_snapshots` (ordered) and:

    - At each future snapshot, checks if the bracket's bid (the price we'd
      hit to close) is high enough that selling there yields >=
      target_pnl_per_contract relative to entry. If yes, sell into the
      bid ladder and return.
    - If time_stop_minutes elapses without hitting target, sell at the
      then-current bid (whatever it is) and return.
    - If no future snapshot has the bracket alive (no bid), and no
      settlement_payout given, return 'no_liquidity'.
    - If hold_to_settle and settlement_payout is given, pay out at
      settlement.

  Fees are subtracted from realized P&L (fee_per_contract × size, applied
  to BOTH entry and exit, so doubled).
  """
  from datetime import datetime, timedelta

  total_fees = 2.0 * fee_per_contract * position.size
  entry_dt = datetime.fromisoformat(position.entry_ts.replace("Z", "+00:00"))
  time_stop_dt = (entry_dt + timedelta(minutes=time_stop_minutes)) if time_stop_minutes else None

  last_seen_bid_ladder = None
  last_seen_ts = None

  for ts, brackets in future_snapshots:
    b = (find_kalshi_bracket(brackets, position.bracket_id) if position.venue == "kalshi"
         else find_polymarket_bracket(brackets, position.bracket_id))
    if b is None:
      continue
    bids = _bid_ladder(b, position.side)
    if bids:
      last_seen_bid_ladder = bids
      last_seen_ts = ts
      # Probe target: would selling all `size` here clear target_pnl_per_contract?
      if target_pnl_per_contract is not None:
        sell = walk_ladder_sell(bids, position.size)
        if sell is not None:
          avg_sell, proceeds = sell
          per_contract_pnl = (proceeds - position.entry_cost) / position.size - 2.0 * fee_per_contract
          if per_contract_pnl >= target_pnl_per_contract:
            return ExitResult(
              exit_ts=ts, exit_reason="target",
              exit_price=avg_sell, exit_proceeds=proceeds,
              realized_pnl=proceeds - position.entry_cost - total_fees,
            )
    # Time stop?
    if time_stop_dt is not None:
      cur_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
      if cur_dt >= time_stop_dt:
        if last_seen_bid_ladder:
          sell = walk_ladder_sell(last_seen_bid_ladder, position.size)
          if sell is not None:
            avg_sell, proceeds = sell
            return ExitResult(
              exit_ts=last_seen_ts, exit_reason="time_stop",
              exit_price=avg_sell, exit_proceeds=proceeds,
              realized_pnl=proceeds - position.entry_cost - total_fees,
            )
        # No bids ever — fall through to settlement / no_liquidity below
        break

  # End of snapshot stream
  if settlement_payout is not None:
    proceeds = settlement_payout * position.size
    return ExitResult(
      exit_ts=last_seen_ts, exit_reason="hold_to_settle",
      exit_price=settlement_payout, exit_proceeds=proceeds,
      realized_pnl=proceeds - position.entry_cost - total_fees,
    )
  # If we saw bids but never hit target and no time stop, sell at last seen
  if last_seen_bid_ladder:
    sell = walk_ladder_sell(last_seen_bid_ladder, position.size)
    if sell is not None:
      avg_sell, proceeds = sell
      return ExitResult(
        exit_ts=last_seen_ts, exit_reason="end_of_window",
        exit_price=avg_sell, exit_proceeds=proceeds,
        realized_pnl=proceeds - position.entry_cost - total_fees,
      )
  return ExitResult(
    exit_ts=None, exit_reason="no_liquidity",
    exit_price=None, exit_proceeds=0.0,
    realized_pnl=-position.entry_cost - total_fees,
  )


def settlement_payout_for(position: Position, kalshi_high: Optional[int],
                          poly_high: Optional[int]) -> Optional[float]:
  """Return the $/contract paid at settlement for `position`, or None if
  outcome unknown. Uses the venue's own reading.
  """
  actual = kalshi_high if position.venue == "kalshi" else poly_high
  if actual is None:
    return None
  in_bracket = actual in position.bracket_degrees
  yes_wins = (position.side == "yes_long" and in_bracket) or \
             (position.side == "no_long" and not in_bracket)
  return 1.0 if yes_wins else 0.0
