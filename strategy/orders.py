"""
Build venue-neutral OrderRequests from ArbOpportunities.

This module is the handoff point from strategy/ to execution/. We produce
OrderRequest objects (defined in contracts.py) — the execution layer is
responsible for translating each into a venue-specific signed payload.

For reference, here is what each venue's API ultimately wants:

  Kalshi POST /portfolio/orders body:
    {
      "ticker": str,             # Kalshi market ticker, e.g. "KXHIGHMIA-26APR27-T82"
      "action": "buy" | "sell",
      "side":   "yes"  | "no",
      "count":  int,             # contracts
      "type":   "limit" | "market",
      "yes_price": int,          # cents (1-99) — required for limit
      "client_order_id": str,    # caller-generated UUID
    }

  Polymarket CLOB order (EIP-712 signed):
    {
      "token_id":   str,         # ERC-1155 asset ID for the YES outcome
      "price":      float,       # 0.00–1.00
      "size":       float,       # shares
      "side":       "BUY" | "SELL",
      "order_type": "GTC" | "FOK" | "GTD",
      # plus signature fields filled in by execution at signing time
    }
"""

import json
import logging
import os
import uuid
from pathlib import Path

from contracts import ArbOpportunity, OrderRequest, PairedArbOpportunity

logger = logging.getLogger(__name__)


def _is_paper_trading() -> bool:
  """
  Paper-trading mode: every logged order also gets a synthetic fill at
  intended_price (zero slippage, zero fee). Lets the existing settlement
  pipeline score every recommendation as if we'd actually filled it,
  feeding the recursive train→evaluate→improve loop without real money.

  Toggle via env var: PAPER_TRADING=1 (on) or PAPER_TRADING=0 (off, default).
  """
  return os.environ.get("PAPER_TRADING", "0") == "1"


# Slippage params written by analysis/slippage.py. Mtime-cached so the
# trading loop picks up new params after each refresh without restart.
_SLIPPAGE_PATH = Path(__file__).resolve().parent.parent / "data" / "slippage_params.json"
_slip_cache: dict[str, dict] = {}
_slip_mtime: float = 0.0


def _load_slippage_params() -> dict[str, dict]:
  global _slip_cache, _slip_mtime
  if not _SLIPPAGE_PATH.exists():
    return {}
  m = _SLIPPAGE_PATH.stat().st_mtime
  if m == _slip_mtime and _slip_cache:
    return _slip_cache
  _slip_cache = json.loads(_SLIPPAGE_PATH.read_text())
  _slip_mtime = m
  return _slip_cache


def _slippage_at(venue: str, size: int) -> float:
  """Predicted slippage in dollars per contract for an order of `size`."""
  v = _load_slippage_params().get(venue, {})
  p = v.get("params", {})
  if not p:
    return 0.0
  return p.get("a", 0.0) + p.get("b", 0.0) * size + p.get("c", 0.0) * size ** 2


def _max_size_under_slippage_budget(
  venue: str, top_of_book: int, expected_edge_per_contract: float,
  safety_fraction: float = 0.5,
) -> int:
  """
  Find the largest size ≤ top_of_book such that predicted slippage stays
  below `safety_fraction * expected_edge_per_contract`. If no slippage
  data exists, returns top_of_book unchanged.
  """
  if not _load_slippage_params() or expected_edge_per_contract <= 0:
    return top_of_book
  budget = safety_fraction * expected_edge_per_contract
  # Scan downward from top_of_book — small N, trivial cost
  for size in range(top_of_book, 0, -1):
    if _slippage_at(venue, size) <= budget:
      return size
  return 0


def from_opportunity(
  opp: ArbOpportunity,
  size: int | None = None,
  log_to_db: bool = True,
) -> OrderRequest:
  """
  Convert one ArbOpportunity (per-bracket value trade) into one OrderRequest.

  Args:
    opp: Trade signal from strategy.arbitrage.find_value_trades.
    size: Override the order size. By default uses opp.max_size (top-of-book
      depth), which trades the full available liquidity at the executable
      price. Pass a smaller size to be conservative, or to follow a sizing
      policy (Kelly fraction, capital cap, etc.) implemented elsewhere.
    log_to_db: If True (default), persist the order as 'pending' in the
      bot.db orders table so the execution layer can pick it up and the
      pnl tracker can later score realized vs predicted edge. Set False
      for in-memory exploration (e.g. backtests) where you don't want to
      pollute the live store.
  """
  base_size = size if size is not None else opp.max_size
  # Apply slippage-aware cap: if data/slippage_params.json exists, shrink
  # base_size so predicted slippage stays under half the per-contract edge.
  # No-op when slippage data hasn't been built yet.
  order_size = _max_size_under_slippage_budget(
    venue=opp.venue,
    top_of_book=base_size,
    expected_edge_per_contract=opp.net_edge,
  )
  if order_size <= 0:
    raise ValueError(
      f"OrderRequest size must be > 0 (got {order_size}). "
      f"opp.max_size={opp.max_size}; pass size= to override."
    )

  order = OrderRequest(
    venue=opp.venue,
    market_id=opp.market_id,
    side=opp.side,
    action=opp.action,
    price=opp.price,
    size=order_size,
    expected_edge=opp.net_edge,
    client_order_id=str(uuid.uuid4()),
    source_opportunity=opp,
  )

  if log_to_db:
    # Local import: keeps strategy/ usable without a database when needed
    # (e.g. unit tests, the in-memory smoke test scripts).
    try:
      from persistence import db as persist
      persist.log_order(order)
      # In paper-trading mode, immediately record a synthetic fill at the
      # intended price. settle_orders.py then scores it once the underlying
      # market settles — the closed loop for recursive model training.
      if _is_paper_trading():
        persist.record_fill(
          client_order_id=order.client_order_id,
          fill_price=order.price,
          fill_size=order.size,
          fee_paid=0.0,
        )
    except Exception as exc:
      logger.warning("Failed to log order to DB: %s", exc)

  return order


def from_opportunities(
  opps: list[ArbOpportunity],
  size: int | None = None,
  log_to_db: bool = True,
) -> list[OrderRequest]:
  """Convenience: convert a list of opportunities to a list of orders."""
  return [from_opportunity(o, size=size, log_to_db=log_to_db) for o in opps]


def from_paired_opportunity(
  pair: PairedArbOpportunity,
  size: int | None = None,
  log_to_db: bool = True,
) -> list[OrderRequest]:
  """
  Decompose one PairedArbOpportunity into the two OrderRequests that
  realize it. Both orders carry the same `arb_pair_id` (set as a tag on
  the embedded opportunity for traceability) so post-hoc P&L can group
  them together as a single hedged position.

  Sizing: by construction `pair.max_size` is min(legA_depth, legB_depth)
  so both legs hedge each other. Caller may downscale via `size` for
  capital limits.
  """
  pair_size = size if size is not None else pair.max_size
  if pair_size <= 0:
    raise ValueError(f"Paired order size must be > 0 (got {pair_size}).")

  pair_id = str(uuid.uuid4())  # both legs share this for join later

  # Leg A: Polymarket
  poly_opp = ArbOpportunity(
    city=pair.city, date=pair.date,
    trade_type=f"PAIRED_{pair.direction.upper()}",
    venue="polymarket",
    market_id=pair.poly_market_id,
    bracket_label=pair.poly_bracket_label,
    degrees=list(pair.poly_degrees),
    side=pair.poly_side,
    action="buy",
    price=pair.poly_price,
    fair_value=pair.expected_payout / 2,  # informational; per-leg fair is fuzzy
    net_edge=pair.net_edge,
    max_size=pair.max_size,
  )
  poly_order = OrderRequest(
    venue="polymarket", market_id=pair.poly_market_id,
    side=pair.poly_side, action="buy",
    price=pair.poly_price, size=pair_size,
    expected_edge=pair.net_edge,
    client_order_id=f"{pair_id}::poly",
    source_opportunity=poly_opp,
  )

  # Leg B: Kalshi
  kalshi_opp = ArbOpportunity(
    city=pair.city, date=pair.date,
    trade_type=f"PAIRED_{pair.direction.upper()}",
    venue="kalshi",
    market_id=pair.kalshi_market_id,
    bracket_label=pair.kalshi_bracket_label,
    degrees=list(pair.kalshi_degrees),
    side=pair.kalshi_side,
    action="buy",
    price=pair.kalshi_price,
    fair_value=pair.expected_payout / 2,
    net_edge=pair.net_edge,
    max_size=pair.max_size,
  )
  kalshi_order = OrderRequest(
    venue="kalshi", market_id=pair.kalshi_market_id,
    side=pair.kalshi_side, action="buy",
    price=pair.kalshi_price, size=pair_size,
    expected_edge=pair.net_edge,
    client_order_id=f"{pair_id}::kalshi",
    source_opportunity=kalshi_opp,
  )

  if log_to_db:
    try:
      from persistence import db as persist
      persist.log_order(poly_order, arb_pair_id=pair_id)
      persist.log_order(kalshi_order, arb_pair_id=pair_id)
    except Exception as exc:
      logger.warning("Failed to log paired orders to DB: %s", exc)

  return [poly_order, kalshi_order]
