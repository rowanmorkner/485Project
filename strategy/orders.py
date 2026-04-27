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

import uuid

from contracts import ArbOpportunity, OrderRequest


def from_opportunity(opp: ArbOpportunity, size: int | None = None) -> OrderRequest:
  """
  Convert one ArbOpportunity (per-bracket value trade) into one OrderRequest.

  Args:
    opp: Trade signal from strategy.arbitrage.find_value_trades.
    size: Override the order size. By default uses opp.max_size (top-of-book
      depth), which trades the full available liquidity at the executable
      price. Pass a smaller size to be conservative, or to follow a sizing
      policy (Kelly fraction, capital cap, etc.) implemented elsewhere.
  """
  order_size = size if size is not None else opp.max_size
  if order_size <= 0:
    raise ValueError(
      f"OrderRequest size must be > 0 (got {order_size}). "
      f"opp.max_size={opp.max_size}; pass size= to override."
    )

  return OrderRequest(
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


def from_opportunities(
  opps: list[ArbOpportunity],
  size: int | None = None,
) -> list[OrderRequest]:
  """Convenience: convert a list of opportunities to a list of orders."""
  return [from_opportunity(o, size=size) for o in opps]
