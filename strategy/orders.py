"""
Build OrderRequest pairs from HedgedPair selections.

Sizing and fill price already came from walking the ASK ladder inside
strategy.arbitrage.find_hedged_pairs, so this module is pure plumbing:
it produces two OrderRequest objects sharing an arb_pair_id and logs
them to bot.db. Real (or paper) fills come from
strategy.execution.execute_order, NOT from this module.
"""

import logging
import uuid

from contracts import HedgedPair, OrderRequest

logger = logging.getLogger(__name__)


def from_hedged_pair(pair: HedgedPair, log_to_db: bool = True) -> list[OrderRequest]:
  """Two OrderRequests sharing an arb_pair_id, ready for execution."""
  pair_id = str(uuid.uuid4())
  per_leg_edge = round(
    (pair.expected_payoff - pair.cost_per_pair) / 2.0, 4
  )

  kalshi_order = OrderRequest(
    venue="kalshi",
    market_id=pair.kalshi_market_id,
    side=pair.kalshi_side,
    action="buy",
    price=pair.kalshi_avg_fill,
    size=pair.size,
    expected_edge=per_leg_edge,
    client_order_id=f"{pair_id}::kalshi",
    condition_id="",
    source_opportunity=pair,
  )
  poly_order = OrderRequest(
    venue="polymarket",
    market_id=pair.poly_market_id,
    side=pair.poly_side,
    action="buy",
    price=pair.poly_avg_fill,
    size=pair.size,
    expected_edge=per_leg_edge,
    client_order_id=f"{pair_id}::poly",
    condition_id=pair.poly_condition_id,
    source_opportunity=pair,
  )

  if log_to_db:
    try:
      from persistence import db as persist
      persist.log_order(kalshi_order, arb_pair_id=pair_id)
      persist.log_order(poly_order, arb_pair_id=pair_id)
    except Exception as exc:
      logger.warning("Failed to log hedged pair to DB: %s", exc)

  return [kalshi_order, poly_order]
