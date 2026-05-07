"""
Adapter from internal OrderRequest → paper_trading_bot's external schema.

paper_trading_bot.CrossPlatformPaperTrader is the live-capital execution
path; this module bridges our typed OrderRequests into the dict format
its `process_order` expects, then mirrors the realized fill back into
bot.db so the existing settlement / pnl pipeline can score it.
"""

import logging
from typing import Optional

from contracts import OrderRequest
from paper_trading_bot import CrossPlatformPaperTrader, PaperOrder, append_ledger
from persistence import db

logger = logging.getLogger(__name__)


_DEFAULT_LEDGER = "paper_trades.jsonl"

# Lazy singleton — KalshiClient and PolymarketClient construction reads
# env vars + private keys, so we want to do it once.
_trader: CrossPlatformPaperTrader | None = None


def _get_trader() -> CrossPlatformPaperTrader:
  global _trader
  if _trader is None:
    _trader = CrossPlatformPaperTrader()
  return _trader


def execute_order(
  order: OrderRequest,
  ledger_path: str = _DEFAULT_LEDGER,
) -> Optional[PaperOrder]:
  """Translate OrderRequest → paper_trading_bot dict, fire it, mirror fill.

  Returns the PaperOrder produced by the trader (None if it couldn't fill,
  e.g. no executable price).
  """
  if order.venue == "kalshi":
    ext = {
      "platform": "kalshi",
      "ticker": order.market_id,
      "action": order.action,
      "side": order.side,
      "count": order.size,
      "type": "limit",
    }
  else:
    # paper_trading_bot accepts either condition_id or token_id; we have
    # both, but token_id is the direct CLOB asset and avoids a side lookup.
    ext = {
      "platform": "polymarket",
      "token_id": order.market_id,
      "action": order.action,
      "side": order.side,
      "count": order.size,
      "type": "limit",
    }

  paper = _get_trader().process_order(ext)
  if paper is None:
    return None

  append_ledger(ledger_path, [paper])
  try:
    db.record_fill(
      client_order_id=order.client_order_id,
      fill_price=paper.fill_price,
      fill_size=paper.count,
      fee_paid=0.0,
    )
  except Exception as exc:
    logger.warning("Failed to mirror paper fill into db: %s", exc)
  return paper
