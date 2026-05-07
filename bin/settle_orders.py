"""
Daily cron job: for every filled order without a pnl row yet, look up the
settled high temperature for that (city, date), determine win/loss, and
write a pnl row.

Win condition: the venue's actual reading falls inside the bracket's
degree range. Δ-divergence between venues is irrelevant here — what
matters is what the trade's OWN venue settled to.

  won = (venue_high in opp.degrees)
  dollars_realized = (1.0 if won else 0.0) − fill_price − fee
                     summed across all fills for the order
  dollars_predicted = expected_edge × intended_size  (per-contract → notional)

Run:
  python -m bin.settle_orders
"""

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from persistence import db


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def venue_high_for(conn, city: str, date: str, venue: str) -> int | None:
  """Return the venue's settled high °F for one (city, date), or None."""
  row = conn.execute(
    "SELECT kalshi_high_f, polymarket_high_f FROM settlements WHERE city=? AND date=?",
    (city, date),
  ).fetchone()
  if not row:
    return None
  if venue == "kalshi":
    return row["kalshi_high_f"]
  if venue == "polymarket":
    return row["polymarket_high_f"]
  return None


def realized_dollars(action: str, won: bool, fill_price: float, fill_size: int,
                     fee_paid: float) -> float:
  """
  Dollars in/out for one fill leg:

    BUY  YES @ p, won → received $1, paid p+fee per contract → +(1 - p) - fee
    BUY  YES @ p, lost → received $0, paid p+fee                 → -(p + fee)
    SELL YES @ p, won → already had liability of $1, received p-fee
                       → -(1 - p) - fee   (i.e. paid 1, received p, paid fee)
    SELL YES @ p, lost → received p, paid fee                    → +(p - fee)
  """
  if action == "buy":
    payoff = (1.0 if won else 0.0) - fill_price
  elif action == "sell":
    payoff = fill_price - (1.0 if won else 0.0)
  else:
    raise ValueError(f"Unknown action {action!r}")
  return (payoff * fill_size) - fee_paid


def main():
  db.init()  # idempotent
  pending = db.filled_orders_without_pnl()
  log.info("Found %d filled orders without pnl rows", len(pending))

  written = 0
  with db.connect() as conn:
    for o in pending:
      high = venue_high_for(conn, o["city"], o["date"], o["venue"])
      if high is None:
        continue  # settlement not yet recorded for that (city, date)
      try:
        opp = json.loads(o["opp_json"])
        bracket_degrees = opp.get("degrees", [])
      except (TypeError, ValueError):
        log.warning("Order %d: failed to parse opp_json", o["id"])
        continue

      # YES side wins iff actual ∈ bracket; NO side wins iff actual ∉ bracket.
      in_bracket = high in bracket_degrees
      won = in_bracket if o["side"] == "yes" else not in_bracket

      # Sum across all fills for this order (multi-fill safe)
      fills = conn.execute(
        "SELECT fill_price, fill_size, fee_paid FROM fills WHERE order_id=?",
        (o["id"],),
      ).fetchall()
      if not fills:
        log.warning("Order %d marked filled but has no fills row; skipping", o["id"])
        continue

      realized = sum(
        realized_dollars(o["action"], won, f["fill_price"], f["fill_size"],
                         f["fee_paid"])
        for f in fills
      )
      # Predicted: edge per contract × total filled size
      total_size = sum(f["fill_size"] for f in fills)
      predicted = o["expected_edge"] * total_size

      db.write_pnl(
        order_id=o["id"],
        settlement_date=o["date"],
        won=won,
        dollars_realized=realized,
        dollars_predicted=predicted,
      )
      written += 1
      log.info(
        "Order %d (%s %s %s @%.4f): won=%s realized=%+.4f predicted=%+.4f",
        o["id"], o["venue"], o["action"], o["market_id"],
        o["intended_price"], won, realized, predicted,
      )

  log.info("Wrote %d pnl rows", written)


if __name__ == "__main__":
  main()
