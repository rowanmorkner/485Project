"""
SQLite persistence layer for the bot's continuous-learning store.

All long-lived state (snapshots, forecasts, settlements, orders, fills, pnl)
goes here. Trading code reads frozen parameter files (data/*.json) — only
training jobs write to those. The DB is the single source of truth that
those parameter files are derived from.

Design choices:
  - Single SQLite file (`data/bot.db`). Portable, file-backed, zero ops.
  - WAL mode for concurrent readers + one writer (the bot's main loop
    writes snapshots while a separate cron job runs analytics queries).
  - Connections are short-lived; helpers open/close per call. Cheap on
    SQLite; avoids long-held writer locks.
  - JSON blobs for nested structures (forecast PDFs, brackets, opps).
    Querying inside JSON is rare; usually we just round-trip the dict.
"""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from contracts import ArbOpportunity, OrderRequest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "bot.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def _now_utc_iso() -> str:
  """Single source of truth for timestamp formatting."""
  return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
  """
  Yield a SQLite connection with foreign keys + WAL mode enabled.
  Each statement autocommits; explicit BEGIN/COMMIT not needed for our
  single-statement writes. Always closes on exit.
  """
  DB_PATH.parent.mkdir(parents=True, exist_ok=True)
  conn = sqlite3.connect(DB_PATH, isolation_level=None)  # autocommit
  conn.row_factory = sqlite3.Row
  conn.execute("PRAGMA foreign_keys = ON")
  conn.execute("PRAGMA journal_mode = WAL")
  try:
    yield conn
  finally:
    conn.close()


def init() -> None:
  """Create the database file and apply schema (idempotent)."""
  DB_PATH.parent.mkdir(parents=True, exist_ok=True)
  schema = SCHEMA_PATH.read_text()
  with connect() as conn:
    conn.executescript(schema)


# ── Writers ───────────────────────────────────────────────────────────────

def write_forecast(
  city: str,
  date: str,
  pdf: dict[int, float],
  source: str = "nws_normal",
  std_dev: float | None = None,
) -> int:
  """Insert a forecast row; returns the new row id."""
  with connect() as conn:
    cur = conn.execute(
      """
      INSERT OR IGNORE INTO forecasts
        (city, date, source, std_dev, pdf_json, fetched_at_utc)
      VALUES (?, ?, ?, ?, ?, ?)
      """,
      (city, date, source, std_dev,
       json.dumps({str(k): v for k, v in pdf.items()}),
       _now_utc_iso()),
    )
    return cur.lastrowid or 0


def write_snapshot(
  city: str,
  date: str,
  venue: str,
  brackets: list[dict],
) -> int:
  """
  Persist one venue's bracket snapshot for one (city, date). The brackets
  list is the same shape main.fetch_*_events produces — we just serialize.
  """
  with connect() as conn:
    cur = conn.execute(
      """
      INSERT INTO snapshots (city, date, venue, fetched_at_utc, brackets_json)
      VALUES (?, ?, ?, ?, ?)
      """,
      (city, date, venue, _now_utc_iso(), json.dumps(brackets)),
    )
    return cur.lastrowid or 0


def write_settlement(
  city: str,
  date: str,
  kalshi_high_f: int | None = None,
  kalshi_event_ticker: str | None = None,
  polymarket_high_f: int | None = None,
  polymarket_event_slug: str | None = None,
) -> int:
  """
  Insert or update the settlement row for one (city, date). Both venues
  may settle at different times — we use INSERT OR REPLACE to merge.
  """
  with connect() as conn:
    # Read existing row if present so we don't clobber the other venue's data
    existing = conn.execute(
      "SELECT * FROM settlements WHERE city=? AND date=?",
      (city, date),
    ).fetchone()

    merged = {
      "kalshi_high_f": kalshi_high_f if kalshi_high_f is not None
        else (existing["kalshi_high_f"] if existing else None),
      "kalshi_event_ticker": kalshi_event_ticker or (
        existing["kalshi_event_ticker"] if existing else None),
      "polymarket_high_f": polymarket_high_f if polymarket_high_f is not None
        else (existing["polymarket_high_f"] if existing else None),
      "polymarket_event_slug": polymarket_event_slug or (
        existing["polymarket_event_slug"] if existing else None),
    }

    cur = conn.execute(
      """
      INSERT INTO settlements
        (city, date, kalshi_high_f, kalshi_event_ticker,
         polymarket_high_f, polymarket_event_slug, recorded_at_utc)
      VALUES (?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT (city, date) DO UPDATE SET
        kalshi_high_f         = excluded.kalshi_high_f,
        kalshi_event_ticker   = excluded.kalshi_event_ticker,
        polymarket_high_f     = excluded.polymarket_high_f,
        polymarket_event_slug = excluded.polymarket_event_slug,
        recorded_at_utc       = excluded.recorded_at_utc
      """,
      (city, date,
       merged["kalshi_high_f"], merged["kalshi_event_ticker"],
       merged["polymarket_high_f"], merged["polymarket_event_slug"],
       _now_utc_iso()),
    )
    return cur.lastrowid or 0


def log_order(order: OrderRequest, arb_pair_id: str | None = None) -> int:
  """
  Persist one OrderRequest as a 'pending' row. The execution layer is
  responsible for updating status to 'filled'/'cancelled'/'rejected' and
  inserting fills via record_fill().

  City and date are denormalized off the embedded ArbOpportunity so the
  pnl join in bin/settle_orders.py is fast.

  arb_pair_id: shared UUID for the two legs of a paired-arb position.
    Pass None for single-leg value trades.
  """
  opp = order.source_opportunity
  if opp is None:
    raise ValueError("OrderRequest must carry source_opportunity for logging")
  with connect() as conn:
    cur = conn.execute(
      """
      INSERT INTO orders
        (client_order_id, created_at_utc, city, date, venue, market_id,
         side, action, intended_price, intended_size, expected_edge,
         opp_json, status, arb_pair_id)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
      """,
      (order.client_order_id, _now_utc_iso(),
       opp.city, opp.date, order.venue, order.market_id,
       order.side, order.action,
       order.price, order.size, order.expected_edge,
       json.dumps(asdict(opp)), arb_pair_id),
    )
    return cur.lastrowid or 0


def update_order_status(client_order_id: str, status: str) -> None:
  """Flip an order's status. Called by the execution layer."""
  if status not in ("pending", "filled", "cancelled", "rejected", "partial"):
    raise ValueError(f"Bad order status: {status}")
  with connect() as conn:
    conn.execute(
      "UPDATE orders SET status=? WHERE client_order_id=?",
      (status, client_order_id),
    )


def record_fill(
  client_order_id: str,
  fill_price: float,
  fill_size: int,
  fee_paid: float = 0.0,
) -> int:
  """
  Append a fill row for an order, looked up by client_order_id.
  Marks the order as 'filled' if it isn't already.
  """
  with connect() as conn:
    row = conn.execute(
      "SELECT id, status FROM orders WHERE client_order_id=?",
      (client_order_id,),
    ).fetchone()
    if not row:
      raise ValueError(f"No order found for client_order_id={client_order_id}")
    cur = conn.execute(
      """
      INSERT INTO fills (order_id, filled_at_utc, fill_price, fill_size, fee_paid)
      VALUES (?, ?, ?, ?, ?)
      """,
      (row["id"], _now_utc_iso(), fill_price, fill_size, fee_paid),
    )
    if row["status"] != "filled":
      conn.execute("UPDATE orders SET status='filled' WHERE id=?", (row["id"],))
    return cur.lastrowid or 0


def write_pnl(
  order_id: int,
  settlement_date: str,
  won: bool,
  dollars_realized: float,
  dollars_predicted: float,
) -> int:
  """One PnL row per order. Idempotent via UNIQUE(order_id) constraint."""
  with connect() as conn:
    cur = conn.execute(
      """
      INSERT OR IGNORE INTO pnl
        (order_id, settlement_date, won_bool, dollars_realized,
         dollars_predicted, computed_at_utc)
      VALUES (?, ?, ?, ?, ?, ?)
      """,
      (order_id, settlement_date, 1 if won else 0,
       dollars_realized, dollars_predicted, _now_utc_iso()),
    )
    return cur.lastrowid or 0


# ── Convenience readers ───────────────────────────────────────────────────

def latest_settlements(limit: int = 365) -> list[sqlite3.Row]:
  """All settlements newest-first; used by bin/refresh_risk_model.py."""
  with connect() as conn:
    return conn.execute(
      "SELECT * FROM settlements ORDER BY date DESC LIMIT ?",
      (limit,),
    ).fetchall()


def filled_orders_without_pnl() -> list[sqlite3.Row]:
  """Used by bin/settle_orders.py to find work."""
  with connect() as conn:
    return conn.execute(
      """
      SELECT o.*
      FROM orders o
      LEFT JOIN pnl p ON p.order_id = o.id
      WHERE o.status='filled' AND p.id IS NULL
      """,
    ).fetchall()


# ── CLI: initialize the database ──────────────────────────────────────────

if __name__ == "__main__":
  init()
  with connect() as conn:
    tables = [r[0] for r in conn.execute(
      "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()]
  print(f"Initialized {DB_PATH}")
  print(f"Tables: {tables}")
