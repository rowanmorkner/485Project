"""
Dashboard data layer — read-only queries against data/bot.db.

All public functions are wrapped in `st.cache_data(ttl=...)` so the UI stays
responsive while the underlying SQLite file is being written to by the live
bot. The cache is deliberately short (a few seconds) — the dashboard is
meant to look "live."
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "bot.db"
LEDGER_PATH = PROJECT_ROOT / "paper_trades.jsonl"

# Only the current cross-venue HedgedPair strategy persists `kalshi_market_id`
# inside opp_json. The old single-leg market-making strategy used a different
# shape — those orders are excluded from every counter so the dashboard
# reflects the deployed strategy's performance, not historical noise from
# deprecated code paths.
NEW_STRATEGY_FILTER = (
  "json_extract(o.opp_json, '$.kalshi_market_id') IS NOT NULL"
)


# ── Connection helper ────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
  """Read-only connection — the dashboard never writes."""
  conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
  conn.row_factory = sqlite3.Row
  return conn


# ── Health / overview ────────────────────────────────────────────────────

@st.cache_data(ttl=10)
def health() -> dict:
  """Headline counters + most-recent timestamps for the bot health row.
  Order/fill/pnl counts are scoped to the deployed (HedgedPair) strategy."""
  with _connect() as c:
    snap_n = c.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
    fc_n = c.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
    settle_n = c.execute("SELECT COUNT(*) FROM settlements").fetchone()[0]
    orders_n = c.execute(
      f"SELECT COUNT(*) FROM orders o WHERE {NEW_STRATEGY_FILTER}"
    ).fetchone()[0]
    fills_n = c.execute(
      f"SELECT COUNT(*) FROM fills f JOIN orders o ON o.id=f.order_id "
      f"WHERE {NEW_STRATEGY_FILTER}"
    ).fetchone()[0]
    pnl_n = c.execute(
      f"SELECT COUNT(*) FROM pnl p JOIN orders o ON o.id=p.order_id "
      f"WHERE {NEW_STRATEGY_FILTER}"
    ).fetchone()[0]
    last_snapshot = c.execute(
      "SELECT MAX(fetched_at_utc) FROM snapshots").fetchone()[0]
    last_forecast = c.execute(
      "SELECT MAX(fetched_at_utc) FROM forecasts").fetchone()[0]
    last_fill = c.execute(
      f"SELECT MAX(f.filled_at_utc) FROM fills f JOIN orders o ON o.id=f.order_id "
      f"WHERE {NEW_STRATEGY_FILTER}"
    ).fetchone()[0]
    last_settlement = c.execute(
      "SELECT MAX(recorded_at_utc) FROM settlements").fetchone()[0]
    cities = [r[0] for r in c.execute(
      "SELECT DISTINCT city FROM snapshots ORDER BY city")]
  return {
    "snapshots": snap_n, "forecasts": fc_n, "settlements": settle_n,
    "orders": orders_n, "fills": fills_n, "pnl": pnl_n,
    "last_snapshot_utc": last_snapshot,
    "last_forecast_utc": last_forecast,
    "last_fill_utc": last_fill,
    "last_settlement_utc": last_settlement,
    "cities": cities,
  }


# ── P&L ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=10)
def pnl_frame() -> pd.DataFrame:
  """One row per settled order — joined with order metadata so we can
  segment by city / venue / side downstream. Scoped to the deployed strategy."""
  with _connect() as c:
    return pd.read_sql_query(
      f"""
      SELECT p.computed_at_utc AS ts, p.settlement_date,
             p.won_bool, p.dollars_realized, p.dollars_predicted,
             o.city, o.venue, o.side, o.market_id, o.intended_price,
             o.intended_size, o.arb_pair_id
      FROM pnl p JOIN orders o ON o.id = p.order_id
      WHERE {NEW_STRATEGY_FILTER}
      ORDER BY p.computed_at_utc
      """, c, parse_dates=["ts", "settlement_date"])


@st.cache_data(ttl=10)
def fills_frame() -> pd.DataFrame:
  """All fills with their order context. Scoped to the deployed strategy."""
  with _connect() as c:
    return pd.read_sql_query(
      f"""
      SELECT f.filled_at_utc AS ts, f.fill_price, f.fill_size, f.fee_paid,
             o.client_order_id, o.city, o.date, o.venue, o.side, o.action,
             o.market_id, o.intended_price, o.expected_edge, o.arb_pair_id,
             o.status
      FROM fills f JOIN orders o ON o.id = f.order_id
      WHERE {NEW_STRATEGY_FILTER}
      ORDER BY f.filled_at_utc DESC
      """, c, parse_dates=["ts"])


# ── Snapshots / forecasts ────────────────────────────────────────────────

@st.cache_data(ttl=10)
def available_city_dates() -> list[tuple[str, str]]:
  """(city, date) pairs that have a snapshot from BOTH venues — these are
  the only rows where the cross-venue strategy is live."""
  with _connect() as c:
    rows = c.execute(
      """
      SELECT city, date
      FROM snapshots
      GROUP BY city, date
      HAVING COUNT(DISTINCT venue) = 2
      ORDER BY date DESC, city
      """).fetchall()
  return [(r["city"], r["date"]) for r in rows]


@st.cache_data(ttl=10)
def latest_brackets(city: str, date: str, venue: str) -> tuple[list[dict], str | None]:
  """Latest snapshot's brackets for one (city, date, venue). Returns
  (brackets, fetched_at_utc) or ([], None) if no snapshot exists."""
  with _connect() as c:
    row = c.execute(
      """
      SELECT brackets_json, fetched_at_utc FROM snapshots
      WHERE city=? AND date=? AND venue=?
      ORDER BY fetched_at_utc DESC LIMIT 1
      """, (city, date, venue)).fetchone()
  if not row:
    return [], None
  return json.loads(row["brackets_json"]), row["fetched_at_utc"]


@st.cache_data(ttl=10)
def latest_forecast(city: str, date: str,
                    source: str | None = None) -> tuple[dict[int, float], str, str | None]:
  """Latest forecast PDF for (city, date), optionally filtered by source.
  Returns (pdf, source, fetched_at_utc) — pdf is {} if none found."""
  q = ("SELECT pdf_json, source, fetched_at_utc FROM forecasts "
       "WHERE city=? AND date=?")
  params: list = [city, date]
  if source:
    q += " AND source=?"
    params.append(source)
  q += " ORDER BY fetched_at_utc DESC LIMIT 1"
  with _connect() as c:
    row = c.execute(q, params).fetchone()
  if not row:
    return {}, "", None
  raw = json.loads(row["pdf_json"])
  return ({int(k): v for k, v in raw.items()},
          row["source"], row["fetched_at_utc"])


@st.cache_data(ttl=30)
def settlements_frame() -> pd.DataFrame:
  with _connect() as c:
    return pd.read_sql_query(
      "SELECT * FROM settlements ORDER BY date DESC", c,
      parse_dates=["date", "recorded_at_utc"])


@st.cache_data(ttl=30)
def settlement_for(city: str, date: str) -> tuple[int | None, int | None]:
  """Per-venue settled high for one (city, date), or (None, None) if not
  yet resolved on either side."""
  with _connect() as c:
    row = c.execute(
      "SELECT kalshi_high_f, polymarket_high_f FROM settlements "
      "WHERE city=? AND date=?", (city, date)).fetchone()
  if not row:
    return None, None
  return row["kalshi_high_f"], row["polymarket_high_f"]


# ── Backtest ─────────────────────────────────────────────────────────────

BACKTEST_PATH = PROJECT_ROOT / "data" / "backtest_results.parquet"


@st.cache_data(ttl=30)
def backtest_frame() -> pd.DataFrame:
  """Replayed strategy results from bin/backtest_strategy.py. Returns an
  empty DataFrame if the parquet hasn't been generated yet."""
  if not BACKTEST_PATH.exists():
    return pd.DataFrame()
  df = pd.read_parquet(BACKTEST_PATH)
  if "snapshot_ts" in df.columns:
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"])
  return df
