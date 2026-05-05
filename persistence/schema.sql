-- SQLite schema for the weather-arbitrage bot's continuous-learning store.
--
-- Six tables; all rows immutable except for `orders.status`. Foreign keys are
-- declared but SQLite needs `PRAGMA foreign_keys = ON;` per-connection
-- (handled in persistence/db.py).
--
-- Naming: snake_case columns; *_at_utc for timestamps (ISO 8601 strings);
-- *_json for serialized blobs; *_f for Fahrenheit integer temperatures.

-- ── forecasts ────────────────────────────────────────────────────────────
-- One row per (city, date, source) — the forecast PDF that was current at
-- the time of fetch. `pdf_json` is a {temp_F: probability} dict.
CREATE TABLE IF NOT EXISTS forecasts (
  id              INTEGER PRIMARY KEY,
  city            TEXT NOT NULL,
  date            TEXT NOT NULL,            -- ISO YYYY-MM-DD (target measurement day)
  source          TEXT NOT NULL,            -- "nws_normal" | "ensemble" | etc.
  std_dev         REAL,                     -- forecast uncertainty in °F (nullable)
  pdf_json        TEXT NOT NULL,            -- JSON {"82": 0.05, "83": 0.18, ...}
  fetched_at_utc  TEXT NOT NULL,
  UNIQUE (city, date, source, fetched_at_utc)
);
CREATE INDEX IF NOT EXISTS idx_forecasts_city_date ON forecasts (city, date);

-- ── snapshots ────────────────────────────────────────────────────────────
-- One row per (city, date, venue, fetched_at_utc) — the venue's market
-- state at a point in time. `brackets_json` carries the full bracket list
-- including bid/ask/sizes/ladders.
CREATE TABLE IF NOT EXISTS snapshots (
  id              INTEGER PRIMARY KEY,
  city            TEXT NOT NULL,
  date            TEXT NOT NULL,            -- target measurement day
  venue           TEXT NOT NULL,            -- "kalshi" | "polymarket"
  fetched_at_utc  TEXT NOT NULL,
  brackets_json   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_snapshots_city_date_venue
  ON snapshots (city, date, venue, fetched_at_utc);

-- ── settlements ─────────────────────────────────────────────────────────
-- One row per (city, date) once both venues have settled. Captures the
-- actual reading reported by each venue, which feeds the per-city Δ
-- histogram in the risk model.
CREATE TABLE IF NOT EXISTS settlements (
  id                       INTEGER PRIMARY KEY,
  city                     TEXT NOT NULL,
  date                     TEXT NOT NULL,
  kalshi_high_f            INTEGER,
  kalshi_event_ticker      TEXT,
  polymarket_high_f        INTEGER,
  polymarket_event_slug    TEXT,
  recorded_at_utc          TEXT NOT NULL,
  UNIQUE (city, date)
);

-- ── orders ───────────────────────────────────────────────────────────────
-- Every OrderRequest the strategy generates is logged here as 'pending'.
-- The execution layer flips status to 'filled'/'cancelled'/'rejected'.
-- `opp_json` carries the full ArbOpportunity for traceability.
CREATE TABLE IF NOT EXISTS orders (
  id                INTEGER PRIMARY KEY,
  client_order_id   TEXT NOT NULL UNIQUE,   -- caller-generated UUID
  created_at_utc    TEXT NOT NULL,
  city              TEXT NOT NULL,          -- denormalized from opp for join speed
  date              TEXT NOT NULL,          -- denormalized from opp for join speed
  venue             TEXT NOT NULL,
  market_id         TEXT NOT NULL,
  side              TEXT NOT NULL,          -- "yes" | "no"
  action            TEXT NOT NULL,          -- "buy" | "sell"
  intended_price    REAL NOT NULL,
  intended_size     INTEGER NOT NULL,
  expected_edge     REAL NOT NULL,          -- predicted_edge from the opportunity
  opp_json          TEXT NOT NULL,
  status            TEXT NOT NULL DEFAULT 'pending',
  -- Paired-arb grouping. NULL for single-leg value trades; shared UUID
  -- across the two legs of a paired-arb position. P&L aggregation joins
  -- on this column so a hedged position is scored as one trade, not two.
  arb_pair_id       TEXT
);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders (status);
CREATE INDEX IF NOT EXISTS idx_orders_city_date ON orders (city, date);
CREATE INDEX IF NOT EXISTS idx_orders_arb_pair_id ON orders (arb_pair_id);

-- ── fills ────────────────────────────────────────────────────────────────
-- One row per execution leg of an order. Multi-fill orders allowed via
-- multiple rows. Used both for slippage analysis and for PnL bookkeeping.
CREATE TABLE IF NOT EXISTS fills (
  id              INTEGER PRIMARY KEY,
  order_id        INTEGER NOT NULL REFERENCES orders(id),
  filled_at_utc   TEXT NOT NULL,
  fill_price      REAL NOT NULL,
  fill_size       INTEGER NOT NULL,
  fee_paid        REAL NOT NULL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_fills_order_id ON fills (order_id);

-- ── pnl ──────────────────────────────────────────────────────────────────
-- One row per order, written by bin/settle_orders.py once the underlying
-- (city, date) market settles. Compares realized vs predicted edge.
CREATE TABLE IF NOT EXISTS pnl (
  id                  INTEGER PRIMARY KEY,
  order_id            INTEGER NOT NULL UNIQUE REFERENCES orders(id),
  settlement_date     TEXT NOT NULL,
  won_bool            INTEGER NOT NULL,     -- 0 | 1 (SQLite has no bool)
  dollars_realized    REAL NOT NULL,        -- (1 if won else 0) - fill_price - fee
  dollars_predicted   REAL NOT NULL,        -- expected_edge from orders
  computed_at_utc     TEXT NOT NULL
);
