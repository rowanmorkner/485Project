"""
Shared data contracts between the three project modules.

These dataclasses define the seams between teammate work areas:

  forecast/  ──Forecast──▶  strategy/  ──OrderRequest──▶  execution/

Each teammate consumes upstream contracts and produces downstream ones,
so changes here ripple to everyone — keep them stable and well-documented.
"""

from dataclasses import dataclass, field


# ── Derek's output: ensemble forecast PDF ─────────────────────────────────

@dataclass
class Forecast:
  """
  Output of the ensemble forecasting model (Derek's domain).

  One Forecast covers a single (city, date) pair and gives a per-degree
  probability mass function over the daily high temperature.
  """
  city: str                       # e.g. "Miami"
  date: str                       # ISO date string, e.g. "2026-04-27"
  pdf: dict[int, float]           # {temp_F: probability}, must sum to 1.0
  model_name: str = "ensemble"    # tag for which model produced it
  std_dev: float | None = None    # optional uncertainty estimate, °F


# ── Strategy's internal: parsed venue snapshot ────────────────────────────

@dataclass
class BracketQuote:
  """
  One venue's executable view of a single bracket market.

  This is the unit at which trades actually happen — you buy or sell N
  contracts of a specific bracket at a specific price. Cross-venue / vs-
  forecast comparisons in arb logic operate on these.
  """
  venue: str                      # "kalshi" | "polymarket"
  market_id: str                  # Kalshi ticker / Polymarket YES token_id
  label: str                      # human-readable bin (e.g. "81° to 82°")
  degrees: list[int]              # integer °F covered by this bracket
  best_bid: float | None          # YES bid in dollars (proceeds when selling 1)
  best_ask: float | None          # YES ask in dollars (cost when buying 1)
  bid_size: float                 # contracts/shares resting at best_bid
  ask_size: float                 # contracts/shares resting at best_ask
  ladder_bids: list[tuple[float, float]] = field(default_factory=list)
  ladder_asks: list[tuple[float, float]] = field(default_factory=list)
  # ladder_* are full books (price, size) ordered worst-to-best, useful
  # for slippage modeling when sizing beyond top-of-book.


@dataclass
class MarketSnapshot:
  """
  A single venue's view of a (city, date) market, normalized into a
  per-degree implied probability distribution.

  Built by strategy/parsers.py from raw client responses. Used internally
  for forecast comparison and the per-degree discrepancy display.
  """
  venue: str                      # "kalshi" | "polymarket"
  city: str
  date: str                       # ISO date string
  bins: dict[int, float]          # {temp_F: implied_probability}
  raw_brackets: list[dict] = field(default_factory=list)


# ── Strategy's output: a flagged arbitrage opportunity ────────────────────

@dataclass
class ArbOpportunity:
  """
  A trade signal large enough (after fees) to act on. Produced by
  strategy/arbitrage.py.

  Today these are per-bracket value trades against the forecast PDF
  (one venue + one direction). Cross-venue locked-arb would require
  bracket-boundary alignment, which Kalshi and Polymarket don't share.
  """
  city: str
  date: str
  trade_type: str                 # "VALUE_BUY" | "VALUE_SELL"
  venue: str                      # "kalshi" | "polymarket"
  market_id: str                  # bracket market ID on that venue
  bracket_label: str              # human-readable bin label
  degrees: list[int]              # degrees covered by this bracket
  side: str                       # "yes" | "no"
  action: str                     # "buy" | "sell"
  price: float                    # executable price (best_ask for buy, best_bid for sell)
  fair_value: float               # forecast EV for this bracket (sum of pdf over degrees)
  net_edge: float                 # |fair - price| - fees, in dollars per contract
  max_size: int                   # top-of-book depth cap (contracts)


# ── Strategy → Execution handoff ──────────────────────────────────────────

@dataclass
class OrderRequest:
  """
  A single venue order ready to be placed (teammate 3's input).

  This is intentionally venue-neutral — the execution layer is responsible
  for translating into venue-specific API payloads (Kalshi REST POST,
  Polymarket EIP-712 signed CLOB order, etc.).
  """
  venue: str                      # "kalshi" | "polymarket"
  market_id: str                  # Kalshi market ticker, OR Polymarket token_id
  side: str                       # "yes" | "no"
  action: str                     # "buy" | "sell"
  price: float                    # 0.0–1.0 (probability / dollar fraction)
  size: int                       # contracts (Kalshi) or shares (Polymarket)
  expected_edge: float            # for logging / sanity-check by execution
  client_order_id: str            # caller-generated UUID for deduplication
  source_opportunity: ArbOpportunity | None = None  # for traceability
