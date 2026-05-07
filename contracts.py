"""
Shared data contracts between strategy and execution layers.

Forecast PDFs are passed as plain dict[int, float]; the only types that
need to cross module boundaries are the venue quote (input to strategy)
and the resulting trade artifacts (output of strategy).
"""

from dataclasses import dataclass, field


# ── Strategy input: parsed venue snapshot ─────────────────────────────────

@dataclass
class BracketQuote:
  """One venue's executable view of a single bracket market."""
  venue: str                      # "kalshi" | "polymarket"
  market_id: str                  # Kalshi ticker / Polymarket YES token_id
  label: str                      # human-readable bin (e.g. "81° to 82°")
  degrees: list[int]              # integer °F covered by this bracket
  best_bid: float | None
  best_ask: float | None
  bid_size: float
  ask_size: float
  condition_id: str = ""          # Polymarket conditionId; "" for Kalshi
  ladder_bids: list[tuple[float, float]] = field(default_factory=list)
  ladder_asks: list[tuple[float, float]] = field(default_factory=list)


# ── Strategy output: a hedged cross-venue pair that passes the CVaR filter ─

@dataclass
class HedgedPair:
  """A 2-leg cross-venue position passing the joint-K-P CVaR filter."""
  city: str
  date: str
  # Kalshi leg
  kalshi_market_id: str
  kalshi_label: str
  kalshi_degrees: list[int]
  kalshi_side: str                # 'yes' | 'no'
  kalshi_avg_fill: float          # avg fill price after walking the ask ladder
  # Polymarket leg
  poly_market_id: str
  poly_condition_id: str
  poly_label: str
  poly_degrees: list[int]
  poly_side: str
  poly_avg_fill: float
  # Sizing & payoff stats (under joint K-P)
  size: int                       # contracts per leg
  cost_per_pair: float
  expected_payoff: float          # E[$/pair]
  q05_payoff: float               # 5th percentile $/pair
  worst_case_payoff: float


# ── Strategy → Execution handoff ──────────────────────────────────────────

@dataclass
class OrderRequest:
  """A single venue order ready to be placed."""
  venue: str                      # "kalshi" | "polymarket"
  market_id: str                  # Kalshi ticker, OR Polymarket token_id
  side: str                       # "yes" | "no"
  action: str                     # "buy" | "sell"
  price: float                    # 0.0–1.0
  size: int
  expected_edge: float            # for logging / sanity check
  client_order_id: str
  condition_id: str
  source_opportunity: "HedgedPair | None" = None
