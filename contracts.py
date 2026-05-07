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
  """One venue's executable view of a single bracket market.

  `degrees` covers the closed degrees (or the TAIL_SPREAD-narrow window
  used for visualization); the `tail_below` / `tail_above` flags + `boundary`
  are what the strategy uses to score open-ended brackets correctly. A
  "72° or below" bracket has degrees=[67..72] (for legacy compatibility)
  AND tail_below=True with boundary=72, so YES wins for ANY high ≤ 72,
  not just highs in [67..72]. See yes_predicate() for the win rule used
  by the trading layer.
  """
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
  tail_below: bool = False        # True for "X° or below" — YES wins iff k ≤ boundary
  tail_above: bool = False        # True for "X° or above" — YES wins iff k ≥ boundary
  boundary: int | None = None     # threshold integer for tail brackets

  def yes_predicate(self):
    """Return a fn(k) -> bool that's True iff a YES bet wins at high=k."""
    if self.tail_below:
      bnd = self.boundary
      return lambda k: k <= bnd
    if self.tail_above:
      bnd = self.boundary
      return lambda k: k >= bnd
    s = frozenset(self.degrees)
    return lambda k: k in s


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
  kalshi_tail_below: bool = False
  kalshi_tail_above: bool = False
  kalshi_boundary: int | None = None
  # Polymarket leg
  poly_market_id: str = ""
  poly_condition_id: str = ""
  poly_label: str = ""
  poly_degrees: list[int] = field(default_factory=list)
  poly_side: str = ""
  poly_avg_fill: float = 0.0
  poly_tail_below: bool = False
  poly_tail_above: bool = False
  poly_boundary: int | None = None
  # Sizing & payoff stats (under joint K-P)
  size: int = 0                   # contracts per leg
  cost_per_pair: float = 0.0
  expected_payoff: float = 0.0    # E[$/pair]
  q05_payoff: float = 0.0         # 5th percentile $/pair
  worst_case_payoff: float = 0.0


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
