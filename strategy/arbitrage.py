"""
Cross-venue hedged-pair selector.

For each (kalshi_quote, kalshi_side, poly_quote, poly_side) candidate,
walk both ASK ladders to size `size` contracts, score the per-pair payoff
under the empirical joint K-P distribution, and accept iff
  q05_payoff > entry_cost AND  E[payoff] > entry_cost + safety_margin.

Each Kalshi (market_id, side) and each Polymarket (market_id, side) is
opened at most once per (city, date) snapshot.
"""

import logging
import uuid

from contracts import BracketQuote, HedgedPair
from strategy.risk import (
  joint_kp_distribution,
  expected_payoff,
  quantile_payoff,
)

logger = logging.getLogger(__name__)


# ── Ladder fills ─────────────────────────────────────────────────────────

def walk_ladder_buy(ask_ladder: list[tuple[float, float]], qty: int
                    ) -> tuple[float, float] | None:
  """Walk an ASK ladder to BUY `qty` contracts.

  Returns (avg_price, total_cost) or None if depth insufficient.
  Sorts ladder ascending defensively.
  """
  if not ask_ladder or qty <= 0:
    return None
  remaining = float(qty)
  total_cost = 0.0
  levels = sorted(
    [(float(p), float(q)) for p, q in ask_ladder if q and q > 0]
  )
  for price, sz in levels:
    take = min(sz, remaining)
    total_cost += take * price
    remaining -= take
    if remaining <= 1e-9:
      return (total_cost / qty, total_cost)
  return None


def _ask_ladder_for_side(quote: BracketQuote, side: str) -> list[tuple[float, float]]:
  """The ladder you'd lift to BUY this side.

  YES → quote.ladder_asks
  NO  → synthesized as [(1-p, q) for p, q in quote.ladder_bids]
        (buying NO = selling YES into the YES bid)
  """
  if side == "yes":
    return list(quote.ladder_asks or [])
  return [(round(1.0 - float(p), 4), float(q))
          for p, q in (quote.ladder_bids or []) if q and q > 0]


# ── Main entry point ─────────────────────────────────────────────────────

def find_hedged_pairs(
  kalshi_quotes: list[BracketQuote],
  poly_quotes: list[BracketQuote],
  forecast_pdf: dict[int, float],
  city: str,
  date: str,
  size: int = 50,
  fee_per_contract: float = 0.005,
  safety_margin: float = 0.05,
) -> list[HedgedPair]:
  """Enumerate cross-venue (k_quote, k_side, p_quote, p_side) candidates,
  walk both ladders for `size` contracts, and accept those whose joint
  K-P payoff distribution satisfies the CVaR + expectation filters.
  """
  if not kalshi_quotes or not poly_quotes or not forecast_pdf:
    return []

  joint = joint_kp_distribution(forecast_pdf, city=city)
  if not joint:
    return []

  fees_per_pair = 2.0 * fee_per_contract  # one fee per leg
  candidates: list[HedgedPair] = []

  for kq in kalshi_quotes:
    k_yes_pred = kq.yes_predicate()
    for k_side in ("yes", "no"):
      k_fill = walk_ladder_buy(_ask_ladder_for_side(kq, k_side), size)
      if k_fill is None:
        continue
      k_avg, k_cost = k_fill
      k_won = (k_yes_pred if k_side == "yes"
               else (lambda k, _y=k_yes_pred: not _y(k)))

      for pq in poly_quotes:
        p_yes_pred = pq.yes_predicate()
        for p_side in ("yes", "no"):
          p_fill = walk_ladder_buy(_ask_ladder_for_side(pq, p_side), size)
          if p_fill is None:
            continue
          p_avg, p_cost = p_fill
          p_won = (p_yes_pred if p_side == "yes"
                   else (lambda p, _y=p_yes_pred: not _y(p)))

          # Cost & payoff are per-pair (one Kalshi + one Polymarket contract)
          cost_per_pair = k_avg + p_avg + fees_per_pair

          def payoff_fn(k: int, p: int, kw=k_won, pw=p_won) -> float:
            return float(kw(k)) + float(pw(p))

          ev = expected_payoff(payoff_fn, joint)
          q05 = quantile_payoff(payoff_fn, joint, q=0.05)

          if q05 <= cost_per_pair:
            continue
          if ev <= cost_per_pair + safety_margin:
            continue

          worst = min(payoff_fn(k, p) for (k, p) in joint)

          candidates.append(HedgedPair(
            city=city, date=date,
            kalshi_market_id=kq.market_id,
            kalshi_label=kq.label,
            kalshi_degrees=list(kq.degrees),
            kalshi_side=k_side,
            kalshi_avg_fill=round(k_avg, 4),
            kalshi_tail_below=kq.tail_below,
            kalshi_tail_above=kq.tail_above,
            kalshi_boundary=kq.boundary,
            poly_market_id=pq.market_id,
            poly_condition_id=pq.condition_id,
            poly_label=pq.label,
            poly_degrees=list(pq.degrees),
            poly_side=p_side,
            poly_avg_fill=round(p_avg, 4),
            poly_tail_below=pq.tail_below,
            poly_tail_above=pq.tail_above,
            poly_boundary=pq.boundary,
            size=size,
            cost_per_pair=round(cost_per_pair, 4),
            expected_payoff=round(ev, 4),
            q05_payoff=round(q05, 4),
            worst_case_payoff=round(worst, 4),
          ))

  # Rank by net E[payoff] - cost
  candidates.sort(key=lambda h: h.expected_payoff - h.cost_per_pair, reverse=True)

  # Strict per-leg dedup: each (kalshi_market_id, kalshi_side) and each
  # (poly_market_id, poly_side) accepted at most once per (city, date).
  used_k: set[tuple[str, str]] = set()
  used_p: set[tuple[str, str]] = set()
  selected: list[HedgedPair] = []
  for h in candidates:
    k_key = (h.kalshi_market_id, h.kalshi_side)
    p_key = (h.poly_market_id, h.poly_side)
    if k_key in used_k or p_key in used_p:
      continue
    used_k.add(k_key)
    used_p.add(p_key)
    selected.append(h)

  return selected
