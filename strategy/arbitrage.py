"""
Arbitrage / value-trade detection.

Two distinct functions live here:

  find_discrepancies()    — diagnostic, per-degree midpoint comparison.
                            Highlights where venues disagree but does NOT
                            specify executable trades. Cheap to compute,
                            useful as a sanity check / display feed.

  find_value_trades()     — actionable, per-bracket comparison against the
                            forecast PDF using EXECUTABLE prices (best_ask
                            for buys, best_bid for sells) and DEPTH-CAPPED
                            sizing (top of book).

Why not "lock-in cross-venue arb"? Kalshi and Polymarket use different
bracket boundaries (Kalshi: 81-82, 83-84, ...; Polymarket: 80-81, 82-83, ...),
so a paired buy/sell across venues never has matched legs — at most degrees
you'd carry directional exposure. The honest framing is per-bracket value
trades against an external fair-value estimate (forecast PDF), which is
what find_value_trades does. If both venues are mispriced relative to
forecast in opposite directions, you naturally get one trade on each venue
and the result resembles a cross-venue arb without pretending the legs are
mechanically locked.
"""

import logging
from typing import Iterable

from contracts import ArbOpportunity, BracketQuote

logger = logging.getLogger(__name__)


# ── Diagnostic: per-degree spread between sources ────────────────────────

def find_discrepancies(
  kalshi_dist: dict[int, float],
  poly_dist: dict[int, float],
  forecast_dist: dict[int, float],
  min_edge: float = 0.05,
) -> list[dict]:
  """
  At each degree present in any source, compute the spread (max - min)
  of implied probabilities and flag rows where it exceeds min_edge.
  Diagnostic only — see find_value_trades() for tradeable signals.
  """
  all_degrees = sorted(
    set(kalshi_dist.keys()) | set(poly_dist.keys()) | set(forecast_dist.keys())
  )

  discrepancies = []
  for deg in all_degrees:
    k_prob = kalshi_dist.get(deg, 0.0)
    p_prob = poly_dist.get(deg, 0.0)
    f_prob = forecast_dist.get(deg, 0.0)

    probs = [k_prob, p_prob, f_prob]
    max_spread = max(probs) - min(probs)

    if max_spread >= min_edge:
      sources = []
      if abs(k_prob - p_prob) >= min_edge:
        sources.append("kalshi-vs-poly")
      if abs(k_prob - f_prob) >= min_edge:
        sources.append("kalshi-vs-forecast")
      if abs(p_prob - f_prob) >= min_edge:
        sources.append("poly-vs-forecast")

      discrepancies.append({
        "degree": deg,
        "kalshi_prob": round(k_prob, 4),
        "poly_prob": round(p_prob, 4),
        "forecast_prob": round(f_prob, 4),
        "max_spread": round(max_spread, 4),
        "sources_disagreeing": sources,
      })

  discrepancies.sort(key=lambda d: d["max_spread"], reverse=True)
  return discrepancies


# ── Actionable: per-bracket value trades using forecast PDF ──────────────

def find_value_trades(
  quotes: Iterable[BracketQuote],
  forecast_pdf: dict[int, float],
  city: str,
  date: str,
  fee_rate: dict[str, float] | None = None,
  min_edge: float = 0.03,
) -> list[ArbOpportunity]:
  """
  For each bracket on each venue, compute the bracket's fair value as the
  sum of forecast probability mass over the degrees it covers. Compare
  against the executable price (best_ask for buying, best_bid for selling)
  and emit a trade signal if the after-fee edge clears min_edge.

  Args:
    quotes: BracketQuote objects from one or more venues (all for the
      same city/date — the caller is responsible for matching).
    forecast_pdf: per-degree probability mass (e.g. Derek's ensemble or
      a normal-from-NWS placeholder). Need not sum to exactly 1.0; the
      bracket EV is simply sum(pdf[d] for d in bracket.degrees).
    city, date: copied through to each ArbOpportunity for traceability.
    fee_rate: per-venue fractional fee on notional (e.g.
      {"kalshi": 0.07, "polymarket": 0.02}). Defaults to zero if omitted.
    min_edge: minimum after-fee edge per contract to flag (in dollars).

  Returns:
    List of ArbOpportunity objects, one per (bracket, direction) signal,
    sorted by net_edge descending.
  """
  fees = fee_rate or {}
  opps: list[ArbOpportunity] = []

  for q in quotes:
    fee = fees.get(q.venue, 0.0)
    fair_value = sum(forecast_pdf.get(d, 0.0) for d in q.degrees)

    # Buy signal: forecast says fair > best_ask (we can buy below fair value)
    if q.best_ask is not None and q.ask_size > 0:
      cost = q.best_ask * (1.0 + fee)
      net_edge = fair_value - cost
      if net_edge >= min_edge:
        opps.append(ArbOpportunity(
          city=city,
          date=date,
          trade_type="VALUE_BUY",
          venue=q.venue,
          market_id=q.market_id,
          bracket_label=q.label,
          degrees=list(q.degrees),
          side="yes",
          action="buy",
          price=q.best_ask,
          fair_value=round(fair_value, 4),
          net_edge=round(net_edge, 4),
          # Top-of-book depth caps how much we can trade at this price.
          # Anything beyond walks the book — handle that separately if needed.
          max_size=int(q.ask_size),
        ))

    # Sell signal: forecast says fair < best_bid (we can sell above fair value)
    if q.best_bid is not None and q.bid_size > 0:
      proceeds = q.best_bid * (1.0 - fee)
      net_edge = proceeds - fair_value
      if net_edge >= min_edge:
        opps.append(ArbOpportunity(
          city=city,
          date=date,
          trade_type="VALUE_SELL",
          venue=q.venue,
          market_id=q.market_id,
          bracket_label=q.label,
          degrees=list(q.degrees),
          side="yes",
          action="sell",
          price=q.best_bid,
          fair_value=round(fair_value, 4),
          net_edge=round(net_edge, 4),
          max_size=int(q.bid_size),
        ))

  opps.sort(key=lambda o: o.net_edge, reverse=True)
  return opps
