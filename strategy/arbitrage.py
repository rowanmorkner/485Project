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

from contracts import ArbOpportunity, BracketQuote, PairedArbOpportunity
from strategy.risk import bracket_fair_value, venue_pdf

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
    # Venue-adjusted fair value: convolves the forecast PDF with the
    # empirical reading-error histogram for the venue (see strategy/risk.py).
    # Falls back to the raw forecast sum if no per-city Δ data is loaded.
    fair_value = bracket_fair_value(q.degrees, forecast_pdf, city, q.venue)

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


# ── Paired (cross-venue, opposite-side) arbitrage ────────────────────────

def find_paired_arbitrage(
  poly_quotes: Iterable[BracketQuote],
  kalshi_quotes: Iterable[BracketQuote],
  forecast_pdf: dict[int, float],
  city: str,
  date: str,
  fee_rate: dict[str, float] | None = None,
  min_edge: float = 0.03,
  gap_zone_risk: float = 0.2,
) -> list[PairedArbOpportunity]:
  """
  Scan for cross-venue YES/NO opposite-side trades on overlapping brackets.

  For each (poly bracket, kalshi bracket) pair whose degree ranges overlap,
  evaluate two trade structures:

    A) BUY YES on Poly  + BUY NO  on Kalshi
       wins if Poly_reading ∈ poly_bracket  OR  Kalshi_reading ∉ kalshi_bracket
       loses only if Poly_reading ∉ poly_bracket  AND  Kalshi_reading ∈ kalshi_bracket
       loss zone (forecast frame) ≈ kalshi_bracket \\ poly_bracket

    B) BUY NO  on Poly  + BUY YES on Kalshi
       symmetric; loss zone ≈ poly_bracket \\ kalshi_bracket

  Pricing — both legs are BUYS, so we pay best ask on each:
    poly  YES: best_ask_poly                      ; NO: 1 - best_bid_poly
    kalshi YES: best_ask_kalshi (= 1 - bid_no)    ; NO: 1 - best_bid_kalshi

  Expected value uses the convolution-adjusted per-venue PDFs (from
  strategy/risk.venue_pdf) so resolution divergence between venues is
  baked into the EV. We then SUBTRACT a gap-zone haircut equal to
  `gap_zone_risk * P(temp ∈ loss_zone)` as an additional safety reserve
  for the basis-risk we may have under-modeled.

  Sizing — each leg's max executable size comes from the relevant
  top-of-book depth; the paired position is capped at min(legA, legB)
  so the legs hedge each other.

  Args:
    gap_zone_risk: extra haircut multiplier (default 0.2). Higher = more
      conservative. Set to 0 to disable (just use convolution-only EV).

  Returns sorted by net_edge descending.
  """
  fees = fee_rate or {}
  poly_list = list(poly_quotes)
  kalshi_list = list(kalshi_quotes)

  # Per-venue convolved reading distributions — one shared computation
  poly_reading_pdf = venue_pdf(forecast_pdf, city, "polymarket")
  kalshi_reading_pdf = venue_pdf(forecast_pdf, city, "kalshi")

  opps: list[PairedArbOpportunity] = []

  for pq in poly_list:
    for kq in kalshi_list:
      # Only consider pairs with overlapping degree ranges. Pure non-overlap
      # is just two independent value trades — handled by find_value_trades.
      poly_set = set(pq.degrees)
      kalshi_set = set(kq.degrees)
      if not (poly_set & kalshi_set):
        continue

      # Direction A: YES Poly + NO Kalshi
      if pq.best_ask is not None and kq.best_bid is not None \
         and pq.ask_size > 0 and kq.bid_size > 0:
        opp_a = _evaluate_paired(
          city=city, date=date, direction="poly_yes_kalshi_no",
          poly_quote=pq, poly_side="yes",
          poly_pay=pq.best_ask, poly_avail=int(pq.ask_size),
          poly_win_set=poly_set,
          kalshi_quote=kq, kalshi_side="no",
          # buying NO on Kalshi means crossing the YES bid: pay (1 - YES_bid)
          kalshi_pay=1.0 - kq.best_bid, kalshi_avail=int(kq.bid_size),
          kalshi_win_set=set(range(kq.degrees[0] - 50, kq.degrees[-1] + 51)) - kalshi_set,
          # NB win_set above is "all plausible degrees NOT in kalshi bracket"
          forecast_pdf=forecast_pdf,
          poly_reading_pdf=poly_reading_pdf,
          kalshi_reading_pdf=kalshi_reading_pdf,
          fees=fees, gap_zone_risk=gap_zone_risk,
        )
        if opp_a and opp_a.net_edge >= min_edge:
          opps.append(opp_a)

      # Direction B: NO Poly + YES Kalshi
      if pq.best_bid is not None and kq.best_ask is not None \
         and pq.bid_size > 0 and kq.ask_size > 0:
        opp_b = _evaluate_paired(
          city=city, date=date, direction="poly_no_kalshi_yes",
          poly_quote=pq, poly_side="no",
          poly_pay=1.0 - pq.best_bid, poly_avail=int(pq.bid_size),
          poly_win_set=set(range(pq.degrees[0] - 50, pq.degrees[-1] + 51)) - poly_set,
          kalshi_quote=kq, kalshi_side="yes",
          kalshi_pay=kq.best_ask, kalshi_avail=int(kq.ask_size),
          kalshi_win_set=kalshi_set,
          forecast_pdf=forecast_pdf,
          poly_reading_pdf=poly_reading_pdf,
          kalshi_reading_pdf=kalshi_reading_pdf,
          fees=fees, gap_zone_risk=gap_zone_risk,
        )
        if opp_b and opp_b.net_edge >= min_edge:
          opps.append(opp_b)

  opps.sort(key=lambda o: o.net_edge, reverse=True)
  return opps


def _evaluate_paired(
  *, city, date, direction,
  poly_quote: BracketQuote, poly_side: str, poly_pay: float, poly_avail: int,
  poly_win_set: set[int],
  kalshi_quote: BracketQuote, kalshi_side: str, kalshi_pay: float, kalshi_avail: int,
  kalshi_win_set: set[int],
  forecast_pdf: dict[int, float],
  poly_reading_pdf: dict[int, float],
  kalshi_reading_pdf: dict[int, float],
  fees: dict[str, float],
  gap_zone_risk: float,
) -> PairedArbOpportunity | None:
  """
  Score a single direction (one specific YES/NO + YES/NO orientation).

  Expected payout treats the two venues' readings as INDEPENDENTLY drawn
  from their convolution-adjusted PDFs. That's a simplification — true
  readings are highly correlated — but it's the right precision tier for
  current data, and the gap-zone haircut adds a safety margin on top.
  """
  cost_per_pair = poly_pay * (1.0 + fees.get("polymarket", 0.0)) \
                + kalshi_pay * (1.0 + fees.get("kalshi", 0.0))

  # Independent-readings expected payout
  p_poly_wins   = sum(poly_reading_pdf.get(d, 0.0)   for d in poly_win_set)
  p_kalshi_wins = sum(kalshi_reading_pdf.get(d, 0.0) for d in kalshi_win_set)
  expected_payout = p_poly_wins + p_kalshi_wins

  expected_value = expected_payout - cost_per_pair

  # Loss zone in the forecast frame (assume venues read = true temp).
  # This is the "structural" loss zone before convolution-induced widening.
  # Convolution-induced widening is already baked into expected_payout
  # (each venue's PDF spreads probability into adjacent degrees per Δ).
  poly_set = set(poly_quote.degrees)
  kalshi_set = set(kalshi_quote.degrees)
  if direction == "poly_yes_kalshi_no":
    # Loss when temp ∈ kalshi AND temp ∉ poly
    loss_zone = sorted(kalshi_set - poly_set)
  else:
    # Loss when temp ∈ poly AND temp ∉ kalshi
    loss_zone = sorted(poly_set - kalshi_set)

  loss_zone_prob = sum(forecast_pdf.get(d, 0.0) for d in loss_zone)
  haircut = gap_zone_risk * loss_zone_prob
  net_edge = expected_value - haircut

  return PairedArbOpportunity(
    city=city, date=date, direction=direction,
    poly_market_id=poly_quote.market_id,
    poly_bracket_label=poly_quote.label,
    poly_degrees=list(poly_quote.degrees),
    poly_side=poly_side,
    poly_price=round(poly_pay, 4),
    poly_size_avail=poly_avail,
    kalshi_market_id=kalshi_quote.market_id,
    kalshi_bracket_label=kalshi_quote.label,
    kalshi_degrees=list(kalshi_quote.degrees),
    kalshi_side=kalshi_side,
    kalshi_price=round(kalshi_pay, 4),
    kalshi_size_avail=kalshi_avail,
    cost_per_pair=round(cost_per_pair, 4),
    expected_payout=round(expected_payout, 4),
    expected_value=round(expected_value, 4),
    loss_zone_degrees=loss_zone,
    loss_zone_prob=round(loss_zone_prob, 4),
    gap_zone_haircut=round(haircut, 4),
    net_edge=round(net_edge, 4),
    max_size=min(poly_avail, kalshi_avail),
  )
