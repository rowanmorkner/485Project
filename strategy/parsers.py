"""
Parsers that turn raw venue bracket data into two views:

  1. PMF view (parse_*_bins) — per-degree dict[int, float] of implied
     probabilities, using midpoints. Used for forecast comparison and
     the per-degree discrepancy display.

  2. Quote view (parse_*_quotes) — list[BracketQuote] preserving each
     bracket's executable bid/ask + depth + market ID. Used by the
     arbitrage layer to generate actual trade signals against forecast EV.

Each venue exposes its bins differently (Kalshi: "77° to 78°" subtitles +
bid/ask; Polymarket: "78-79°F" questions + bid/ask via CLOB), but both
collapse into the same downstream shapes via these parsers.
"""

import re
import logging
from typing import Optional

from contracts import BracketQuote
from strategy.distributions import normalize

logger = logging.getLogger(__name__)

# How many degrees beyond an open-ended boundary ("X or below"/"X or above")
# to spread the tail-bin probability across.
TAIL_SPREAD = 5


# ── Kalshi ────────────────────────────────────────────────────────────────

def parse_kalshi_bins(brackets: list[dict]) -> dict[int, float]:
  """
  Convert Kalshi bracket markets into a per-degree PMF.

  Each bracket dict should contain:
    - subtitle: "77° to 78°" / "76° or below" / "85° or above"
    - best_yes_bid, best_yes_ask: floats (or None)

  The midpoint of bid/ask becomes the bin's implied probability, then is
  spread uniformly across the integer degrees inside the bin.
  """
  distribution: dict[int, float] = {}

  for bracket in brackets:
    subtitle = bracket.get("subtitle", "")
    bid = bracket.get("best_yes_bid")
    ask = bracket.get("best_yes_ask")

    implied_prob = _midpoint_price(bid, ask)
    if implied_prob is None or implied_prob <= 0:
      continue

    degrees = _parse_kalshi_range(subtitle)
    if not degrees:
      logger.warning("Could not parse Kalshi subtitle: '%s'", subtitle)
      continue

    prob_per_degree = implied_prob / len(degrees)
    for deg in degrees:
      distribution[deg] = distribution.get(deg, 0.0) + prob_per_degree

  return normalize(distribution)


def _parse_kalshi_range(subtitle: str) -> list[int]:
  """
  Extract the integer degrees covered by a Kalshi bracket subtitle.
  Kalshi uses exactly three formats:

    "77° to 78°"    -> [77, 78]
    "76° or below"  -> [71..76]   (boundary - TAIL_SPREAD .. boundary)
    "85° or above"  -> [85..90]   (boundary .. boundary + TAIL_SPREAD)
  """
  # "X° to Y°"
  m = re.search(r"(\d+)°\s*to\s*(\d+)°", subtitle)
  if m:
    return list(range(int(m.group(1)), int(m.group(2)) + 1))

  # "X° or below" / "X° or above"
  m = re.search(r"(\d+)°\s*or\s+(below|above)", subtitle, re.IGNORECASE)
  if m:
    boundary = int(m.group(1))
    if m.group(2).lower() == "below":
      return list(range(boundary - TAIL_SPREAD, boundary + 1))
    return list(range(boundary, boundary + TAIL_SPREAD + 1))

  return []


# ── Polymarket ────────────────────────────────────────────────────────────

def parse_polymarket_bins(brackets: list[dict]) -> dict[int, float]:
  """
  Convert Polymarket bracket markets into a per-degree PMF.

  Each bracket dict should contain:
    - question: "78-79°F" / "73°F or less" / "92°F or more"
    - yes_price: float (or None)
  """
  distribution: dict[int, float] = {}

  for bracket in brackets:
    question = bracket.get("question", "")
    yes_price = bracket.get("yes_price")

    if yes_price is None or yes_price <= 0:
      continue

    degrees = _parse_polymarket_range(question)
    if not degrees:
      logger.warning("Could not parse Polymarket question: '%s'", question)
      continue

    prob_per_degree = yes_price / len(degrees)
    for deg in degrees:
      distribution[deg] = distribution.get(deg, 0.0) + prob_per_degree

  return normalize(distribution)


def _parse_polymarket_range(question: str) -> list[int]:
  """
  Extract the integer degrees covered by a Polymarket bracket question.
  Polymarket uses exactly three formats:

    "78-79°F"        -> [78, 79]
    "73°F or below"  -> [68..73]
    "92°F or higher" -> [92..97]
  """
  # "X-Y°F"
  m = re.search(r"(\d+)-(\d+)°F", question)
  if m:
    return list(range(int(m.group(1)), int(m.group(2)) + 1))

  # "X°F or below" / "X°F or higher"
  m = re.search(r"(\d+)°F\s*or\s+(below|higher)", question, re.IGNORECASE)
  if m:
    boundary = int(m.group(1))
    if m.group(2).lower() == "below":
      return list(range(boundary - TAIL_SPREAD, boundary + 1))
    return list(range(boundary, boundary + TAIL_SPREAD + 1))

  return []


# ── Shared helpers ────────────────────────────────────────────────────────

def _midpoint_price(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
  """
  Midpoint of bid and ask. Falls back to whichever side exists if only
  one is present; returns None if both are missing.
  """
  if bid is not None and ask is not None:
    return (bid + ask) / 2.0
  if bid is not None:
    return bid
  if ask is not None:
    return ask
  return None


# ── Quote parsers (executable bid/ask + depth) ────────────────────────────

def parse_kalshi_quotes(brackets: list[dict]) -> list[BracketQuote]:
  """
  Convert raw Kalshi bracket dicts into BracketQuote objects.

  Skips brackets whose subtitle doesn't parse — they show up as a warning
  and are excluded from arb consideration (better than fabricating a quote).
  """
  quotes = []
  for b in brackets:
    subtitle = b.get("subtitle", "")
    degrees = _parse_kalshi_range(subtitle)
    if not degrees:
      logger.warning("Could not parse Kalshi subtitle for quote: '%s'", subtitle)
      continue

    quotes.append(BracketQuote(
      venue="kalshi",
      market_id=b.get("ticker", ""),
      label=subtitle,
      degrees=degrees,
      best_bid=b.get("best_yes_bid"),
      best_ask=b.get("best_yes_ask"),
      bid_size=float(b.get("best_yes_bid_size", 0) or 0),
      ask_size=float(b.get("best_yes_ask_size", 0) or 0),
      ladder_bids=b.get("yes_bid_ladder", []),
      ladder_asks=b.get("yes_ask_ladder", []),
    ))
  return quotes


def parse_polymarket_quotes(brackets: list[dict]) -> list[BracketQuote]:
  """
  Convert raw Polymarket bracket dicts into BracketQuote objects.
  """
  quotes = []
  for b in brackets:
    question = b.get("question", "")
    degrees = _parse_polymarket_range(question)
    if not degrees:
      logger.warning("Could not parse Polymarket question for quote: '%s'", question)
      continue

    quotes.append(BracketQuote(
      venue="polymarket",
      market_id=b.get("token_id", "") or "",
      label=question,
      degrees=degrees,
      best_bid=b.get("best_yes_bid"),
      best_ask=b.get("best_yes_ask"),
      bid_size=float(b.get("best_yes_bid_size", 0) or 0),
      ask_size=float(b.get("best_yes_ask_size", 0) or 0),
      condition_id=b.get("condition_id", "") or "",
      ladder_bids=b.get("yes_bid_ladder", []),
      ladder_asks=b.get("yes_ask_ladder", []),
    ))
  return quotes
