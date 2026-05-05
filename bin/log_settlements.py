"""
Daily cron job: pull every recently-settled (city, date) market from both
Kalshi and Polymarket, extract the winning bracket on each side, and write
the implied "venue reading" to the settlements table.

This is the data-acquisition half of Component 2. The companion job
bin/refresh_risk_model.py reads from the settlements table and rebuilds
the per-city Δ histogram used by strategy/risk.py.

Reading extraction rules:
  - Narrow brackets (e.g. Kalshi "85° to 86°", Polymarket "84-85°F"):
    record the LOW degree as the venue's "reading". Midpoints would be
    cleaner conceptually but produce systematic ±1°F bias from the
    Kalshi-vs-Polymarket bracket-edge offset (Kalshi: odd-aligned;
    Polymarket: even-aligned).
  - Tail brackets ("X or above" / "X or below"): skip — we can't infer
    a single reading from an open-ended interval.

Run:
  python -m bin.log_settlements [--days N]
"""

import argparse
import logging
import sys
from pathlib import Path

# Project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from clients.kalshi import KalshiClient
from clients.polymarket import PolymarketClient
from config import CITIES
from persistence import db
# Reuse the proven settlement-fetching helpers from the backtest module
from backtests.run_settlement_backtest import (
  fetch_all_settled_kalshi_events,
  fetch_all_closed_weather_events,
  kalshi_event_date,
  kalshi_winning_market,
  kalshi_market_degrees,
  poly_event_date,
  poly_winner,
  ABOVE_RE,
  BELOW_RE,
)
from strategy.parsers import _parse_polymarket_range


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def reading_or_none(degrees: list[int], label: str) -> int | None:
  """
  Return a deterministic integer "reading" for a closed bracket, or None
  for open-ended brackets ("X or above" / "X or below") which can't pin
  the venue's reading to a single value.

  We use the LOW edge (min(degrees)) instead of the midpoint. Midpoints
  with banker's rounding produce systematic ±1°F bias because Kalshi
  brackets (X° to X+1°) and Polymarket brackets (X-X+1°F) have offset
  alignments. The low edge is invariant to alignment and gives a stable
  Δ statistic with bounded artifact (≤ 1°F bracket-alignment noise).
  """
  if not degrees:
    return None
  if ABOVE_RE.search(label) or BELOW_RE.search(label):
    return None
  return min(degrees)


def log_kalshi_for_city(k: KalshiClient, series: str, city: str, days: int) -> int:
  """Fetch Kalshi settled events for one city; write to settlements table."""
  events = fetch_all_settled_kalshi_events(k, series)
  log.info("[%s] %d settled Kalshi events fetched", city, len(events))
  written = 0
  for ev in events:
    d = kalshi_event_date(ev.get("event_ticker", ""))
    if not d:
      continue
    try:
      w = kalshi_winning_market(k, ev["event_ticker"])
    except Exception as exc:
      log.warning("[%s] kalshi market fetch failed for %s: %s",
                  city, ev["event_ticker"], exc)
      continue
    if not w:
      continue
    label, degrees = kalshi_market_degrees(w)
    high = reading_or_none(degrees, label)
    if high is None:
      continue
    db.write_settlement(
      city=city,
      date=d,
      kalshi_high_f=high,
      kalshi_event_ticker=ev["event_ticker"],
    )
    written += 1
  log.info("[%s] wrote %d kalshi readings to settlements", city, written)
  return written


def log_polymarket_for_city(poly_events_all: list[dict], poly_name: str,
                            city: str) -> int:
  """Filter the closed-weather event list for one city; write settlements."""
  city_lower = poly_name.lower()
  written = 0
  for e in poly_events_all:
    title = (e.get("title") or "").lower()
    if "highest temperature" not in title or city_lower not in title:
      continue
    d = poly_event_date(e)
    if not d:
      continue
    w = poly_winner(e)
    if not w:
      continue
    label = w.get("groupItemTitle") or w.get("question") or ""
    degrees = _parse_polymarket_range(label)
    high = reading_or_none(degrees, label)
    if high is None:
      continue
    db.write_settlement(
      city=city,
      date=d,
      polymarket_high_f=high,
      polymarket_event_slug=e.get("slug"),
    )
    written += 1
  log.info("[%s] wrote %d polymarket readings to settlements", city, written)
  return written


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  # `days` is currently advisory — both venue clients fetch all settled
  # events and we filter on insertion (UPSERT idempotent on city+date).
  parser.add_argument("--days", type=int, default=365,
                      help="Soft cap on lookback window (currently advisory).")
  args = parser.parse_args()
  _ = args.days  # placeholder for future filtering

  db.init()  # idempotent

  k = KalshiClient()
  p = PolymarketClient()

  # Polymarket: pull the entire closed-weather event list once and filter
  # per-city below. This single fetch is what backtests already do.
  log.info("Fetching all closed Polymarket weather events...")
  poly_all = fetch_all_closed_weather_events(p)
  log.info("  %d total closed weather events", len(poly_all))

  total_k, total_p = 0, 0
  for city_name, cfg in CITIES.items():
    log.info("=== %s ===", city_name)
    total_k += log_kalshi_for_city(k, cfg["kalshi_series"], city_name, args.days)
    total_p += log_polymarket_for_city(poly_all, cfg["polymarket_name"], city_name)

  log.info("Done. Kalshi rows touched: %d, Polymarket rows touched: %d",
           total_k, total_p)


if __name__ == "__main__":
  main()
