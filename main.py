"""
Weather Arbitrage Bot — Main Entry Point

Scans multiple cities for daily high temperature markets on Kalshi and
Polymarket, compares them against NWS forecasts, and identifies pricing
discrepancies using reconstructed per-degree probability distributions.
"""

import json
import logging
import re
from dotenv import load_dotenv

# Load env vars before importing clients
load_dotenv()

from cities import CITIES
from clients.kalshi import KalshiClient
from clients.polymarket import PolymarketClient
from clients.weather import NWSClient
from strategy.arbitrage import (
  parse_kalshi_bins,
  parse_polymarket_bins,
  forecast_to_distribution,
  find_discrepancies,
  find_arbitrage_opportunities,
  display_comparison,
)

# Configure logging
logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# ── Data fetching ───────────────────────────────────────────────────────────

def fetch_kalshi_events(client: KalshiClient, series_ticker: str) -> list[dict]:
  """
  Fetch open high temp events and orderbook data for a given Kalshi series.
  Returns a list of event dicts with bracket pricing from the orderbook.
  """
  events = client.search_high_temp_events(series_ticker, status="open")

  results = []
  for event in events:
    event_ticker = event.get("event_ticker", "")
    title = event.get("title", "N/A")

    # Get all bracket markets for this event
    markets_resp = client.get_markets(event_ticker=event_ticker)
    brackets = []
    for m in markets_resp.get("markets", []):
      ticker = m["ticker"]
      # Some Kalshi markets use "subtitle", others use "yes_sub_title"
      subtitle = m.get("subtitle", "") or m.get("yes_sub_title", "")

      # Fetch orderbook for real pricing
      ob = client.get_orderbook(ticker)
      book = ob.get("orderbook_fp", {})
      yes_levels = book.get("yes_dollars", [])
      no_levels = book.get("no_dollars", [])

      best_yes_bid = float(yes_levels[-1][0]) if yes_levels else None
      best_no_bid = float(no_levels[-1][0]) if no_levels else None
      best_yes_ask = round(1 - best_no_bid, 4) if best_no_bid is not None else None

      brackets.append({
        "ticker": ticker,
        "subtitle": subtitle,
        "best_yes_bid": best_yes_bid,
        "best_yes_ask": best_yes_ask,
        "yes_depth": len(yes_levels),
        "no_depth": len(no_levels),
      })

    brackets.sort(key=lambda x: x["ticker"])
    date_str = _parse_kalshi_date(event_ticker)

    results.append({
      "source": "kalshi",
      "event_ticker": event_ticker,
      "title": title,
      "date": date_str,
      "brackets": brackets,
    })

  return results


def fetch_polymarket_events(client: PolymarketClient, city_name: str) -> list[dict]:
  """
  Fetch open temperature events from Polymarket for a given city.
  Returns a list of event dicts with bracket pricing.
  """
  events = client.search_weather_events(city=city_name, only_open=True)

  results = []
  for event in events:
    title = event.get("title", "N/A")
    markets = event.get("markets", [])
    brackets = []
    for mkt in markets:
      outcome_prices = mkt.get("outcomePrices", "")
      yes_price = None
      if outcome_prices:
        try:
          prices_list = json.loads(outcome_prices) if isinstance(outcome_prices, str) else outcome_prices
          yes_price = float(prices_list[0]) if prices_list else None
        except (ValueError, IndexError):
          pass

      brackets.append({
        "question": mkt.get("groupItemTitle", mkt.get("question", "?")),
        "condition_id": mkt.get("conditionId"),
        "yes_price": yes_price,
      })

    date_str = _parse_polymarket_date(event.get("slug", ""), event.get("endDate", ""))

    results.append({
      "source": "polymarket",
      "title": title,
      "slug": event.get("slug"),
      "date": date_str,
      "brackets": brackets,
    })

  return results


# ── Date parsing helpers ────────────────────────────────────────────────────

MONTH_MAP = {
  "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
  "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
  "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
}

def _parse_kalshi_date(event_ticker: str) -> str:
  """Parse date from Kalshi ticker like KXHIGHMIA-26APR01 -> 2026-04-01."""
  match = re.search(r"-(\d{2})([A-Z]{3})(\d{2})$", event_ticker)
  if match:
    year = f"20{match.group(1)}"
    month = MONTH_MAP.get(match.group(2), "01")
    day = match.group(3)
    return f"{year}-{month}-{day}"
  return ""


def _parse_polymarket_date(slug: str, end_date: str) -> str:
  """Extract date from Polymarket event endDate (ISO format)."""
  if end_date and "T" in end_date:
    return end_date.split("T")[0]
  return ""


# ── Event matching ──────────────────────────────────────────────────────────

def _match_events(kalshi_data: list[dict], poly_data: list[dict]) -> list[tuple]:
  """
  Match Kalshi and Polymarket events by date.
  Returns list of (date, kalshi_event_or_None, poly_event_or_None).
  """
  by_date: dict[str, dict] = {}

  for event in kalshi_data:
    d = event.get("date", "")
    if d:
      by_date.setdefault(d, {"kalshi": None, "poly": None})
      by_date[d]["kalshi"] = event

  for event in poly_data:
    d = event.get("date", "")
    if d:
      by_date.setdefault(d, {"kalshi": None, "poly": None})
      by_date[d]["poly"] = event

  return [
    (date, pair["kalshi"], pair["poly"])
    for date, pair in sorted(by_date.items())
  ]


# ── Main ────────────────────────────────────────────────────────────────────

def analyze_city(
  city_name: str,
  city_config: dict,
  kalshi_client: KalshiClient,
  poly_client: PolymarketClient,
  nws_client: NWSClient,
):
  """Run the full arbitrage analysis for a single city."""
  print("\n" + "~" * 78)
  print(f"  CITY: {city_name}")
  print("~" * 78)

  series = city_config["kalshi_series"]
  poly_name = city_config["polymarket_name"]

  # Fetch market data
  logger.info("[%s] Fetching Kalshi events (series=%s)...", city_name, series)
  kalshi_data = fetch_kalshi_events(kalshi_client, series)
  logger.info("[%s] Found %d Kalshi events.", city_name, len(kalshi_data))

  logger.info("[%s] Fetching Polymarket events...", city_name)
  poly_data = fetch_polymarket_events(poly_client, poly_name)
  logger.info("[%s] Found %d Polymarket events.", city_name, len(poly_data))

  # Match events by date
  matched = _match_events(kalshi_data, poly_data)
  if not matched:
    print(f"  No events found for {city_name}.\n")
    return

  # NWS gridpoints differ per platform — fetch forecasts for each
  k_office, k_gx, k_gy = city_config["kalshi_nws"]
  p_office, p_gx, p_gy = city_config["poly_nws"]
  k_station = city_config["kalshi_station"]
  p_station = city_config["poly_station"]

  dates = [date for date, _, _ in matched]
  kalshi_forecasts: dict[str, float] = {}
  poly_forecasts: dict[str, float] = {}

  logger.info("[%s] Fetching NWS forecasts (Kalshi station: %s)...", city_name, k_station)
  for date_str in dates:
    high = nws_client.get_forecast_high_for_date(date_str, k_office, k_gx, k_gy)
    if high is not None:
      kalshi_forecasts[date_str] = high

  # Only fetch separately if Polymarket uses a different station
  if (k_office, k_gx, k_gy) != (p_office, p_gx, p_gy):
    logger.info("[%s] Fetching NWS forecasts (Polymarket station: %s)...", city_name, p_station)
    for date_str in dates:
      high = nws_client.get_forecast_high_for_date(date_str, p_office, p_gx, p_gy)
      if high is not None:
        poly_forecasts[date_str] = high
  else:
    poly_forecasts = kalshi_forecasts

  # Analyze each date
  for date, kalshi_event, poly_event in matched:
    k_forecast = kalshi_forecasts.get(date)
    p_forecast = poly_forecasts.get(date)

    print("\n" + "#" * 78)
    print(f"  {city_name} — {date}")
    if kalshi_event:
      print(f"  Kalshi: {kalshi_event['event_ticker']} (resolves at {k_station})")
    if poly_event:
      print(f"  Polymarket: {poly_event['title']} (resolves at {p_station})")
    if k_forecast and p_forecast and k_forecast != p_forecast:
      print(f"  NWS Forecast: {k_forecast:.0f}°F (Kalshi station) / {p_forecast:.0f}°F (Poly station)")
    elif k_forecast:
      print(f"  NWS Forecast High: {k_forecast:.0f}°F")
    else:
      print("  NWS Forecast: not available")
    print("#" * 78)

    # Use the Kalshi station forecast for the shared distribution comparison
    # (it's the more widely-used NWS climatological station)
    forecast_high = k_forecast

    # Build distributions
    kalshi_dist = parse_kalshi_bins(kalshi_event["brackets"]) if kalshi_event else {}
    poly_dist = parse_polymarket_bins(poly_event["brackets"]) if poly_event else {}
    forecast_dist = forecast_to_distribution(forecast_high) if forecast_high else {}

    source_count = sum([bool(kalshi_dist), bool(poly_dist), bool(forecast_dist)])
    if source_count < 2:
      print("  Need at least 2 data sources for comparison. Skipping.\n")
      continue

    # Display comparison table
    display_comparison(kalshi_dist, poly_dist, forecast_dist)

    # Find discrepancies
    discrepancies = find_discrepancies(kalshi_dist, poly_dist, forecast_dist)
    print("=" * 78)
    print("  DISCREPANCIES (min_edge = 0.05)")
    print("=" * 78)
    if discrepancies:
      for d in discrepancies:
        sources = ", ".join(d.get("sources_disagreeing", []))
        print(
          f"  {d['degree']:>4}°F  |  "
          f"K={d['kalshi_prob']:.4f}  P={d['poly_prob']:.4f}  F={d['forecast_prob']:.4f}  "
          f"|  spread={d['max_spread']:.4f}  ({sources})"
        )
    else:
      print("  No significant discrepancies found.")

    # Determine if platforms use the same weather station for this city
    same_station = city_config["kalshi_nws"] == city_config["poly_nws"]

    # Find arbitrage opportunities
    opportunities = find_arbitrage_opportunities(
      kalshi_dist, poly_dist, forecast_dist,
      same_station=same_station,
    )
    print("\n" + "=" * 78)
    print("  ARBITRAGE OPPORTUNITIES (min_edge = 0.03, fees = 0%)")
    print("=" * 78)
    if opportunities:
      for opp in opportunities:
        trade_type = opp.get("trade_type", "?")
        print(f"\n  [{trade_type}] {opp['action']}")
        print(
          f"    Buy @ {opp['buy_price']:.4f} on {opp['buy_platform']}, "
          f"sell @ {opp['sell_price']:.4f} on {opp['sell_platform']}"
        )
        print(f"    Net edge: {opp['net_edge']:.4f} — {opp.get('risk_note', '')}")
    else:
      print("  No arbitrage opportunities found above threshold.")

    print()


def main():
  """Run the arbitrage analysis across all configured cities."""
  logger.info("Starting Weather Arbitrage Bot — scanning %d cities...", len(CITIES))

  # Initialize clients once (shared across all cities)
  kalshi_client = KalshiClient()
  poly_client = PolymarketClient()
  nws_client = NWSClient()

  for city_name, city_config in CITIES.items():
    try:
      analyze_city(city_name, city_config, kalshi_client, poly_client, nws_client)
    except Exception as exc:
      logger.error("[%s] Error: %s", city_name, exc)
      continue

  logger.info("Done — all cities scanned.")


if __name__ == "__main__":
  main()
