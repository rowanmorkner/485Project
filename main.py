"""
Weather Arbitrage Bot — main entry point.

Per city: fetch venue brackets + NWS forecast, persist snapshots and
forecast PDF, then run the cross-venue hedged-pair selector. Accepted
pairs are turned into OrderRequest pairs (logged to bot.db; if
PAPER_TRADING=1, synthetic fills are recorded immediately).
"""

import json
import logging
from dotenv import load_dotenv

# Load .env before importing clients (they read env vars at construction time)
load_dotenv()

from config import CITIES
from clients.kalshi import KalshiClient
from clients.polymarket import PolymarketClient
from clients.weather import NWSClient
from strategy.parsers import parse_kalshi_quotes, parse_polymarket_quotes
from strategy.forecasting import get_forecast_pdf
from strategy.arbitrage import find_hedged_pairs
from strategy.orders import from_hedged_pair
from strategy.execution import execute_order
from strategy.matching import (
  parse_kalshi_date,
  parse_polymarket_date,
  match_events_by_date,
)
from strategy.display import display_hedged_pairs
from persistence import db as persist


logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# ── Venue data fetching ─────────────────────────────────────────────────────

def fetch_kalshi_events(client: KalshiClient, series_ticker: str) -> list[dict]:
  """
  Fetch open events + per-bracket orderbook data for one Kalshi series.
  Returns list of dicts: {source, event_ticker, title, date, brackets[]}.
  """
  events = client.search_high_temp_events(series_ticker, status="open")

  results = []
  for event in events:
    event_ticker = event.get("event_ticker", "")
    title = event.get("title", "N/A")

    markets_resp = client.get_markets(event_ticker=event_ticker)
    brackets = []
    for m in markets_resp.get("markets", []):
      ticker = m["ticker"]
      subtitle = m.get("subtitle", "") or m.get("yes_sub_title", "")

      # Kalshi puts best price at the END of each ladder. yes_dollars = YES
      # bid ladder, no_dollars = NO bid ladder. Best YES ask = 1 - best NO bid.
      ob = client.get_orderbook(ticker)
      book = ob.get("orderbook_fp", {})
      yes_levels = book.get("yes_dollars", [])
      no_levels = book.get("no_dollars", [])

      best_yes_bid = float(yes_levels[-1][0]) if yes_levels else None
      best_yes_bid_size = float(yes_levels[-1][1]) if yes_levels else 0.0
      best_no_bid = float(no_levels[-1][0]) if no_levels else None
      best_no_bid_size = float(no_levels[-1][1]) if no_levels else 0.0
      best_yes_ask = round(1 - best_no_bid, 4) if best_no_bid is not None else None
      best_yes_ask_size = best_no_bid_size

      yes_bid_ladder = [(float(p), float(s)) for p, s in yes_levels]
      yes_ask_ladder = [(round(1 - float(p), 4), float(s)) for p, s in no_levels]

      brackets.append({
        "ticker": ticker,
        "subtitle": subtitle,
        "best_yes_bid": best_yes_bid,
        "best_yes_ask": best_yes_ask,
        "best_yes_bid_size": best_yes_bid_size,
        "best_yes_ask_size": best_yes_ask_size,
        "yes_bid_ladder": yes_bid_ladder,
        "yes_ask_ladder": yes_ask_ladder,
      })

    brackets.sort(key=lambda x: x["ticker"])
    results.append({
      "source": "kalshi",
      "event_ticker": event_ticker,
      "title": title,
      "date": parse_kalshi_date(event_ticker),
      "brackets": brackets,
    })

  return results


def fetch_polymarket_events(client: PolymarketClient, city_name: str) -> list[dict]:
  """
  Fetch open temperature events for one city from Polymarket.
  Returns list of dicts: {source, title, slug, date, brackets[]}.
  """
  events = client.search_weather_events(city=city_name, only_open=True)

  results = []
  for event in events:
    title = event.get("title", "N/A")
    brackets = []
    for mkt in event.get("markets", []):
      # Polymarket events keep already-resolved brackets in their markets
      # list. The CLOB orderbook endpoint 404s on those, flooding the log
      # with "No orderbook exists for the requested token id" warnings.
      # Skip them — there's no executable book to read.
      if mkt.get("closed") or mkt.get("archived"):
        continue
      outcome_prices = mkt.get("outcomePrices", "")
      yes_price = None
      if outcome_prices:
        try:
          prices_list = (
            json.loads(outcome_prices) if isinstance(outcome_prices, str)
            else outcome_prices
          )
          yes_price = float(prices_list[0]) if prices_list else None
        except (ValueError, IndexError):
          pass

      yes_token_id = None
      clob_token_ids = mkt.get("clobTokenIds")
      if clob_token_ids:
        try:
          tokens = (
            json.loads(clob_token_ids) if isinstance(clob_token_ids, str)
            else clob_token_ids
          )
          yes_token_id = tokens[0] if tokens else None
        except (ValueError, IndexError):
          pass

      best_yes_bid = mkt.get("bestBid")
      best_yes_ask = mkt.get("bestAsk")
      best_yes_bid_size = 0.0
      best_yes_ask_size = 0.0
      yes_bid_ladder: list[tuple[float, float]] = []
      yes_ask_ladder: list[tuple[float, float]] = []

      if yes_token_id:
        try:
          ob = client.get_orderbook(yes_token_id)
          bids = ob.get("bids", [])
          asks = ob.get("asks", [])
          if bids:
            best_yes_bid_size = float(bids[-1].get("size", 0))
          if asks:
            best_yes_ask_size = float(asks[-1].get("size", 0))
          yes_bid_ladder = [(float(b["price"]), float(b["size"])) for b in bids]
          yes_ask_ladder = [(float(a["price"]), float(a["size"])) for a in asks]
        except Exception as exc:
          logger.warning("CLOB orderbook fetch failed for token %s: %s",
                         yes_token_id, exc)

      brackets.append({
        "question": mkt.get("groupItemTitle", mkt.get("question", "?")),
        "condition_id": mkt.get("conditionId"),
        "token_id": yes_token_id,
        "yes_price": yes_price,
        "best_yes_bid": float(best_yes_bid) if best_yes_bid is not None else None,
        "best_yes_ask": float(best_yes_ask) if best_yes_ask is not None else None,
        "best_yes_bid_size": best_yes_bid_size,
        "best_yes_ask_size": best_yes_ask_size,
        "yes_bid_ladder": yes_bid_ladder,
        "yes_ask_ladder": yes_ask_ladder,
      })

    results.append({
      "source": "polymarket",
      "title": title,
      "slug": event.get("slug"),
      "date": parse_polymarket_date(event.get("slug", ""), event.get("endDate", "")),
      "brackets": brackets,
    })

  return results


# ── Per-city analysis ───────────────────────────────────────────────────────

def analyze_city(
  city_name: str,
  city_config: dict,
  kalshi_client: KalshiClient,
  poly_client: PolymarketClient,
  nws_client: NWSClient,
):
  """Run the hedged-pair selector for one city across every matched date."""
  print("\n" + "~" * 78)
  print(f"  CITY: {city_name}")
  print("~" * 78)

  series = city_config["kalshi_series"]
  poly_name = city_config["polymarket_name"]
  station = city_config["station"]
  office, gx, gy = city_config["nws"]

  logger.info("[%s] Fetching Kalshi events (series=%s)...", city_name, series)
  kalshi_data = fetch_kalshi_events(kalshi_client, series)
  logger.info("[%s] Found %d Kalshi events.", city_name, len(kalshi_data))

  logger.info("[%s] Fetching Polymarket events...", city_name)
  poly_data = fetch_polymarket_events(poly_client, poly_name)
  logger.info("[%s] Found %d Polymarket events.", city_name, len(poly_data))

  matched = match_events_by_date(kalshi_data, poly_data)
  if not matched:
    print(f"  No events found for {city_name}.\n")
    return

  # Persist every venue snapshot we just fetched.
  for ev in kalshi_data:
    if ev.get("date") and ev.get("brackets"):
      persist.write_snapshot(city_name, ev["date"], "kalshi", ev["brackets"])
  for ev in poly_data:
    if ev.get("date") and ev.get("brackets"):
      persist.write_snapshot(city_name, ev["date"], "polymarket", ev["brackets"])

  logger.info("[%s] Fetching NWS forecasts (station: %s)...", city_name, station)
  forecasts: dict[str, float] = {}
  for date_str in (date for date, _, _ in matched):
    high = nws_client.get_forecast_high_for_date(date_str, office, gx, gy)
    if high is not None:
      forecasts[date_str] = high

  for date, kalshi_event, poly_event in matched:
    forecast_high = forecasts.get(date)

    # Both venues must be present — the strategy is a cross-venue pair.
    if not kalshi_event:
      print(f"\n  [{city_name} — {date}] no Kalshi event yet — skipping.")
      continue
    if not poly_event:
      print(f"\n  [{city_name} — {date}] no Polymarket event yet — skipping.")
      continue

    print("\n" + "#" * 78)
    print(f"  {city_name} — {date}")
    print(f"  Kalshi: {kalshi_event['event_ticker']}")
    print(f"  Polymarket: {poly_event['title']}")
    if forecast_high is not None:
      print(f"  NWS Forecast High: {forecast_high:.0f}°F")
    forecast_pdf, fc_source = get_forecast_pdf(city_name, date, forecast_high)
    if not forecast_pdf:
      print("  No forecast available (ensemble + NWS both failed) — skipping.")
      print("#" * 78)
      continue
    print(f"  Forecast PDF: {fc_source} ({len(forecast_pdf)} bins)")
    print("#" * 78)

    persist.write_forecast(city_name, date, forecast_pdf, source=fc_source)

    kalshi_quotes = parse_kalshi_quotes(kalshi_event["brackets"])
    poly_quotes = parse_polymarket_quotes(poly_event["brackets"])

    pairs = find_hedged_pairs(
      kalshi_quotes, poly_quotes, forecast_pdf,
      city=city_name, date=date,
    )
    display_hedged_pairs(pairs)

    for pair in pairs:
      try:
        orders = from_hedged_pair(pair)
        for o in orders:
          execute_order(o)
      except Exception as exc:
        logger.warning("Failed to execute hedged pair %s/%s: %s",
                       pair.kalshi_market_id, pair.poly_market_id, exc)


def main():
  """Run the analysis across all configured cities."""
  logger.info("Starting Weather Arbitrage Bot — scanning %d cities...", len(CITIES))

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
