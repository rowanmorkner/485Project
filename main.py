"""
Weather Arbitrage Bot — Main Entry Point.

Thin orchestrator: for each configured city, fetch venue data + NWS forecast,
build per-degree distributions, print comparison, and flag arb opportunities.

Heavy lifting lives in:
  strategy/parsers.py        — venue bracket → PMF
  strategy/distributions.py  — PMF/CDF math
  strategy/arbitrage.py      — discrepancy & arb detection
  strategy/matching.py       — date parsing & cross-venue event pairing
  strategy/display.py        — console tables
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
from strategy.parsers import (
  parse_kalshi_bins,
  parse_polymarket_bins,
  parse_kalshi_quotes,
  parse_polymarket_quotes,
)
from strategy.distributions import forecast_to_distribution
from strategy.arbitrage import find_discrepancies, find_value_trades
from strategy.matching import (
  parse_kalshi_date,
  parse_polymarket_date,
  match_events_by_date,
)
from strategy.display import display_comparison


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
      # Kalshi exposes the human range under either 'subtitle' or 'yes_sub_title'
      subtitle = m.get("subtitle", "") or m.get("yes_sub_title", "")

      # Pull pricing from the orderbook. Kalshi puts best price at the END
      # of each ladder. yes_dollars = YES bid ladder, no_dollars = NO bid
      # ladder. Best YES ask = 1 - best NO bid (binary-market identity).
      ob = client.get_orderbook(ticker)
      book = ob.get("orderbook_fp", {})
      yes_levels = book.get("yes_dollars", [])
      no_levels = book.get("no_dollars", [])

      best_yes_bid = float(yes_levels[-1][0]) if yes_levels else None
      best_yes_bid_size = float(yes_levels[-1][1]) if yes_levels else 0.0
      best_no_bid = float(no_levels[-1][0]) if no_levels else None
      best_no_bid_size = float(no_levels[-1][1]) if no_levels else 0.0
      best_yes_ask = round(1 - best_no_bid, 4) if best_no_bid is not None else None
      # Size at best YES ask = size at best NO bid (same liquidity, opposite sign)
      best_yes_ask_size = best_no_bid_size

      # Convert ladders to (price, size) tuples ordered worst→best, matching
      # BracketQuote.ladder_* convention. YES ask ladder is the NO bid ladder
      # with prices flipped (1 - p), so we reverse direction to keep worst→best.
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
      # Polymarket returns outcomePrices as a JSON-encoded string of [yes, no]
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

      # The YES token ID is the first entry of clobTokenIds (also JSON-encoded).
      # We need it to call the CLOB orderbook endpoint for executable depth.
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

      # Pull executable bid/ask + depth from the CLOB orderbook. Polymarket
      # returns ladders worst→best (best bid = bids[-1], best ask = asks[-1]),
      # matching Kalshi's convention.
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
        # groupItemTitle is the short bin label ("78-79°F"); fall back to question
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
  """Run the full arbitrage analysis for one city."""
  print("\n" + "~" * 78)
  print(f"  CITY: {city_name}")
  print("~" * 78)

  series = city_config["kalshi_series"]
  poly_name = city_config["polymarket_name"]
  station = city_config["station"]
  office, gx, gy = city_config["nws"]

  # Fetch venue data
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

  # Cities now share one resolution station per row, so one forecast per date
  logger.info("[%s] Fetching NWS forecasts (station: %s)...", city_name, station)
  forecasts: dict[str, float] = {}
  for date_str in (date for date, _, _ in matched):
    high = nws_client.get_forecast_high_for_date(date_str, office, gx, gy)
    if high is not None:
      forecasts[date_str] = high

  for date, kalshi_event, poly_event in matched:
    forecast_high = forecasts.get(date)

    print("\n" + "#" * 78)
    print(f"  {city_name} — {date}")
    if kalshi_event:
      print(f"  Kalshi: {kalshi_event['event_ticker']}")
    if poly_event:
      print(f"  Polymarket: {poly_event['title']}")
    print(f"  Resolution station: {station}")
    if forecast_high is not None:
      print(f"  NWS Forecast High: {forecast_high:.0f}°F")
    else:
      print("  NWS Forecast: not available")
    print("#" * 78)

    # Build per-degree distributions for whichever sources are available
    kalshi_dist = parse_kalshi_bins(kalshi_event["brackets"]) if kalshi_event else {}
    poly_dist = parse_polymarket_bins(poly_event["brackets"]) if poly_event else {}
    forecast_dist = (
      forecast_to_distribution(forecast_high) if forecast_high is not None else {}
    )

    source_count = sum([bool(kalshi_dist), bool(poly_dist), bool(forecast_dist)])
    if source_count < 2:
      print("  Need at least 2 data sources for comparison. Skipping.\n")
      continue

    display_comparison(kalshi_dist, poly_dist, forecast_dist)

    # Discrepancies (informational)
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

    # Per-bracket value trades against the forecast PDF (executable prices,
    # depth-capped sizing). Both venues are scored independently — if both
    # are mispriced in opposite directions you naturally get one trade per side.
    if forecast_dist:
      kalshi_quotes = (
        parse_kalshi_quotes(kalshi_event["brackets"]) if kalshi_event else []
      )
      poly_quotes = (
        parse_polymarket_quotes(poly_event["brackets"]) if poly_event else []
      )
      opportunities = find_value_trades(
        list(kalshi_quotes) + list(poly_quotes),
        forecast_pdf=forecast_dist,
        city=city_name,
        date=date,
      )
    else:
      opportunities = []

    print("\n" + "=" * 78)
    print("  VALUE TRADES vs forecast (min_edge = 0.03, fees = 0%)")
    print("=" * 78)
    if opportunities:
      for opp in opportunities:
        print(
          f"\n  [{opp.trade_type}] {opp.venue:10s}  {opp.bracket_label}"
        )
        print(
          f"    {opp.action.upper():4s} @ {opp.price:.4f}  "
          f"(fair={opp.fair_value:.4f}, net edge={opp.net_edge:+.4f}, "
          f"max size={opp.max_size})"
        )
        print(f"    market_id={opp.market_id}")
    elif not forecast_dist:
      print("  No forecast available — cannot score value trades.")
    else:
      print("  No value trades found above threshold.")

    print()


def main():
  """Run the arbitrage analysis across all configured cities."""
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
