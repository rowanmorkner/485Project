"""
One-shot script: pull tomorrow's high-temp brackets for every city in
config.CITIES from both venues and dump each to a JSON snapshot.

Usage:
  python -m test_data.fetch_snapshot                              # all cities, tomorrow
  python -m test_data.fetch_snapshot --city Miami                 # one city, tomorrow
  python -m test_data.fetch_snapshot --city LA --date 2026-04-28  # one city, specific date
  python -m test_data.fetch_snapshot --date 2026-04-28            # all cities, specific date

Output files are named test_data/<city_slug>_<date>.json and contain
the raw bracket lists ready to feed into strategy/parsers.py.
"""

import argparse
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

# Make the project root importable when run as `python -m test_data.fetch_snapshot`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from config import CITIES
from clients.kalshi import KalshiClient
from clients.polymarket import PolymarketClient
from main import fetch_kalshi_events, fetch_polymarket_events


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def snapshot(
  city_name: str,
  target_date: str,
  kalshi_client: KalshiClient,
  poly_client: PolymarketClient,
) -> dict:
  """Fetch one date's worth of bracket data for one city from both venues."""
  city_config = CITIES[city_name]
  series = city_config["kalshi_series"]
  poly_name = city_config["polymarket_name"]

  logger.info("[%s] Fetching Kalshi events (series=%s)...", city_name, series)
  kalshi_events = fetch_kalshi_events(kalshi_client, series)
  logger.info("[%s]   %d total Kalshi events fetched.", city_name, len(kalshi_events))

  logger.info("[%s] Fetching Polymarket events for '%s'...", city_name, poly_name)
  poly_events = fetch_polymarket_events(poly_client, poly_name)
  logger.info("[%s]   %d total Polymarket events fetched.", city_name, len(poly_events))

  # Filter both venues to the requested date
  k_match = next((e for e in kalshi_events if e.get("date") == target_date), None)
  p_match = next((e for e in poly_events if e.get("date") == target_date), None)

  if not k_match:
    logger.warning("[%s] No Kalshi event on %s", city_name, target_date)
  if not p_match:
    logger.warning("[%s] No Polymarket event on %s", city_name, target_date)

  return {
    "city": city_name,
    "date": target_date,
    "city_config": city_config,
    "kalshi": k_match,
    "polymarket": p_match,
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--city",
    default=None,
    help="City name from config.CITIES (omit to fetch all cities)",
  )
  parser.add_argument(
    "--date",
    default=(date.today() + timedelta(days=1)).isoformat(),
    help="Target date in YYYY-MM-DD (default: tomorrow)",
  )
  args = parser.parse_args()

  # Validate / pick the target city list
  if args.city is not None:
    if args.city not in CITIES:
      raise SystemExit(f"Unknown city: {args.city}. Choices: {list(CITIES)}")
    targets = [args.city]
  else:
    targets = list(CITIES.keys())

  # Build clients once and reuse across cities
  kalshi_client = KalshiClient()
  poly_client = PolymarketClient()

  out_dir = Path(__file__).resolve().parent
  results: list[tuple[str, Path | None, str]] = []  # (city, out_path, status)

  for city_name in targets:
    try:
      data = snapshot(city_name, args.date, kalshi_client, poly_client)
    except Exception as exc:
      logger.error("[%s] Fetch failed: %s", city_name, exc)
      results.append((city_name, None, f"ERROR: {exc}"))
      continue

    city_slug = city_name.lower().replace(" ", "_")
    out_path = out_dir / f"{city_slug}_{args.date}.json"
    # default=list converts tuples (e.g. city_config["nws"]) — JSON has no tuple type
    out_path.write_text(json.dumps(data, indent=2, default=list))

    has_k = data["kalshi"] is not None
    has_p = data["polymarket"] is not None
    status = f"K={'✓' if has_k else '✗'} P={'✓' if has_p else '✗'}"
    results.append((city_name, out_path, status))
    logger.info("[%s] Wrote %s (%d bytes) %s",
                city_name, out_path.name, out_path.stat().st_size, status)

  # Summary
  print("\n=== SUMMARY ===")
  for city, path, status in results:
    print(f"  {city:18s} {status}  {path.name if path else ''}")


if __name__ == "__main__":
  main()
