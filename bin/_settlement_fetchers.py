"""
Helpers for fetching settled events from Kalshi and Polymarket and
extracting the winning bracket on each side.

Used by bin/log_settlements.py. Cached responses live under bin/_cache/.
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path

from clients.kalshi import KalshiClient
from clients.polymarket import PolymarketClient
from strategy.parsers import _parse_kalshi_range


CACHE_DIR = Path(__file__).resolve().parent / "_cache"
CACHE_DIR.mkdir(exist_ok=True)
POLY_CACHE = CACHE_DIR / "poly_closed_weather_events.json"


# Open-ended bracket detection — "X or above/higher/more" / "X or below/lower/less".
ABOVE_RE = re.compile(r"(\d+)\s*°?\s*F?\s*or\s+(?:above|higher|more)", re.IGNORECASE)
BELOW_RE = re.compile(r"(\d+)\s*°?\s*F?\s*or\s+(?:below|lower|less)", re.IGNORECASE)


# ── Kalshi ────────────────────────────────────────────────────────────────

def fetch_all_settled_kalshi_events(k: KalshiClient, series: str) -> list[dict]:
  """Paginate every settled event for one Kalshi series."""
  out: list[dict] = []
  cursor = None
  while True:
    params = {"series_ticker": series, "status": "settled", "limit": 200}
    if cursor:
      params["cursor"] = cursor
    resp = k._request("GET", "/events", params=params)
    events = resp.get("events", []) or []
    out.extend(events)
    cursor = resp.get("cursor")
    if not cursor or not events:
      break
  return out


def kalshi_event_date(event_ticker: str) -> str | None:
  """Parse YYMMMDD suffix from a Kalshi event ticker (e.g. KXHIGHMIA-26APR27)."""
  m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})$", event_ticker)
  if not m:
    return None
  try:
    return datetime.strptime(
      f"{m.group(3)}{m.group(2)}{m.group(1)}", "%d%b%y"
    ).strftime("%Y-%m-%d")
  except ValueError:
    return None


def kalshi_winning_market(k: KalshiClient, event_ticker: str) -> dict | None:
  """Cached: fetch markets for an event and return the YES-resolved one."""
  cache_path = CACHE_DIR / f"k_event_{event_ticker}.json"
  if cache_path.exists():
    markets = json.loads(cache_path.read_text())
  else:
    resp = k.get_markets(event_ticker=event_ticker, limit=200)
    markets = resp.get("markets", []) or []
    cache_path.write_text(json.dumps(markets))
    time.sleep(0.05)  # gentle pacing
  for m in markets:
    if (m.get("result") or "").lower() == "yes":
      return m
  return None


def kalshi_market_degrees(market: dict) -> tuple[str, list[int]]:
  """Return (label, degrees) for a Kalshi YES bracket."""
  label = market.get("subtitle") or market.get("yes_sub_title") or ""
  return label, _parse_kalshi_range(label)


# ── Polymarket ────────────────────────────────────────────────────────────

def fetch_all_closed_weather_events(p: PolymarketClient) -> list[dict]:
  """Cached pull of every closed Polymarket weather event."""
  if POLY_CACHE.exists():
    return json.loads(POLY_CACHE.read_text())
  events: list[dict] = []
  offset = 0
  page = 500
  while True:
    batch = p.search_events(tag_slug="weather", closed=True, limit=page, offset=offset)
    if not batch:
      break
    events.extend(batch)
    if len(batch) < page:
      break
    offset += page
    if offset > 20000:
      break
  POLY_CACHE.write_text(json.dumps(events))
  return events


def poly_event_date(event: dict) -> str | None:
  """
  Get the LOCAL measurement date from a Polymarket event title.
  Format: "Highest temperature in Austin on March 28?".
  """
  title = event.get("title") or ""
  m = re.search(
    r"on\s+(January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+(\d{1,2})(?:,\s*(\d{4}))?",
    title,
  )
  if not m:
    return None
  month_name, day, year = m.group(1), int(m.group(2)), m.group(3)
  if year:
    year_int = int(year)
  else:
    end = event.get("endDate") or ""
    try:
      year_int = datetime.fromisoformat(end.replace("Z", "+00:00")).year
    except (ValueError, TypeError):
      return None
  try:
    return datetime.strptime(
      f"{day} {month_name} {year_int}", "%d %B %Y"
    ).strftime("%Y-%m-%d")
  except ValueError:
    return None


def poly_winner(event: dict) -> dict | None:
  """Find the YES-resolved bracket inside a closed Polymarket event."""
  for m in event.get("markets", []) or []:
    raw = m.get("outcomePrices")
    try:
      prices = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
      continue
    if not prices:
      continue
    try:
      yes_price = float(prices[0])
    except (TypeError, ValueError, IndexError):
      continue
    if yes_price >= 0.99:
      return m
  return None
