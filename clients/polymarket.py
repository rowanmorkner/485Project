"""
Polymarket API client for weather/temperature market discovery and pricing.

Uses two Polymarket APIs:
  - Gamma API (https://gamma-api.polymarket.com): Public market discovery/search.
    No authentication required for read operations.
  - CLOB API (https://clob.polymarket.com): Central Limit Order Book for
    prices, orderbooks, and trading. Read endpoints are public; trading
    requires API key + wallet signing.

Reference docs: https://docs.polymarket.com/
"""

import os
import logging
from typing import Optional
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure module logger
logger = logging.getLogger(__name__)

# ── API base URLs ──────────────────────────────────────────────────────────
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
CLOB_API_BASE = "https://clob.polymarket.com"


def _build_session(max_retries: int = 3, backoff_factor: float = 0.3) -> Session:
  """
  Build a requests.Session with automatic retry logic for transient errors.
  Retries on 429 (rate-limit) and 5xx server errors.
  """
  session = Session()
  retry_strategy = Retry(
    total=max_retries,
    backoff_factor=backoff_factor,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
  )
  adapter = HTTPAdapter(max_retries=retry_strategy)
  session.mount("https://", adapter)
  session.mount("http://", adapter)
  return session


class PolymarketClient:
  """
  Client for reading public Polymarket data — market search, prices,
  and orderbooks. No API key is required for read-only operations.

  Optional env vars (only needed if you want to place orders later):
    POLYMARKET_API_KEY   – CLOB API key
    POLYMARKET_SECRET    – CLOB API secret
    POLYMARKET_PASSPHRASE – CLOB API passphrase
  """

  def __init__(
    self,
    gamma_base: str = GAMMA_API_BASE,
    clob_base: str = CLOB_API_BASE,
  ):
    self.gamma_base = gamma_base.rstrip("/")
    self.clob_base = clob_base.rstrip("/")
    self.session = _build_session()

    # Load optional credentials from environment (only for future write ops)
    self.api_key = os.environ.get("POLYMARKET_API_KEY")
    self.api_secret = os.environ.get("POLYMARKET_SECRET")
    self.api_passphrase = os.environ.get("POLYMARKET_PASSPHRASE")

  # ── Gamma API: Event & Market discovery ────────────────────────────────

  def search_events(
    self,
    query: str = "",
    slug: Optional[str] = None,
    tag_slug: Optional[str] = None,
    limit: int = 25,
    offset: int = 0,
    closed: Optional[bool] = None,
  ) -> list[dict]:
    """
    Search for events on the Gamma API. Events group related markets
    together (e.g., all temperature brackets for a single day).

    Note: the _q free-text search is unreliable for weather markets.
    Prefer tag_slug for reliable filtering (e.g. tag_slug="temperature").

    Args:
      query: Free-text search via _q param (unreliable for some categories).
      slug: Exact event slug match.
      tag_slug: Filter by tag slug (e.g. "temperature", "miami", "weather").
      limit: Max results to return.
      offset: Pagination offset.
      closed: Filter by open (False) or resolved (True).

    Returns list of event dicts, each containing nested "markets" list.
    """
    params: dict = {"limit": limit, "offset": offset}
    if query:
      params["_q"] = query
    if slug:
      params["slug"] = slug
    if tag_slug:
      params["tag_slug"] = tag_slug
    if closed is not None:
      params["closed"] = str(closed).lower()

    url = f"{self.gamma_base}/events"
    logger.debug("GET %s params=%s", url, params)
    resp = self.session.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()

  def get_event_by_slug(self, slug: str) -> Optional[dict]:
    """
    Fetch a single event by its URL slug.
    Returns the event dict with nested markets, or None if not found.
    """
    events = self.search_events(slug=slug, limit=1)
    return events[0] if events else None

  def search_weather_events(
    self,
    city: str = "Miami",
    only_open: bool = True,
  ) -> list[dict]:
    """
    Search for daily-high-temperature events by city name.

    Polymarket files these under tag_slug="weather" (~180 events globally).
    We pull all of them, then filter by both city name AND the "Highest
    temperature" prefix to exclude the matching "Lowest temperature" events.
    """
    closed = False if only_open else None
    events = self.search_events(
      tag_slug="weather",
      closed=closed,
      limit=500,
    )
    city_lower = city.lower()
    return [
      e for e in events
      if city_lower in e.get("title", "").lower()
      and "highest temperature" in e.get("title", "").lower()
    ]

  def search_markets(
    self,
    query: str,
    limit: int = 25,
    offset: int = 0,
    closed: Optional[bool] = None,
    tag: Optional[str] = None,
  ) -> list[dict]:
    """
    Search for markets on the Gamma API by free-text query and optional filters.

    The Gamma /markets endpoint supports these query parameters:
      - limit / offset  : pagination
      - closed          : filter by open (false) or resolved (true)
      - tag             : filter by tag slug (e.g. "weather", "sports")
      - slug            : exact slug match
      - active          : boolean, whether market is actively trading
      - _q              : free-text keyword search against question field

    Returns a list of market dicts, each containing fields like:
      id, question, slug, conditionId, tokens[], outcomes[], outcomePrices[],
      startDate, endDate, volume, liquidity, ...
    """
    params: dict = {
      "limit": limit,
      "offset": offset,
    }
    # Free-text search — Gamma uses the undocumented but widely-used _q param
    if query:
      params["_q"] = query
    if closed is not None:
      params["closed"] = str(closed).lower()
    if tag:
      params["tag"] = tag

    url = f"{self.gamma_base}/markets"
    logger.debug("GET %s params=%s", url, params)
    resp = self.session.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()

  def search_weather_markets(
    self,
    city: str = "Miami",
    keyword: str = "highest temperature",
    limit: int = 25,
    only_open: bool = True,
  ) -> list[dict]:
    """
    Convenience method: search for weather/temperature markets for a city.

    Combines a free-text query with the 'weather' tag filter to narrow
    results to temperature-related prediction markets.

    Args:
      city: City name to search for (default "Miami").
      keyword: Additional keyword for the search (default "highest temperature").
      limit: Max results to return.
      only_open: If True, exclude resolved/closed markets.

    Returns list of matching market dicts.
    """
    query_text = f"{keyword} {city}"
    closed = False if only_open else None
    return self.search_markets(
      query=query_text,
      limit=limit,
      closed=closed,
    )

  def get_market_by_id(self, market_id: str) -> dict:
    """
    Fetch a single market by its Gamma market ID (numeric string).
    The Gamma API exposes individual markets at /markets/{id}.
    """
    url = f"{self.gamma_base}/markets/{market_id}"
    logger.debug("GET %s", url)
    resp = self.session.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json()

  def get_market_by_slug(self, slug: str) -> list[dict]:
    """
    Fetch market(s) by their URL slug. Slugs are the human-readable
    identifiers used in Polymarket URLs (e.g. "will-miami-hit-100f-in-2026").
    """
    url = f"{self.gamma_base}/markets"
    params = {"slug": slug, "limit": 1}
    logger.debug("GET %s params=%s", url, params)
    resp = self.session.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()

  # ── CLOB API: Prices & orderbook ───────────────────────────────────────

  def get_clob_market(self, condition_id: str) -> dict:
    """
    Fetch market details from the CLOB by condition_id.

    The CLOB /markets/{condition_id} endpoint returns:
      condition_id, tokens (list of {token_id, outcome}), min_tick_size,
      question, description, ...

    The token_id values returned here are what you pass to the price
    and orderbook endpoints.
    """
    url = f"{self.clob_base}/markets/{condition_id}"
    logger.debug("GET %s", url)
    resp = self.session.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json()

  def get_price(self, token_id: str, side: str = "buy") -> dict:
    """
    Get the current best price for a token.

    Args:
      token_id: The CLOB token ID (from market.tokens[].token_id).
      side: "buy" or "sell" — which side of the book to query.

    Returns dict with {"price": "0.65"} or similar.
    """
    url = f"{self.clob_base}/price"
    params = {"token_id": token_id, "side": side}
    logger.debug("GET %s params=%s", url, params)
    resp = self.session.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()

  def get_midpoint_price(self, token_id: str) -> dict:
    """
    Get the midpoint price for a token (average of best bid and best ask).

    Returns dict with {"mid": "0.63"} or similar.
    """
    url = f"{self.clob_base}/midpoint"
    params = {"token_id": token_id}
    logger.debug("GET %s params=%s", url, params)
    resp = self.session.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()

  def get_orderbook(self, token_id: str) -> dict:
    """
    Fetch the full orderbook (bids and asks) for a token.

    Returns dict with structure:
      {
        "market": "...",
        "asset_id": "...",
        "bids": [{"price": "0.60", "size": "100"}, ...],
        "asks": [{"price": "0.65", "size": "50"}, ...],
        "hash": "...",
        "timestamp": "..."
      }
    """
    url = f"{self.clob_base}/book"
    params = {"token_id": token_id}
    logger.debug("GET %s params=%s", url, params)
    resp = self.session.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()

  def get_spread(self, token_id: str) -> dict:
    """
    Get the bid-ask spread for a token.

    Returns dict with {"spread": "0.02"} or similar.
    """
    url = f"{self.clob_base}/spread"
    params = {"token_id": token_id}
    logger.debug("GET %s params=%s", url, params)
    resp = self.session.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()

  # ── Convenience: end-to-end market pricing ─────────────────────────────

  def get_market_prices(self, condition_id: str) -> list[dict]:
    """
    High-level method: given a condition_id, fetch the CLOB market info
    and then retrieve buy/sell prices + midpoint for each token (outcome).

    Returns a list of dicts, one per outcome:
      [
        {
          "outcome": "Yes",
          "token_id": "abc123...",
          "buy_price": "0.65",
          "sell_price": "0.63",
          "midpoint": "0.64",
        },
        ...
      ]
    """
    # Step 1: Get market details to learn the token IDs
    market = self.get_clob_market(condition_id)
    tokens = market.get("tokens", [])
    if not tokens:
      logger.warning("No tokens found for condition_id=%s", condition_id)
      return []

    # Step 2: For each token/outcome, fetch pricing data
    results = []
    for token in tokens:
      token_id = token["token_id"]
      outcome = token.get("outcome", "Unknown")
      try:
        buy_price = self.get_price(token_id, side="buy")
        sell_price = self.get_price(token_id, side="sell")
        midpoint = self.get_midpoint_price(token_id)
        results.append({
          "outcome": outcome,
          "token_id": token_id,
          "buy_price": buy_price.get("price"),
          "sell_price": sell_price.get("price"),
          "midpoint": midpoint.get("mid"),
        })
      except Exception as exc:
        logger.error("Failed to get prices for token %s: %s", token_id, exc)
        results.append({
          "outcome": outcome,
          "token_id": token_id,
          "error": str(exc),
        })
    return results

  def get_market_summary(self, condition_id: str) -> dict:
    """
    High-level method: fetch a combined summary with CLOB market info,
    orderbook depth, and pricing for all outcomes.

    Useful for getting a complete snapshot of a market's state.
    """
    market = self.get_clob_market(condition_id)
    tokens = market.get("tokens", [])

    outcomes = []
    for token in tokens:
      token_id = token["token_id"]
      outcome = token.get("outcome", "Unknown")
      try:
        book = self.get_orderbook(token_id)
        midpoint = self.get_midpoint_price(token_id)
        outcomes.append({
          "outcome": outcome,
          "token_id": token_id,
          "midpoint": midpoint.get("mid"),
          "bid_count": len(book.get("bids", [])),
          "ask_count": len(book.get("asks", [])),
          "best_bid": book["bids"][0]["price"] if book.get("bids") else None,
          "best_ask": book["asks"][0]["price"] if book.get("asks") else None,
        })
      except Exception as exc:
        logger.error("Failed to get summary for token %s: %s", token_id, exc)
        outcomes.append({
          "outcome": outcome,
          "token_id": token_id,
          "error": str(exc),
        })

    return {
      "condition_id": condition_id,
      "question": market.get("question"),
      "description": market.get("description"),
      "min_tick_size": market.get("min_tick_size"),
      "outcomes": outcomes,
    }


# ── CLI demo / quick test ────────────────────────────────────────────────

if __name__ == "__main__":
  import json

  logging.basicConfig(level=logging.DEBUG)

  client = PolymarketClient()

  # Search for Miami temperature markets
  print("\n=== Searching for Miami temperature markets ===")
  markets = client.search_weather_markets(city="Miami")
  print(f"Found {len(markets)} market(s):\n")

  for mkt in markets[:5]:
    # Each gamma market has an 'id', 'question', 'conditionId', 'outcomePrices'
    print(f"  ID: {mkt.get('id')}")
    print(f"  Question: {mkt.get('question')}")
    print(f"  Slug: {mkt.get('slug')}")
    print(f"  Condition ID: {mkt.get('conditionId')}")
    print(f"  Outcome Prices: {mkt.get('outcomePrices')}")
    print(f"  Active: {mkt.get('active')}")
    print(f"  Closed: {mkt.get('closed')}")
    print()

    # If the market has a conditionId, fetch CLOB prices
    condition_id = mkt.get("conditionId")
    if condition_id:
      print(f"  --- CLOB Prices for condition {condition_id} ---")
      try:
        prices = client.get_market_prices(condition_id)
        print(f"  {json.dumps(prices, indent=4)}")
      except Exception as e:
        print(f"  Error fetching CLOB prices: {e}")
      print()
