"""
Kalshi API Client

Handles RSA-signed authentication and provides methods for interacting
with the Kalshi prediction market API. Specifically targets weather/temperature
markets for the MATH485 weather arbitrage project.

API Reference: https://trading-api.readme.io/reference
Production base URL: https://trading-api.kalshi.com/trade-api/v2
"""

import os
import time
import base64
from typing import Optional
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
import requests


# --- Constants ---

# Kalshi production API base URL
BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"



class KalshiClient:
  """
  Client for the Kalshi prediction market API.

  Authentication uses RSA-PSS signing with SHA-256. Each request
  is signed by concatenating: timestamp_ms + method + path, then signing
  that string with the user's RSA private key.

  Required environment variables:
    KALSHI_API_KEY_ID      - Your Kalshi API key ID (UUID format)
    KALSHI_PRIVATE_KEY_PATH - Path to your RSA private key PEM file
  """

  def __init__(
    self,
    api_key_id: Optional[str] = None,
    private_key_path: Optional[str] = None,
    base_url: str = BASE_URL,
  ):
    """
    Initialize the Kalshi client.

    Args:
      api_key_id: Kalshi API key ID. Falls back to KALSHI_API_KEY_ID env var.
      private_key_path: Path to RSA private key PEM file. Falls back to
                        KALSHI_PRIVATE_KEY_PATH env var.
      base_url: API base URL. Defaults to Kalshi production.
    """
    self.api_key_id = api_key_id or os.environ.get("KALSHI_API_KEY_ID", "")
    self.private_key_path = private_key_path or os.environ.get("KALSHI_PRIVATE_KEY_PATH", "")
    self.base_url = base_url.rstrip("/")
    self.session = requests.Session()


    # Load and parse the RSA private key once at init
    self._private_key = self._load_private_key()

  def _load_private_key(self):
    """
    Load the RSA private key from the PEM file specified by private_key_path.

    Returns:
      An RSA private key object from the cryptography library.

    Raises:
      FileNotFoundError: If the key file doesn't exist.
      ValueError: If the key file is not a valid RSA private key.
    """
    key_path = os.path.expanduser(self.private_key_path)
    with open(key_path, "rb") as key_file:
      private_key = serialization.load_pem_private_key(
        key_file.read(),
        password=None,  # Kalshi keys are typically unencrypted
      )
    return private_key

  def _sign_request(self, method: str, path: str, timestamp_ms: str) -> str:
    """
    Generate the RSA signature for a Kalshi API request.

    Kalshi's auth scheme signs the concatenation of:
      timestamp_ms (string) + HTTP_METHOD (uppercase) + path (no host, e.g. /trade-api/v2/markets)

    The signature uses RSA-PSS padding with SHA-256 (MGF1 + SHA-256,
    salt length = digest length), then is base64-encoded for the header.

    Args:
      method: HTTP method (GET, POST, etc.), will be uppercased.
      path: Request path including /trade-api/v2 prefix (no host/scheme).
      timestamp_ms: Unix timestamp in milliseconds as a string.

    Returns:
      Base64-encoded RSA signature string.
    """
    # Build the message to sign: timestamp + method + path
    message = timestamp_ms + method.upper() + path
    signature = self._private_key.sign(
      message.encode("utf-8"),
      padding.PSS(
        mgf=padding.MGF1(hashes.SHA256()),
        salt_length=padding.PSS.DIGEST_LENGTH,
      ),
      hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("utf-8")

  def _build_auth_headers(self, method: str, path: str) -> dict:
    """
    Build the authentication headers required for a Kalshi API request.

    Headers:
      KALSHI-ACCESS-KEY       - The API key ID
      KALSHI-ACCESS-TIMESTAMP - Current unix timestamp in milliseconds
      KALSHI-ACCESS-SIGNATURE - RSA signature of (timestamp + method + path)

    Args:
      method: HTTP method for the request.
      path: Full request path (e.g., /trade-api/v2/markets).

    Returns:
      Dictionary of authentication headers.
    """
    # Kalshi expects the timestamp as an integer string of milliseconds
    timestamp_ms = str(int(time.time() * 1000))
    signature = self._sign_request(method, path, timestamp_ms)

    return {
      "KALSHI-ACCESS-KEY": self.api_key_id,
      "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
      "KALSHI-ACCESS-SIGNATURE": signature,
      "Content-Type": "application/json",
      "Accept": "application/json",
    }

  def _request(self, method: str, endpoint: str, params: Optional[dict] = None, json_body: Optional[dict] = None) -> dict:
    """
    Make an authenticated request to the Kalshi API.

    Args:
      method: HTTP method (GET, POST, DELETE, etc.).
      endpoint: API endpoint path relative to base URL (e.g., "/markets").
      params: Optional query parameters for GET requests.
      json_body: Optional JSON body for POST/PUT requests.

    Returns:
      Parsed JSON response as a dictionary.

    Raises:
      requests.HTTPError: If the API returns a non-2xx status code.
    """
    # Construct the full path (used for signing) and full URL
    path = f"/trade-api/v2{endpoint}"
    url = f"{self.base_url}{endpoint}"

    # Build auth headers using the path (no query params in signature)
    headers = self._build_auth_headers(method.upper(), path)

    response = self.session.request(
      method=method.upper(),
      url=url,
      headers=headers,
      params=params,
      json=json_body,
    )

    # Raise on HTTP error status codes
    response.raise_for_status()
    return response.json()

  # --- Event & Market Discovery ---

  def get_events(
    self,
    series_ticker: Optional[str] = None,
    status: Optional[str] = None,
    cursor: Optional[str] = None,
    limit: int = 100,
  ) -> dict:
    """
    List events, optionally filtered by series ticker or status.

    Events group related markets together (e.g., all temperature brackets
    for a specific day in Miami).

    Args:
      series_ticker: Filter by series (e.g., "KXHIGHTMIAMI" for Miami high temp).
      status: Filter by status: "open", "closed", "settled".
      cursor: Pagination cursor from a previous response.
      limit: Max results per page (default 100, max 200).

    Returns:
      API response dict with "events" list and "cursor" for pagination.
    """
    params = {"limit": limit}
    if series_ticker:
      params["series_ticker"] = series_ticker
    if status:
      params["status"] = status
    if cursor:
      params["cursor"] = cursor

    return self._request("GET", "/events", params=params)

  def get_event(self, event_ticker: str) -> dict:
    """
    Get details for a specific event by its ticker.

    Args:
      event_ticker: The event ticker (e.g., "KXHIGHTMIAMI-26APR01").

    Returns:
      API response dict with event details and its child markets.
    """
    return self._request("GET", f"/events/{event_ticker}")

  def get_markets(
    self,
    event_ticker: Optional[str] = None,
    series_ticker: Optional[str] = None,
    status: Optional[str] = None,
    tickers: Optional[str] = None,
    cursor: Optional[str] = None,
    limit: int = 100,
  ) -> dict:
    """
    List markets, optionally filtered by event, series, or status.

    Each market represents a specific yes/no contract (e.g., "Miami high
    temp >= 85F on April 1").

    Args:
      event_ticker: Filter to markets under a specific event.
      series_ticker: Filter to markets under a specific series.
      status: Filter by status: "open", "closed", "settled".
      tickers: Comma-separated list of specific market tickers.
      cursor: Pagination cursor from a previous response.
      limit: Max results per page (default 100, max 1000).

    Returns:
      API response dict with "markets" list and "cursor" for pagination.
    """
    params = {"limit": limit}
    if event_ticker:
      params["event_ticker"] = event_ticker
    if series_ticker:
      params["series_ticker"] = series_ticker
    if status:
      params["status"] = status
    if tickers:
      params["tickers"] = tickers
    if cursor:
      params["cursor"] = cursor

    return self._request("GET", "/markets", params=params)

  def get_market(self, ticker: str) -> dict:
    """
    Get details for a specific market by its ticker.

    Includes current pricing info (yes_bid, yes_ask, last_price, volume, etc.).

    Args:
      ticker: The market ticker (e.g., "KXHIGHTMIAMI-26APR01-T85").

    Returns:
      API response dict with full market details.
    """
    return self._request("GET", f"/markets/{ticker}")

  # --- Pricing & Orderbook ---

  def get_orderbook(self, ticker: str, depth: Optional[int] = None) -> dict:
    """
    Get the full orderbook for a specific market.

    The orderbook contains arrays of [price, quantity] for both yes and no
    sides at each price level.

    Args:
      ticker: The market ticker to get the orderbook for.
      depth: Optional limit on the number of price levels returned.

    Returns:
      API response dict with "orderbook" containing:
        - "yes": list of [price, quantity] entries (best ask first)
        - "no":  list of [price, quantity] entries (best ask first)
    """
    params = {}
    if depth is not None:
      params["depth"] = depth

    return self._request("GET", f"/markets/{ticker}/orderbook", params=params)

  def get_market_history(
    self,
    ticker: str,
    cursor: Optional[str] = None,
    limit: int = 100,
    min_ts: Optional[int] = None,
    max_ts: Optional[int] = None,
  ) -> dict:
    """
    Get trade history / candlestick data for a market.

    Args:
      ticker: The market ticker.
      cursor: Pagination cursor.
      limit: Max results per page.
      min_ts: Minimum unix timestamp (seconds) filter.
      max_ts: Maximum unix timestamp (seconds) filter.

    Returns:
      API response dict with historical trade/price snapshots.
    """
    params = {"limit": limit}
    if cursor:
      params["cursor"] = cursor
    if min_ts is not None:
      params["min_ts"] = min_ts
    if max_ts is not None:
      params["max_ts"] = max_ts

    return self._request("GET", f"/markets/{ticker}/history", params=params)

  # --- Weather-Specific Helpers ---

  def search_high_temp_events(self, series_ticker: str, status: str = "open") -> list:
    """
    Search for highest temperature events for a given series.

    Args:
      series_ticker: Kalshi series ticker (e.g. "KXHIGHMIA", "KXHIGHDEN").
      status: Filter by event status. Default "open" to get active events.

    Returns:
      List of event dicts (each event typically corresponds to one day).
    """
    results = []
    cursor = None

    while True:
      response = self.get_events(
        series_ticker=series_ticker,
        status=status,
        cursor=cursor,
      )
      events = response.get("events", [])
      results.extend(events)

      cursor = response.get("cursor")
      if not cursor or not events:
        break

    return results


# --- Standalone usage example ---

if __name__ == "__main__":
  import json

  # Initialize client from environment variables
  client = KalshiClient()

  # Find open Miami high temperature events
  print("Searching for Miami high temperature events...")
  events = client.search_miami_high_temp_events(status="open")
  print(f"Found {len(events)} open Miami high temp events.\n")

  if events:
    # Show the first event
    first_event = events[0]
    event_ticker = first_event.get("event_ticker", "")
    print(f"First event: {event_ticker}")
    print(f"  Title: {first_event.get('title', 'N/A')}")
    print(f"  Category: {first_event.get('category', 'N/A')}")
    print()

    # Get prices for all brackets in that event
    print(f"Fetching prices for {event_ticker}...")
    prices = client.get_miami_temp_prices(event_ticker)
    for p in prices:
      print(
        f"  {p['ticker']:40s} | "
        f"Yes: {p['yes_bid']}/{p['yes_ask']}  "
        f"Last: {p['last_price']}  "
        f"Vol: {p['volume']}"
      )
    print()

    # Get the orderbook for the first market
    if prices:
      ticker = prices[0]["ticker"]
      print(f"Orderbook for {ticker}:")
      ob = client.get_orderbook(ticker)
      print(json.dumps(ob, indent=2))
