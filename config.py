"""
Project configuration: env-loaded credentials + the CITIES table.

All settings the rest of the project depends on live here. Secrets come
from .env (gitignored); the CITIES dict is hand-curated.
"""

import os
from dotenv import load_dotenv

# Load .env from project root before reading any env vars below
load_dotenv()


# ── Credentials ──────────────────────────────────────────────────────────

# Kalshi: RSA key-based auth (read + write)
KALSHI_API_KEY_ID = os.environ.get("KALSHI_API_KEY_ID", "")
KALSHI_PRIVATE_KEY_PATH = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "")

# Polymarket: no auth needed for read; loaded for future order placement
POLYMARKET_API_KEY = os.environ.get("POLYMARKET_API_KEY", "")

# Weather data (currently unused — NWS API is unauthenticated)
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY", "")


# ── Cities ───────────────────────────────────────────────────────────────
#
# Only cities where Kalshi and Polymarket resolve against the SAME physical
# weather station are listed — that's the precondition for clean arbitrage.
# Kalshi reads the NWS CLI (Climatological Report) product for the station;
# Polymarket reads Wunderground historical data for the same station.
#
# Per-city fields:
#   icao             ICAO code of the shared resolution station
#   station          Human-readable station name
#   nws              (office, gridX, gridY) for NWS forecast lookups
#   nws_cli_product  NWS CLI product ID Kalshi resolves against (e.g. "CLIMIA")
#   kalshi_series    Kalshi series ticker for daily high temp events
#   polymarket_name  City name as it appears in Polymarket event titles

CITIES: dict[str, dict] = {
  "Miami": {
    "icao": "KMIA",
    "station": "Miami International Airport",
    "nws": ("MFL", 106, 51),
    "nws_cli_product": "CLIMIA",
    "kalshi_series": "KXHIGHMIA",
    "polymarket_name": "Miami",
  },
  "LA": {
    "icao": "KLAX",
    "station": "Los Angeles International Airport",
    "nws": ("LOX", 148, 41),
    "nws_cli_product": "CLILAX",
    "kalshi_series": "KXHIGHLAX",
    "polymarket_name": "Los Angeles",
  },
  "Austin": {
    "icao": "KAUS",
    "station": "Austin-Bergstrom International Airport",
    "nws": ("EWX", 159, 88),
    "nws_cli_product": "CLIAUS",
    "kalshi_series": "KXHIGHAUS",
    "polymarket_name": "Austin",
  },
  "San Francisco": {
    "icao": "KSFO",
    "station": "San Francisco International Airport",
    "nws": ("MTR", 85, 98),
    "nws_cli_product": "CLISFO",
    "kalshi_series": "KXHIGHTSFO",
    "polymarket_name": "San Francisco",
  },
  "Seattle": {
    "icao": "KSEA",
    "station": "Seattle-Tacoma International Airport",
    "nws": ("SEW", 124, 61),
    "nws_cli_product": "CLISEA",
    "kalshi_series": "KXHIGHTSEA",
    "polymarket_name": "Seattle",
  },
}
