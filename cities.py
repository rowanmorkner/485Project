"""
City configuration for the weather arbitrage bot.

Maps each supported city to its Kalshi series ticker, Polymarket search name,
and NWS gridpoints. IMPORTANT: Kalshi and Polymarket often resolve against
DIFFERENT weather stations, so we track separate NWS gridpoints for each.

Resolution stations (verified from market rules):
  - Kalshi: Uses NWS Climatological Report (Daily) for a specific station
  - Polymarket: Uses Wunderground historical data for a specific station
"""

# Each city entry contains:
#   kalshi_series:    Kalshi series ticker for daily high temp events
#   kalshi_station:   Human-readable Kalshi resolution station name
#   kalshi_nws:       (office, gridX, gridY) for the Kalshi resolution station
#   poly_station:     Human-readable Polymarket resolution station name
#   poly_nws:         (office, gridX, gridY) for the Polymarket resolution station
#   polymarket_name:  City name as it appears in Polymarket event titles
CITIES: dict[str, dict] = {
  "Miami": {
    "kalshi_series": "KXHIGHMIA",
    "kalshi_station": "Miami Intl Airport (KMIA)",
    "kalshi_nws": ("MFL", 106, 51),
    "poly_station": "Miami Intl Airport (KMIA)",
    "poly_nws": ("MFL", 106, 51),
    "polymarket_name": "Miami",
  },
  "Denver": {
    "kalshi_series": "KXHIGHDEN",
    "kalshi_station": "Denver Intl Airport (KDEN)",
    "kalshi_nws": ("BOU", 74, 66),
    "poly_station": "Buckley Space Force Base (KBKF)",
    "poly_nws": ("BOU", 71, 60),
    "polymarket_name": "Denver",
  },
  "Phoenix": {
    "kalshi_series": "KXHIGHTPHX",
    "kalshi_station": "Phoenix Sky Harbor (KPHX)",
    "kalshi_nws": ("PSR", 161, 57),
    "poly_station": "Phoenix Sky Harbor (KPHX)",
    "poly_nws": ("PSR", 161, 57),
    "polymarket_name": "Phoenix",
  },
  "NYC": {
    "kalshi_series": "KXHIGHNY",
    "kalshi_station": "Central Park, New York",
    "kalshi_nws": ("OKX", 34, 38),
    "poly_station": "LaGuardia Airport (KLGA)",
    "poly_nws": ("OKX", 37, 39),
    "polymarket_name": "NYC",
  },
  "Boston": {
    "kalshi_series": "KXHIGHTBOS",
    "kalshi_station": "Boston Logan Airport (KBOS)",
    "kalshi_nws": ("BOX", 73, 91),
    "poly_station": "Boston Logan Airport (KBOS)",
    "poly_nws": ("BOX", 73, 91),
    "polymarket_name": "Boston",
  },
  "LA": {
    "kalshi_series": "KXHIGHLAX",
    "kalshi_station": "Los Angeles Airport (KLAX)",
    "kalshi_nws": ("LOX", 148, 41),
    "poly_station": "Los Angeles Intl Airport (KLAX)",
    "poly_nws": ("LOX", 148, 41),
    "polymarket_name": "Los Angeles",
  },
  "Chicago": {
    "kalshi_series": "KXHIGHCHI",
    "kalshi_station": "Chicago Midway (KMDW)",
    "kalshi_nws": ("LOT", 72, 69),
    "poly_station": "Chicago O'Hare (KORD)",
    "poly_nws": ("LOT", 66, 77),
    "polymarket_name": "Chicago",
  },
}
