"""
National Weather Service (NWS) API client for weather forecast data.

Fetches hourly and daily forecasts from the NWS public API at
https://api.weather.gov. No API key is required, but the NWS mandates
a descriptive User-Agent header.

Currently configured for Miami International Airport (MFL/75,53).

API Reference: https://www.weather.gov/documentation/services-web-api
"""

import logging
from datetime import datetime
from typing import Optional
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Configure module logger
logger = logging.getLogger(__name__)

# ── API configuration ────────────────────────────────────────────────────
NWS_BASE_URL = "https://api.weather.gov"

# NWS policy requires a descriptive User-Agent with contact info
USER_AGENT = "WeatherArbBot (rowanmorkner@gmail.com)"

# Miami International Airport gridpoint (office=MFL, gridX=75, gridY=53)
MIAMI_OFFICE = "MFL"
MIAMI_GRID_X = 75
MIAMI_GRID_Y = 53


def _build_session(max_retries: int = 3, backoff_factor: float = 0.5) -> Session:
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


def _celsius_to_fahrenheit(celsius: float) -> float:
  """Convert a Celsius temperature to Fahrenheit."""
  return round(celsius * 9.0 / 5.0 + 32, 1)


class NWSClient:
  """
  Client for the National Weather Service API.

  Provides methods to fetch hourly and daily forecasts, with convenience
  methods for extracting daily high temperatures. Currently focused on
  the Miami International Airport gridpoint but easily extensible.

  No authentication is required. The NWS requires a User-Agent header
  with a contact email, which is set automatically.
  """

  def __init__(
    self,
    base_url: str = NWS_BASE_URL,
    user_agent: str = USER_AGENT,
  ):
    """
    Initialize the NWS client.

    Args:
      base_url: NWS API base URL. Defaults to https://api.weather.gov.
      user_agent: User-Agent header value. NWS requires a descriptive
                  string with contact info.
    """
    self.base_url = base_url.rstrip("/")
    self.user_agent = user_agent
    self.session = _build_session()

    # Set default headers for all requests
    self.session.headers.update({
      "User-Agent": self.user_agent,
      "Accept": "application/geo+json",
    })

  def _get(self, endpoint: str, params: Optional[dict] = None) -> dict:
    """
    Make a GET request to the NWS API.

    Args:
      endpoint: API path relative to base URL (e.g., "/gridpoints/MFL/75,53/forecast").
      params: Optional query parameters.

    Returns:
      Parsed JSON response as a dictionary.

    Raises:
      requests.HTTPError: If the API returns a non-2xx status code.
    """
    url = f"{self.base_url}{endpoint}"
    logger.debug("GET %s params=%s", url, params)
    resp = self.session.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()

  # ── Raw forecast endpoints ─────────────────────────────────────────────

  def get_gridpoint_forecast(
    self,
    office: str = MIAMI_OFFICE,
    grid_x: int = MIAMI_GRID_X,
    grid_y: int = MIAMI_GRID_Y,
  ) -> dict:
    """
    Fetch the daily forecast for a gridpoint.

    The NWS daily forecast returns ~14 periods, alternating between
    daytime and nighttime forecasts with high/low temperatures.

    Args:
      office: NWS forecast office code (e.g., "MFL" for Miami).
      grid_x: Grid X coordinate for the forecast point.
      grid_y: Grid Y coordinate for the forecast point.

    Returns:
      Full NWS forecast response (GeoJSON feature with "periods" array).
    """
    endpoint = f"/gridpoints/{office}/{grid_x},{grid_y}/forecast"
    return self._get(endpoint)

  def get_gridpoint_hourly_forecast(
    self,
    office: str = MIAMI_OFFICE,
    grid_x: int = MIAMI_GRID_X,
    grid_y: int = MIAMI_GRID_Y,
  ) -> dict:
    """
    Fetch the hourly forecast for a gridpoint.

    Returns per-hour temperature forecasts for ~156 hours (about 7 days).
    More granular than the daily forecast.

    Args:
      office: NWS forecast office code (e.g., "MFL" for Miami).
      grid_x: Grid X coordinate for the forecast point.
      grid_y: Grid Y coordinate for the forecast point.

    Returns:
      Full NWS hourly forecast response (GeoJSON feature with "periods" array).
    """
    endpoint = f"/gridpoints/{office}/{grid_x},{grid_y}/forecast/hourly"
    return self._get(endpoint)

  # ── Processed forecast methods ─────────────────────────────────────────

  def _ensure_fahrenheit(self, temperature: float, unit_code: str) -> float:
    """
    Ensure a temperature value is in Fahrenheit.

    The NWS API may return temps in Celsius (unitCode "wmoUnit:degC")
    or Fahrenheit (unitCode "wmoUnit:degF"). This method normalizes
    to Fahrenheit.

    Args:
      temperature: The raw temperature value from the API.
      unit_code: The unitCode field from the NWS period data.

    Returns:
      Temperature in Fahrenheit.
    """
    # NWS unitCode strings look like "wmoUnit:degC" or "wmoUnit:degF"
    if "degC" in unit_code or "celsius" in unit_code.lower():
      return _celsius_to_fahrenheit(temperature)
    # Already Fahrenheit (or assumed F if unrecognized)
    return float(temperature)

  def get_hourly_forecast(
    self,
    office: str = MIAMI_OFFICE,
    grid_x: int = MIAMI_GRID_X,
    grid_y: int = MIAMI_GRID_Y,
  ) -> list[dict]:
    """
    Get the hourly temperature forecast for a gridpoint.

    Returns a list of dicts, each containing:
      - start_time: ISO 8601 timestamp for the start of the hour
      - end_time: ISO 8601 timestamp for the end of the hour
      - temperature_f: Temperature in Fahrenheit
      - short_forecast: Brief text description (e.g., "Partly Cloudy")
      - wind_speed: Wind speed string (e.g., "10 mph")
      - wind_direction: Cardinal direction (e.g., "SE")
    """
    raw = self.get_gridpoint_hourly_forecast(office, grid_x, grid_y)
    periods = raw.get("properties", {}).get("periods", [])

    hourly_data = []
    for period in periods:
      unit_code = period.get("temperatureUnit", "F")
      # The hourly endpoint may use "temperatureUnit" (short) or nested unitCode
      # Check for the full unitCode field as well
      if "temperature" in period and isinstance(period["temperature"], dict):
        temp_val = period["temperature"]["value"]
        unit_code = period["temperature"].get("unitCode", unit_code)
      else:
        temp_val = period.get("temperature", 0)
        # NWS hourly forecast typically uses "temperatureUnit" field ("F" or "C")
        unit_str = period.get("temperatureUnit", "F")
        # Normalize short unit codes to match _ensure_fahrenheit expectations
        if unit_str == "C":
          unit_code = "wmoUnit:degC"
        else:
          unit_code = "wmoUnit:degF"

      temp_f = self._ensure_fahrenheit(temp_val, unit_code)

      hourly_data.append({
        "start_time": period.get("startTime"),
        "end_time": period.get("endTime"),
        "temperature_f": temp_f,
        "short_forecast": period.get("shortForecast", ""),
        "wind_speed": period.get("windSpeed", ""),
        "wind_direction": period.get("windDirection", ""),
      })

    return hourly_data

  def get_daily_high_forecast(
    self,
    office: str = MIAMI_OFFICE,
    grid_x: int = MIAMI_GRID_X,
    grid_y: int = MIAMI_GRID_Y,
  ) -> list[dict]:
    """
    Get the forecasted daily high temperature for a gridpoint.

    Extracts daytime periods from the NWS daily forecast (isDaytime=True)
    and returns the high temperature for each day.

    Returns a list of dicts with date, high_temperature_f, name,
    short_forecast, and detailed_forecast.
    """
    raw = self.get_gridpoint_forecast(office, grid_x, grid_y)
    periods = raw.get("properties", {}).get("periods", [])

    daily_highs = []
    for period in periods:
      # Only include daytime periods (these contain the daily high)
      if not period.get("isDaytime", False):
        continue

      unit_code = period.get("temperatureUnit", "F")
      if isinstance(period.get("temperature"), dict):
        temp_val = period["temperature"]["value"]
        unit_code = period["temperature"].get("unitCode", unit_code)
      else:
        temp_val = period.get("temperature", 0)
        unit_str = period.get("temperatureUnit", "F")
        if unit_str == "C":
          unit_code = "wmoUnit:degC"
        else:
          unit_code = "wmoUnit:degF"

      high_f = self._ensure_fahrenheit(temp_val, unit_code)

      # Parse the date from startTime (ISO 8601 format)
      start_time = period.get("startTime", "")
      try:
        date_str = datetime.fromisoformat(start_time).strftime("%Y-%m-%d")
      except (ValueError, TypeError):
        # Fallback: extract date from the ISO string directly
        date_str = start_time[:10] if len(start_time) >= 10 else "unknown"

      daily_highs.append({
        "date": date_str,
        "high_temperature_f": high_f,
        "name": period.get("name", ""),
        "short_forecast": period.get("shortForecast", ""),
        "detailed_forecast": period.get("detailedForecast", ""),
      })

    return daily_highs

  def get_forecast_high_for_date(
    self,
    date_str: str,
    office: str = MIAMI_OFFICE,
    grid_x: int = MIAMI_GRID_X,
    grid_y: int = MIAMI_GRID_Y,
  ) -> Optional[float]:
    """
    Get the forecasted daily high temperature for a specific date and gridpoint.

    First checks the daily forecast for an exact match. Falls back to the
    hourly forecast and takes the max temperature for that day.

    Args:
      date_str: Target date in "YYYY-MM-DD" format (e.g., "2026-04-01").
      office: NWS forecast office code.
      grid_x: Grid X coordinate.
      grid_y: Grid Y coordinate.

    Returns:
      Forecasted high temperature in Fahrenheit, or None if unavailable.
    """
    # Try the daily forecast first
    daily_highs = self.get_daily_high_forecast(office, grid_x, grid_y)
    for day in daily_highs:
      if day["date"] == date_str:
        logger.debug("Found daily high for %s: %.1f F", date_str, day["high_temperature_f"])
        return day["high_temperature_f"]

    # Fallback: derive daily high from hourly forecast data
    logger.debug("Date %s not in daily forecast, checking hourly...", date_str)
    hourly = self.get_hourly_forecast(office, grid_x, grid_y)
    temps_for_date = [
      h["temperature_f"]
      for h in hourly
      if h["start_time"] and h["start_time"].startswith(date_str)
    ]

    if temps_for_date:
      high = max(temps_for_date)
      logger.debug("Derived hourly high for %s: %.1f F (%d hours)", date_str, high, len(temps_for_date))
      return high

    logger.warning("Date %s is not within the NWS forecast range.", date_str)
    return None


# ── CLI demo / quick test ────────────────────────────────────────────────

if __name__ == "__main__":
  logging.basicConfig(level=logging.DEBUG)

  client = NWSClient()

  # Fetch and display the daily high forecast for Miami
  print("\n=== Miami Daily High Temperature Forecast ===")
  daily_highs = client.get_miami_daily_high_forecast()
  for day in daily_highs:
    print(
      f"  {day['date']} ({day['name']:>12s}): "
      f"{day['high_temperature_f']:.0f} F  --  {day['short_forecast']}"
    )

  # Fetch and display the first 24 hours of hourly forecast
  print("\n=== Miami Hourly Forecast (next 24 hours) ===")
  hourly = client.get_miami_hourly_forecast()
  for h in hourly[:24]:
    print(
      f"  {h['start_time'][:16]}  "
      f"{h['temperature_f']:5.1f} F  "
      f"{h['wind_speed']:>10s} {h['wind_direction']:>3s}  "
      f"{h['short_forecast']}"
    )

  # Demo: get the forecasted high for a specific date (tomorrow)
  if daily_highs:
    target_date = daily_highs[0]["date"]
    print(f"\n=== Forecast High for {target_date} ===")
    high = client.get_forecast_high_for_date(target_date)
    if high is not None:
      print(f"  Forecasted high: {high:.0f} F")
    else:
      print(f"  No forecast available for {target_date}")
