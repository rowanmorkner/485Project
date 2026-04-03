"""
Configuration loader — reads .env and exposes settings for the project.
"""

import os
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()

# --- Kalshi ---
KALSHI_API_KEY_ID = os.environ.get("KALSHI_API_KEY_ID", "")
KALSHI_PRIVATE_KEY_PATH = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "")

# --- Polymarket ---
# No auth needed for read-only, but load if present for future trading
POLYMARKET_API_KEY = os.environ.get("POLYMARKET_API_KEY", "")

# --- Weather API ---
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY", "")
