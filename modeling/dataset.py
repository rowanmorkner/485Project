"""
Build training datasets for all five cities.

The CITIES dict below merges the team's downstream-trading fields
(icao, station, kalshi_series, etc.) with the modeling-side fields
needed here (lat, lon, acis_station, timezone). One canonical registry
that both halves of the project share.

Run from project root:
    py dataset.py        # builds all five datasets
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# ===== City registry =====
# Each entry contains:
#   Trading-side: icao, station, nws, nws_cli_product, kalshi_series, polymarket_name
#   Modeling-side: lat, lon, acis_station, timezone

CITIES: dict[str, dict] = {
    "Miami": {
        "icao": "KMIA",
        "station": "Miami International Airport",
        "nws": ("MFL", 106, 51),
        "nws_cli_product": "CLIMIA",
        "kalshi_series": "KXHIGHMIA",
        "polymarket_name": "Miami",
        "lat": 25.7959,
        "lon": -80.2870,
        "acis_station": "USW00012839",
        "timezone": "America/New_York",
    },
    "LA": {
        "icao": "KLAX",
        "station": "Los Angeles International Airport",
        "nws": ("LOX", 148, 41),
        "nws_cli_product": "CLILAX",
        "kalshi_series": "KXHIGHLAX",
        "polymarket_name": "Los Angeles",
        "lat": 33.9425,
        "lon": -118.4081,
        "acis_station": "USW00023174",
        "timezone": "America/Los_Angeles",
    },
    "Austin": {
        "icao": "KAUS",
        "station": "Austin-Bergstrom International Airport",
        "nws": ("EWX", 159, 88),
        "nws_cli_product": "CLIAUS",
        "kalshi_series": "KXHIGHAUS",
        "polymarket_name": "Austin",
        "lat": 30.1944,
        "lon": -97.6700,
        "acis_station": "USW00013904",
        "timezone": "America/Chicago",
    },
    "San Francisco": {
        "icao": "KSFO",
        "station": "San Francisco International Airport",
        "nws": ("MTR", 85, 98),
        "nws_cli_product": "CLISFO",
        "kalshi_series": "KXHIGHTSFO",
        "polymarket_name": "San Francisco",
        "lat": 37.6213,
        "lon": -122.3790,
        "acis_station": "USW00023234",
        "timezone": "America/Los_Angeles",
    },
    "Seattle": {
        "icao": "KSEA",
        "station": "Seattle-Tacoma International Airport",
        "nws": ("SEW", 124, 61),
        "nws_cli_product": "CLISEA",
        "kalshi_series": "KXHIGHTSEA",
        "polymarket_name": "Seattle",
        "lat": 47.4444,
        "lon": -122.3139,
        "acis_station": "USW00024233",
        "timezone": "America/Los_Angeles",
    },
}


def city_slug(city_name: str) -> str:
    """Filename-safe key derived from the display name."""
    return city_name.lower().replace(" ", "_")


# ===== NOAA ACIS targets =====

ACIS_URL = "http://data.rcc-acis.org/StnData"


def fetch_acis_daily(station_id, start_date, end_date, timeout=60):
    """Pull daily max temperature observations from NOAA ACIS for a station.
    Returns DataFrame indexed by date with column 'tmax_actual' in °F."""
    payload = {
        "sid": station_id,
        "sdate": start_date,
        "edate": end_date,
        "elems": [{"name": "maxt", "interval": "dly"}],
    }
    resp = requests.post(ACIS_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        raise RuntimeError(f"ACIS error: {data['error']}")

    df = pd.DataFrame(data["data"], columns=["date", "tmax_actual"])
    df["date"] = pd.to_datetime(df["date"])
    df["tmax_actual"] = df["tmax_actual"].replace({"M": pd.NA, "T": "0.0"})
    df["tmax_actual"] = pd.to_numeric(df["tmax_actual"], errors="coerce")
    return df.set_index("date").sort_index()


# ===== Open-Meteo features (ERA5 reanalysis archive) =====

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

DAILY_VARS = [
    "temperature_2m_max", "temperature_2m_min",
    "apparent_temperature_max", "apparent_temperature_min",
    "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours",
    "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
    "shortwave_radiation_sum", "et0_fao_evapotranspiration", "cloudcover_mean",
]

HOURLY_VARS = [
    "surface_pressure",
    "cloudcover_low", "cloudcover_high",
    "soil_temperature_0_to_7cm", "soil_moisture_0_to_7cm",
]


def fetch_archive_features(lat, lon, start_date, end_date, timezone, timeout=120):
    """Pull atmospheric features from Open-Meteo's ERA5 reanalysis archive.
    All temperatures in °F, precipitation in inches, wind in mph.
    Returns DataFrame indexed by date including day-of-year sin/cos encoding."""
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "daily": ",".join(DAILY_VARS),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": timezone,
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "windspeed_unit": "mph",
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    daily = pd.DataFrame(data["daily"])
    daily["time"] = pd.to_datetime(daily["time"])
    daily = daily.set_index("time")

    hourly = pd.DataFrame(data["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"])
    hourly_daily = hourly.set_index("time").resample("D").mean()

    features = daily.join(hourly_daily)
    features.index.name = "date"

    doy = features.index.dayofyear
    features["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    features["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return features.sort_index()


# ===== Build dataset =====

ARCHIVE_LAG_DAYS = 7   # Open-Meteo archive lags real-time by a few days


def build_dataset(city_name, start_date, end_date, output_dir="data/datasets"):
    """Build the training parquet for one city. Day-ahead target: features
    at date D predict the high temperature on date D+1."""
    cfg = CITIES[city_name]
    slug = city_slug(city_name)
    print(f"Building dataset for {city_name} [{start_date} → {end_date}]")

    print("  → ACIS targets...")
    targets = fetch_acis_daily(cfg["acis_station"], start_date, end_date)
    print(f"    got {len(targets)} rows")

    print("  → Open-Meteo features...")
    features = fetch_archive_features(
        cfg["lat"], cfg["lon"], start_date, end_date, cfg["timezone"]
    )
    print(f"    got {len(features)} rows × {features.shape[1]} cols")

    # Day-ahead shift
    joined = features.join(targets[["tmax_actual"]], how="inner").sort_index()
    joined["tmax_actual"] = joined["tmax_actual"].shift(-1)
    df = joined.dropna(subset=["tmax_actual"])

    out_path = Path(output_dir) / f"{slug}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)

    print(f"  ✓ {len(df)} rows → {out_path}")
    print(f"    {df.index.min().date()} → {df.index.max().date()}")
    print(f"    target tmax_actual: "
          f"min={df['tmax_actual'].min():.1f}°F  "
          f"max={df['tmax_actual'].max():.1f}°F  "
          f"mean={df['tmax_actual'].mean():.1f}°F")
    return df


if __name__ == "__main__":
    if __name__ == "__main__":
        import time

        end = date.today() - timedelta(days=ARCHIVE_LAG_DAYS)
        start = end - timedelta(days=365 * 10)

        for i, city in enumerate(CITIES):
            if i > 0:
                print(f"  (sleeping 30s to avoid rate limit)")
                time.sleep(30)
            try:
                build_dataset(city, start.isoformat(), end.isoformat())
                print()
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    print(f"  ! rate-limited, waiting 90s and retrying once...")
                    time.sleep(90)
                    try:
                        build_dataset(city, start.isoformat(), end.isoformat())
                        print()
                    except Exception as e2:
                        print(f"  ! retry failed: {e2}\n")
                else:
                    print(f"  ! failed: {e}\n")
            except Exception as e:
                print(f"  ! failed: {e}\n")
