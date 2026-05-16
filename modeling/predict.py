"""
predict.py — Live deployment.

Trains the ensemble on all historical data for a city, fetches the current
atmospheric state from Open-Meteo's live forecast API, and produces a
probabilistic forecast for tomorrow's daily high temperature.

Run from project root:
    py predict.py                   # all cities in the registry
    py predict.py "San Francisco"   # one city
    py predict.py "Miami" "LA"      # multiple cities

Limitations to be aware of:
    - Training features come from ERA5 reanalysis (smoothed, post-hoc).
      Inference features come from Open-Meteo's live forecast model
      (noisier, real-time). Deployment MAE will be modestly worse than
      what train.py reports on the test set.
    - "Today's" daily aggregates are partial during the day. For cleanest
      match to training-time semantics, run this in the evening (local time).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from dataset import CITIES, DAILY_VARS, HOURLY_VARS, city_slug
from models import (EnsembleModel, HeteroscedasticRidgeModel,
                     NGBoostNormalModel, XGBMeanVarModel)

LIVE_URL = "https://api.open-meteo.com/v1/forecast"
DATASETS_DIR = Path("data/datasets")
TARGET = "tmax_actual"


def fetch_live_features(city_name: str) -> pd.DataFrame:
    """Fetch live atmospheric features (yesterday + today + tomorrow) for a city."""
    cfg = CITIES[city_name]
    params = {
        "latitude": cfg["lat"],
        "longitude": cfg["lon"],
        "daily": ",".join(DAILY_VARS),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": cfg["timezone"],
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "windspeed_unit": "mph",
        # Yesterday + today + tomorrow + day-after gives enough feature rows
        # to predict the next ~3 target tmax days using the day-ahead model.
        "past_days": 1,
        "forecast_days": 3,
    }
    resp = requests.get(LIVE_URL, params=params, timeout=60)
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


def train_ensemble(city_name: str):
    """Train the ensemble on ALL historical data for the city (no holdout)."""
    slug = city_slug(city_name)
    parquet_path = DATASETS_DIR / f"{slug}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"No parquet at {parquet_path}. Run `py dataset.py` first."
        )

    df = pd.read_parquet(parquet_path).sort_index()
    feature_names = [c for c in df.columns if c != TARGET]
    X, y = df[feature_names], df[TARGET]
    print(f"  Training on {len(X)} rows "
          f"({df.index.min().date()} → {df.index.max().date()})")

    ridge = HeteroscedasticRidgeModel().fit(X, y)
    xgb = XGBMeanVarModel().fit(X, y)
    ngb = NGBoostNormalModel().fit(X, y)
    ensemble = EnsembleModel([ridge, xgb, ngb])
    return ensemble, feature_names, df


def align_features(X_today: pd.DataFrame,
                   feature_names: list[str],
                   df_train: pd.DataFrame) -> pd.DataFrame:
    X = X_today.copy()
    for c in feature_names:
        if c not in X.columns:
            print(f"  ! live API didn't return '{c}'; using historical mean")
            X[c] = float(df_train[c].mean())
        elif X[c].isna().any():
            X[c] = X[c].astype(float).fillna(float(df_train[c].mean()))
    return X[feature_names]


def predict_for(city_name: str, target_dates: list[str] | None = None):
    """Train ensemble, fetch live features, predict each target date.

    target_dates: ISO strings (YYYY-MM-DD). If None, defaults to tomorrow.
    Returns (forecasts, source_date) where forecasts has one Forecast per
    successfully-predicted target date and source_date is the live-features
    observation day. Targets without available features are skipped.
    """
    print(f"\n--- {city_name} ---")
    ensemble, feature_names, df_train = train_ensemble(city_name)

    print("  Fetching live features...")
    live = fetch_live_features(city_name)

    today = pd.Timestamp.now(tz=CITIES[city_name]["timezone"]) \
              .normalize().tz_localize(None)
    if today not in live.index:
        print(f"  ! today ({today.date()}) not in live response; "
              f"falling back to most recent: {live.index.max().date()}")
        today = live.index.max()

    if target_dates is None:
        target_dates = [(today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")]

    forecasts = predict_with(ensemble, feature_names, df_train,
                             live, city_name, target_dates)
    return forecasts, today


def predict_with(ensemble, feature_names: list[str], df_train: pd.DataFrame,
                 live_features: pd.DataFrame, city_name: str,
                 target_dates: list[str]):
    """Predict for each target date using features observed the day before.

    The ensemble is trained as "features on day D → tmax on D+1", so for
    each target date we look up the feature row at (target - 1 day) in
    `live_features`. Targets without a corresponding feature row are
    silently dropped.
    """
    rows: list[pd.DataFrame] = []
    valid_dates: list[str] = []
    for td in target_dates:
        feature_date = pd.Timestamp(td) - pd.Timedelta(days=1)
        if feature_date not in live_features.index:
            print(f"  ! features for {feature_date.date()} (target {td}) "
                  f"not in live response; skipping")
            continue
        rows.append(align_features(
            live_features.loc[[feature_date]], feature_names, df_train))
        valid_dates.append(td)
    if not rows:
        return []
    X = pd.concat(rows, ignore_index=False)
    return ensemble.forecast(
        X, cities=[city_name] * len(valid_dates), dates=valid_dates,
    )


def display(fc, source_date) -> None:
    """Pretty-print a Forecast: median, 80% interval, top bins, mini bar chart."""
    pdf = fc.pdf
    sorted_bins = sorted(pdf.items())

    # Median (50th percentile)
    median = None
    cum = 0.0
    for t, p in sorted_bins:
        cum += p
        if cum >= 0.5:
            median = t
            break

    # 80% prediction interval (10th to 90th percentile)
    lo80 = hi80 = None
    cum = 0.0
    for t, p in sorted_bins:
        cum += p
        if lo80 is None and cum >= 0.10:
            lo80 = t
        if cum >= 0.90:
            hi80 = t
            break

    top5 = sorted(pdf.items(), key=lambda kv: -kv[1])[:5]

    print(f"\n  Forecast for {fc.city} on {fc.date}")
    print(f"  Features observed on: {source_date.date()}")
    print(f"  Predicted high (median): {median}°F")
    print(f"  80% prediction interval: [{lo80}°F, {hi80}°F]")
    print(f"  σ (uncertainty): {fc.std_dev:.2f}°F")
    print(f"  Most likely outcomes:")
    for t, p in top5:
        bar = "█" * int(round(p * 60))
        print(f"    {t}°F  {p * 100:5.1f}%  {bar}")


def main():
    cities = sys.argv[1:] if len(sys.argv) > 1 else list(CITIES.keys())
    for city in cities:
        if city not in CITIES:
            print(f"\nUnknown city: {city!r}")
            print(f"Available: {list(CITIES.keys())}")
            continue
        try:
            forecasts, source_date = predict_for(city)
            for fc in forecasts:
                display(fc, source_date)
        except Exception as e:
            print(f"  ! Failed for {city}: {e}")


if __name__ == "__main__":
    main()
