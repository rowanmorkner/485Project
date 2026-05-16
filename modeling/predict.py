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

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .dataset import (ARCHIVE_LAG_DAYS, CITIES, DAILY_VARS, HOURLY_VARS,
                      add_lag_features, city_slug, fetch_archive_features)
from .models import (EnsembleModel, HeteroscedasticRidgeModel,
                     NGBoostNormalModel, XGBMeanVarModel)

LIVE_URL = "https://api.open-meteo.com/v1/forecast"
DATASETS_DIR = Path("data/datasets")
LIVE_OFFSETS_DIR = Path("data/calibration")
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
        # past_days=2 gives lag-1 coverage for "today" even at the earliest
        # possible run time; forecast_days=3 covers tomorrow + the next two.
        "past_days": 2,
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

    features = features.sort_index()
    features = add_lag_features(features, n_lags=1)
    return features


def calibrate_live_features(city_name: str, lookback_days: int = 60) -> dict:
    """Compute per-feature mean offsets between archive (training) features
    and live forecast-API features over a recent overlap window.

    Saves a JSON file under ``data/calibration/`` keyed by feature name and
    returns the offsets dict. ``apply_live_offsets`` adds these offsets to
    live features at inference time so they better match the training
    distribution (which comes from ERA5 reanalysis).
    """
    cfg = CITIES[city_name]
    end_date = date.today() - timedelta(days=ARCHIVE_LAG_DAYS)
    start_date = end_date - timedelta(days=lookback_days)

    print(f"\n--- Calibrating {city_name} "
          f"[{start_date.isoformat()} → {end_date.isoformat()}] ---")

    # Live features for the same window via the forecast API.
    params = {
        "latitude": cfg["lat"],
        "longitude": cfg["lon"],
        "daily": ",".join(DAILY_VARS),
        "hourly": ",".join(HOURLY_VARS),
        "timezone": cfg["timezone"],
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "windspeed_unit": "mph",
        "past_days": lookback_days,
        "forecast_days": 1,
    }
    resp = requests.get(LIVE_URL, params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    daily = pd.DataFrame(data["daily"])
    daily["time"] = pd.to_datetime(daily["time"])
    daily = daily.set_index("time")

    hourly = pd.DataFrame(data["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"])
    hourly_daily = hourly.set_index("time").resample("D").mean()

    live = daily.join(hourly_daily)
    live.index.name = "date"
    doy = live.index.dayofyear
    live["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    live["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    live = live.sort_index()
    live = add_lag_features(live, n_lags=1)

    # Archive (ERA5) features over the same window.
    arch = fetch_archive_features(
        cfg["lat"], cfg["lon"],
        start_date.isoformat(), end_date.isoformat(),
        cfg["timezone"],
    )
    arch = add_lag_features(arch, n_lags=1)

    # Per-feature mean offset (archive minus live) on the overlapping index.
    common_idx = arch.index.intersection(live.index)
    offsets: dict[str, float] = {}
    for col in arch.columns:
        if col in ("doy_sin", "doy_cos"):
            continue
        if col not in live.columns:
            continue
        diffs = (arch.loc[common_idx, col] - live.loc[common_idx, col]).dropna()
        if len(diffs) >= 10:
            offsets[col] = float(diffs.mean())

    LIVE_OFFSETS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LIVE_OFFSETS_DIR / f"{city_slug(city_name)}_live_offsets.json"
    with open(out_path, "w") as f:
        json.dump(offsets, f, indent=2, sort_keys=True)
    print(f"  Saved {len(offsets)} offsets → {out_path}")

    return offsets


def apply_live_offsets(features: pd.DataFrame, city_name: str) -> pd.DataFrame:
    """Add saved per-feature offsets to live features (no-op if missing)."""
    path = LIVE_OFFSETS_DIR / f"{city_slug(city_name)}_live_offsets.json"
    if not path.exists():
        print(f"  No live-offset calibration found for {city_name}; "
              f"using raw live features.")
        return features

    with open(path) as f:
        offsets = json.load(f)

    corrected = features.copy()
    applied = 0
    for col, off in offsets.items():
        if col in corrected.columns:
            corrected[col] = corrected[col] + off
            applied += 1
    print(f"  Applied {applied} live-feature offsets.")
    return corrected


def train_ensemble(city_name: str):
    """Train the ensemble. Components fit on sub-train (first 80%), σ-calibrate
    and learn ensemble weights on a val tail (last 20%), then refit on full
    data — calibration scalars and weights carry over to the deployed model.
    """
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

    # Carve val tail
    val_frac = 0.2
    n_val = max(int(len(X) * val_frac), 30)
    X_sub, y_sub = X.iloc[:-n_val], y.iloc[:-n_val]
    X_val, y_val = X.iloc[-n_val:], y.iloc[-n_val:]

    # Sub-train fits + calibration on val
    ridge_sub = HeteroscedasticRidgeModel().fit(X_sub, y_sub); ridge_sub.calibrate(X_val, y_val)
    xgb_sub = XGBMeanVarModel().fit(X_sub, y_sub); xgb_sub.calibrate(X_val, y_val)
    ngb_sub = NGBoostNormalModel().fit(X_sub, y_sub); ngb_sub.calibrate(X_val, y_val)

    ens_sub = EnsembleModel([ridge_sub, xgb_sub, ngb_sub])
    weights = ens_sub.set_weights_from_validation(X_val, y_val, score="crps")
    print(f"  Ensemble weights (CRPS-derived): "
          f"Ridge={weights[0]:.2f}  XGB={weights[1]:.2f}  NGB={weights[2]:.2f}")

    # Refit on full data; transfer calibration scalars
    ridge = HeteroscedasticRidgeModel().fit(X, y)
    ridge.mean_offset_, ridge.sigma_scale_ = ridge_sub.mean_offset_, ridge_sub.sigma_scale_
    xgb = XGBMeanVarModel().fit(X, y)
    xgb.mean_offset_, xgb.sigma_scale_ = xgb_sub.mean_offset_, xgb_sub.sigma_scale_
    ngb = NGBoostNormalModel().fit(X, y)
    ngb.mean_offset_, ngb.sigma_scale_ = ngb_sub.mean_offset_, ngb_sub.sigma_scale_

    ensemble = EnsembleModel([ridge, xgb, ngb], weights=weights)
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
    live = apply_live_offsets(live, city_name)

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
    args = sys.argv[1:]

    # --calibrate-live: compute archive-vs-live offsets and exit.
    if "--calibrate-live" in args:
        args = [a for a in args if a != "--calibrate-live"]
        cities = args if args else list(CITIES.keys())
        for city in cities:
            if city not in CITIES:
                print(f"\nUnknown city: {city!r}")
                print(f"Available: {list(CITIES.keys())}")
                continue
            try:
                calibrate_live_features(city)
            except Exception as e:
                print(f"  ! Calibration failed for {city}: {e}")
        return

    cities = args if args else list(CITIES.keys())
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
