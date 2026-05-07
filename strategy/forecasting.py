"""
Forecast PDF dispatcher: ensemble first, NWS fallback.

The trading strategy needs `dict[int, float]` PDFs over daily-high temperature.
Two sources are available:

  1. The teammate's ensemble (modeling/) — Ridge + XGBoost + NGBoost mixture
     trained on ERA5 reanalysis features. Live inference uses Open-Meteo
     forecast features. Better calibrated, more accurate.
  2. The NWS daily-high point forecast widened into a Gaussian PDF
     (strategy/distributions.py). Always available, less accurate.

Training the ensemble is expensive (~1 min/city), inference is cheap. We
cache one trained ensemble per (city, date_trained) in memory and let
bin/run_loop's daily training pass refresh it once a day. If anything in
the ensemble path fails (missing parquet, API error, etc.) we fall back
to the NWS Gaussian — the bot keeps trading.
"""
from __future__ import annotations

import logging
import threading
from datetime import date, datetime, timezone

from strategy.distributions import forecast_to_distribution

logger = logging.getLogger(__name__)


# ── Per-city ensemble cache ──────────────────────────────────────────────

# `(ensemble, feature_names, df_train, trained_on_date)` per city.
# Lock guards concurrent training under the parallel poll loop.
_cache: dict[str, tuple] = {}
_cache_lock = threading.Lock()


def _today_utc() -> date:
  return datetime.now(timezone.utc).date()


def _train_ensemble_cached(city: str):
  """Train (or reuse) one ensemble per city. Idempotent within a UTC day."""
  with _cache_lock:
    cached = _cache.get(city)
    if cached and cached[3] == _today_utc():
      return cached
  # Train outside the lock — releases other cities to proceed in parallel
  from modeling.predict import train_ensemble
  ensemble, feature_names, df_train = train_ensemble(city)
  entry = (ensemble, feature_names, df_train, _today_utc())
  with _cache_lock:
    _cache[city] = entry
  return entry


def warm_cache(cities: list[str]) -> None:
  """Train every city up front. Called from bin/run_loop after each daily
  training pass so the next polling cycle starts with fresh models."""
  for c in cities:
    try:
      _train_ensemble_cached(c)
    except Exception as exc:
      logger.warning("[%s] ensemble train failed: %s", c, exc)


# ── Public entry point ───────────────────────────────────────────────────

def get_forecast_pdf(
  city: str, date_iso: str, nws_high: float | None,
) -> tuple[dict[int, float], str]:
  """Return (pdf, source) for the given (city, date).

  Tries the ensemble first; falls back to a Gaussian on `nws_high`. If
  neither works, returns ({}, "none").

  `source` is one of: "ensemble", "nws_normal", "none". The caller
  persists this so we can attribute predictions to their model later.
  """
  pdf = _ensemble_pdf(city, date_iso)
  if pdf:
    return pdf, "ensemble"
  if nws_high is not None:
    return forecast_to_distribution(nws_high), "nws_normal"
  return {}, "none"


def _ensemble_pdf(city: str, date_iso: str) -> dict[int, float]:
  """Try the ensemble path. Returns {} on any failure."""
  try:
    ensemble, feature_names, df_train, _ = _train_ensemble_cached(city)
    from modeling.predict import fetch_live_features, predict_with
    live = fetch_live_features(city)
    forecasts = predict_with(
      ensemble, feature_names, df_train, live, city, [date_iso]
    )
    if forecasts:
      return dict(forecasts[0].pdf)
  except Exception as exc:
    logger.warning("[%s %s] ensemble forecast failed: %s", city, date_iso, exc)
  return {}
