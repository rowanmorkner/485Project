"""
Per-venue reading-error model for fair-value adjustment.

The empirical insight from backtests/cli_vs_asos_divergence.json is that
Polymarket (Wunderground) and Kalshi (NWS CLI) systematically read the
same airport's daily high differently — typically Kalshi 1°F warmer,
with a per-city tail.

We model this as: Polymarket_reading = Kalshi_reading − Δ, where Δ is
distributed per the city's empirical histogram. Since the forecast PDF
(Derek's ensemble or our normal-from-NWS placeholder) is built from
NWS data and aligns with Kalshi's source, we assume:

    P(Kalshi reads d | forecast) = forecast_pdf[d]
    P(Polymarket reads d | forecast) = Σ_k forecast_pdf[k] · P(Δ = k − d)

i.e. Polymarket's reading distribution is the forecast PDF convolved
with the empirical Δ distribution. Bracket fair value is then summed
over the bracket's degrees against the appropriate per-venue distribution.

Effects this captures vs a flat haircut:
  - Directional bias (CLI runs warmer → Polymarket brackets shift cooler)
  - Tail shape (SF's wider tail spreads more probability outside-bracket)
  - Boundary distance (brackets at the forecast peak edge are penalized
    proportionally more than brackets clearly inside the peak)
"""

import json
from pathlib import Path

# Two-tier lookup: prefer the live data refreshed by bin/refresh_risk_model.py,
# fall back to the static backtest output. The fallback gives sane defaults
# before the bot has accumulated enough settlements to refresh on its own.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIVE_RISK_PATH = _PROJECT_ROOT / "data" / "risk_histograms.json"
FALLBACK_RISK_PATH = _PROJECT_ROOT / "backtests" / "cli_vs_asos_divergence.json"

# Manual cache + mtime tracker so the running bot picks up new params after
# the daily cron writes a fresh file — without restart and without holding
# stale data via lru_cache.
_cache: dict[str, dict] = {}
_cache_path: Path | None = None
_cache_mtime: float = 0.0


def _resolve_risk_path() -> Path:
  """Live file if present, otherwise the static backtest fallback."""
  return LIVE_RISK_PATH if LIVE_RISK_PATH.exists() else FALLBACK_RISK_PATH


def _load_risk_data() -> dict[str, dict]:
  """
  Read the per-city Δ histogram JSON, refreshing from disk when the source
  file's mtime changes (or when the resolved path itself flips between
  live and fallback).
  """
  global _cache, _cache_path, _cache_mtime
  path = _resolve_risk_path()
  if not path.exists():
    return {}
  mtime = path.stat().st_mtime
  if path == _cache_path and mtime == _cache_mtime and _cache:
    return _cache
  _cache = json.loads(path.read_text())
  _cache_path = path
  _cache_mtime = mtime
  return _cache


def delta_pdf(city: str) -> dict[int, float]:
  """
  Empirical PDF over Δ = Kalshi_reading − Polymarket_reading for one city,
  normalized to sum to 1.0. Returns {} if no data exists for that city.

  Δ is integer °F; positive Δ means Kalshi reads warmer than Polymarket.
  """
  data = _load_risk_data().get(city)
  if not data or not data.get("histogram"):
    return {}
  raw = {int(k): float(v) for k, v in data["histogram"].items()}
  total = sum(raw.values())
  if total <= 0:
    return {}
  return {k: v / total for k, v in raw.items()}


def venue_pdf(forecast_pdf: dict[int, float], city: str, venue: str) -> dict[int, float]:
  """
  Convert the forecast PDF (assumed to match Kalshi's reading distribution)
  into the venue's own reading distribution.

    Kalshi:     pass-through (P(K = d) = forecast_pdf[d])
    Polymarket: convolve with the empirical Δ:
                P(P = d) = Σ_k forecast_pdf[k] · P(Δ = k − d)

  If no per-city Δ data exists, falls back to the forecast PDF unchanged
  for both venues. The caller's behavior is then identical to pre-risk-model.
  """
  if venue == "kalshi":
    return forecast_pdf

  d_pdf = delta_pdf(city)
  if not d_pdf:
    return forecast_pdf

  # Output keys: every degree d such that some k in forecast_pdf and some
  # delta in d_pdf give k - delta = d.
  out: dict[int, float] = {}
  for k, p_k in forecast_pdf.items():
    for delta, p_delta in d_pdf.items():
      d = k - delta
      out[d] = out.get(d, 0.0) + p_k * p_delta
  return out


def bracket_fair_value(
  bracket_degrees: list[int],
  forecast_pdf: dict[int, float],
  city: str,
  venue: str,
) -> float:
  """
  Compute a venue-adjusted fair value for one bracket.

  This is the probability that the venue will read a value falling inside
  bracket_degrees, given the forecast and the venue's empirical reading
  distribution relative to Kalshi.
  """
  vpdf = venue_pdf(forecast_pdf, city, venue)
  return sum(vpdf.get(d, 0.0) for d in bracket_degrees)
