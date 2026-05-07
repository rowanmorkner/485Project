"""
Venue divergence + joint K-P distribution under the empirical Δ histogram.

Round 2 finding: only 25.5% of historical (city, date) settlements report
identical Kalshi and Polymarket highs. Cross-venue strategies must propagate
the empirical Δ = K_high − P_high distribution into their scoring.

Reads ONLY from data/bot.db settlements. No JSON cache layer; the joint
distribution is recomputed per call to find_hedged_pairs (cheap).
"""

import sqlite3
from pathlib import Path
from typing import Callable

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "bot.db"


def _conn() -> sqlite3.Connection:
  c = sqlite3.connect(str(DB_PATH))
  c.row_factory = sqlite3.Row
  return c


def venue_divergence_histogram(city: str | None = None) -> dict[int, float]:
  """{Δ: prob} where Δ = K_high - P_high. None pools all cities.

  Falls back to the pooled distribution if a per-city query is empty.
  """
  q = """SELECT kalshi_high_f - polymarket_high_f AS delta
           FROM settlements
          WHERE kalshi_high_f IS NOT NULL
            AND polymarket_high_f IS NOT NULL"""
  args: tuple = ()
  if city:
    q += " AND city = ?"
    args = (city,)
  with _conn() as c:
    rows = c.execute(q, args).fetchall()
  hist: dict[int, float] = {}
  for r in rows:
    d = int(r["delta"])
    hist[d] = hist.get(d, 0.0) + 1.0
  if not hist and city is not None:
    return venue_divergence_histogram(city=None)
  if not hist:
    return {}
  total = sum(hist.values())
  return {k: v / total for k, v in hist.items()}


def joint_kp_distribution(forecast_pdf: dict[int, float],
                          city: str | None = None) -> dict[tuple[int, int], float]:
  """P(K=k, P=p) ≈ forecast_pdf[k] * P(Δ=k-p).

  Treats the forecast PDF as P(K=k) (NWS source aligns with Kalshi reads)
  and assumes Δ independent of K.
  """
  delta_hist = venue_divergence_histogram(city=city)
  if not delta_hist:
    # Fallback: assume venues agree perfectly
    return {(k, k): p for k, p in forecast_pdf.items()}
  joint: dict[tuple[int, int], float] = {}
  for k, p_k in forecast_pdf.items():
    for d, p_d in delta_hist.items():
      key = (k, k - d)
      joint[key] = joint.get(key, 0.0) + p_k * p_d
  return joint


def expected_payoff(payoff_fn: Callable[[int, int], float],
                    joint: dict[tuple[int, int], float]) -> float:
  """E[payoff(k, p)] over the joint K,P distribution."""
  return sum(w * payoff_fn(k, p) for (k, p), w in joint.items())


def quantile_payoff(payoff_fn: Callable[[int, int], float],
                    joint: dict[tuple[int, int], float],
                    q: float = 0.05) -> float:
  """Lower q-quantile of payoff(k, p) under the joint distribution."""
  outcomes = sorted(
    ((payoff_fn(k, p), w) for (k, p), w in joint.items()),
    key=lambda t: t[0],
  )
  cum = 0.0
  for payoff, w in outcomes:
    cum += w
    if cum >= q:
      return payoff
  return outcomes[-1][0] if outcomes else 0.0
