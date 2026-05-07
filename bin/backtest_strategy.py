"""
Backtest the live HedgedPair strategy against historical snapshots.

For each (city, date) where we have snapshots from BOTH venues plus a forecast
PDF, replay the strategy as if the bot were running at the time the latest
snapshot was captured:

  1. Pick the latest pre-close snapshot per venue + the latest forecast PDF.
  2. Run `find_hedged_pairs` exactly as the live bot would.
  3. Resolve each accepted pair against the actual venue settlement when
     present; otherwise fall back to a *synthetic* settlement derived from
     the highest-probability bracket in each venue's final snapshot.
  4. Score per-pair realized $ = gross_payoff - cost_per_pair, multiplied
     by `size`, written to data/backtest_results.parquet.

Run from project root:
    .venv/bin/python -m bin.backtest_strategy
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from strategy.arbitrage import find_hedged_pairs  # noqa: E402
from strategy.parsers import (  # noqa: E402
  parse_kalshi_quotes, parse_polymarket_quotes,
  _parse_kalshi_range, _parse_polymarket_range,
)


DB_PATH = PROJECT_ROOT / "data" / "bot.db"
OUT_PATH = PROJECT_ROOT / "data" / "backtest_results.parquet"


# ── DB readers ───────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
  c = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
  c.row_factory = sqlite3.Row
  return c


def candidate_city_dates(c: sqlite3.Connection) -> list[tuple[str, str]]:
  """(city, date) pairs with snapshots from both venues + at least one
  forecast — backtestable iff all three are present."""
  rows = c.execute(
    """
    SELECT s.city, s.date
    FROM (SELECT DISTINCT city, date FROM snapshots) s
    WHERE EXISTS (SELECT 1 FROM snapshots WHERE city=s.city AND date=s.date AND venue='kalshi')
      AND EXISTS (SELECT 1 FROM snapshots WHERE city=s.city AND date=s.date AND venue='polymarket')
      AND EXISTS (SELECT 1 FROM forecasts WHERE city=s.city AND date=s.date)
    ORDER BY s.date, s.city
    """).fetchall()
  return [(r["city"], r["date"]) for r in rows]


def latest_brackets(c, city: str, date: str, venue: str) -> tuple[list[dict], str | None]:
  row = c.execute(
    """
    SELECT brackets_json, fetched_at_utc FROM snapshots
    WHERE city=? AND date=? AND venue=?
    ORDER BY fetched_at_utc DESC LIMIT 1
    """, (city, date, venue)).fetchone()
  if not row:
    return [], None
  return json.loads(row["brackets_json"]), row["fetched_at_utc"]


def all_snapshots(c, city: str, date: str, venue: str) -> list[tuple[str, list[dict]]]:
  """Every snapshot for (city, date, venue), oldest-first, parsed."""
  rows = c.execute(
    """
    SELECT fetched_at_utc, brackets_json FROM snapshots
    WHERE city=? AND date=? AND venue=?
    ORDER BY fetched_at_utc ASC
    """, (city, date, venue)).fetchall()
  return [(r["fetched_at_utc"], json.loads(r["brackets_json"])) for r in rows]


def all_forecasts(c, city: str, date: str) -> list[tuple[str, dict[int, float]]]:
  rows = c.execute(
    """
    SELECT fetched_at_utc, pdf_json FROM forecasts
    WHERE city=? AND date=?
    ORDER BY fetched_at_utc ASC
    """, (city, date)).fetchall()
  return [(r["fetched_at_utc"],
           {int(k): v for k, v in json.loads(r["pdf_json"]).items()})
          for r in rows]


def latest_forecast(c, city: str, date: str) -> tuple[dict[int, float], str, str | None]:
  row = c.execute(
    """
    SELECT pdf_json, source, fetched_at_utc FROM forecasts
    WHERE city=? AND date=?
    ORDER BY fetched_at_utc DESC LIMIT 1
    """, (city, date)).fetchone()
  if not row:
    return {}, "", None
  return ({int(k): v for k, v in json.loads(row["pdf_json"]).items()},
          row["source"], row["fetched_at_utc"])


def settlement(c, city: str, date: str) -> tuple[int | None, int | None]:
  row = c.execute(
    "SELECT kalshi_high_f, polymarket_high_f FROM settlements WHERE city=? AND date=?",
    (city, date)).fetchone()
  if not row:
    return None, None
  return row["kalshi_high_f"], row["polymarket_high_f"]


# ── Synthetic settlement ─────────────────────────────────────────────────

def _bracket_mid(b: dict) -> float | None:
  bid = b.get("best_yes_bid"); ask = b.get("best_yes_ask")
  if bid is not None and ask is not None: return (bid + ask) / 2.0
  return bid if bid is not None else ask


def synthetic_high_from_brackets(brackets: list[dict], venue: str) -> int | None:
  """Highest-probability bracket's middle integer degree. Used as a proxy
  for the close when no official settlement is available."""
  best_p = -1.0
  best_deg: int | None = None
  for b in brackets:
    if venue == "kalshi":
      degs = _parse_kalshi_range(b.get("subtitle", ""))
    else:
      degs = _parse_polymarket_range(b.get("question", ""))
    if not degs:
      continue
    mid = _bracket_mid(b)
    if mid is None:
      continue
    if mid > best_p:
      best_p = mid
      best_deg = degs[len(degs) // 2]
  return best_deg


# ── Per-pair scoring ─────────────────────────────────────────────────────

def realized_payoff(pair, kalshi_high: int, poly_high: int) -> float:
  """Per-contract realized $ for one HedgedPair given each venue's close."""
  # Reproduce the win-set logic from strategy.arbitrage:
  # YES side wins iff settled high lies in the bracket's degrees;
  # NO side wins iff it does NOT.
  k_in = kalshi_high in pair.kalshi_degrees
  p_in = poly_high in pair.poly_degrees
  k_won = k_in if pair.kalshi_side == "yes" else not k_in
  p_won = p_in if pair.poly_side == "yes" else not p_in
  gross = (1.0 if k_won else 0.0) + (1.0 if p_won else 0.0)
  return gross - pair.cost_per_pair


# ── Backtest driver ──────────────────────────────────────────────────────

def _pair_snapshots(k_snaps, p_snaps, max_skew_sec: int = 600):
  """Pair a kalshi snapshot with the polymarket snapshot closest in time,
  dropping pairs whose skew exceeds `max_skew_sec`. Yields (ts, k_brackets,
  p_brackets) tuples in chronological order."""
  from datetime import datetime
  def parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))
  p_times = [parse(t) for t, _ in p_snaps]
  for k_ts, k_brackets in k_snaps:
    if not p_snaps:
      return
    kt = parse(k_ts)
    j = min(range(len(p_snaps)), key=lambda i: abs((p_times[i] - kt).total_seconds()))
    if abs((p_times[j] - kt).total_seconds()) <= max_skew_sec:
      yield k_ts, k_brackets, p_snaps[j][1]


def _forecast_at(forecasts, ts: str) -> dict[int, float]:
  """Most recent forecast at or before `ts`. Forecasts list must be ascending."""
  best = None
  for f_ts, pdf in forecasts:
    if f_ts <= ts:
      best = pdf
    else:
      break
  return best or (forecasts[0][1] if forecasts else {})


def run() -> pd.DataFrame:
  """Replay the strategy at every paired snapshot timestamp and score each
  newly-accepted leg (deduped per city/date) against the close.

  This mirrors live behaviour: the bot polls every ~5 min, opens at most
  one position per (kalshi_market_id, kalshi_side) and (poly_market_id,
  poly_side) per (city, date), and holds to settlement.
  """
  rows: list[dict] = []
  with _connect() as c:
    targets = candidate_city_dates(c)
    print(f"Backtesting {len(targets)} (city, date) combinations.")
    for city, date in targets:
      k_snaps = all_snapshots(c, city, date, "kalshi")
      p_snaps = all_snapshots(c, city, date, "polymarket")
      forecasts = all_forecasts(c, city, date)
      if not k_snaps or not p_snaps or not forecasts:
        continue

      # Resolve the close — actual settlement if we have one, else
      # synthetic from the highest-probability bracket in the LAST snapshot.
      k_settle, p_settle = settlement(c, city, date)
      kalshi_high = k_settle if k_settle is not None \
        else synthetic_high_from_brackets(k_snaps[-1][1], "kalshi")
      poly_high = p_settle if p_settle is not None \
        else synthetic_high_from_brackets(p_snaps[-1][1], "polymarket")
      if kalshi_high is None or poly_high is None:
        continue
      if k_settle is not None and p_settle is not None:
        kind = "settled"
      elif k_settle is None and p_settle is None:
        kind = "synthetic"
      else:
        kind = "partial"

      # Walk paired snapshots, accept each leg at most once per (city, date)
      seen_k: set[tuple[str, str]] = set()
      seen_p: set[tuple[str, str]] = set()
      day_pairs = 0
      _, last_pdf = forecasts[-1]
      fc_source = "ensemble" if len(last_pdf) > 30 else "nws_normal"

      for ts, kb, pb in _pair_snapshots(k_snaps, p_snaps):
        kq = parse_kalshi_quotes(kb)
        pq = parse_polymarket_quotes(pb)
        pdf = _forecast_at(forecasts, ts) or last_pdf
        if not kq or not pq or not pdf:
          continue
        for h in find_hedged_pairs(kq, pq, pdf, city=city, date=date):
          k_key = (h.kalshi_market_id, h.kalshi_side)
          p_key = (h.poly_market_id, h.poly_side)
          if k_key in seen_k or p_key in seen_p:
            continue
          seen_k.add(k_key); seen_p.add(p_key)
          per_contract = realized_payoff(h, kalshi_high, poly_high)
          rows.append({
            "city": city, "date": date,
            "snapshot_ts": ts,
            "kalshi_high": kalshi_high, "poly_high": poly_high,
            "outcome_source": kind, "forecast_source": fc_source,
            "kalshi_market_id": h.kalshi_market_id,
            "kalshi_label": h.kalshi_label, "kalshi_side": h.kalshi_side,
            "kalshi_avg_fill": h.kalshi_avg_fill,
            "poly_market_id": h.poly_market_id,
            "poly_label": h.poly_label, "poly_side": h.poly_side,
            "poly_avg_fill": h.poly_avg_fill,
            "size": h.size, "cost_per_pair": h.cost_per_pair,
            "expected_payoff": h.expected_payoff, "q05_payoff": h.q05_payoff,
            "predicted_edge": h.expected_payoff - h.cost_per_pair,
            "realized_per_pair": per_contract,
            "realized_total": per_contract * h.size,
            "won": int(per_contract > 0),
          })
          day_pairs += 1
      print(f"  {city:14s} {date}  kalshi={kalshi_high}°F poly={poly_high}°F "
            f"({kind}) — {day_pairs} pairs across "
            f"{len(k_snaps)} snapshots")

  df = pd.DataFrame(rows)
  if df.empty:
    print("\nNo backtestable pairs produced.")
  else:
    print(f"\n{len(df)} pairs scored. "
          f"Total realized: ${df['realized_total'].sum():,.2f}  "
          f"win rate: {df['won'].mean() * 100:.1f}%")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")
  return df


if __name__ == "__main__":
  run()
