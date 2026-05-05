"""
Backtest #1: empirical Kalshi-vs-Polymarket settlement divergence.

For every past resolved (city, date) pair where both venues had a market,
compare which bracket settled YES on each side. Classify as:

  AGREE     — winning ranges overlap (share at least one integer degree)
  ADJACENT  — no overlap, but ≤ 1°F apart (likely rounding/QC noise)
  DISAGREE  — no overlap and ≥ 2°F apart (real basis risk)

Key correctness fix vs the original draft: Kalshi runs multiple bracket-
structure variants per day (some 1°F wide, some wider with tail-only bins).
We dedup by date and prefer the NARROWEST winning bracket — that's the
tightest estimate of Kalshi's reading. Polymarket has one event per
(city, date) so no dedup needed.

Cache: poly events and per-event Kalshi market lookups go through the
existing _cache/ dir written by the prior agent. Reuse what's there.
"""

import json
import logging
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from clients.kalshi import KalshiClient
from clients.polymarket import PolymarketClient
from strategy.parsers import _parse_kalshi_range, _parse_polymarket_range


# ── Setup ─────────────────────────────────────────────────────────────────

CACHE_DIR = Path(__file__).resolve().parent / "_cache"
CACHE_DIR.mkdir(exist_ok=True)
POLY_CACHE = CACHE_DIR / "poly_closed_weather_events.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


CITIES = [
  {"name": "Miami",         "kalshi_series": "KXHIGHMIA",  "poly_name": "Miami"},
  {"name": "LA",            "kalshi_series": "KXHIGHLAX",  "poly_name": "Los Angeles"},
  {"name": "Austin",        "kalshi_series": "KXHIGHAUS",  "poly_name": "Austin"},
  {"name": "San Francisco", "kalshi_series": "KXHIGHTSFO", "poly_name": "San Francisco"},
  {"name": "Seattle",       "kalshi_series": "KXHIGHTSEA", "poly_name": "Seattle"},
]


# ── Kalshi side: paginate every settled event for a series ────────────────

def fetch_all_settled_kalshi_events(k: KalshiClient, series: str) -> list[dict]:
  """Paginate through every settled event for one series."""
  out: list[dict] = []
  cursor = None
  while True:
    params = {"series_ticker": series, "status": "settled", "limit": 200}
    if cursor:
      params["cursor"] = cursor
    resp = k._request("GET", "/events", params=params)
    events = resp.get("events", []) or []
    out.extend(events)
    cursor = resp.get("cursor")
    if not cursor or not events:
      break
  return out


def kalshi_event_date(event_ticker: str) -> str | None:
  """Parse YYMMMDD suffix from a Kalshi event ticker, e.g. KXHIGHMIA-26APR27."""
  m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})$", event_ticker)
  if not m:
    return None
  try:
    return datetime.strptime(
      f"{m.group(3)}{m.group(2)}{m.group(1)}", "%d%b%y"
    ).strftime("%Y-%m-%d")
  except ValueError:
    return None


def kalshi_winning_market(k: KalshiClient, event_ticker: str) -> dict | None:
  """Cached: fetch markets for an event and return the YES-resolved one."""
  cache_path = CACHE_DIR / f"k_event_{event_ticker}.json"
  if cache_path.exists():
    markets = json.loads(cache_path.read_text())
  else:
    resp = k.get_markets(event_ticker=event_ticker, limit=200)
    markets = resp.get("markets", []) or []
    cache_path.write_text(json.dumps(markets))
    time.sleep(0.05)  # gentle pacing
  for m in markets:
    if (m.get("result") or "").lower() == "yes":
      return m
  return None


def kalshi_market_degrees(market: dict) -> tuple[str, list[int]]:
  """Return (label, degrees) for a Kalshi YES bracket."""
  label = market.get("subtitle") or market.get("yes_sub_title") or ""
  return label, _parse_kalshi_range(label)


# ── Polymarket side ───────────────────────────────────────────────────────

def fetch_all_closed_weather_events(p: PolymarketClient) -> list[dict]:
  """Cached pull of every closed Polymarket weather event."""
  if POLY_CACHE.exists():
    return json.loads(POLY_CACHE.read_text())
  events: list[dict] = []
  offset = 0
  page = 500
  while True:
    batch = p.search_events(tag_slug="weather", closed=True, limit=page, offset=offset)
    if not batch:
      break
    events.extend(batch)
    if len(batch) < page:
      break
    offset += page
    if offset > 20000:
      break
  POLY_CACHE.write_text(json.dumps(events))
  return events


def poly_event_date(event: dict) -> str | None:
  """
  Get the LOCAL measurement date from a Polymarket event title.
  Polymarket titles are reliable: "Highest temperature in Austin on March 28?".
  Fallback: endDate's date in UTC (less reliable since endDate is settle time).
  """
  title = event.get("title") or ""
  m = re.search(
    r"on\s+(January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+(\d{1,2})(?:,\s*(\d{4}))?",
    title,
  )
  if m:
    month_name, day, year = m.group(1), int(m.group(2)), m.group(3)
    if year:
      year_int = int(year)
    else:
      end = event.get("endDate") or ""
      try:
        year_int = datetime.fromisoformat(end.replace("Z", "+00:00")).year
      except (ValueError, TypeError):
        return None
    try:
      return datetime.strptime(
        f"{day} {month_name} {year_int}", "%d %B %Y"
      ).strftime("%Y-%m-%d")
    except ValueError:
      return None
  return None


def poly_winner(event: dict) -> dict | None:
  """Find the YES-resolved bracket inside a closed Polymarket event."""
  for m in event.get("markets", []) or []:
    raw = m.get("outcomePrices")
    try:
      prices = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
      continue
    if not prices:
      continue
    try:
      yes_price = float(prices[0])
    except (TypeError, ValueError, IndexError):
      continue
    if yes_price >= 0.99:
      return m
  return None


# ── Classification ────────────────────────────────────────────────────────

# Open-ended bracket detection — these labels mean "X or anything higher/lower",
# i.e. semi-infinite ranges. The PMF parsers in strategy/parsers.py expand them
# to a fixed 5°F window for probability-spreading purposes, but for SETTLEMENT
# classification we need the true open-ended interval.

ABOVE_RE = re.compile(r"(\d+)\s*°?\s*F?\s*or\s+(?:above|higher|more)", re.IGNORECASE)
BELOW_RE = re.compile(r"(\d+)\s*°?\s*F?\s*or\s+(?:below|lower|less)", re.IGNORECASE)


def label_to_interval(label: str, parsed_degrees: list[int]) -> tuple[int, int]:
  """
  Return (lo, hi) for a bracket. Open-ended brackets get -inf/+inf bounds
  encoded as ±9999 (sentinel — temperatures never reach that).
  """
  am = ABOVE_RE.search(label)
  if am:
    return (int(am.group(1)), 9999)
  bm = BELOW_RE.search(label)
  if bm:
    return (-9999, int(bm.group(1)))
  if parsed_degrees:
    return (min(parsed_degrees), max(parsed_degrees))
  return (0, 0)


def classify(
  k_degrees: list[int], k_label: str,
  p_degrees: list[int], p_label: str,
) -> str:
  if not k_degrees or not p_degrees:
    return "UNPARSED"
  k_lo, k_hi = label_to_interval(k_label, k_degrees)
  p_lo, p_hi = label_to_interval(p_label, p_degrees)
  # Intervals overlap iff k_lo <= p_hi AND p_lo <= k_hi
  if k_lo <= p_hi and p_lo <= k_hi:
    return "AGREE"
  # No overlap; gap is the distance between the closer endpoints
  if p_lo > k_hi:
    gap = p_lo - k_hi - 1
  else:
    gap = k_lo - p_hi - 1
  # ADJACENT: gap of 0°F (touching) counts as 1°F apart at most
  return "ADJACENT" if gap <= 0 else "DISAGREE"


# ── Main ──────────────────────────────────────────────────────────────────

def build_kalshi_winners(k: KalshiClient, series: str) -> dict[str, dict]:
  """
  date → {"label", "degrees", "tickers": [list of all winning market tickers]}

  Dedup strategy: many Kalshi event variants resolve on the same date with
  different bracket widths. Keep the NARROWEST winning bracket per date (most
  informative). Track all variant tickers for traceability.
  """
  events = fetch_all_settled_kalshi_events(k, series)
  log.info("  %d settled Kalshi events for %s", len(events), series)

  by_date: dict[str, list[dict]] = defaultdict(list)
  for ev in events:
    d = kalshi_event_date(ev.get("event_ticker", ""))
    if not d:
      continue
    try:
      w = kalshi_winning_market(k, ev["event_ticker"])
    except Exception as exc:
      log.warning("  Kalshi market fetch failed for %s: %s", ev["event_ticker"], exc)
      continue
    if not w:
      continue
    label, degrees = kalshi_market_degrees(w)
    if not degrees:
      continue
    by_date[d].append({
      "ticker": w.get("ticker"),
      "label": label,
      "degrees": degrees,
      "width": len(degrees),
    })

  # Pick narrowest per date
  winners: dict[str, dict] = {}
  for d, candidates in by_date.items():
    candidates.sort(key=lambda c: c["width"])
    best = candidates[0]
    winners[d] = {
      "label": best["label"],
      "degrees": best["degrees"],
      "ticker": best["ticker"],
      "all_variants": [c["ticker"] for c in candidates],
    }
  return winners


def build_poly_winners(all_events: list[dict], poly_name: str) -> dict[str, dict]:
  """date → {"question", "label", "degrees"} for one city."""
  city_lower = poly_name.lower()
  winners: dict[str, dict] = {}
  for e in all_events:
    title = (e.get("title") or "").lower()
    if "highest temperature" not in title or city_lower not in title:
      continue
    d = poly_event_date(e)
    if not d:
      continue
    w = poly_winner(e)
    if not w:
      continue
    label = w.get("groupItemTitle") or w.get("question") or ""
    degrees = _parse_polymarket_range(label)
    if not degrees:
      continue
    winners[d] = {
      "question": w.get("question"),
      "label": label,
      "degrees": degrees,
    }
  return winners


def main():
  k = KalshiClient()
  p = PolymarketClient()

  log.info("Fetching cached Polymarket closed-weather events...")
  poly_all = fetch_all_closed_weather_events(p)
  log.info("Got %d closed Polymarket weather events", len(poly_all))

  pairs_by_city: dict[str, dict] = {}
  summary: dict[str, dict] = {}

  for city in CITIES:
    cname = city["name"]
    log.info("=== %s ===", cname)

    kw = build_kalshi_winners(k, city["kalshi_series"])
    pw = build_poly_winners(poly_all, city["poly_name"])
    log.info("  %d Kalshi winners, %d Polymarket winners", len(kw), len(pw))

    common = sorted(set(kw) & set(pw))
    log.info("  %d overlapping dates", len(common))

    pairs: dict[str, dict] = {}
    counts = {"AGREE": 0, "ADJACENT": 0, "DISAGREE": 0, "UNPARSED": 0}
    for d in common:
      cls = classify(
        kw[d]["degrees"], kw[d]["label"],
        pw[d]["degrees"], pw[d]["label"],
      )
      counts[cls] += 1
      pairs[d] = {
        "kalshi_ticker": kw[d]["ticker"],
        "kalshi_bracket": kw[d]["label"],
        "kalshi_degrees": kw[d]["degrees"],
        "kalshi_variants": kw[d]["all_variants"],
        "poly_question": pw[d]["question"],
        "poly_bracket": pw[d]["label"],
        "poly_degrees": pw[d]["degrees"],
        "classification": cls,
      }
    pairs_by_city[cname] = pairs

    n = sum(counts.values())
    summary[cname] = {
      "n_pairs": n,
      "agree": counts["AGREE"],
      "adjacent": counts["ADJACENT"],
      "disagree": counts["DISAGREE"],
      "agree_pct": (counts["AGREE"] / n) if n else None,
      "adjacent_pct": (counts["ADJACENT"] / n) if n else None,
      "disagree_pct": (counts["DISAGREE"] / n) if n else None,
    }
    log.info("  → %s", summary[cname])

  # Write outputs
  out_dir = Path(__file__).resolve().parent
  (out_dir / "settlement_pairs.json").write_text(
    json.dumps(pairs_by_city, indent=2, ensure_ascii=False)
  )
  (out_dir / "settlement_divergence.json").write_text(
    json.dumps(summary, indent=2)
  )

  # Markdown report
  md = ["# Kalshi vs Polymarket — Settlement Divergence Backtest\n"]
  md.append("For each (city, date) where both venues had a resolved market, "
            "we compare which bracket settled YES on each side. "
            "Kalshi reads NWS CLI products; Polymarket reads Wunderground. "
            "Even at the same airport these can differ.\n")
  md.append("**Classification:**\n")
  md.append("- **AGREE**: winning ranges overlap (share ≥ 1°F)")
  md.append("- **ADJACENT**: no overlap, ≤ 0°F gap (touching — typical 1°F rounding)")
  md.append("- **DISAGREE**: no overlap, ≥ 1°F gap (real basis risk)\n")
  md.append("## Per-city results\n")
  md.append("| City | N pairs | Agree | Adjacent | Disagree | Agree % | Adjacent % | Disagree % |")
  md.append("|------|--------:|------:|---------:|---------:|--------:|-----------:|-----------:|")
  for cname, s in summary.items():
    if s["n_pairs"] == 0:
      md.append(f"| {cname} | 0 | – | – | – | – | – | – |")
      continue
    md.append(
      f"| {cname} | {s['n_pairs']} | {s['agree']} | {s['adjacent']} | {s['disagree']} | "
      f"{s['agree_pct']:.1%} | {s['adjacent_pct']:.1%} | {s['disagree_pct']:.1%} |"
    )
  md.append("\n## Caveats\n")
  md.append("- Sample sizes are small — Kalshi per-city daily temp markets started in 2024-ish, "
            "Polymarket's are even newer.")
  md.append("- Kalshi runs multiple bracket-structure variants per date; we keep the narrowest "
            "winning bracket as the most informative reading. All variant tickers are saved in "
            "settlement_pairs.json.")
  md.append("- AGREE is the only signal that the two venues read the same temperature. ADJACENT "
            "is most likely 1°F rounding / observation timing. DISAGREE is real basis risk and "
            "should drive the EV haircut in find_value_trades.")
  (out_dir / "settlement_report.md").write_text("\n".join(md) + "\n")
  log.info("Wrote settlement_pairs.json, settlement_divergence.json, settlement_report.md")


if __name__ == "__main__":
  main()
