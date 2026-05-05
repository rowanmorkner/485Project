"""
Backtest #2: NWS CLI (Kalshi's source) vs ASOS hourly max (free Wunderground proxy).

For each of our 5 airports, pull the past year of:
  - NWS CLI products via Iowa Mesonet (parse "MAXIMUM" line for daily high °F)
  - ASOS hourly tmpf via Iowa Mesonet bulk download (compute per-local-day max)

Then compute Δ = CLI_high - ASOS_max distribution and tail probabilities per
city. The output feeds into the EV haircut / risk-adjusted sizing in
strategy/arbitrage.py.

Caveats baked in:
  * CLI products are issued multiple times/day; the AM product (8-9am LST)
    reports YESTERDAY's data — that's what we parse. Afternoon products
    cover today-so-far and are skipped.
  * ASOS reports at :53 hourly (METAR convention); other timestamps are "M".
  * Daily max must be computed in LOCAL time, not UTC, since "the daily high"
    is by local calendar day.
  * If a day has fewer than 18 valid hourly observations, drop it (gaps).
"""

import csv
import json
import logging
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(__file__).resolve().parent / "_cache"
CACHE_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


STATIONS = [
  {"city": "Miami",         "icao": "KMIA", "asos": "MIA", "cli": "CLIMIA", "tz": "America/New_York"},
  {"city": "LA",            "icao": "KLAX", "asos": "LAX", "cli": "CLILAX", "tz": "America/Los_Angeles"},
  {"city": "Austin",        "icao": "KAUS", "asos": "AUS", "cli": "CLIAUS", "tz": "America/Chicago"},
  {"city": "San Francisco", "icao": "KSFO", "asos": "SFO", "cli": "CLISFO", "tz": "America/Los_Angeles"},
  {"city": "Seattle",       "icao": "KSEA", "asos": "SEA", "cli": "CLISEA", "tz": "America/Los_Angeles"},
]

# Mesonet base URLs
MESONET_API = "https://mesonet.agron.iastate.edu/api/1"
ASOS_BULK = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

# Date range: last ~12 months ending today
END_DATE = datetime.now(timezone.utc).date()
START_DATE = END_DATE - timedelta(days=365)


# ── CLI side ──────────────────────────────────────────────────────────────

def list_cli_products(pil: str, date: datetime.date) -> list[dict]:
  """List CLI products for one (pil, date). Cached per (pil, date)."""
  cache = CACHE_DIR / f"list_{pil}_{date.isoformat()}.json"
  if cache.exists():
    return json.loads(cache.read_text())
  url = f"{MESONET_API}/nws/afos/list.json?pil={pil}&date={date.isoformat()}"
  r = requests.get(url, timeout=30)
  r.raise_for_status()
  data = r.json().get("data", [])
  cache.write_text(json.dumps(data))
  return data


def fetch_cli_text(product_id: str) -> str:
  """Cached fetch of one CLI product's text."""
  cache = CACHE_DIR / f"text_{product_id}.txt"
  if cache.exists():
    return cache.read_text()
  url = f"{MESONET_API}/nwstext/{product_id}"
  r = requests.get(url, timeout=30)
  r.raise_for_status()
  text = r.text
  cache.write_text(text)
  return text


# MAXIMUM may carry a suffix flag in CLI products: R = record tying,
# E = estimated, etc. (single uppercase letter, no whitespace before it).
# Followed by whitespace and the time-of-observation column.
CLI_MAX_RE = re.compile(
  r"^\s*MAXIMUM\s+(\d{1,3})[A-Z]?\s",
  re.MULTILINE,
)
CLI_FOR_DATE_RE = re.compile(
  r"CLIMATE SUMMARY FOR ([A-Z]+)\s+(\d{1,2})\s+(\d{4})",
  re.IGNORECASE,
)


def parse_cli_high(text: str) -> tuple[str | None, int | None]:
  """
  From a CLI text product, extract (date_iso_summarized, high_F).
  The "CLIMATE SUMMARY FOR <MONTH> <DAY> <YEAR>" line names the date this
  product covers; "MAXIMUM <N>" gives the high.
  """
  date_m = CLI_FOR_DATE_RE.search(text)
  date_iso = None
  if date_m:
    try:
      dt = datetime.strptime(
        f"{date_m.group(2)} {date_m.group(1).title()} {date_m.group(3)}",
        "%d %B %Y",
      )
      date_iso = dt.strftime("%Y-%m-%d")
    except ValueError:
      pass

  max_m = CLI_MAX_RE.search(text)
  high = int(max_m.group(1)) if max_m else None
  return date_iso, high


def build_cli_highs(pil: str) -> dict[str, int]:
  """
  Walk all dates in the range and pull the CLI-reported high for each.
  Two phases run in a thread pool: (1) list products per date, (2) fetch
  each product's text. Cache hits are free; only network calls hit the
  thread pool. Iowa Mesonet handles ~10 concurrent requests fine.
  """
  log.info("  CLI %s: fetching products day-by-day from %s to %s...",
           pil, START_DATE, END_DATE)

  # Phase 1: list products for every date concurrently
  all_dates = []
  d = START_DATE
  while d <= END_DATE:
    all_dates.append(d)
    d += timedelta(days=1)

  product_meta: list[dict] = []
  with ThreadPoolExecutor(max_workers=10) as ex:
    futures = {ex.submit(list_cli_products, pil, dt): dt for dt in all_dates}
    for fut in as_completed(futures):
      try:
        product_meta.extend(fut.result())
      except Exception as exc:
        log.warning("    list_cli_products(%s, %s) failed: %s",
                    pil, futures[fut], exc)

  # Phase 2: fetch every product's text concurrently
  highs: dict[str, int] = {}
  # Sort so that later (corrected) products overwrite earlier ones via dict assignment
  product_meta.sort(key=lambda x: x["entered"])
  with ThreadPoolExecutor(max_workers=10) as ex:
    futures = {ex.submit(fetch_cli_text, p["product_id"]): p for p in product_meta}
    results = []
    for fut in as_completed(futures):
      try:
        text = fut.result()
        results.append((futures[fut]["entered"], text))
      except Exception as exc:
        log.warning("    fetch_cli_text(%s) failed: %s",
                    futures[fut]["product_id"], exc)

  # Replay in chronological order so corrections supersede
  results.sort(key=lambda x: x[0])
  for _, text in results:
    summarized_date, high = parse_cli_high(text)
    if summarized_date and high is not None:
      highs[summarized_date] = high

  log.info("  CLI %s: parsed %d daily highs", pil, len(highs))
  return highs


# ── ASOS side ─────────────────────────────────────────────────────────────

def fetch_asos_csv(station: str, start: datetime.date, end: datetime.date) -> str:
  """Cached bulk download of ASOS hourly tmpf."""
  cache = CACHE_DIR / f"asos_{station}_{start}_{end}.csv"
  if cache.exists():
    return cache.read_text()
  params = {
    "station": station,
    "data": "tmpf",
    "year1": start.year, "month1": start.month, "day1": start.day,
    "year2": end.year, "month2": end.month, "day2": end.day,
    "tz": "Etc/UTC",
    "format": "onlycomma",
    "latlon": "no",
    "missing": "M",
    "trace": "T",
  }
  log.info("  ASOS %s: bulk download %s..%s", station, start, end)
  r = requests.get(ASOS_BULK, params=params, timeout=120)
  r.raise_for_status()
  cache.write_text(r.text)
  return r.text


def build_asos_highs(station: str, tz_name: str) -> dict[str, int]:
  """Group ASOS hourly tmpf by LOCAL date, return per-day max as int °F."""
  csv_text = fetch_asos_csv(station, START_DATE, END_DATE)
  tz = ZoneInfo(tz_name)
  by_date: dict[str, list[float]] = defaultdict(list)

  reader = csv.DictReader(csv_text.splitlines())
  for row in reader:
    val = row.get("tmpf", "")
    if not val or val in ("M", "T"):
      continue
    try:
      f = float(val)
    except ValueError:
      continue
    valid = row.get("valid", "")
    try:
      # Mesonet timestamps are "YYYY-MM-DD HH:MM" in the requested tz (Etc/UTC here)
      dt_utc = datetime.strptime(valid, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    except ValueError:
      continue
    local_date = dt_utc.astimezone(tz).date().isoformat()
    by_date[local_date].append(f)

  # Drop days with < 18 hourly obs (gaps); compute integer max for the rest
  highs: dict[str, int] = {}
  dropped = 0
  for d, vals in by_date.items():
    if len(vals) < 18:
      dropped += 1
      continue
    highs[d] = int(round(max(vals)))
  log.info("  ASOS %s: %d days with ≥18 obs (dropped %d sparse days)",
           station, len(highs), dropped)
  return highs


# ── Comparison ────────────────────────────────────────────────────────────

def compare_one_city(s: dict) -> dict:
  log.info("=== %s (%s) ===", s["city"], s["icao"])
  cli_highs = build_cli_highs(s["cli"])
  asos_highs = build_asos_highs(s["asos"], s["tz"])

  rows: list[dict] = []
  for date, cli_h in cli_highs.items():
    asos_h = asos_highs.get(date)
    if asos_h is None:
      continue
    rows.append({
      "city": s["city"], "date": date,
      "cli_high": cli_h, "asos_max": asos_h,
      "delta": cli_h - asos_h,
    })

  if not rows:
    log.warning("  %s: no overlapping days!", s["city"])
    return {"rows": [], "summary": None}

  # Histogram + tail probabilities
  hist: dict[int, int] = defaultdict(int)
  for r in rows:
    hist[r["delta"]] += 1
  abs_deltas = [abs(r["delta"]) for r in rows]
  n = len(rows)
  summary = {
    "n_days": n,
    "median_abs": float(sorted(abs_deltas)[n // 2]),
    "mean_abs": round(sum(abs_deltas) / n, 3),
    "p_diff_ge_1": round(sum(1 for d in abs_deltas if d >= 1) / n, 4),
    "p_diff_ge_2": round(sum(1 for d in abs_deltas if d >= 2) / n, 4),
    "p_diff_ge_3": round(sum(1 for d in abs_deltas if d >= 3) / n, 4),
    "histogram": {str(k): v for k, v in sorted(hist.items())},
  }
  log.info("  %s: n=%d  P(|Δ|≥1)=%.1f%%  P(|Δ|≥2)=%.1f%%  P(|Δ|≥3)=%.2f%%",
           s["city"], n,
           100*summary["p_diff_ge_1"],
           100*summary["p_diff_ge_2"],
           100*summary["p_diff_ge_3"])
  return {"rows": rows, "summary": summary}


def main():
  raw_rows: list[dict] = []
  summaries: dict[str, dict] = {}

  for s in STATIONS:
    res = compare_one_city(s)
    raw_rows.extend(res["rows"])
    if res["summary"]:
      summaries[s["city"]] = res["summary"]

  out_dir = Path(__file__).resolve().parent

  # Raw CSV
  csv_path = out_dir / "cli_vs_asos_raw.csv"
  with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["city", "date", "cli_high", "asos_max", "delta"])
    w.writeheader()
    w.writerows(raw_rows)

  # JSON summary
  (out_dir / "cli_vs_asos_divergence.json").write_text(
    json.dumps(summaries, indent=2)
  )

  # Markdown report
  md = ["# NWS CLI vs ASOS Hourly Max — Backtest\n"]
  md.append("CLI is what Kalshi reads (NWS Climatological Report).")
  md.append("ASOS hourly max is a free proxy for Wunderground (Polymarket's source).")
  md.append("Δ = CLI_high − ASOS_max, in integer °F. Both observed at the same airport.\n")
  md.append("## Per-city results\n")
  md.append("| City | N days | median \\|Δ\\| | mean \\|Δ\\| | P(\\|Δ\\|≥1) | P(\\|Δ\\|≥2) | P(\\|Δ\\|≥3) |")
  md.append("|------|------:|-----------:|---------:|----------:|----------:|----------:|")
  for city, s in summaries.items():
    md.append(
      f"| {city} | {s['n_days']} | {s['median_abs']:.0f}°F | {s['mean_abs']:.2f}°F | "
      f"{s['p_diff_ge_1']:.1%} | {s['p_diff_ge_2']:.1%} | {s['p_diff_ge_3']:.1%} |"
    )
  md.append("\n## Caveats\n")
  md.append("- ASOS is a *lower bound* proxy for Wunderground divergence. Wunderground does some "
            "smoothing and occasional source-switching that pure ASOS-max won't replicate, so real "
            "Polymarket vs Kalshi disagreement is likely slightly higher than these numbers.")
  md.append("- Days with fewer than 18 valid hourly observations were dropped (data gaps).")
  md.append("- CLI corrections supersede earlier products for the same date (last-write-wins).")
  md.append("- Time zones: daily highs are computed in the airport's local time, not UTC.")
  (out_dir / "cli_vs_asos_report.md").write_text("\n".join(md) + "\n")
  log.info("Wrote cli_vs_asos_raw.csv, cli_vs_asos_divergence.json, cli_vs_asos_report.md")


if __name__ == "__main__":
  main()
