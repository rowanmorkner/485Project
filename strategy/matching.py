"""
Cross-venue event matching by date.

Kalshi events embed the date in the ticker (e.g. "KXHIGHMIA-26APR27");
Polymarket events expose it via the ISO endDate field. We normalize both
to "YYYY-MM-DD" strings, then pair events that share a date.
"""

import re

MONTH_MAP = {
  "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
  "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
  "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
}


def parse_kalshi_date(event_ticker: str) -> str:
  """
  Extract the resolution date from a Kalshi event ticker.

  Format: "<series>-YYMMMDD"  e.g. "KXHIGHMIA-26APR01" -> "2026-04-01".
  Returns "" if the ticker doesn't match the expected pattern.
  """
  match = re.search(r"-(\d{2})([A-Z]{3})(\d{2})$", event_ticker)
  if not match:
    return ""
  year = f"20{match.group(1)}"
  month = MONTH_MAP.get(match.group(2), "01")
  day = match.group(3)
  return f"{year}-{month}-{day}"


def parse_polymarket_date(slug: str, end_date: str) -> str:
  """
  Extract the resolution date from a Polymarket event.

  Polymarket exposes the endDate as an ISO timestamp; we just take the
  date portion. The slug parameter is accepted for symmetry with the
  Kalshi parser and for future fallback if endDate is missing.
  """
  if end_date and "T" in end_date:
    return end_date.split("T")[0]
  return ""


def match_events_by_date(
  kalshi_events: list[dict],
  poly_events: list[dict],
) -> list[tuple[str, dict | None, dict | None]]:
  """
  Pair Kalshi and Polymarket events that resolve on the same date.

  Returns:
    List of (date, kalshi_event_or_None, poly_event_or_None), sorted by date.
    A date may appear with only one side populated if the other venue
    doesn't have a market for it.
  """
  by_date: dict[str, dict] = {}

  for event in kalshi_events:
    d = event.get("date", "")
    if d:
      by_date.setdefault(d, {"kalshi": None, "poly": None})
      by_date[d]["kalshi"] = event

  for event in poly_events:
    d = event.get("date", "")
    if d:
      by_date.setdefault(d, {"kalshi": None, "poly": None})
      by_date[d]["poly"] = event

  return [
    (date, pair["kalshi"], pair["poly"])
    for date, pair in sorted(by_date.items())
  ]
