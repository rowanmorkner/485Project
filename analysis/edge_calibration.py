"""
Read-only analytics over the pnl table: how does realized edge compare
to predicted edge across (venue, trade_type, city) buckets?

This is the model-validation half of Component 3. It does not feed back
into the trading loop; it produces tables for human review (or eventually
for an automated alert when a bucket goes systematically off).

Run:
  python -m analysis.edge_calibration                 # all buckets
  python -m analysis.edge_calibration --by venue      # one dimension
  python -m analysis.edge_calibration --drift-alert   # flag bad buckets
"""

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from persistence import db


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ALLOWED_GROUP_DIMS = {"venue", "trade_type", "city"}


def bucket_stats(group_by: list[str]) -> list[dict]:
  """
  Aggregate pnl rows by the given dimensions. trade_type comes from the
  embedded ArbOpportunity JSON in orders.opp_json.

  Returns a list of dicts with: group key, n, mean_realized, mean_predicted,
  mean_residual, hit_rate, sharpe (= mean / stdev of per-trade pnl).
  """
  for d in group_by:
    if d not in ALLOWED_GROUP_DIMS:
      raise ValueError(f"Bad group dim: {d}. Allowed: {ALLOWED_GROUP_DIMS}")

  with db.connect() as conn:
    rows = conn.execute(
      """
      SELECT o.id, o.venue, o.city, o.arb_pair_id, o.opp_json,
             p.dollars_realized, p.dollars_predicted, p.won_bool
      FROM orders o
      JOIN pnl p ON p.order_id = o.id
      """,
    ).fetchall()

  # Aggregate paired-arb legs first: a hedged position is one trade, not
  # two. Sum dollars_realized and dollars_predicted across legs sharing
  # an arb_pair_id; the trade "won" iff combined realized > 0.
  unpaired: list[dict] = []
  pairs: dict[str, dict] = defaultdict(
    lambda: {"realized": 0.0, "predicted": 0.0, "venues": [], "city": None,
             "trade_type": None, "n_legs": 0}
  )
  for r in rows:
    try:
      opp = json.loads(r["opp_json"])
      trade_type = opp.get("trade_type", "?")
    except (TypeError, ValueError):
      trade_type = "?"
    pid = r["arb_pair_id"]
    if pid:
      agg = pairs[pid]
      agg["realized"] += r["dollars_realized"]
      agg["predicted"] += r["dollars_predicted"]
      agg["venues"].append(r["venue"])
      agg["city"] = r["city"]
      agg["trade_type"] = trade_type
      agg["n_legs"] += 1
    else:
      unpaired.append({
        "venue": r["venue"], "city": r["city"], "trade_type": trade_type,
        "realized": r["dollars_realized"],
        "predicted": r["dollars_predicted"],
        "won": bool(r["won_bool"]),
      })

  # Convert paired aggregates into the same per-trade record shape
  for pid, agg in pairs.items():
    venue_label = "+".join(sorted(set(agg["venues"]))) or "paired"
    unpaired.append({
      "venue": venue_label, "city": agg["city"], "trade_type": agg["trade_type"],
      "realized": agg["realized"],
      "predicted": agg["predicted"],
      "won": agg["realized"] > 0,
    })

  buckets: dict[tuple, list[dict]] = defaultdict(list)
  for trade in unpaired:
    key_parts = []
    for dim in group_by:
      key_parts.append(trade.get(dim))
    buckets[tuple(key_parts)].append({
      "realized": trade["realized"],
      "predicted": trade["predicted"],
      "won": trade["won"],
    })

  out: list[dict] = []
  for key, trades in sorted(buckets.items()):
    n = len(trades)
    realized = [t["realized"] for t in trades]
    predicted = [t["predicted"] for t in trades]
    mean_r = sum(realized) / n
    mean_p = sum(predicted) / n
    if n > 1:
      var = sum((x - mean_r) ** 2 for x in realized) / (n - 1)
      stdev = math.sqrt(var)
    else:
      stdev = 0.0
    sharpe = (mean_r / stdev) if stdev > 0 else float("nan")
    hit_rate = sum(1 for t in trades if t["won"]) / n
    row = {dim: key[i] for i, dim in enumerate(group_by)}
    row.update({
      "n": n,
      "mean_realized": round(mean_r, 4),
      "mean_predicted": round(mean_p, 4),
      "mean_residual": round(mean_r - mean_p, 4),
      "hit_rate": round(hit_rate, 4),
      "sharpe": round(sharpe, 4) if not math.isnan(sharpe) else None,
    })
    out.append(row)
  return out


def model_drift_alert(min_n: int = 30, residual_threshold: float = -0.02) -> list[dict]:
  """
  Buckets where the model is systematically miscalibrated downward — i.e.
  realized consistently underperforms predicted, with enough sample size
  to be confident it's not noise.

  Returns the same row shape as bucket_stats(), restricted to flagged buckets.
  """
  stats = bucket_stats(["venue", "trade_type", "city"])
  return [
    s for s in stats
    if s["n"] >= min_n and s["mean_residual"] < residual_threshold
  ]


def _print_table(rows: list[dict]) -> None:
  if not rows:
    print("(no data)")
    return
  cols = list(rows[0].keys())
  widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
  header = "  ".join(c.ljust(widths[c]) for c in cols)
  print(header)
  print("-" * len(header))
  for r in rows:
    print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--by", nargs="+", default=["venue", "trade_type", "city"],
                      choices=sorted(ALLOWED_GROUP_DIMS),
                      help="Dimensions to group by.")
  parser.add_argument("--drift-alert", action="store_true",
                      help="Print only buckets with miscalibration signals.")
  parser.add_argument("--min-n", type=int, default=30,
                      help="Minimum sample size for drift-alert mode.")
  args = parser.parse_args()

  if args.drift_alert:
    rows = model_drift_alert(min_n=args.min_n)
    print(f"\n=== DRIFT ALERTS (min_n={args.min_n}) ===")
  else:
    rows = bucket_stats(args.by)
    print(f"\n=== BUCKET STATS (group_by={args.by}) ===")
  _print_table(rows)


if __name__ == "__main__":
  main()
