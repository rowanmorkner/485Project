"""
Fit a per-venue slippage model from logged fills.

Run this when ≥ 100 fills exist per venue. Output is data/slippage_params.json
which strategy/orders.py reads to cap order sizes such that expected slippage
doesn't erase the predicted edge.

Model: signed slippage = a + b * size + c * size^2
  - For BUY  orders: slippage = fill_price − intended_price (positive = paid more)
  - For SELL orders: slippage = intended_price − fill_price (positive = received less)

So in both cases, "more positive slippage" means "more cost than expected."

Run:
  python -m analysis.slippage
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from persistence import db


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

OUTPUT_PATH = PROJECT_ROOT / "data" / "slippage_params.json"
MIN_FILLS = 100


def fit_quadratic(sizes: list[float], slippages: list[float]) -> dict[str, float]:
  """
  Least-squares fit of slippage = a + b*size + c*size^2 using numpy.polyfit.
  Returns {a, b, c}. Falls back to constants if numpy is unavailable.
  """
  try:
    import numpy as np
    coeffs = np.polyfit(sizes, slippages, deg=2)  # returns highest-order first
    return {"a": float(coeffs[2]), "b": float(coeffs[1]), "c": float(coeffs[0])}
  except ImportError:
    # Fallback: report mean slippage as constant offset, no size dependence
    if not slippages:
      return {"a": 0.0, "b": 0.0, "c": 0.0}
    return {"a": sum(slippages) / len(slippages), "b": 0.0, "c": 0.0}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--min-fills", type=int, default=MIN_FILLS,
                      help="Minimum fills per venue before publishing.")
  args = parser.parse_args()

  with db.connect() as conn:
    rows = conn.execute(
      """
      SELECT o.venue AS venue, o.action AS action,
             o.intended_price AS intended_price,
             f.fill_price AS fill_price, f.fill_size AS fill_size
      FROM fills f
      JOIN orders o ON o.id = f.order_id
      """
    ).fetchall()

  by_venue: dict[str, list[tuple[float, float]]] = defaultdict(list)
  for r in rows:
    if r["action"] == "buy":
      slip = r["fill_price"] - r["intended_price"]
    elif r["action"] == "sell":
      slip = r["intended_price"] - r["fill_price"]
    else:
      continue
    by_venue[r["venue"]].append((float(r["fill_size"]), slip))

  out: dict[str, dict] = {}
  for venue, samples in by_venue.items():
    n = len(samples)
    if n < args.min_fills:
      log.info("%s: only %d fills (min=%d) — skipping",
               venue, n, args.min_fills)
      continue
    sizes = [s[0] for s in samples]
    slips = [s[1] for s in samples]
    params = fit_quadratic(sizes, slips)
    out[venue] = {
      "n_fills": n,
      "params": params,
      "mean_slippage": round(sum(slips) / n, 5),
    }
    log.info("%s: n=%d  a=%.4f b=%.4f c=%.6f  mean_slip=%.4f",
             venue, n, params["a"], params["b"], params["c"],
             out[venue]["mean_slippage"])

  if not out:
    log.warning("Not enough fills to publish slippage params.")
    return

  OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
  tmp = OUTPUT_PATH.with_suffix(".json.tmp")
  tmp.write_text(json.dumps(out, indent=2))
  os.replace(tmp, OUTPUT_PATH)
  log.info("Wrote %s with %d venues", OUTPUT_PATH, len(out))


if __name__ == "__main__":
  main()
