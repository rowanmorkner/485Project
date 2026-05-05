"""
Consolidate the two backtests into a single risk_model.json that
strategy/ code can load to apply per-city basis-risk haircuts.

Inputs:
  cli_vs_asos_divergence.json   — empirical |Δ| histogram for CLI vs ASOS-max
  settlement_divergence.json    — Polymarket-vs-Kalshi settled-bracket disagreement

Output:
  risk_model.json with per-city:
    p_divergence_1f / 2f / 3f   — empirical tail probabilities
    settlement_disagree_pct     — actual past disagreement rate
    recommended_min_edge_haircut — extra dollar edge required to clear bar,
                                   indexed off P(|Δ|≥2°F) which is the most
                                   relevant for 2°F-wide Polymarket brackets
"""

import json
from pathlib import Path

BACKTESTS = Path(__file__).resolve().parent

asos = json.loads((BACKTESTS / "cli_vs_asos_divergence.json").read_text())
settle = json.loads((BACKTESTS / "settlement_divergence.json").read_text())

risk_model: dict[str, dict] = {}

for city, asos_stats in asos.items():
  s = settle.get(city, {})

  # Heuristic: extra edge required = expected loss from a 2°F slip into an
  # adjacent (losing) bracket. With a 2°F-wide Polymarket bracket, a 2°F
  # divergence between CLI and Wunderground roughly halves the bracket's
  # win probability, costing on average ~$0.50 per contract. So the
  # haircut in dollar edge per contract ≈ 0.5 * P(|Δ|≥2°F).
  p2 = asos_stats.get("p_diff_ge_2", 0.0)
  haircut = round(0.5 * p2, 4)

  risk_model[city] = {
    "n_days_observed": asos_stats.get("n_days"),
    "p_divergence_1f": asos_stats.get("p_diff_ge_1"),
    "p_divergence_2f": p2,
    "p_divergence_3f": asos_stats.get("p_diff_ge_3"),
    "median_abs_divergence": asos_stats.get("median_abs"),
    "settlement_n_pairs": s.get("n_pairs"),
    "settlement_disagree_pct": s.get("disagree_pct"),
    "recommended_min_edge_haircut": haircut,
    "notes": (
      f"CLI vs ASOS-max measured over {asos_stats.get('n_days')} days; "
      f"Kalshi-vs-Polymarket settlement compared over {s.get('n_pairs', 0)} dates. "
      f"Haircut = 0.5 * P(|Δ|>=2°F)."
    ),
  }

out = BACKTESTS / "risk_model.json"
out.write_text(json.dumps(risk_model, indent=2))
print(f"Wrote {out}")

# Pretty-print the model
print("\nPer-city risk parameters:")
print(f"{'City':<16} {'P(|Δ|≥1)':>10} {'P(|Δ|≥2)':>10} {'P(|Δ|≥3)':>10} "
      f"{'SettleDisagree':>14} {'EdgeHaircut':>12}")
for city, r in risk_model.items():
  sdis = r["settlement_disagree_pct"]
  print(f"{city:<16} {r['p_divergence_1f']:>10.1%} {r['p_divergence_2f']:>10.1%} "
        f"{r['p_divergence_3f']:>10.1%} "
        f"{sdis:>14.1%}" if sdis is not None else f"{'n/a':>14}",
        f"{r['recommended_min_edge_haircut']:>12.4f}")
