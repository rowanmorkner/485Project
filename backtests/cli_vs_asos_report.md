# NWS CLI vs ASOS Hourly Max — Backtest

CLI is what Kalshi reads (NWS Climatological Report).
ASOS hourly max is a free proxy for Wunderground (Polymarket's source).
Δ = CLI_high − ASOS_max, in integer °F. Both observed at the same airport.

## Per-city results

| City | N days | median \|Δ\| | mean \|Δ\| | P(\|Δ\|≥1) | P(\|Δ\|≥2) | P(\|Δ\|≥3) |
|------|------:|-----------:|---------:|----------:|----------:|----------:|
| Miami | 363 | 1°F | 0.80°F | 64.7% | 13.5% | 1.7% |
| LA | 363 | 1°F | 0.79°F | 63.6% | 13.2% | 1.9% |
| Austin | 365 | 1°F | 0.79°F | 65.5% | 11.2% | 1.9% |
| San Francisco | 361 | 1°F | 0.89°F | 66.5% | 16.3% | 4.4% |
| Seattle | 362 | 1°F | 0.67°F | 59.7% | 7.5% | 0.3% |

## Caveats

- ASOS is a *lower bound* proxy for Wunderground divergence. Wunderground does some smoothing and occasional source-switching that pure ASOS-max won't replicate, so real Polymarket vs Kalshi disagreement is likely slightly higher than these numbers.
- Days with fewer than 18 valid hourly observations were dropped (data gaps).
- CLI corrections supersede earlier products for the same date (last-write-wins).
- Time zones: daily highs are computed in the airport's local time, not UTC.
