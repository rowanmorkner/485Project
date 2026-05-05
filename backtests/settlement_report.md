# Kalshi vs Polymarket — Settlement Divergence Backtest

For each (city, date) where both venues had a resolved market, we compare which bracket settled YES on each side. Kalshi reads NWS CLI products; Polymarket reads Wunderground. Even at the same airport these can differ.

**Classification:**

- **AGREE**: winning ranges overlap (share ≥ 1°F)
- **ADJACENT**: no overlap, ≤ 0°F gap (touching — typical 1°F rounding)
- **DISAGREE**: no overlap, ≥ 1°F gap (real basis risk)

## Per-city results

| City | N pairs | Agree | Adjacent | Disagree | Agree % | Adjacent % | Disagree % |
|------|--------:|------:|---------:|---------:|--------:|-----------:|-----------:|
| Miami | 66 | 49 | 11 | 6 | 74.2% | 16.7% | 9.1% |
| LA | 32 | 22 | 9 | 1 | 68.8% | 28.1% | 3.1% |
| Austin | 32 | 25 | 6 | 1 | 78.1% | 18.8% | 3.1% |
| San Francisco | 32 | 24 | 7 | 1 | 75.0% | 21.9% | 3.1% |
| Seattle | 67 | 54 | 12 | 1 | 80.6% | 17.9% | 1.5% |

## Caveats

- Sample sizes are small — Kalshi per-city daily temp markets started in 2024-ish, Polymarket's are even newer.
- Kalshi runs multiple bracket-structure variants per date; we keep the narrowest winning bracket as the most informative reading. All variant tickers are saved in settlement_pairs.json.
- AGREE is the only signal that the two venues read the same temperature. ADJACENT is most likely 1°F rounding / observation timing. DISAGREE is real basis risk and should drive the EV haircut in find_value_trades.
