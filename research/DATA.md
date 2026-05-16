# Research data dictionary

All data lives in `data/bot.db` (SQLite). Helpers in `research/loaders.py`.

## Tables

### `forecasts` (2,863 rows, 2026-05-06 to 2026-05-07)
- `pdf_json` — JSON dict {degree(int) -> probability}
- `std_dev` — forecast σ (often null pre-calibration)
- One row per (city, date, source, fetched_at_utc). For backtests, take the
  first NWS forecast per (city, date).

### `snapshots` (8,158 rows)
- Polled every ~5 min between 2026-05-06 00:47 UTC and now.
- `brackets_json` — venue-shaped JSON list of bracket dicts. Schema
  varies by venue:
  - Kalshi: `{ticker, subtitle ("78° to 79°"), best_yes_bid, best_yes_ask,
    best_yes_bid_size, best_yes_ask_size, yes_bid_ladder, yes_ask_ladder}`
  - Polymarket: `{token_id, condition_id, question ("78-79°F"),
    best_yes_bid, best_yes_ask, best_yes_bid_size, best_yes_ask_size,
    yes_bid_ladder, yes_ask_ladder}`
- Snapshot density: ~335 per (city, venue) for the FULL polling day
  (2026-05-06). Earlier dates partial.

### `settlements` (400 rows, Dec 2025 to May 2026)
- `kalshi_high_f`, `polymarket_high_f` — actual realized highs.
- 188 rows have BOTH venues set (use these for cross-venue Δ stats).
- Use `parsers.parse_kalshi_quotes` / `parse_polymarket_quotes` to turn
  brackets into BracketQuote objects, then check which bracket contains
  the settled high.

### `orders` / `fills` / `pnl`
- 18,541 orders + 515 pnl rows from the live bot. Treat as REFERENCE only
  — the live strategy was wrong-sided; do NOT train on its predictions.

## Backtest reality

The bot started polling at 2026-05-06 00:47 UTC. Implications:

- **2026-05-05 markets** — useless. By the time we started polling those
  markets had already locked in (every snapshot shows 0/6 brackets with
  both bid+ask alive). Skip these.
- **2026-05-06 markets** — GOLD. 340 snapshots/(city,venue) covering the
  full trading window: brackets start 6/6 active, decay through the day,
  end 0/6 active. This is where mark-to-market backtests run. Settles
  tonight ~23:55 UTC, at which point hold-to-settle scoring also lights up.
- **2026-05-07 markets** — 181 snapshots, all brackets still actively
  trading at latest timestamp. Good for forward simulation; not yet
  scoreable.

**Mark-to-market exit is the universal scoring primitive.** A position
opens by lifting the ask ladder at snapshot t, then closes by hitting
the bid ladder at any later snapshot t+k where the close condition is
met (target hit, time stop expired, or end-of-window). Realized P&L is
strictly a function of snapshot data — no settlement required for
non-arb strategies.

For arb strategies that want hold-to-settle scoring, only 5/5 LA/SF are
currently settled (and those data are compromised; see above). 5/6 will
land 5 more cities tonight.

Only YES-side ladders are captured. To trade NO synthetically:
NO_ask = 1 − YES_bid, NO_bid = 1 − YES_ask. `loaders.py` does this
automatically when side='no_long'.

## Useful queries

```sql
-- All settled (city, date) pairs with both venues
SELECT city, date, kalshi_high_f, polymarket_high_f
  FROM settlements
 WHERE kalshi_high_f IS NOT NULL AND polymarket_high_f IS NOT NULL;

-- All snapshot-only dates (no settlement yet)
SELECT DISTINCT s.city, s.date FROM snapshots s
  LEFT JOIN settlements t USING(city, date)
 WHERE t.kalshi_high_f IS NULL AND t.polymarket_high_f IS NULL;
```
