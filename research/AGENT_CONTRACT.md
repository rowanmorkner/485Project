# Strategy agent contract

You are one of 5 parallel agents researching a trading strategy for a
weather-arbitrage bot (MATH485 project). Read this file in full before
starting.

## Project background

`/home/rowan/githuball/485Project/CLAUDE.md` has the full project intro.
Short version: the bot polls Kalshi and Polymarket for daily-high
temperature brackets in 5 US cities (Miami, LA, Austin, San Francisco,
Seattle), compares quotes against an NWS forecast PDF, and tries to
trade mispricings. The original strategy lost money in paper trading
(-$430K predicted profit, +$18.5M actual loss). We now have ~28 hours of
fresh snapshot data and want to research alternative strategies.

## Data

Read `/home/rowan/githuball/485Project/research/DATA.md` for the full
data dictionary. Quick reference:

- Database: `data/bot.db` (SQLite, 6 tables)
- Helpers: `research/loaders.py` — **read-only for you**. Do not modify.
- Backtestable window: **2026-05-06 markets** (340 snapshots/(city,venue),
  full trading day). Skip 2026-05-05 (compromised) and 2026-05-07 (not
  yet settled, but usable for forward sim).
- Historical settlements: 400 rows for distribution-fit training only
  (no snapshots accompany them).

## Your output

Create your own subdirectory under `/home/rowan/githuball/485Project/research/`
named `strategy_<N>_<short_name>/` (your number and slug given in the
launch prompt). Inside, you must produce by the end:

1. **`metrics.json`** — standardized metrics dict (schema below).
2. **`RESULTS.md`** — narrative writeup (1–3 pages markdown). Sections:
   - Strategy idea (1 paragraph)
   - Key implementation choices (what you tried, what you discarded)
   - Backtest results table (per-iteration metrics)
   - Failure modes you observed
   - Recommended next steps
3. **Code** — your strategy lives in this directory. Whatever Python
   files you need. Use the project venv: `/home/rowan/githuball/485Project/.venv/bin/python`.

### `metrics.json` schema

```json
{
  "strategy_name": "calibrated_gaussian",
  "iterations": [
    {
      "version": "v1",
      "n_trades": 142,
      "n_winners": 78,
      "win_rate": 0.549,
      "total_pnl_dollars": 12.34,
      "avg_pnl_per_trade": 0.087,
      "median_pnl_per_trade": 0.05,
      "sharpe_per_trade": 0.32,
      "max_drawdown_dollars": -3.21,
      "settled_n": 5,
      "settled_pnl_dollars": 1.40,
      "exit_reason_counts": {"target": 89, "time_stop": 41, "end_of_window": 12},
      "notes": "v1 baseline; high false-positive rate on illiquid brackets"
    },
    { "version": "v2", "...": "..." }
  ]
}
```

`settled_pnl_dollars` and `settled_n` only apply to trades where
hold-to-settle was used AND the (city, date) settlement is known.
For pure mark-to-market strategies, set them to 0/null.

## Required iterations

You MUST produce **at least 2 versions** (v1, v2). v1 is a working
baseline. v2 is informed improvement based on what v1 told you. If you
have time, do v3+. Document each version's metrics row.

## Constraints

- **DO NOT** modify `research/loaders.py`, `research/DATA.md`, anything
  outside your strategy subdirectory, or anything in `data/bot.db`.
  Read-only access elsewhere.
- **DO NOT** call out to external APIs (Kalshi, Polymarket, NWS) — you
  have full backtest data locally.
- Use Python via `/home/rowan/githuball/485Project/.venv/bin/python` (deps installed there).
- You may import existing project modules (`from strategy.parsers import ...`,
  `from contracts import ...`) but treat them as read-only.
- Be honest: if your strategy underperforms, say so in RESULTS.md.
  Negative findings are valuable.
- Keep your code self-contained in your subdirectory. No new top-level
  files outside research/strategy_<N>_<slug>/.

## Backtest mechanics

Use `research/loaders.py` primitives:

```python
from research.loaders import (
  iter_snapshots, load_forecast_pdf, list_settled_pairs,
  list_backtestable_pairs, list_dates_with_snapshots,
  open_position, simulate_exit, settlement_payout_for,
  walk_ladder_buy, walk_ladder_sell, max_size_at_or_better,
  find_kalshi_bracket, find_polymarket_bracket,
)
from strategy.parsers import _parse_kalshi_range, _parse_polymarket_range
```

Sketch of a typical backtest loop:

```python
results = []
for city in CITIES:
  date = '2026-05-06'
  snaps = list(iter_snapshots(city, date))   # [(ts, {venue: brackets}), ...]
  for entry_idx, (ts, venues) in enumerate(snaps):
    signal = my_signal_fn(venues, city, date, ts)
    if not signal: continue
    bracket = signal['bracket']  # one of venues[venue]
    pos = open_position(city, date, signal['venue'], bracket,
                        side=signal['side'], size=signal['size'], entry_ts=ts)
    if pos is None: continue
    pos.bracket_degrees = signal['degrees']
    future = [(t, v.get(signal['venue'], [])) for t, v in snaps[entry_idx+1:]]
    exit_result = simulate_exit(
      pos, future,
      target_pnl_per_contract=signal.get('target'),
      time_stop_minutes=signal.get('time_stop'),
      fee_per_contract=0.005,
    )
    results.append((pos, exit_result))
```

Be careful about double-counting: don't open the same position twice
across consecutive snapshots; track which (bracket_id, side) is already
open and skip duplicates within a session.

## Quality bar

A "good" strategy in this exercise demonstrates:

1. Positive total P&L on the 2026-05-06 mark-to-market window after
   spread + fees (we model fees at $0.005/contract round-trip-half by
   default; you can argue for different).
2. Sharpe per trade > 0.2 over n_trades >= 30 (some statistical weight).
3. The strategy doesn't depend on lookahead: at any point, you only use
   information available as of the entry timestamp.
4. Honest accounting of failure modes (illiquid brackets, wide spreads,
   convergence not happening, etc.).

A "great" strategy also has interpretable wins (you can point at why a
trade made money), and graceful failure when its assumption breaks.

If you find your strategy fundamentally cannot work on this data, say
so clearly. Don't invent a story.
