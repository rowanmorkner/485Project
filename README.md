# MATH485 Weather Arbitrage Bot

A Python bot that identifies pricing discrepancies between prediction markets (Kalshi, Polymarket) and NWS weather forecasts for daily high temperature events across 7 US cities.

## How It Works

The bot reconstructs **per-degree probability distributions** from each source's incompatible temperature bins, then compares them to find mispriced contracts. Since Kalshi and Polymarket use different bin widths (e.g., `77-78°F` vs `78-79°F`), a piecewise-uniform interpolation normalizes everything to a common integer-degree resolution before comparison.

Opportunities are classified by risk:
- **CLEAN ARB** — Both platforms resolve against the same weather station. True arbitrage.
- **BASIS RISK** — Platforms resolve against different stations (e.g., Central Park vs LaGuardia for NYC). Edge may be absorbed by 1-3°F station divergence. Requires a higher threshold.
- **FORECAST** — Market is mispriced relative to the NWS forecast. Depends on forecast accuracy.

## Architecture

```
main.py                    # Orchestrator — iterates cities, fetches data, runs analysis
cities.py                  # Per-city config: series tickers, station names, NWS gridpoints
config.py                  # Loads .env secrets (Kalshi key, key path)

clients/
  kalshi.py                # Kalshi API — RSA-PSS signed auth, event/market/orderbook queries
  polymarket.py            # Polymarket Gamma + CLOB APIs — event search, pricing, orderbooks
  weather.py               # NWS API — daily/hourly forecasts for arbitrary gridpoints

strategy/
  arbitrage.py             # Distribution reconstruction, CDF comparison, opportunity detection
```

### Data Flow

```
  Kalshi API ──────────> parse_kalshi_bins() ──┐
                                               ├──> find_discrepancies()
  Polymarket API ──────> parse_polymarket_bins()│    find_arbitrage_opportunities()
                                               │
  NWS API ─────────────> forecast_to_distribution()
  (per-station)            (Gaussian, σ=2°F)
```

1. **Fetch** — `main.py` pulls open events from both platforms and NWS forecasts for the correct resolution station per platform.
2. **Parse** — Bin prices are converted to per-degree PMFs via piecewise-uniform spreading. Tail bins (e.g., "85°F or above") are spread over `TAIL_SPREAD` degrees beyond the boundary.
3. **Compare** — PMF and CDF tables are printed side-by-side. Discrepancies exceeding a configurable threshold are flagged.
4. **Classify** — Cross-platform opportunities are labeled CLEAN ARB or BASIS RISK based on whether `cities.py` maps both platforms to the same NWS gridpoint.

### Supported Cities

| City | Kalshi Series | Kalshi Station | Polymarket Station | Same Station? |
|------|--------------|----------------|-------------------|--------------|
| Miami | `KXHIGHMIA` | KMIA | KMIA | Yes |
| Denver | `KXHIGHDEN` | KDEN | KBKF (Buckley SFB) | No |
| Phoenix | `KXHIGHTPHX` | KPHX | KPHX | Yes |
| NYC | `KXHIGHNY` | Central Park | KLGA (LaGuardia) | No |
| Boston | `KXHIGHTBOS` | KBOS | KBOS | Yes |
| LA | `KXHIGHLAX` | KLAX | KLAX | Yes |
| Chicago | `KXHIGHCHI` | KMDW (Midway) | KORD (O'Hare) | No |

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure credentials
cp .env.example .env
# Edit .env with your Kalshi API key ID and private key path
```

### Required Environment Variables

| Variable | Description |
|----------|-------------|
| `KALSHI_API_KEY_ID` | Kalshi API key (UUID format) |
| `KALSHI_PRIVATE_KEY_PATH` | Path to RSA private key PEM file |

Polymarket and NWS require no authentication for read operations.

## Usage

```bash
# Run full scan across all cities
python main.py

# Test individual clients
python clients/kalshi.py
python clients/polymarket.py
python clients/weather.py

# Run strategy module with example data
python strategy/arbitrage.py
```

## Key API Details

- **Kalshi**: RSA-PSS signing (SHA-256, MGF1, salt=digest length). Base URL: `https://api.elections.kalshi.com/trade-api/v2`
- **Polymarket**: Gamma API (`gamma-api.polymarket.com`) for event discovery via `tag_slug="temperature"`. CLOB API (`clob.polymarket.com`) for orderbook pricing.
- **NWS**: Free public API at `api.weather.gov`. Requires `User-Agent` header. Forecasts fetched per-gridpoint to match each platform's resolution station.
