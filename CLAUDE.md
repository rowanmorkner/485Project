# MATH485 Weather Arbitrage Bot

## Project Overview
Python bot that compares weather prediction markets (Kalshi, Polymarket) against real forecasts to find arbitrage opportunities. Currently focused on Miami daily high temperature.

## Allowed Tools
allowedTools:
  - Edit
  - Write
  - Bash
  - Read
  - Glob
  - Grep
  - Agent

## Build & Run
- **Install deps**: `pip install -r requirements.txt`
- **Run bot**: `python main.py`
- **Test individual clients**: `python clients/kalshi.py` or `python clients/polymarket.py`
- **Environment**: Config lives in `.env` (gitignored), template in `.env.example`

## Code Style
- Python, 2-space indentation
- camelCase for variables, snake_case for functions
- Comment complex logic
- Use `logging` module (not print) for debug output
- Load secrets from environment variables via `python-dotenv`

## Architecture
- `clients/` — API clients (kalshi.py, polymarket.py, weather.py)
- `strategy/` — Arbitrage detection logic
- `config.py` — Central config loader
- `main.py` — Entry point / orchestrator

## Key API Details
- **Kalshi**: RSA-PSS signed auth, base URL `https://api.elections.kalshi.com/trade-api/v2`,
- **Polymarket**: Gamma API for event discovery (use `tag_slug` param, not `_q`), CLOB API for pricing. No auth for reads.


# Claude chats to resume later: 
*polymarket orders:* 26edb4cf-13ac-4ce8-87df-5c53fbe766cb
