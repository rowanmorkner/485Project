#!/usr/bin/env python3
"""
Cross-platform paper trading bot for Kalshi + Polymarket.

Consumes externally generated ORDER INSTRUCTIONS.

Expected input:

{
  "orders": [
    {
      "platform": "kalshi",
      "ticker": "KXHIGHMIA-26APR26-B84.5",
      "action": "buy",
      "side": "yes",
      "count": 1,
      "type": "limit"
    },
    {
      "platform": "polymarket",
      "condition_id": "0xabc123...",
      "action": "buy",
      "side": "yes",
      "count": 1,
      "type": "limit"
    }
  ]
}

Paper only. No live orders are placed.
"""

from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

from clients.kalshi import KalshiClient
from clients.polymarket import PolymarketClient


def get_kalshi_best_prices(orderbook_fp: dict[str, Any]) -> dict[str, Optional[float]]:
    yes_levels = orderbook_fp.get("yes_dollars", []) or []
    no_levels = orderbook_fp.get("no_dollars", []) or []

    best_yes_bid = float(yes_levels[-1][0]) if yes_levels else None
    best_no_bid = float(no_levels[-1][0]) if no_levels else None

    best_yes_ask = round(1 - best_no_bid, 4) if best_no_bid is not None else None
    best_no_ask = round(1 - best_yes_bid, 4) if best_yes_bid is not None else None

    return {
        "best_yes_bid": best_yes_bid,
        "best_yes_ask": best_yes_ask,
        "best_no_bid": best_no_bid,
        "best_no_ask": best_no_ask,
    }


def normalize_outcome_name(name: str) -> str:
    return (name or "").strip().lower()


@dataclass
class PaperOrder:
    timestamp_utc: str
    client_order_id: str
    platform: str
    ticker: str
    action: str
    side: str
    count: int
    order_type: str
    fill_price: float
    market_title: str
    market_subtitle: str
    status: str
    note: str


class CrossPlatformPaperTrader:
    def __init__(self) -> None:
        self.kalshi = KalshiClient()
        self.poly = PolymarketClient()

    def validate_order(self, order: dict[str, Any]) -> None:
        required = ["platform", "action", "side", "count", "type"]
        missing = [field for field in required if field not in order]
        if missing:
            raise ValueError(f"Order missing required field(s): {', '.join(missing)}")

        if order["platform"] not in {"kalshi", "polymarket"}:
            raise ValueError("platform must be 'kalshi' or 'polymarket'")

        if order["action"] not in {"buy", "sell"}:
            raise ValueError("action must be 'buy' or 'sell'")

        if order["side"] not in {"yes", "no"}:
            raise ValueError("side must be 'yes' or 'no'")

        if order["type"] not in {"limit", "market"}:
            raise ValueError("type must be 'limit' or 'market'")

        if not isinstance(order["count"], int) or order["count"] <= 0:
            raise ValueError("count must be a positive integer")

        if order["platform"] == "kalshi" and "ticker" not in order:
            raise ValueError("Kalshi orders require 'ticker'")

        if order["platform"] == "polymarket" and "condition_id" not in order and "token_id" not in order:
            raise ValueError("Polymarket orders require 'condition_id' or 'token_id'")

    # ---------------------------
    # Kalshi
    # ---------------------------

    def fetch_kalshi_snapshot(self, ticker: str) -> dict[str, Any]:
        market = self.kalshi.get_market(ticker)
        orderbook = self.kalshi.get_orderbook(ticker)
        prices = get_kalshi_best_prices(orderbook.get("orderbook_fp", {}) or {})

        return {
            "platform": "kalshi",
            "ticker": ticker,
            "title": market.get("title", "") or market.get("event_ticker", ""),
            "subtitle": market.get("subtitle", "") or market.get("yes_sub_title", "") or "",
            **prices,
        }

    def kalshi_fill_price(self, action: str, side: str, snapshot: dict[str, Any]) -> Optional[float]:
        if action == "buy" and side == "yes":
            return snapshot.get("best_yes_ask")
        if action == "buy" and side == "no":
            return snapshot.get("best_no_ask")
        if action == "sell" and side == "yes":
            return snapshot.get("best_yes_bid")
        if action == "sell" and side == "no":
            return snapshot.get("best_no_bid")
        return None

    # ---------------------------
    # Polymarket
    # ---------------------------

    def resolve_polymarket_token(self, order: dict[str, Any]) -> tuple[str, str, str]:
        """
        Returns: (token_id, title, subtitle)
        subtitle is just the chosen outcome name for logging.
        """
        side = normalize_outcome_name(order["side"])

        if "token_id" in order:
            token_id = order["token_id"]
            title = order.get("title", "")
            subtitle = side
            return token_id, title, subtitle

        condition_id = order["condition_id"]
        market = self.poly.get_clob_market(condition_id)
        tokens = market.get("tokens", []) or []

        desired_outcome = "yes" if side == "yes" else "no"
        chosen = None
        for token in tokens:
            outcome = normalize_outcome_name(token.get("outcome", ""))
            if outcome == desired_outcome:
                chosen = token
                break

        if chosen is None:
            raise ValueError(f"Could not find outcome token for side='{side}' in condition_id={condition_id}")

        token_id = chosen["token_id"]
        title = market.get("question", "") or market.get("description", "") or condition_id
        subtitle = chosen.get("outcome", desired_outcome)
        return token_id, title, subtitle

    def fetch_polymarket_snapshot(self, order: dict[str, Any]) -> dict[str, Any]:
        token_id, title, subtitle = self.resolve_polymarket_token(order)
        book = self.poly.get_orderbook(token_id)

        bids = book.get("bids", []) or []
        asks = book.get("asks", []) or []

        best_bid = float(bids[0]["price"]) if bids else None
        best_ask = float(asks[0]["price"]) if asks else None

        return {
            "platform": "polymarket",
            "ticker": token_id,
            "title": title,
            "subtitle": subtitle,
            "best_bid": best_bid,
            "best_ask": best_ask,
        }

    def polymarket_fill_price(self, action: str, snapshot: dict[str, Any]) -> Optional[float]:
        if action == "buy":
            return snapshot.get("best_ask")
        if action == "sell":
            return snapshot.get("best_bid")
        return None

    # ---------------------------
    # Shared processing
    # ---------------------------

    def process_order(self, order: dict[str, Any]) -> Optional[PaperOrder]:
        self.validate_order(order)

        platform = order["platform"]
        action = order["action"]
        side = order["side"]
        count = order["count"]
        order_type = order["type"]

        if platform == "kalshi":
            snapshot = self.fetch_kalshi_snapshot(order["ticker"])
            fill_price = self.kalshi_fill_price(action, side, snapshot)
            ticker_for_log = order["ticker"]
        else:
            snapshot = self.fetch_polymarket_snapshot(order)
            fill_price = self.polymarket_fill_price(action, snapshot)
            ticker_for_log = snapshot["ticker"]

        if fill_price is None:
            print(f"[SKIP] {platform} {ticker_for_log}: no executable paper fill price")
            return None

        paper_order = PaperOrder(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            client_order_id=str(uuid.uuid4()),
            platform=platform,
            ticker=ticker_for_log,
            action=action,
            side=side,
            count=count,
            order_type=order_type,
            fill_price=float(fill_price),
            market_title=snapshot["title"],
            market_subtitle=snapshot["subtitle"],
            status="filled_paper",
            note="paper trade filled at current best executable price",
        )

        print(
            f"[PAPER {platform.upper()} {action.upper()}] "
            f"{ticker_for_log} {side.upper()} x{count} @ {fill_price:.4f} | {snapshot['title']} | {snapshot['subtitle']}"
        )
        return paper_order


def load_orders(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    orders = payload.get("orders", [])
    if not isinstance(orders, list):
        raise ValueError("Order file must contain a top-level 'orders' list.")
    return orders


def append_ledger(path: str, orders: list[PaperOrder]) -> None:
    if not orders:
        return

    with open(path, "a", encoding="utf-8") as f:
        for order in orders:
            f.write(json.dumps(asdict(order)) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-platform paper execution bot")
    parser.add_argument("--orders", required=True, help="Path to JSON file containing flagged order instructions")
    parser.add_argument("--ledger", default="paper_trades.jsonl", help="Path to JSONL ledger")
    parser.add_argument("--dry-run", action="store_true", help="Do not write ledger entries")
    args = parser.parse_args()

    trader = CrossPlatformPaperTrader()
    incoming_orders = load_orders(args.orders)

    paper_orders: list[PaperOrder] = []
    for order in incoming_orders:
        try:
            result = trader.process_order(order)
            if result is not None:
                paper_orders.append(result)
        except Exception as exc:
            ident = order.get("ticker") or order.get("condition_id") or order.get("token_id") or "<unknown>"
            print(f"[ERROR] {ident}: {exc}")

    if args.dry_run:
        print(f"\nDry run complete. Would write {len(paper_orders)} paper order(s).")
        return

    append_ledger(args.ledger, paper_orders)
    print(f"\nWrote {len(paper_orders)} paper order(s) to {args.ledger}")


if __name__ == "__main__":
    main()