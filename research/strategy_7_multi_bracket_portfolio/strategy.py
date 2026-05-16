"""
Strategy 7 — Multi-bracket portfolio hedges.

Idea
----
A "leg" is a SUM of brackets on the same venue. By combining 1-2 brackets
per venue, we can synthesize equal-degree spreads even when individual
Kalshi/Polymarket grids don't align (SF/Seattle have a 1-degree offset).

For each cross-venue PORTFOLIO (legs_K, legs_P), each leg is
(bracket, side, qty) with qty=1 (one contract per leg). At settlement
with Kalshi reading k and Polymarket reading p, total payout =
  Σ_{leg in K} 1{k ∈ leg.win_set} + Σ_{leg in P} 1{p ∈ leg.win_set}

Total entry cost = Σ leg.avg_fill_for_qty=1 contract.
We compare the joint K-P payout distribution to entry cost.

Iterations
----------
- v1 STRICT STRUCTURAL: min payoff(k, p) over the (k,p) plausible support
  ≥ entry_cost + fees+safety. (Worst-case-positive across the joint
  support.)
- v2 CVaR-bounded EV: q05 payoff > entry_cost AND E[payoff] > entry_cost
  + 5¢ safety. Uses joint K-P from `joint_kp_distribution`.
- v3 NEW-ONLY: exclude any portfolio that reduces to a single-bracket-pair
  (size-1×size-1) configuration that S3 v2 would have selected. Reports
  *additive* trades from genuine multi-bracket combos.

Caveats
-------
- Each LEG (bracket+side) lifts the ASK ladder for qty=1 contract.
  This is per-pair sizing; we then scale the whole portfolio by
  PAIR_SIZE.
- Fees: $0.005/contract per leg. A K=2,P=2 portfolio has 4 legs ≈
  $0.02/pair entry fees. Hold-to-settle pays no exit fees.
- Joint K-P distribution uses `joint_kp_distribution(forecast_pdf,
  city)` — combines NWS PDF (treated as P(K)) with empirical Δ histogram
  (treated as P(P|K), K-independent approximation).
- We DEDUP by canonical portfolio key:
  ((city, sorted(K legs), sorted(P legs))).
- We restrict to portfolios with leg count L_K, L_P ∈ {1, 2}. K=1,P=1 is
  the S3 v2 baseline (kept for a comparison point but flagged in v3 as
  "reducible").
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from statistics import mean, median, stdev
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from research.loaders import (
    iter_snapshots,
    load_forecast_pdf,
    walk_ladder_buy,
    venue_divergence_histogram,
    joint_kp_distribution,
)
from strategy.parsers import _parse_kalshi_range, _parse_polymarket_range


CITIES = ["Miami", "LA", "Austin", "San Francisco", "Seattle"]
DATE = "2026-05-06"

FEE_PER_CONTRACT = 0.005
PAIR_SIZE = 50            # per-portfolio scale
MIN_PAIR_SIZE = 5
MAX_LEGS_PER_SIDE = 2
SAFETY_BUFFER = 0.05      # cents per pair ($0.05) — required edge above cost

# Time-budget controls: subsample snapshots for brute-force enumeration.
# Dedup is per (city, canonical-portfolio-key) so subsampling only delays
# the earliest-observation timestamp; it doesn't change the unique trade
# set materially.
SNAPSHOT_STRIDE = 6       # take every Nth snapshot
ALLOW_2x2 = False         # skip 2x2 portfolios to keep enumeration tractable


# ── Leg construction ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class Leg:
    """A single venue leg: one bracket + side + ask ladder."""
    venue: str             # 'K' or 'P'
    bracket_key: str       # ticker (K) or token_id (P) — stable id
    label: str
    side: str              # 'yes_long' | 'no_long'
    win_set: frozenset[int]


def _ask_ladder_yes(b: dict) -> list:
    return list(b.get("yes_ask_ladder") or [])


def _ask_ladder_no(b: dict) -> list:
    return [(round(1.0 - p, 4), q) for p, q in (b.get("yes_bid_ladder") or [])]


def _ask_ladder_for_leg(b: dict, side: str) -> list:
    return _ask_ladder_yes(b) if side == "yes_long" else _ask_ladder_no(b)


def build_legs(
    kalshi_brackets: list, poly_brackets: list, universe: set[int]
) -> tuple[list[tuple[Leg, dict]], list[tuple[Leg, dict]]]:
    """Build all legs with their bracket-dicts (for ladder access).
    Returns (kalshi_legs, poly_legs). Each item = (Leg, bracket_dict)."""
    k_legs: list[tuple[Leg, dict]] = []
    for b in kalshi_brackets:
        sub = b.get("subtitle", "") or ""
        deg = _parse_kalshi_range(sub)
        if not deg:
            continue
        d_set = frozenset(deg) & frozenset(universe)
        if not d_set:
            continue
        for side in ("yes_long", "no_long"):
            win = d_set if side == "yes_long" else frozenset(universe) - d_set
            ladder = _ask_ladder_for_leg(b, side)
            if not ladder:
                continue
            leg = Leg(
                venue="K", bracket_key=b.get("ticker", ""),
                label=sub, side=side, win_set=win,
            )
            k_legs.append((leg, b))

    p_legs: list[tuple[Leg, dict]] = []
    for b in poly_brackets:
        q = b.get("question", "") or ""
        deg = _parse_polymarket_range(q)
        if not deg:
            continue
        d_set = frozenset(deg) & frozenset(universe)
        if not d_set:
            continue
        for side in ("yes_long", "no_long"):
            win = d_set if side == "yes_long" else frozenset(universe) - d_set
            ladder = _ask_ladder_for_leg(b, side)
            if not ladder:
                continue
            leg = Leg(
                venue="P", bracket_key=b.get("token_id", ""),
                label=q, side=side, win_set=win,
            )
            p_legs.append((leg, b))
    return k_legs, p_legs


def universe_for_snapshot(kbs: list, pbs: list) -> set[int]:
    deg = set()
    for b in kbs:
        deg.update(_parse_kalshi_range(b.get("subtitle", "") or ""))
    for b in pbs:
        deg.update(_parse_polymarket_range(b.get("question", "") or ""))
    return deg


# ── Portfolio cost & payoff ──────────────────────────────────────────────


def fill_leg_cost(b: dict, side: str, qty: int) -> Optional[float]:
    """Avg fill price (per contract) for buying `qty` contracts of leg.
    Returns None if depth insufficient.
    """
    ladder = _ask_ladder_for_leg(b, side)
    fill = walk_ladder_buy(ladder, qty)
    if fill is None:
        return None
    avg, _ = fill
    return avg


def portfolio_cost_per_pair(
    k_legs_b: list[tuple[Leg, dict]], p_legs_b: list[tuple[Leg, dict]],
    qty: int,
) -> Optional[float]:
    """Sum of avg fill prices across all legs for a portfolio scaled to
    `qty` contracts per leg. Returns None if any leg can't fill."""
    total = 0.0
    for _, b in k_legs_b + p_legs_b:
        # Side resolved via Leg, but ladder is from bracket and side.
        pass
    total = 0.0
    for leg, b in k_legs_b:
        c = fill_leg_cost(b, leg.side, qty)
        if c is None:
            return None
        total += c
    for leg, b in p_legs_b:
        c = fill_leg_cost(b, leg.side, qty)
        if c is None:
            return None
        total += c
    return total


def portfolio_payoff(legs: list[Leg], reading: int) -> int:
    """How many of these legs pay $1 when the relevant venue reads `reading`."""
    return sum(1 for leg in legs if reading in leg.win_set)


def portfolio_payoff_kp(
    k_legs: list[Leg], p_legs: list[Leg], k: int, p: int
) -> float:
    return float(portfolio_payoff(k_legs, k) + portfolio_payoff(p_legs, p))


# ── Portfolio enumeration ─────────────────────────────────────────────────


def enumerate_portfolios(
    k_legs_b: list[tuple[Leg, dict]],
    p_legs_b: list[tuple[Leg, dict]],
    max_k: int = MAX_LEGS_PER_SIDE,
    max_p: int = MAX_LEGS_PER_SIDE,
    universe: Optional[set[int]] = None,
):
    """Yield (k_subset, p_subset) where each subset is a list of (Leg, bracket).
    Subset sizes 1..max_*. Skip "trivial" duplicates by canonicalising on
    (venue+bracket_key+side) tuples sorted.
    """
    # K subsets
    k_subsets = []
    for r in range(1, max_k + 1):
        for combo in combinations(range(len(k_legs_b)), r):
            sub = [k_legs_b[i] for i in combo]
            # Skip if multiple legs have the same bracket+side identity
            keys = sorted((l.bracket_key, l.side) for l, _ in sub)
            if len(set(keys)) != len(keys):
                continue
            k_subsets.append(sub)
    p_subsets = []
    for r in range(1, max_p + 1):
        for combo in combinations(range(len(p_legs_b)), r):
            sub = [p_legs_b[i] for i in combo]
            keys = sorted((l.bracket_key, l.side) for l, _ in sub)
            if len(set(keys)) != len(keys):
                continue
            p_subsets.append(sub)

    for k_sub in k_subsets:
        for p_sub in p_subsets:
            if not ALLOW_2x2 and len(k_sub) > 1 and len(p_sub) > 1:
                continue
            yield k_sub, p_sub


def portfolio_canonical_key(
    city: str, k_sub: list[tuple[Leg, dict]], p_sub: list[tuple[Leg, dict]]
) -> tuple:
    k_part = tuple(sorted((l.bracket_key, l.side) for l, _ in k_sub))
    p_part = tuple(sorted((l.bracket_key, l.side) for l, _ in p_sub))
    return (city, k_part, p_part)


# ── Filters ──────────────────────────────────────────────────────────────


def support_set_from_joint(joint: dict[tuple[int, int], float], min_prob: float = 0.0):
    return [(k, p) for (k, p), w in joint.items() if w > min_prob]


def evaluate_portfolio(
    k_legs: list[Leg], p_legs: list[Leg],
    cost: float, joint: dict[tuple[int, int], float],
    fee_per_pair: float,
):
    """Compute summary stats for one portfolio: worst-case payoff,
    expected payoff, q05 payoff, etc."""
    payoffs = []
    for (k, p), w in joint.items():
        payoffs.append((portfolio_payoff_kp(k_legs, p_legs, k, p), w))
    if not payoffs:
        return None

    worst = min(p for p, _ in payoffs)
    exp = sum(pay * w for pay, w in payoffs)

    # q05 = 5th percentile from sorted payoffs by payoff ascending
    sorted_p = sorted(payoffs, key=lambda x: x[0])
    cum = 0.0
    q05 = sorted_p[0][0]
    for pay, w in sorted_p:
        cum += w
        if cum >= 0.05:
            q05 = pay
            break
    # q01
    cum = 0.0
    q01 = sorted_p[0][0]
    for pay, w in sorted_p:
        cum += w
        if cum >= 0.01:
            q01 = pay
            break

    return {
        "cost_per_pair": cost,
        "fee_per_pair": fee_per_pair,
        "worst_case_payoff": worst,
        "expected_payoff": exp,
        "q05_payoff": q05,
        "q01_payoff": q01,
        "worst_case_pnl_per_pair": worst - cost - fee_per_pair,
        "expected_pnl_per_pair": exp - cost - fee_per_pair,
        "q05_pnl_per_pair": q05 - cost - fee_per_pair,
    }


# ── Backtest driver ──────────────────────────────────────────────────────


@dataclass
class TradeRecord:
    city: str
    date: str
    entry_ts: str
    pair_size: int
    k_legs: list[dict]
    p_legs: list[dict]
    cost_per_pair: float
    fee_per_pair: float
    worst_case_payoff: float
    expected_payoff: float
    q05_payoff: float
    q01_payoff: float
    worst_case_pnl: float
    expected_pnl: float
    q05_pnl: float
    n_legs_k: int
    n_legs_p: int
    is_multi: bool        # True if either side has >1 leg
    canonical_key: tuple


def run_backtest(
    cities: list[str],
    date: str,
    mode: str,                    # 'strict' | 'cvar'
    pair_size: int = PAIR_SIZE,
    safety_buffer: float = SAFETY_BUFFER,
    multi_only: bool = False,
    log_progress: bool = True,
) -> dict:
    """Iterate snapshots, enumerate portfolios at each one, dedupe within
    a city, score every passing portfolio against `mode`'s filter.
    Returns aggregate metrics + trade list.
    """
    trades: list[TradeRecord] = []
    raw_signal_count = 0
    n_snapshots_visited = 0
    # Per-city stats
    per_city: dict[str, int] = {c: 0 for c in cities}

    fee_per_contract = FEE_PER_CONTRACT
    # Each leg pays fee_per_contract on entry; hold-to-settle has no exit fee.
    # fee_per_pair depends on # legs.

    for city in cities:
        forecast_pdf = load_forecast_pdf(city, date) or {}
        if not forecast_pdf:
            if log_progress:
                print(f"  [{city}] no forecast PDF; skipping")
            continue
        joint = joint_kp_distribution(forecast_pdf, city)
        if not joint:
            joint = {(k, k): w for k, w in forecast_pdf.items()}

        snaps = list(iter_snapshots(city, date))
        if not snaps:
            continue
        # Subsample to stay under time budget; dedup makes this safe.
        if SNAPSHOT_STRIDE > 1:
            snaps = snaps[::SNAPSHOT_STRIDE]
        opened_keys: set[tuple] = set()

        for ts, venues in snaps:
            n_snapshots_visited += 1
            kbs = venues.get("kalshi", [])
            pbs = venues.get("polymarket", [])
            if not kbs or not pbs:
                continue
            uni = universe_for_snapshot(kbs, pbs)
            if not uni:
                continue
            k_legs_b, p_legs_b = build_legs(kbs, pbs, uni)
            if not k_legs_b or not p_legs_b:
                continue

            for k_sub, p_sub in enumerate_portfolios(k_legs_b, p_legs_b):
                raw_signal_count += 1
                n_k = len(k_sub)
                n_p = len(p_sub)
                if multi_only and n_k == 1 and n_p == 1:
                    continue

                # Quick prune: top-of-book min bound. Sum of best asks for
                # all legs at qty=1 — if even that exceeds the max-payout,
                # don't bother. Max payoff per pair = n_k + n_p.
                max_payoff = float(n_k + n_p)
                # Cost prune
                cost = 0.0
                fillable = True
                for leg, b in k_sub:
                    c = fill_leg_cost(b, leg.side, 1)
                    if c is None:
                        fillable = False
                        break
                    cost += c
                if not fillable:
                    continue
                for leg, b in p_sub:
                    c = fill_leg_cost(b, leg.side, 1)
                    if c is None:
                        fillable = False
                        break
                    cost += c
                if not fillable:
                    continue

                fee_per_pair = (n_k + n_p) * fee_per_contract  # entry only

                if cost + fee_per_pair >= max_payoff:
                    continue  # can't possibly profit

                # Compute payoff stats
                k_legs = [leg for leg, _ in k_sub]
                p_legs = [leg for leg, _ in p_sub]
                stats = evaluate_portfolio(k_legs, p_legs, cost, joint, fee_per_pair)
                if stats is None:
                    continue

                # Apply mode filter
                if mode == "strict":
                    # Worst-case-positive across the JOINT support
                    if stats["worst_case_pnl_per_pair"] <= safety_buffer:
                        continue
                elif mode == "cvar":
                    # q05 > cost AND E > cost + safety
                    if stats["q05_pnl_per_pair"] <= 0:
                        continue
                    if stats["expected_pnl_per_pair"] <= safety_buffer:
                        continue
                else:
                    raise ValueError(f"unknown mode {mode}")

                # Now check actual fillability at PAIR_SIZE
                actual_cost = 0.0
                fill_ok = True
                for leg, b in k_sub:
                    c = fill_leg_cost(b, leg.side, pair_size)
                    if c is None:
                        fill_ok = False
                        break
                    actual_cost += c
                if not fill_ok:
                    continue
                for leg, b in p_sub:
                    c = fill_leg_cost(b, leg.side, pair_size)
                    if c is None:
                        fill_ok = False
                        break
                    actual_cost += c
                if not fill_ok:
                    continue

                # Re-check filter at actual fill price (same pair-size)
                stats2 = evaluate_portfolio(k_legs, p_legs, actual_cost, joint, fee_per_pair)
                if stats2 is None:
                    continue
                if mode == "strict" and stats2["worst_case_pnl_per_pair"] <= safety_buffer:
                    continue
                if mode == "cvar":
                    if stats2["q05_pnl_per_pair"] <= 0:
                        continue
                    if stats2["expected_pnl_per_pair"] <= safety_buffer:
                        continue

                ckey = portfolio_canonical_key(city, k_sub, p_sub)
                if ckey in opened_keys:
                    continue
                opened_keys.add(ckey)

                rec = TradeRecord(
                    city=city, date=date, entry_ts=ts,
                    pair_size=pair_size,
                    k_legs=[
                        {"label": leg.label, "side": leg.side,
                         "key": leg.bracket_key,
                         "win_set": sorted(leg.win_set)}
                        for leg, _ in k_sub
                    ],
                    p_legs=[
                        {"label": leg.label, "side": leg.side,
                         "key": leg.bracket_key,
                         "win_set": sorted(leg.win_set)}
                        for leg, _ in p_sub
                    ],
                    cost_per_pair=actual_cost,
                    fee_per_pair=fee_per_pair,
                    worst_case_payoff=stats2["worst_case_payoff"],
                    expected_payoff=stats2["expected_payoff"],
                    q05_payoff=stats2["q05_payoff"],
                    q01_payoff=stats2["q01_payoff"],
                    worst_case_pnl=stats2["worst_case_pnl_per_pair"] * pair_size,
                    expected_pnl=stats2["expected_pnl_per_pair"] * pair_size,
                    q05_pnl=stats2["q05_pnl_per_pair"] * pair_size,
                    n_legs_k=n_k,
                    n_legs_p=n_p,
                    is_multi=(n_k > 1 or n_p > 1),
                    canonical_key=ckey,
                )
                trades.append(rec)
                per_city[city] += 1

    return _summarize(trades, raw_signal_count, n_snapshots_visited, per_city)


def _summarize(trades, raw_signal_count, n_snapshots, per_city) -> dict:
    if not trades:
        return {
            "raw_signal_count": raw_signal_count,
            "n_snapshots": n_snapshots,
            "n_trades": 0,
            "n_winners": 0,
            "win_rate": 0.0,
            "total_pnl_dollars": 0.0,
            "total_pnl_worst_case": 0.0,
            "total_pnl_q05": 0.0,
            "avg_pnl_per_trade": 0.0,
            "median_pnl_per_trade": 0.0,
            "sharpe_per_trade": 0.0,
            "max_drawdown_dollars": 0.0,
            "settled_n": 0,
            "settled_pnl_dollars": 0.0,
            "exit_reason_counts": {},
            "per_city": per_city,
            "trades": [],
        }

    pnls_exp = [t.expected_pnl for t in trades]
    pnls_worst = [t.worst_case_pnl for t in trades]
    pnls_q05 = [t.q05_pnl for t in trades]
    n = len(trades)
    n_winners = sum(1 for p in pnls_worst if p > 0)
    win_rate = n_winners / n
    per_pair_pnls = [t.expected_pnl / t.pair_size for t in trades]
    if len(per_pair_pnls) > 1:
        s = stdev(per_pair_pnls) or 1e-9
        sharpe = mean(per_pair_pnls) / s
    else:
        sharpe = 0.0

    sorted_pnls = sorted(zip([t.entry_ts for t in trades], pnls_exp),
                         key=lambda t: t[0])
    cum, peak, max_dd = 0.0, 0.0, 0.0
    for _, p in sorted_pnls:
        cum += p
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)

    multi_count = sum(1 for t in trades if t.is_multi)
    n_legs_dist: dict[tuple[int, int], int] = {}
    for t in trades:
        k = (t.n_legs_k, t.n_legs_p)
        n_legs_dist[k] = n_legs_dist.get(k, 0) + 1

    return {
        "raw_signal_count": raw_signal_count,
        "n_snapshots": n_snapshots,
        "n_trades": n,
        "n_winners": n_winners,
        "win_rate": round(win_rate, 4),
        "total_pnl_dollars": round(sum(pnls_exp), 4),
        "total_pnl_worst_case": round(sum(pnls_worst), 4),
        "total_pnl_q05": round(sum(pnls_q05), 4),
        "avg_pnl_per_trade": round(sum(pnls_exp) / n, 4),
        "median_pnl_per_trade": round(median(pnls_exp), 4),
        "sharpe_per_trade": round(sharpe, 4),
        "max_drawdown_dollars": round(max_dd, 4),
        "settled_n": 0,
        "settled_pnl_dollars": 0.0,
        "exit_reason_counts": {"hold_to_settle": n},
        "per_city": per_city,
        "n_multi_trades": multi_count,
        "n_legs_distribution": {f"{k[0]}x{k[1]}": v for k, v in sorted(n_legs_dist.items())},
        "trades": [_trade_to_dict(t) for t in trades],
    }


def _trade_to_dict(t: TradeRecord) -> dict:
    return {
        "city": t.city, "date": t.date, "entry_ts": t.entry_ts,
        "pair_size": t.pair_size,
        "n_legs_k": t.n_legs_k, "n_legs_p": t.n_legs_p,
        "is_multi": t.is_multi,
        "k_legs": t.k_legs, "p_legs": t.p_legs,
        "cost_per_pair": round(t.cost_per_pair, 4),
        "fee_per_pair": round(t.fee_per_pair, 4),
        "worst_case_payoff": round(t.worst_case_payoff, 4),
        "expected_payoff": round(t.expected_payoff, 4),
        "q05_payoff": round(t.q05_payoff, 4),
        "q01_payoff": round(t.q01_payoff, 4),
        "worst_case_pnl": round(t.worst_case_pnl, 4),
        "expected_pnl": round(t.expected_pnl, 4),
        "q05_pnl": round(t.q05_pnl, 4),
    }


def load_s3_v2_trades_universe() -> set[tuple]:
    """Load S3 v2 trade keys from `strategy_3_structural_arb/trades_v2.json`.
    Returns set of canonical keys: (city, ((k_ticker, k_side),), ((p_token, p_side),)).
    """
    p = ROOT / "research" / "strategy_3_structural_arb" / "trades_v2.json"
    if not p.exists():
        return set()
    data = json.loads(p.read_text())
    keys: set[tuple] = set()
    for tr in data:
        # trades_v2 stores 'kalshi_label' = ticker, 'poly_label' = first 16 of token
        # We cannot reconstruct the full token_id; we approximate by using
        # (ticker, side) for K and (poly_label_prefix, side) for P.
        k_ticker = tr.get("kalshi_label", "")
        k_side = tr.get("kalshi_side", "")
        p_label = tr.get("poly_label", "")
        p_side = tr.get("poly_side", "")
        keys.add((tr.get("city", ""),
                  ((k_ticker, k_side),),
                  ((p_label, p_side),)))
    return keys


def trade_in_s3_v2_universe(t: TradeRecord, s3_keys: set[tuple]) -> bool:
    """Approximate match: same city, single K-leg ticker+side, single P-leg
    token-prefix+side."""
    if t.n_legs_k != 1 or t.n_legs_p != 1:
        return False
    k_leg = t.k_legs[0]
    p_leg = t.p_legs[0]
    candidate = (t.city,
                 ((k_leg["key"], k_leg["side"]),),
                 ((p_leg["key"][:16], p_leg["side"]),))
    return candidate in s3_keys


# ── CLI ──────────────────────────────────────────────────────────────────


def main():
    out_dir = Path(__file__).resolve().parent
    print("=" * 70)
    print("Strategy 7 — Multi-bracket portfolio hedges")
    print("=" * 70)
    print(f"Cities: {CITIES}; Date: {DATE}; Pair size: {PAIR_SIZE}")
    print(f"Max legs/side: {MAX_LEGS_PER_SIDE}; Safety buffer: ${SAFETY_BUFFER}")
    print()

    iterations = []

    # ── v1: STRICT structural across joint K-P support ──
    print("--- v1: strict worst-case across joint K-P support ---")
    v1 = run_backtest(CITIES, DATE, mode="strict")
    print(f"  raw signals enumerated: {v1['raw_signal_count']}")
    print(f"  filled trades: {v1['n_trades']}  (multi: {v1.get('n_multi_trades', 0)})")
    print(f"  per-city: {v1['per_city']}")
    print(f"  n_legs_dist: {v1.get('n_legs_distribution', {})}")
    print(f"  total expected PnL: ${v1['total_pnl_dollars']:.2f}")
    print(f"  total worst-case PnL: ${v1['total_pnl_worst_case']:.2f}")
    print(f"  total q05 PnL: ${v1['total_pnl_q05']:.2f}")
    print()

    iterations.append(_iter_dict("v1", v1,
        "Strict worst-case structural multi-bracket hedge. Worst-case PnL/pair "
        ">= $0.05 over the JOINT (K, P) support (using `joint_kp_distribution` "
        "= NWS PDF × empirical Δ histogram). Allows up to 2 legs per venue. "
        "Hold-to-settle scoring; entry fees only."))

    # ── v2: CVaR (q05 > 0) + EV > cost + safety ──
    print("--- v2: CVaR-bounded EV (q05 > cost AND E > cost + 5¢) ---")
    v2 = run_backtest(CITIES, DATE, mode="cvar")
    print(f"  raw signals enumerated: {v2['raw_signal_count']}")
    print(f"  filled trades: {v2['n_trades']}  (multi: {v2.get('n_multi_trades', 0)})")
    print(f"  per-city: {v2['per_city']}")
    print(f"  n_legs_dist: {v2.get('n_legs_distribution', {})}")
    print(f"  total expected PnL: ${v2['total_pnl_dollars']:.2f}")
    print(f"  total worst-case PnL: ${v2['total_pnl_worst_case']:.2f}")
    print(f"  total q05 PnL: ${v2['total_pnl_q05']:.2f}")
    print()

    iterations.append(_iter_dict("v2", v2,
        "CVaR-bounded EV: q05 PnL/pair > 0 AND expected PnL/pair > $0.05. "
        "Loosens v1 to permit small-probability tail losses while keeping a "
        "positive-EV cushion. Otherwise same multi-bracket portfolio search."))

    # ── v3: NEW-ONLY (exclude reducible-to-S3-v2 pairs) ──
    print("--- v3: NEW-only — exclude single-leg portfolios in S3 v2's universe ---")
    s3_keys = load_s3_v2_trades_universe()
    print(f"  S3 v2 universe loaded: {len(s3_keys)} keys")
    # Reuse v2's selection but filter
    v2_trades = []
    for tdict in v2["trades"]:
        # Recover TradeRecord-lite for matching: only need the matcher's fields
        rec = _dict_to_record(tdict)
        if trade_in_s3_v2_universe(rec, s3_keys):
            continue
        v2_trades.append(tdict)
    # Also filter: keep ONLY genuinely multi-bracket trades (n_legs_k>1 OR n_legs_p>1)
    v3_trades = [t for t in v2_trades if t["is_multi"]]

    v3 = _summarize_dicts(v3_trades, v2["raw_signal_count"], v2["n_snapshots"], CITIES)
    print(f"  filled trades: {v3['n_trades']}  (multi: {v3.get('n_multi_trades', 0)})")
    print(f"  per-city: {v3['per_city']}")
    print(f"  n_legs_dist: {v3.get('n_legs_distribution', {})}")
    print(f"  total expected PnL: ${v3['total_pnl_dollars']:.2f}")
    print(f"  total worst-case PnL: ${v3['total_pnl_worst_case']:.2f}")
    print(f"  total q05 PnL: ${v3['total_pnl_q05']:.2f}")
    print()

    iterations.append(_iter_dict("v3", v3,
        "v2's selection minus any portfolio reducible to a single Kalshi leg + "
        "single Polymarket leg matching S3 v2's universe. Reports the *additive* "
        "trades available only via multi-bracket combinations."))

    # ── Save ──
    metrics = {
        "strategy_name": "multi_bracket_portfolio_hedges",
        "cities": CITIES,
        "date": DATE,
        "pair_size": PAIR_SIZE,
        "max_legs_per_side": MAX_LEGS_PER_SIDE,
        "fee_per_contract": FEE_PER_CONTRACT,
        "safety_buffer_per_pair": SAFETY_BUFFER,
        "iterations": iterations,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (out_dir / "trades_v1.json").write_text(json.dumps(v1["trades"], indent=2, default=str))
    (out_dir / "trades_v2.json").write_text(json.dumps(v2["trades"], indent=2, default=str))
    (out_dir / "trades_v3.json").write_text(json.dumps(v3_trades, indent=2, default=str))
    print(f"Wrote {out_dir / 'metrics.json'}")


def _iter_dict(version, v, notes):
    return {
        "version": version,
        "n_trades": v["n_trades"],
        "n_winners": v["n_winners"],
        "win_rate": v["win_rate"],
        "total_pnl_dollars": v["total_pnl_dollars"],
        "total_pnl_worst_case": v.get("total_pnl_worst_case", 0.0),
        "total_pnl_q05": v.get("total_pnl_q05", 0.0),
        "avg_pnl_per_trade": v["avg_pnl_per_trade"],
        "median_pnl_per_trade": v["median_pnl_per_trade"],
        "sharpe_per_trade": v["sharpe_per_trade"],
        "max_drawdown_dollars": v["max_drawdown_dollars"],
        "settled_n": v["settled_n"],
        "settled_pnl_dollars": v["settled_pnl_dollars"],
        "exit_reason_counts": v["exit_reason_counts"],
        "raw_signal_count": v.get("raw_signal_count", 0),
        "per_city": v["per_city"],
        "n_multi_trades": v.get("n_multi_trades", 0),
        "n_legs_distribution": v.get("n_legs_distribution", {}),
        "notes": notes,
    }


def _dict_to_record(td: dict) -> TradeRecord:
    """Reconstruct just enough of a TradeRecord for matching/summary."""
    return TradeRecord(
        city=td["city"], date=td["date"], entry_ts=td["entry_ts"],
        pair_size=td["pair_size"],
        k_legs=td["k_legs"], p_legs=td["p_legs"],
        cost_per_pair=td["cost_per_pair"], fee_per_pair=td["fee_per_pair"],
        worst_case_payoff=td["worst_case_payoff"],
        expected_payoff=td["expected_payoff"],
        q05_payoff=td["q05_payoff"], q01_payoff=td["q01_payoff"],
        worst_case_pnl=td["worst_case_pnl"],
        expected_pnl=td["expected_pnl"], q05_pnl=td["q05_pnl"],
        n_legs_k=td["n_legs_k"], n_legs_p=td["n_legs_p"],
        is_multi=td["is_multi"], canonical_key=tuple(),
    )


def _summarize_dicts(trade_dicts, raw_signal_count, n_snapshots, cities) -> dict:
    if not trade_dicts:
        return {
            "raw_signal_count": raw_signal_count, "n_snapshots": n_snapshots,
            "n_trades": 0, "n_winners": 0, "win_rate": 0.0,
            "total_pnl_dollars": 0.0, "total_pnl_worst_case": 0.0,
            "total_pnl_q05": 0.0, "avg_pnl_per_trade": 0.0,
            "median_pnl_per_trade": 0.0, "sharpe_per_trade": 0.0,
            "max_drawdown_dollars": 0.0, "settled_n": 0,
            "settled_pnl_dollars": 0.0,
            "exit_reason_counts": {}, "per_city": {c: 0 for c in cities},
            "n_multi_trades": 0, "n_legs_distribution": {},
            "trades": [],
        }
    per_city = {c: 0 for c in cities}
    n_legs_dist: dict[str, int] = {}
    for td in trade_dicts:
        per_city[td["city"]] = per_city.get(td["city"], 0) + 1
        k = f"{td['n_legs_k']}x{td['n_legs_p']}"
        n_legs_dist[k] = n_legs_dist.get(k, 0) + 1
    pnls_exp = [td["expected_pnl"] for td in trade_dicts]
    pnls_worst = [td["worst_case_pnl"] for td in trade_dicts]
    pnls_q05 = [td["q05_pnl"] for td in trade_dicts]
    n = len(trade_dicts)
    n_winners = sum(1 for p in pnls_worst if p > 0)
    per_pair = [td["expected_pnl"] / td["pair_size"] for td in trade_dicts]
    if len(per_pair) > 1:
        s = stdev(per_pair) or 1e-9
        sharpe = mean(per_pair) / s
    else:
        sharpe = 0.0
    sorted_pnls = sorted([(td["entry_ts"], td["expected_pnl"]) for td in trade_dicts])
    cum, peak, max_dd = 0.0, 0.0, 0.0
    for _, p in sorted_pnls:
        cum += p
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)
    return {
        "raw_signal_count": raw_signal_count, "n_snapshots": n_snapshots,
        "n_trades": n, "n_winners": n_winners,
        "win_rate": round(n_winners / n, 4),
        "total_pnl_dollars": round(sum(pnls_exp), 4),
        "total_pnl_worst_case": round(sum(pnls_worst), 4),
        "total_pnl_q05": round(sum(pnls_q05), 4),
        "avg_pnl_per_trade": round(sum(pnls_exp) / n, 4),
        "median_pnl_per_trade": round(median(pnls_exp), 4),
        "sharpe_per_trade": round(sharpe, 4),
        "max_drawdown_dollars": round(max_dd, 4),
        "settled_n": 0, "settled_pnl_dollars": 0.0,
        "exit_reason_counts": {"hold_to_settle": n},
        "per_city": per_city,
        "n_multi_trades": sum(1 for td in trade_dicts if td["is_multi"]),
        "n_legs_distribution": dict(sorted(n_legs_dist.items())),
        "trades": trade_dicts,
    }


if __name__ == "__main__":
    main()
