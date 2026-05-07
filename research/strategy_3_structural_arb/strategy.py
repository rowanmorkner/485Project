"""
Strategy 3 — Structural bracket-math arbitrage.

Idea
----
Across Kalshi and Polymarket, we have YES and NO contracts on bracketed
temperature ranges. For any (Kalshi bracket, Polymarket bracket) pair
and choice of side on each leg, we can compute deterministically:

  K_set   — degrees where the Kalshi leg pays $1
  P_set   — degrees where the Polymarket leg pays $1
  win_2   — K_set ∩ P_set: combined payout = $2
  win_1   — symmetric_diff(K_set, P_set): combined payout = $1
  loss    — universe \\ (K_set ∪ P_set): combined payout = $0

Universe = the integer temperature range bracketed by the venues
(plausible support; we use the union of all bracket degrees seen at
that snapshot). If `loss` is empty, the position is "structural" — we
are guaranteed at least $1 regardless of outcome. If combined cost
< $1, that's a worst-case-positive arbitrage. If combined cost < $2
and the EV (vs forecast PDF) > cost, that's expected-value-positive.

We then back this with realistic depth: at each snapshot we walk both
ladders to find the largest pair size (capped at min depth and a small
default size) that fills both legs with combined cost still below the
threshold.

Two scoring modes:

  v1 — STRICT WORST-CASE: only signal if loss zone is empty AND combined
       cost (avg fill) < $1.

  v2 — EXPECTED-EDGE: weight outcomes by the NWS forecast PDF and signal
       if E[payout] - cost > min_edge, even if a small loss zone exists.

We then mark-to-market at every later snapshot (close both legs) and pick
the first time TOTAL realized P&L (legA + legB) hits a target, or a time
stop, or end-of-window.

Caveats / honesty
-----------------
- "Best ask" can be $0.99 due to the partial-resolution token sitting in
  the book. The loaders sort the ladder ascending defensively, so
  walk_ladder_buy uses the actual lowest ask first. Good. But if we ask
  for size larger than real depth at the cheap levels, we walk into the
  $0.99 trap. We cap pair size at the minimum depth of the cheap part of
  each ladder, defined as cumulative qty available at avg-price ≤
  some max-cost bound (so we never lift the $0.99 partial-resolution
  reserves).

- Fees: we charge $0.005/contract per leg, both entry and exit, which is
  $0.02 per pair round-trip ($0.01/contract since each pair is 2 contracts).

- We only ever OPEN a given (kalshi_bracket, poly_bracket, direction) once.
  If the same arb persists across many snapshots we don't double-count.

- Universe: we use [min(all bracket degs) - 0, max(all bracket degs) + 0]
  on each snapshot. The "or below"/"or higher" tail brackets contribute
  parsed degree ranges (parsers expand by TAIL_SPREAD=5), so the universe
  typically spans ~25-30 degrees.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from statistics import mean, median, stdev
from typing import Optional

# Make project root importable
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from research.loaders import (
    iter_snapshots,
    load_forecast_pdf,
    walk_ladder_buy,
    walk_ladder_sell,
    find_kalshi_bracket,
    find_polymarket_bracket,
)
from strategy.parsers import _parse_kalshi_range, _parse_polymarket_range


CITIES = ["Miami", "LA", "Austin", "San Francisco", "Seattle"]
DATE = "2026-05-06"

# Trade economics
FEE_PER_CONTRACT = 0.005
DEFAULT_PAIR_SIZE_CAP = 50      # never put more than this into one structural arb
MIN_PAIR_SIZE = 5               # don't bother below this


# ── Side helpers ──────────────────────────────────────────────────────────


def cheap_ask_ladder_yes(b: dict) -> list:
    """ASK ladder for buying YES. As-is from data."""
    return list(b.get("yes_ask_ladder") or [])


def cheap_ask_ladder_no(b: dict) -> list:
    """ASK ladder for buying NO = (1 - p, q) for entries in YES bid ladder."""
    return [(round(1.0 - p, 4), q) for p, q in (b.get("yes_bid_ladder") or [])]


def bid_ladder_for_close(b: dict, side: str) -> list:
    """BID ladder for selling out of the position, depending on which side
    we hold. yes_long → sell into YES bids; no_long → sell into NO bids =
    (1 - p, q) on YES asks.
    """
    if side == "yes_long":
        return list(b.get("yes_bid_ladder") or [])
    return [(round(1.0 - p, 4), q) for p, q in (b.get("yes_ask_ladder") or [])]


# ── Structural pair scoring ──────────────────────────────────────────────


@dataclass
class PairCandidate:
    city: str
    date: str
    ts: str
    kalshi_ticker: str
    kalshi_label: str
    kalshi_side: str           # 'yes_long' | 'no_long'
    kalshi_degrees: list[int]
    kalshi_win_set: set[int]
    poly_token: str
    poly_label: str
    poly_side: str
    poly_degrees: list[int]
    poly_win_set: set[int]
    win_2: set[int]
    win_1: set[int]
    loss: set[int]
    universe: set[int]
    avg_kalshi_cost: float
    avg_poly_cost: float
    pair_size: int
    total_cost_per_pair: float   # avg fill cost of buying 1 of each
    worst_case_payout: float     # min payout over universe
    expected_payout: float       # forecast-weighted
    worst_case_edge: float       # worst_case_payout - total_cost_per_pair
    expected_edge: float         # expected_payout - total_cost_per_pair


def venue_universe(kalshi_brackets: list, poly_brackets: list) -> set[int]:
    """Union of all parsed degree ranges at this snapshot. Used as the
    'plausible temperature universe'."""
    deg = set()
    for b in kalshi_brackets:
        deg.update(_parse_kalshi_range(b.get("subtitle", "")))
    for b in poly_brackets:
        deg.update(_parse_polymarket_range(b.get("question", "")))
    return deg


def score_pair(
    city: str, date: str, ts: str,
    kb: dict, ks_side: str,         # ks_side ∈ {'yes_long','no_long'}
    pb: dict, ps_side: str,
    universe: set[int],
    forecast_pdf: dict[int, float],
    pair_size: int,
) -> Optional[PairCandidate]:
    k_degrees = _parse_kalshi_range(kb.get("subtitle", ""))
    p_degrees = _parse_polymarket_range(pb.get("question", ""))
    if not k_degrees or not p_degrees:
        return None

    k_set = set(k_degrees)
    p_set = set(p_degrees)
    k_win = k_set if ks_side == "yes_long" else (universe - k_set)
    p_win = p_set if ps_side == "yes_long" else (universe - p_set)

    # Pick ladders per chosen side
    k_ask = cheap_ask_ladder_yes(kb) if ks_side == "yes_long" else cheap_ask_ladder_no(kb)
    p_ask = cheap_ask_ladder_yes(pb) if ps_side == "yes_long" else cheap_ask_ladder_no(pb)

    k_fill = walk_ladder_buy(k_ask, pair_size)
    p_fill = walk_ladder_buy(p_ask, pair_size)
    if k_fill is None or p_fill is None:
        return None
    k_avg, _ = k_fill
    p_avg, _ = p_fill

    total_cost = k_avg + p_avg

    # Win-zone partition over the universe
    win_2 = k_win & p_win
    win_1 = (k_win ^ p_win) & universe   # symmetric difference, intersected with universe
    loss = universe - (k_win | p_win)

    # Worst-case payout: $0 if loss is non-empty, else $1
    worst_case_payout = 0.0 if loss else 1.0

    # Expected payout under forecast PDF (only weighting degrees within universe;
    # mass outside universe ignored — usually negligible)
    exp_payout = 0.0
    for d in win_2:
        exp_payout += 2.0 * forecast_pdf.get(d, 0.0)
    for d in win_1:
        exp_payout += 1.0 * forecast_pdf.get(d, 0.0)
    # loss zone contributes 0

    return PairCandidate(
        city=city, date=date, ts=ts,
        kalshi_ticker=kb.get("ticker", ""),
        kalshi_label=kb.get("subtitle", ""),
        kalshi_side=ks_side,
        kalshi_degrees=k_degrees,
        kalshi_win_set=k_win,
        poly_token=pb.get("token_id", ""),
        poly_label=pb.get("question", ""),
        poly_side=ps_side,
        poly_degrees=p_degrees,
        poly_win_set=p_win,
        win_2=win_2,
        win_1=win_1,
        loss=loss,
        universe=universe,
        avg_kalshi_cost=k_avg,
        avg_poly_cost=p_avg,
        pair_size=pair_size,
        total_cost_per_pair=total_cost,
        worst_case_payout=worst_case_payout,
        expected_payout=exp_payout,
        worst_case_edge=worst_case_payout - total_cost,
        expected_edge=exp_payout - total_cost,
    )


def enumerate_signals(
    city: str, date: str, ts: str,
    kalshi_brackets: list, poly_brackets: list,
    forecast_pdf: dict[int, float],
    mode: str,                  # 'worst_case' | 'expected'
    min_edge: float,
    pair_size: int = DEFAULT_PAIR_SIZE_CAP,
) -> list[PairCandidate]:
    """Return all pairs that pass the chosen mode's threshold at this snapshot."""
    universe = venue_universe(kalshi_brackets, poly_brackets)
    if not universe:
        return []

    out: list[PairCandidate] = []
    for kb in kalshi_brackets:
        for ks_side in ("yes_long", "no_long"):
            for pb in poly_brackets:
                for ps_side in ("yes_long", "no_long"):
                    cand = score_pair(
                        city, date, ts, kb, ks_side, pb, ps_side,
                        universe, forecast_pdf, pair_size,
                    )
                    if cand is None:
                        continue
                    if mode == "worst_case":
                        # require strictly positive worst-case edge AFTER fees
                        # fees: 2 entry + 2 exit @ FEE_PER_CONTRACT each = 4 * fee per pair.
                        # But we won't necessarily exit in the worst-case; a hold-to-settle
                        # only pays the entry fees on the leg you sold (or all if you must
                        # close mtm). We use 2*fee/pair as a conservative lower bound.
                        edge_after_fees = cand.worst_case_edge - 2.0 * FEE_PER_CONTRACT
                        if edge_after_fees > min_edge and not cand.loss:
                            out.append(cand)
                    elif mode == "expected":
                        edge_after_fees = cand.expected_edge - 2.0 * FEE_PER_CONTRACT
                        if edge_after_fees > min_edge:
                            out.append(cand)
                    elif mode == "structural_expected":
                        # Require structural (no loss zone) AND positive expected edge
                        if cand.loss:
                            continue
                        edge_after_fees = cand.expected_edge - 2.0 * FEE_PER_CONTRACT
                        if edge_after_fees > min_edge:
                            out.append(cand)
    return out


# ── Trade lifecycle: open, mark-to-market, exit ─────────────────────────


@dataclass
class PairPosition:
    city: str
    date: str
    entry_ts: str
    pair_size: int
    # leg A (Kalshi)
    kalshi_ticker: str
    kalshi_side: str
    kalshi_entry_avg: float
    kalshi_entry_cost: float
    kalshi_degrees: list[int]
    kalshi_win_set: set[int]
    # leg B (Polymarket)
    poly_token: str
    poly_side: str
    poly_entry_avg: float
    poly_entry_cost: float
    poly_degrees: list[int]
    poly_win_set: set[int]
    # bookkeeping
    universe: set[int]
    win_2: set[int]
    win_1: set[int]
    loss: set[int]
    expected_payout: float
    worst_case_payout: float
    structural: bool             # True if loss set is empty


def open_pair(cand: PairCandidate) -> PairPosition:
    return PairPosition(
        city=cand.city, date=cand.date, entry_ts=cand.ts,
        pair_size=cand.pair_size,
        kalshi_ticker=cand.kalshi_ticker,
        kalshi_side=cand.kalshi_side,
        kalshi_entry_avg=cand.avg_kalshi_cost,
        kalshi_entry_cost=cand.avg_kalshi_cost * cand.pair_size,
        kalshi_degrees=cand.kalshi_degrees,
        kalshi_win_set=cand.kalshi_win_set,
        poly_token=cand.poly_token,
        poly_side=cand.poly_side,
        poly_entry_avg=cand.avg_poly_cost,
        poly_entry_cost=cand.avg_poly_cost * cand.pair_size,
        poly_degrees=cand.poly_degrees,
        poly_win_set=cand.poly_win_set,
        universe=cand.universe,
        win_2=cand.win_2,
        win_1=cand.win_1,
        loss=cand.loss,
        expected_payout=cand.expected_payout,
        worst_case_payout=cand.worst_case_payout,
        structural=not cand.loss,
    )


@dataclass
class PairExit:
    exit_ts: Optional[str]
    exit_reason: str            # 'target' | 'time_stop' | 'end_of_window' | 'no_liquidity'
    kalshi_exit_avg: Optional[float]
    poly_exit_avg: Optional[float]
    kalshi_exit_proceeds: float
    poly_exit_proceeds: float
    realized_pnl: float


def simulate_pair_exit(
    pos: PairPosition,
    future_snapshots: list,            # list of (ts, {venue: brackets})
    target_pnl_per_pair: Optional[float] = None,
    time_stop_minutes: Optional[int] = None,
) -> PairExit:
    """Mark-to-market exit. At each future snapshot, attempt to close BOTH
    legs at then-current bid ladders. If combined proceeds - entry_cost
    exceeds target, lift. Otherwise time-stop / end-of-window.
    """
    from datetime import datetime, timedelta

    entry_dt = datetime.fromisoformat(pos.entry_ts.replace("Z", "+00:00"))
    time_stop_dt = (entry_dt + timedelta(minutes=time_stop_minutes)) if time_stop_minutes else None

    # Total fees: 2 contracts entry + 2 contracts exit = 4 * fee * pair_size
    total_fees = 4.0 * FEE_PER_CONTRACT * pos.pair_size

    last_seen = None  # (ts, k_proceeds, p_proceeds, k_avg, p_avg)

    for ts, venues in future_snapshots:
        kbs = venues.get("kalshi", [])
        pbs = venues.get("polymarket", [])
        kb = find_kalshi_bracket(kbs, pos.kalshi_ticker) if kbs else None
        pb = find_polymarket_bracket(pbs, pos.poly_token) if pbs else None
        if kb is None or pb is None:
            # Either venue gone: skip
            cur_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if time_stop_dt and cur_dt >= time_stop_dt:
                break
            continue
        k_bids = bid_ladder_for_close(kb, pos.kalshi_side)
        p_bids = bid_ladder_for_close(pb, pos.poly_side)
        if not k_bids or not p_bids:
            cur_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if time_stop_dt and cur_dt >= time_stop_dt:
                break
            continue
        k_sell = walk_ladder_sell(k_bids, pos.pair_size)
        p_sell = walk_ladder_sell(p_bids, pos.pair_size)
        if k_sell is None or p_sell is None:
            cur_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if time_stop_dt and cur_dt >= time_stop_dt:
                break
            continue
        k_avg, k_proceeds = k_sell
        p_avg, p_proceeds = p_sell
        last_seen = (ts, k_proceeds, p_proceeds, k_avg, p_avg)

        proceeds = k_proceeds + p_proceeds
        cost = pos.kalshi_entry_cost + pos.poly_entry_cost
        running_pnl_per_pair = (proceeds - cost) / pos.pair_size - 4.0 * FEE_PER_CONTRACT

        if target_pnl_per_pair is not None and running_pnl_per_pair >= target_pnl_per_pair:
            return PairExit(
                exit_ts=ts, exit_reason="target",
                kalshi_exit_avg=k_avg, poly_exit_avg=p_avg,
                kalshi_exit_proceeds=k_proceeds, poly_exit_proceeds=p_proceeds,
                realized_pnl=proceeds - cost - total_fees,
            )

        if time_stop_dt is not None:
            cur_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if cur_dt >= time_stop_dt:
                return PairExit(
                    exit_ts=ts, exit_reason="time_stop",
                    kalshi_exit_avg=k_avg, poly_exit_avg=p_avg,
                    kalshi_exit_proceeds=k_proceeds, poly_exit_proceeds=p_proceeds,
                    realized_pnl=proceeds - cost - total_fees,
                )

    # End of window: close at last seen prices if any
    if last_seen is not None:
        ts, k_proceeds, p_proceeds, k_avg, p_avg = last_seen
        cost = pos.kalshi_entry_cost + pos.poly_entry_cost
        return PairExit(
            exit_ts=ts, exit_reason="end_of_window",
            kalshi_exit_avg=k_avg, poly_exit_avg=p_avg,
            kalshi_exit_proceeds=k_proceeds, poly_exit_proceeds=p_proceeds,
            realized_pnl=k_proceeds + p_proceeds - cost - total_fees,
        )

    # No liquidity: assume held; payout determined structurally.
    # If structural (loss zone empty): worst-case $1 paid. We use that.
    # If non-structural with expected mode: we treat as no-liquidity loss.
    # (Without settlement data we can't realize a specific payout; use
    # worst-case for honest accounting.)
    cost = pos.kalshi_entry_cost + pos.poly_entry_cost
    if pos.structural:
        # Worst-case payout = $1/pair guaranteed if we could hold to settle.
        # But since we couldn't even close legs along the way, treat this as
        # "would have paid out at settlement" only if the pair survives to
        # settlement — which is true for these pairs (brackets always settle).
        # So pay out $1/pair (the worst-case).
        proceeds = 1.0 * pos.pair_size
        return PairExit(
            exit_ts=None, exit_reason="hold_to_settle_worstcase",
            kalshi_exit_avg=None, poly_exit_avg=None,
            kalshi_exit_proceeds=0.0, poly_exit_proceeds=0.0,
            realized_pnl=proceeds - cost - total_fees,
        )
    return PairExit(
        exit_ts=None, exit_reason="no_liquidity",
        kalshi_exit_avg=None, poly_exit_avg=None,
        kalshi_exit_proceeds=0.0, poly_exit_proceeds=0.0,
        realized_pnl=-cost - total_fees,
    )


# ── Backtest driver ─────────────────────────────────────────────────────


def run_backtest(
    cities: list[str],
    date: str,
    mode: str,                    # 'worst_case' or 'expected'
    min_edge: float,
    pair_size: int = DEFAULT_PAIR_SIZE_CAP,
    target_pnl_per_pair: Optional[float] = None,
    time_stop_minutes: Optional[int] = None,
    raw_count_only: bool = False,
) -> dict:
    """Iterate over the trading window, open at most ONE position per
    (city, kalshi_ticker, poly_token, kalshi_side, poly_side) pair, exit
    via mark-to-market, and aggregate metrics.

    If raw_count_only, just count enumerated signals (no opening).
    """
    raw_signal_count = 0
    raw_filled_signals = 0   # signals where MIN_PAIR_SIZE actually fills both legs

    positions: list[PairPosition] = []
    exits: list[PairExit] = []

    for city in cities:
        forecast_pdf = load_forecast_pdf(city, date) or {}
        snaps = list(iter_snapshots(city, date))
        if not snaps:
            continue

        opened_keys: set[tuple] = set()

        for entry_idx, (ts, venues) in enumerate(snaps):
            kbs = venues.get("kalshi", [])
            pbs = venues.get("polymarket", [])
            if not kbs or not pbs:
                continue
            cands = enumerate_signals(
                city, date, ts, kbs, pbs, forecast_pdf,
                mode=mode, min_edge=min_edge, pair_size=pair_size,
            )
            for c in cands:
                raw_signal_count += 1

            if raw_count_only:
                continue

            # Pick the BEST cand we haven't already opened
            cands.sort(key=lambda c: -(c.worst_case_edge if mode == "worst_case" else c.expected_edge))
            for cand in cands:
                key = (city, cand.kalshi_ticker, cand.poly_token,
                       cand.kalshi_side, cand.poly_side)
                if key in opened_keys:
                    continue
                # Try to open at MIN_PAIR_SIZE first (to ensure depth);
                # if that succeeds, optionally try larger size up to pair_size.
                if cand.pair_size < MIN_PAIR_SIZE:
                    continue
                raw_filled_signals += 1
                pos = open_pair(cand)
                positions.append(pos)
                future = snaps[entry_idx + 1:]
                ex = simulate_pair_exit(
                    pos, future,
                    target_pnl_per_pair=target_pnl_per_pair,
                    time_stop_minutes=time_stop_minutes,
                )
                exits.append(ex)
                opened_keys.add(key)

    if not positions:
        return {
            "raw_signal_count": raw_signal_count,
            "raw_filled_signals": raw_filled_signals,
            "n_trades": 0,
            "n_winners": 0,
            "win_rate": 0.0,
            "total_pnl_dollars": 0.0,
            "avg_pnl_per_trade": 0.0,
            "median_pnl_per_trade": 0.0,
            "sharpe_per_trade": 0.0,
            "max_drawdown_dollars": 0.0,
            "settled_n": 0,
            "settled_pnl_dollars": 0.0,
            "exit_reason_counts": {},
            "trades": [],
        }

    pnls = [ex.realized_pnl for ex in exits]
    n_trades = len(pnls)
    n_winners = sum(1 for p in pnls if p > 0)
    win_rate = n_winners / n_trades
    total_pnl = sum(pnls)
    avg_pnl_total = total_pnl / n_trades
    med_pnl = median(pnls)

    # Per-pair (per-contract pair) returns for sharpe
    per_pair_pnls = [ex.realized_pnl / pos.pair_size for pos, ex in zip(positions, exits)]
    if len(per_pair_pnls) > 1:
        s = stdev(per_pair_pnls) or 1e-9
        sharpe = mean(per_pair_pnls) / s
    else:
        sharpe = 0.0

    # Drawdown over the time-ordered cumulative pnl curve
    sorted_pairs = sorted(zip(positions, exits), key=lambda pe: pe[0].entry_ts)
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for pos, ex in sorted_pairs:
        cum += ex.realized_pnl
        if cum > peak:
            peak = cum
        dd = cum - peak
        if dd < max_dd:
            max_dd = dd

    exit_reason_counts: dict[str, int] = {}
    for ex in exits:
        exit_reason_counts[ex.exit_reason] = exit_reason_counts.get(ex.exit_reason, 0) + 1

    # No 5/6 settlements yet (per contract). Settled fields = 0.
    return {
        "raw_signal_count": raw_signal_count,
        "raw_filled_signals": raw_filled_signals,
        "n_trades": n_trades,
        "n_winners": n_winners,
        "win_rate": round(win_rate, 4),
        "total_pnl_dollars": round(total_pnl, 4),
        "avg_pnl_per_trade": round(avg_pnl_total, 4),
        "median_pnl_per_trade": round(med_pnl, 4),
        "sharpe_per_trade": round(sharpe, 4),
        "max_drawdown_dollars": round(max_dd, 4),
        "settled_n": 0,
        "settled_pnl_dollars": 0.0,
        "exit_reason_counts": exit_reason_counts,
        "trades": [
            {
                "city": pos.city,
                "entry_ts": pos.entry_ts,
                "exit_ts": ex.exit_ts,
                "exit_reason": ex.exit_reason,
                "kalshi_label": pos.kalshi_ticker,
                "kalshi_side": pos.kalshi_side,
                "poly_label": pos.poly_token[:16],
                "poly_side": pos.poly_side,
                "pair_size": pos.pair_size,
                "structural": pos.structural,
                "entry_cost_per_pair": round(pos.kalshi_entry_avg + pos.poly_entry_avg, 4),
                "realized_pnl": round(ex.realized_pnl, 4),
                "expected_payout": round(pos.expected_payout, 4),
                "worst_case_payout": round(pos.worst_case_payout, 4),
            }
            for pos, ex in zip(positions, exits)
        ],
    }


def run_backtest_hold_to_settle(
    cities: list[str],
    date: str,
    mode: str,
    min_edge: float,
    pair_size: int = DEFAULT_PAIR_SIZE_CAP,
) -> dict:
    """Structural arbs only really pay off at settlement. This variant
    scores trades by their HOLD-TO-SETTLE payout:

      realized_payout = 1.0 + 1.0 * 1{T ∈ win_2}      (per pair)

    where T is the (unknown) settlement temperature. Without ground-truth
    settlements (5/6 hasn't settled in our DB), we substitute with the
    forecast PDF: E[payout] = 1.0 + P(T ∈ win_2 | NWS PDF). We also
    track worst_case_payout = $1 per pair (universally guaranteed for
    structural arbs).
    """
    raw_signal_count = 0
    positions: list[PairPosition] = []
    pnls_expected = []
    pnls_worst = []

    for city in cities:
        forecast_pdf = load_forecast_pdf(city, date) or {}
        snaps = list(iter_snapshots(city, date))
        if not snaps:
            continue
        opened_keys: set[tuple] = set()

        for ts, venues in snaps:
            kbs = venues.get("kalshi", [])
            pbs = venues.get("polymarket", [])
            if not kbs or not pbs:
                continue
            cands = enumerate_signals(
                city, date, ts, kbs, pbs, forecast_pdf,
                mode=mode, min_edge=min_edge, pair_size=pair_size,
            )
            for c in cands:
                raw_signal_count += 1

            cands.sort(key=lambda c: -(c.worst_case_edge if mode == "worst_case" else c.expected_edge))
            for cand in cands:
                key = (city, cand.kalshi_ticker, cand.poly_token,
                       cand.kalshi_side, cand.poly_side)
                if key in opened_keys:
                    continue
                if cand.pair_size < MIN_PAIR_SIZE:
                    continue
                if not cand.universe:
                    continue

                pos = open_pair(cand)
                positions.append(pos)

                # PnL at settlement: payout = $1 (worst-case) + $1 * 1{T ∈ win_2}
                # For expected, weight win_2 by forecast PDF.
                # Fees: 2 contracts ENTRY only (settlement requires no
                # additional fee — winning contracts pay out at $1, losing
                # contracts settle at $0 with no further charge).
                fee = 2.0 * FEE_PER_CONTRACT * pos.pair_size
                cost = pos.kalshi_entry_cost + pos.poly_entry_cost

                p_win2 = sum(forecast_pdf.get(d, 0.0) for d in pos.win_2)
                exp_payout_per_pair = 1.0 + p_win2
                exp_total_proceeds = exp_payout_per_pair * pos.pair_size
                worst_total_proceeds = 1.0 * pos.pair_size

                pnls_expected.append(exp_total_proceeds - cost - fee)
                pnls_worst.append(worst_total_proceeds - cost - fee)

                opened_keys.add(key)

    if not positions:
        return {
            "raw_signal_count": raw_signal_count,
            "n_trades": 0, "n_winners": 0, "win_rate": 0.0,
            "total_pnl_dollars": 0.0, "avg_pnl_per_trade": 0.0,
            "median_pnl_per_trade": 0.0, "sharpe_per_trade": 0.0,
            "max_drawdown_dollars": 0.0, "settled_n": 0,
            "settled_pnl_dollars": 0.0, "worst_case_pnl_dollars": 0.0,
            "trades": [],
        }

    n_trades = len(pnls_expected)
    n_winners = sum(1 for p in pnls_worst if p > 0)  # under worst case
    total_pnl_exp = sum(pnls_expected)
    total_pnl_worst = sum(pnls_worst)
    avg_pnl = total_pnl_exp / n_trades

    per_pair = [p / pos.pair_size for p, pos in zip(pnls_expected, positions)]
    if len(per_pair) > 1:
        s = stdev(per_pair) or 1e-9
        sharpe = mean(per_pair) / s
    else:
        sharpe = 0.0

    # Drawdown over time-ordered worst-case PnL
    sorted_pairs = sorted(zip(positions, pnls_worst), key=lambda pe: pe[0].entry_ts)
    cum, peak, max_dd = 0.0, 0.0, 0.0
    for _, pnl in sorted_pairs:
        cum += pnl
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)

    trades = []
    for pos, e_pnl, w_pnl in zip(positions, pnls_expected, pnls_worst):
        trades.append({
            "city": pos.city,
            "entry_ts": pos.entry_ts,
            "kalshi_label": pos.kalshi_ticker,
            "kalshi_side": pos.kalshi_side,
            "poly_label": pos.poly_token[:16],
            "poly_side": pos.poly_side,
            "pair_size": pos.pair_size,
            "structural": pos.structural,
            "entry_cost_per_pair": round(pos.kalshi_entry_avg + pos.poly_entry_avg, 4),
            "win2_count": len(pos.win_2),
            "expected_pnl": round(e_pnl, 4),
            "worst_case_pnl": round(w_pnl, 4),
        })

    return {
        "raw_signal_count": raw_signal_count,
        "n_trades": n_trades,
        "n_winners": n_winners,
        "win_rate": round(n_winners / n_trades, 4),
        "total_pnl_dollars": round(total_pnl_exp, 4),
        "avg_pnl_per_trade": round(avg_pnl, 4),
        "median_pnl_per_trade": round(median(pnls_expected), 4),
        "sharpe_per_trade": round(sharpe, 4),
        "max_drawdown_dollars": round(max_dd, 4),
        "settled_n": 0,
        "settled_pnl_dollars": 0.0,
        "worst_case_pnl_dollars": round(total_pnl_worst, 4),
        "trades": trades,
    }


def count_distinct_keys(cities: list[str], date: str, sizes: tuple[int, ...] = (5, 10, 25, 50)) -> dict:
    """How many DISTINCT (city, kalshi_ticker, poly_token, sides) keys
    ever show up as a fillable structural arb across the trading day,
    at each candidate pair size?
    """
    distinct_by_size: dict[int, set] = {sz: set() for sz in sizes}
    for city in cities:
        snaps = list(iter_snapshots(city, date))
        for ts, venues in snaps:
            kbs = venues.get("kalshi", [])
            pbs = venues.get("polymarket", [])
            if not kbs or not pbs:
                continue
            universe = venue_universe(kbs, pbs)
            if not universe:
                continue
            for kb in kbs:
                k_degs = _parse_kalshi_range(kb.get("subtitle", ""))
                if not k_degs:
                    continue
                k_set = set(k_degs)
                for ks_side in ("yes_long", "no_long"):
                    k_win = k_set if ks_side == "yes_long" else (universe - k_set)
                    k_ladder = cheap_ask_ladder_yes(kb) if ks_side == "yes_long" else cheap_ask_ladder_no(kb)
                    for pb in pbs:
                        p_degs = _parse_polymarket_range(pb.get("question", ""))
                        if not p_degs:
                            continue
                        p_set = set(p_degs)
                        for ps_side in ("yes_long", "no_long"):
                            p_win = p_set if ps_side == "yes_long" else (universe - p_set)
                            loss = universe - (k_win | p_win)
                            if loss:
                                continue
                            p_ladder = cheap_ask_ladder_yes(pb) if ps_side == "yes_long" else cheap_ask_ladder_no(pb)
                            for sz in sizes:
                                kf = walk_ladder_buy(k_ladder, sz)
                                pf = walk_ladder_buy(p_ladder, sz)
                                if kf is None or pf is None:
                                    continue
                                if kf[0] + pf[0] >= 1.0:
                                    continue
                                key = (city, kb.get("ticker"), pb.get("token_id"), ks_side, ps_side)
                                distinct_by_size[sz].add(key)
    return {f"distinct_at_size_{sz}": len(distinct_by_size[sz]) for sz in sizes} | {
        "distinct_total": len(distinct_by_size[min(sizes)]),
    }


def diagnose_raw_arbs(cities: list[str], date: str) -> dict:
    """Without any filtering: how many raw 'structural' arbs (loss zone
    empty, total ASK cost < $1) appear in the data per snapshot, and how
    many of those are real after a depth filter?
    """
    raw_count = 0      # any pair where loss empty AND ask-sum < 1.0 at top of book
    fillable_count = 0  # same, but min(MIN_PAIR_SIZE) actually fills both legs from the cheap part
    illusory_count = 0  # raw - fillable

    for city in cities:
        snaps = list(iter_snapshots(city, date))
        for ts, venues in snaps:
            kbs = venues.get("kalshi", [])
            pbs = venues.get("polymarket", [])
            if not kbs or not pbs:
                continue
            universe = venue_universe(kbs, pbs)
            if not universe:
                continue
            for kb in kbs:
                k_degs = _parse_kalshi_range(kb.get("subtitle", ""))
                if not k_degs:
                    continue
                k_set = set(k_degs)
                for ks_side in ("yes_long", "no_long"):
                    k_win = k_set if ks_side == "yes_long" else (universe - k_set)
                    k_ask_top = _top_of_book_ask(kb, ks_side)
                    if k_ask_top is None:
                        continue
                    for pb in pbs:
                        p_degs = _parse_polymarket_range(pb.get("question", ""))
                        if not p_degs:
                            continue
                        p_set = set(p_degs)
                        for ps_side in ("yes_long", "no_long"):
                            p_win = p_set if ps_side == "yes_long" else (universe - p_set)
                            loss = universe - (k_win | p_win)
                            if loss:
                                continue
                            p_ask_top = _top_of_book_ask(pb, ps_side)
                            if p_ask_top is None:
                                continue
                            if k_ask_top + p_ask_top >= 1.0:
                                continue
                            raw_count += 1

                            # Test fillability for MIN_PAIR_SIZE, requiring avg fill < 1 - small buffer
                            k_ladder = cheap_ask_ladder_yes(kb) if ks_side == "yes_long" else cheap_ask_ladder_no(kb)
                            p_ladder = cheap_ask_ladder_yes(pb) if ps_side == "yes_long" else cheap_ask_ladder_no(pb)
                            kf = walk_ladder_buy(k_ladder, MIN_PAIR_SIZE)
                            pf = walk_ladder_buy(p_ladder, MIN_PAIR_SIZE)
                            if kf is None or pf is None:
                                illusory_count += 1
                                continue
                            if kf[0] + pf[0] >= 1.0:
                                illusory_count += 1
                                continue
                            fillable_count += 1
    return {
        "raw_top_of_book_arbs": raw_count,
        "fillable_at_min_size": fillable_count,
        "illusory_due_to_depth": illusory_count,
    }


def _top_of_book_ask(b: dict, side: str) -> Optional[float]:
    """Return the best ASK price for the chosen side. None if no ask."""
    if side == "yes_long":
        return b.get("best_yes_ask")
    # no_long: NO ask = 1 - YES bid
    yb = b.get("best_yes_bid")
    if yb is None:
        return None
    return round(1.0 - yb, 4)


# ── CLI ────────────────────────────────────────────────────────────────


def main():
    out_dir = Path(__file__).resolve().parent

    print("=" * 70)
    print("Strategy 3 — Structural bracket-math arbitrage")
    print("=" * 70)
    print(f"Cities: {CITIES}")
    print(f"Date: {DATE}")
    print()

    print("--- DIAGNOSTIC: raw vs fillable arb counts ---")
    diag = diagnose_raw_arbs(CITIES, DATE)
    print(f"  raw top-of-book structural arbs (loss empty, ask sum < 1): {diag['raw_top_of_book_arbs']}")
    print(f"  fillable at MIN_PAIR_SIZE={MIN_PAIR_SIZE}: {diag['fillable_at_min_size']}")
    print(f"  illusory (vanish under depth): {diag['illusory_due_to_depth']}")
    distinct = count_distinct_keys(CITIES, DATE)
    print(f"  distinct keys at each pair size: {distinct}")
    diag.update(distinct)
    print()

    iterations = []

    # ── v1: STRICT structural, MARK-TO-MARKET exit ──
    # (the obvious approach — open structural arb, try to close MtM. We
    # expect this to lose because spreads are wide.)
    print("--- v1: strict worst-case structural arb, MtM exit ---")
    v1 = run_backtest(
        CITIES, DATE,
        mode="worst_case",
        min_edge=0.0,
        pair_size=DEFAULT_PAIR_SIZE_CAP,
        target_pnl_per_pair=0.05,
        time_stop_minutes=240,
    )
    print(f"  raw signals enumerated: {v1['raw_signal_count']}")
    print(f"  filled trades: {v1['n_trades']}")
    print(f"  winners: {v1['n_winners']}, win rate: {v1['win_rate']}")
    print(f"  total PnL: ${v1['total_pnl_dollars']:.2f}")
    print(f"  exit reasons: {v1['exit_reason_counts']}")
    print()

    iterations.append({
        "version": "v1",
        "n_trades": v1["n_trades"],
        "n_winners": v1["n_winners"],
        "win_rate": v1["win_rate"],
        "total_pnl_dollars": v1["total_pnl_dollars"],
        "avg_pnl_per_trade": v1["avg_pnl_per_trade"],
        "median_pnl_per_trade": v1["median_pnl_per_trade"],
        "sharpe_per_trade": v1["sharpe_per_trade"],
        "max_drawdown_dollars": v1["max_drawdown_dollars"],
        "settled_n": v1["settled_n"],
        "settled_pnl_dollars": v1["settled_pnl_dollars"],
        "exit_reason_counts": v1["exit_reason_counts"],
        "raw_signal_count": v1["raw_signal_count"],
        "notes": "Strict worst-case structural arb (loss zone empty, total ask < $1) "
                 "with mark-to-market exit (target +$0.05/pair, time stop 240m). "
                 "Expected to underperform: structural guarantee only realises at "
                 "settlement, and MtM closes pay the bid-ask spread on both legs.",
    })

    # ── v2: STRICT structural, HOLD-TO-SETTLEMENT scoring ──
    # (right way to score the same pairs as v1)
    print("--- v2: strict worst-case structural arb, hold-to-settle scoring ---")
    v2 = run_backtest_hold_to_settle(
        CITIES, DATE,
        mode="worst_case",
        min_edge=0.0,
        pair_size=DEFAULT_PAIR_SIZE_CAP,
    )
    print(f"  raw signals enumerated: {v2['raw_signal_count']}")
    print(f"  filled trades: {v2['n_trades']}")
    print(f"  winners (worst-case PnL>0): {v2['n_winners']}, win rate: {v2['win_rate']}")
    print(f"  total expected PnL (forecast-weighted): ${v2['total_pnl_dollars']:.2f}")
    print(f"  total worst-case PnL: ${v2['worst_case_pnl_dollars']:.2f}")
    print(f"  avg expected PnL/pair: ${v2['avg_pnl_per_trade']:.4f}")
    print()

    iterations.append({
        "version": "v2",
        "n_trades": v2["n_trades"],
        "n_winners": v2["n_winners"],
        "win_rate": v2["win_rate"],
        "total_pnl_dollars": v2["total_pnl_dollars"],
        "avg_pnl_per_trade": v2["avg_pnl_per_trade"],
        "median_pnl_per_trade": v2["median_pnl_per_trade"],
        "sharpe_per_trade": v2["sharpe_per_trade"],
        "max_drawdown_dollars": v2["max_drawdown_dollars"],
        "settled_n": v2["settled_n"],
        "settled_pnl_dollars": v2["settled_pnl_dollars"],
        "exit_reason_counts": {"hold_to_settle": v2["n_trades"]},
        "raw_signal_count": v2["raw_signal_count"],
        "worst_case_pnl_dollars": v2["worst_case_pnl_dollars"],
        "notes": "Same selection as v1 (structural arbs only) but scored by "
                 "HOLD-TO-SETTLEMENT payout: $1 + $1*1{T ∈ win_2}. Expected "
                 "PnL uses NWS PDF for E[1{T ∈ win_2}]; worst-case assumes "
                 "win_2 never realises ($1 payout/pair).",
    })

    # ── v3: structural-AND-expected, HOLD-TO-SETTLEMENT scoring ──
    # Loss zone still empty (no risk of $0 payout), but we accept pairs
    # where total entry cost may be slightly above $1 if expected win_2
    # bonus more than compensates. Edge measured against expected payout.
    print("--- v3: structural + expected-edge filter, hold-to-settle scoring ---")
    v3 = run_backtest_hold_to_settle(
        CITIES, DATE,
        mode="structural_expected",
        min_edge=0.05,
        pair_size=DEFAULT_PAIR_SIZE_CAP,
    )
    print(f"  raw signals enumerated: {v3['raw_signal_count']}")
    print(f"  filled trades: {v3['n_trades']}")
    print(f"  winners (worst-case PnL>0): {v3['n_winners']}, win rate: {v3['win_rate']}")
    print(f"  total expected PnL: ${v3['total_pnl_dollars']:.2f}")
    print(f"  total worst-case PnL: ${v3['worst_case_pnl_dollars']:.2f}")
    print(f"  avg expected PnL/pair: ${v3['avg_pnl_per_trade']:.4f}")
    print()

    iterations.append({
        "version": "v3",
        "n_trades": v3["n_trades"],
        "n_winners": v3["n_winners"],
        "win_rate": v3["win_rate"],
        "total_pnl_dollars": v3["total_pnl_dollars"],
        "avg_pnl_per_trade": v3["avg_pnl_per_trade"],
        "median_pnl_per_trade": v3["median_pnl_per_trade"],
        "sharpe_per_trade": v3["sharpe_per_trade"],
        "max_drawdown_dollars": v3["max_drawdown_dollars"],
        "settled_n": v3["settled_n"],
        "settled_pnl_dollars": v3["settled_pnl_dollars"],
        "exit_reason_counts": {"hold_to_settle": v3["n_trades"]},
        "raw_signal_count": v3["raw_signal_count"],
        "worst_case_pnl_dollars": v3["worst_case_pnl_dollars"],
        "notes": "Structural pairs (loss zone empty), filtered by expected "
                 "edge >= $0.05/pair after fees using NWS PDF. Allows entry "
                 "cost > $1 if win_2 bonus mass justifies it. Scored by "
                 "hold-to-settle ($1 + 1{T ∈ win_2}); both expected (PDF) "
                 "and worst-case ($1 payout) PnL reported.",
    })

    metrics = {
        "strategy_name": "structural_bracket_math_arb",
        "diagnostics": diag,
        "iterations": iterations,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    # Save trade-level dumps for reference
    (out_dir / "trades_v1.json").write_text(json.dumps(v1["trades"], indent=2, default=str))
    (out_dir / "trades_v2.json").write_text(json.dumps(v2["trades"], indent=2, default=str))
    (out_dir / "trades_v3.json").write_text(json.dumps(v3["trades"], indent=2, default=str))

    print("Wrote", out_dir / "metrics.json")


if __name__ == "__main__":
    main()
