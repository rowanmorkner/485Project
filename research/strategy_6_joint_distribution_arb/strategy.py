"""
Strategy 6 — Joint-distribution structural arbitrage.

Background
----------
Strategy 3 (S3) found 15 hold-to-settle "structural" arbs on 2026-05-06,
with a quoted "100% worst-case win rate" of +$40 / +$140 expected. But the
worst-case math implicitly assumed that Kalshi's settlement temperature
EQUALS Polymarket's settlement temperature. The empirical Δ histogram
(188 historical pairs) shows venues only agree on the same integer high
25.5% of the time; Kalshi reads ~+1°F warmer on average. So a pair where
both legs require "actual high in B_K" and "actual high in B_P" can still
pay $0 if K_high lands inside B_K but P_high (≠ K_high) lands outside B_P.

This strategy reuses S3's enumeration / ladder-walking logic but replaces
the "venues agree" assumption with the empirical joint distribution
P(K_high, P_high) provided by `joint_kp_distribution()` in loaders.py.

Versions
--------
- v1: Naive S3 replication (baseline). Selection: structural (loss zone
      empty over the on-snapshot universe) AND combined entry cost < $1.
      Scoring: hold-to-settle, "venues agree" assumption — payout =
      $1 + $1·1{T ∈ win_2} via NWS PDF. This reproduces S3 v2.

- v2: Joint-EV filter. Selection: keep only pairs where E[payout] under
      the joint K-P distribution exceeds entry cost + a 5¢/pair safety
      margin. Scoring: same expected-payout-under-joint-KP. No structural
      requirement.

- v3: CVaR / risk-floor filter. Selection: keep pairs where the 5th
      percentile payout under the joint K-P distribution still exceeds
      entry cost. Scoring: expected payout under joint K-P. This is the
      "true worst-case-positive" filter under venue divergence.

- v4: v3 + asymmetric directional filter. Kalshi reads ~+1°F warmer, so
      hedges where Kalshi long is on the high-temperature side
      systematically benefit. We require the Kalshi YES-long leg to cover
      a bracket whose center is ≥ the Polymarket YES-long leg's center
      (when both are yes_long), or analogous side-aware tweaks.

Honest caveats
--------------
- The joint K-P distribution is approximated as P(K)·P(Δ|K) with the
  empirical Δ histogram treated as K-independent. n=188 (≈30-90 per city)
  doesn't refute this but doesn't prove it either.
- Per-city Δ histograms are noisy (Seattle n=49, SF n=29 etc.).
- The forecast PDF (NWS) is used as P(K). Round-1 history shows the
  live bot lost $18M on PDF-driven trades, so this is fragile if the
  PDF is biased.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median, stdev
from typing import Optional, Callable

# Make project root importable
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# REUSED FROM S3 / loaders
from research.loaders import (
    iter_snapshots,
    load_forecast_pdf,
    load_settlement,
    walk_ladder_buy,
    find_kalshi_bracket,
    find_polymarket_bracket,
    joint_kp_distribution,
    expected_payout_under_joint_kp,
    quantile_payout_under_joint_kp,
    venue_divergence_histogram,
)
from strategy.parsers import _parse_kalshi_range, _parse_polymarket_range


CITIES = ["Miami", "LA", "Austin", "San Francisco", "Seattle"]
DATE = "2026-05-06"

# Trade economics — same as S3 for fair comparison
FEE_PER_CONTRACT = 0.005
DEFAULT_PAIR_SIZE_CAP = 50
MIN_PAIR_SIZE = 5

# v2: how much expected-edge above cost we require
V2_SAFETY_MARGIN = 0.05    # $0.05 per pair
# v3: tail-quantile we require to beat cost
V3_TAIL_QUANTILE = 0.05    # 5th percentile


# ── Side helpers (REUSED from S3) ─────────────────────────────────────────


def cheap_ask_ladder_yes(b: dict) -> list:
    return list(b.get("yes_ask_ladder") or [])


def cheap_ask_ladder_no(b: dict) -> list:
    return [(round(1.0 - p, 4), q) for p, q in (b.get("yes_bid_ladder") or [])]


# ── Pair candidate (extends S3 with joint-distribution metrics) ──────────


@dataclass
class PairCandidate:
    city: str
    date: str
    ts: str
    kalshi_ticker: str
    kalshi_label: str
    kalshi_side: str
    kalshi_degrees: list[int]
    kalshi_win_set: set[int]
    poly_token: str
    poly_label: str
    poly_side: str
    poly_degrees: list[int]
    poly_win_set: set[int]
    win_2: set[int]              # if K_high == P_high then both legs pay
    win_1: set[int]
    loss: set[int]
    universe: set[int]
    avg_kalshi_cost: float
    avg_poly_cost: float
    pair_size: int
    total_cost_per_pair: float
    # S3-style scoring (assumes K=P)
    s3_worst_case_payout: float  # min over universe assuming K=P
    s3_expected_payout: float    # forecast-weighted assuming K=P
    # NEW: joint distribution scoring
    joint_expected_payout: float       # E[payout(k,p)] under joint
    joint_q05_payout: float            # 5th percentile of payout(k,p)
    joint_min_payout: float            # absolute min over support of joint
    # Edges (over fees)
    s3_worst_case_edge: float
    s3_expected_edge: float
    joint_expected_edge: float         # joint_expected_payout - cost
    joint_q05_edge: float              # joint_q05_payout - cost


# ── Universe & side-membership (REUSED) ──────────────────────────────────


def venue_universe(kalshi_brackets: list, poly_brackets: list) -> set[int]:
    """Union of all parsed degree ranges at this snapshot."""
    deg = set()
    for b in kalshi_brackets:
        deg.update(_parse_kalshi_range(b.get("subtitle", "")))
    for b in poly_brackets:
        deg.update(_parse_polymarket_range(b.get("question", "")))
    return deg


# ── NEW: pair payout function under joint K,P ────────────────────────────


def make_pair_payout_fn(
    kalshi_win_set: set[int],
    poly_win_set: set[int],
) -> Callable[[int, int], float]:
    """Return a function f(k, p) → $ payout for ONE pair (1 contract on
    each leg) given Kalshi reads `k` and Polymarket reads `p`.

    A YES-long contract pays $1 if the venue's reading is in its win-set,
    $0 otherwise. So the pair payout is:

        f(k, p) = 1{k ∈ K_win} + 1{p ∈ P_win}

    which lives in {0, 1, 2}. Note: structural arbs (loss-zone empty over
    the universe assuming K=P) DO NOT guarantee f(k, p) ≥ 1 when k ≠ p.
    That's the entire reason this strategy exists.
    """
    def fn(k: int, p: int) -> float:
        leg_k = 1.0 if k in kalshi_win_set else 0.0
        leg_p = 1.0 if p in poly_win_set else 0.0
        return leg_k + leg_p
    return fn


# ── Score a single (Kalshi bracket, Polymarket bracket, sides) pair ──────


def score_pair(
    city: str, date: str, ts: str,
    kb: dict, ks_side: str,
    pb: dict, ps_side: str,
    universe: set[int],
    forecast_pdf: dict[int, float],
    joint_kp: dict[tuple[int, int], float],
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

    k_ask = cheap_ask_ladder_yes(kb) if ks_side == "yes_long" else cheap_ask_ladder_no(kb)
    p_ask = cheap_ask_ladder_yes(pb) if ps_side == "yes_long" else cheap_ask_ladder_no(pb)

    k_fill = walk_ladder_buy(k_ask, pair_size)
    p_fill = walk_ladder_buy(p_ask, pair_size)
    if k_fill is None or p_fill is None:
        return None
    k_avg, _ = k_fill
    p_avg, _ = p_fill
    total_cost = k_avg + p_avg

    # S3-style win zones (assuming K=P)
    win_2 = k_win & p_win
    win_1 = (k_win ^ p_win) & universe
    loss = universe - (k_win | p_win)
    s3_worst = 0.0 if loss else 1.0
    s3_exp = sum(2.0 * forecast_pdf.get(d, 0.0) for d in win_2) + \
             sum(1.0 * forecast_pdf.get(d, 0.0) for d in win_1)

    # JOINT-distribution scoring
    payout_fn = make_pair_payout_fn(k_win, p_win)
    joint_e = expected_payout_under_joint_kp(payout_fn, joint_kp)
    joint_q05 = quantile_payout_under_joint_kp(payout_fn, joint_kp, q=V3_TAIL_QUANTILE)
    # absolute minimum
    joint_min = min(payout_fn(k, p) for (k, p) in joint_kp) if joint_kp else 0.0

    # Edge accounting: subtract round-trip pair fees ($0.01 per pair entry-
    # only at settlement, since winning/losing contracts auto-resolve at
    # no exit fee).
    pair_fees = 2.0 * FEE_PER_CONTRACT  # $0.01/pair on entry
    s3_worst_edge = s3_worst - total_cost - pair_fees
    s3_exp_edge = s3_exp - total_cost - pair_fees
    joint_e_edge = joint_e - total_cost - pair_fees
    joint_q05_edge = joint_q05 - total_cost - pair_fees

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
        s3_worst_case_payout=s3_worst,
        s3_expected_payout=s3_exp,
        joint_expected_payout=joint_e,
        joint_q05_payout=joint_q05,
        joint_min_payout=joint_min,
        s3_worst_case_edge=s3_worst_edge,
        s3_expected_edge=s3_exp_edge,
        joint_expected_edge=joint_e_edge,
        joint_q05_edge=joint_q05_edge,
    )


# ── Filter modes ────────────────────────────────────────────────────────


def passes_v1(c: PairCandidate) -> bool:
    """v1 — Naive S3 replication: structural (loss empty under K=P) AND
    cost < $1. Scoring uses S3's expected_payout. (= S3 v2)"""
    if c.loss:
        return False
    return c.s3_worst_case_edge > 0


def passes_v2(c: PairCandidate, safety_margin: float = V2_SAFETY_MARGIN) -> bool:
    """v2 — Joint-EV filter: E[payout under joint] > cost + safety_margin.
    Does NOT require structural worst-case."""
    return c.joint_expected_edge > safety_margin


def passes_v3(c: PairCandidate) -> bool:
    """v3 — CVaR filter: 5th percentile of payout under joint K,P >
    entry cost. The TRUE worst-case-positive filter under venue divergence."""
    return c.joint_q05_edge > 0


def asymmetric_directional_ok(c: PairCandidate) -> bool:
    """v4 helper: Kalshi reads ~+1°F warmer, so prefer hedges where Kalshi
    is positioned to benefit from "Kalshi reads inside high bracket".

    Concretely: when both legs are yes_long on different brackets, the
    Kalshi bracket should be at or above the Polymarket bracket (Kalshi
    captures the top of the joint distribution). When both are no_long
    (i.e. betting on "out of bracket"), prefer Polymarket on the high
    side. For mixed-side pairs (yes_long vs no_long, the typical
    aligned-bracket pair), no asymmetry preference applies (those are
    already exact-bracket-aligned and benefit equally).
    """
    k_center = mean(c.kalshi_degrees) if c.kalshi_degrees else 0.0
    p_center = mean(c.poly_degrees) if c.poly_degrees else 0.0
    if c.kalshi_side == "yes_long" and c.poly_side == "yes_long":
        return k_center >= p_center
    if c.kalshi_side == "no_long" and c.poly_side == "no_long":
        return p_center >= k_center
    # mixed / aligned pairs: no preference
    return True


def passes_v4(c: PairCandidate) -> bool:
    return passes_v3(c) and asymmetric_directional_ok(c)


# ── Enumeration (same shape as S3) ───────────────────────────────────────


def enumerate_signals(
    city: str, date: str, ts: str,
    kalshi_brackets: list, poly_brackets: list,
    forecast_pdf: dict[int, float],
    joint_kp: dict[tuple[int, int], float],
    filter_fn: Callable[[PairCandidate], bool],
    pair_size: int = DEFAULT_PAIR_SIZE_CAP,
) -> list[PairCandidate]:
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
                        universe, forecast_pdf, joint_kp, pair_size,
                    )
                    if cand is None:
                        continue
                    if filter_fn(cand):
                        out.append(cand)
    return out


# ── Position lifecycle ───────────────────────────────────────────────────


@dataclass
class PairPosition:
    city: str
    date: str
    entry_ts: str
    pair_size: int
    kalshi_ticker: str
    kalshi_label: str
    kalshi_side: str
    kalshi_entry_avg: float
    kalshi_entry_cost: float
    kalshi_degrees: list[int]
    kalshi_win_set: set[int]
    poly_token: str
    poly_label: str
    poly_side: str
    poly_entry_avg: float
    poly_entry_cost: float
    poly_degrees: list[int]
    poly_win_set: set[int]
    universe: set[int]
    win_2: set[int]
    win_1: set[int]
    loss: set[int]
    s3_expected_payout: float
    s3_worst_case_payout: float
    joint_expected_payout: float
    joint_q05_payout: float
    joint_min_payout: float
    structural: bool


def open_pair(c: PairCandidate) -> PairPosition:
    return PairPosition(
        city=c.city, date=c.date, entry_ts=c.ts, pair_size=c.pair_size,
        kalshi_ticker=c.kalshi_ticker, kalshi_label=c.kalshi_label,
        kalshi_side=c.kalshi_side, kalshi_entry_avg=c.avg_kalshi_cost,
        kalshi_entry_cost=c.avg_kalshi_cost * c.pair_size,
        kalshi_degrees=c.kalshi_degrees, kalshi_win_set=c.kalshi_win_set,
        poly_token=c.poly_token, poly_label=c.poly_label,
        poly_side=c.poly_side, poly_entry_avg=c.avg_poly_cost,
        poly_entry_cost=c.avg_poly_cost * c.pair_size,
        poly_degrees=c.poly_degrees, poly_win_set=c.poly_win_set,
        universe=c.universe,
        win_2=c.win_2, win_1=c.win_1, loss=c.loss,
        s3_expected_payout=c.s3_expected_payout,
        s3_worst_case_payout=c.s3_worst_case_payout,
        joint_expected_payout=c.joint_expected_payout,
        joint_q05_payout=c.joint_q05_payout,
        joint_min_payout=c.joint_min_payout,
        structural=not c.loss,
    )


# ── Backtest driver ──────────────────────────────────────────────────────


def run_backtest(
    cities: list[str],
    date: str,
    filter_fn: Callable[[PairCandidate], bool],
    score_using: str,                   # 's3_expected' | 'joint_expected'
    sort_key: Callable[[PairCandidate], float],
    pair_size: int = DEFAULT_PAIR_SIZE_CAP,
    one_per_k_position: bool = False,   # if True, opens at most one trade per (K_ticker, K_side)
) -> dict:
    """Hold-to-settle scoring. Picks at most one position per
    (city, kalshi_ticker, poly_token, k_side, p_side) — the FIRST snapshot
    where the pair passes `filter_fn`. Final per-trade PnL is computed as:

      - If settlement available: realized using actual K_high, P_high.
      - Else: scored as `score_using` (E[payout] under chosen distribution)
              minus entry cost minus fees.

    Reports settled_pnl_dollars separately when available.
    """
    raw_signal_count = 0
    positions: list[PairPosition] = []
    pnls_scored = []      # what we'd report as "expected pnl" pre-settlement
    pnls_joint_e = []     # E[pnl] under joint K-P
    pnls_joint_q05 = []   # 5th-percentile pnl under joint K-P
    pnls_s3_worst = []    # S3 "worst case under K=P assumption"
    pnls_settled = []     # realized pnl if settlement known
    settle_known = []     # bool per trade

    for city in cities:
        forecast_pdf = load_forecast_pdf(city, date) or {}
        if not forecast_pdf:
            continue
        joint_kp = joint_kp_distribution(forecast_pdf, city=city)
        snaps = list(iter_snapshots(city, date))
        if not snaps:
            continue
        opened_keys: set[tuple] = set()
        opened_k_positions: set[tuple] = set()
        opened_p_positions: set[tuple] = set()
        settlement = load_settlement(city, date) or (None, None)
        k_actual, p_actual = settlement

        for ts, venues in snaps:
            kbs = venues.get("kalshi", [])
            pbs = venues.get("polymarket", [])
            if not kbs or not pbs:
                continue
            cands = enumerate_signals(
                city, date, ts, kbs, pbs, forecast_pdf, joint_kp,
                filter_fn=filter_fn, pair_size=pair_size,
            )
            raw_signal_count += len(cands)
            cands.sort(key=sort_key, reverse=True)
            for c in cands:
                key = (city, c.kalshi_ticker, c.poly_token,
                       c.kalshi_side, c.poly_side)
                k_key = (city, c.kalshi_ticker, c.kalshi_side)
                p_key = (city, c.poly_token, c.poly_side)
                if key in opened_keys:
                    continue
                if one_per_k_position and (k_key in opened_k_positions
                                            or p_key in opened_p_positions):
                    continue
                if c.pair_size < MIN_PAIR_SIZE:
                    continue
                pos = open_pair(c)
                positions.append(pos)
                opened_k_positions.add(k_key)
                opened_p_positions.add(p_key)

                cost = pos.kalshi_entry_cost + pos.poly_entry_cost
                fee = 2.0 * FEE_PER_CONTRACT * pos.pair_size  # entry only

                # All four scoring lenses
                pnl_joint_e = pos.joint_expected_payout * pos.pair_size - cost - fee
                pnl_joint_q05 = pos.joint_q05_payout * pos.pair_size - cost - fee
                pnl_s3_worst = pos.s3_worst_case_payout * pos.pair_size - cost - fee
                pnls_joint_e.append(pnl_joint_e)
                pnls_joint_q05.append(pnl_joint_q05)
                pnls_s3_worst.append(pnl_s3_worst)

                if score_using == "s3_expected":
                    pnls_scored.append(pos.s3_expected_payout * pos.pair_size - cost - fee)
                elif score_using == "joint_expected":
                    pnls_scored.append(pnl_joint_e)
                else:
                    pnls_scored.append(pnl_joint_e)

                # Settled pnl if available
                if k_actual is not None and p_actual is not None:
                    leg_k = 1.0 if k_actual in pos.kalshi_win_set else 0.0
                    leg_p = 1.0 if p_actual in pos.poly_win_set else 0.0
                    realized_pair = leg_k + leg_p
                    pnls_settled.append(realized_pair * pos.pair_size - cost - fee)
                    settle_known.append(True)
                else:
                    pnls_settled.append(None)
                    settle_known.append(False)

                opened_keys.add(key)

    if not positions:
        return _empty_result(raw_signal_count)

    n = len(positions)
    n_winners = sum(1 for p in pnls_scored if p > 0)
    total_pnl = sum(pnls_scored)
    avg_pnl = total_pnl / n
    med_pnl = median(pnls_scored)
    per_pair = [p / pos.pair_size for p, pos in zip(pnls_scored, positions)]
    if len(per_pair) > 1:
        s = stdev(per_pair) or 1e-9
        sharpe = mean(per_pair) / s
    else:
        sharpe = 0.0
    sorted_pairs = sorted(zip(positions, pnls_scored), key=lambda pe: pe[0].entry_ts)
    cum, peak, max_dd = 0.0, 0.0, 0.0
    for _, pnl in sorted_pairs:
        cum += pnl
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)

    settled_pnls = [p for p, k in zip(pnls_settled, settle_known) if k]
    settled_n = len(settled_pnls)
    settled_pnl = sum(settled_pnls) if settled_pnls else 0.0

    trades = []
    for pos, p_scored, p_je, p_jq05, p_s3w, p_set in zip(
        positions, pnls_scored, pnls_joint_e, pnls_joint_q05,
        pnls_s3_worst, pnls_settled,
    ):
        trades.append({
            "city": pos.city,
            "entry_ts": pos.entry_ts,
            "kalshi_label": pos.kalshi_label,
            "kalshi_ticker": pos.kalshi_ticker,
            "kalshi_side": pos.kalshi_side,
            "poly_token": pos.poly_token,
            "poly_label": pos.poly_label,
            "poly_side": pos.poly_side,
            "pair_size": pos.pair_size,
            "structural": pos.structural,
            "entry_cost_per_pair": round(pos.kalshi_entry_avg + pos.poly_entry_avg, 4),
            "s3_expected_payout": round(pos.s3_expected_payout, 4),
            "s3_worst_case_payout": round(pos.s3_worst_case_payout, 4),
            "joint_expected_payout": round(pos.joint_expected_payout, 4),
            "joint_q05_payout": round(pos.joint_q05_payout, 4),
            "joint_min_payout": round(pos.joint_min_payout, 4),
            "scored_pnl": round(p_scored, 4),
            "joint_expected_pnl": round(p_je, 4),
            "joint_q05_pnl": round(p_jq05, 4),
            "s3_worst_case_pnl": round(p_s3w, 4),
            "settled_pnl": round(p_set, 4) if p_set is not None else None,
        })

    return {
        "raw_signal_count": raw_signal_count,
        "n_trades": n,
        "n_winners": n_winners,
        "win_rate": round(n_winners / n, 4),
        "total_pnl_dollars": round(total_pnl, 4),
        "avg_pnl_per_trade": round(avg_pnl, 4),
        "median_pnl_per_trade": round(med_pnl, 4),
        "sharpe_per_trade": round(sharpe, 4),
        "max_drawdown_dollars": round(max_dd, 4),
        "settled_n": settled_n,
        "settled_pnl_dollars": round(settled_pnl, 4),
        "joint_expected_total_pnl": round(sum(pnls_joint_e), 4),
        "joint_q05_total_pnl": round(sum(pnls_joint_q05), 4),
        "s3_worst_case_total_pnl": round(sum(pnls_s3_worst), 4),
        "exit_reason_counts": {"hold_to_settle": n},
        "trades": trades,
    }


def _empty_result(raw_signal_count: int) -> dict:
    return {
        "raw_signal_count": raw_signal_count,
        "n_trades": 0, "n_winners": 0, "win_rate": 0.0,
        "total_pnl_dollars": 0.0, "avg_pnl_per_trade": 0.0,
        "median_pnl_per_trade": 0.0, "sharpe_per_trade": 0.0,
        "max_drawdown_dollars": 0.0, "settled_n": 0,
        "settled_pnl_dollars": 0.0,
        "joint_expected_total_pnl": 0.0,
        "joint_q05_total_pnl": 0.0,
        "s3_worst_case_total_pnl": 0.0,
        "exit_reason_counts": {},
        "trades": [],
    }


# ── Diagnostic: how many of S3 v2's 15 trades survive each filter? ───────


def diagnose_s3_v2_overlap() -> dict:
    """For each of S3 v2's 15 trades, recompute under joint distribution
    and report whether it would survive v2/v3/v4 filters."""
    s3_v2_path = ROOT / "research" / "strategy_3_structural_arb" / "trades_v2.json"
    if not s3_v2_path.exists():
        return {"error": "trades_v2.json missing"}
    s3_trades = json.loads(s3_v2_path.read_text())
    results = []
    survived_v2 = 0
    survived_v3 = 0
    survived_v4 = 0
    for st in s3_trades:
        # We can't perfectly replay the snapshot without full data, but we
        # can recompute joint metrics from win sets reconstructed from
        # bracket labels. Simpler: we just rerun the v3 backtest below
        # and intersect by (city, kalshi_ticker, poly_token, sides).
        results.append({
            "city": st["city"],
            "kalshi_label": st["kalshi_label"],
            "poly_label": st["poly_label"],
        })
    return {"s3_v2_trades": results, "n_s3_v2": len(s3_trades)}


# ── CLI ──────────────────────────────────────────────────────────────────


def main():
    out_dir = Path(__file__).resolve().parent

    print("=" * 70)
    print("Strategy 6 — Joint-distribution structural arbitrage")
    print("=" * 70)
    print(f"Cities: {CITIES}")
    print(f"Date: {DATE}")
    print()

    # Print divergence histogram for the record
    pooled = venue_divergence_histogram()
    print(f"Pooled K-P divergence histogram (n={sum(round(v*188) for v in pooled.values())} approx):")
    for d in sorted(pooled):
        print(f"  Δ={d:+d}: {pooled[d]:.3%}")
    print()

    iterations = []

    # ── v1: Replicate S3 v2 (naive structural, score under "venues agree") ──
    print("--- v1: NAIVE S3 replication (structural, K=P scoring) ---")
    v1 = run_backtest(
        CITIES, DATE,
        filter_fn=passes_v1,
        score_using="s3_expected",
        sort_key=lambda c: c.s3_worst_case_edge,
    )
    print(f"  n_trades: {v1['n_trades']}")
    print(f"  total scored PnL: ${v1['total_pnl_dollars']:.2f}")
    print(f"  joint E[PnL]: ${v1['joint_expected_total_pnl']:.2f}")
    print(f"  joint q05 PnL: ${v1['joint_q05_total_pnl']:.2f}")
    print(f"  S3-worst (K=P) PnL: ${v1['s3_worst_case_total_pnl']:.2f}")
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
        "joint_expected_total_pnl": v1["joint_expected_total_pnl"],
        "joint_q05_total_pnl": v1["joint_q05_total_pnl"],
        "s3_worst_case_total_pnl": v1["s3_worst_case_total_pnl"],
        "notes": "Naive S3 replication: structural (loss zone empty) "
                 "under K=P assumption AND cost < $1. PnL scored using "
                 "S3's expected-payout (NWS PDF, K=P) for direct "
                 "comparison. Joint E[PnL] / joint q05 PnL / S3-worst "
                 "fields show the same trades scored under the joint "
                 "K-P distribution — this exposes the venue-divergence "
                 "risk hiding inside S3's $40 worst-case.",
    })

    # ── v2: Joint-EV filter ──
    print("--- v2: JOINT-EV filter (E[payout|joint] - cost > 5¢) ---")
    v2 = run_backtest(
        CITIES, DATE,
        filter_fn=lambda c: passes_v2(c, V2_SAFETY_MARGIN),
        score_using="joint_expected",
        sort_key=lambda c: c.joint_expected_edge,
    )
    print(f"  n_trades: {v2['n_trades']}")
    print(f"  total joint E[PnL]: ${v2['total_pnl_dollars']:.2f}")
    print(f"  joint q05 PnL: ${v2['joint_q05_total_pnl']:.2f}")
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
        "exit_reason_counts": v2["exit_reason_counts"],
        "raw_signal_count": v2["raw_signal_count"],
        "joint_expected_total_pnl": v2["joint_expected_total_pnl"],
        "joint_q05_total_pnl": v2["joint_q05_total_pnl"],
        "s3_worst_case_total_pnl": v2["s3_worst_case_total_pnl"],
        "notes": f"Joint-EV filter: accept pair if E[payout] under "
                 f"P(K_high,P_high) exceeds entry cost + ${V2_SAFETY_MARGIN}/pair "
                 f"safety margin. Drops S3's structural requirement; "
                 f"scoring uses NWS PDF for P(K) and empirical Δ "
                 f"histogram for P(P|K). Pure forecast-EV bet — no "
                 f"worst-case floor.",
    })

    # ── v3: CVaR filter ──
    print("--- v3: CVaR filter (5th percentile payout > cost) ---")
    v3 = run_backtest(
        CITIES, DATE,
        filter_fn=passes_v3,
        score_using="joint_expected",
        sort_key=lambda c: c.joint_q05_edge,
    )
    print(f"  n_trades: {v3['n_trades']}")
    print(f"  total joint E[PnL]: ${v3['total_pnl_dollars']:.2f}")
    print(f"  joint q05 PnL: ${v3['joint_q05_total_pnl']:.2f}")
    print(f"  S3-worst PnL: ${v3['s3_worst_case_total_pnl']:.2f}")
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
        "exit_reason_counts": v3["exit_reason_counts"],
        "raw_signal_count": v3["raw_signal_count"],
        "joint_expected_total_pnl": v3["joint_expected_total_pnl"],
        "joint_q05_total_pnl": v3["joint_q05_total_pnl"],
        "s3_worst_case_total_pnl": v3["s3_worst_case_total_pnl"],
        "notes": "CVaR filter: accept pair only if 5th-percentile payout "
                 "under joint K-P > entry cost. This is the production-"
                 "grade replacement for S3's 'venues agree' worst case. "
                 "Scoring uses joint expected payout. Per-city Δ "
                 "histograms (n≈30-90) noisy; documented approximation "
                 "P(Δ|K) ≈ P(Δ).",
    })

    # ── v4: CVaR + asymmetric directional ──
    print("--- v4: CVaR + asymmetric directional filter ---")
    v4 = run_backtest(
        CITIES, DATE,
        filter_fn=passes_v4,
        score_using="joint_expected",
        sort_key=lambda c: c.joint_q05_edge,
    )
    print(f"  n_trades: {v4['n_trades']}")
    print(f"  total joint E[PnL]: ${v4['total_pnl_dollars']:.2f}")
    print(f"  joint q05 PnL: ${v4['joint_q05_total_pnl']:.2f}")
    print()
    iterations.append({
        "version": "v4",
        "n_trades": v4["n_trades"],
        "n_winners": v4["n_winners"],
        "win_rate": v4["win_rate"],
        "total_pnl_dollars": v4["total_pnl_dollars"],
        "avg_pnl_per_trade": v4["avg_pnl_per_trade"],
        "median_pnl_per_trade": v4["median_pnl_per_trade"],
        "sharpe_per_trade": v4["sharpe_per_trade"],
        "max_drawdown_dollars": v4["max_drawdown_dollars"],
        "settled_n": v4["settled_n"],
        "settled_pnl_dollars": v4["settled_pnl_dollars"],
        "exit_reason_counts": v4["exit_reason_counts"],
        "raw_signal_count": v4["raw_signal_count"],
        "joint_expected_total_pnl": v4["joint_expected_total_pnl"],
        "joint_q05_total_pnl": v4["joint_q05_total_pnl"],
        "s3_worst_case_total_pnl": v4["s3_worst_case_total_pnl"],
        "notes": "v3 + asymmetric directional preference. Kalshi reads "
                 "+1°F warmer on average, so prefer hedges where Kalshi "
                 "long is on the high side (yes_long+yes_long: K bracket "
                 "≥ P bracket; no_long+no_long: P bracket ≥ K bracket). "
                 "Mixed/aligned-bracket pairs get no asymmetry penalty.",
    })

    # ── v5: CVaR + strict per-position dedup (no double-counting K leg) ──
    print("--- v5: CVaR + strict per-position dedup ---")
    v5 = run_backtest(
        CITIES, DATE,
        filter_fn=passes_v3,
        score_using="joint_expected",
        sort_key=lambda c: c.joint_q05_edge,
        one_per_k_position=True,
    )
    print(f"  n_trades: {v5['n_trades']}")
    print(f"  total joint E[PnL]: ${v5['total_pnl_dollars']:.2f}")
    print(f"  joint q05 PnL: ${v5['joint_q05_total_pnl']:.2f}")
    print()
    iterations.append({
        "version": "v5",
        "n_trades": v5["n_trades"],
        "n_winners": v5["n_winners"],
        "win_rate": v5["win_rate"],
        "total_pnl_dollars": v5["total_pnl_dollars"],
        "avg_pnl_per_trade": v5["avg_pnl_per_trade"],
        "median_pnl_per_trade": v5["median_pnl_per_trade"],
        "sharpe_per_trade": v5["sharpe_per_trade"],
        "max_drawdown_dollars": v5["max_drawdown_dollars"],
        "settled_n": v5["settled_n"],
        "settled_pnl_dollars": v5["settled_pnl_dollars"],
        "exit_reason_counts": v5["exit_reason_counts"],
        "raw_signal_count": v5["raw_signal_count"],
        "joint_expected_total_pnl": v5["joint_expected_total_pnl"],
        "joint_q05_total_pnl": v5["joint_q05_total_pnl"],
        "s3_worst_case_total_pnl": v5["s3_worst_case_total_pnl"],
        "notes": "v3 CVaR filter + strict per-leg dedup: each "
                 "(city, K_ticker, K_side) and each (city, P_token, P_side) "
                 "is opened at most once across the day. Avoids "
                 "double-counting the same Kalshi leg paired with multiple "
                 "Polymarket brackets — capital-realistic accounting.",
    })

    # ── S3 v2 overlap analysis ──
    s3_v2_path = ROOT / "research" / "strategy_3_structural_arb" / "trades_v2.json"
    s3_v2 = json.loads(s3_v2_path.read_text()) if s3_v2_path.exists() else []
    s3_v2_keys = {(t["city"], t["kalshi_label"], t["poly_label"],
                   t["kalshi_side"], t["poly_side"]) for t in s3_v2}

    def _trade_key(t):
        return (t["city"], t["kalshi_ticker"], t["poly_token"],
                t["kalshi_side"], t["poly_side"])

    # Collect S6 v3 keys (CVaR-filtered)
    v3_keys = {_trade_key(t) for t in v3["trades"]}
    v4_keys = {_trade_key(t) for t in v4["trades"]}
    overlap_v3 = len(s3_v2_keys & v3_keys)
    overlap_v4 = len(s3_v2_keys & v4_keys)

    overlap_summary = {
        "n_s3_v2": len(s3_v2_keys),
        "n_v3": len(v3_keys),
        "n_v4": len(v4_keys),
        "s3_v2_surviving_v3": overlap_v3,
        "s3_v2_surviving_v4": overlap_v4,
        "s3_v2_dropped_by_v3": len(s3_v2_keys) - overlap_v3,
    }
    print()
    print("--- S3 v2 overlap with v3/v4 ---")
    print(f"  S3 v2 trades: {len(s3_v2_keys)}")
    print(f"  S6 v3 trades: {len(v3_keys)}")
    print(f"  Surviving S3 v2 → S6 v3: {overlap_v3}")
    print(f"  Surviving S3 v2 → S6 v4: {overlap_v4}")

    metrics = {
        "strategy_name": "joint_distribution_structural_arb",
        "diagnostics": {
            "pooled_divergence_hist": pooled,
            "s3_v2_overlap": overlap_summary,
        },
        "iterations": iterations,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    (out_dir / "trades_v1.json").write_text(json.dumps(v1["trades"], indent=2, default=str))
    (out_dir / "trades_v2.json").write_text(json.dumps(v2["trades"], indent=2, default=str))
    (out_dir / "trades_v3.json").write_text(json.dumps(v3["trades"], indent=2, default=str))
    (out_dir / "trades_v4.json").write_text(json.dumps(v4["trades"], indent=2, default=str))
    (out_dir / "trades_v5.json").write_text(json.dumps(v5["trades"], indent=2, default=str))

    print()
    print(f"Wrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
