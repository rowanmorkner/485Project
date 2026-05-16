"""
Strategy 8 — Late-day certainty capture.

Idea
----
Round-1 finding (S2): the only convergence that meaningfully closes the
bid-ask gap is the day's final price collapse to {0, 1}. Round-1 finding
(S4): even perfect-foresight intraday exits are unprofitable.

Hypothesis: as the trading day progresses the actual high becomes more
known (intraday observations confirm the reading), and brackets concentrate
toward {0, 1}. In this regime, structural arbs become CHEAPER (closer to
free money) but also rarer. The right play might be to wait, then pounce.

We re-use Strategy 3's structural-arb selection (loss zone empty AND
combined cost < $1, walked through real ladders) but only enter when a
"certainty score" derived from the snapshot itself crosses a threshold.

Certainty score (per snapshot):
  Combines four observable signals:
    * mean (yes_ask - yes_bid) across alive brackets — TIGHTER → certain
    * concentration: max bracket mid price across alive brackets — HIGHER → certain
    * alive-bracket count — LOWER → certain (book is consolidating)
    * snapshot index normalized to [0,1] over the day (proxy time only)

We DELIBERATELY avoid hard-coding a UTC time, because cities settle at
different local times. Instead we use the data's own structure.

We compare two regimes:
  * all-day: structural arb at every snapshot (S3 v2 baseline)
  * late-day: structural arb only when certainty score >= threshold

Versions:
  v1 — fixed window: last N alive snapshots before market death.
  v2 — adaptive: certainty threshold (sweep).
  v3 — v2 + joint K-P risk filter (q05_payout > entry_cost).

Notes:
  * "Dead" brackets (bid=None, ask=0.99 reserve) DO NOT count as alive.
  * Settle-payouts are estimated via E[payout] under the joint K-P
    distribution, which is the honest replacement for "venues agree".
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Optional

# Project root importable
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from research.loaders import (
    iter_snapshots,
    load_forecast_pdf,
    walk_ladder_buy,
    venue_divergence_histogram,
    joint_kp_distribution,
    expected_payout_under_joint_kp,
    quantile_payout_under_joint_kp,
)
from strategy.parsers import _parse_kalshi_range, _parse_polymarket_range


CITIES = ["Miami", "LA", "Austin", "San Francisco", "Seattle"]
DATE = "2026-05-06"

# Trade economics
FEE_PER_CONTRACT = 0.005
DEFAULT_PAIR_SIZE_CAP = 50
MIN_PAIR_SIZE = 5

# Bracket "alive" definition: must have BOTH a bid and an ask, with
# spread < ALIVE_SPREAD_MAX (filters the 0.99 partial-reserve trap).
ALIVE_SPREAD_MAX = 0.30
# Bracket counts as "dead one-sided" if bid >= 0.95 with no ask (consolidated to YES)
# or ask <= 0.02 with no bid (consolidated to NO).


# ── Bracket / ladder helpers (mirror S3) ──────────────────────────────────

def cheap_ask_ladder_yes(b: dict) -> list:
    return list(b.get("yes_ask_ladder") or [])


def cheap_ask_ladder_no(b: dict) -> list:
    return [(round(1.0 - p, 4), q) for p, q in (b.get("yes_bid_ladder") or [])]


def is_alive(b: dict) -> bool:
    """A bracket is 'alive' if it has both a real bid and real ask, with a
    moderate spread. Brackets that have collapsed to {0, 1} or are dead
    one-sided (bid=None ask=0.01, or bid=0.99 ask=None) are NOT alive."""
    yb = b.get("best_yes_bid")
    ya = b.get("best_yes_ask")
    if yb is None or ya is None:
        return False
    if ya - yb >= ALIVE_SPREAD_MAX:
        return False
    return True


# ── Certainty score ───────────────────────────────────────────────────────

@dataclass
class CertaintyMetrics:
    n_alive: int
    mean_spread: float
    max_concentration: float    # max bracket mid across alive brackets
    elapsed_frac: float         # snapshot index / total snapshots
    score: float                # 0..1, larger = more certain


def compute_certainty(kbs: list, pbs: list, idx: int, n_total: int) -> CertaintyMetrics:
    """Compute a 0..1 'certainty' score from snapshot observables.

    Components:
      spread_term       (lower spread → more certain)
      concentration_term (higher max-mid → more certain)
      alive_term        (fewer alive brackets → more certain)
      time_term         (later in day → more certain — small weight, proxy only)

    Weights: spread 0.30, concentration 0.40, alive 0.20, time 0.10.
    Concentration matters most — the book agreeing on a particular
    bracket is the strongest "we know the answer" signal.
    """
    alive = [b for b in kbs + pbs if is_alive(b)]
    n_alive = len(alive)
    spreads = [b["best_yes_ask"] - b["best_yes_bid"] for b in alive]
    mean_spread = mean(spreads) if spreads else 1.0
    # Concentration: across ALL brackets (alive or one-sided dead), what's
    # the largest mid we see? A bid=0.99 ask=None counts as 0.99
    # concentration — that's signal of certainty even though that bracket
    # itself is dead-untradeable.
    mids = []
    for b in kbs + pbs:
        yb = b.get("best_yes_bid")
        ya = b.get("best_yes_ask")
        if yb is not None and ya is not None:
            mids.append((yb + ya) / 2.0)
        elif yb is not None:
            mids.append(yb)
        elif ya is not None:
            mids.append(ya)
    max_conc = max(mids) if mids else 0.0
    # Term mappings (each in [0,1], higher = more certain):
    spread_term = max(0.0, 1.0 - (mean_spread / 0.05))   # spread 0 → 1, spread 0.05 → 0
    concentration_term = max(0.0, (max_conc - 0.5) / 0.5)  # 0.5 → 0, 1.0 → 1
    # Alive term: ~1 alive → highly certain; ~12 alive → uncertain
    alive_term = max(0.0, 1.0 - n_alive / 12.0)
    elapsed_frac = idx / max(1, n_total - 1)
    time_term = elapsed_frac

    score = 0.30 * spread_term + 0.40 * concentration_term + 0.20 * alive_term + 0.10 * time_term
    return CertaintyMetrics(
        n_alive=n_alive,
        mean_spread=mean_spread,
        max_concentration=max_conc,
        elapsed_frac=elapsed_frac,
        score=score,
    )


def is_book_dead(kbs: list, pbs: list) -> bool:
    """Return True if no bracket on either venue is alive — books gone."""
    if not any(is_alive(b) for b in kbs + pbs):
        return True
    return False


# ── Structural-arb pair scoring (lifted from S3) ─────────────────────────

@dataclass
class PairCandidate:
    city: str
    date: str
    ts: str
    snap_idx: int
    certainty: float
    kalshi_ticker: str
    kalshi_label: str
    kalshi_side: str
    kalshi_degrees: list
    kalshi_win_set: set
    poly_token: str
    poly_label: str
    poly_side: str
    poly_degrees: list
    poly_win_set: set
    win_2: set
    win_1: set
    loss: set
    universe: set
    avg_kalshi_cost: float
    avg_poly_cost: float
    pair_size: int
    total_cost_per_pair: float
    worst_case_payout: float
    expected_payout_pdf: float
    expected_payout_joint: float
    q05_payout_joint: float


def venue_universe(kbs: list, pbs: list) -> set:
    deg = set()
    for b in kbs:
        deg.update(_parse_kalshi_range(b.get("subtitle", "")))
    for b in pbs:
        deg.update(_parse_polymarket_range(b.get("question", "")))
    return deg


def score_pair(
    city: str, date: str, ts: str, snap_idx: int, certainty: float,
    kb: dict, ks_side: str,
    pb: dict, ps_side: str,
    universe: set,
    forecast_pdf: dict,
    joint_kp: dict,
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

    win_2 = k_win & p_win
    win_1 = (k_win ^ p_win) & universe
    loss = universe - (k_win | p_win)

    worst_case_payout = 0.0 if loss else 1.0

    # Expected payout under NWS PDF (treats venues as identical)
    exp_pdf = 0.0
    for d in win_2:
        exp_pdf += 2.0 * forecast_pdf.get(d, 0.0)
    for d in win_1:
        exp_pdf += 1.0 * forecast_pdf.get(d, 0.0)

    # Expected payout under joint K-P distribution (true cross-venue model)
    def payout_fn(k, p):
        # Kalshi leg pays $1 when k ∈ k_win; Poly leg pays $1 when p ∈ p_win.
        kp = (1.0 if k in k_win else 0.0)
        pp = (1.0 if p in p_win else 0.0)
        return kp + pp

    exp_joint = expected_payout_under_joint_kp(payout_fn, joint_kp) if joint_kp else exp_pdf
    q05_joint = quantile_payout_under_joint_kp(payout_fn, joint_kp, q=0.05) if joint_kp else worst_case_payout

    return PairCandidate(
        city=city, date=date, ts=ts, snap_idx=snap_idx, certainty=certainty,
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
        win_2=win_2, win_1=win_1, loss=loss, universe=universe,
        avg_kalshi_cost=k_avg, avg_poly_cost=p_avg,
        pair_size=pair_size, total_cost_per_pair=total_cost,
        worst_case_payout=worst_case_payout,
        expected_payout_pdf=exp_pdf,
        expected_payout_joint=exp_joint,
        q05_payout_joint=q05_joint,
    )


def enumerate_pair_candidates(
    city: str, date: str, ts: str, snap_idx: int, certainty: float,
    kbs: list, pbs: list,
    forecast_pdf: dict, joint_kp: dict,
    pair_size: int,
    require_loss_empty: bool = True,
    max_total_cost: float = 1.0,
) -> list:
    universe = venue_universe(kbs, pbs)
    if not universe:
        return []
    out = []
    for kb in kbs:
        for ks_side in ("yes_long", "no_long"):
            for pb in pbs:
                for ps_side in ("yes_long", "no_long"):
                    cand = score_pair(
                        city, date, ts, snap_idx, certainty,
                        kb, ks_side, pb, ps_side,
                        universe, forecast_pdf, joint_kp, pair_size,
                    )
                    if cand is None:
                        continue
                    if require_loss_empty and cand.loss:
                        continue
                    edge_after_fees = (cand.worst_case_payout - cand.total_cost_per_pair) - 2.0 * FEE_PER_CONTRACT
                    if edge_after_fees <= 0:
                        continue
                    if cand.total_cost_per_pair >= max_total_cost:
                        continue
                    out.append(cand)
    return out


# ── Backtest driver ─────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    city: str
    date: str
    entry_ts: str
    snap_idx: int
    certainty: float
    kalshi_ticker: str
    kalshi_label: str
    kalshi_side: str
    poly_token: str
    poly_label: str
    poly_side: str
    pair_size: int
    entry_cost_per_pair: float
    worst_case_pnl: float
    expected_pnl_pdf: float
    expected_pnl_joint: float
    q05_pnl_joint: float
    structural: bool


def run_strategy(
    cities: list,
    date: str,
    *,
    mode: str,                       # 'all_day' | 'fixed_window' | 'threshold' | 'threshold_q05'
    fixed_window_size: int = 30,     # for 'fixed_window'
    certainty_threshold: float = 0.5,  # for 'threshold' and 'threshold_q05'
    min_q05: Optional[float] = None,  # for 'threshold_q05', skip pairs with q05_payout < cost+min_q05
    pair_size: int = DEFAULT_PAIR_SIZE_CAP,
) -> dict:
    positions: list = []
    rejects = 0  # candidates rejected by certainty/risk filter (not by structural rule)

    # Per-city certainty trace for the report
    certainty_trace: dict = {}

    for city in cities:
        forecast_pdf = load_forecast_pdf(city, date) or {}
        joint_kp = joint_kp_distribution(forecast_pdf, city=city) if forecast_pdf else {}
        snaps = list(iter_snapshots(city, date))
        if not snaps:
            continue
        n_total = len(snaps)

        # Find death index — first snapshot where book is dead and stays dead
        death_idx = n_total
        for i in range(n_total):
            ts, venues = snaps[i]
            kbs = venues.get("kalshi", [])
            pbs = venues.get("polymarket", [])
            if is_book_dead(kbs, pbs):
                # Confirm next 5 are also dead (avoid transient gaps)
                tail_dead = all(
                    is_book_dead(snaps[j][1].get("kalshi", []), snaps[j][1].get("polymarket", []))
                    for j in range(i, min(n_total, i + 5))
                )
                if tail_dead:
                    death_idx = i
                    break

        # For fixed_window: trades only allowed when idx in [death_idx - W, death_idx)
        window_start_idx = death_idx - fixed_window_size

        # Track per-city certainty at each snapshot for plotting/reporting
        city_trace = []
        opened_keys = set()

        for idx, (ts, venues) in enumerate(snaps):
            kbs = venues.get("kalshi", [])
            pbs = venues.get("polymarket", [])

            cert = compute_certainty(kbs, pbs, idx, n_total)
            city_trace.append({
                "idx": idx,
                "ts": ts,
                "n_alive": cert.n_alive,
                "mean_spread": round(cert.mean_spread, 4),
                "max_conc": round(cert.max_concentration, 4),
                "elapsed_frac": round(cert.elapsed_frac, 4),
                "score": round(cert.score, 4),
            })

            if not kbs or not pbs:
                continue
            if is_book_dead(kbs, pbs):
                continue

            # Mode gating
            if mode == "all_day":
                pass
            elif mode == "fixed_window":
                if idx < window_start_idx:
                    continue
            elif mode in ("threshold", "threshold_q05"):
                if cert.score < certainty_threshold:
                    continue
            else:
                raise ValueError(f"Unknown mode {mode}")

            cands = enumerate_pair_candidates(
                city, date, ts, idx, cert.score,
                kbs, pbs, forecast_pdf, joint_kp,
                pair_size=pair_size,
                require_loss_empty=True,
                max_total_cost=1.0,
            )

            # Sort by worst-case edge (most-positive first)
            cands.sort(key=lambda c: -(c.worst_case_payout - c.total_cost_per_pair))

            for cand in cands:
                # Apply risk filter
                if mode == "threshold_q05" and min_q05 is not None:
                    # Reject when q05_payout < entry_cost + min_q05 (probabilistic worst-case)
                    if cand.q05_payout_joint < cand.total_cost_per_pair + min_q05:
                        rejects += 1
                        continue
                key = (city, cand.kalshi_ticker, cand.poly_token,
                       cand.kalshi_side, cand.poly_side)
                if key in opened_keys:
                    continue
                if cand.pair_size < MIN_PAIR_SIZE:
                    continue

                # Settle this trade
                fee_pair = 2.0 * FEE_PER_CONTRACT * cand.pair_size
                cost = cand.total_cost_per_pair * cand.pair_size

                worst_proceeds = cand.worst_case_payout * cand.pair_size
                exp_proceeds_pdf = cand.expected_payout_pdf * cand.pair_size
                exp_proceeds_joint = cand.expected_payout_joint * cand.pair_size
                q05_proceeds = cand.q05_payout_joint * cand.pair_size

                rec = TradeRecord(
                    city=city, date=date, entry_ts=ts, snap_idx=idx,
                    certainty=cert.score,
                    kalshi_ticker=cand.kalshi_ticker,
                    kalshi_label=cand.kalshi_label,
                    kalshi_side=cand.kalshi_side,
                    poly_token=cand.poly_token,
                    poly_label=cand.poly_label,
                    poly_side=cand.poly_side,
                    pair_size=cand.pair_size,
                    entry_cost_per_pair=cand.total_cost_per_pair,
                    worst_case_pnl=worst_proceeds - cost - fee_pair,
                    expected_pnl_pdf=exp_proceeds_pdf - cost - fee_pair,
                    expected_pnl_joint=exp_proceeds_joint - cost - fee_pair,
                    q05_pnl_joint=q05_proceeds - cost - fee_pair,
                    structural=not cand.loss,
                )
                positions.append(rec)
                opened_keys.add(key)

        certainty_trace[city] = {
            "n_total": n_total,
            "death_idx": death_idx,
            "trace": city_trace,
        }

    # Aggregate
    n_trades = len(positions)
    if n_trades == 0:
        return {
            "mode": mode,
            "params": {
                "fixed_window_size": fixed_window_size,
                "certainty_threshold": certainty_threshold,
                "min_q05": min_q05,
                "pair_size": pair_size,
            },
            "n_trades": 0,
            "n_winners": 0,
            "win_rate": 0.0,
            "total_worst_case_pnl": 0.0,
            "total_expected_pnl_pdf": 0.0,
            "total_expected_pnl_joint": 0.0,
            "total_q05_pnl_joint": 0.0,
            "avg_pnl_per_trade": 0.0,
            "median_pnl_per_trade": 0.0,
            "sharpe_per_trade": 0.0,
            "max_drawdown_dollars": 0.0,
            "rejects": rejects,
            "trades": [],
            "certainty_trace": certainty_trace,
        }

    worst_pnls = [p.worst_case_pnl for p in positions]
    exp_pnls = [p.expected_pnl_joint for p in positions]
    q05_pnls = [p.q05_pnl_joint for p in positions]
    pdf_pnls = [p.expected_pnl_pdf for p in positions]

    n_winners = sum(1 for p in worst_pnls if p > 0)

    per_pair = [pnl / pos.pair_size for pnl, pos in zip(exp_pnls, positions)]
    if len(per_pair) > 1:
        s = stdev(per_pair) or 1e-9
        sharpe = mean(per_pair) / s
    else:
        sharpe = 0.0

    # Drawdown over time-ordered worst-case
    sorted_recs = sorted(positions, key=lambda p: p.entry_ts)
    cum, peak, max_dd = 0.0, 0.0, 0.0
    for rec in sorted_recs:
        cum += rec.worst_case_pnl
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)

    return {
        "mode": mode,
        "params": {
            "fixed_window_size": fixed_window_size,
            "certainty_threshold": certainty_threshold,
            "min_q05": min_q05,
            "pair_size": pair_size,
        },
        "n_trades": n_trades,
        "n_winners": n_winners,
        "win_rate": round(n_winners / n_trades, 4),
        "total_worst_case_pnl": round(sum(worst_pnls), 4),
        "total_expected_pnl_pdf": round(sum(pdf_pnls), 4),
        "total_expected_pnl_joint": round(sum(exp_pnls), 4),
        "total_q05_pnl_joint": round(sum(q05_pnls), 4),
        "avg_pnl_per_trade": round(mean(exp_pnls), 4),
        "median_pnl_per_trade": round(median(exp_pnls), 4),
        "sharpe_per_trade": round(sharpe, 4),
        "max_drawdown_dollars": round(max_dd, 4),
        "rejects": rejects,
        "trades": [
            {
                "city": p.city,
                "entry_ts": p.entry_ts,
                "snap_idx": p.snap_idx,
                "certainty": round(p.certainty, 4),
                "kalshi_ticker": p.kalshi_ticker,
                "kalshi_label": p.kalshi_label,
                "kalshi_side": p.kalshi_side,
                "poly_token": p.poly_token[:16],
                "poly_label": p.poly_label,
                "poly_side": p.poly_side,
                "pair_size": p.pair_size,
                "entry_cost_per_pair": round(p.entry_cost_per_pair, 4),
                "worst_case_pnl": round(p.worst_case_pnl, 4),
                "expected_pnl_pdf": round(p.expected_pnl_pdf, 4),
                "expected_pnl_joint": round(p.expected_pnl_joint, 4),
                "q05_pnl_joint": round(p.q05_pnl_joint, 4),
                "structural": p.structural,
            }
            for p in positions
        ],
        "certainty_trace": certainty_trace,
    }


def main():
    out_dir = Path(__file__).resolve().parent

    print("=" * 70)
    print("Strategy 8 — Late-day certainty capture")
    print("=" * 70)
    print(f"Cities: {CITIES}")
    print(f"Date: {DATE}")
    print()

    iterations = []

    # ── Baseline: all-day structural arb ──
    print("--- BASELINE: all-day structural arb (S3-style selection) ---")
    base = run_strategy(CITIES, DATE, mode="all_day")
    print(f"  n_trades: {base['n_trades']}, winners: {base['n_winners']}, "
          f"win_rate: {base['win_rate']}")
    print(f"  worst-case $: {base['total_worst_case_pnl']:.2f}, "
          f"expected (joint K-P) $: {base['total_expected_pnl_joint']:.2f}, "
          f"q05 $: {base['total_q05_pnl_joint']:.2f}")
    print()
    iterations.append({
        "version": "baseline_all_day",
        "n_trades": base["n_trades"],
        "n_winners": base["n_winners"],
        "win_rate": base["win_rate"],
        "total_pnl_dollars": base["total_expected_pnl_joint"],
        "avg_pnl_per_trade": base["avg_pnl_per_trade"],
        "median_pnl_per_trade": base["median_pnl_per_trade"],
        "sharpe_per_trade": base["sharpe_per_trade"],
        "max_drawdown_dollars": base["max_drawdown_dollars"],
        "settled_n": 0, "settled_pnl_dollars": 0.0,
        "exit_reason_counts": {"hold_to_settle": base["n_trades"]},
        "worst_case_pnl_dollars": base["total_worst_case_pnl"],
        "q05_pnl_dollars": base["total_q05_pnl_joint"],
        "expected_pnl_pdf_dollars": base["total_expected_pnl_pdf"],
        "notes": "Baseline replication of S3 selection logic with joint K-P "
                 "expected/q05 PnL columns. Hold-to-settle scoring. No "
                 "late-day filter.",
    })

    # ── v1: fixed last-30-snapshot window ──
    print("--- v1: late-day fixed window (last 30 snapshots before book death) ---")
    v1 = run_strategy(CITIES, DATE, mode="fixed_window", fixed_window_size=30)
    print(f"  n_trades: {v1['n_trades']}, winners: {v1['n_winners']}, "
          f"win_rate: {v1['win_rate']}")
    print(f"  worst-case $: {v1['total_worst_case_pnl']:.2f}, "
          f"expected (joint) $: {v1['total_expected_pnl_joint']:.2f}, "
          f"q05 $: {v1['total_q05_pnl_joint']:.2f}")
    print()
    iterations.append({
        "version": "v1_fixed_window_30",
        "n_trades": v1["n_trades"],
        "n_winners": v1["n_winners"],
        "win_rate": v1["win_rate"],
        "total_pnl_dollars": v1["total_expected_pnl_joint"],
        "avg_pnl_per_trade": v1["avg_pnl_per_trade"],
        "median_pnl_per_trade": v1["median_pnl_per_trade"],
        "sharpe_per_trade": v1["sharpe_per_trade"],
        "max_drawdown_dollars": v1["max_drawdown_dollars"],
        "settled_n": 0, "settled_pnl_dollars": 0.0,
        "exit_reason_counts": {"hold_to_settle": v1["n_trades"]},
        "worst_case_pnl_dollars": v1["total_worst_case_pnl"],
        "q05_pnl_dollars": v1["total_q05_pnl_joint"],
        "expected_pnl_pdf_dollars": v1["total_expected_pnl_pdf"],
        "fixed_window_size": 30,
        "notes": "Only enter during last 30 alive-snapshot window before per-city "
                 "book death. Hold-to-settle scoring under joint K-P distribution.",
    })

    # ── v2: certainty threshold sweep (find best) ──
    print("--- v2: adaptive certainty threshold sweep ---")
    threshold_results = []
    for th in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        r = run_strategy(CITIES, DATE, mode="threshold", certainty_threshold=th)
        threshold_results.append({
            "threshold": th,
            "n_trades": r["n_trades"],
            "winners": r["n_winners"],
            "worst_case_pnl": r["total_worst_case_pnl"],
            "expected_pnl_joint": r["total_expected_pnl_joint"],
            "q05_pnl": r["total_q05_pnl_joint"],
            "sharpe": r["sharpe_per_trade"],
        })
        print(f"  threshold={th:.2f}: n={r['n_trades']}, "
              f"worst=${r['total_worst_case_pnl']:.2f}, "
              f"exp=${r['total_expected_pnl_joint']:.2f}, "
              f"q05=${r['total_q05_pnl_joint']:.2f}, sharpe={r['sharpe_per_trade']:.2f}")

    # Pick best by expected_pnl_joint as headline
    best = max(threshold_results, key=lambda x: x["expected_pnl_joint"])
    best_th = best["threshold"]
    print(f"  best threshold: {best_th}")
    v2 = run_strategy(CITIES, DATE, mode="threshold", certainty_threshold=best_th)
    iterations.append({
        "version": f"v2_threshold_{best_th}",
        "n_trades": v2["n_trades"],
        "n_winners": v2["n_winners"],
        "win_rate": v2["win_rate"],
        "total_pnl_dollars": v2["total_expected_pnl_joint"],
        "avg_pnl_per_trade": v2["avg_pnl_per_trade"],
        "median_pnl_per_trade": v2["median_pnl_per_trade"],
        "sharpe_per_trade": v2["sharpe_per_trade"],
        "max_drawdown_dollars": v2["max_drawdown_dollars"],
        "settled_n": 0, "settled_pnl_dollars": 0.0,
        "exit_reason_counts": {"hold_to_settle": v2["n_trades"]},
        "worst_case_pnl_dollars": v2["total_worst_case_pnl"],
        "q05_pnl_dollars": v2["total_q05_pnl_joint"],
        "expected_pnl_pdf_dollars": v2["total_expected_pnl_pdf"],
        "certainty_threshold": best_th,
        "threshold_sweep": threshold_results,
        "notes": (
            "Adaptive certainty threshold. Score = 0.30*spread + 0.40*conc "
            "+ 0.20*alive + 0.10*time. Picked best threshold by expected PnL "
            "under joint K-P. Hold-to-settle scoring."
        ),
    })

    # ── v3: best threshold + joint K-P q05 risk filter ──
    print("--- v3: best threshold + q05 risk filter (joint K-P) ---")
    v3_results = []
    for min_q05 in [-0.10, -0.05, 0.0, 0.05, 0.10]:
        r = run_strategy(
            CITIES, DATE, mode="threshold_q05",
            certainty_threshold=best_th, min_q05=min_q05,
        )
        v3_results.append({
            "min_q05": min_q05,
            "n_trades": r["n_trades"],
            "winners": r["n_winners"],
            "worst": r["total_worst_case_pnl"],
            "exp": r["total_expected_pnl_joint"],
            "q05": r["total_q05_pnl_joint"],
            "rejects": r["rejects"],
        })
        print(f"  min_q05={min_q05:+.2f}: n={r['n_trades']}, rejects={r['rejects']}, "
              f"worst=${r['total_worst_case_pnl']:.2f}, "
              f"exp=${r['total_expected_pnl_joint']:.2f}, "
              f"q05=${r['total_q05_pnl_joint']:.2f}")
    # Pick the min_q05 that maximizes expected PnL given non-zero trades
    nonempty = [r for r in v3_results if r["n_trades"] > 0]
    best_q05 = max(nonempty, key=lambda x: x["exp"]) if nonempty else v3_results[0]
    print(f"  best min_q05: {best_q05['min_q05']}")

    v3 = run_strategy(
        CITIES, DATE, mode="threshold_q05",
        certainty_threshold=best_th, min_q05=best_q05["min_q05"],
    )
    iterations.append({
        "version": f"v3_threshold_{best_th}_q05_{best_q05['min_q05']}",
        "n_trades": v3["n_trades"],
        "n_winners": v3["n_winners"],
        "win_rate": v3["win_rate"],
        "total_pnl_dollars": v3["total_expected_pnl_joint"],
        "avg_pnl_per_trade": v3["avg_pnl_per_trade"],
        "median_pnl_per_trade": v3["median_pnl_per_trade"],
        "sharpe_per_trade": v3["sharpe_per_trade"],
        "max_drawdown_dollars": v3["max_drawdown_dollars"],
        "settled_n": 0, "settled_pnl_dollars": 0.0,
        "exit_reason_counts": {"hold_to_settle": v3["n_trades"]},
        "worst_case_pnl_dollars": v3["total_worst_case_pnl"],
        "q05_pnl_dollars": v3["total_q05_pnl_joint"],
        "expected_pnl_pdf_dollars": v3["total_expected_pnl_pdf"],
        "certainty_threshold": best_th,
        "min_q05": best_q05["min_q05"],
        "min_q05_sweep": v3_results,
        "rejects": v3["rejects"],
        "notes": (
            "Same as v2 plus a probabilistic-worst-case (q05) risk filter "
            "computed under the joint K-P distribution. Filters out pairs "
            "where 5%-tail payout is below entry cost + min_q05."
        ),
    })

    # ── Save outputs ──
    metrics = {
        "strategy_name": "late_day_certainty",
        "iterations": iterations,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    # Per-iteration trade dumps
    (out_dir / "trades_baseline.json").write_text(json.dumps(base["trades"], indent=2, default=str))
    (out_dir / "trades_v1.json").write_text(json.dumps(v1["trades"], indent=2, default=str))
    (out_dir / "trades_v2.json").write_text(json.dumps(v2["trades"], indent=2, default=str))
    (out_dir / "trades_v3.json").write_text(json.dumps(v3["trades"], indent=2, default=str))

    # Certainty trace dump (for the regime-analysis "plot")
    (out_dir / "certainty_trace.json").write_text(json.dumps(base["certainty_trace"], indent=2, default=str))

    # Render an ASCII regime-analysis table and save
    render_regime_table(base["certainty_trace"], v2["trades"], out_dir / "regime_analysis.txt")

    print()
    print("Wrote", out_dir / "metrics.json")
    print("Wrote", out_dir / "regime_analysis.txt")


def render_regime_table(trace_by_city: dict, trades: list, out_path: Path):
    """ASCII regime-analysis: certainty score over snapshots per city.
    Markers at snapshots where the strategy took trades."""
    lines = []
    trades_by_city: dict = {}
    for t in trades:
        trades_by_city.setdefault(t["city"], set()).add(t["snap_idx"])
    lines.append("Regime analysis: certainty score across the trading day per city.")
    lines.append("Each line shows snapshot idx, score (1-pct bins), * = trade taken at that idx.")
    lines.append("")
    for city, info in trace_by_city.items():
        n_total = info["n_total"]
        death = info["death_idx"]
        traces = info["trace"]
        lines.append(f"== {city}  (n_snapshots={n_total}, book_death_idx={death}) ==")
        # Sample every 10th snapshot to fit
        rendered = []
        for tr in traces[::10]:
            score = tr["score"]
            bar_len = int(round(score * 40))
            bar = "#" * bar_len + "." * (40 - bar_len)
            mark = "*" if tr["idx"] in trades_by_city.get(city, set()) else " "
            rendered.append(
                f"  {mark} idx={tr['idx']:3d}  score={score:.2f}  alive={tr['n_alive']:2d}  "
                f"sp={tr['mean_spread']:.3f}  conc={tr['max_conc']:.2f}  |{bar}|"
            )
        # Also include any snapshot at which a trade was taken (in case missed by sampling)
        trade_idxs = sorted(trades_by_city.get(city, set()))
        for ti in trade_idxs:
            if ti % 10 == 0:
                continue
            tr = traces[ti] if ti < len(traces) else None
            if tr is None:
                continue
            bar_len = int(round(tr["score"] * 40))
            bar = "#" * bar_len + "." * (40 - bar_len)
            rendered.append(
                f"  * idx={tr['idx']:3d}  score={tr['score']:.2f}  alive={tr['n_alive']:2d}  "
                f"sp={tr['mean_spread']:.3f}  conc={tr['max_conc']:.2f}  |{bar}|  (trade)"
            )
        lines.extend(rendered)
        lines.append("")
    out_path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
