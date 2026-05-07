"""
Strategy 9 — Cross-venue PDF arbitrage.

Idea
----
Round 1 lesson: the NWS forecast PDF is too stale to value brackets against
(off by 2°F in 4/5 cities on 2026-05-06). But each venue's bracket prices,
themselves, encode an implied per-degree PMF that updates intra-day. Use
each venue's PMF as a prior for the OTHER venue's bracket prices, after
applying the empirical (K - P) divergence shift.

Concretely, at each snapshot:
  1. pmf_K(d) = parse_kalshi_bins(brackets) — implied PMF on Kalshi's reading.
  2. pmf_P(d) = parse_polymarket_bins(brackets) — implied PMF on Polymarket's reading.
  3. Translate K's distribution into P's space:
       pmf_P_predicted_by_K(p) = sum_k pmf_K(k) * P(Δ = k - p)
     where Δ is the empirical city-specific (K − P) histogram.
  4. For each Polymarket bracket B with price P_market(B):
       E_K[B] = sum_{d in B.degrees} pmf_P_predicted_by_K(d)
       gap = E_K[B] − P_market(B)
     If |gap| > threshold AND the bracket has fillable depth, signal.
  5. Symmetrically translate P's view into K's space and signal on Kalshi
     brackets.

Iterations:
  - v1: directional single-leg trade based on the disagreement.
        Hold-to-settle scored under the city's joint K-P distribution
        using the venues' OWN consensus as the K marginal (NOT NWS).
        We expect this to be poor because we are exposed to BOTH venue-
        divergence noise AND any systematic bias in the implied PMFs.
  - v2: hedged version. For every directional Poly signal, simultaneously
        take a hedging Kalshi position on the same temperature region
        (shifted by the average Δ). Score the pair under joint K-P with
        E[payout] and 5%-quantile payout filters.
  - v3: only enter when BOTH the NWS PDF (centered on the official
        forecast) AND the cross-venue gap agree on direction. This
        adds a second prior to reduce false positives from one venue
        running stale.

Reality checks / honest caveats:
  - parse_kalshi_bins / parse_polymarket_bins normalize to sum=1 (verified).
    So pmf_K and pmf_P don't suffer from "midpoints don't sum to 1".
  - pmf_K is on K_high; pmf_P is on P_high. They live in different spaces;
    comparing them directly under-uses the +1°F shift. The convolution
    above is the rigorous translation.
  - 188 historical (city, date) pairs is thin for per-city Δ histograms
    (Miami n=61, others smaller). We pool fall-back when a city has < 30
    rows and document this.
  - Without 2026-05-06 settlements, all PnL is "expected under joint K-P
    distribution". Worst-case is the 5% quantile under that joint.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median, stdev
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.loaders import (
    iter_snapshots,
    load_forecast_pdf,
    venue_divergence_histogram,
    walk_ladder_buy,
    walk_ladder_sell,
)
from strategy.parsers import (
    parse_kalshi_bins,
    parse_polymarket_bins,
    _parse_kalshi_range,
    _parse_polymarket_range,
)


CITIES = ["Miami", "LA", "Austin", "San Francisco", "Seattle"]
DATE = "2026-05-06"

FEE_PER_CONTRACT = 0.005
DEFAULT_PAIR_SIZE = 25
MIN_DEPTH = 5
MIN_CITY_DELTA_N = 30  # if fewer pairs, fall back to pooled Δ histogram


# ── Δ histogram helpers ───────────────────────────────────────────────────


def _city_delta_count(city: str) -> int:
    """Approximate count of historical pairs feeding the per-city Δ hist."""
    raw = venue_divergence_histogram(city, normalize=False)
    return int(sum(raw.values()))


def get_delta_hist(city: str) -> tuple[dict[int, float], str]:
    """Return (Δ histogram, source) for `city`. Falls back to pooled if
    sample size below MIN_CITY_DELTA_N."""
    n = _city_delta_count(city)
    if n >= MIN_CITY_DELTA_N:
        return venue_divergence_histogram(city), f"per_city_n={n}"
    return venue_divergence_histogram(), f"pooled_n=188_(city_n={n})"


# ── PMF translation ──────────────────────────────────────────────────────


def translate_pmf_K_to_P_space(pmf_K: dict[int, float],
                                delta_hist: dict[int, float]) -> dict[int, float]:
    """Given Kalshi's implied PMF on K_high and the empirical Δ = K − P
    histogram, return predicted PMF on P_high:
        pmf_P_pred(p) = sum_k pmf_K(k) * P(Δ = k - p)
    """
    pmf_P_pred: dict[int, float] = {}
    for k, p_k in pmf_K.items():
        for d, p_d in delta_hist.items():
            p = k - d
            pmf_P_pred[p] = pmf_P_pred.get(p, 0.0) + p_k * p_d
    return pmf_P_pred


def translate_pmf_P_to_K_space(pmf_P: dict[int, float],
                                delta_hist: dict[int, float]) -> dict[int, float]:
    """Predicted PMF on K_high given Polymarket's PMF and Δ:
        pmf_K_pred(k) = sum_p pmf_P(p) * P(Δ = k - p)
    """
    pmf_K_pred: dict[int, float] = {}
    for p, p_p in pmf_P.items():
        for d, p_d in delta_hist.items():
            k = p + d
            pmf_K_pred[k] = pmf_K_pred.get(k, 0.0) + p_p * p_d
    return pmf_K_pred


def bracket_yes_prob_under_pmf(degrees: list[int], pmf: dict[int, float]) -> float:
    return sum(pmf.get(d, 0.0) for d in degrees)


# ── Side / ladder helpers ────────────────────────────────────────────────


def yes_ask_ladder(b: dict) -> list:
    return list(b.get("yes_ask_ladder") or [])


def no_ask_ladder(b: dict) -> list:
    """Buying NO ≡ selling YES into bids ⇒ NO ask = 1 − YES bid price."""
    return [(round(1.0 - p, 4), q) for p, q in (b.get("yes_bid_ladder") or [])]


def best_yes_ask(b: dict) -> Optional[float]:
    return b.get("best_yes_ask")


def best_no_ask(b: dict) -> Optional[float]:
    yb = b.get("best_yes_bid")
    return None if yb is None else round(1.0 - yb, 4)


# ── Pair-payout closures (for joint-distribution scoring) ────────────────


def make_single_leg_payout_fn(venue: str, side: str, degrees: list[int],
                              entry_cost_per_contract: float, size: int):
    """Return f(k, p) → $/pair payout for a SINGLE leg. Settlement uses
    the venue's own reading (Kalshi → k, Polymarket → p)."""
    deg_set = set(degrees)
    def fn(k: int, p: int) -> float:
        actual = k if venue == "kalshi" else p
        in_bracket = actual in deg_set
        if side == "yes_long":
            payout_per_contract = 1.0 if in_bracket else 0.0
        else:  # no_long
            payout_per_contract = 0.0 if in_bracket else 1.0
        return (payout_per_contract - entry_cost_per_contract) * size
    return fn


def make_pair_payout_fn(legs: list[tuple[str, str, list[int], float, int]]):
    """legs is a list of (venue, side, degrees, entry_cost_per_contract, size).
    Returns f(k, p) → total $ payout across legs."""
    leg_fns = [make_single_leg_payout_fn(*leg) for leg in legs]
    def fn(k: int, p: int) -> float:
        return sum(lf(k, p) for lf in leg_fns)
    return fn


# ── Joint K,P distribution constructed FROM venue PMFs (not NWS) ─────────


def joint_from_venues(pmf_K: dict[int, float],
                      delta_hist: dict[int, float]) -> dict[tuple[int, int], float]:
    """Build joint P(K=k, P=p) from the venue-implied PMF on K and the
    empirical Δ histogram, exactly mirroring loaders.joint_kp_distribution
    but using the live VENUE PMF (not NWS)."""
    joint: dict[tuple[int, int], float] = {}
    for k, p_k in pmf_K.items():
        for d, p_d in delta_hist.items():
            p = k - d
            joint[(k, p)] = joint.get((k, p), 0.0) + p_k * p_d
    return joint


def joint_from_venues_polymarket_marginal(
    pmf_P: dict[int, float], delta_hist: dict[int, float],
) -> dict[tuple[int, int], float]:
    """Build joint P(K=k, P=p) treating live pmf_P as the marginal on P:
        P(K=k, P=p) = pmf_P(p) * P(Δ = k − p)
    Used when we mistrust Kalshi's marginal (e.g. when the trade thesis
    is 'Kalshi is overpricing this'). Symmetric counterpart of
    `joint_from_venues`.
    """
    joint: dict[tuple[int, int], float] = {}
    for p, pp in pmf_P.items():
        for d, pd in delta_hist.items():
            k = p + d
            joint[(k, p)] = joint.get((k, p), 0.0) + pp * pd
    return joint


def joint_consensus(
    pmf_K: dict[int, float], pmf_P: dict[int, float],
    delta_hist: dict[int, float],
) -> dict[tuple[int, int], float]:
    """Build joint with a CONSENSUS prior: average the two venues'
    implied K-marginals after translating Polymarket into K's space.
    More robust when neither venue is obviously stale."""
    pmf_K_from_P = translate_pmf_P_to_K_space(pmf_P, delta_hist)
    joint: dict[tuple[int, int], float] = {}
    keys = set(pmf_K) | set(pmf_K_from_P)
    for k in keys:
        avg_k = 0.5 * pmf_K.get(k, 0.0) + 0.5 * pmf_K_from_P.get(k, 0.0)
        if avg_k <= 0:
            continue
        for d, pd in delta_hist.items():
            p = k - d
            joint[(k, p)] = joint.get((k, p), 0.0) + avg_k * pd
    # Renormalize defensively
    total = sum(joint.values())
    if total > 0:
        for key in joint:
            joint[key] /= total
    return joint


def joint_average_K(pmf_K1: dict[int, float], pmf_K2: dict[int, float],
                    delta_hist: dict[int, float],
                    weight_K1: float = 0.5) -> dict[tuple[int, int], float]:
    """Build joint using a CONSENSUS K-marginal: weight pmf_K1 (live K) and
    pmf_K2 (translated from P). This is the 'two-prior' joint."""
    joint: dict[tuple[int, int], float] = {}
    keys = set(pmf_K1) | set(pmf_K2)
    for k in keys:
        p_k = weight_K1 * pmf_K1.get(k, 0.0) + (1 - weight_K1) * pmf_K2.get(k, 0.0)
        if p_k <= 0:
            continue
        for d, p_d in delta_hist.items():
            p = k - d
            joint[(k, p)] = joint.get((k, p), 0.0) + p_k * p_d
    return joint


def expected_under_joint(pair_fn, joint: dict[tuple[int, int], float]) -> float:
    return sum(joint[k, p] * pair_fn(k, p) for (k, p) in joint)


def quantile_under_joint(pair_fn, joint: dict[tuple[int, int], float],
                         q: float = 0.05) -> float:
    pairs = sorted(((pair_fn(k, p), w) for (k, p), w in joint.items()),
                   key=lambda t: t[0])
    cum = 0.0
    for payout, w in pairs:
        cum += w
        if cum >= q:
            return payout
    return pairs[-1][0] if pairs else 0.0


# ── Signal types ─────────────────────────────────────────────────────────


@dataclass
class DirectionalSignal:
    """A single-leg directional bet."""
    city: str
    ts: str
    venue: str               # 'kalshi' | 'polymarket'
    bracket_id: str
    bracket_label: str
    degrees: list[int]
    side: str                # 'yes_long' | 'no_long'
    entry_avg_price: float
    size: int
    fair_value_yes: float    # E_other[YES] under translated PMF
    market_yes_price: float  # market YES price (best_yes_ask if yes_long, 1-best_yes_bid for no_long)
    gap: float               # signed gap (positive = market underprices YES vs other-venue view)
    expected_payout: float   # per-pair $ payout under joint K,P
    q05_payout: float        # 5th percentile $ payout under joint K,P
    nws_yes_prob: float      # NWS-implied YES prob (for v3 / diagnostics)


@dataclass
class HedgedPair:
    """Two legs scored jointly against the same joint K,P distribution."""
    city: str
    ts: str
    primary: DirectionalSignal
    hedge: DirectionalSignal
    expected_payout: float
    q05_payout: float
    pair_size: int
    total_entry_cost: float


# ── v1: directional cross-venue PDF disagreement ─────────────────────────


def find_directional_signals(
    city: str, date: str, ts: str,
    kalshi_brackets: list, poly_brackets: list,
    delta_hist: dict[int, float],
    forecast_pdf: dict[int, float],
    threshold: float,
    pair_size: int,
) -> list[DirectionalSignal]:
    """Enumerate directional signals on BOTH venues based on the other
    venue's PMF translated through Δ."""
    pmf_K = parse_kalshi_bins(kalshi_brackets)
    pmf_P = parse_polymarket_bins(poly_brackets)
    if not pmf_K or not pmf_P:
        return []

    pmf_P_pred_by_K = translate_pmf_K_to_P_space(pmf_K, delta_hist)
    pmf_K_pred_by_P = translate_pmf_P_to_K_space(pmf_P, delta_hist)

    signals: list[DirectionalSignal] = []

    # --- Polymarket brackets (priced on P_high; K's view via translation)
    for b in poly_brackets:
        degrees = _parse_polymarket_range(b.get("question", ""))
        if not degrees:
            continue
        fv_yes = bracket_yes_prob_under_pmf(degrees, pmf_P_pred_by_K)
        nws_yes = bracket_yes_prob_under_pmf(degrees, forecast_pdf)
        ya = best_yes_ask(b)
        na = best_no_ask(b)

        # YES underpriced: buy YES.
        if ya is not None and 0.02 <= ya <= 0.95:
            gap = fv_yes - ya
            if gap > threshold:
                fill = walk_ladder_buy(yes_ask_ladder(b), pair_size)
                if fill is not None:
                    avg, _ = fill
                    if avg <= 0.95 and avg <= ya + 0.02:
                        signals.append(DirectionalSignal(
                            city=city, ts=ts, venue="polymarket",
                            bracket_id=b.get("token_id", ""),
                            bracket_label=b.get("question", ""),
                            degrees=degrees, side="yes_long",
                            entry_avg_price=avg, size=pair_size,
                            fair_value_yes=fv_yes, market_yes_price=ya,
                            gap=gap, expected_payout=0.0, q05_payout=0.0,
                            nws_yes_prob=nws_yes,
                        ))

        # NO underpriced (i.e. YES overpriced): buy NO.
        if na is not None and 0.02 <= na <= 0.95:
            yes_bid = b.get("best_yes_bid")
            gap = (1.0 - fv_yes) - na  # FV(NO) − ask(NO)
            if gap > threshold and yes_bid is not None:
                fill = walk_ladder_buy(no_ask_ladder(b), pair_size)
                if fill is not None:
                    avg, _ = fill
                    if avg <= 0.95 and avg <= na + 0.02:
                        signals.append(DirectionalSignal(
                            city=city, ts=ts, venue="polymarket",
                            bracket_id=b.get("token_id", ""),
                            bracket_label=b.get("question", ""),
                            degrees=degrees, side="no_long",
                            entry_avg_price=avg, size=pair_size,
                            fair_value_yes=1.0 - fv_yes,
                            market_yes_price=na, gap=gap,
                            expected_payout=0.0, q05_payout=0.0,
                            nws_yes_prob=1.0 - nws_yes,
                        ))

    # --- Kalshi brackets (priced on K_high; P's view via translation)
    for b in kalshi_brackets:
        degrees = _parse_kalshi_range(b.get("subtitle", ""))
        if not degrees:
            continue
        fv_yes = bracket_yes_prob_under_pmf(degrees, pmf_K_pred_by_P)
        nws_yes = bracket_yes_prob_under_pmf(degrees, forecast_pdf)
        ya = best_yes_ask(b)
        na = best_no_ask(b)

        if ya is not None and 0.02 <= ya <= 0.95:
            gap = fv_yes - ya
            if gap > threshold:
                fill = walk_ladder_buy(yes_ask_ladder(b), pair_size)
                if fill is not None:
                    avg, _ = fill
                    if avg <= 0.95 and avg <= ya + 0.02:
                        signals.append(DirectionalSignal(
                            city=city, ts=ts, venue="kalshi",
                            bracket_id=b.get("ticker", ""),
                            bracket_label=b.get("subtitle", ""),
                            degrees=degrees, side="yes_long",
                            entry_avg_price=avg, size=pair_size,
                            fair_value_yes=fv_yes, market_yes_price=ya,
                            gap=gap, expected_payout=0.0, q05_payout=0.0,
                            nws_yes_prob=nws_yes,
                        ))

        if na is not None and 0.02 <= na <= 0.95:
            yes_bid = b.get("best_yes_bid")
            gap = (1.0 - fv_yes) - na
            if gap > threshold and yes_bid is not None:
                fill = walk_ladder_buy(no_ask_ladder(b), pair_size)
                if fill is not None:
                    avg, _ = fill
                    if avg <= 0.95 and avg <= na + 0.02:
                        signals.append(DirectionalSignal(
                            city=city, ts=ts, venue="kalshi",
                            bracket_id=b.get("ticker", ""),
                            bracket_label=b.get("subtitle", ""),
                            degrees=degrees, side="no_long",
                            entry_avg_price=avg, size=pair_size,
                            fair_value_yes=1.0 - fv_yes,
                            market_yes_price=na, gap=gap,
                            expected_payout=0.0, q05_payout=0.0,
                            nws_yes_prob=1.0 - nws_yes,
                        ))

    return signals


def score_signal_under_joint(sig: DirectionalSignal,
                              joint: dict[tuple[int, int], float]) -> tuple[float, float]:
    """Return (E[payout], q05[payout]) for a single-leg signal in $."""
    fn = make_single_leg_payout_fn(
        sig.venue, sig.side, sig.degrees, sig.entry_avg_price, sig.size,
    )
    e = expected_under_joint(fn, joint)
    q = quantile_under_joint(fn, joint, q=0.05)
    # Subtract entry-side fee (no exit fee — held to settlement; losing
    # contracts settle at $0 with no further charge)
    fees = FEE_PER_CONTRACT * sig.size
    return (e - fees, q - fees)


# ── v2: hedged-pair construction ─────────────────────────────────────────


def find_hedge_for_signal(
    sig: DirectionalSignal,
    kalshi_brackets: list, poly_brackets: list,
    delta_hist: dict[int, float],
    pair_size: int,
) -> Optional[DirectionalSignal]:
    """Construct a hedge on the OTHER venue. Strategy:

      Primary: long YES on bracket B (degrees D) on venue V → pays when
               V's reading lands in D.
      Hedge:   long YES on the OTHER venue's brackets covering the SHIFTED
               equivalent (translated by mean Δ).

    For a Polymarket primary on degrees D, hedge on Kalshi covering the
    shifted region {d + round(mean_delta) for d in D} ∩ available Kalshi
    brackets. Side is the OPPOSITE of primary's bracket-direction so a
    correlated outcome leaves both positions correlated.

    Concretely: if primary is YES_long on Poly bracket B_P (pays when
    P_high ∈ B_P), the natural hedge is NO_long on Kalshi bracket B_K
    where B_K covers the shifted region. If both venues agree (Δ ≈ +1°F)
    and the high actually IS in the predicted region, primary wins ($1)
    and hedge wins (Kalshi reading NOT in B_K — only if our shifted
    region is wrong). Hmm — this is subtle. Let's just be explicit:
        primary YES on V, B_V → wants V_reading ∈ B_V
        hedge   NO  on W, B_W → wants W_reading ∉ B_W
    For the pair to be profitable across more outcomes, B_W should be
    the bracket on W that LARGELY OVERLAPS B_V (after Δ shift). Then if
    the reading IS in the predicted region on both venues, primary wins
    and hedge loses. If the reading is OUTSIDE on both, hedge wins and
    primary loses. The hedge protects against "primary's view is wrong
    AND the other venue agrees with the original price."

    This is the canonical hedged pair from S2.
    """
    if sig.side == "yes_long":
        hedge_side = "no_long"
    else:
        hedge_side = "yes_long"

    # Determine mean delta (rounded) for shift
    mean_delta_int = int(round(sum(d * w for d, w in delta_hist.items())))

    # Hedge on the OTHER venue
    if sig.venue == "polymarket":
        # Hedge on Kalshi. Shift degrees: K = P + Δ.
        target_degrees = set(d + mean_delta_int for d in sig.degrees)
        candidate_brackets = kalshi_brackets
        is_kalshi = True
    else:
        # Hedge on Polymarket. Shift degrees: P = K - Δ.
        target_degrees = set(d - mean_delta_int for d in sig.degrees)
        candidate_brackets = poly_brackets
        is_kalshi = False

    # Find the bracket whose degrees BEST overlap target_degrees
    best = None
    best_overlap = 0
    for b in candidate_brackets:
        if is_kalshi:
            degs = _parse_kalshi_range(b.get("subtitle", ""))
        else:
            degs = _parse_polymarket_range(b.get("question", ""))
        if not degs:
            continue
        overlap = len(set(degs) & target_degrees)
        if overlap > best_overlap:
            best_overlap = overlap
            best = (b, degs)

    if best is None or best_overlap == 0:
        return None
    hedge_bracket, hedge_degrees = best

    # Get fillable price for hedge
    if hedge_side == "yes_long":
        ladder = yes_ask_ladder(hedge_bracket)
        market_ask = best_yes_ask(hedge_bracket)
    else:
        ladder = no_ask_ladder(hedge_bracket)
        market_ask = best_no_ask(hedge_bracket)
    if market_ask is None or market_ask < 0.02 or market_ask > 0.95:
        return None
    fill = walk_ladder_buy(ladder, pair_size)
    if fill is None:
        return None
    avg, _ = fill
    if avg > 0.95:
        return None

    hedge_venue = "kalshi" if is_kalshi else "polymarket"
    hedge_id = (hedge_bracket.get("ticker", "") if is_kalshi
                else hedge_bracket.get("token_id", ""))
    hedge_label = (hedge_bracket.get("subtitle", "") if is_kalshi
                   else hedge_bracket.get("question", ""))

    return DirectionalSignal(
        city=sig.city, ts=sig.ts, venue=hedge_venue,
        bracket_id=hedge_id, bracket_label=hedge_label,
        degrees=hedge_degrees, side=hedge_side,
        entry_avg_price=avg, size=pair_size,
        fair_value_yes=0.0, market_yes_price=market_ask,
        gap=0.0, expected_payout=0.0, q05_payout=0.0,
        nws_yes_prob=0.0,
    )


def score_pair_under_joint(primary: DirectionalSignal,
                            hedge: DirectionalSignal,
                            joint: dict[tuple[int, int], float]) -> tuple[float, float, float]:
    """Returns (E[payout], q05[payout], total_entry_cost) for the hedged pair, in $."""
    legs = [
        (primary.venue, primary.side, primary.degrees,
         primary.entry_avg_price, primary.size),
        (hedge.venue, hedge.side, hedge.degrees,
         hedge.entry_avg_price, hedge.size),
    ]
    fn = make_pair_payout_fn(legs)
    e = expected_under_joint(fn, joint)
    q = quantile_under_joint(fn, joint, q=0.05)
    # Fees: 2 entry contracts (each leg) — settlement requires no further
    # charge. So fee = 2 * fee_per_contract * pair_size.
    fees = 2.0 * FEE_PER_CONTRACT * primary.size
    cost = (primary.entry_avg_price + hedge.entry_avg_price) * primary.size
    return (e - fees, q - fees, cost)


# ── Backtest drivers ─────────────────────────────────────────────────────


def _build_joint_for_signal(sig: DirectionalSignal,
                              pmf_K: dict[int, float], pmf_P: dict[int, float],
                              delta_hist: dict[int, float]) -> dict[tuple[int, int], float]:
    """Score a directional signal against a CONSENSUS prior (50/50 average
    of both venues' implied marginals after Δ-translation).

    Note: scoring against the OTHER venue's marginal would be circular —
    by construction, if X believes Y is underpriced, X's view says the
    trade is positive. Consensus is the neutral prior."""
    return joint_consensus(pmf_K, pmf_P, delta_hist)


def run_v1(threshold: float = 0.05, pair_size: int = DEFAULT_PAIR_SIZE) -> dict:
    """v1: directional single-leg signals, hold-to-settle scoring under
    a CONSENSUS joint K,P (average both venues' implied marginals)."""
    raw_signal_count = 0
    trades = []

    for city in CITIES:
        delta_hist, delta_src = get_delta_hist(city)
        forecast_pdf = load_forecast_pdf(city, DATE) or {}
        snaps = list(iter_snapshots(city, DATE))
        if not snaps:
            continue

        opened_keys: set[tuple] = set()
        for ts, venues in snaps:
            kbs = venues.get("kalshi", [])
            pbs = venues.get("polymarket", [])
            if not kbs or not pbs:
                continue
            sigs = find_directional_signals(
                city, DATE, ts, kbs, pbs, delta_hist, forecast_pdf,
                threshold=threshold, pair_size=pair_size,
            )
            raw_signal_count += len(sigs)
            if not sigs:
                continue

            pmf_K = parse_kalshi_bins(kbs)
            pmf_P = parse_polymarket_bins(pbs)

            # Sort by gap (best first) and dedup
            sigs.sort(key=lambda s: -s.gap)
            for sig in sigs:
                key = (city, sig.venue, sig.bracket_id, sig.side)
                if key in opened_keys:
                    continue
                joint = _build_joint_for_signal(sig, pmf_K, pmf_P, delta_hist)
                e, q = score_signal_under_joint(sig, joint)
                sig.expected_payout = e
                sig.q05_payout = q
                trades.append({
                    "city": city, "ts": ts, "venue": sig.venue,
                    "bracket_id": sig.bracket_id,
                    "bracket_label": sig.bracket_label,
                    "degrees": sig.degrees, "side": sig.side,
                    "entry_price": sig.entry_avg_price,
                    "size": sig.size,
                    "fv_yes": sig.fair_value_yes,
                    "market_yes_price": sig.market_yes_price,
                    "gap": sig.gap,
                    "expected_pnl": e,
                    "q05_pnl": q,
                    "nws_yes_prob": sig.nws_yes_prob,
                    "delta_src": delta_src,
                })
                opened_keys.add(key)

    return _summarize_trades(trades, "v1")


def run_v2(threshold: float = 0.05,
           min_q05: float = 0.0,
           require_positive_e: bool = True,
           pair_size: int = DEFAULT_PAIR_SIZE) -> dict:
    """v2: hedged pairs. For each directional signal, build a hedge on the
    OTHER venue (shifted bracket, opposite side). Score the pair under the
    joint distribution. Only keep when E[payout] > 0 and q05[payout] > min_q05."""
    raw_signal_count = 0
    trades = []
    skipped_no_hedge = 0
    skipped_negative_e = 0
    skipped_q05 = 0

    for city in CITIES:
        delta_hist, delta_src = get_delta_hist(city)
        forecast_pdf = load_forecast_pdf(city, DATE) or {}
        snaps = list(iter_snapshots(city, DATE))
        if not snaps:
            continue

        opened_keys: set[tuple] = set()
        for ts, venues in snaps:
            kbs = venues.get("kalshi", [])
            pbs = venues.get("polymarket", [])
            if not kbs or not pbs:
                continue
            sigs = find_directional_signals(
                city, DATE, ts, kbs, pbs, delta_hist, forecast_pdf,
                threshold=threshold, pair_size=pair_size,
            )
            raw_signal_count += len(sigs)
            if not sigs:
                continue

            pmf_K = parse_kalshi_bins(kbs)
            pmf_P = parse_polymarket_bins(pbs)
            # For hedged pairs, score against a CONSENSUS joint (averaging
            # both venues' implied marginals) — neither venue is privileged.
            joint = joint_consensus(pmf_K, pmf_P, delta_hist)

            sigs.sort(key=lambda s: -s.gap)
            for sig in sigs:
                key = (city, sig.venue, sig.bracket_id, sig.side)
                if key in opened_keys:
                    continue
                hedge = find_hedge_for_signal(sig, kbs, pbs, delta_hist, pair_size)
                if hedge is None:
                    skipped_no_hedge += 1
                    continue
                # Don't double-open hedges either
                hedge_key = (city, hedge.venue, hedge.bracket_id, hedge.side)
                if hedge_key in opened_keys:
                    continue
                e, q, cost = score_pair_under_joint(sig, hedge, joint)
                if require_positive_e and e <= 0:
                    skipped_negative_e += 1
                    continue
                if q < min_q05:
                    skipped_q05 += 1
                    continue
                trades.append({
                    "city": city, "ts": ts,
                    "primary_venue": sig.venue,
                    "primary_label": sig.bracket_label,
                    "primary_degrees": sig.degrees,
                    "primary_side": sig.side,
                    "primary_entry": sig.entry_avg_price,
                    "primary_gap": sig.gap,
                    "hedge_venue": hedge.venue,
                    "hedge_label": hedge.bracket_label,
                    "hedge_degrees": hedge.degrees,
                    "hedge_side": hedge.side,
                    "hedge_entry": hedge.entry_avg_price,
                    "size": pair_size,
                    "total_entry_cost": cost,
                    "expected_pnl": e,
                    "q05_pnl": q,
                    "delta_src": delta_src,
                })
                opened_keys.add(key)
                opened_keys.add(hedge_key)

    summary = _summarize_trades(trades, "v2")
    summary["skipped_no_hedge"] = skipped_no_hedge
    summary["skipped_negative_e"] = skipped_negative_e
    summary["skipped_q05"] = skipped_q05
    return summary


def run_v3(threshold: float = 0.05,
           nws_threshold: float = 0.03,
           min_q05: float = 0.0,
           pair_size: int = DEFAULT_PAIR_SIZE) -> dict:
    """v3: hedged + two-prior agreement filter. Only enter if:
      - cross-venue gap (E_other[YES] - market_YES) > threshold
      - AND NWS PDF agrees with a margin (NWS YES − market YES > nws_threshold)
    Then build hedge and apply E>0, q05>min_q05 as in v2.
    """
    raw_signal_count = 0
    trades = []
    skipped_no_hedge = 0
    skipped_negative_e = 0
    skipped_q05 = 0
    skipped_nws_disagree = 0

    for city in CITIES:
        delta_hist, delta_src = get_delta_hist(city)
        forecast_pdf = load_forecast_pdf(city, DATE) or {}
        snaps = list(iter_snapshots(city, DATE))
        if not snaps or not forecast_pdf:
            continue

        opened_keys: set[tuple] = set()
        for ts, venues in snaps:
            kbs = venues.get("kalshi", [])
            pbs = venues.get("polymarket", [])
            if not kbs or not pbs:
                continue
            sigs = find_directional_signals(
                city, DATE, ts, kbs, pbs, delta_hist, forecast_pdf,
                threshold=threshold, pair_size=pair_size,
            )
            raw_signal_count += len(sigs)
            if not sigs:
                continue

            pmf_K = parse_kalshi_bins(kbs)
            pmf_P = parse_polymarket_bins(pbs)
            joint = joint_consensus(pmf_K, pmf_P, delta_hist)

            sigs.sort(key=lambda s: -s.gap)
            for sig in sigs:
                key = (city, sig.venue, sig.bracket_id, sig.side)
                if key in opened_keys:
                    continue

                # NWS agreement check: NWS YES probability also exceeds
                # market_yes_price by some margin (small one — just signed
                # agreement).
                # For yes_long entry: we want fv_yes > market AND nws_yes > market
                # For no_long entry: we are "buying NO" with market NO ask = sig.market_yes_price (already 1-yes_bid).
                # In that branch, sig.fair_value_yes IS the FV(NO) under translated PMF,
                # and sig.nws_yes_prob is FV(NO) under NWS. We want NWS to also
                # exceed market NO ask by the same minimum.
                nws_gap = sig.nws_yes_prob - sig.market_yes_price
                if nws_gap < nws_threshold:
                    skipped_nws_disagree += 1
                    continue

                hedge = find_hedge_for_signal(sig, kbs, pbs, delta_hist, pair_size)
                if hedge is None:
                    skipped_no_hedge += 1
                    continue
                hedge_key = (city, hedge.venue, hedge.bracket_id, hedge.side)
                if hedge_key in opened_keys:
                    continue
                e, q, cost = score_pair_under_joint(sig, hedge, joint)
                if e <= 0:
                    skipped_negative_e += 1
                    continue
                if q < min_q05:
                    skipped_q05 += 1
                    continue
                trades.append({
                    "city": city, "ts": ts,
                    "primary_venue": sig.venue,
                    "primary_label": sig.bracket_label,
                    "primary_degrees": sig.degrees,
                    "primary_side": sig.side,
                    "primary_entry": sig.entry_avg_price,
                    "primary_gap": sig.gap,
                    "primary_nws_gap": nws_gap,
                    "hedge_venue": hedge.venue,
                    "hedge_label": hedge.bracket_label,
                    "hedge_degrees": hedge.degrees,
                    "hedge_side": hedge.side,
                    "hedge_entry": hedge.entry_avg_price,
                    "size": pair_size,
                    "total_entry_cost": cost,
                    "expected_pnl": e,
                    "q05_pnl": q,
                    "delta_src": delta_src,
                })
                opened_keys.add(key)
                opened_keys.add(hedge_key)

    summary = _summarize_trades(trades, "v3")
    summary["skipped_no_hedge"] = skipped_no_hedge
    summary["skipped_negative_e"] = skipped_negative_e
    summary["skipped_q05"] = skipped_q05
    summary["skipped_nws_disagree"] = skipped_nws_disagree
    return summary


# ── Reporting ────────────────────────────────────────────────────────────


def _summarize_trades(trades: list[dict], version: str) -> dict:
    if not trades:
        return {
            "version": version, "n_trades": 0,
            "n_winners": 0, "win_rate": 0.0,
            "total_pnl_dollars": 0.0,
            "avg_pnl_per_trade": 0.0, "median_pnl_per_trade": 0.0,
            "sharpe_per_trade": 0.0, "max_drawdown_dollars": 0.0,
            "settled_n": 0, "settled_pnl_dollars": 0.0,
            "exit_reason_counts": {}, "trades": [],
            "total_q05_pnl_dollars": 0.0, "avg_q05_pnl": 0.0,
        }
    e_pnls = [t["expected_pnl"] for t in trades]
    q_pnls = [t["q05_pnl"] for t in trades]
    n = len(e_pnls)
    n_winners = sum(1 for p in e_pnls if p > 0)
    total = sum(e_pnls)
    avg = total / n
    med = median(e_pnls)
    if n > 1:
        s = stdev(e_pnls) or 1e-9
        sharpe = mean(e_pnls) / s
    else:
        sharpe = 0.0
    sorted_t = sorted(trades, key=lambda t: t["ts"])
    cum = 0.0
    peak = 0.0
    dd = 0.0
    for t in sorted_t:
        cum += t["expected_pnl"]
        peak = max(peak, cum)
        dd = min(dd, cum - peak)
    return {
        "version": version, "n_trades": n,
        "n_winners": n_winners,
        "win_rate": round(n_winners / n, 4),
        "total_pnl_dollars": round(total, 4),
        "avg_pnl_per_trade": round(avg, 4),
        "median_pnl_per_trade": round(med, 4),
        "sharpe_per_trade": round(sharpe, 4),
        "max_drawdown_dollars": round(dd, 4),
        "settled_n": 0, "settled_pnl_dollars": 0.0,
        "exit_reason_counts": {"hold_to_settle": n},
        "total_q05_pnl_dollars": round(sum(q_pnls), 4),
        "avg_q05_pnl": round(sum(q_pnls) / n, 4),
        "trades": trades,
    }


# ── CLI ──────────────────────────────────────────────────────────────────


def main():
    out_dir = Path(__file__).resolve().parent

    print("=" * 70)
    print("Strategy 9 — Cross-venue PDF arbitrage")
    print("=" * 70)
    print(f"Cities: {CITIES}")
    print(f"Date: {DATE}")
    print()

    # Diagnostics: per-city Δ histograms
    print("--- Δ histogram source per city ---")
    for c in CITIES:
        dh, src = get_delta_hist(c)
        print(f"  {c:14s} src={src}")
        for d in sorted(dh.keys()):
            print(f"     Δ={d:+d}  P={dh[d]:.3f}")
    print()

    iterations = []

    # v1
    print("--- v1: directional single-leg, threshold=0.05 ---")
    v1 = run_v1(threshold=0.05, pair_size=DEFAULT_PAIR_SIZE)
    print(f"  n_trades={v1['n_trades']}  win_rate={v1['win_rate']:.3f}")
    print(f"  total E[PnL]=${v1['total_pnl_dollars']:.2f}  total q05=${v1['total_q05_pnl_dollars']:.2f}")
    print(f"  avg E[PnL]/trade=${v1['avg_pnl_per_trade']:.4f}  sharpe={v1['sharpe_per_trade']:.3f}")
    print()

    iter_v1 = {k: v for k, v in v1.items() if k != "trades"}
    iter_v1["notes"] = (
        "Directional single-leg using cross-venue translated-PMF mispricings. "
        "Threshold=0.05 (gap > 5¢ between FV under translated other-venue PMF "
        "and market price). Hold-to-settle scoring under joint K,P built from "
        "the live venue-implied K-PMF and city Δ histogram. NO hedge — "
        "expected to be exposed to venue-divergence risk."
    )
    iterations.append(iter_v1)

    # v2
    print("--- v2: hedged pair, threshold=0.05, E>0, q05>=-1.0 ---")
    v2 = run_v2(threshold=0.05, min_q05=-1.0, require_positive_e=True,
                pair_size=DEFAULT_PAIR_SIZE)
    print(f"  n_trades={v2['n_trades']}  win_rate={v2['win_rate']:.3f}")
    print(f"  total E[PnL]=${v2['total_pnl_dollars']:.2f}  total q05=${v2['total_q05_pnl_dollars']:.2f}")
    print(f"  avg E[PnL]/trade=${v2['avg_pnl_per_trade']:.4f}  sharpe={v2['sharpe_per_trade']:.3f}")
    print(f"  skipped_no_hedge={v2.get('skipped_no_hedge')} skipped_negative_e={v2.get('skipped_negative_e')} skipped_q05={v2.get('skipped_q05')}")
    print()

    iter_v2 = {k: v for k, v in v2.items() if k != "trades"}
    iter_v2["notes"] = (
        "Hedged pair: each cross-venue PDF disagreement is paired with an "
        "opposite-side hedge on the OTHER venue at the Δ-shifted bracket. "
        "Filter: E[payout] > 0 AND q05[payout] >= -$1/pair (CVaR-style). "
        "Joint distribution = venue-implied pmf_K convolved with Δ histogram."
    )
    iterations.append(iter_v2)

    # v3
    print("--- v3: hedged + NWS agreement (>= 3¢ NWS gap), threshold=0.05, q05>=-1.0 ---")
    v3 = run_v3(threshold=0.05, nws_threshold=0.03, min_q05=-1.0,
                pair_size=DEFAULT_PAIR_SIZE)
    print(f"  n_trades={v3['n_trades']}  win_rate={v3['win_rate']:.3f}")
    print(f"  total E[PnL]=${v3['total_pnl_dollars']:.2f}  total q05=${v3['total_q05_pnl_dollars']:.2f}")
    print(f"  avg E[PnL]/trade=${v3['avg_pnl_per_trade']:.4f}  sharpe={v3['sharpe_per_trade']:.3f}")
    print(f"  skipped_nws_disagree={v3.get('skipped_nws_disagree')} skipped_no_hedge={v3.get('skipped_no_hedge')}")
    print(f"  skipped_negative_e={v3.get('skipped_negative_e')} skipped_q05={v3.get('skipped_q05')}")
    print()

    iter_v3 = {k: v for k, v in v3.items() if k != "trades"}
    iter_v3["notes"] = (
        "Two-prior agreement: only enter if cross-venue gap AND NWS PDF "
        "agree directionally on the bracket being underpriced. Then hedge "
        "as in v2. Strictest filter."
    )
    iterations.append(iter_v3)

    # v4: structural-quality hedged pairs (q05 >= 0)
    print("--- v4: hedged STRICT structural (q05 >= 0), threshold=0.03 ---")
    v4 = run_v2(threshold=0.03, min_q05=0.0, require_positive_e=True,
                pair_size=DEFAULT_PAIR_SIZE)
    v4["version"] = "v4"
    print(f"  n_trades={v4['n_trades']}  win_rate={v4['win_rate']:.3f}")
    print(f"  total E[PnL]=${v4['total_pnl_dollars']:.2f}  total q05=${v4['total_q05_pnl_dollars']:.2f}")
    print(f"  avg E[PnL]/trade=${v4['avg_pnl_per_trade']:.4f}  sharpe={v4['sharpe_per_trade']:.3f}")
    print()

    iter_v4 = {k: v for k, v in v4.items() if k != "trades"}
    iter_v4["notes"] = (
        "Strict-structural quality hedged pairs: only entries where the 5%-"
        "quantile payout is non-negative. This makes each pair a 'cannot-lose-"
        "much-at-worst' position under the joint K,P distribution. "
        "Threshold loosened to 3¢ since the q05 filter does most of the work."
    )
    iterations.append(iter_v4)

    metrics = {
        "strategy_name": "cross_venue_pdf_arb",
        "diagnostics": {
            "per_city_delta_n": {c: _city_delta_count(c) for c in CITIES},
            "min_city_delta_n_for_per_city_hist": MIN_CITY_DELTA_N,
            "default_pair_size": DEFAULT_PAIR_SIZE,
            "fee_per_contract": FEE_PER_CONTRACT,
            "settlement_data_for_2026_05_06": "absent — PnL is E[under joint K,P]",
        },
        "iterations": iterations,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (out_dir / "trades_v1.json").write_text(json.dumps(v1["trades"], indent=2, default=str))
    (out_dir / "trades_v2.json").write_text(json.dumps(v2["trades"], indent=2, default=str))
    (out_dir / "trades_v3.json").write_text(json.dumps(v3["trades"], indent=2, default=str))
    (out_dir / "trades_v4.json").write_text(json.dumps(v4["trades"], indent=2, default=str))

    print("Wrote", out_dir / "metrics.json")


if __name__ == "__main__":
    main()
