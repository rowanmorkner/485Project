"""
Forecast dataclass and probabilistic evaluation metrics.

The Forecast dataclass is the agreed output format for downstream consumers.
Evaluation functions cover both Gaussian closed-form metrics (faster, exact for
single-Gaussian models) and discrete-PMF versions (work for any distribution
including the ensemble's mixture).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import norm


# ===== Output format =====

@dataclass
class Forecast:
    """One Forecast covers a single (city, date) pair and gives a per-degree
    probability mass function over the daily high temperature."""
    city: str
    date: str                         # ISO date, e.g. "2026-04-27"
    pdf: dict[int, float]             # {temp_F: probability}, sums to 1.0
    model_name: str = "ensemble"
    std_dev: Optional[float] = None   # uncertainty estimate, °F


def gaussian_to_pmf(
    mean_f: float, std_f: float,
    lo: int = -20, hi: int = 130, eps: float = 1e-6,
) -> dict[int, float]:
    """Discretize N(mean_f, std_f) into per-integer-°F probability mass.
    Uses bin-integral form: P(T=t) = Φ((t+0.5-μ)/σ) - Φ((t-0.5-μ)/σ).
    Matches the unit-width bins that Kalshi strikes use."""
    if std_f <= 0:
        raise ValueError(f"std_f must be positive, got {std_f}")
    temps = np.arange(lo, hi + 1)
    upper = norm.cdf(temps + 0.5, loc=mean_f, scale=std_f)
    lower = norm.cdf(temps - 0.5, loc=mean_f, scale=std_f)
    mass = upper - lower
    mass = mass / mass.sum()  # renormalize for tail truncation
    return {int(t): float(m) for t, m in zip(temps, mass) if m > eps}


# ===== Metrics on Gaussian (mean, std) =====

def crps_gaussian(y_true, mean, std):
    """Closed-form CRPS for N(mean, std). Lower is better."""
    z = (y_true - mean) / std
    return std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))


def log_score_gaussian(y_true, mean, std):
    """Per-observation NLL under N(mean, std)."""
    return -norm.logpdf(y_true, loc=mean, scale=std)


# ===== Metrics on PMF (works for any discrete distribution) =====

def crps_from_pmf(y_true: float, pmf: dict[int, float]) -> float:
    """CRPS for a discrete PMF on integer bins (1°F wide)."""
    bins = np.array(sorted(pmf.keys()), dtype=float)
    masses = np.array([pmf[int(b)] for b in bins])
    cdf = np.cumsum(masses)
    indicator = (y_true <= bins).astype(float)
    return float(np.sum((cdf - indicator) ** 2))


def log_score_from_pmf(y_true: float, pmf: dict[int, float], eps: float = 1e-12) -> float:
    """Per-observation NLL: -log P(round(y_true)) under the PMF."""
    p = pmf.get(int(round(y_true)), eps)
    return -float(np.log(max(p, eps)))


# ===== Reporters =====

def report(name: str, y, mean, std) -> dict:
    """One-line summary for a Gaussian forecaster."""
    y = np.asarray(y, dtype=float)
    metrics = {
        "crps": float(crps_gaussian(y, mean, std).mean()),
        "nll":  float(log_score_gaussian(y, mean, std).mean()),
        "mae":  float(np.abs(y - mean).mean()),
        "rmse": float(np.sqrt(((y - mean) ** 2).mean())),
        "mean_std": float(np.asarray(std).mean()),
    }
    print(f"{name:20s} CRPS={metrics['crps']:.3f}  NLL={metrics['nll']:.3f}  "
          f"MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}  "
          f"avg σ={metrics['mean_std']:.2f}")
    return metrics


def report_pmf(name: str, y, pmfs: list[dict[int, float]]) -> dict:
    """One-line summary for a list of PMF predictions (e.g. ensemble mixture)."""
    y = np.asarray(y, dtype=float)
    crps = np.array([crps_from_pmf(yi, p) for yi, p in zip(y, pmfs)])
    nll  = np.array([log_score_from_pmf(yi, p) for yi, p in zip(y, pmfs)])
    means = np.array([sum(t * p for t, p in pmf.items()) for pmf in pmfs])
    metrics = {
        "crps": float(crps.mean()),
        "nll":  float(nll.mean()),
        "mae":  float(np.abs(y - means).mean()),
        "rmse": float(np.sqrt(((y - means) ** 2).mean())),
    }
    print(f"{name:20s} CRPS={metrics['crps']:.3f}  NLL={metrics['nll']:.3f}  "
          f"MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}")
    return metrics


# ===== Hit-rate metrics =====

def hit_rates(y_true, mean, std, pmfs=None) -> dict:
    """Three notions of correctness: exact mode, ±N°F, mass on truth."""
    y = np.asarray(y_true, dtype=float)
    y_int = np.round(y).astype(int)

    if pmfs is None:
        pmfs = [gaussian_to_pmf(float(m), float(s)) for m, s in zip(mean, std)]

    modes = np.array([max(p.items(), key=lambda kv: kv[1])[0] for p in pmfs])
    return {
        "mode_hit":      float(np.mean(modes == y_int)),
        "within_2":      float(np.mean(np.abs(mean - y) <= 2.0)),
        "within_3":      float(np.mean(np.abs(mean - y) <= 3.0)),
        "mass_on_truth": float(np.mean([p.get(int(yi), 0.0) for p, yi in zip(pmfs, y_int)])),
        "n": len(y),
    }


def report_hits(name: str, hits: dict) -> None:
    n = hits["n"]
    print(f"{name:12s} "
          f"exact={hits['mode_hit']*100:5.1f}% ({int(hits['mode_hit']*n):3d}/{n}) "
          f"±2°F={hits['within_2']*100:5.1f}% ({int(hits['within_2']*n):3d}/{n}) "
          f"±3°F={hits['within_3']*100:5.1f}% ({int(hits['within_3']*n):3d}/{n}) "
          f"avg P(truth)={hits['mass_on_truth']*100:.1f}%")
