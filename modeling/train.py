"""
Train and evaluate four probabilistic forecasters on day-ahead max temperature
for every city in the registry. Prints per-city results, then summary tables
comparing all cities × all models.

Run from project root:
    py dataset.py     # build all city parquets
    py train.py       # train and evaluate everything
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .dataset import CITIES, city_slug
from .forecast import hit_rates, report, report_hits, report_pmf
from .models import (EnsembleModel, HeteroscedasticRidgeModel,
                     NGBoostNormalModel, XGBMeanVarModel)

DATASETS_DIR = Path("data/datasets")
TARGET = "tmax_actual"
SPLIT_DATE = "2024-01-01"
MODELS_TO_REPORT = ["Ridge", "XGBoost", "NGBoost", "Ensemble"]


def train_one_city(city_name: str) -> dict:
    """Train all four models for one city. Returns nested dict of metrics."""
    slug = city_slug(city_name)
    dataset_path = DATASETS_DIR / f"{slug}.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(f"No parquet at {dataset_path} — run `py dataset.py`")

    df = pd.read_parquet(dataset_path).sort_index()
    feat = [c for c in df.columns if c != TARGET]

    train = df[df.index < SPLIT_DATE]
    test  = df[df.index >= SPLIT_DATE]
    X_train, y_train = train[feat], train[TARGET]
    X_test,  y_test  = test[feat],  test[TARGET]

    print(f"\n{'=' * 78}")
    print(f"  {city_name}: train={len(train)}  test={len(test)}  "
          f"features={len(feat)}")
    print(f"  train: {train.index.min().date()} → {train.index.max().date()}")
    print(f"  test : {test.index.min().date()} → {test.index.max().date()}")
    print(f"{'=' * 78}")

    # Fit models
    print("  Fitting Ridge...")
    ridge = HeteroscedasticRidgeModel().fit(X_train, y_train)
    print("  Fitting XGBoost...")
    xgb = XGBMeanVarModel().fit(X_train, y_train)
    print("  Fitting NGBoost...")
    ngb = NGBoostNormalModel().fit(X_train, y_train)
    ensemble = EnsembleModel([ridge, xgb, ngb])

    # Predict
    r_mu, r_sd = ridge.predict_mean_std(X_test)
    x_mu, x_sd = xgb.predict_mean_std(X_test)
    n_mu, n_sd = ngb.predict_mean_std(X_test)
    e_mu, e_sd = ensemble.predict_mean_std(X_test)
    e_pmfs = ensemble.predict_pmfs(X_test)

    # Standard metrics
    print()
    metrics = {
        "Ridge":    report("Ridge",       y_test.values, r_mu, r_sd),
        "XGBoost":  report("XGBoost",     y_test.values, x_mu, x_sd),
        "NGBoost":  report("NGBoost",     y_test.values, n_mu, n_sd),
        "Ensemble": report_pmf("Ensemble",   y_test.values, e_pmfs),
    }

    # Hit rates
    print("\n  Hit rates:")
    hits = {
        "Ridge":    hit_rates(y_test.values, r_mu, r_sd),
        "XGBoost":  hit_rates(y_test.values, x_mu, x_sd),
        "NGBoost":  hit_rates(y_test.values, n_mu, n_sd),
        "Ensemble": hit_rates(y_test.values, e_mu, e_sd, pmfs=e_pmfs),
    }
    for m in MODELS_TO_REPORT:
        report_hits(m, hits[m])

    return {"metrics": metrics, "hits": hits}


def print_summary(all_results: dict) -> None:
    """Print the cross-city comparison tables."""
    cities = list(all_results.keys())
    if not cities:
        return

    print(f"\n\n{'=' * 78}")
    print("SUMMARY: CRPS (lower is better)")
    print(f"{'=' * 78}")
    print(f"{'':16s}" + "".join(f"{m:>11s}" for m in MODELS_TO_REPORT))
    for city in cities:
        row = f"{city:16s}"
        for m in MODELS_TO_REPORT:
            row += f"{all_results[city]['metrics'][m]['crps']:>11.3f}"
        print(row)

    print(f"\n{'=' * 78}")
    print("SUMMARY: MAE °F (lower is better)")
    print(f"{'=' * 78}")
    print(f"{'':16s}" + "".join(f"{m:>11s}" for m in MODELS_TO_REPORT))
    for city in cities:
        row = f"{city:16s}"
        for m in MODELS_TO_REPORT:
            row += f"{all_results[city]['metrics'][m]['mae']:>11.3f}"
        print(row)

    print(f"\n{'=' * 78}")
    print("SUMMARY: ±2°F Hit Rate (higher is better)")
    print(f"{'=' * 78}")
    print(f"{'':16s}" + "".join(f"{m:>11s}" for m in MODELS_TO_REPORT))
    for city in cities:
        row = f"{city:16s}"
        for m in MODELS_TO_REPORT:
            row += f"{all_results[city]['hits'][m]['within_2'] * 100:>10.1f}%"
        print(row)

    print(f"\n{'=' * 78}")
    print("Best model per city (by CRPS):")
    print(f"{'=' * 78}")
    for city in cities:
        best = min(MODELS_TO_REPORT,
                   key=lambda m: all_results[city]["metrics"][m]["crps"])
        m_data = all_results[city]["metrics"][best]
        h_data = all_results[city]["hits"][best]
        print(f"  {city:16s}  {best:8s}  "
              f"CRPS={m_data['crps']:.3f}  "
              f"MAE={m_data['mae']:.2f}°F  "
              f"±2°F={h_data['within_2'] * 100:.1f}%")
        print(f"  {city:16s}  {best:8s}  "
          f"CRPS={m_data['crps']:.3f}  "
          f"MAE={m_data['mae']:.2f}°F  "
          f"exact={h_data['mode_hit'] * 100:.1f}%  "
          f"±2°F={h_data['within_2'] * 100:.1f}%")


def main():
    all_results = {}
    for city in CITIES:
        try:
            all_results[city] = train_one_city(city)
        except FileNotFoundError as e:
            print(f"\nSkipping {city}: {e}")
        except Exception as e:
            print(f"\n! Error training {city}: {e}")

    if all_results:
        print_summary(all_results)


if __name__ == "__main__":
    main()
