"""
SHAP analysis on the XGBoost mean model for each city.

Run after building datasets:
    pip install shap
    py explain.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap

from dataset import CITIES, city_slug
from models import XGBMeanVarModel

DATASETS_DIR = Path("data/datasets")
OUTPUT_DIR = Path("data/shap")
TARGET = "tmax_actual"


def explain_city(city_name: str, top_k: int = 10):
    slug = city_slug(city_name)
    df = pd.read_parquet(DATASETS_DIR / f"{slug}.parquet").sort_index()
    feat = [c for c in df.columns if c != TARGET]
    X, y = df[feat], df[TARGET]

    print(f"\n--- {city_name} ---")
    model = XGBMeanVarModel().fit(X, y)

    # use the mean-prediction model; that's the one whose SHAP values are
    # easiest to interpret ("which features push the predicted high up or down")
    explainer = shap.TreeExplainer(model.mean_model_)
    shap_values = explainer.shap_values(X)

    # ranked feature importance: mean(|shap value|) across all training rows
    importance = pd.Series(
        abs(shap_values).mean(axis=0), index=feat
    ).sort_values(ascending=False)

    print(f"  Top {top_k} features by mean |SHAP|:")
    for name, val in importance.head(top_k).items():
        print(f"    {name:35s}  {val:6.3f}°F")

    # save a beeswarm plot — best summary visualization
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feat,
                      max_display=top_k, show=False)
    out_path = OUTPUT_DIR / f"{slug}_shap.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"  saved beeswarm plot → {out_path}")

    return importance


def main():
    all_importance = {}
    for city in CITIES:
        try:
            all_importance[city] = explain_city(city)
        except FileNotFoundError:
            print(f"\nSkipping {city}: dataset not built")

    # cross-city comparison: which features matter where?
    print(f"\n\n{'=' * 78}")
    print("TOP 5 FEATURES BY CITY")
    print(f"{'=' * 78}")
    for city, imp in all_importance.items():
        top5 = imp.head(5).index.tolist()
        print(f"  {city:16s}  {', '.join(top5)}")


if __name__ == "__main__":
    main()