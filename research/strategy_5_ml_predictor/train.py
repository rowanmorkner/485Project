"""
Train ML classifiers (LogisticRegression, GradientBoosting) to predict
the direction of bracket midprice over the next K=6 snapshots (~30 min).

Temporal split per (city, venue, bracket): first 70% snapshots train,
last 30% test. No random shuffling — that would leak future into past.

Two label schemes:
  - 3-class: UP / SIDEWAYS / DOWN with epsilon threshold
  - binary: UP vs NOT-UP (used for the trading signal)

Saves:
  - models/v1_logreg.pkl
  - models/v2_gbm.pkl
  - feature_importances.json
  - train_metrics.json
"""
from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, log_loss)
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).parent
FEATURES_CSV = HERE / "features.csv"
MODELS_DIR = HERE / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Epsilon for "sideways" — bracket prices live on 0..1 with $0.01 ticks.
# 1 cent move is roughly the typical spread. Use 1.5 cents as the threshold
# so "sideways" captures genuine no-meaningful-move (and would not pay fees).
EPS = 0.015

FEATURE_COLS = [
  "mid_t", "spread", "bid_size", "ask_size", "bid_depth", "ask_depth",
  "dist_peak", "hour_utc", "mins_since_open", "has_two_sided",
  "mid_dt3", "mid_dt6", "mid_dt12", "cross_diff",
]


def label_3class(delta: float) -> int:
  if delta > EPS:
    return 1   # UP
  if delta < -EPS:
    return -1  # DOWN
  return 0     # SIDEWAYS


def temporal_split(df: pd.DataFrame, train_frac: float = 0.7):
  """For each (city, venue, bracket), use first train_frac of timestamps
  as train, last as test. Returns (train_df, test_df)."""
  parts_train = []
  parts_test = []
  for (city, venue, bid), grp in df.groupby(["city", "venue", "bracket_id"]):
    grp = grp.sort_values("snap_idx")
    n = len(grp)
    if n < 10:
      continue
    cut = int(n * train_frac)
    parts_train.append(grp.iloc[:cut])
    parts_test.append(grp.iloc[cut:])
  return pd.concat(parts_train), pd.concat(parts_test)


def baseline_majority(y_train, y_test):
  vals, counts = np.unique(y_train, return_counts=True)
  majority = vals[np.argmax(counts)]
  preds = np.full_like(y_test, majority)
  return accuracy_score(y_test, preds), int(majority)


def fit_eval(model, X_train, y_train, X_test, y_test, name: str):
  model.fit(X_train, y_train)
  train_acc = accuracy_score(y_train, model.predict(X_train))
  test_pred = model.predict(X_test)
  test_acc = accuracy_score(y_test, test_pred)
  cm = confusion_matrix(y_test, test_pred, labels=sorted(np.unique(y_train).tolist()))
  rpt = classification_report(y_test, test_pred, output_dict=True, zero_division=0)
  try:
    proba = model.predict_proba(X_test)
    ll = float(log_loss(y_test, proba, labels=sorted(np.unique(y_train).tolist())))
  except Exception:
    ll = None
  print(f"\n=== {name} ===")
  print(f"  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}  log_loss={ll}")
  print(f"  confusion (rows=true, cols=pred), labels={sorted(np.unique(y_train).tolist())}:\n{cm}")
  return {
    "train_acc": float(train_acc),
    "test_acc": float(test_acc),
    "log_loss": ll,
    "confusion": cm.tolist(),
    "classification_report": rpt,
  }


def main():
  print(f"Loading {FEATURES_CSV} ...")
  df = pd.read_csv(FEATURES_CSV)
  print(f"  total rows: {len(df)}")

  # Drop rows with NaN in any feature column or label
  before = len(df)
  df = df.dropna(subset=FEATURE_COLS + ["mid_delta"])
  print(f"  after dropping NaNs: {len(df)} (lost {before - len(df)})")

  # Build labels
  df["y3"] = df["mid_delta"].apply(label_3class)
  df["y_up"] = (df["mid_delta"] > EPS).astype(int)
  df["y_down"] = (df["mid_delta"] < -EPS).astype(int)

  # Temporal split
  train, test = temporal_split(df, train_frac=0.7)
  print(f"  train rows: {len(train)}  test rows: {len(test)}")

  X_train = train[FEATURE_COLS].values
  X_test = test[FEATURE_COLS].values

  # Standardize for logreg; trees don't need it but it's cheap
  scaler = StandardScaler().fit(X_train)
  X_train_s = scaler.transform(X_train)
  X_test_s = scaler.transform(X_test)

  results = {}

  # 3-class label distribution
  y3_train = train["y3"].values
  y3_test = test["y3"].values
  classes, counts = np.unique(y3_train, return_counts=True)
  dist = {int(c): int(n) for c, n in zip(classes, counts)}
  print(f"\n3-class train distribution: {dist}")
  print(f"  test distribution: {dict(zip(*np.unique(y3_test, return_counts=True)))}")

  base_acc, base_label = baseline_majority(y3_train, y3_test)
  print(f"  3-class majority baseline: predict {base_label} -> test_acc={base_acc:.4f}")

  # ── v1: Logistic Regression, 3-class
  v1 = LogisticRegression(
    max_iter=1000, solver="lbfgs",
    class_weight="balanced",
  )
  res_v1 = fit_eval(v1, X_train_s, y3_train, X_test_s, y3_test, "v1 LogReg 3-class (balanced)")
  res_v1["majority_baseline_acc"] = float(base_acc)
  results["v1_logreg_3class"] = res_v1

  with open(MODELS_DIR / "v1_logreg.pkl", "wb") as fh:
    pickle.dump({"model": v1, "scaler": scaler, "features": FEATURE_COLS,
                 "classes": v1.classes_.tolist(), "label_eps": EPS}, fh)

  # Coefficients for interpretation
  coefs = {}
  for cls, row in zip(v1.classes_.tolist(), v1.coef_):
    coefs[str(int(cls))] = {f: float(c) for f, c in zip(FEATURE_COLS, row)}
  results["v1_logreg_coefs"] = coefs

  # ── v2: Gradient boosting binary UP / NOT-UP (the trading-relevant question)
  y_up_train = train["y_up"].values
  y_up_test = test["y_up"].values
  print(f"\nBinary UP train distribution: {dict(zip(*np.unique(y_up_train, return_counts=True)))}")

  v2 = GradientBoostingClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0,
  )
  res_v2 = fit_eval(v2, X_train, y_up_train, X_test, y_up_test, "v2 GBM binary UP")
  results["v2_gbm_up"] = res_v2

  imp = sorted(zip(FEATURE_COLS, v2.feature_importances_.tolist()),
               key=lambda kv: -kv[1])
  results["v2_gbm_feature_importance"] = [{"feature": f, "importance": float(i)} for f, i in imp]

  with open(MODELS_DIR / "v2_gbm_up.pkl", "wb") as fh:
    pickle.dump({"model": v2, "features": FEATURE_COLS, "label_eps": EPS}, fh)

  # ── v2b: Gradient boosting binary DOWN
  y_down_train = train["y_down"].values
  y_down_test = test["y_down"].values
  v2b = GradientBoostingClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0,
  )
  res_v2b = fit_eval(v2b, X_train, y_down_train, X_test, y_down_test, "v2b GBM binary DOWN")
  results["v2b_gbm_down"] = res_v2b
  with open(MODELS_DIR / "v2_gbm_down.pkl", "wb") as fh:
    pickle.dump({"model": v2b, "features": FEATURE_COLS, "label_eps": EPS}, fh)

  with open(HERE / "train_metrics.json", "w") as fh:
    json.dump(results, fh, indent=2, default=str)

  print(f"\nWrote train_metrics.json")
  print(f"Top 5 GBM (UP) features:")
  for f, i in imp[:5]:
    print(f"  {f:20s} {i:.4f}")


if __name__ == "__main__":
  main()
