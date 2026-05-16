"""Ensemble probabilistic forecaster for daily-high temperatures.

Public surface: see `modeling.forecasting` for the orchestrator entrypoint
used by the trading bot. The lower-level pieces are:
  - dataset.py — feature/label registry; offline parquet build
  - models.py  — Ridge / XGBoost / NGBoost / EnsembleModel
  - forecast.py — Forecast dataclass + evaluation metrics
  - predict.py — live deployment (train + Open-Meteo features)
  - train.py   — offline cross-val training script
"""
