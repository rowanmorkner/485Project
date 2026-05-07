"""
Probabilistic forecasting models with a unified interface.

Every model implements:
    fit(X, y)               -> returns self
    predict_mean_std(X)     -> returns (mean, std) numpy arrays
    forecast(X, cities, dates) -> returns list of Forecast objects

In increasing complexity:
    HeteroscedasticRidgeModel  - linear, two-stage ridge regression
    XGBMeanVarModel            - tree ensemble, two-stage with OOF residuals
    NGBoostNormalModel         - tree ensemble, jointly optimized distribution
    EnsembleModel              - mixture of multiple component models
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from forecast import Forecast, gaussian_to_pmf

# Floor on predicted std (°F). Without this, models can predict near-zero σ
# on easy days and get crushed on the rare days they're wrong.
MIN_STD_F = 1.0


def _build_forecasts(means, stds, cities, dates, model_name):
    """Shared helper: turn (mean, std) arrays into Forecast objects."""
    return [
        Forecast(city=c, date=d, pdf=gaussian_to_pmf(float(mu), float(sd)),
                 model_name=model_name, std_dev=float(sd))
        for c, d, mu, sd in zip(cities, dates, means, stds)
    ]


# ===== 1. Heteroscedastic Ridge =====

class HeteroscedasticRidgeModel:
    """Linear baseline. Ridge for the mean, second ridge on |residuals| for σ."""

    def __init__(self, alphas=(0.1, 1.0, 10.0, 100.0)):
        self.alphas = alphas
        self.mean_pipe_: Optional[Pipeline] = None
        self.std_pipe_: Optional[Pipeline] = None

    def fit(self, X, y):
        X_arr = X.to_numpy()
        y_arr = np.asarray(y, dtype=float)

        self.mean_pipe_ = Pipeline([("scaler", StandardScaler()),
                                    ("ridge", RidgeCV(alphas=self.alphas))])
        self.mean_pipe_.fit(X_arr, y_arr)

        residuals = y_arr - self.mean_pipe_.predict(X_arr)
        self.std_pipe_ = Pipeline([("scaler", StandardScaler()),
                                   ("ridge", RidgeCV(alphas=self.alphas))])
        self.std_pipe_.fit(X_arr, np.abs(residuals))
        return self

    def predict_mean_std(self, X):
        X_arr = X.to_numpy()
        mean = self.mean_pipe_.predict(X_arr)
        # E[|r|] = σ·√(2/π), so σ ≈ E[|r|] · √(π/2)
        abs_resid = np.maximum(self.std_pipe_.predict(X_arr), 0.0)
        std = np.maximum(abs_resid * np.sqrt(np.pi / 2), MIN_STD_F)
        return mean, std

    def forecast(self, X, cities, dates, model_name="ridge"):
        means, stds = self.predict_mean_std(X)
        return _build_forecasts(means, stds, cities, dates, model_name)


# ===== 2. XGBoost mean+variance =====

class XGBMeanVarModel:
    """Tree ensemble. Mean model, then variance model on out-of-fold residuals."""

    def __init__(self, n_oof_splits=5):
        self.params = dict(n_estimators=400, max_depth=6, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, random_state=42,
                           tree_method="hist")
        self.var_params = {**self.params, "objective": "reg:gamma"}
        self.n_oof_splits = n_oof_splits
        self.mean_model_: Optional[XGBRegressor] = None
        self.var_model_: Optional[XGBRegressor] = None

    def fit(self, X, y):
        X_arr = X.to_numpy()
        y_arr = np.asarray(y, dtype=float)

        # Out-of-fold predictions for honest residuals (in-sample residuals
        # would be artificially small and make the variance model overconfident)
        oof_pred = np.full_like(y_arr, np.nan)
        for tr, va in TimeSeriesSplit(n_splits=self.n_oof_splits).split(X_arr):
            fold = XGBRegressor(**self.params).fit(X_arr[tr], y_arr[tr])
            oof_pred[va] = fold.predict(X_arr[va])

        valid = ~np.isnan(oof_pred)
        residual_sq = np.maximum((y_arr[valid] - oof_pred[valid]) ** 2, 1e-3)

        self.mean_model_ = XGBRegressor(**self.params).fit(X_arr, y_arr)
        self.var_model_ = XGBRegressor(**self.var_params).fit(X_arr[valid], residual_sq)
        return self

    def predict_mean_std(self, X):
        X_arr = X.to_numpy()
        mean = self.mean_model_.predict(X_arr)
        var = self.var_model_.predict(X_arr)
        std = np.sqrt(np.maximum(var, MIN_STD_F ** 2))
        return mean, std

    def forecast(self, X, cities, dates, model_name="xgboost"):
        means, stds = self.predict_mean_std(X)
        return _build_forecasts(means, stds, cities, dates, model_name)


# ===== 3. NGBoost =====

class NGBoostNormalModel:
    """Tree ensemble. Jointly optimizes location and scale via natural gradient."""

    def __init__(self, n_estimators=500, learning_rate=0.01, random_state=42):
        self.model = NGBRegressor(
            Dist=Normal, Score=LogScore,
            n_estimators=n_estimators, learning_rate=learning_rate,
            random_state=random_state, verbose=False,
        )

    def fit(self, X, y):
        self.model.fit(X.to_numpy(), np.asarray(y, dtype=float))
        return self

    def predict_mean_std(self, X):
        dist = self.model.pred_dist(X.to_numpy())
        mean = np.asarray(dist.loc, dtype=float)
        std = np.maximum(np.asarray(dist.scale, dtype=float), MIN_STD_F)
        return mean, std

    def forecast(self, X, cities, dates, model_name="ngboost"):
        means, stds = self.predict_mean_std(X)
        return _build_forecasts(means, stds, cities, dates, model_name)


# ===== 4. Equal-weight ensemble (mixture distribution) =====

class EnsembleModel:
    """
    Equal-weight mixture of probabilistic forecasters.
        Mixture PMF: P_mix(T) = (1/N) · Σᵢ Pᵢ(T)
        Variance:    Var(mix) = within-model variance + between-model variance
    Pass already-fitted models in to avoid retraining.
    """

    def __init__(self, models, model_name="ensemble"):
        if len(models) < 2:
            raise ValueError("Ensemble needs at least 2 component models")
        self.models = models
        self.model_name = model_name

    def fit(self, X, y):
        for i, m in enumerate(self.models, start=1):
            print(f"  fitting member {i}/{len(self.models)}...")
            m.fit(X, y)
        return self

    def predict_mean_std(self, X):
        means, vars_ = [], []
        for m in self.models:
            mu, sd = m.predict_mean_std(X)
            means.append(np.asarray(mu, dtype=float))
            vars_.append(np.asarray(sd, dtype=float) ** 2)
        means, vars_ = np.array(means), np.array(vars_)

        mix_mean = means.mean(axis=0)
        within_var  = vars_.mean(axis=0)
        between_var = ((means - mix_mean) ** 2).mean(axis=0)
        mix_std = np.maximum(np.sqrt(within_var + between_var), MIN_STD_F)
        return mix_mean, mix_std

    def predict_pmfs(self, X):
        """Average component PMFs at each integer bin → mixture PMF."""
        per_model = []
        for m in self.models:
            mu, sd = m.predict_mean_std(X)
            per_model.append([gaussian_to_pmf(float(mu_i), float(sd_i))
                              for mu_i, sd_i in zip(mu, sd)])

        n_models = len(self.models)
        mixed = []
        for i in range(len(X)):
            all_temps = set()
            for j in range(n_models):
                all_temps.update(per_model[j][i].keys())
            avg_mass = {}
            for t in all_temps:
                m = sum(per_model[j][i].get(t, 0.0) for j in range(n_models)) / n_models
                if m > 1e-6:
                    avg_mass[t] = m
            total = sum(avg_mass.values())
            if total > 0:
                avg_mass = {t: m / total for t, m in avg_mass.items()}
            mixed.append(avg_mass)
        return mixed

    def forecast(self, X, cities, dates, model_name=None):
        pmfs = self.predict_pmfs(X)
        _, stds = self.predict_mean_std(X)
        name = model_name or self.model_name
        return [
            Forecast(city=c, date=d, pdf=pmf, model_name=name, std_dev=float(sd))
            for c, d, pmf, sd in zip(cities, dates, pmfs, stds)
        ]
