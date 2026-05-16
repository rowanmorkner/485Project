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

# floor on predicted σ — without it, models predict near-zero spread on
# easy days and the resulting PMF blows up log-loss on rare misses
MIN_STD_F = 1.0


def _build_forecasts(means, stds, cities, dates, model_name):
    """Turn parallel (mean, std) arrays into a list of Forecast objects."""
    return [
        Forecast(city=c, date=d, pdf=gaussian_to_pmf(float(mu), float(sd)),
                 model_name=model_name, std_dev=float(sd))
        for c, d, mu, sd in zip(cities, dates, means, stds)
    ]


# ===== 1. Heteroscedastic Ridge =====

class HeteroscedasticRidgeModel:
    """Linear baseline. Ridge for the mean, second ridge on |residuals| for σ."""

    def __init__(self, alphas=(0.1, 1.0, 10.0, 100.0)):
        # alpha values to search over — RidgeCV picks the best one
        self.alphas = alphas
        self.mean_pipe_: Optional[Pipeline] = None
        self.std_pipe_: Optional[Pipeline] = None

    def fit(self, X, y):
        X_arr = X.to_numpy()
        y_arr = np.asarray(y, dtype=float)

        # stage 1: linear model for the mean
        # StandardScaler is needed because ridge's penalty is not scale-invariant
        self.mean_pipe_ = Pipeline([("scaler", StandardScaler()),
                                    ("ridge", RidgeCV(alphas=self.alphas))])
        self.mean_pipe_.fit(X_arr, y_arr)

        # stage 2: predict the size of the errors from stage 1
        residuals = y_arr - self.mean_pipe_.predict(X_arr)
        self.std_pipe_ = Pipeline([("scaler", StandardScaler()),
                                   ("ridge", RidgeCV(alphas=self.alphas))])
        # target is |residual|, not residual — we care about magnitude, not sign
        self.std_pipe_.fit(X_arr, np.abs(residuals))
        return self

    def predict_mean_std(self, X):
        X_arr = X.to_numpy()
        mean = self.mean_pipe_.predict(X_arr)

        # stage 2 predicted E[|r|]; for a Gaussian E[|r|] = σ·√(2/π),
        # so σ ≈ E[|r|] · √(π/2)
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
        # standard XGBoost regression params — default objective is squared error
        self.params = dict(n_estimators=400, max_depth=6, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, random_state=42,
                           tree_method="hist")
        # variance target is non-negative and heavy-tailed, so Gamma deviance
        # is more stable than squared-error here
        self.var_params = {**self.params, "objective": "reg:gamma"}
        self.n_oof_splits = n_oof_splits
        self.mean_model_: Optional[XGBRegressor] = None
        self.var_model_: Optional[XGBRegressor] = None

    def fit(self, X, y):
        X_arr = X.to_numpy()
        y_arr = np.asarray(y, dtype=float)

        # generate out-of-fold predictions — each row's prediction comes from
        # a model that didn't see it during training. in-sample residuals would
        # be artificially small because the model memorized those days
        oof_pred = np.full_like(y_arr, np.nan)
        for tr, va in TimeSeriesSplit(n_splits=self.n_oof_splits).split(X_arr):
            fold = XGBRegressor(**self.params).fit(X_arr[tr], y_arr[tr])
            oof_pred[va] = fold.predict(X_arr[va])

        # squared residuals are the target for the variance model
        # clip at 1e-3 to keep reg:gamma happy (it requires strictly positive targets)
        valid = ~np.isnan(oof_pred)
        residual_sq = np.maximum((y_arr[valid] - oof_pred[valid]) ** 2, 1e-3)

        # final mean model trains on all data (we already extracted clean residuals above)
        self.mean_model_ = XGBRegressor(**self.params).fit(X_arr, y_arr)
        self.var_model_ = XGBRegressor(**self.var_params).fit(X_arr[valid], residual_sq)
        return self

    def predict_mean_std(self, X):
        X_arr = X.to_numpy()
        mean = self.mean_model_.predict(X_arr)
        # variance model predicts σ² directly; take the square root for σ
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
        # Dist=Normal: predict a Gaussian per sample
        # Score=LogScore: optimize negative log-likelihood (proper scoring rule)
        self.model = NGBRegressor(
            Dist=Normal, Score=LogScore,
            n_estimators=n_estimators, learning_rate=learning_rate,
            random_state=random_state, verbose=False,
        )

    def fit(self, X, y):
        # NGBoost handles μ and σ together — no two-stage trick needed
        self.model.fit(X.to_numpy(), np.asarray(y, dtype=float))
        return self

    def predict_mean_std(self, X):
        # pred_dist returns a frozen scipy-style distribution per row
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
        # fit each component on the same data
        for i, m in enumerate(self.models, start=1):
            print(f"  fitting member {i}/{len(self.models)}...")
            m.fit(X, y)
        return self

    def predict_mean_std(self, X):
        # collect per-model (μ, σ²) predictions
        means, vars_ = [], []
        for m in self.models:
            mu, sd = m.predict_mean_std(X)
            means.append(np.asarray(mu, dtype=float))
            vars_.append(np.asarray(sd, dtype=float) ** 2)
        means, vars_ = np.array(means), np.array(vars_)

        # variance decomposition for a mixture:
        #   total variance = average within-model variance
        #                  + variance between model means (disagreement)
        mix_mean = means.mean(axis=0)
        within_var  = vars_.mean(axis=0)
        between_var = ((means - mix_mean) ** 2).mean(axis=0)
        mix_std = np.maximum(np.sqrt(within_var + between_var), MIN_STD_F)
        return mix_mean, mix_std

    def predict_pmfs(self, X):
        """Average component PMFs at each integer bin → mixture PMF."""
        # build a PMF per (model, sample) — shape: [n_models][n_samples] of dicts
        per_model = []
        for m in self.models:
            mu, sd = m.predict_mean_std(X)
            per_model.append([gaussian_to_pmf(float(mu_i), float(sd_i))
                              for mu_i, sd_i in zip(mu, sd)])

        n_models = len(self.models)
        mixed = []
        for i in range(len(X)):
            # collect every temperature bin any model considered for this sample
            all_temps = set()
            for j in range(n_models):
                all_temps.update(per_model[j][i].keys())

            # average the per-model mass at each bin
            avg_mass = {}
            for t in all_temps:
                m = sum(per_model[j][i].get(t, 0.0) for j in range(n_models)) / n_models
                if m > 1e-6:
                    avg_mass[t] = m

            # renormalize — some mass may have been dropped by the eps cutoff
            total = sum(avg_mass.values())
            if total > 0:
                avg_mass = {t: m / total for t, m in avg_mass.items()}
            mixed.append(avg_mass)
        return mixed

    def forecast(self, X, cities, dates, model_name=None):
        pmfs = self.predict_pmfs(X)
        # std for display purposes — actual PMF is the real output
        _, stds = self.predict_mean_std(X)
        name = model_name or self.model_name
        return [
            Forecast(city=c, date=d, pdf=pmf, model_name=name, std_dev=float(sd))
            for c, d, pmf, sd in zip(cities, dates, pmfs, stds)
        ]