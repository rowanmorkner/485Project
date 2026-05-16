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

from .forecast import Forecast, crps_gaussian, gaussian_to_pmf, log_score_gaussian

# Numerical safeguard floor on predicted std (°F). Per-model σ calibration
# (see `_calibrate_raw` and each component's `calibrate` method) is the
# principled fix for overconfidence; this floor only prevents division-by-zero
# / pathological zero-σ predictions.
MIN_STD_F = 0.1


def _calibrate_raw(raw_mu, raw_sd, y):
    """Fit scalar (mean_offset, sigma_scale) so post-calibration residuals
    are approximately unit-z on the held-out set.

        residual   = y - raw_mu
        offset     = mean(residual)
        z          = (residual - offset) / max(raw_sd, 1e-6)
        sigma_scale= std(z)

    After applying mu := raw_mu + offset and sd := raw_sd * sigma_scale,
    residuals/sd should have mean ~0 and std ~1.
    """
    raw_mu = np.asarray(raw_mu, dtype=float)
    raw_sd = np.asarray(raw_sd, dtype=float)
    residuals = np.asarray(y, dtype=float) - raw_mu
    mean_offset = float(residuals.mean())
    z = (residuals - mean_offset) / np.maximum(raw_sd, 1e-6)
    sigma_scale = float(z.std())
    if not np.isfinite(sigma_scale) or sigma_scale <= 0:
        sigma_scale = 1.0
    return mean_offset, sigma_scale


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
        # Per-model calibration parameters (see calibrate())
        self.mean_offset_ = 0.0
        self.sigma_scale_ = 1.0

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

    def _predict_raw(self, X):
        """Raw, un-calibrated, un-floored (mu, sd)."""
        X_arr = X.to_numpy()
        raw_mu = self.mean_pipe_.predict(X_arr)
        # E[|r|] = σ·√(2/π), so σ ≈ E[|r|] · √(π/2)
        abs_resid = np.maximum(self.std_pipe_.predict(X_arr), 0.0)
        raw_sd = abs_resid * np.sqrt(np.pi / 2)
        return raw_mu, raw_sd

    def predict_mean_std(self, X):
        raw_mu, raw_sd = self._predict_raw(X)
        mu = raw_mu + self.mean_offset_
        sd = np.maximum(raw_sd * self.sigma_scale_, MIN_STD_F)
        return mu, sd

    def calibrate(self, X_val, y_val):
        raw_mu, raw_sd = self._predict_raw(X_val)
        self.mean_offset_, self.sigma_scale_ = _calibrate_raw(raw_mu, raw_sd, y_val)
        return self.mean_offset_, self.sigma_scale_

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
        # Per-model calibration parameters (see calibrate())
        self.mean_offset_ = 0.0
        self.sigma_scale_ = 1.0

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

    def _predict_raw(self, X):
        """Raw, un-calibrated, un-floored (mu, sd)."""
        X_arr = X.to_numpy()
        raw_mu = self.mean_model_.predict(X_arr)
        var = self.var_model_.predict(X_arr)
        raw_sd = np.sqrt(np.maximum(var, 1e-6))
        return raw_mu, raw_sd

    def predict_mean_std(self, X):
        raw_mu, raw_sd = self._predict_raw(X)
        mu = raw_mu + self.mean_offset_
        sd = np.maximum(raw_sd * self.sigma_scale_, MIN_STD_F)
        return mu, sd

    def calibrate(self, X_val, y_val):
        raw_mu, raw_sd = self._predict_raw(X_val)
        self.mean_offset_, self.sigma_scale_ = _calibrate_raw(raw_mu, raw_sd, y_val)
        return self.mean_offset_, self.sigma_scale_

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
        # Per-model calibration parameters (see calibrate())
        self.mean_offset_ = 0.0
        self.sigma_scale_ = 1.0

    def fit(self, X, y):
        # NGBoost handles μ and σ together — no two-stage trick needed
        self.model.fit(X.to_numpy(), np.asarray(y, dtype=float))
        return self

    def _predict_raw(self, X):
        """Raw, un-calibrated, un-floored (mu, sd)."""
        dist = self.model.pred_dist(X.to_numpy())
        raw_mu = np.asarray(dist.loc, dtype=float)
        # Tiny numerical guard only — calibration handles overconfidence.
        raw_sd = np.maximum(np.asarray(dist.scale, dtype=float), 1e-6)
        return raw_mu, raw_sd

    def predict_mean_std(self, X):
        raw_mu, raw_sd = self._predict_raw(X)
        mu = raw_mu + self.mean_offset_
        sd = np.maximum(raw_sd * self.sigma_scale_, MIN_STD_F)
        return mu, sd

    def calibrate(self, X_val, y_val):
        raw_mu, raw_sd = self._predict_raw(X_val)
        self.mean_offset_, self.sigma_scale_ = _calibrate_raw(raw_mu, raw_sd, y_val)
        return self.mean_offset_, self.sigma_scale_

    def forecast(self, X, cities, dates, model_name="ngboost"):
        means, stds = self.predict_mean_std(X)
        return _build_forecasts(means, stds, cities, dates, model_name)


# ===== 4. Weighted ensemble (mixture distribution) =====

class EnsembleModel:
    """
    Weighted mixture of probabilistic forecasters.
        Mixture PMF: P_mix(T) = Σᵢ wᵢ · Pᵢ(T)
        Variance:    Var(mix) = within-model variance + between-model variance
    Weights default to uniform 1/N. Use `set_weights_from_validation` to learn
    weights from CRPS (or NLL) on a held-out set. Pass already-fitted models in
    to avoid retraining.
    """

    def __init__(self, models, weights=None, model_name="ensemble"):
        if len(models) < 2:
            raise ValueError("Ensemble needs at least 2 component models")
        self.models = models
        self.model_name = model_name
        self.weights = self._validate_weights(weights, len(models))

    @staticmethod
    def _validate_weights(weights, n_models):
        """Normalize and validate component weights → numpy float array sum=1."""
        if weights is None:
            return np.ones(n_models, dtype=float) / n_models
        w = np.asarray(weights, dtype=float)
        if w.shape != (n_models,):
            raise ValueError(f"weights length {w.shape} != n_models {n_models}")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        s = w.sum()
        if s <= 0:
            raise ValueError("weights must have positive sum")
        return w / s

    def set_weights_from_validation(self, X_val, y_val, score="crps", temperature=1.0):
        """Learn per-component weights via softmax over negative validation losses.

        Lower loss → larger weight. Temperature controls sharpness; higher T → more uniform.
        """
        y_arr = np.asarray(y_val, dtype=float)
        losses = np.empty(len(self.models), dtype=float)
        for i, m in enumerate(self.models):
            mu, sd = m.predict_mean_std(X_val)
            mu = np.asarray(mu, dtype=float)
            sd = np.asarray(sd, dtype=float)
            if score == "crps":
                losses[i] = crps_gaussian(y_arr, mu, sd).mean()
            elif score == "nll":
                losses[i] = log_score_gaussian(y_arr, mu, sd).mean()
            else:
                raise ValueError(f"Unknown score '{score}' (use 'crps' or 'nll')")

        # Softmax of negative losses, shifted for numerical stability.
        logits = -(losses - losses.min()) / float(temperature)
        exp = np.exp(logits)
        self.weights = exp / exp.sum()
        return self.weights

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

        # Broadcast weights over (n_models, n_obs).
        w = self.weights[:, None]
        mix_mean    = (w * means).sum(axis=0)
        within_var  = (w * vars_).sum(axis=0)
        between_var = (w * (means - mix_mean) ** 2).sum(axis=0)
        mix_std     = np.maximum(np.sqrt(within_var + between_var), MIN_STD_F)
        return mix_mean, mix_std

    def predict_pmfs(self, X):
        """Weighted average of component PMFs at each integer bin → mixture PMF."""
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
                m = sum(self.weights[j] * per_model[j][i].get(t, 0.0) for j in range(n_models))
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