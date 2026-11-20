"""Prediction models (Section 3.4).

One pooled model per dataset is trained on all users together (not one per user),
which stays fast with hundreds of users and lets users share patterns.

  LastValue        y_hat = lag_1                      (baseline)
  MovingAverage    y_hat = mean_1h                    (baseline)
  LinearRegression StandardScaler (fit on train) + OLS
  RandomForest     sklearn RandomForestRegressor, raw features
  GradientBoosting sklearn HistGradientBoostingRegressor (histogram-based version of
                   gradient boosting; same idea as GradientBoostingRegressor but
                   much faster on 10^5+ rows)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class LastValue:
    name = "Last Value"
    is_baseline = True

    def fit(self, X, y):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["lag_1"].to_numpy()


class MovingAverage:
    name = "Moving Average (1h)"
    is_baseline = True

    def fit(self, X, y):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["mean_1h"].to_numpy()


class SklearnModel:
    is_baseline = False

    def __init__(self, name: str, estimator, params: dict | None = None):
        self.name = name
        self.estimator = estimator
        self.params = params or {}
        self.features_: list[str] = []

    def fit(self, X: pd.DataFrame, y):
        self.features_ = list(X.columns)
        self.estimator.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.estimator.predict(X[self.features_]), 0, None)

    def feature_importance(self) -> pd.Series:
        est = self.estimator
        if hasattr(est, "feature_importances_"):
            vals = est.feature_importances_
        elif hasattr(est, "named_steps"):  # LR pipeline: |standardised coefficient|
            vals = np.abs(est.named_steps["linearregression"].coef_)
        else:
            return pd.Series(dtype=float)
        return pd.Series(vals, index=self.features_).sort_values(ascending=False)


def make_linear() -> SklearnModel:
    return SklearnModel("Linear Regression", make_pipeline(StandardScaler(), LinearRegression()))


def make_random_forest(seed: int, n_estimators: int = 200, max_depth=None,
                       min_samples_leaf: int = 1) -> SklearnModel:
    est = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                min_samples_leaf=min_samples_leaf, n_jobs=-1,
                                random_state=seed)
    return SklearnModel("Random Forest", est,
                        {"max_depth": max_depth, "min_samples_leaf": min_samples_leaf})


def make_gradient_boosting(seed: int, n_estimators: int = 300, learning_rate: float = 0.05,
                           max_depth: int = 3) -> SklearnModel:
    est = HistGradientBoostingRegressor(max_iter=n_estimators, learning_rate=learning_rate,
                                        max_depth=max_depth, random_state=seed)
    return SklearnModel("Gradient Boosting", est,
                        {"n_estimators": n_estimators, "learning_rate": learning_rate,
                         "max_depth": max_depth})


def candidate_models(cfg: dict, for_cv: bool = True) -> list:
    """All models compared in walk-forward CV, including the RF grid (Section 3.7).

    With for_cv=True the forests use cv_n_estimators trees (cheaper tuning).
    """
    seed = cfg["seed"]
    m = cfg["models"]
    rf, gb = m["random_forest"], m["gradient_boosting"]
    n_trees = m.get("cv_n_estimators", rf["n_estimators"]) if for_cv else rf["n_estimators"]
    cands = [LastValue(), MovingAverage(), make_linear()]
    for depth in rf["max_depth"]:
        for leaf in rf["min_samples_leaf"]:
            cands.append(make_random_forest(seed, n_trees, depth, leaf))
    cands.append(make_gradient_boosting(seed, gb["n_estimators"], gb["learning_rate"],
                                        gb["max_depth"]))
    return cands


def final_version(model, cfg: dict):
    """Same model/config as chosen in CV, but with the full number of trees."""
    if getattr(model, "name", "") == "Random Forest":
        p = model.params
        return make_random_forest(cfg["seed"], cfg["models"]["random_forest"]["n_estimators"],
                                  p["max_depth"], p["min_samples_leaf"])
    return model


def model_label(model) -> str:
    if getattr(model, "params", None) and model.name == "Random Forest":
        p = model.params
        return f"Random Forest (depth={p['max_depth']}, leaf={p['min_samples_leaf']})"
    return model.name
