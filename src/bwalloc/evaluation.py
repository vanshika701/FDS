"""Testing the prediction models (Section 3.5).

* Split by time, never randomly: the last 15% of windows is the final holdout,
  touched once. The first 85% is split into 5 growing walk-forward folds using
  sklearn's TimeSeriesSplit on the *unique timestamps* (so all users of a window
  land on the same side of the split).
* Metrics: MAE, RMSE, % improvement of MAE over Last Value, and NMAE
  (MAE / mean demand) so errors can be compared between datasets.
* Cross-dataset matrix: train on one dataset, test on every dataset's holdout.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

from .models import LastValue, SklearnModel, model_label


# ------------------------------------------------------------------ splits

def holdout_boundary(timestamps: pd.Series, holdout_frac: float) -> pd.Timestamp:
    """First timestamp of the final holdout."""
    uniq = np.sort(timestamps.unique())
    return pd.Timestamp(uniq[int(np.floor(len(uniq) * (1 - holdout_frac)))])


def walk_forward_folds(timestamps: pd.Series, n_splits: int):
    """Yield (train_mask, val_mask) row masks; training always precedes validation."""
    uniq = np.sort(timestamps.unique())
    for tr_idx, va_idx in TimeSeriesSplit(n_splits=n_splits).split(uniq):
        tr_times, va_times = uniq[tr_idx], uniq[va_idx]
        yield timestamps.isin(tr_times).to_numpy(), timestamps.isin(va_times).to_numpy()


# ------------------------------------------------------------------ metrics

def forecast_metrics(y_true, y_pred, y_last=None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    out = {
        "MAE": mae,
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "NMAE": mae / float(np.mean(y_true) + 1e-12),
    }
    if y_last is not None:
        mae_last = float(np.mean(np.abs(np.asarray(y_last) - y_true)))
        out["improvement_vs_last_%"] = 100 * (mae_last - mae) / (mae_last + 1e-12)
    return out


# ------------------------------------------------------------------ helpers

def fresh(model):
    """Untrained copy of a model (baselines are stateless)."""
    if isinstance(model, SklearnModel):
        return SklearnModel(model.name, clone(model.estimator), dict(model.params))
    return model


def _subsample(n_rows: int, max_rows: int | None, seed: int) -> np.ndarray:
    idx = np.arange(n_rows)
    if max_rows and n_rows > max_rows:
        idx = np.sort(np.random.default_rng(seed).choice(n_rows, max_rows, replace=False))
    return idx


def fit_model(model, X: pd.DataFrame, y: pd.Series, max_rows: int | None, seed: int):
    if isinstance(model, SklearnModel):
        idx = _subsample(len(X), max_rows, seed)
        model.fit(X.iloc[idx], y.iloc[idx])
    return model


# ------------------------------------------------------------------ CV

def walk_forward_cv(frame: pd.DataFrame, feature_cols: list[str], candidates: list,
                    n_splits: int, holdout_frac: float, max_rows: int | None,
                    seed: int, log=print) -> pd.DataFrame:
    """Score every candidate on every fold of the development (first 85%) data."""
    start_hold = holdout_boundary(frame["timestamp"], holdout_frac)
    dev = frame[frame["timestamp"] < start_hold].reset_index(drop=True)
    X, y = dev[feature_cols], dev["y"]
    rows = []
    for fold, (tr, va) in enumerate(walk_forward_folds(dev["timestamp"], n_splits), 1):
        y_last = X.loc[va, "lag_1"]
        for cand in candidates:
            t0 = time.perf_counter()
            m = fit_model(fresh(cand), X[tr], y[tr], max_rows, seed)
            pred = m.predict(X[va])
            rows.append({"fold": fold, "model": model_label(cand),
                         "train_rows": int(tr.sum()), "val_rows": int(va.sum()),
                         "fit_seconds": time.perf_counter() - t0,
                         **forecast_metrics(y[va], pred, y_last)})
        log(f"    fold {fold}/{n_splits} done")
    return pd.DataFrame(rows)


def summarize_cv(cv: pd.DataFrame) -> pd.DataFrame:
    """Mean and std across folds per model, sorted by mean MAE."""
    agg = cv.groupby("model")[["MAE", "RMSE", "NMAE", "improvement_vs_last_%", "fit_seconds"]]
    out = agg.mean().join(agg.std(), rsuffix="_std")
    return out.sort_values("MAE")


def select_best(cv_summary: pd.DataFrame, candidates: list):
    best_label = cv_summary.index[0]
    for c in candidates:
        if model_label(c) == best_label:
            return c
    raise KeyError(best_label)


def final_holdout(frame: pd.DataFrame, feature_cols: list[str], models: list,
                  holdout_frac: float, max_rows: int | None, seed: int):
    """Fit each model on all development data, evaluate ONCE on the holdout.

    Returns (metrics table, holdout frame with a prediction column per model,
    dict of fitted models).
    """
    start_hold = holdout_boundary(frame["timestamp"], holdout_frac)
    dev = frame[frame["timestamp"] < start_hold]
    hold = frame[frame["timestamp"] >= start_hold].copy()
    rows, fitted = [], {}
    for cand in models:
        m = fit_model(fresh(cand), dev[feature_cols], dev["y"], max_rows, seed)
        t0 = time.perf_counter()
        pred = m.predict(hold[feature_cols])
        pred_seconds = time.perf_counter() - t0
        label = model_label(cand)
        hold[f"pred::{label}"] = pred
        fitted[label] = m
        rows.append({"model": label,
                     "predict_ms_per_window": 1000 * pred_seconds / hold["timestamp"].nunique(),
                     **forecast_metrics(hold["y"], pred, hold["lag_1"])})
    return pd.DataFrame(rows).set_index("model").sort_values("MAE"), hold, fitted


# ------------------------------------------------------------------ cross-dataset

def normalize_by_entity(long: pd.DataFrame, holdout_frac: float) -> pd.DataFrame:
    """Divide each user's demand by its development-period mean (scale-free units).

    Datasets measure very different things (Milan activity counts vs backbone Mbps
    vs home Mbps), so cross-dataset errors are only comparable in relative units.
    """
    start_hold = holdout_boundary(long["timestamp"], holdout_frac)
    means = long[long["timestamp"] < start_hold].groupby("entity_id")["demand_mbps"].mean()
    out = long.copy()
    out["demand_mbps"] = out["demand_mbps"] / out["entity_id"].map(means).clip(lower=1e-9)
    return out


def cross_dataset_matrix(frames: dict[str, pd.DataFrame], feature_cols: list[str],
                         model, holdout_frac: float, max_rows: int | None,
                         seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows = training dataset, columns = test dataset. Returns (MAE matrix, gain vs last value)."""
    names = list(frames)
    mae = pd.DataFrame(index=names, columns=names, dtype=float)
    gain = pd.DataFrame(index=names, columns=names, dtype=float)
    for src in names:
        f = frames[src]
        start = holdout_boundary(f["timestamp"], holdout_frac)
        dev = f[f["timestamp"] < start]
        m = fit_model(fresh(model), dev[feature_cols], dev["y"], max_rows, seed)
        for tgt in names:
            g = frames[tgt]
            hold = g[g["timestamp"] >= holdout_boundary(g["timestamp"], holdout_frac)]
            met = forecast_metrics(hold["y"], m.predict(hold[feature_cols]), hold["lag_1"])
            mae.loc[src, tgt] = met["MAE"]
            gain.loc[src, tgt] = met["improvement_vs_last_%"]
    mae.index.name, gain.index.name = "trained_on", "trained_on"
    return mae, gain
