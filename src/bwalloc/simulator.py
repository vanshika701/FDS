"""Shared-link simulator (Sections 3.6, 3.8 and the pseudocode).

For every window tau in the holdout:
    d_prev = demand seen in tau-1        (what reactive methods know)
    d_hat  = model forecast for tau      (what the predictive method knows)
    r      = true demand in tau          (what everyone is scored against)
    a      = method(signal, C);  served = min(a, r);  record metrics.

C is fixed per dataset at capacity_frac x the peak total demand seen in the
development (non-holdout) period, so the link is busy at peak times and no
holdout information is used to set it.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .allocators import ABLATIONS, METHODS
from .metrics import window_metrics


def holdout_matrices(hold: pd.DataFrame, pred_col: str):
    """Pivot the holdout frame into (R, D_prev, D_hat) time x user matrices."""
    piv = lambda col: hold.pivot(index="timestamp", columns="entity_id", values=col).sort_index()
    R = piv("y")
    return R, piv("lag_1").loc[R.index, R.columns], piv(pred_col).loc[R.index, R.columns]


def link_capacity(dev_long: pd.DataFrame, capacity_frac: float) -> float:
    return float(capacity_frac * dev_long.groupby("timestamp")["demand_mbps"].sum().max())


def simulate(R: pd.DataFrame, D_prev: pd.DataFrame, D_hat: pd.DataFrame, C: float,
             starvation_threshold: float = 0.5, include_ablations: bool = True,
             methods: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all methods. Returns (per-window metrics long table, per-method summary)."""
    signals = {"prev": D_prev.to_numpy(), "pred": D_hat.to_numpy(), "true": R.to_numpy()}
    r = R.to_numpy()
    methods = dict(methods or METHODS)
    if include_ablations:
        methods.update(ABLATIONS)

    per_window, summary = [], []
    for name, (fn, signal, family) in methods.items():
        D = signals[signal]
        t0 = time.perf_counter()
        A = np.vstack([fn(D[t], C) for t in range(D.shape[0])])
        alloc_ms = 1000 * (time.perf_counter() - t0) / D.shape[0]
        assert (A.sum(1) <= C * (1 + 1e-9)).all(), f"{name} exceeded capacity"
        m = window_metrics(A, r, C, starvation_threshold)
        df = pd.DataFrame(m, index=R.index)
        df["method"], df["family"] = name, family
        per_window.append(df.reset_index())
        summary.append({"method": name, "family": family, "signal": signal,
                        "alloc_ms_per_window": alloc_ms,
                        **{k: float(np.mean(v)) for k, v in m.items()},
                        **{f"{k}_std": float(np.std(v)) for k, v in m.items()
                           if k in ("jain", "utilization", "starvation", "unmet_frac")}})
    return pd.concat(per_window, ignore_index=True), pd.DataFrame(summary).set_index("method")


def capacity_sweep(R, D_prev, D_hat, peak_total: float, fracs: list[float],
                   starvation_threshold: float = 0.5) -> pd.DataFrame:
    rows = []
    for f in fracs:
        _, s = simulate(R, D_prev, D_hat, f * peak_total, starvation_threshold)
        s = s.reset_index()
        s["capacity_frac"] = f
        rows.append(s)
    return pd.concat(rows, ignore_index=True)


def noise_sensitivity(R, D_prev, C: float, noise_levels: list[float], seed: int,
                      starvation_threshold: float = 0.5, n_repeats: int = 3) -> pd.DataFrame:
    """Replace the forecast by the truth with multiplicative log-normal noise.

    d_hat = r * exp(e - sigma^2/2),  e ~ N(0, sigma)   (mean-preserving)
    Shows how good a forecast must be before the predictive allocator helps.
    The forecast error (NMAE) at each sigma is reported so it can be compared to
    the NMAE the real model achieves.
    """
    rng = np.random.default_rng(seed)
    r = R.to_numpy()
    rows = []
    for sigma in noise_levels:
        for rep in range(n_repeats):
            noise = np.exp(rng.normal(0, sigma, r.shape) - sigma ** 2 / 2) if sigma > 0 else 1.0
            D_noisy = pd.DataFrame(r * noise, index=R.index, columns=R.columns)
            nmae = float(np.abs(D_noisy.to_numpy() - r).mean() / r.mean())
            _, s = simulate(R, D_prev, D_noisy, C, starvation_threshold)
            s = s.reset_index()
            s["noise_sigma"], s["forecast_nmae"], s["repeat"] = sigma, nmae, rep
            rows.append(s)
    out = pd.concat(rows, ignore_index=True)
    return (out.groupby(["noise_sigma", "method", "family"], as_index=False)
               [["forecast_nmae", "jain", "utilization", "starvation", "unmet_frac"]].mean())
