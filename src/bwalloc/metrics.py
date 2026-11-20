"""Allocation quality metrics (Section 4.3), computed per window.

served_i = min(a_i, r_i)                 bandwidth a user can actually use
s_i      = served_i / r_i                satisfaction (users with r_i = 0 count as 1)
Jain     = (sum s)^2 / (n * sum s^2)     1/n (worst) .. 1 (everyone equally satisfied)
util     = sum served / C
unmet    = sum (r - served)              demand that could not be met
starved  = share of users with s_i < threshold (default 0.5)
wasted   = sum (a - served)              reserved for a user who did not need it
stranded = min(wasted, unmet)            capacity that sat idle WHILE someone else was
                                         short; the part of "wasted" that actually hurt
"""

from __future__ import annotations

import numpy as np


def window_metrics(A: np.ndarray, R: np.ndarray, C: np.ndarray | float,
                   starvation_threshold: float = 0.5) -> dict[str, np.ndarray]:
    """A, R: (T, n) allocations and true demand. C: scalar or (T,). Returns per-window arrays."""
    A, R = np.atleast_2d(A), np.atleast_2d(R)
    C = np.broadcast_to(np.asarray(C, dtype=float), (A.shape[0],))
    served = np.minimum(A, R)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(R > 0, served / R, 1.0)
    n = s.shape[1]
    jain = s.sum(1) ** 2 / (n * (s ** 2).sum(1))
    total_r = R.sum(1)
    unmet = (R - served).sum(1)
    wasted = (A - served).sum(1)
    return {
        "jain": jain,
        "utilization": served.sum(1) / C,
        "unmet": unmet,
        "unmet_frac": np.where(total_r > 0, unmet / total_r, 0.0),
        "starvation": (s < starvation_threshold).mean(1),
        "wasted_frac": wasted / C,
        "stranded_frac": np.minimum(wasted, unmet) / C,
        "mean_satisfaction": s.mean(1),
        "min_satisfaction": s.min(1),
    }


def jain_index(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(x.sum() ** 2 / (len(x) * (x ** 2).sum()))
