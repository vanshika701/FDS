"""Bandwidth allocation methods (Section 3.6).

Every allocator takes a demand signal d (one value per user) and link capacity C,
and returns allocations a with sum(a) <= C. What signal it receives decides whether
it is reactive, predictive or the oracle:

  reactive    d = demand seen in the current window t      (d_prev)
  predictive  d = forecast of the next window t+1          (d_hat)
  oracle      d = true demand of the next window t+1       (r)

The simulator then scores a against the true demand r of window t+1.

Methods
  round_robin        a_i = min(d_i, C/n); spare capacity is NOT re-distributed
  max_min            progressive filling ("water-filling"): users needing less than an
                     equal share get what they need, the rest is re-split
  proportional       a_i = d_i * min(1, C / sum d)                       (Eq. 1)
  proportional_full  a_i = C * d_i / sum d                               (Eq. 2)
                     always hands out all of C in proportion to d. Used by the
                     Predictive allocator; we also run it on d_prev as an ablation.
  max_min_full       max_min, then any capacity left over is split in proportion
                     to d (headroom), used by Predictive Max-Min.
"""

from __future__ import annotations

import numpy as np


def round_robin(d: np.ndarray, C: float) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    return np.minimum(d, C / len(d))


def max_min(d: np.ndarray, C: float) -> np.ndarray:
    d = np.clip(np.asarray(d, dtype=float), 0, None)
    n = len(d)
    a = np.zeros(n)
    remaining = float(C)
    for k, i in enumerate(np.argsort(d, kind="stable")):
        give = min(d[i], remaining / (n - k))
        a[i] = give
        remaining -= give
    return a


def proportional(d: np.ndarray, C: float) -> np.ndarray:
    d = np.clip(np.asarray(d, dtype=float), 0, None)
    total = d.sum()
    return d * min(1.0, C / total) if total > 0 else np.zeros_like(d)


def proportional_full(d: np.ndarray, C: float) -> np.ndarray:
    d = np.clip(np.asarray(d, dtype=float), 0, None)
    total = d.sum()
    return C * d / total if total > 0 else np.full_like(d, C / len(d))


def max_min_full(d: np.ndarray, C: float) -> np.ndarray:
    a = max_min(d, C)
    left = C - a.sum()
    if left > 1e-12:
        d = np.clip(np.asarray(d, dtype=float), 0, None)
        w = d / d.sum() if d.sum() > 0 else np.full_like(d, 1 / len(d))
        a = a + left * w
    return a


# name -> (allocator, signal it uses, family)
METHODS = {
    "Round Robin":            (round_robin,       "prev",   "traditional"),
    "Max-Min Fairness":       (max_min,           "prev",   "traditional"),
    "Proportional Share":     (proportional,      "prev",   "traditional"),
    "Predictive (Prop.)":     (proportional_full, "pred",   "proposed"),
    "Predictive (Max-Min)":   (max_min_full,      "pred",   "proposed"),
    # Oracle = proportional on TRUE demand: every user gets the same satisfaction
    # min(1, C/sum r), which is the optimum for Jain's index on satisfaction and also
    # maximises utilization. (Max-Min on true demand is kept as a second reference:
    # it protects small users but leaves satisfaction unequal, so its Jain is lower.)
    "Oracle (Prop.)":         (proportional,      "true",   "benchmark"),
    "Oracle (Max-Min)":       (max_min,           "true",   "benchmark"),
}

# Ablation: Eq. 2's "use all of C" rule fed with reactive demand instead of forecasts.
# (A Max-Min variant was also tried; its numbers matched this one to ~0.001.)
ABLATIONS = {
    "Reactive Prop. + headroom": (proportional_full, "prev", "ablation"),
}
