"""Cleaning steps from Section 3.2, applied identically to every dataset.

1. Timestamps to UTC, sort each user by time.
2. Aggregate to one window length (mean demand within the window).
3. (packet traces are converted to Mbps in loaders.packets_to_demand)
4. Fill gaps of <= 2 windows by linear interpolation; drop users missing > 10%.
5. Clip spikes at the 99.9th percentile per user, fitted on the training part only.
6. Keep a fixed group of the n most active users.
   (7. Feature scaling happens inside the model pipeline, fitted on train folds.)

Remaining longer gaps (only possible for users with <= 10% missing) are forward-
filled so the simulator has a complete user x window matrix; how many values this
touched is recorded in the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from .schema import validate_demand_table


@dataclass
class PreprocessReport:
    dataset: str
    window: str
    raw_rows: int = 0
    raw_users: int = 0
    windows: int = 0
    users_dropped_missing: int = 0
    users_kept: int = 0
    values_interpolated: int = 0
    values_ffilled: int = 0
    values_clipped: int = 0
    train_end: str = ""
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def preprocess(df: pd.DataFrame, dataset: str, window: str, n_users: int,
               holdout_frac: float = 0.15, max_gap_windows: int = 2,
               max_missing_frac: float = 0.10, clip_quantile: float = 0.999,
               ) -> tuple[pd.DataFrame, PreprocessReport]:
    df = validate_demand_table(df)
    rep = PreprocessReport(dataset=dataset, window=window,
                           raw_rows=len(df), raw_users=df["entity_id"].nunique())

    # 1-2. UTC + aggregate to window, then pivot to a time x user matrix
    df["timestamp"] = df["timestamp"].dt.floor(window)
    wide = (df.groupby(["timestamp", "entity_id"])["demand_mbps"].mean()
              .unstack("entity_id").sort_index())
    full_index = pd.date_range(wide.index.min(), wide.index.max(), freq=window, tz="UTC")
    wide = wide.reindex(full_index)
    wide.index.name = "timestamp"
    rep.windows = len(wide)

    # 4. drop users with too many missing windows, then fill short gaps
    miss_frac = wide.isna().mean()
    keep = miss_frac[miss_frac <= max_missing_frac].index
    rep.users_dropped_missing = int(len(miss_frac) - len(keep))
    wide = wide[keep]
    before = int(wide.isna().sum().sum())
    wide = wide.interpolate(method="linear", limit=max_gap_windows, limit_area="inside")
    after_interp = int(wide.isna().sum().sum())
    rep.values_interpolated = before - after_interp
    wide = wide.ffill().bfill()
    rep.values_ffilled = after_interp

    # time split boundary: everything before the holdout counts as "training data"
    n_train = int(np.floor(len(wide) * (1 - holdout_frac)))
    train_end = wide.index[n_train - 1]
    rep.train_end = str(train_end)

    # 5. clip spikes using per-user quantile from the training part only
    caps = wide.iloc[:n_train].quantile(clip_quantile)
    rep.values_clipped = int((wide > caps).sum().sum())
    wide = wide.clip(upper=caps, axis=1)

    # 6. fixed group of the most active users (by training-period volume)
    top = wide.iloc[:n_train].mean().nlargest(n_users).index
    if len(top) < n_users:
        rep.notes.append(f"only {len(top)} users available (asked for {n_users})")
    wide = wide[sorted(top)]
    rep.users_kept = wide.shape[1]

    long = (wide.reset_index().melt(id_vars="timestamp", var_name="entity_id",
                                    value_name="demand_mbps")
                .sort_values(["entity_id", "timestamp"]).reset_index(drop=True))
    return long[["entity_id", "timestamp", "demand_mbps"]], rep


def to_wide(long: pd.DataFrame) -> pd.DataFrame:
    """Long demand table -> time x user matrix."""
    return long.pivot(index="timestamp", columns="entity_id", values="demand_mbps").sort_index()


def resample(long: pd.DataFrame, window: str) -> pd.DataFrame:
    """Coarsen a clean demand table (e.g. Milan 10min -> 1h for cross-dataset tests)."""
    out = long.copy()
    out["timestamp"] = out["timestamp"].dt.floor(window)
    return out.groupby(["entity_id", "timestamp"], as_index=False)["demand_mbps"].mean()
