"""Feature builder (Section 3.3).

Each row is indexed by the *target* window tau. The target is y = demand(tau);
every feature uses data from windows <= tau-1 only. In the proposal's notation
t = tau-1 is "now" and we predict t+1.

Groups (about 20 features):
  lag      lag_1, lag_2, lag_3, lag_24h, lag_7d
  rolling  mean_1h, std_1h, mean_6h, max_6h, mean_24h
  trend    trend_short (lag_1 - lag_2), pct_change_1h ((lag_1 - mean_1h) / mean_1h)
  time     hour_sin, hour_cos, day_of_week, is_weekend, is_peak_hour (of the target window)
  context  total_load, share_of_total, entity_avg

Deliberate choices (stricter than the proposal, documented in the report):
  * entity_avg is an *expanding* mean of the user's past demand, so no fold ever
    sees an average that includes its own validation period.
  * Peak hours are found on the first walk-forward training block only.
  * For hourly data "1h" is a single window, so mean_1h/std_1h use 3 windows
    (otherwise mean_1h == lag_1 and std_1h is undefined).
  * A feature whose look-back is longer than 40% of the series is dropped for that
    dataset (e.g. lag_7d on a short packet trace) and the drop is reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIME_FEATURES = ["hour_sin", "hour_cos", "day_of_week", "is_weekend", "is_peak_hour"]


def steps_per_hour(window: str) -> int:
    return max(int(pd.Timedelta("1h") / pd.Timedelta(window)), 1)


def find_peak_hours(long: pd.DataFrame, until: pd.Timestamp, top_k: int = 4) -> list[int]:
    """Hours of day with the highest average total load, using data up to `until`."""
    d = long[long["timestamp"] <= until]
    total = d.groupby("timestamp")["demand_mbps"].sum()
    by_hour = total.groupby(total.index.hour).mean()
    return sorted(by_hour.nlargest(top_k).index.tolist())


def build_features(long: pd.DataFrame, window: str, peak_hours: list[int],
                   max_lookback_frac: float = 0.4) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return (frame, feature_columns, dropped_features).

    `frame` has entity_id, timestamp (target window), y, and the feature columns.
    """
    sph = steps_per_hour(window)
    w1 = max(sph, 3)
    lookbacks = {
        "lag_1": 1, "lag_2": 2, "lag_3": 3, "lag_24h": 24 * sph, "lag_7d": 7 * 24 * sph,
        "mean_1h": w1, "std_1h": w1, "mean_6h": 6 * sph, "max_6h": 6 * sph,
        "mean_24h": 24 * sph,
    }
    n_windows = long["timestamp"].nunique()
    dropped = [f for f, lb in lookbacks.items() if lb > max_lookback_frac * n_windows]
    if "mean_1h" in dropped:  # pct_change_1h is built from mean_1h
        dropped.append("pct_change_1h")

    df = long.sort_values(["entity_id", "timestamp"]).reset_index(drop=True).copy()
    df = df.rename(columns={"demand_mbps": "y"})
    g = df.groupby("entity_id", sort=False)["y"]
    past = g.shift(1)  # demand at tau-1 ("now")
    gp = past.groupby(df["entity_id"], sort=False)

    feats = pd.DataFrame(index=df.index)
    feats["lag_1"] = past
    feats["lag_2"] = g.shift(2)
    feats["lag_3"] = g.shift(3)
    feats["lag_24h"] = g.shift(24 * sph)
    feats["lag_7d"] = g.shift(7 * 24 * sph)

    roll = lambda w, fn: gp.transform(lambda s: getattr(s.rolling(w, min_periods=w), fn)())
    feats["mean_1h"] = roll(w1, "mean")
    feats["std_1h"] = roll(w1, "std")
    feats["mean_6h"] = roll(6 * sph, "mean")
    feats["max_6h"] = roll(6 * sph, "max")
    feats["mean_24h"] = roll(24 * sph, "mean")

    feats["trend_short"] = feats["lag_1"] - feats["lag_2"]
    feats["pct_change_1h"] = (feats["lag_1"] - feats["mean_1h"]) / (feats["mean_1h"] + 1e-9)

    ts = df["timestamp"]
    hour = ts.dt.hour + ts.dt.minute / 60.0
    feats["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    feats["day_of_week"] = ts.dt.day_of_week
    feats["is_weekend"] = (ts.dt.day_of_week >= 5).astype(int)
    feats["is_peak_hour"] = ts.dt.hour.isin(peak_hours).astype(int)

    # context at "now" (tau-1): total load of all users and this user's share of it
    total_now = df.groupby("timestamp")["y"].sum().shift(1)
    feats["total_load"] = ts.map(total_now)
    feats["share_of_total"] = feats["lag_1"] / (feats["total_load"] + 1e-9)
    feats["entity_avg"] = gp.transform(lambda s: s.expanding().mean())

    feats = feats.drop(columns=dropped)
    feature_cols = list(feats.columns)
    out = pd.concat([df[["entity_id", "timestamp", "y"]], feats], axis=1)
    out = out.dropna(subset=feature_cols).reset_index(drop=True)
    return out, feature_cols, dropped
