import numpy as np
import pandas as pd
import pytest

from bwalloc.data import generate_dataset, preprocess
from bwalloc.features import build_features, find_peak_hours


@pytest.fixture(scope="module")
def clean():
    raw = generate_dataset("fcc", n_users=10, days=14, window="1h", seed=1)
    df, _ = preprocess(raw, "fcc", "1h", n_users=10)
    return df


def test_no_future_leakage(clean):
    """Changing demand at tau and later must not change any feature of row tau."""
    peaks = find_peak_hours(clean, clean["timestamp"].min() + pd.Timedelta(days=3))
    base, cols, _ = build_features(clean, "1h", peaks)

    tau = sorted(clean["timestamp"].unique())[300]
    tampered = clean.copy()
    future = tampered["timestamp"] >= tau
    tampered.loc[future, "demand_mbps"] = tampered.loc[future, "demand_mbps"] * 7 + 100
    after, _, _ = build_features(tampered, "1h", peaks)

    a = base[base["timestamp"] == tau].set_index("entity_id")[cols].sort_index()
    b = after[after["timestamp"] == tau].set_index("entity_id")[cols].sort_index()
    assert len(a) == 10
    pd.testing.assert_frame_equal(a, b)
    # sanity: the target itself did change
    ya = base[base["timestamp"] == tau]["y"].values
    yb = after[after["timestamp"] == tau]["y"].values
    assert not np.allclose(ya, yb)


def test_lag_definitions(clean):
    feats, _, _ = build_features(clean, "1h", [20])
    e = feats["entity_id"].iloc[0]
    series = clean[clean["entity_id"] == e].set_index("timestamp")["demand_mbps"]
    row = feats[feats["entity_id"] == e].iloc[5]
    tau = row["timestamp"]
    assert row["lag_1"] == pytest.approx(series[tau - pd.Timedelta("1h")])
    assert row["lag_24h"] == pytest.approx(series[tau - pd.Timedelta("24h")])
    assert row["y"] == pytest.approx(series[tau])


def test_long_lookback_dropped_for_short_series():
    raw = generate_dataset("caida", n_users=5, days=10, window="1h", seed=2)
    df, _ = preprocess(raw, "caida", "1h", n_users=5)
    _, cols, dropped = build_features(df, "1h", [14])
    assert "lag_7d" in dropped and "lag_7d" not in cols
    assert len(cols) == 19


def test_very_short_series_keeps_rows_when_hourly_features_dropped():
    """e.g. a 15-minute packet trace in 1-minute windows: hourly features must be dropped
    together with pct_change_1h, otherwise every row becomes NaN and is deleted."""
    raw = generate_dataset("mawi", n_users=5, days=1, window="1min", seed=3)
    raw = raw[raw["timestamp"] < raw["timestamp"].min() + pd.Timedelta("15min")]
    df, _ = preprocess(raw, "mawi", "1min", n_users=5)
    frame, cols, dropped = build_features(df, "1min", [14])
    assert "mean_1h" in dropped and "pct_change_1h" in dropped
    assert len(frame) > 0
