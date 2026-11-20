"""Synthetic stand-ins for the four datasets in Section 3.2.

These generators are NOT the real datasets. They exist so the full pipeline can be
developed, tested and demonstrated before (or without) the multi-GB raw downloads.
Each profile copies the *shape* of its real counterpart as described in the
literature and dataset documentation:

  milan  – Telecom Italia grid cells: strong daily cycle (lunch + evening peaks),
           weekday/weekend difference, mix of business and residential cells.
  mawi   – Backbone source IPs (WIDE, Japan): heavy-tailed volumes, bursty on/off
           behaviour, weak daily cycle.
  caida  – Backbone source IPs (US provider): like MAWI but a different level,
           burstiness and time-zone phase.
  fcc    – Home routers (Measuring Broadband America): low weekday daytime use,
           strong evening peak (streaming), heavier weekends.

Imperfections are injected on purpose (missing windows, a few users with >10%
missing, rare spikes) so the preprocessing steps have real work to do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    entity_prefix: str
    base_scale: float          # median demand level (Mbps)
    base_sigma: float          # spread of per-user levels (lognormal)
    daily_amp: float           # strength of the daily cycle (0..1)
    weekend_factor: float      # weekend multiplier
    peaks: tuple               # (hour, width, weight) Gaussian bumps of the daily curve
    ar_phi: float              # AR(1) persistence of the noise
    noise_sigma: float         # noise std (log-space)
    burst_prob: float          # chance a user is in an "on" burst in a window
    burst_scale: float         # burst multiplier
    tz_shift_h: float          # phase shift of the daily curve vs UTC
    spike_prob: float = 0.0005 # rare measurement spikes (to test clipping)


PROFILES: dict[str, Profile] = {
    "milan": Profile(
        name="milan",
        description="Synthetic stand-in for Telecom Italia Milan grid (mobile internet per cell)",
        entity_prefix="cell",
        base_scale=40.0, base_sigma=0.9, daily_amp=0.85, weekend_factor=0.8,
        peaks=((11.5, 2.5, 1.0), (18.5, 3.0, 0.9)),
        ar_phi=0.85, noise_sigma=0.12, burst_prob=0.01, burst_scale=1.8, tz_shift_h=-1.0,
    ),
    "mawi": Profile(
        name="mawi",
        description="Synthetic stand-in for MAWI backbone traces (per source IP)",
        entity_prefix="ip",
        base_scale=8.0, base_sigma=1.4, daily_amp=0.35, weekend_factor=0.95,
        peaks=((21.0, 4.0, 1.0),),
        ar_phi=0.6, noise_sigma=0.35, burst_prob=0.08, burst_scale=4.0, tz_shift_h=-9.0,
    ),
    "caida": Profile(
        name="caida",
        description="Synthetic stand-in for CAIDA anonymized backbone traces (per source IP)",
        entity_prefix="ip",
        base_scale=12.0, base_sigma=1.2, daily_amp=0.45, weekend_factor=0.9,
        peaks=((14.0, 5.0, 1.0),),
        ar_phi=0.65, noise_sigma=0.30, burst_prob=0.06, burst_scale=3.0, tz_shift_h=5.0,
    ),
    "fcc": Profile(
        name="fcc",
        description="Synthetic stand-in for FCC Measuring Broadband America (per home)",
        entity_prefix="home",
        base_scale=5.0, base_sigma=0.8, daily_amp=0.9, weekend_factor=1.3,
        peaks=((20.5, 2.0, 1.0), (13.0, 3.0, 0.3)),
        ar_phi=0.5, noise_sigma=0.25, burst_prob=0.12, burst_scale=3.5, tz_shift_h=5.0,
    ),
}


def _daily_curve(hours_local: np.ndarray, peaks: tuple, amp: float) -> np.ndarray:
    """Daily shape in [1-amp, 1] built from Gaussian bumps on a 24h circle."""
    curve = np.zeros_like(hours_local, dtype=float)
    for center, width, weight in peaks:
        d = np.abs(hours_local - center)
        d = np.minimum(d, 24 - d)  # circular distance
        curve += weight * np.exp(-0.5 * (d / width) ** 2)
    curve = curve / curve.max()
    return (1 - amp) + amp * curve


def generate_dataset(
    name: str,
    n_users: int = 100,
    days: int = 28,
    window: str = "1h",
    start: str = "2013-11-04",
    seed: int = 42,
    extra_users_frac: float = 0.15,
) -> pd.DataFrame:
    """Generate a long-format demand table (entity_id, timestamp, demand_mbps).

    Generates n_users * (1 + extra_users_frac) users so that preprocessing still
    has n_users left after dropping users with too much missing data.
    """
    p = PROFILES[name]
    # Fixed per-profile offset (str hash() is randomised per process, so not used)
    rng = np.random.default_rng(seed + list(PROFILES).index(name) * 1000)

    ts = pd.date_range(start, periods=int(pd.Timedelta(days=days) / pd.Timedelta(window)),
                       freq=window, tz="UTC")
    T = len(ts)
    U = int(round(n_users * (1 + extra_users_frac)))

    hours_utc = ts.hour.values + ts.minute.values / 60.0
    hours_local = (hours_utc - p.tz_shift_h) % 24
    dow_local = (ts - pd.Timedelta(hours=p.tz_shift_h)).day_of_week.values
    weekend = np.where(dow_local >= 5, p.weekend_factor, 1.0)

    # Per-user heterogeneity: level, phase jitter, own daily strength.
    levels = p.base_scale * rng.lognormal(0, p.base_sigma, U)
    phase = rng.normal(0, 1.0, U)
    amp_u = np.clip(p.daily_amp + rng.normal(0, 0.08, U), 0.05, 0.98)
    # For Milan, ~30% of cells are "business" cells: big lunch peak, quiet weekends.
    business = rng.random(U) < (0.3 if name == "milan" else 0.0)

    steps_per_hour = pd.Timedelta("1h") / pd.Timedelta(window)
    phi = p.ar_phi ** (1 / max(steps_per_hour, 1))  # keep persistence per hour comparable

    demand = np.empty((U, T))
    for u in range(U):
        peaks = ((11.0, 2.0, 1.0), (15.0, 2.0, 0.7)) if business[u] else p.peaks
        wk = np.where(dow_local >= 5, 0.45, 1.0) if business[u] else weekend
        shape = _daily_curve((hours_local + phase[u]) % 24, peaks, amp_u[u]) * wk

        # AR(1) log-noise
        eps = rng.normal(0, p.noise_sigma * np.sqrt(1 - phi**2), T)
        z = np.empty(T)
        z[0] = rng.normal(0, p.noise_sigma)
        for t in range(1, T):
            z[t] = phi * z[t - 1] + eps[t]

        # On/off bursts that last a few windows
        start_burst = rng.random(T) < p.burst_prob / max(steps_per_hour, 1)
        burst_len = max(int(steps_per_hour), 1) * 2
        burst = np.convolve(start_burst.astype(float), np.ones(burst_len), mode="full")[:T] > 0
        burst_mult = np.where(burst, p.burst_scale * rng.uniform(0.6, 1.4), 1.0)

        # Slow drift over weeks (users change behaviour)
        drift = 1 + 0.1 * np.sin(2 * np.pi * np.arange(T) / (T * rng.uniform(0.8, 2.0)) + rng.uniform(0, 6))

        demand[u] = levels[u] * shape * np.exp(z) * burst_mult * drift

    # Rare measurement spikes (to be clipped by preprocessing)
    spikes = rng.random((U, T)) < p.spike_prob
    demand[spikes] *= rng.uniform(15, 40, spikes.sum())

    # Missing windows: light random loss for everyone, heavy loss for a few users
    missing = rng.random((U, T)) < 0.01
    heavy = rng.choice(U, size=max(1, int(0.06 * U)), replace=False)
    for u in heavy:
        start_gap = rng.integers(0, T // 2)
        missing[u, start_gap:start_gap + int(0.2 * T)] = True
    demand[missing] = np.nan

    ids = [f"{p.entity_prefix}_{i:04d}" for i in range(U)]
    df = pd.DataFrame({
        "entity_id": np.repeat(ids, T),
        "timestamp": np.tile(ts, U),
        "demand_mbps": demand.ravel(),
    })
    # Drop the missing rows entirely, like real logs that simply have no record.
    return df.dropna(subset=["demand_mbps"]).reset_index(drop=True)
