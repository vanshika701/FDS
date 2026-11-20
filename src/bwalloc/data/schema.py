"""The common demand-table format every dataset is converted into (Section 3.2).

entity_id   : str              one "user" (Milan grid cell, source IP, or home router)
timestamp   : datetime64[UTC]  start of the time window
demand_mbps : float >= 0       demand in that window
"""

from __future__ import annotations

import pandas as pd

SCHEMA_COLUMNS = ["entity_id", "timestamp", "demand_mbps"]


def validate_demand_table(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in SCHEMA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"demand table is missing columns: {missing}")
    out = df[SCHEMA_COLUMNS].copy()
    out["entity_id"] = out["entity_id"].astype(str)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["demand_mbps"] = pd.to_numeric(out["demand_mbps"], errors="coerce")
    if (out["demand_mbps"].dropna() < 0).any():
        raise ValueError("demand_mbps contains negative values")
    if out.duplicated(["entity_id", "timestamp"]).any():
        raise ValueError("duplicate (entity_id, timestamp) rows")
    return out.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
