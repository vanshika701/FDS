"""Real MAWI backbone demand via the public agurim aggregate-flow service.

Why not raw pcaps: one MAWI samplepoint-F capture is ~24 GB for only 15 minutes, which
is far too large to process here and too short for daily/weekly patterns.
agurim (http://mawi.wide.ad.jp/~agurim/) is run by the MAWI project on the same WIDE
backbone traffic and publishes continuous "significant aggregate flows", meaning
traffic grouped by source/destination address prefix, at 10-minute resolution.

Entity ("user") = one aggregate flow, identified by its address label,
e.g. "150.37.248.94 *"  (one source host to anywhere), or "* 202.203.193.64/26".
The catch-all "* *" remainder flow is excluded.

Caveat (reported in the write-up): agurim only lists a flow when it is "significant"
(>= threshold % of traffic) in that day's query, so a flow can be missing on quiet
days. Preprocessing then drops flows missing > 10% of windows, as for every dataset.

    python scripts/fetch_mawi_agurim.py --days 42 --end 2026-11-16
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

import pandas as pd

from bwalloc.config import PROCESSED_DIR

CGI = "http://mawi.wide.ad.jp/~agurim/cgi-bin/myagurim.cgi"
LABEL_RE = re.compile(r"^\[\s*\d+\]\s+(\S+)\s+(\S+)\s+[\d.]+%")


def fetch_day(start: int, threshold: int, nflows: int, retries: int = 3) -> dict:
    body = urllib.parse.urlencode({
        "criteria": "byte", "view": "addr", "outfmt": "json", "interval": 600,
        "nflows": nflows, "threshold": threshold,
        "startTime": start, "endTime": start + 86400, "duration": 86400,
    }).encode()
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(CGI, data=body), timeout=120) as r:
                return json.load(r)
        except Exception as e:  # network hiccup: back off and retry
            if attempt == retries - 1:
                raise
            print(f"  retry after error: {e}")
            time.sleep(5 * (attempt + 1))


def day_to_long(d: dict) -> pd.DataFrame:
    keys = []
    for lab in d["labels"]:
        m = LABEL_RE.match(lab)
        keys.append(f"{m.group(1)} {m.group(2)}" if m else None)
    interval = d["interval"]
    rows = []
    for rec in d["data"]:
        ts, vals = rec[0], rec[1:1 + len(keys)]  # trailing column is the total
        for k, v in zip(keys, vals):
            if k and k != "* *" and k != "*:: *::":
                rows.append((k, ts, v * 8 / interval / 1e6))  # bytes per interval -> Mbps
    df = pd.DataFrame(rows, columns=["entity_id", "ts", "demand_mbps"])
    df["timestamp"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    return df.drop(columns="ts")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=42)
    ap.add_argument("--end", default="2026-11-16", help="UTC date (exclusive)")
    ap.add_argument("--threshold", type=int, default=1, help="agurim significance threshold, in %%")
    ap.add_argument("--nflows", type=int, default=100)
    a = ap.parse_args()

    end = datetime.fromisoformat(a.end).replace(tzinfo=timezone.utc)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    parts, meta = [], []
    for k in range(a.days, 0, -1):
        day = end - timedelta(days=k)
        d = fetch_day(int(day.timestamp()), a.threshold, a.nflows)
        part = day_to_long(d)
        parts.append(part)
        meta.append({"day": day.date().isoformat(), "flows": d["nflows"],
                     "interval_s": d["interval"], "points": len(d["data"]), "cmd": d["cmd"]})
        print(f"{day.date()}  flows={d['nflows']}  points={len(d['data'])}", flush=True)
        time.sleep(1.0)  # be polite to the public service

    df = pd.concat(parts, ignore_index=True)
    # Same flow can appear in two adjacent day-files at the boundary: keep one value.
    df = df.groupby(["entity_id", "timestamp"], as_index=False)["demand_mbps"].max()
    out = PROCESSED_DIR / "mawi.csv"
    df[["entity_id", "timestamp", "demand_mbps"]].to_csv(out, index=False)

    prov_path = PROCESSED_DIR / "provenance.json"
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
    prov["mawi"] = {"source": CGI, "service": "MAWI agurim aggregate flows (WIDE backbone)",
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                    "threshold_pct": a.threshold, "days": meta}
    prov_path.write_text(json.dumps(prov, indent=2))

    days_present = df.assign(day=df["timestamp"].dt.date).groupby("entity_id")["day"].nunique()
    print(f"wrote {out}: {df['entity_id'].nunique()} flows, "
          f"{(days_present >= 0.9 * a.days).sum()} present on >= 90% of days")


if __name__ == "__main__":
    main()
