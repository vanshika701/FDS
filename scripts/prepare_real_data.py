"""Turn raw public datasets into small processed demand tables (data/processed/<name>.csv).

Designed for low disk space: raw files are streamed ONE AT A TIME, reduced to the
selected users, and deleted immediately. Works the same locally and in Colab.

    # Milan: Telecom Italia Big Data Challenge, Harvard Dataverse doi:10.7910/DVN/EGZHFV
    python scripts/prepare_real_data.py milan --days 28 --cells 150

    # Packet traces (MAWI / CAIDA): first export each pcap to CSV with tshark, e.g.
    #   tshark -r trace.pcap -T fields -E separator=, -e frame.time_epoch -e ip.src -e frame.len > trace.csv
    python scripts/prepare_real_data.py packets --name mawi --inputs trace1.csv trace2.csv --window 1min

    # FCC Measuring Broadband America: data-usage CSV(s) from fcc.gov
    python scripts/prepare_real_data.py fcc --inputs curr_datausage.csv

Afterwards set `source: real` for that dataset in config.yaml.
Every download is logged to data/processed/provenance.json (URL + date), as the
proposal requires.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (adds src/ to sys.path)

import argparse
import os
import json
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bwalloc.config import PROCESSED_DIR
from bwalloc.data.loaders import packets_to_demand, parse_fcc_usage, parse_milan_file

DATAVERSE = "https://dataverse.harvard.edu"
MILAN_DOI = "doi:10.7910/DVN/EGZHFV"


def record_provenance(name: str, entry: dict) -> None:
    path = PROCESSED_DIR / "provenance.json"
    prov = json.loads(path.read_text()) if path.exists() else {}
    prov.setdefault(name, []).append({**entry, "downloaded_at": datetime.now(timezone.utc).isoformat()})
    path.write_text(json.dumps(prov, indent=2))


# Harvard Dataverse's firewall rejects Python's default "Python-urllib" User-Agent (HTTP 403).
USER_AGENT = "bwalloc-data-prep/0.1"


def _request(url: str, **kw) -> urllib.request.Request:
    headers = {"User-Agent": USER_AGENT, **kw.pop("headers", {})}
    return urllib.request.Request(url, headers=headers, **kw)


def _download(url: str, dest: Path) -> None:
    with urllib.request.urlopen(_request(url), timeout=600) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def milan_file_list() -> list[dict]:
    url = f"{DATAVERSE}/api/datasets/:persistentId/?persistentId={MILAN_DOI}"
    with urllib.request.urlopen(_request(url), timeout=60) as r:
        files = json.load(r)["data"]["latestVersion"]["files"]
    out = [{"id": f["dataFile"]["id"], "name": f["dataFile"]["filename"],
            "size": f["dataFile"].get("filesize", 0)} for f in files
           if f["dataFile"]["filename"].startswith("sms-call-internet-mi-")]
    return sorted(out, key=lambda f: f["name"])


def signed_download_url(file_id: int, guestbook: dict) -> str:
    """The Milan dataset has a guestbook: POST a response, receive a short-lived signed URL.

    `guestbook` holds what the person agreed to submit (email is required by the dataset).
    """
    req = _request(
        f"{DATAVERSE}/api/access/datafile/{file_id}",
        data=json.dumps({"guestbookResponse": guestbook}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["data"]["signedUrl"]


def prepare_milan(days: int, n_cells: int, start: str | None, guestbook: dict,
                  resume: bool = True) -> Path:
    files = milan_file_list()
    if start:
        files = [f for f in files if f["name"] >= f"sms-call-internet-mi-{start}"]
    files = files[:days]
    out_path = PROCESSED_DIR / "milan.csv"
    parts, cells = [], None
    if resume and out_path.exists():  # continue an interrupted run
        prev = pd.read_csv(out_path, parse_dates=["timestamp"])
        parts.append(prev)
        cells = prev["entity_id"].str.replace("cell_", "").astype(int).unique().tolist()
        done_days = set(prev["timestamp"].dt.tz_convert("Europe/Rome").dt.date.astype(str))
        files = [f for f in files if f["name"][-14:-4] not in done_days]
        print(f"resuming: {len(done_days)} days already done, {len(files)} to go")
    with tempfile.TemporaryDirectory() as tmp:
        for k, f in enumerate(files, 1):
            url = f"{DATAVERSE}/api/access/datafile/{f['id']}"
            raw = Path(tmp) / f["name"]
            print(f"[{k}/{len(files)}] {f['name']} ({f['size'] / 1e6:.0f} MB)", flush=True)
            _download(signed_download_url(f["id"], guestbook), raw)
            if cells is None:
                # Fix the cell group on the FIRST day only (most active cells), then keep it.
                day = parse_milan_file(raw)
                cells = (day.groupby("entity_id")["demand_mbps"].sum().nlargest(n_cells)
                            .index.str.replace("cell_", "").astype(int).tolist())
                parts.append(day[day["entity_id"].isin([f"cell_{c}" for c in cells])])
            else:
                parts.append(parse_milan_file(raw, cells=cells))
            raw.unlink()  # keep disk usage to one raw file at a time
            record_provenance("milan", {"url": url, "file": f["name"]})
            # checkpoint so an interrupted run is not lost
            pd.concat(parts).to_csv(out_path, index=False)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


def prepare_packets(name: str, inputs: list[str], window: str, top_n: int) -> Path:
    cols = ["time_epoch", "src_ip", "frame_len"]
    parts = []
    for p in inputs:
        for chunk in pd.read_csv(p, names=cols, header=None, chunksize=2_000_000):
            parts.append(packets_to_demand(chunk, window=window, top_n=10 * top_n))
        record_provenance(name, {"file": str(p)})
    df = (pd.concat(parts).groupby(["entity_id", "timestamp"], as_index=False)["demand_mbps"].sum())
    top = df.groupby("entity_id")["demand_mbps"].sum().nlargest(top_n).index
    out_path = PROCESSED_DIR / f"{name}.csv"
    df[df["entity_id"].isin(top)].to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    return out_path


def prepare_fcc(inputs: list[str], window: str, top_n: int, **cols) -> Path:
    df = pd.concat([parse_fcc_usage(p, window=window, **cols) for p in inputs])
    df = df.groupby(["entity_id", "timestamp"], as_index=False)["demand_mbps"].sum()
    top = df.groupby("entity_id")["demand_mbps"].count().nlargest(top_n).index  # most complete homes
    out_path = PROCESSED_DIR / "fcc.csv"
    df[df["entity_id"].isin(top)].to_csv(out_path, index=False)
    for p in inputs:
        record_provenance("fcc", {"file": str(p)})
    print(f"wrote {out_path}")
    return out_path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="kind", required=True)
    m = sub.add_parser("milan")
    m.add_argument("--days", type=int, default=28)
    m.add_argument("--cells", type=int, default=150)
    m.add_argument("--start", default="2013-11-04", help="first day, YYYY-MM-DD (a Monday)")
    # Guestbook details come from environment variables that the person sets themselves,
    # so no personal data lives in the code or the command history.
    m.add_argument("--email", default=os.environ.get("MILAN_GUESTBOOK_EMAIL"),
                   help="guestbook email (required); default: $MILAN_GUESTBOOK_EMAIL")
    m.add_argument("--name", default=os.environ.get("MILAN_GUESTBOOK_NAME", ""),
                   help="guestbook name (optional); default: $MILAN_GUESTBOOK_NAME")
    m.add_argument("--institution", default=os.environ.get("MILAN_GUESTBOOK_INSTITUTION", ""),
                   help="guestbook institution (optional)")
    p = sub.add_parser("packets")
    p.add_argument("--name", required=True, choices=["mawi", "caida"])
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--window", default="1min")
    p.add_argument("--top-n", type=int, default=150)
    f = sub.add_parser("fcc")
    f.add_argument("--inputs", nargs="+", required=True)
    f.add_argument("--window", default="1h")
    f.add_argument("--top-n", type=int, default=150)
    f.add_argument("--unit-col", default="unit_id")
    f.add_argument("--time-col", default="dtime")
    f.add_argument("--bytes-cols", nargs="+", default=["wan_rx_bytes", "wan_tx_bytes"])
    a = ap.parse_args(argv)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if a.kind == "milan":
        if not a.email:
            ap.error("the Milan guestbook needs an email: set MILAN_GUESTBOOK_EMAIL or pass --email")
        gb = {"email": a.email, "name": a.name, "institution": a.institution, "position": ""}
        prepare_milan(a.days, a.cells, a.start, gb)
    elif a.kind == "packets":
        prepare_packets(a.name, a.inputs, a.window, a.top_n)
    else:
        prepare_fcc(a.inputs, a.window, a.top_n, unit_col=a.unit_col, time_col=a.time_col,
                    bytes_cols=tuple(a.bytes_cols))


if __name__ == "__main__":
    sys.exit(main())
