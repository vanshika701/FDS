"""Load each dataset into the common (entity_id, timestamp, demand_mbps) table.

Two paths:
  * source="synthetic": generate a stand-in (see synthetic.py).
  * source="real":      read data/processed/<name>.csv, which is produced from the raw
                        files by the parsers below (run in Colab or locally via
                        scripts/prepare_real_data.py) so the multi-GB raw files never
                        need to live on this machine.

Raw parsers
-----------
parse_milan_file    Telecom Italia "sms-call-internet-mi" daily .txt (tab separated):
                    square_id, time_ms, country_code, sms_in, sms_out, call_in,
                    call_out, internet. We sum `internet` over country codes.
                    Note: Milan's internet value is an anonymised *activity* count,
                    not Mbps. We keep it as a relative demand proxy and say so.
packets_to_demand   Packet traces (MAWI / CAIDA) exported to CSV with
                    (time_epoch, src_ip, frame_len) e.g. via
                    `tshark -r trace.pcap -T fields -e frame.time_epoch -e ip.src -e frame.len`.
                    Bytes per source IP per window -> Mbps; keep the most active sources.
parse_fcc_usage     FCC Measuring Broadband America data-usage table. Column names
                    vary between releases, so they are passed in explicitly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import PROCESSED_DIR
from .schema import validate_demand_table
from .synthetic import generate_dataset

MILAN_COLUMNS = ["square_id", "time_ms", "country_code", "sms_in", "sms_out",
                 "call_in", "call_out", "internet"]


def load_dataset(name: str, ds_cfg: dict, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Return (demand_table, info). `info` records provenance for the dashboard/report."""
    source = ds_cfg.get("source", "synthetic")
    if source == "synthetic":
        df = generate_dataset(name, n_users=ds_cfg["n_users"], days=ds_cfg["days"],
                              window=ds_cfg["window"], seed=seed)
        info = {"source": "synthetic", "path": None}
    elif source == "real":
        path = Path(ds_cfg.get("path") or PROCESSED_DIR / f"{name}.csv")
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run notebooks/colab_data_prep.ipynb (or "
                f"scripts/prepare_real_data.py) and copy the output here, or set "
                f"source: synthetic for '{name}' in config.yaml.")
        df = pd.read_csv(path)
        info = {"source": "real", "path": str(path)}
    else:
        raise ValueError(f"unknown source '{source}' for dataset {name}")
    return validate_demand_table(df), info


# ---------------------------------------------------------------- raw parsers

def parse_milan_file(path: str | Path, cells: list[int] | None = None) -> pd.DataFrame:
    """One Milan daily file -> demand table (10-minute windows, summed over countries)."""
    raw = pd.read_csv(path, sep="\t", header=None, names=MILAN_COLUMNS,
                      usecols=["square_id", "time_ms", "internet"])
    if cells is not None:
        raw = raw[raw["square_id"].isin(cells)]
    agg = raw.groupby(["square_id", "time_ms"], as_index=False)["internet"].sum()
    return pd.DataFrame({
        "entity_id": "cell_" + agg["square_id"].astype(str),
        "timestamp": pd.to_datetime(agg["time_ms"], unit="ms", utc=True),
        "demand_mbps": agg["internet"].astype(float),
    })


def packets_to_demand(packets: pd.DataFrame, window: str = "1min",
                      top_n: int = 200) -> pd.DataFrame:
    """Packet rows (time_epoch, src_ip, frame_len) -> per-source Mbps per window."""
    p = packets.dropna(subset=["src_ip"]).copy()
    p["timestamp"] = pd.to_datetime(p["time_epoch"], unit="s", utc=True).dt.floor(window)
    agg = p.groupby(["src_ip", "timestamp"], as_index=False)["frame_len"].sum()
    seconds = pd.Timedelta(window).total_seconds()
    agg["demand_mbps"] = agg["frame_len"] * 8 / seconds / 1e6
    top = agg.groupby("src_ip")["frame_len"].sum().nlargest(top_n).index
    agg = agg[agg["src_ip"].isin(top)]
    return pd.DataFrame({
        "entity_id": "ip_" + agg["src_ip"].astype(str),
        "timestamp": agg["timestamp"],
        "demand_mbps": agg["demand_mbps"],
    })


def parse_cesnet_ip_dir(series_dir: str | Path, times_csv: str | Path, start: str,
                        days: int, window: str = "1h") -> pd.DataFrame:
    """CESNET-TimeSeries24 per-IP files (one CSV per IP, columns id_time, n_bytes, ...).

    CESNET only writes a row for windows in which the IP had traffic, so a missing
    window means ZERO traffic, not missing data; it is filled with 0 here.
    Entity = one anonymised IP address of an ISP customer (closest public analogue to
    FCC's "one home").
    """
    times = pd.read_csv(times_csv)
    times["time"] = pd.to_datetime(times["time"], utc=True)
    t_start = pd.Timestamp(start, tz="UTC")
    sel = times[(times["time"] >= t_start) & (times["time"] < t_start + pd.Timedelta(days=days))]
    id_to_time = sel.set_index("id_time")["time"]
    seconds = pd.Timedelta(window).total_seconds()
    parts = []
    for f in sorted(Path(series_dir).glob("*.csv")):
        x = pd.read_csv(f, usecols=["id_time", "n_bytes"])
        x = x[x["id_time"].isin(id_to_time.index)]
        s = pd.Series(0.0, index=id_to_time.values)  # full grid, zeros = no traffic
        s.loc[id_to_time.loc[x["id_time"]].values] = x["n_bytes"].to_numpy() * 8 / seconds / 1e6
        parts.append(pd.DataFrame({"entity_id": f"ip_{f.stem}", "timestamp": s.index,
                                   "demand_mbps": s.to_numpy()}))
    return pd.concat(parts, ignore_index=True)


def parse_sndlib_tm_tar(tar_path: str | Path, start: str, days: int) -> pd.DataFrame:
    """GÉANT traffic matrices in SNDlib native format (one .txt per 15-min interval).

    Each DEMANDS line is "<id> ( <src> <dst> ) <routing unit> <value Mbit/s> <max path>".
    Entity = one origin->destination pair of GÉANT points of presence.
    Inside an existing matrix an unlisted pair means zero traffic. A matrix FILE that
    does not exist is a missing interval (left as NaN for preprocessing to handle).
    """
    import re
    import tarfile

    t0 = pd.Timestamp(start, tz="UTC")
    t1 = t0 + pd.Timedelta(days=days)
    name_re = re.compile(r"15min-(\d{8}-\d{4})\.txt$")
    line_re = re.compile(r"^\s*\S+\s+\(\s*(\S+)\s+(\S+)\s*\)\s+\S+\s+([\d.eE+-]+)")
    rows, stamps = [], []
    with tarfile.open(tar_path) as tar:
        for m in tar:
            hit = name_re.search(m.name)
            if not hit:
                continue
            ts = pd.Timestamp(pd.to_datetime(hit.group(1), format="%Y%m%d-%H%M"), tz="UTC")
            if not (t0 <= ts < t1):
                continue
            stamps.append(ts)
            in_demands = False
            for raw in tar.extractfile(m).read().decode().splitlines():
                if raw.startswith("DEMANDS"):
                    in_demands = True
                    continue
                if in_demands:
                    if raw.startswith(")"):
                        break
                    g = line_re.match(raw)
                    if g:
                        rows.append((f"{g.group(1)}>{g.group(2)}", ts, float(g.group(3))))
    df = pd.DataFrame(rows, columns=["entity_id", "timestamp", "demand_mbps"])
    # unlisted pair in an existing matrix = 0 traffic: complete the grid over existing files
    full = pd.MultiIndex.from_product([df["entity_id"].unique(), sorted(set(stamps))],
                                      names=["entity_id", "timestamp"])
    df = df.set_index(["entity_id", "timestamp"]).reindex(full, fill_value=0.0).reset_index()
    return df


def parse_fcc_usage(path: str | Path, unit_col: str = "unit_id", time_col: str = "dtime",
                    bytes_cols: tuple[str, ...] = ("wan_rx_bytes", "wan_tx_bytes"),
                    window: str = "1h") -> pd.DataFrame:
    """FCC MBA data-usage table -> per-home Mbps per window."""
    raw = pd.read_csv(path, usecols=[unit_col, time_col, *bytes_cols])
    raw["timestamp"] = pd.to_datetime(raw[time_col], utc=True).dt.floor(window)
    raw["bytes"] = raw[list(bytes_cols)].sum(axis=1)
    agg = raw.groupby([unit_col, "timestamp"], as_index=False)["bytes"].sum()
    seconds = pd.Timedelta(window).total_seconds()
    return pd.DataFrame({
        "entity_id": "home_" + agg[unit_col].astype(str),
        "timestamp": agg["timestamp"],
        "demand_mbps": agg["bytes"] * 8 / seconds / 1e6,
    })
