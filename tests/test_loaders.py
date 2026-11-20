"""Raw-format parsers, tested on tiny hand-made files in each real format."""

import pandas as pd
import pytest

from bwalloc.data.loaders import load_dataset, packets_to_demand, parse_fcc_usage, parse_milan_file


def test_parse_milan_sums_over_country_codes(tmp_path):
    # square_id, time_ms, country_code, sms_in, sms_out, call_in, call_out, internet
    rows = [
        "1\t1383260400000\t39\t0.1\t0.1\t\t\t10.0",
        "1\t1383260400000\t0\t\t\t\t\t2.5",       # same cell+time, another country
        "2\t1383260400000\t39\t\t\t\t\t4.0",
        "1\t1383261000000\t39\t\t\t\t\t7.0",       # next 10-minute window
    ]
    p = tmp_path / "sms-call-internet-mi-2013-11-01.txt"
    p.write_text("\n".join(rows) + "\n")
    df = parse_milan_file(p, cells=[1])
    assert list(df["entity_id"].unique()) == ["cell_1"]
    assert df["demand_mbps"].tolist() == [12.5, 7.0]
    assert df["timestamp"].iloc[0] == pd.Timestamp("2013-10-31 23:00", tz="UTC")


def test_packets_to_demand_mbps():
    pk = pd.DataFrame({
        "time_epoch": [0.0, 10.0, 30.0, 70.0],
        "src_ip": ["a", "a", "b", "a"],
        "frame_len": [750_000, 750_000, 100, 1500],
    })
    df = packets_to_demand(pk, window="1min", top_n=1).set_index("timestamp")
    # 1.5 MB in 60 s = 0.2 Mbps; only the top source "a" is kept
    assert set(df["entity_id"]) == {"ip_a"}
    assert df["demand_mbps"].iloc[0] == pytest.approx(1_500_000 * 8 / 60 / 1e6)


def test_parse_fcc_usage(tmp_path):
    p = tmp_path / "curr_datausage.csv"
    pd.DataFrame({
        "unit_id": [7, 7, 7],
        "dtime": ["2023-01-01 10:05:00", "2023-01-01 10:40:00", "2023-01-01 11:00:00"],
        "wan_rx_bytes": [450_000_000, 450_000_000, 0],
        "wan_tx_bytes": [0, 0, 36_000_000],
    }).to_csv(p, index=False)
    df = parse_fcc_usage(p)
    assert df["demand_mbps"].tolist() == pytest.approx([2.0, 0.08])


def test_real_source_requires_processed_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="colab_data_prep"):
        load_dataset("milan", {"source": "real", "path": tmp_path / "missing.csv"})
