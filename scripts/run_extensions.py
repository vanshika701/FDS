"""Optional extension (Section 3.4): SARIMA and LSTM vs the main forecasters.

Every forecaster (Last Value, Moving Average, Linear Regression, Gradient Boosting,
Random Forest, LSTM, SARIMA) is scored on the SAME holdout windows and users, twice:
  1. forecast error (MAE, NMAE, % gain over Last Value)
  2. what that forecast does to the allocation when fed into the Predictive (Prop.)
     allocator (Jain, utilization, unmet demand, starvation, stranded capacity)

    python scripts/run_extensions.py                  # all datasets in results/
    python scripts/run_extensions.py --datasets fcc --no-sarima

SARIMA is per user (slow, as the proposal notes), with a daily season. It is skipped
for 10-minute data (a 144-step season is impractically slow) and that is reported.
Run run_pipeline.py first.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import time

import pandas as pd

from bwalloc.allocators import METHODS
from bwalloc.config import RESULTS_DIR, load_config
from bwalloc.evaluation import forecast_metrics
from bwalloc.extensions import lstm_forecast, sarima_forecast
from bwalloc.features import steps_per_hour
from bwalloc.simulator import simulate


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_matrix(path) -> pd.DataFrame:
    m = pd.read_csv(path, index_col=0)
    m.index = pd.to_datetime(m.index, utc=True)
    return m


def run(name: str, info: dict, cfg: dict, do_sarima: bool) -> pd.DataFrame:
    d = RESULTS_DIR / name
    R, D_prev = read_matrix(d / "holdout_true.csv"), read_matrix(d / "holdout_prev.csv")
    clean = pd.read_csv(d / "clean_demand.csv", parse_dates=["timestamp"])
    wide = clean.pivot(index="timestamp", columns="entity_id", values="demand_mbps").sort_index()
    wide = wide[R.columns]

    preds_long = pd.read_csv(d / "holdout_predictions.csv", parse_dates=["timestamp"])
    forecasts, timing = {}, {}
    for col in [c for c in preds_long.columns if c.startswith("pred::")]:
        label = col.split("::", 1)[1]
        if label.startswith("Random Forest"):
            label = "Random Forest"
        forecasts[label] = (preds_long.pivot(index="timestamp", columns="entity_id", values=col)
                            .reindex(index=R.index, columns=R.columns))

    sph = steps_per_hour(info["window"])
    log(f"[{name}] LSTM")
    t0 = time.time()
    forecasts["LSTM"], lstm_info = lstm_forecast(wide, R.index, L=24, seed=cfg["seed"], log=log)
    timing["LSTM"] = time.time() - t0

    if do_sarima and sph == 1:
        log(f"[{name}] SARIMA(1,0,1)x(1,0,1,24) for {wide.shape[1]} users")
        t0 = time.time()
        forecasts["SARIMA"] = sarima_forecast(wide, R.index, season=24 * sph, log=log)
        timing["SARIMA"] = time.time() - t0
    elif do_sarima:
        log(f"[{name}] SARIMA skipped: {info['window']} windows -> season of {24 * sph} steps")

    C = info["capacity"]
    method = {"Predictive (Prop.)": METHODS["Predictive (Prop.)"]}
    rows = []
    for label, F in forecasts.items():
        mask = F.notna().to_numpy() & R.notna().to_numpy()
        fm = forecast_metrics(R.to_numpy()[mask], F.to_numpy()[mask], D_prev.to_numpy()[mask])
        _, s = simulate(R, D_prev, F.fillna(D_prev), C, cfg["simulator"]["starvation_threshold"],
                        include_ablations=False, methods=method)
        s = s.iloc[0]
        rows.append({"dataset": name, "forecaster": label, **fm,
                     **{k: s[k] for k in ("jain", "utilization", "unmet_frac", "starvation",
                                          "stranded_frac")},
                     "train_plus_predict_s": timing.get(label)})
    out = pd.DataFrame(rows).sort_values("MAE")
    out.to_csv(d / "extension_forecasters.csv", index=False)
    (d / "extension_lstm_info.json").write_text(json.dumps(lstm_info, indent=2))
    return out


def main() -> None:
    cfg = load_config()
    summary = json.loads((RESULTS_DIR / "run_summary.json").read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=[n for n in cfg["datasets"]
                                                       if n in summary["datasets"]])
    ap.add_argument("--no-sarima", action="store_true")
    a = ap.parse_args()

    allres = []
    for name in a.datasets:
        res = run(name, summary["datasets"][name], cfg, not a.no_sarima)
        log(f"[{name}]\n" + res[["forecaster", "NMAE", "improvement_vs_last_%", "jain",
                                 "unmet_frac"]].round(3).to_string(index=False))
        allres.append(res)
    # Combined table is rebuilt from every configured dataset's own file, so running a
    # subset (--datasets x) never drops the other datasets' rows.
    parts = [pd.read_csv(RESULTS_DIR / n / "extension_forecasters.csv") for n in cfg["datasets"]
             if (RESULTS_DIR / n / "extension_forecasters.csv").exists()]
    pd.concat(parts).to_csv(RESULTS_DIR / "forecaster_comparison.csv", index=False)
    log("done")


if __name__ == "__main__":
    main()
