"""Run the full experiment end to end and write everything to results/.

    python scripts/run_pipeline.py                  # all datasets in config.yaml
    python scripts/run_pipeline.py --datasets milan fcc
    python scripts/run_pipeline.py --skip-cross     # skip the 4x4 cross-dataset matrix

Follows the pseudocode in Section 3.8 of the proposal.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (adds src/ to sys.path)

import argparse
import json
import time
from datetime import datetime, timezone

import pandas as pd

from bwalloc.config import RESULTS_DIR, load_config
from bwalloc.data import load_dataset, preprocess
from bwalloc.data.preprocess import resample
from bwalloc.evaluation import (cross_dataset_matrix, final_holdout, holdout_boundary,
                                normalize_by_entity, select_best, summarize_cv,
                                walk_forward_cv)
from bwalloc.features import build_features, find_peak_hours
from bwalloc.models import (LastValue, MovingAverage, candidate_models, final_version,
                            make_gradient_boosting, make_linear, model_label)
from bwalloc.simulator import (capacity_sweep, holdout_matrices, link_capacity,
                               noise_sensitivity, simulate)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def first_fold_end(timestamps: pd.Series, holdout_frac: float, n_splits: int) -> pd.Timestamp:
    """End of the first walk-forward training block (used to find peak hours leak-free)."""
    uniq = sorted(timestamps.unique())
    n_dev = int(len(uniq) * (1 - holdout_frac))
    return pd.Timestamp(uniq[n_dev // (n_splits + 1) - 1])


def run_dataset(name: str, cfg: dict, out_dir) -> dict:
    ds = cfg["datasets"][name]
    sp, mc, sim = cfg["split"], cfg["models"], cfg["simulator"]
    seed = cfg["seed"]
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"[{name}] load ({ds['source']}) + preprocess")
    raw, info = load_dataset(name, ds, seed)
    clean, rep = preprocess(raw, name, ds["window"], ds["n_users"],
                            holdout_frac=sp["holdout_frac"], **cfg["preprocess"])
    clean.to_csv(out_dir / "clean_demand.csv", index=False)

    peaks = find_peak_hours(clean, first_fold_end(clean["timestamp"], sp["holdout_frac"],
                                                  sp["n_splits"]))
    frame, feat_cols, dropped = build_features(clean, ds["window"], peaks)
    log(f"[{name}] {len(frame):,} feature rows, {len(feat_cols)} features, "
        f"dropped={dropped}, peak hours={peaks}")

    log(f"[{name}] walk-forward CV ({sp['n_splits']} folds)")
    cands = candidate_models(cfg, for_cv=True)
    cv = walk_forward_cv(frame, feat_cols, cands, sp["n_splits"], sp["holdout_frac"],
                         mc["cv_max_train_rows"], seed, log=log)
    cv.to_csv(out_dir / "cv_folds.csv", index=False)
    cv_sum = summarize_cv(cv)
    cv_sum.to_csv(out_dir / "cv_summary.csv")
    best = select_best(cv_sum, cands)
    best_label = model_label(best)
    log(f"[{name}] best in CV: {best_label}")

    # Final fit on all development data, ONE evaluation on the untouched holdout.
    finals = [LastValue(), MovingAverage(), make_linear(), make_gradient_boosting(
        seed, **{k: mc["gradient_boosting"][k] for k in ("learning_rate", "max_depth")},
        n_estimators=mc["gradient_boosting"]["n_estimators"])]
    rf_best = cv_sum.index[cv_sum.index.str.startswith("Random Forest")][0]
    finals.append(final_version(next(c for c in cands if model_label(c) == rf_best), cfg))
    if not any(model_label(f) == best_label for f in finals):
        finals.append(final_version(best, cfg))
    log(f"[{name}] final fit + holdout evaluation")
    hold_metrics, hold, fitted = final_holdout(frame, feat_cols, finals, sp["holdout_frac"],
                                               mc["max_train_rows"], seed)
    hold_metrics.to_csv(out_dir / "holdout_forecast_metrics.csv")
    # every model's holdout forecast (lets later steps compare forecasters under allocation)
    pred_cols = [c for c in hold.columns if c.startswith("pred::")]
    hold[["entity_id", "timestamp", "y", "lag_1", *pred_cols]].to_csv(
        out_dir / "holdout_predictions.csv", index=False)

    importances = {}
    for label, m in fitted.items():
        if hasattr(m, "feature_importance"):
            fi = m.feature_importance()
            if len(fi):
                importances[label] = fi
    pd.DataFrame(importances).to_csv(out_dir / "feature_importance.csv")

    # ------------------------------------------------------------ allocation
    start_hold = holdout_boundary(frame["timestamp"], sp["holdout_frac"])
    dev_long = clean[clean["timestamp"] < start_hold]
    peak_total = link_capacity(dev_long, 1.0)
    C = sim["capacity_frac"] * peak_total
    R, D_prev, D_hat = holdout_matrices(hold, f"pred::{best_label}")
    for nm, M in (("true", R), ("prev", D_prev), ("pred", D_hat)):
        M.to_csv(out_dir / f"holdout_{nm}.csv")
    run_allocation(name, cfg, out_dir, R, D_prev, D_hat, peak_total,
                   hold_metrics.loc[best_label, "predict_ms_per_window"])

    return {
        "dataset": name, **info, "window": ds["window"], "preprocess": rep.to_dict(),
        "features": feat_cols, "features_dropped": dropped, "peak_hours": peaks,
        "best_model": best_label, "capacity": C, "peak_total_demand": peak_total,
        "holdout_start": str(start_hold), "holdout_windows": int(R.shape[0]),
        "users": int(R.shape[1]),
    }


def run_allocation(name, cfg, out_dir, R, D_prev, D_hat, peak_total: float,
                   predict_ms: float) -> None:
    """Allocation stage only: simulate, capacity sweep, noise sensitivity."""
    sim = cfg["simulator"]
    C = sim["capacity_frac"] * peak_total
    log(f"[{name}] simulate allocation (C = {C:,.1f})")
    per_window, alloc = simulate(R, D_prev, D_hat, C, sim["starvation_threshold"])
    per_window.to_csv(out_dir / "allocation_per_window.csv", index=False)
    alloc["predict_ms_per_window"] = predict_ms
    alloc.to_csv(out_dir / "allocation_summary.csv")

    log(f"[{name}] capacity sweep + noise sensitivity")
    capacity_sweep(R, D_prev, D_hat, peak_total, sim["capacity_sweep"],
                   sim["starvation_threshold"]).to_csv(out_dir / "capacity_sweep.csv", index=False)
    noise_sensitivity(R, D_prev, C, sim["noise_levels"], cfg["seed"],
                      sim["starvation_threshold"]).to_csv(out_dir / "noise_sensitivity.csv",
                                                          index=False)


def rerun_allocation_from_saved(name: str, cfg: dict, info: dict) -> None:
    """Re-run only the allocation stage using the saved holdout forecasts (no retraining)."""
    out_dir = RESULTS_DIR / name
    load = lambda nm: pd.read_csv(out_dir / f"holdout_{nm}.csv", index_col=0, parse_dates=True)
    hold = pd.read_csv(out_dir / "holdout_forecast_metrics.csv", index_col=0)
    run_allocation(name, cfg, out_dir, load("true"), load("prev"), load("pred"),
                   info["peak_total_demand"],
                   hold.loc[info["best_model"], "predict_ms_per_window"])
    info["capacity"] = cfg["simulator"]["capacity_frac"] * info["peak_total_demand"]


def run_cross_dataset(names: list[str], cfg: dict, best_rf_label: str) -> None:
    """Train on one dataset, test on each holdout, in scale-free (per-user-mean) units."""
    sp, mc = cfg["split"], cfg["models"]
    win = cfg["cross_dataset_window"]
    frames, col_sets = {}, []
    for name in names:
        clean = pd.read_csv(RESULTS_DIR / name / "clean_demand.csv", parse_dates=["timestamp"])
        hourly = normalize_by_entity(resample(clean, win), sp["holdout_frac"])
        peaks = find_peak_hours(hourly, first_fold_end(hourly["timestamp"], sp["holdout_frac"],
                                                       sp["n_splits"]))
        f, cols, _ = build_features(hourly, win, peaks)
        frames[name] = f
        col_sets.append(cols)
    common = [c for c in col_sets[0] if all(c in s for s in col_sets)]
    # Same RF configuration that won CV on the first (main) dataset, full trees.
    cand = next(c for c in candidate_models(cfg, for_cv=True) if model_label(c) == best_rf_label)
    model = final_version(cand, cfg)
    log(f"cross-dataset matrix with {best_rf_label}, {len(common)} common features")
    mae, gain = cross_dataset_matrix(frames, common, model, sp["holdout_frac"],
                                     mc["max_train_rows"], cfg["seed"])
    mae.to_csv(RESULTS_DIR / "cross_dataset_nmae.csv")
    gain.to_csv(RESULTS_DIR / "cross_dataset_gain_vs_last.csv")


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(cfg["datasets"]))
    ap.add_argument("--skip-cross", action="store_true")
    ap.add_argument("--sim-only", action="store_true",
                    help="reuse saved holdout forecasts; re-run only the allocation stage")
    args = ap.parse_args()

    t0 = time.time()
    RESULTS_DIR.mkdir(exist_ok=True)
    summary_path = RESULTS_DIR / "run_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"datasets": {}}
    for name in args.datasets:
        if args.sim_only:
            rerun_allocation_from_saved(name, cfg, summary["datasets"][name])
        else:
            summary["datasets"][name] = run_dataset(name, cfg, RESULTS_DIR / name)

    names = [n for n in cfg["datasets"] if (RESULTS_DIR / n / "clean_demand.csv").exists()]
    if not args.skip_cross and not args.sim_only and len(names) > 1:
        main_ds = names[0]
        cv_sum = pd.read_csv(RESULTS_DIR / main_ds / "cv_summary.csv", index_col=0)
        rf_label = cv_sum.index[cv_sum.index.str.startswith("Random Forest")][0]
        run_cross_dataset(names, cfg, rf_label)
        summary["cross_dataset"] = {"model": rf_label, "window": cfg["cross_dataset_window"],
                                    "units": "MAE in multiples of each user's own mean demand"}

    summary["config"] = cfg
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    summary["runtime_seconds"] = round(time.time() - t0, 1)
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    log(f"done in {time.time() - t0:.0f}s -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
