"""Static figures for the report, built from results/ (run run_pipeline.py first).

    python scripts/make_figures.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (adds src/ to sys.path)

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from bwalloc.config import FIGURES_DIR, RESULTS_DIR
from bwalloc.style import (DATASET_LABELS, FAMILY_MARKERS, INK_2, METHOD_COLORS, METHOD_ORDER,
                           MUTED, SEQUENTIAL_BLUE, apply_matplotlib_style)

KEY_METHODS = ["Round Robin", "Max-Min Fairness", "Proportional Share",
               "Predictive (Prop.)", "Oracle (Prop.)"]


def datasets() -> list[str]:
    summary = json.loads((RESULTS_DIR / "run_summary.json").read_text())
    return [d for d in summary["config"]["datasets"] if d in summary["datasets"]]


def grid(n):
    cols = min(n, 2)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.9 * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes.flat


def fairness_vs_utilization(names):
    fig, axes = grid(len(names))
    for ax, name in zip(axes, names):
        s = pd.read_csv(RESULTS_DIR / name / "allocation_summary.csv", index_col=0)
        for rank, m in enumerate(METHOD_ORDER):
            if m not in s.index:
                continue
            r = s.loc[m]
            star = r["family"] == "benchmark"
            # earlier palette slots = more important methods -> drawn on top
            ax.scatter(100 * r["utilization"], r["jain"], s=190 if star else 75,
                       color=METHOD_COLORS[m], marker=FAMILY_MARKERS[r["family"]],
                       edgecolor="white", linewidth=1.5, zorder=20 - rank, label=m)
            if m in ("Predictive (Prop.)", "Max-Min Fairness", "Oracle (Prop.)"):
                ax.annotate(m, (100 * r["utilization"], r["jain"]), xytext=(8, -3),
                            textcoords="offset points", fontsize=8, color=INK_2, zorder=30)
        ax.set_title(DATASET_LABELS.get(name, name))
        ax.set_xlabel("Link utilization (%)")
        ax.set_ylabel("Jain's fairness index")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Fairness vs utilization on the holdout (top-right is best)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fairness_vs_utilization.png")
    plt.close(fig)


def noise_curve(names, summary):
    fig, axes = grid(len(names))
    for ax, name in zip(axes, names):
        nz = pd.read_csv(RESULTS_DIR / name / "noise_sensitivity.csv")
        hold = pd.read_csv(RESULTS_DIR / name / "holdout_forecast_metrics.csv", index_col=0)
        model_nmae = hold.loc[summary["datasets"][name]["best_model"], "NMAE"]
        for m in ("Predictive (Prop.)", "Predictive (Max-Min)"):
            d = nz[nz["method"] == m].sort_values("forecast_nmae")
            ax.plot(d["forecast_nmae"], d["jain"], color=METHOD_COLORS[m], marker="D",
                    markersize=5, label=f"{m} with noisy forecast")
        for m in ("Max-Min Fairness", "Reactive Prop. + headroom"):
            y = nz[nz["method"] == m]["jain"].mean()
            ax.axhline(y, color=METHOD_COLORS[m], linestyle="--", linewidth=1.5,
                       label=f"{m} (no forecast)")
        ax.axvline(model_nmae, color=MUTED, linestyle=":", linewidth=1.5)
        ax.text(model_nmae, ax.get_ylim()[0], " our model", color=INK_2, fontsize=8,
                va="bottom")
        ax.set_title(DATASET_LABELS.get(name, name))
        ax.set_xlabel("Forecast error (NMAE = MAE / mean demand)")
        ax.set_ylabel("Jain's fairness index")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("How good must the forecast be? Fairness as forecast error grows",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "noise_sensitivity.png")
    plt.close(fig)


def capacity_sweep(names):
    metrics = [("jain", "Jain's fairness index"), ("unmet_frac", "Unmet demand (share)")]
    fig, axes = plt.subplots(len(metrics), len(names), figsize=(4.0 * len(names), 6.6),
                             squeeze=False, sharex=True)
    for j, name in enumerate(names):
        sw = pd.read_csv(RESULTS_DIR / name / "capacity_sweep.csv")
        for i, (col, label) in enumerate(metrics):
            ax = axes[i, j]
            for m in [m for m in METHOD_ORDER if m in KEY_METHODS]:
                d = sw[sw["method"] == m]
                ax.plot(100 * d["capacity_frac"], d[col], color=METHOD_COLORS[m],
                        marker=FAMILY_MARKERS[d["family"].iloc[0]], markersize=5, label=m)
            if i == 0:
                ax.set_title(DATASET_LABELS.get(name, name))
            if j == 0:
                ax.set_ylabel(label)
            if i == len(metrics) - 1:
                ax.set_xlabel("Capacity C (% of peak demand)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Effect of link capacity", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "capacity_sweep.png")
    plt.close(fig)


def cross_dataset():
    p = RESULTS_DIR / "cross_dataset_nmae.csv"
    if not p.exists():
        return
    mae = pd.read_csv(p, index_col=0)
    labels = [DATASET_LABELS.get(n, n) for n in mae.index]
    cmap = LinearSegmentedColormap.from_list("seq", SEQUENTIAL_BLUE)
    fig, ax = plt.subplots(figsize=(6.4, 5))
    im = ax.imshow(mae.values, cmap=cmap)
    ax.grid(False)
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Tested on (holdout)")
    ax.set_ylabel("Trained on")
    vmax = mae.values.max()
    for i in range(mae.shape[0]):
        for j in range(mae.shape[1]):
            v = mae.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if v > 0.6 * vmax else "#0b0b0b")
    fig.colorbar(im, ax=ax, label="MAE in units of each user's mean demand")
    ax.set_title("Cross-dataset forecast error (lower is better)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cross_dataset_matrix.png")
    plt.close(fig)


def feature_importance(names, summary):
    fig, axes = grid(len(names))
    for ax, name in zip(axes, names):
        fi = pd.read_csv(RESULTS_DIR / name / "feature_importance.csv", index_col=0)
        col = next(c for c in fi.columns if c.startswith("Random Forest"))
        top = fi[col].dropna().sort_values().tail(10)
        ax.barh(top.index, top.values, color=METHOD_COLORS["Predictive (Prop.)"], height=0.6)
        ax.set_title(DATASET_LABELS.get(name, name))
        ax.set_xlabel("Random Forest importance")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Top-10 features", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png")
    plt.close(fig)


def model_comparison(names):
    rows = []
    for name in names:
        cv = pd.read_csv(RESULTS_DIR / name / "cv_summary.csv", index_col=0)
        rf = cv[cv.index.str.startswith("Random Forest")].iloc[:1].rename(lambda _: "Random Forest (best)")
        cv = pd.concat([cv[~cv.index.str.startswith("Random Forest")], rf])
        cv["dataset"] = name
        rows.append(cv)
    df = pd.concat(rows)
    models = ["Moving Average (1h)", "Linear Regression", "Gradient Boosting", "Random Forest (best)"]
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / len(names)
    for k, name in enumerate(names):
        d = df[df["dataset"] == name].reindex(models)
        x = np.arange(len(models)) + k * width - 0.4 + width / 2
        vals = d["improvement_vs_last_%"]
        ax.bar(x, vals.clip(lower=-50), width * 0.92, yerr=d["improvement_vs_last_%_std"],
               color=list(METHOD_COLORS.values())[k], label=DATASET_LABELS.get(name, name),
               error_kw={"ecolor": MUTED, "elinewidth": 1})
        for xi, v in zip(x, vals):  # bars cut at -50%: print the true value
            if v < -50:
                ax.text(xi, -48, f"{v:.0f}%", ha="center", va="bottom", fontsize=7,
                        color="white", rotation=90)
    ax.set_ylim(-52, None)
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.set_xticks(range(len(models)), models)
    ax.set_ylabel("MAE improvement over Last Value (%)")
    ax.set_title("Walk-forward CV: forecast gain over the Last Value baseline (mean ± std over 5 folds)")
    ax.legend(fontsize=8, ncol=len(names))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png")
    plt.close(fig)


def forecaster_comparison(names):
    """Forecast gain (x) vs the fairness it buys (y), one panel per dataset.

    One colour + direct labels (7 forecasters in a scatter is past the safe
    categorical limit, so identity is carried by the label, not the hue).
    """
    path = RESULTS_DIR / "forecaster_comparison.csv"
    if not path.exists():
        return
    fc = pd.read_csv(path)
    fig, axes = grid(len(names))
    for ax, name in zip(axes, names):
        d = fc[fc["dataset"] == name]
        d = d[d["improvement_vs_last_%"] > -60]  # very poor moving averages would squash the axis
        ax.scatter(d["improvement_vs_last_%"], d["jain"], s=60,
                   color=METHOD_COLORS["Predictive (Prop.)"], edgecolor="white", linewidth=1.5,
                   zorder=3)
        # Greedy label placement: stack labels of clustered points vertically (12 pt apart).
        xr = max(d["improvement_vs_last_%"].max() - d["improvement_vs_last_%"].min(), 1e-9)
        yr = max(d["jain"].max() - d["jain"].min(), 1e-9)
        placed = []  # (x_norm, y_slot_pt)
        for _, r in d.sort_values("jain", ascending=False).iterrows():
            xn = (r["improvement_vs_last_%"] - d["improvement_vs_last_%"].min()) / xr
            yn = (r["jain"] - d["jain"].min()) / yr
            dy = 3
            # 1 unit of normalised y is roughly 300 pt of axis height
            while any(abs(xn - px) < 0.35 and abs(yn * 300 + dy - py) < 11 for px, py in placed):
                dy -= 12
            placed.append((xn, yn * 300 + dy))
            ax.annotate(r["forecaster"], (r["improvement_vs_last_%"], r["jain"]),
                        xytext=(7, dy), textcoords="offset points", fontsize=8, color=INK_2,
                        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.6}
                        if dy != 3 else None)
        ax.axvline(0, color=MUTED, linewidth=1, linestyle=":")
        ax.set_title(DATASET_LABELS.get(name, name))
        ax.set_xlabel("Forecast MAE gain over Last Value (%)")
        ax.set_ylabel("Jain's index of Predictive (Prop.)")
    fig.suptitle("Does a better forecast buy a fairer link?  (holdout, every forecaster)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "forecaster_comparison.png")
    plt.close(fig)


def main():
    apply_matplotlib_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads((RESULTS_DIR / "run_summary.json").read_text())
    names = datasets()
    fairness_vs_utilization(names)
    noise_curve(names, summary)
    capacity_sweep(names)
    cross_dataset()
    feature_importance(names, summary)
    model_comparison(names)
    forecaster_comparison(names)
    print("figures ->", FIGURES_DIR)


if __name__ == "__main__":
    main()
