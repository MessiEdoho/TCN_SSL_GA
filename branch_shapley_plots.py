"""
branch_shapley_plots.py
=======================
Visualises per-prediction branch Shapley attributions produced by
branch_shapley_analysis.py.

For each model (M3, M4) and partition (val, test), reads the streaming CSV
and the population-summary JSON written by branch_shapley_analysis.py and
produces five PNG figures plus a run log. Runs locally on the user's
Windows machine -- no GPU, no model load, no test-set inference.

Inputs (defaults derived from --model and --partition)
------------------------------------------------------
<LOCAL_OUTPUT_ROOTS[model]>/interpret_branch_ablation/<partition>/
    <model>_shapley_<partition>.csv             (~1.3 GB for M3 test)
    <model>_shapley_<partition>_summary.json

Outputs
-------
<LOCAL_OUTPUT_ROOTS[model]>/interpret_branch_ablation/<partition>/figures/
    <model>_shapley_<partition>_fig_a_mean_abs_phi.png
    <model>_shapley_<partition>_fig_b_winner_branch_histogram.png
    <model>_shapley_<partition>_fig_c_mean_signed_phi.png
    <model>_shapley_<partition>_fig_d_per_mouse_heatmap_TP.png
    <model>_shapley_<partition>_fig_d_per_mouse_heatmap_FP.png
    <model>_shapley_<partition>_fig_e_phi_violin.png

<LOCAL_OUTPUT_ROOTS[model]>/interpret_branch_ablation/<partition>/logs/
    branch_shapley_<model>_<partition>_plots.log

Usage
-----
python branch_shapley_plots.py --model M3 --partition test
python branch_shapley_plots.py --model all --partition test
"""

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Local Windows paths mirroring the cluster MODEL*_OUTPUT layout.
LOCAL_ROOT = Path(r"C:\Users\messi\OneDrive\Desktop\Desktop\TCN_UNIQURE_PROJECT")
LOCAL_OUTPUT_ROOTS = {
    "M3": LOCAL_ROOT / "MultiScaleTCN",
    "M4": LOCAL_ROOT / "MultiScaleTCNAttention",
}
BRANCH_ABLATION_SUBDIR = "interpret_branch_ablation"

CATEGORIES = ["TP", "FP", "TN", "FN"]
BRANCH_KEYS = ["B1", "B2", "B3"]
BRANCH_LABELS = ["B1 (fine, [1,2,4])",
                 "B2 (medium, [8,16,32])",
                 "B3 (coarse, [32,64,128])"]
BRANCH_COLORS = [ "#08519C", "#4292C6", "#9ECAE1"]  # dark blue/mid/light
BRANCH_EDGE = "black"
CATEGORY_COLORS = {"TP": "#2ca02c", "FP": "#d62728",
                   "TN": "#7f7f7f", "FN": "#ff7f0e"}

DPI = 300
VIOLIN_MAX_PER_GROUP = 50_000   # subsample per (category, branch) for Fig E

# CSV columns required for plotting (file_basename is dropped to save RAM).
PLOT_COLS = ["mouse_id", "y_true", "y_pred",
             "phi_B1", "phi_B2", "phi_B3",
             "winner_branch_abs", "category"]
DTYPES = {
    "mouse_id":            "category",
    "y_true":              np.int8,
    "y_pred":              np.int8,
    "phi_B1":              np.float32,
    "phi_B2":              np.float32,
    "phi_B3":              np.float32,
    "winner_branch_abs":   "category",
    "category":            "category",
}


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot per-prediction branch Shapley attributions.")
    parser.add_argument("--model", choices=["M3", "M4", "all"], default="all",
                        help="Model to plot (default: all).")
    parser.add_argument("--partition", choices=["val", "test"], required=True)
    parser.add_argument("--csv-path", type=Path, default=None,
                        help="Override CSV path. Default derived from --model "
                             "and --partition using LOCAL_OUTPUT_ROOTS.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override figure output dir. Default is the "
                             "'figures/' subdirectory next to the CSV.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def csv_path_for(model_name, partition):
    return (LOCAL_OUTPUT_ROOTS[model_name] / BRANCH_ABLATION_SUBDIR
            / partition / ("%s_shapley_%s.csv" % (model_name.lower(), partition)))


def summary_path_for(model_name, partition):
    return (LOCAL_OUTPUT_ROOTS[model_name] / BRANCH_ABLATION_SUBDIR
            / partition / ("%s_shapley_%s_summary.json" % (model_name.lower(), partition)))


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------
def setup_logging(out_dir, model_name, partition):
    log_dir = out_dir.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / ("branch_shapley_%s_%s_plots.log"
                          % (model_name.lower(), partition))
    logger = logging.getLogger("branch_shapley_plots_%s_%s" % (model_name, partition))
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    logger.info("=" * 70)
    logger.info("branch_shapley_plots.py -- %s / %s", model_name, partition.upper())
    logger.info("Timestamp: %s", datetime.datetime.now().isoformat())
    logger.info("Log file:  %s", log_file)
    logger.info("=" * 70)
    return logger


# ---------------------------------------------------------------------------
# load_csv
# ---------------------------------------------------------------------------
def load_csv(csv_path, logger):
    logger.info("Reading CSV: %s", csv_path)
    df = pd.read_csv(csv_path, dtype=DTYPES, usecols=PLOT_COLS)
    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    logger.info("Loaded %d rows | %.0f MB in memory", len(df), mem_mb)
    # Add absolute-phi columns once (used by multiple figures).
    df["abs_B1"] = df["phi_B1"].abs()
    df["abs_B2"] = df["phi_B2"].abs()
    df["abs_B3"] = df["phi_B3"].abs()
    logger.info("Category counts: %s",
                dict(df["category"].value_counts().reindex(CATEGORIES, fill_value=0)))
    return df


# ---------------------------------------------------------------------------
# Figure A -- mean |phi| per branch by category
# ---------------------------------------------------------------------------
def plot_a_mean_abs_phi(df, out_dir, model_name, partition, logger):
    means = (df.groupby("category", observed=True)[["abs_B1", "abs_B2", "abs_B3"]]
               .mean().reindex(CATEGORIES))
    counts = df["category"].value_counts().reindex(CATEGORIES, fill_value=0)

    x = np.arange(len(CATEGORIES))
    bw = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (col, label, color) in enumerate(zip(
            ["abs_B1", "abs_B2", "abs_B3"], BRANCH_LABELS, BRANCH_COLORS)):
        vals = means[col].values
        bars = ax.bar(x + (i - 1) * bw, vals, bw, label=label, color=color,
                      edgecolor=BRANCH_EDGE, linewidth=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, "%.2f" % v,
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["%s\n(n=%s)" % (c, "{:,}".format(int(counts[c]))) for c in CATEGORIES])
    ax.set_ylabel("Mean |phi| (logit units)")
    ax.set_title("%s -- Mean branch contribution magnitude by prediction category "
                 "(%s)" % (model_name, partition))
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = out_dir / ("%s_shapley_%s_fig_a_mean_abs_phi.png"
                      % (model_name.lower(), partition))
    plt.savefig(path, dpi=DPI, bbox_inches="tight"); plt.close()
    logger.info("Saved: %s", path)


# ---------------------------------------------------------------------------
# Figure B -- winner-branch percentage by category (HEADLINE)
# ---------------------------------------------------------------------------
def plot_b_winner_histogram(df, out_dir, model_name, partition, logger):
    ct = pd.crosstab(df["category"], df["winner_branch_abs"], normalize="index") * 100
    ct = ct.reindex(index=CATEGORIES, columns=BRANCH_KEYS, fill_value=0.0)
    counts = df["category"].value_counts().reindex(CATEGORIES, fill_value=0)

    x = np.arange(len(CATEGORIES))
    bw = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (k, label, color) in enumerate(zip(BRANCH_KEYS, BRANCH_LABELS, BRANCH_COLORS)):
        vals = ct[k].values
        bars = ax.bar(x + (i - 1) * bw, vals, bw, label=label, color=color,
                      edgecolor=BRANCH_EDGE, linewidth=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, "%.1f%%" % v,
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["%s\n(n=%s)" % (c, "{:,}".format(int(counts[c]))) for c in CATEGORIES])
    ax.set_ylabel("% of segments where branch is argmax|phi|")
    ax.set_title("%s -- Dominant branch by prediction category (%s)"
                 % (model_name, partition))
    ax.set_ylim(0, max(105, ct.values.max() + 8))
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = out_dir / ("%s_shapley_%s_fig_b_winner_branch_histogram.png"
                      % (model_name.lower(), partition))
    plt.savefig(path, dpi=DPI, bbox_inches="tight"); plt.close()
    logger.info("Saved: %s", path)


# ---------------------------------------------------------------------------
# Figure C -- mean signed phi per branch by category
# ---------------------------------------------------------------------------
def plot_c_mean_signed_phi(df, out_dir, model_name, partition, logger):
    means = (df.groupby("category", observed=True)[["phi_B1", "phi_B2", "phi_B3"]]
               .mean().reindex(CATEGORIES))
    counts = df["category"].value_counts().reindex(CATEGORIES, fill_value=0)

    x = np.arange(len(CATEGORIES))
    bw = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (col, label, color) in enumerate(zip(
            ["phi_B1", "phi_B2", "phi_B3"], BRANCH_LABELS, BRANCH_COLORS)):
        vals = means[col].values
        bars = ax.bar(x + (i - 1) * bw, vals, bw, label=label, color=color,
                      edgecolor=BRANCH_EDGE, linewidth=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2,
                    v + (0.05 if v >= 0 else -0.15),
                    "%+0.2f" % v,
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(["%s\n(n=%s)" % (c, "{:,}".format(int(counts[c]))) for c in CATEGORIES])
    ax.set_ylabel("Mean signed phi (logit units; positive = pushes toward seizure)")
    ax.set_title("%s -- Signed branch contribution by prediction category "
                 "(%s)" % (model_name, partition))
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = out_dir / ("%s_shapley_%s_fig_c_mean_signed_phi.png"
                      % (model_name.lower(), partition))
    plt.savefig(path, dpi=DPI, bbox_inches="tight"); plt.close()
    logger.info("Saved: %s", path)


# ---------------------------------------------------------------------------
# Figure D -- per-mouse heatmap of mean |phi| for a given category
# ---------------------------------------------------------------------------
def plot_d_per_mouse_heatmap(df, out_dir, model_name, partition, category, logger):
    sub = df[df["category"] == category]
    if len(sub) == 0:
        logger.warning("No %s segments -- skipping Fig D (%s).", category, category)
        return
    per_mouse = (sub.groupby("mouse_id", observed=True)[["abs_B1", "abs_B2", "abs_B3"]]
                    .mean())
    counts = sub.groupby("mouse_id", observed=True).size().rename("n")
    per_mouse = per_mouse.join(counts).sort_values("n", ascending=False)
    if len(per_mouse) == 0:
        logger.warning("No mice with %s segments -- skipping Fig D.", category)
        return
    heat = per_mouse[["abs_B1", "abs_B2", "abs_B3"]].values

    fig, (ax_heat, ax_count) = plt.subplots(
        1, 2, figsize=(8, max(4, 0.35 * len(per_mouse))),
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05})

    im = ax_heat.imshow(heat, aspect="auto", cmap="viridis")
    ax_heat.set_xticks(range(3))
    ax_heat.set_xticklabels(BRANCH_LABELS, rotation=20, ha="right")
    ax_heat.set_yticks(range(len(per_mouse)))
    ax_heat.set_yticklabels(per_mouse.index.tolist())
    ax_heat.set_title("%s -- Mean |phi| per mouse over %s segments (%s)"
                      % (model_name, category, partition))
    for i in range(len(per_mouse)):
        for j in range(3):
            v = heat[i, j]
            ax_heat.text(j, i, "%.2f" % v, ha="center", va="center",
                         color="white" if v < heat.max() * 0.55 else "black",
                         fontsize=8)
    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.85, pad=0.02)
    cbar.set_label("Mean |phi|")

    cat_color = CATEGORY_COLORS[category]
    y = np.arange(len(per_mouse))
    ax_count.barh(y, per_mouse["n"].values, color=cat_color, edgecolor="black",
                  linewidth=0.3)
    ax_count.set_yticks(y)
    ax_count.set_yticklabels([])
    ax_count.invert_yaxis()
    ax_heat.invert_yaxis()
    for i, v in enumerate(per_mouse["n"].values):
        ax_count.text(v, i, " %s" % "{:,}".format(int(v)),
                      va="center", ha="left", fontsize=8)
    ax_count.set_xlabel("%s count" % category)
    ax_count.grid(axis="x", alpha=0.3)
    ax_count.set_xlim(0, per_mouse["n"].max() * 1.25)

    plt.tight_layout()
    path = out_dir / ("%s_shapley_%s_fig_d_per_mouse_heatmap_%s.png"
                      % (model_name.lower(), partition, category))
    plt.savefig(path, dpi=DPI, bbox_inches="tight"); plt.close()
    logger.info("Saved: %s  (%d mice with >=1 %s)", path, len(per_mouse), category)


# ---------------------------------------------------------------------------
# Figure E -- |phi| distribution violins per branch per category
# ---------------------------------------------------------------------------
def plot_e_phi_violin(df, out_dir, model_name, partition, logger):
    # Subsample within each (category, branch) group to keep render time
    # bounded; KDE on millions of points is unnecessarily slow.
    pieces = []
    for cat in CATEGORIES:
        sub = df[df["category"] == cat]
        if len(sub) == 0:
            continue
        n = min(VIOLIN_MAX_PER_GROUP, len(sub))
        s = sub.sample(n=n, random_state=42)
        for bk, col in zip(BRANCH_KEYS, ["abs_B1", "abs_B2", "abs_B3"]):
            pieces.append(pd.DataFrame({"category": cat,
                                        "branch":   bk,
                                        "abs_phi":  s[col].values}))
    long_df = pd.concat(pieces, ignore_index=True)
    logger.info("Violin subsample: %d rows total (%d per cat-branch cap)",
                len(long_df), VIOLIN_MAX_PER_GROUP)

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.violinplot(data=long_df, x="category", y="abs_phi",
                   hue="branch", hue_order=BRANCH_KEYS, order=CATEGORIES,
                   palette=BRANCH_COLORS, inner="quartile", cut=0,
                   density_norm="width", ax=ax)
    ax.set_xlabel("Prediction category")
    ax.set_ylabel("|phi| (logit units)")
    ax.set_title("%s -- Distribution of branch contribution magnitude (%s)"
                 % (model_name, partition))
    # Replace the seaborn default branch labels in the legend with the full names.
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, BRANCH_LABELS, loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = out_dir / ("%s_shapley_%s_fig_e_phi_violin.png"
                      % (model_name.lower(), partition))
    plt.savefig(path, dpi=DPI, bbox_inches="tight"); plt.close()
    logger.info("Saved: %s", path)


# ---------------------------------------------------------------------------
# run_session
# ---------------------------------------------------------------------------
def run_session(model_name, partition, csv_override, output_dir_override):
    csv_path = csv_override or csv_path_for(model_name, partition)
    summary_path = summary_path_for(model_name, partition)
    out_dir = output_dir_override or (csv_path.parent / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(out_dir, model_name, partition)
    if not csv_path.exists():
        logger.error("CSV not found: %s -- skipping %s.", csv_path, model_name)
        return
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        logger.info("Summary  : n_segments=%d | efficiency_residual_max=%.2e",
                    summary.get("n_segments", -1),
                    summary.get("efficiency_residual_max", float("nan")))
    else:
        logger.warning("Summary JSON not found: %s", summary_path)

    df = load_csv(csv_path, logger)

    plot_a_mean_abs_phi(df, out_dir, model_name, partition, logger)
    plot_b_winner_histogram(df, out_dir, model_name, partition, logger)
    plot_c_mean_signed_phi(df, out_dir, model_name, partition, logger)
    plot_d_per_mouse_heatmap(df, out_dir, model_name, partition, "TP", logger)
    plot_d_per_mouse_heatmap(df, out_dir, model_name, partition, "FP", logger)
    plot_e_phi_violin(df, out_dir, model_name, partition, logger)

    logger.info("Done. Figures under %s", out_dir)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    requested = ["M3", "M4"] if args.model == "all" else [args.model]
    for mname in requested:
        run_session(mname, args.partition, args.csv_path, args.output_dir)


if __name__ == "__main__":
    main()
