"""
replot_postproc_sweep.py
========================
Regenerate the postproc_sweep val-vs-test impact figure from the ALREADY
COMPUTED comparison CSV -- no NPZ, no model, no data reload. This exists
purely to fix legibility ("images small, hard to see"): it reproduces the
exact curves/points of postproc_sweep.plot_impact but with larger fonts
and a VECTOR (PDF) output that stays sharp at any zoom in the LaTeX PDF.

The numbers are untouched; only font sizes, line/marker weights, figure
size and output format change.

Usage
-----
    python replot_postproc_sweep.py \
        --csv  "<...>/post_process_varing_sec/comparison_val_vs_test.csv" \
        --out  "<...>/post_process_varing_sec/impact_val_vs_test_MSTCN_big" \
        --model-label "M3 (MultiScaleTCN)"

--out is a stem: both <out>.pdf (vector, for LaTeX) and <out>.png
(dpi=300 raster preview) are written. Defaults point at the MultiScaleTCN
val/test sweep folder.
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Style constants (mirrors postproc_sweep.py) ──────────────────────────
ORDERINGS = ["refractory_then_min", "min_then_refractory"]
SWEEP_SECS = [10, 15, 20, 25, 30]

ORDER_LABEL = {
    "refractory_then_min": "Refractory -> Min-dur",
    "min_then_refractory": "Min-dur -> Refractory",
}
ORDER_COLOR = {
    "refractory_then_min": "#5A7DC8",
    "min_then_refractory": "#C85A5A",
}
ORDER_LINESTYLE = {"refractory_then_min": "-",  "min_then_refractory": "--"}
ORDER_MARKER    = {"refractory_then_min": "o",  "min_then_refractory": "^"}

# Fixed per-metric colours in the P/R/F1 panel (same as source script).
PREC_COLOR, REC_COLOR, F1_COLOR = "#5A7DC8", "#5AC880", "#C8A05A"

DEFAULT_ROOT = Path(
    r"C:\Users\messi\OneDrive\Desktop\Desktop\TCN_UNIQURE_PROJECT"
    r"\MultiScaleTCN\evaluation\post_process_varing_sec")

# ── Legibility: bump every default font well above the source script's
#    fontsize=7/8 legends and matplotlib's default ~10 pt ticks. ──────────
plt.rcParams.update({
    "font.size":        15,
    "axes.titlesize":   17,
    "axes.labelsize":   16,
    "xtick.labelsize":  14,
    "ytick.labelsize":  14,
    "legend.fontsize":  12,
    "figure.titlesize": 20,
})

NUM_FIELDS = {
    "min_event_sec", "event_precision", "event_recall", "event_f1",
    "event_far_per_hour",
}


def load_rows(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in NUM_FIELDS:
                r[k] = float(r[k])
            r["min_event_sec"] = int(r["min_event_sec"])
            rows.append(r)
    return rows


def subset(rows, partition, order):
    sub = [r for r in rows if r["partition"] == partition and r["order"] == order]
    sub.sort(key=lambda r: r["min_event_sec"])
    return sub


def plot(rows, out_stem, model_label, partitions):
    n_rows = len(partitions)
    # Wider + taller than the source (15 x 4.5/row) for breathing room.
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5.6 * n_rows + 0.6))
    if n_rows == 1:
        axes = axes.reshape(1, 3)

    for row_i, partition in enumerate(partitions):
        # ── Panel 0: Precision / Recall / F1 ────────────────────────────
        ax = axes[row_i, 0]
        for order in ORDERINGS:
            sub = subset(rows, partition, order)
            if not sub:
                continue
            secs = [r["min_event_sec"]   for r in sub]
            ls, mk, lbl = ORDER_LINESTYLE[order], ORDER_MARKER[order], ORDER_LABEL[order]
            ax.plot(secs, [r["event_precision"] for r in sub], linestyle=ls, marker=mk,
                    color=PREC_COLOR, linewidth=2.4, markersize=8, label=f"Precision ({lbl})")
            ax.plot(secs, [r["event_recall"] for r in sub], linestyle=ls, marker=mk,
                    color=REC_COLOR, linewidth=2.4, markersize=8, label=f"Recall ({lbl})")
            ax.plot(secs, [r["event_f1"] for r in sub], linestyle=ls, marker=mk,
                    color=F1_COLOR, linewidth=2.4, markersize=8, label=f"F1 ({lbl})")
        ax.set_xlabel("MIN_EVENT_SEC (s)"); ax.set_ylabel("Score")
        ax.set_title(f"[{partition.upper()}] Event Precision / Recall / F1")
        ax.set_xticks(SWEEP_SECS); ax.grid(True, alpha=0.3)

        # ── Panel 1: FAR/hr ─────────────────────────────────────────────
        ax = axes[row_i, 1]
        for order in ORDERINGS:
            sub = subset(rows, partition, order)
            if not sub:
                continue
            ax.plot([r["min_event_sec"] for r in sub],
                    [r["event_far_per_hour"] for r in sub],
                    linestyle=ORDER_LINESTYLE[order], marker=ORDER_MARKER[order],
                    color=ORDER_COLOR[order], linewidth=2.4, markersize=9,
                    label=ORDER_LABEL[order])
        ax.set_xlabel("MIN_EVENT_SEC (s)"); ax.set_ylabel("Event-level FAR/hr")
        ax.set_title(f"[{partition.upper()}] Event-level FAR/hr")
        ax.set_xticks(SWEEP_SECS); ax.grid(True, alpha=0.3)

        # ── Panel 2: Recall vs FAR/hr Pareto ────────────────────────────
        ax = axes[row_i, 2]
        for order in ORDERINGS:
            sub = subset(rows, partition, order)
            if not sub:
                continue
            recs = [r["event_recall"]       for r in sub]
            fars = [r["event_far_per_hour"] for r in sub]
            secs = [r["min_event_sec"]      for r in sub]
            ax.scatter(recs, fars, s=130, color=ORDER_COLOR[order],
                       marker=ORDER_MARKER[order], zorder=3, label=ORDER_LABEL[order])
            for s, x_, y_ in zip(secs, recs, fars):
                ax.annotate(f"{s}s", xy=(x_, y_), xytext=(7, 5),
                            textcoords="offset points", fontsize=12,
                            color=ORDER_COLOR[order])
        ax.set_xlabel("Event recall (sensitivity)")
        ax.set_ylabel("Event-level FAR/hr")
        ax.set_title(f"[{partition.upper()}] Recall vs FAR/hr Pareto\n(bottom-right = ideal)")
        ax.grid(True, alpha=0.3)

    if partitions == ["val", "test"]:
        suptitle = ("Impact of post-processing order x MIN_EVENT_SEC on %s "
                    "-- val (top) vs test (bottom)" % model_label)
    else:
        suptitle = ("Impact of post-processing order x MIN_EVENT_SEC on %s "
                    "-- %s" % (model_label, " / ".join(p.upper() for p in partitions)))
    plt.suptitle(suptitle)

    # ── ONE shared legend for the whole figure ──────────────────────────
    # Every panel repeats the same two orderings; the P/R/F1 panel adds the
    # three metric colours. A single bottom strip explains all of it, so no
    # per-panel legend crowds the data or its neighbour, and the panels
    # reclaim the space. Metrics = colour (P/R/F1 panel); order = line style
    # + marker (black, so it reads in every panel regardless of that panel's
    # colour meaning).
    metric_handles = [
        Line2D([0], [0], color=PREC_COLOR, lw=2.6, label="Precision"),
        Line2D([0], [0], color=REC_COLOR,  lw=2.6, label="Recall"),
        Line2D([0], [0], color=F1_COLOR,   lw=2.6, label="F1"),
    ]
    order_handles = [
        Line2D([0], [0], color="black", lw=2.6, linestyle="-",  marker="o",
               markersize=9, label="Refractory -> Min-dur"),
        Line2D([0], [0], color="black", lw=2.6, linestyle="--", marker="^",
               markersize=9, label="Min-dur -> Refractory"),
    ]
    fig.legend(handles=metric_handles + order_handles,
               loc="lower center", ncol=5, fontsize=14, frameon=True,
               bbox_to_anchor=(0.5, 0.0),
               columnspacing=1.6, handlelength=2.6)

    # Reserve a thin strip at top (suptitle) and bottom (shared legend);
    # everything between is plot area. Small hspace/wspace since no panel
    # carries its own legend anymore -> maximise the plotted region.
    plt.tight_layout(rect=(0, 0.05, 1, 0.955))
    fig.subplots_adjust(hspace=0.30, wspace=0.28)

    out_stem = Path(out_stem)
    pdf_path = out_stem.with_suffix(".pdf")
    png_path = out_stem.with_suffix(".png")
    plt.savefig(pdf_path, bbox_inches="tight")            # vector -> LaTeX
    plt.savefig(png_path, dpi=300, bbox_inches="tight")   # high-res preview
    plt.close()
    print(f"Saved vector figure : {pdf_path}")
    print(f"Saved raster preview: {png_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=DEFAULT_ROOT / "comparison_val_vs_test.csv")
    p.add_argument("--out", type=Path, default=DEFAULT_ROOT / "impact_val_vs_test_MSTCN_big")
    p.add_argument("--model-label", default="M3 (MultiScaleTCN)")
    p.add_argument("--partitions", nargs="+", default=["val", "test"])
    args = p.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")
    rows = load_rows(args.csv)
    print(f"Loaded {len(rows)} rows from {args.csv}")
    plot(rows, args.out, args.model_label, args.partitions)


if __name__ == "__main__":
    main()
