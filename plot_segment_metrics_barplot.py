"""
plot_segment_metrics_barplot.py
===============================
Local plotting utility. Reads the per-row classification report JSONs
and per-partition event-level summary JSONs from recover_event_metrics.py
and emits paper-ready bar plots.

Outputs (3 PNG files):
  segment_metrics_barplot_val.png         -- segment-level, val
  segment_metrics_barplot_test.png        -- segment-level, test
  event_metrics_barplot_val_vs_test.png   -- event-level, val vs test
                                             (clinical headline figure)

Each segment-level plot mirrors the make_classreport_barplot layout:
  Top panel    -- 7 grouped bars per metric (raw vs post-processed):
                  accuracy, precision, recall (sensitivity),
                  specificity, F1-score, AUROC, PRAUC.
  Bottom panel -- segment-level FAR/hr (corrected, step_sec=2.5),
                  raw vs post-processed.

Event-level plot:
  Top panel    -- Recall, Precision, F1 (val vs test bars per metric)
                  with TP/(TP+FN) and TP/(TP+FP) absolute count labels.
  Bottom panel -- event-level FAR/hr (val vs test) with primary axis
                  in events/hour and secondary axis in events/24h.

Inputs  : MultiScaleTCN/event_metrics_recovery/recover_classification_report_{val,test}_row{1,2}.json
          MultiScaleTCN/event_metrics_recovery/recover_event_metrics_{val,test}.json
Outputs : MultiScaleTCN/event_metrics_recovery/figures/

Run     : python plot_segment_metrics_barplot.py
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LOCAL_ROOT  = Path(r"C:\Users\messi\OneDrive\Desktop\Desktop\TCN_UNIQURE_PROJECT")
RECOVERY    = LOCAL_ROOT / "MultiScaleTCN" / "event_metrics_recovery"
FIGURE_DIR  = RECOVERY / "figures"

PARTITIONS  = ("val", "test")
DISPLAY_NAMES = ["accuracy", "precision", "recall\n(sensitivity)",
                 "specificity", "F1-score", "AUROC", "PRAUC"]


def _truncate(v):
    return math.floor(float(v) * 100.0) / 100.0


def load_row_report(partition, row):
    path = RECOVERY / f"recover_classification_report_{partition}_row{row}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def row_values(report):
    macro = report.get("macro avg", {})
    return [
        float(report.get("accuracy", 0.0)),
        float(macro.get("precision", 0.0)),
        float(macro.get("recall", 0.0)),
        float(report.get("specificity", 0.0)),
        float(macro.get("f1-score", 0.0)),
        float(report.get("auroc", 0.0)),
        float(report.get("prauc", 0.0)),
    ]


def far_corrected(report):
    return float(report.get("far_per_hour_seg_CORRECTED_2_5s_denom", 0.0))


def plot_one_partition(partition, out_path):
    r1 = load_row_report(partition, 1)
    r2 = load_row_report(partition, 2)

    r1_vals = row_values(r1)
    r2_vals = row_values(r2)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12, 7),
        gridspec_kw={"height_ratios": [7, 3]})

    x = np.arange(len(DISPLAY_NAMES))
    w = 0.35
    b1 = ax_top.bar(x - w/2, r1_vals, w, color="#E8A87C", label="Raw score")
    b2 = ax_top.bar(x + w/2, r2_vals, w, color="#5A7DC8", label="Post-processed score")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax_top.annotate("%.2f" % _truncate(h),
                            xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", fontsize=7)
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(DISPLAY_NAMES, fontsize=9)
    ax_top.set_ylabel("Score")
    ax_top.set_title("Multi-Scale TCN Segment-Level Metrics -- %s set "
                     "(Raw vs Post-processed)" % partition.upper())
    ax_top.legend(fontsize=8)
    ax_top.set_ylim(0, 1.15)

    far_labels = ["Raw\nsegment-level", "Post-processed\nsegment-level"]
    far_vals   = [far_corrected(r1), far_corrected(r2)]
    bars_far = ax_bot.bar(far_labels, far_vals,
                          color=["#C85A5A", "#5A7DC8"], edgecolor="white")
    for bar in bars_far:
        h = bar.get_height()
        ax_bot.annotate("%.2f" % _truncate(h),
                        xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=8)
    ax_bot.set_ylabel("FAR/hr")
    ax_bot.set_title("Segment-Level False Alarm Rate per Hour")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def load_event_summary(partition):
    path = RECOVERY / f"recover_event_metrics_{partition}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def plot_event_level_val_vs_test(out_path):
    val_summary  = load_event_summary("val")
    test_summary = load_event_summary("test")
    val_em  = val_summary["event_level_metrics"]
    test_em = test_summary["event_level_metrics"]

    metric_keys   = ["recall", "precision", "f1"]
    metric_labels = ["Recall\n(Sensitivity)", "Precision", "F1"]
    val_vals  = [float(val_em[k])  for k in metric_keys]
    test_vals = [float(test_em[k]) for k in metric_keys]

    def count_label(metric, em):
        tp, fp, fn = em["tp"], em["fp"], em["fn"]
        if metric == "recall":
            return f"{tp}/{tp+fn}"
        if metric == "precision":
            return f"{tp}/{tp+fp}"
        return f"TP={tp}"

    val_counts  = [count_label(k, val_em)  for k in metric_keys]
    test_counts = [count_label(k, test_em) for k in metric_keys]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 7),
        gridspec_kw={"height_ratios": [7, 3]})

    x = np.arange(len(metric_labels))
    w = 0.35
    b_val  = ax_top.bar(x - w/2, val_vals,  w, color="#5A7DC8", label="Validation")
    b_test = ax_top.bar(x + w/2, test_vals, w, color="#E8A87C", label="Test")

    for bars, vals, counts in [(b_val, val_vals, val_counts),
                               (b_test, test_vals, test_counts)]:
        for bar, v, c in zip(bars, vals, counts):
            h = bar.get_height()
            ax_top.annotate("%.2f" % _truncate(v),
                            xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 12), textcoords="offset points",
                            ha="center", fontsize=9, fontweight="bold")
            ax_top.annotate(c,
                            xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", fontsize=7, color="#444")

    ax_top.set_xticks(x)
    ax_top.set_xticklabels(metric_labels, fontsize=10)
    ax_top.set_ylabel("Score")
    ax_top.set_title("Multi-Scale TCN Event-Level Metrics -- Validation vs Test")
    ax_top.legend(fontsize=9, loc="upper right")
    ax_top.set_ylim(0, 1.20)

    far_val  = float(val_em["far_per_hour_event_CORRECTED"])
    far_test = float(test_em["far_per_hour_event_CORRECTED"])
    far_labels = ["Validation", "Test"]
    far_vals   = [far_val, far_test]
    bars_far = ax_bot.bar(far_labels, far_vals,
                          color=["#5A7DC8", "#E8A87C"], edgecolor="white")
    for bar, v in zip(bars_far, far_vals):
        h = bar.get_height()
        ax_bot.annotate("%.3f / hr   (~%.1f / 24 h)" % (v, v * 24),
                        xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=9)
    ax_bot.set_ylabel("Event-level FAR per hour")
    ax_bot.set_title("Event-Level False Alarm Rate")
    ax_bot.set_ylim(0, max(far_vals) * 1.45)
    ax_bot.secondary_yaxis("right",
                           functions=(lambda x: x * 24, lambda x: x / 24)).set_ylabel(
                               "Events per 24 h")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for partition in PARTITIONS:
        out_path = FIGURE_DIR / f"segment_metrics_barplot_{partition}.png"
        plot_one_partition(partition, out_path)
    plot_event_level_val_vs_test(FIGURE_DIR / "event_metrics_barplot_val_vs_test.png")


if __name__ == "__main__":
    main()
