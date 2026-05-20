"""
plot_segment_metrics_barplot.py
===============================
Local plotting utility (no GPU, no model forward pass). Reads the per-row
classification report JSONs and per-partition event-level summary JSONs
that the recovery / evaluation scripts produced, and emits paper-ready
bar plots.

Variant-parameterised in the same style as train_eval.py:
    python plot_segment_metrics_barplot.py --variant MultiScaleTCN
    python plot_segment_metrics_barplot.py --variant MultiScaleTCNWithAttention
    python plot_segment_metrics_barplot.py --variant TCN                       # future
    python plot_segment_metrics_barplot.py --variant TCNWithAttention          # future

Outputs (per variant, three PNG files prefixed with the variant's
output_prefix):
    <prefix>_segment_metrics_barplot_val.png     -- segment-level, val
    <prefix>_segment_metrics_barplot_test.png    -- segment-level, test
    <prefix>_event_metrics_barplot_val_vs_test.png -- event-level, val vs test

Each segment-level plot mirrors the make_classreport_barplot layout:
    Top panel    -- 7 grouped bars per metric (raw vs post-processed):
                    accuracy, precision (macro), recall (sensitivity),
                    specificity, F1-score (macro), AUROC, PRAUC.
    Bottom panel -- segment-level FAR/hr (corrected, step_sec=2.5),
                    raw vs post-processed.

Event-level plot:
    Top panel    -- Recall, Precision, F1 (val vs test bars per metric)
                    with TP/(TP+FN) and TP/(TP+FP) absolute count labels.
    Bottom panel -- event-level FAR/hr (val vs test) with primary axis
                    in events/hour and secondary axis in events/24 h.

Variant inputs (per partition) and outputs are described in VARIANT_CONFIG
below; override paths via --local-root or per-variant flags.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LOCAL_DEFAULT = Path(r"C:\Users\messi\OneDrive\Desktop\Desktop\TCN_UNIQURE_PROJECT")

PARTITIONS    = ("val", "test")
DISPLAY_NAMES = ["accuracy", "precision", "recall\n(sensitivity)",
                 "specificity", "F1-score", "AUROC", "PRAUC"]


# ---------------------------------------------------------------------------
# Per-variant config. Each partition entry gives the directory holding its
# classification-report + event-summary JSONs and the filename templates.
# {row} is filled with 1 or 2 for the per-row classification report.
# ---------------------------------------------------------------------------
def variant_config(local_root):
    return {
        "TCN": {
            "model_label":   "TCN",
            "output_prefix": "tcn",
            "figure_dir":    local_root / "TCN" / "figures",
            "partitions": {
                "val": {
                    "dir":         local_root / "TCN",
                    "row_report":  "tcn_classification_report_row{row}.json",
                    "summary":     "tcn_evaluation_report.json",
                },
                "test": {
                    "dir":         local_root / "TCN" / "evaluation",
                    "row_report":  "tcn_classification_report_row{row}.json",
                    "summary":     "tcn_evaluation_report.json",
                },
            },
        },
        "TCNWithAttention": {
            "model_label":   "TCN + Attention",
            "output_prefix": "tcn_attention",
            "figure_dir":    local_root / "TCNAttention" / "figures",
            "partitions": {
                "val": {
                    "dir":         local_root / "TCNAttention",
                    "row_report":  "tcn_attention_classification_report_row{row}.json",
                    "summary":     "tcn_attention_evaluation_report.json",
                },
                "test": {
                    "dir":         local_root / "TCNAttention" / "evaluation",
                    "row_report":  "tcn_attention_classification_report_row{row}.json",
                    "summary":     "tcn_attention_evaluation_report.json",
                },
            },
        },
        "MultiScaleTCN": {
            "model_label":   "Multi-Scale TCN",
            "output_prefix": "multiscale_tcn",
            "figure_dir":    local_root / "MultiScaleTCN" / "event_metrics_recovery" / "figures",
            "partitions": {
                "val": {
                    "dir":         local_root / "MultiScaleTCN" / "event_metrics_recovery",
                    "row_report":  "recover_classification_report_val_row{row}.json",
                    "summary":     "recover_event_metrics_val.json",
                },
                "test": {
                    "dir":         local_root / "MultiScaleTCN" / "event_metrics_recovery",
                    "row_report":  "recover_classification_report_test_row{row}.json",
                    "summary":     "recover_event_metrics_test.json",
                },
            },
        },
        "MultiScaleTCNWithAttention": {
            "model_label":   "Multi-Scale TCN + Attention",
            "output_prefix": "ms_attn",
            "figure_dir":    local_root / "MultiScaleTCNAttention" / "evaluation" / "figures_2",
            "partitions": {
                "val": {
                    "dir":         local_root / "MultiScaleTCNAttention" / "val_event_metrics",
                    "row_report":  "ms_attn_classification_report_row{row}.json",
                    "summary":     "ms_attn_evaluation_report.json",
                },
                "test": {
                    "dir":         local_root / "MultiScaleTCNAttention" / "evaluation",
                    "row_report":  "ms_attn_classification_report_row{row}.json",
                    "summary":     "ms_attn_evaluation_report.json",
                },
            },
        },
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True,
                   choices=["TCN", "TCNWithAttention",
                            "MultiScaleTCN", "MultiScaleTCNWithAttention"])
    p.add_argument("--local-root", type=Path, default=LOCAL_DEFAULT,
                   help="Local data root (default: %(default)s).")
    p.add_argument("--figure-dir", type=Path, default=None,
                   help="Override figure-output dir (default: per-variant).")
    return p.parse_args()


def _truncate(v):
    return math.floor(float(v) * 100.0) / 100.0


def load_row_report(cfg, partition, row):
    pcfg = cfg["partitions"][partition]
    path = pcfg["dir"] / pcfg["row_report"].format(row=row)
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_event_summary(cfg, partition):
    pcfg = cfg["partitions"][partition]
    path = pcfg["dir"] / pcfg["summary"]
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


def plot_one_partition(cfg, partition, out_path):
    r1 = load_row_report(cfg, partition, 1)
    r2 = load_row_report(cfg, partition, 2)

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
    ax_top.set_title("%s Segment-Level Metrics -- %s set "
                     "(Raw vs Post-processed)"
                     % (cfg["model_label"], partition.upper()))
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


def plot_event_level_val_vs_test(cfg, out_path):
    val_summary  = load_event_summary(cfg, "val")
    test_summary = load_event_summary(cfg, "test")
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
    ax_top.set_title("%s Event-Level Metrics -- Validation vs Test"
                     % cfg["model_label"])
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
    ax_bot.set_ylim(0, max(far_vals) * 1.45 if max(far_vals) > 0 else 1.0)
    ax_bot.secondary_yaxis("right",
                           functions=(lambda x: x * 24, lambda x: x / 24)).set_ylabel(
                               "Events per 24 h")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    args = parse_args()
    cfg = variant_config(args.local_root)[args.variant]
    figure_dir = args.figure_dir or cfg["figure_dir"]
    figure_dir.mkdir(parents=True, exist_ok=True)

    prefix = cfg["output_prefix"]
    print(f"Variant     : {args.variant}")
    print(f"Local root  : {args.local_root}")
    print(f"Figure dir  : {figure_dir}")
    print(f"Output stem : {prefix}_*")
    print("-" * 60)

    for partition in PARTITIONS:
        out_path = figure_dir / f"{prefix}_segment_metrics_barplot_{partition}.png"
        plot_one_partition(cfg, partition, out_path)
    plot_event_level_val_vs_test(
        cfg, figure_dir / f"{prefix}_event_metrics_barplot_val_vs_test.png")


if __name__ == "__main__":
    main()
