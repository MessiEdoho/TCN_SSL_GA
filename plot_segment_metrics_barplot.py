"""
plot_segment_metrics_barplot.py
===============================
Local plotting utility (no GPU, no model forward pass). Reads the per-row
classification report JSONs and per-partition event-level summary JSONs
that the recovery / evaluation scripts produced, and emits paper-ready
bar plots.

Variant-parameterised in the same style as raw_segment_level_3partitions.py:
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
import csv
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
    """All partition inputs read from the postproc_sweep.py output for the
    LOCKED operating point: order = min_then_refractory, MIN_EVENT_SEC = 25 s.
    Per-partition file naming is uniform across variants because the sweep
    writes generic '{partition}_summary.json' / '{partition}_classification_
    report_row{row}.json'. Figure output goes to a per-variant
    'evaluation_figure_paper/' folder, distinct from earlier 'figures_2/'
    or 'event_metrics_recovery/figures/' folders that hold artefacts from
    the prior operating point.
    """
    LOCK = ("evaluation", "post_process_varing_sec",
            "{partition}", "min_then_refractory", "MIN_EVENT_SEC_25s")

    def _partition_dir(model_root, partition):
        # Build path from the LOCK template with the partition substituted.
        return model_root.joinpath(*(s.format(partition=partition) for s in LOCK))

    m1 = local_root / "TCN"
    m2 = local_root / "TCNAttention"
    m3 = local_root / "MultiScaleTCN"
    m4 = local_root / "MultiScaleTCNAttention"

    def _variant_entry(model_label, output_prefix, model_root):
        return {
            "model_label":   model_label,
            "output_prefix": output_prefix,
            "figure_dir":    model_root / "evaluation_figure_paper",
            "partitions": {
                "val": {
                    "dir":         _partition_dir(model_root, "val"),
                    "row_report":  "val_classification_report_row{row}.json",
                    "summary":     "val_summary.json",
                },
                "test": {
                    "dir":         _partition_dir(model_root, "test"),
                    "row_report":  "test_classification_report_row{row}.json",
                    "summary":     "test_summary.json",
                },
            },
        }

    return {
        "TCN":                        _variant_entry("TCN",                          "tcn",            m1),
        "TCNWithAttention":           _variant_entry("TCN + Attention",              "tcn_attention",  m2),
        "MultiScaleTCN":              _variant_entry("Multi-Scale TCN",              "multiscale_tcn", m3),
        "MultiScaleTCNWithAttention": _variant_entry("Multi-Scale TCN + Attention",  "ms_attn",        m4),
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


def far_from_eval_report(seg_block, class_report):
    """Resolve segment-level FAR/hr (corrected, step_sec=2.5) from the most
    authoritative source available. Preference order:
      1. eval_report.json -> segment_level_metrics -> row{1,2} block
         (always present for runs that used eval_utils.evaluate_event_level).
      2. classification_report JSON top-level key (only present for newer
         runs that called eval_utils.build_classification_report after the
         FAR field was added).
    Returns 0.0 if neither source has it (e.g., legacy non-chronology runs).
    """
    far = seg_block.get("far_per_hour_seg_CORRECTED_2_5s_denom")
    if far is None:
        far = class_report.get("far_per_hour_seg_CORRECTED_2_5s_denom")
    return float(far) if far is not None else 0.0


def plot_one_partition(cfg, partition, out_path):
    r1 = load_row_report(cfg, partition, 1)
    r2 = load_row_report(cfg, partition, 2)

    # FAR/hr lives in the eval-report's segment_level_metrics block; the
    # per-row classification_report JSONs from older runs may not have it.
    summary  = load_event_summary(cfg, partition)
    seg      = summary.get("segment_level_metrics", {}) or {}
    seg_row1 = seg.get("row1_raw_threshold_0_5", {}) or {}
    seg_row2 = seg.get("row2_postproc_threshold_0_5", {}) or {}

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
    far_vals   = [far_from_eval_report(seg_row1, r1),
                  far_from_eval_report(seg_row2, r2)]
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


def plot_far_3stage_val_vs_test(cfg, out_path):
    """Three-stage FAR/hr improvement figure -- six bars (2 partitions x 3
    pipeline stages). Stages, from left to right within each partition:
      (A) Raw segment-level FAR/hr        -- raw classifier at tau=0.5.
      (B) Post-processed segment-level    -- after smoothing + threshold
                                             + run-detection + min-dur
                                             + refractory; scored per
                                             segment on smoothed_preds.
      (C) Post-processed event-level      -- per detected event after
                                             any-overlap GT matching.
    Stages (A) and (B) are scored on segments; (C) is scored on events,
    so the y-axis is FAR/hr but the units of "alarm" change between
    (A,B) and (C). The figure exists precisely to make that contrast
    visible -- it is the headline narrative of the post-processing
    contribution: noise spikes -> filtered segments -> clinical events.
    """
    rows = {}
    for p in PARTITIONS:
        s = load_event_summary(cfg, p)
        seg = s.get("segment_level_metrics", {}) or {}
        em  = s.get("event_level_metrics", {}) or {}
        r1  = seg.get("row1_raw_threshold_0_5", {}) or {}
        r2  = seg.get("row2_postproc_threshold_0_5", {}) or {}
        rows[p] = (
            float(r1.get("far_per_hour_seg_CORRECTED_2_5s_denom", 0.0)),
            float(r2.get("far_per_hour_seg_CORRECTED_2_5s_denom", 0.0)),
            float(em.get("far_per_hour_event_CORRECTED", 0.0)),
        )

    stage_labels = ["Raw\nsegment-level",
                    "Post-processed\nsegment-level",
                    "Post-processed\nevent-level"]
    stage_colors = ["#C85A5A", "#E8A87C", "#5A7DC8"]   # red -> orange -> blue

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(PARTITIONS))           # 0 = val, 1 = test
    w = 0.27

    for i, (lbl, color) in enumerate(zip(stage_labels, stage_colors)):
        x_offset = (i - 1) * w
        vals = [rows[p][i] for p in PARTITIONS]
        bars = ax.bar(x + x_offset, vals, w, color=color, label=lbl,
                      edgecolor="white")
        for bar, v in zip(bars, vals):
            h = bar.get_height()
            ax.annotate("%.2f" % v,
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["Validation", "Test"], fontsize=11)
    ax.set_ylabel("FAR per hour", fontsize=11)
    ax.set_title("%s False Alarm Rate -- Three-stage pipeline impact "
                 "(raw segment -> post-processed segment -> event-level)"
                 % cfg["model_label"], fontsize=11)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def write_per_mouse_csv(cfg, partition, out_path):
    """Extract the per-mouse breakdown block from the eval summary JSON
    and write a paper-table CSV. Source is the locked-operating-point
    sweep output (MIN_EVENT_SEC=25 s, order=min_then_refractory). One
    row per mouse, sorted by mouse_id for stable diffing.
    """
    summary = load_event_summary(cfg, partition)
    per_mouse = summary.get("per_mouse", {}) or {}
    fieldnames = [
        "mouse_id",
        "n_ground_truth_seizures", "n_predicted_events",
        "tp", "fp", "fn",
        "precision", "recall", "f1",
        "non_ictal_segments", "non_ictal_hours",
        "far_per_hour", "mean_detection_latency_sec", "n_chunks",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for mouse_id in sorted(per_mouse.keys()):
            row = dict(per_mouse[mouse_id])
            row["mouse_id"] = mouse_id
            writer.writerow(row)
    print(f"Saved: {out_path} ({len(per_mouse)} mice)")


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
    print(f"Source      : MIN_EVENT_SEC=25s, order=min_then_refractory "
          f"(locked operating point; sweep output)")
    print("-" * 60)

    # Row 1 vs Row 2 segment-level barplots (per partition; 7 metrics + seg FAR)
    for partition in PARTITIONS:
        out_path = figure_dir / f"{prefix}_segment_metrics_barplot_{partition}.png"
        plot_one_partition(cfg, partition, out_path)

    # Event-level P/R/F1 + event FAR/hr (val vs test) -- primary headline result
    plot_event_level_val_vs_test(
        cfg, figure_dir / f"{prefix}_event_metrics_barplot_val_vs_test.png")

    # Three-stage FAR/hr improvement (raw seg -> post seg -> post event)
    plot_far_3stage_val_vs_test(
        cfg, figure_dir / f"{prefix}_far_3stage_val_vs_test.png")

    # Per-mouse breakdown CSVs (val + test) -- paper appendix table
    for partition in PARTITIONS:
        out_path = figure_dir / f"{prefix}_per_mouse_table_{partition}.csv"
        write_per_mouse_csv(cfg, partition, out_path)


if __name__ == "__main__":
    main()
