"""
postproc_sweep.py
=================
Sweep post-processing parameters (refractory-vs-min-duration order;
MIN_EVENT_SEC) for any of the four model variants, on both val and test
partitions in a single run. Reads cached _full predictions NPZ -- no
model forward pass, no GPU.

Sweep grid per partition:
    order ∈ {refractory_then_min, min_then_refractory}
    MIN_EVENT_SEC ∈ {10, 15, 20, 30}    (seconds)
= 8 configurations per partition × 2 partitions = 16 evaluations.

Smoothing window (W=3), threshold (tau=0.5), refractory period (30 s),
and FAR/hr denominator (step_sec = 2.5 s) are held at canonical values;
only the post-processing order and MIN_EVENT_SEC vary.

Two orderings under comparison:
  - "refractory_then_min" (legacy default): refractory merge, then drop
    survivors shorter than MIN_EVENT_SEC. Can rescue fragmented true
    detections (and fragmented false alarms).
  - "min_then_refractory" (clinical-standard order; Saab 2020, Tang 2022,
    Persyst, Encevis): drop short candidate events first, then merge any
    survivors that are < 30 s apart. Strictly more conservative.

Variant + partition + order sweep is the methodological ablation the
paper reports as "Section X.Y: Post-processing parameter ablation".

Dual-mode execution
-------------------
By default reads/writes the local Windows-mirror layout under
TCN_UNIQURE_PROJECT/. Pass --cluster to switch to the cluster paths under
/home/people/22206468/scratch/OUTPUT/. Override individual paths via
--val-npz / --test-npz / --val-annot-dir / --test-annot-dir / --metadata
/ --output-dir.

Variant flag
------------
    python postproc_sweep.py --variant MultiScaleTCN              # M3
    python postproc_sweep.py --variant MultiScaleTCNWithAttention # M4
    python postproc_sweep.py --variant TCN                        # M1 (when NPZs exist)
    python postproc_sweep.py --variant TCNWithAttention           # M2 (when NPZs exist)

Outputs (per variant, under <output-dir>)
    val/
        refractory_then_min/  MIN_EVENT_SEC_10s/  15s/  20s/  30s/
        min_then_refractory/  MIN_EVENT_SEC_10s/  15s/  20s/  30s/
    test/
        refractory_then_min/  MIN_EVENT_SEC_10s/  15s/  20s/  30s/
        min_then_refractory/  MIN_EVENT_SEC_10s/  15s/  20s/  30s/
    comparison_val_vs_test.csv         (16 rows: partition x order x sec)
    impact_val_vs_test.png             (2x3 panels: rows = partitions)
    postproc_sweep.log
"""

import argparse
import csv
import datetime
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import eval_utils
from eval_utils import (
    evaluate_event_level, build_classification_report,
)


CLUSTER_OUTPUT  = Path("/home/people/22206468/scratch/OUTPUT")
CLUSTER_SCRATCH = Path("/home/people/22206468/scratch")
LOCAL_DEFAULT   = Path(r"C:\Users\messi\OneDrive\Desktop\Desktop\TCN_UNIQURE_PROJECT")

PARTITIONS = ["val", "test"]
ORDERINGS  = ["refractory_then_min", "min_then_refractory"]
SWEEP_SECS = [10, 15, 20, 30]

ORDER_LABEL = {
    "refractory_then_min": "Refractory -> Min-dur",
    "min_then_refractory": "Min-dur -> Refractory",
}
ORDER_COLOR = {
    "refractory_then_min": "#5A7DC8",
    "min_then_refractory": "#C85A5A",
}
ORDER_LINESTYLE = {
    "refractory_then_min": "-",
    "min_then_refractory": "--",
}
ORDER_MARKER = {
    "refractory_then_min": "o",
    "min_then_refractory": "^",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True,
                   choices=["TCN", "TCNWithAttention",
                            "MultiScaleTCN", "MultiScaleTCNWithAttention"])
    p.add_argument("--cluster", action="store_true",
                   help="Use cluster paths (/home/people/22206468/scratch/...). "
                        "Default: local Windows-mirror paths.")
    p.add_argument("--local-root",   type=Path, default=LOCAL_DEFAULT)
    p.add_argument("--cluster-root", type=Path, default=CLUSTER_OUTPUT,
                   help="Override the cluster OUTPUT root (default /home/people/22206468/scratch/OUTPUT).")
    p.add_argument("--val-npz",        type=Path, default=None, help="Override val NPZ path.")
    p.add_argument("--test-npz",       type=Path, default=None, help="Override test NPZ path.")
    p.add_argument("--metadata",       type=Path, default=None, help="Override mouse metadata JSON path.")
    p.add_argument("--val-annot-dir",  type=Path, default=None, help="Override val annotations dir.")
    p.add_argument("--test-annot-dir", type=Path, default=None, help="Override test annotations dir.")
    p.add_argument("--output-dir",     type=Path, default=None, help="Override output root.")
    return p.parse_args()


def variant_config(local_root, cluster_root, cluster_mode):
    """Return per-variant input/output paths plus partition-level annotation
    dirs and the global mouse_metadata path. Cluster mode collapses val and
    test annotations into the single shared folder (build_chronology.py
    convention); local mirror keeps them in separate folders.
    """
    if cluster_mode:
        m1 = cluster_root / "MODEL1_OUTPUT" / "TCN"
        m2 = cluster_root / "MODEL2_OUTPUT" / "TCNAttention"
        m3 = cluster_root / "MODEL3_OUTPUT" / "MultiScaleTCN"
        m4 = cluster_root / "MODEL4_OUTPUT" / "MultiScaleTCNAttention"
        val_annot  = CLUSTER_SCRATCH / "seizure_times_updated"
        test_annot = CLUSTER_SCRATCH / "seizure_times_updated"
        metadata   = CLUSTER_SCRATCH / "INPUT_DATA" / "Data_diagnostic" / "mouse_recording_metadata.json"
    else:
        m1 = local_root / "TCN"
        m2 = local_root / "TCNAttention"
        m3 = local_root / "MultiScaleTCN"
        m4 = local_root / "MultiScaleTCNAttention"
        val_annot  = local_root / "val_seizure_annot_updated"
        test_annot = local_root / "test_seizure_annot_updated"
        metadata   = local_root / "mouse_recording_metadata.json"

    variants = {
        "TCN": {
            "model_label": "M1 (TCN)",
            "val_npz":  m1 / "tcn_val_predictions_full.npz",
            "test_npz": m1 / "evaluation" / "tcn_test_predictions_full.npz",
            "output":   m1 / "evaluation" / "post_process_varing_sec",
        },
        "TCNWithAttention": {
            "model_label": "M2 (TCNAttention)",
            "val_npz":  m2 / "tcn_attention_val_predictions_full.npz",
            "test_npz": m2 / "evaluation" / "tcn_attention_test_predictions_full.npz",
            "output":   m2 / "evaluation" / "post_process_varing_sec",
        },
        "MultiScaleTCN": {
            "model_label": "M3 (MultiScaleTCN)",
            "val_npz":  m3 / "multiscale_tcn_val_predictions_full.npz",
            "test_npz": m3 / "evaluation" / "multiscale_tcn_test_predictions_full.npz",
            "output":   m3 / "evaluation" / "post_process_varing_sec",
        },
        "MultiScaleTCNWithAttention": {
            "model_label": "M4 (MultiScaleTCNAttention)",
            "val_npz":  m4 / "val_event_metrics" / "ms_attn_val_predictions_full.npz",
            "test_npz": m4 / "evaluation" / "ms_attn_test_predictions_full.npz",
            "output":   m4 / "evaluation" / "post_process_varing_sec",
        },
    }
    return variants, val_annot, test_annot, metadata


def resolve_paths(args):
    variants, val_annot, test_annot, metadata = variant_config(
        args.local_root, args.cluster_root, args.cluster)
    vcfg = variants[args.variant]
    return {
        "model_label":    vcfg["model_label"],
        "val_npz":        args.val_npz        or vcfg["val_npz"],
        "test_npz":       args.test_npz       or vcfg["test_npz"],
        "val_annot_dir":  args.val_annot_dir  or val_annot,
        "test_annot_dir": args.test_annot_dir or test_annot,
        "metadata":       args.metadata       or metadata,
        "output":         args.output_dir     or vcfg["output"],
    }


def setup_logging(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "postproc_sweep.log"
    logger = logging.getLogger("postproc_sweep")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8"); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger, log_path


def records_from_npz(npz, logger):
    """Build per-segment record dicts directly from a `_full` predictions
    NPZ. The NPZ schema (saved by every *_evaluation.py and the M4 recovery
    script via eval_utils.evaluate_event_level['reordered_arrays']) stores
    y_true, y_prob, mouse_id, chrono_idx, t_start_sec as parallel arrays in
    chronological-per-mouse order. Building records from these fields
    avoids the manifest entirely and guarantees record[i] aligns with
    y_true[i] / y_prob[i].

    Returns (y_true_arr, y_prob_arr, records). records[i] has keys:
        mouse_id, chrono_idx, t_start_sec, label, filepath (sentinel "")
    """
    required = ("y_true", "y_prob", "mouse_id", "chrono_idx", "t_start_sec")
    missing = [k for k in required if k not in npz.files]
    if missing:
        raise KeyError(
            "NPZ missing required _full-schema fields: %s. Available: %s. "
            "This script requires the chronologically-reordered predictions "
            "NPZ (filename ends in _predictions_full.npz). The older _raw "
            "manifest-order NPZs are not supported here -- use a script that "
            "reads them via the data-splits manifest if you must."
            % (missing, list(npz.files)))

    y_true       = np.asarray(npz["y_true"]).astype(np.int64)
    y_prob       = np.asarray(npz["y_prob"]).astype(np.float64)
    mouse_id_arr = np.asarray(npz["mouse_id"])
    chrono_idx_arr   = np.asarray(npz["chrono_idx"]).astype(np.int64)
    t_start_sec_arr  = np.asarray(npz["t_start_sec"]).astype(np.float64)

    records = []
    for i in range(len(y_true)):
        records.append({
            "mouse_id":    str(mouse_id_arr[i]),
            "chrono_idx":  int(chrono_idx_arr[i]),
            "t_start_sec": float(t_start_sec_arr[i]),
            "label":       int(y_true[i]),
            "filepath":    "",
        })
    logger.info("Built %d records from NPZ (%d unique mice)",
                len(records),
                len({r["mouse_id"] for r in records}))
    return y_true, y_prob, records


def write_summary_json(out_path, model_label, partition, sec, order,
                       result, logger):
    payload = {
        "model":     model_label,
        "partition": partition,
        "timestamp": datetime.datetime.now().isoformat(),
        "post_processing_params": {
            "order":                  order,
            "smoothing_window":       eval_utils.SMOOTHING_WIN,
            "refractory_period_sec":  eval_utils.REFRACTORY_SEC,
            "min_event_duration_sec": sec,
            "threshold":              eval_utils.THRESHOLD,
            "step_sec":               eval_utils.STEP_SEC,
            "matching_rule":          "any-overlap",
        },
        "totals":                result["totals"],
        "segment_level_metrics": result["segment_level_metrics"],
        "event_level_metrics":   result["event_level_metrics"],
        "per_mouse":             result["per_mouse_results"],
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str),
                        encoding="utf-8")
    logger.info("      summary -> %s", out_path)


def write_event_details_csv(out_path, partition, result, logger):
    fieldnames = ["mouse_id", "partition", "is_true_alarm", "start_sec", "end_sec",
                  "duration_sec", "max_prob", "matched_gt_idx",
                  "matched_gt_start_sec", "matched_gt_end_sec"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["all_event_details"]:
            r = dict(row)
            r["partition"] = partition
            writer.writerow(r)
    logger.info("      event details -> %s (%d rows)",
                out_path, len(result["all_event_details"]))


def write_classification_reports(out_dir, partition, result, logger):
    seg = result["segment_level_metrics"]
    reord = result["reordered_arrays"]
    y_true = reord["y_true"]
    r1 = build_classification_report(y_true, reord["y_pred_row1"],
                                     seg["row1_raw_threshold_0_5"])
    r2 = build_classification_report(y_true, reord["y_pred_row2"],
                                     seg["row2_postproc_threshold_0_5"])
    p1 = out_dir / f"{partition}_classification_report_row1.json"
    p2 = out_dir / f"{partition}_classification_report_row2.json"
    p1.write_text(json.dumps(r1, indent=2, sort_keys=True, default=str), encoding="utf-8")
    p2.write_text(json.dumps(r2, indent=2, sort_keys=True, default=str), encoding="utf-8")
    logger.info("      row1 report -> %s", p1)
    logger.info("      row2 report -> %s", p2)


def write_comparison_csv(out_path, rows, logger):
    fieldnames = [
        "partition", "order", "min_event_sec",
        "n_predicted_events", "tp", "fp", "fn",
        "event_precision", "event_recall", "event_f1",
        "event_far_per_hour", "event_mean_latency_sec",
        "seg_row2_f1_macro", "seg_row2_auroc", "seg_row2_precision",
        "seg_row2_recall", "seg_row2_specificity", "seg_row2_far_per_hour",
        "seg_row2_mcc",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    logger.info("Saved comparison CSV: %s (%d rows)", out_path, len(rows))


def plot_impact(rows, out_path, model_label, logger):
    """2 rows (val/test) x 3 cols (P/R/F1 lines, FAR line, Pareto)."""
    def _subset(partition, order):
        sub = [r for r in rows if r["partition"] == partition and r["order"] == order]
        sub.sort(key=lambda r: r["min_event_sec"])
        return sub

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for row_i, partition in enumerate(PARTITIONS):
        ax = axes[row_i, 0]
        for order in ORDERINGS:
            sub = _subset(partition, order)
            if not sub:
                continue
            secs = [r["min_event_sec"]   for r in sub]
            prec = [r["event_precision"] for r in sub]
            rec  = [r["event_recall"]    for r in sub]
            f1v  = [r["event_f1"]        for r in sub]
            ls = ORDER_LINESTYLE[order]; mk = ORDER_MARKER[order]
            lbl = ORDER_LABEL[order]
            ax.plot(secs, prec, linestyle=ls, marker=mk, color="#5A7DC8",
                    linewidth=1.6, label=f"Precision ({lbl})")
            ax.plot(secs, rec,  linestyle=ls, marker=mk, color="#5AC880",
                    linewidth=1.6, label=f"Recall ({lbl})")
            ax.plot(secs, f1v,  linestyle=ls, marker=mk, color="#C8A05A",
                    linewidth=1.6, label=f"F1 ({lbl})")
        ax.set_xlabel("MIN_EVENT_SEC (s)"); ax.set_ylabel("Score")
        ax.set_title(f"[{partition.upper()}] Event Precision / Recall / F1")
        ax.set_xticks(SWEEP_SECS); ax.grid(True, alpha=0.3)
        # Legend BELOW the axes so the 6 entries never overlap with the
        # P/R/F1 curves (loc="best" picked the data-covered lower-left
        # corner for this dataset).
        ax.legend(fontsize=7, ncol=3, loc="upper center",
                  bbox_to_anchor=(0.5, -0.18), frameon=True)

        ax = axes[row_i, 1]
        for order in ORDERINGS:
            sub = _subset(partition, order)
            if not sub:
                continue
            secs = [r["min_event_sec"]      for r in sub]
            fars = [r["event_far_per_hour"] for r in sub]
            ax.plot(secs, fars,
                    linestyle=ORDER_LINESTYLE[order],
                    marker=ORDER_MARKER[order],
                    color=ORDER_COLOR[order],
                    linewidth=1.6,
                    label=ORDER_LABEL[order])
        ax.set_xlabel("MIN_EVENT_SEC (s)"); ax.set_ylabel("Event-level FAR/hr")
        ax.set_title(f"[{partition.upper()}] Event-level FAR/hr (corrected)")
        ax.set_xticks(SWEEP_SECS); ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

        ax = axes[row_i, 2]
        for order in ORDERINGS:
            sub = _subset(partition, order)
            if not sub:
                continue
            recs = [r["event_recall"]       for r in sub]
            fars = [r["event_far_per_hour"] for r in sub]
            secs = [r["min_event_sec"]      for r in sub]
            ax.scatter(recs, fars, s=80, color=ORDER_COLOR[order],
                       marker=ORDER_MARKER[order], zorder=3,
                       label=ORDER_LABEL[order])
            for s, x_, y_ in zip(secs, recs, fars):
                ax.annotate(f"{s}s", xy=(x_, y_), xytext=(6, 4),
                            textcoords="offset points", fontsize=9,
                            color=ORDER_COLOR[order])
        ax.set_xlabel("Event recall (sensitivity)")
        ax.set_ylabel("Event-level FAR/hr")
        ax.set_title(f"[{partition.upper()}] Recall vs FAR/hr Pareto\n(bottom-right = ideal)")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="best")

    plt.suptitle("Impact of post-processing order x MIN_EVENT_SEC on %s "
                 "-- val (top) vs test (bottom)" % model_label,
                 fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    # Add vertical breathing room between the two rows so the [VAL] P/R/F1
    # legend (sitting below its axes) doesn't crowd the [TEST] row's titles.
    fig.subplots_adjust(hspace=0.55)
    # bbox_inches="tight" expands the saved figure to fit the below-axis
    # legend on the bottom row (which would otherwise sit outside the
    # default canvas).
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved impact figure: %s", out_path)


def sweep_partition(partition, npz_path, annot_dir, mouse_metadata,
                    output_root, model_label, logger):
    """Run the 2-order x 4-sec sweep for one partition. Returns list of
    comparison-row dicts (one per (order, sec))."""
    if not npz_path.exists():
        logger.warning("[%s] NPZ missing at %s -- skipping this partition.",
                       partition, npz_path)
        return []
    if not annot_dir.exists():
        logger.warning("[%s] Annotations dir missing at %s -- skipping this partition.",
                       partition, annot_dir)
        return []

    npz = np.load(npz_path, allow_pickle=True)
    try:
        y_true_kept, y_prob_kept, records = records_from_npz(npz, logger)
    except KeyError as exc:
        logger.error("[%s] %s", partition, exc)
        return []
    logger.info("[%s] NPZ loaded: %d segments | %d positive (%.3f%%)",
                partition, len(y_true_kept), int(y_true_kept.sum()),
                100.0 * float(y_true_kept.sum()) / max(1, len(y_true_kept)))

    rows = []
    for order in ORDERINGS:
        for sec in SWEEP_SECS:
            logger.info("-" * 65)
            logger.info("[%s] order=%s | MIN_EVENT_SEC=%d s", partition, order, sec)
            eval_utils.MIN_EVENT_SEC = float(sec)
            result = evaluate_event_level(
                records, y_true_kept, y_prob_kept,
                annot_dir, mouse_metadata, logger, order=order)

            out_dir = output_root / partition / order / f"MIN_EVENT_SEC_{sec}s"
            out_dir.mkdir(parents=True, exist_ok=True)

            write_summary_json(out_dir / f"{partition}_summary.json",
                               model_label, partition, sec, order,
                               result, logger)
            write_event_details_csv(out_dir / f"{partition}_event_details.csv",
                                    partition, result, logger)
            write_classification_reports(out_dir, partition, result, logger)

            em = result["event_level_metrics"]
            seg_r2 = result["segment_level_metrics"]["row2_postproc_threshold_0_5"]
            rows.append({
                "partition":              partition,
                "order":                  order,
                "min_event_sec":          sec,
                "n_predicted_events":     result["totals"]["n_predicted_events"],
                "tp":                     em["tp"],
                "fp":                     em["fp"],
                "fn":                     em["fn"],
                "event_precision":        em["precision"],
                "event_recall":           em["recall"],
                "event_f1":               em["f1"],
                "event_far_per_hour":     em["far_per_hour_event_CORRECTED"],
                "event_mean_latency_sec": em["mean_detection_latency_sec"],
                "seg_row2_f1_macro":      seg_r2["f1_macro"],
                "seg_row2_auroc":         seg_r2["auroc"],
                "seg_row2_precision":     seg_r2["precision"],
                "seg_row2_recall":        seg_r2["recall"],
                "seg_row2_specificity":   seg_r2["specificity"],
                "seg_row2_far_per_hour":  seg_r2["far_per_hour_seg_CORRECTED_2_5s_denom"],
                "seg_row2_mcc":           seg_r2.get("mcc"),
            })
            logger.info("      TP=%d FP=%d FN=%d | Prec=%.4f Rec=%.4f F1=%.4f | FAR/hr=%.4f",
                        em["tp"], em["fp"], em["fn"],
                        em["precision"], em["recall"], em["f1"],
                        em["far_per_hour_event_CORRECTED"])
    return rows


def main():
    args   = parse_args()
    paths  = resolve_paths(args)
    logger, log_path = setup_logging(paths["output"])

    output_root    = paths["output"]
    comparison_csv = output_root / "comparison_val_vs_test.csv"
    impact_plot    = output_root / "impact_val_vs_test.png"

    logger.info("=" * 65)
    logger.info("postproc_sweep.py")
    logger.info("Variant         : %s (%s)", args.variant, paths["model_label"])
    logger.info("Mode            : %s", "cluster" if args.cluster else "local")
    logger.info("Val NPZ         : %s", paths["val_npz"])
    logger.info("Test NPZ        : %s", paths["test_npz"])
    logger.info("Metadata        : %s", paths["metadata"])
    logger.info("Val annot dir   : %s", paths["val_annot_dir"])
    logger.info("Test annot dir  : %s", paths["test_annot_dir"])
    logger.info("Output root     : %s", output_root)
    logger.info("Log             : %s", log_path)
    logger.info("Orderings       : %s", ORDERINGS)
    logger.info("Sweep secs      : %s", SWEEP_SECS)
    logger.info("=" * 65)

    if not paths["metadata"].exists():
        logger.error("Required input missing: %s", paths["metadata"]); sys.exit(1)
    mouse_metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))

    npz_map        = {"val": paths["val_npz"],       "test": paths["test_npz"]}
    annot_dir_map  = {"val": paths["val_annot_dir"], "test": paths["test_annot_dir"]}

    all_rows = []
    for partition in PARTITIONS:
        logger.info("#" * 65)
        logger.info("PARTITION: %s", partition.upper())
        logger.info("#" * 65)
        rows = sweep_partition(
            partition, npz_map[partition], annot_dir_map[partition],
            mouse_metadata, output_root, paths["model_label"], logger)
        all_rows.extend(rows)

    if not all_rows:
        logger.error("No sweep rows produced -- comparison CSV / plot will not be written.")
        sys.exit(1)

    write_comparison_csv(comparison_csv, all_rows, logger)
    plot_impact(all_rows, impact_plot, paths["model_label"], logger)

    logger.info("=" * 65)
    logger.info("DONE")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
