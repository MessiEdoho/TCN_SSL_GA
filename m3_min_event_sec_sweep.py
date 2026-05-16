"""
m3_min_event_sec_sweep.py
=========================
Local sweep over MIN_EVENT_SEC for M3 (MultiScaleTCN) on the test
partition. Reads cached test predictions (no model forward pass, no
GPU required) and recomputes the chronology-aware event-level
evaluation for each candidate value of the post-processing
min-event-duration filter, to quantify the precision / recall /
FAR-per-hour tradeoff of raising the filter.

Smoothing window (W=3), threshold (tau=0.5), refractory period (30 s),
and FAR/hr denominator (step_sec = 2.5 s) are held at their canonical
values; only MIN_EVENT_SEC varies.

Sweep values: MIN_EVENT_SEC in [10, 15, 20, 30] seconds (10 s is the
current production baseline; 15 / 20 / 30 are FP-reduction candidates).

Inputs  (local Windows mirror)
    NPZ        : TCN_UNIQURE_PROJECT/MultiScaleTCN/evaluation/multiscale_tcn_test_predictions_raw.npz
    Manifest   : TCN_UNIQURE_PROJECT/data_splits_nonictal_sampled_filtered.json
    Metadata   : TCN_UNIQURE_PROJECT/mouse_recording_metadata.json
    Annotations: TCN_UNIQURE_PROJECT/test_seizure_annot_updated/

Outputs (under TCN_UNIQURE_PROJECT/MultiScaleTCN/evaluation/post_process_varing_sec/)
    MIN_EVENT_SEC_<sec>s/
        test_summary.json
        test_classification_report_row1.json
        test_classification_report_row2.json
        test_event_details.csv
    test_postproc_min_event_sec_comparison.csv
    test_postproc_min_event_sec_impact.png
    m3_min_event_sec_sweep.log
"""

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
    load_annotations, build_mouse_chronology, evaluate_event_level,
    build_classification_report,
)


LOCAL_ROOT       = Path(r"C:\Users\messi\OneDrive\Desktop\Desktop\TCN_UNIQURE_PROJECT")
NPZ_PATH         = LOCAL_ROOT / "MultiScaleTCN" / "evaluation" / "multiscale_tcn_test_predictions_raw.npz"
MANIFEST_PATH    = LOCAL_ROOT / "data_splits_nonictal_sampled_filtered.json"
METADATA_PATH    = LOCAL_ROOT / "mouse_recording_metadata.json"
ANNOT_DIR        = LOCAL_ROOT / "test_seizure_annot_updated"

OUTPUT_ROOT      = LOCAL_ROOT / "MultiScaleTCN" / "evaluation" / "post_process_varing_sec"
COMPARISON_CSV   = OUTPUT_ROOT / "test_postproc_min_event_sec_comparison.csv"
IMPACT_PLOT      = OUTPUT_ROOT / "test_postproc_min_event_sec_impact.png"
LOG_PATH         = OUTPUT_ROOT / "m3_min_event_sec_sweep.log"

MODEL_NAME       = "MultiScaleTCN"
PARTITION        = "test"
SWEEP_SECS       = [10, 15, 20, 30]


def setup_logging():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("m3_min_event_sec_sweep")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger


def enrich_test_records(records, mouse_metadata, annotations_dir, logger):
    """For each test record, add mouse_id / chrono_idx / t_start_sec by
    replaying per-mouse chronology from EDF metadata + Excel annotations
    (build_mouse_chronology). Records that can't be enriched are dropped.

    Returns (enriched_records, kept_indices). kept_indices is an ordered
    list of original positions in `records` so the caller can trim the
    NPZ arrays in lockstep.
    """
    mice = sorted({Path(r["filepath"]).stem.split("_")[0] for r in records})
    logger.info("Test partition: %d records, %d unique mice", len(records), len(mice))

    chrono_maps = {}
    for mouse_id in mice:
        if mouse_id not in mouse_metadata:
            logger.warning("  %s : no metadata; dropping all its records.", mouse_id)
            chrono_maps[mouse_id] = None
            continue
        xlsx = annotations_dir / f"{mouse_id}_xlsx.xlsx"
        if not xlsx.exists():
            logger.warning("  %s : no annotation at %s; dropping all its records.",
                           mouse_id, xlsx)
            chrono_maps[mouse_id] = None
            continue
        meta = mouse_metadata[mouse_id]
        rec_start = datetime.datetime.fromisoformat(meta["recording_start_dt"])
        seizure_intervals = load_annotations(xlsx, rec_start)
        cmap, _, _ = build_mouse_chronology(
            mouse_id, int(meta["n_samples"]), seizure_intervals, logger)
        chrono_maps[mouse_id] = cmap

    enriched, kept = [], []
    for npz_idx, r in enumerate(records):
        fname = Path(r["filepath"]).name
        mouse = fname.split("_")[0]
        cmap = chrono_maps.get(mouse)
        if cmap is None:
            continue
        info = cmap.get(fname)
        if info is None:
            continue
        r2 = dict(r)
        r2["mouse_id"]    = mouse
        r2["chrono_idx"]  = info["chrono_idx"]
        r2["t_start_sec"] = info["t_start_sec"]
        enriched.append(r2)
        kept.append(npz_idx)

    logger.info("Enriched: %d retained | %d dropped (no chronology)",
                len(enriched), len(records) - len(enriched))
    return enriched, kept


def write_summary_json(out_path, model, partition, sec,
                       manifest_filter_state, result, logger):
    payload = {
        "model":     model,
        "partition": partition,
        "timestamp": datetime.datetime.now().isoformat(),
        "post_processing_params": {
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
    if manifest_filter_state is not None:
        payload["manifest_filter_history"] = manifest_filter_state
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str),
                        encoding="utf-8")
    logger.info("    summary -> %s", out_path)


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
    logger.info("    event details -> %s (%d rows)",
                out_path, len(result["all_event_details"]))


def write_classification_reports(out_dir, result, logger):
    seg = result["segment_level_metrics"]
    reord = result["reordered_arrays"]
    y_true = reord["y_true"]
    r1 = build_classification_report(y_true, reord["y_pred_row1"],
                                     seg["row1_raw_threshold_0_5"])
    r2 = build_classification_report(y_true, reord["y_pred_row2"],
                                     seg["row2_postproc_threshold_0_5"])
    p1 = out_dir / "test_classification_report_row1.json"
    p2 = out_dir / "test_classification_report_row2.json"
    p1.write_text(json.dumps(r1, indent=2, sort_keys=True, default=str), encoding="utf-8")
    p2.write_text(json.dumps(r2, indent=2, sort_keys=True, default=str), encoding="utf-8")
    logger.info("    row1 report -> %s", p1)
    logger.info("    row2 report -> %s", p2)


def write_comparison_csv(out_path, rows, logger):
    fieldnames = [
        "min_event_sec",
        "n_predicted_events", "tp", "fp", "fn",
        "event_precision", "event_recall", "event_f1",
        "event_far_per_hour", "event_mean_latency_sec",
        "seg_row2_f1_macro", "seg_row2_auroc", "seg_row2_precision",
        "seg_row2_recall", "seg_row2_specificity", "seg_row2_far_per_hour",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    logger.info("Saved comparison CSV: %s", out_path)


def plot_impact(rows, out_path, logger):
    """2x2 panel.
      (0,0) Event Precision / Recall / F1 vs MIN_EVENT_SEC (line+marker)
      (0,1) Event TP / FP / FN grouped bars
      (1,0) Event-level FAR/hr vs MIN_EVENT_SEC (line+marker)
      (1,1) Recall vs FAR/hr Pareto scatter (each point labelled with sec)
    """
    secs   = [r["min_event_sec"]      for r in rows]
    prec   = [r["event_precision"]    for r in rows]
    rec    = [r["event_recall"]       for r in rows]
    f1     = [r["event_f1"]           for r in rows]
    tps    = [r["tp"]                 for r in rows]
    fps    = [r["fp"]                 for r in rows]
    fns    = [r["fn"]                 for r in rows]
    fars   = [r["event_far_per_hour"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(secs, prec, "o-", color="#5A7DC8", linewidth=1.6, label="Precision")
    ax.plot(secs, rec,  "s-", color="#5AC880", linewidth=1.6, label="Recall")
    ax.plot(secs, f1,   "^-", color="#C8A05A", linewidth=1.6, label="F1")
    ax.set_xlabel("MIN_EVENT_SEC (s)"); ax.set_ylabel("Score")
    ax.set_title("Event-level Precision / Recall / F1")
    ax.set_xticks(secs); ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    ax = axes[0, 1]
    x = np.arange(len(secs)); w = 0.27
    ax.bar(x - w, tps, width=w, color="#5AC880", label="TP")
    ax.bar(x,     fps, width=w, color="#C85A5A", label="FP")
    ax.bar(x + w, fns, width=w, color="#5A7DC8", label="FN")
    ax.set_xticks(x); ax.set_xticklabels([str(s) for s in secs])
    ax.set_xlabel("MIN_EVENT_SEC (s)"); ax.set_ylabel("Event count")
    ax.set_title("Event-level TP / FP / FN counts")
    ax.grid(True, alpha=0.3, axis="y"); ax.legend(fontsize=9)

    ax = axes[1, 0]
    ax.plot(secs, fars, "o-", color="#C85A5A", linewidth=1.6)
    ax.set_xlabel("MIN_EVENT_SEC (s)"); ax.set_ylabel("Event-level FAR/hr (corrected)")
    ax.set_title("Event-level FAR/hr")
    ax.set_xticks(secs); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.scatter(rec, fars, s=90, color="#5A7DC8", zorder=3)
    for s, x_, y_ in zip(secs, rec, fars):
        ax.annotate(f"{s}s", xy=(x_, y_), xytext=(6, 4),
                    textcoords="offset points", fontsize=10)
    ax.set_xlabel("Event recall (sensitivity)")
    ax.set_ylabel("Event-level FAR/hr")
    ax.set_title("Recall vs FAR/hr Pareto\n(label = MIN_EVENT_SEC, bottom-right = ideal)")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Impact of MIN_EVENT_SEC on M3 (MultiScaleTCN) test partition",
                 fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved impact figure: %s", out_path)


def main():
    logger = setup_logging()
    logger.info("=" * 65)
    logger.info("m3_min_event_sec_sweep.py")
    logger.info("Local root      : %s", LOCAL_ROOT)
    logger.info("NPZ             : %s", NPZ_PATH)
    logger.info("Manifest        : %s", MANIFEST_PATH)
    logger.info("Metadata        : %s", METADATA_PATH)
    logger.info("Annotations dir : %s", ANNOT_DIR)
    logger.info("Output root     : %s", OUTPUT_ROOT)
    logger.info("Log             : %s", LOG_PATH)
    logger.info("Sweep values    : %s", SWEEP_SECS)
    logger.info("=" * 65)

    for p in [NPZ_PATH, MANIFEST_PATH, METADATA_PATH, ANNOT_DIR]:
        if not p.exists():
            logger.error("Required input missing: %s", p); sys.exit(1)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    test_records = manifest.get(PARTITION) or []
    if not test_records:
        logger.error("Empty test partition in %s.", MANIFEST_PATH); sys.exit(1)
    manifest_filter_state = (manifest.get("meta", {}) or {}).get("filter_history", None)

    mouse_metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    npz = np.load(NPZ_PATH, allow_pickle=False)
    y_true_all = npz["y_true"].astype(np.int64)
    y_prob_all = npz["y_prob"].astype(np.float64)
    if len(y_true_all) != len(test_records):
        logger.error("NPZ length %d != manifest test length %d",
                     len(y_true_all), len(test_records)); sys.exit(1)
    logger.info("NPZ loaded: %d segments | %d positive (%.3f%%)",
                len(y_true_all), int(y_true_all.sum()),
                100.0 * float(y_true_all.sum()) / max(1, len(y_true_all)))

    logger.info("Building per-mouse chronologies and enriching records (one-time)...")
    enriched, kept_idx = enrich_test_records(
        test_records, mouse_metadata, ANNOT_DIR, logger)
    y_true_kept = y_true_all[kept_idx]
    y_prob_kept = y_prob_all[kept_idx]

    comparison_rows = []
    for sec in SWEEP_SECS:
        logger.info("-" * 65)
        logger.info("Sweep value: MIN_EVENT_SEC = %d s", sec)
        eval_utils.MIN_EVENT_SEC = float(sec)
        result = evaluate_event_level(
            enriched, y_true_kept, y_prob_kept, ANNOT_DIR, mouse_metadata, logger)

        out_dir = OUTPUT_ROOT / f"MIN_EVENT_SEC_{sec}s"
        out_dir.mkdir(parents=True, exist_ok=True)

        write_summary_json(out_dir / "test_summary.json",
                           MODEL_NAME, PARTITION, sec, manifest_filter_state,
                           result, logger)
        write_event_details_csv(out_dir / "test_event_details.csv",
                                PARTITION, result, logger)
        write_classification_reports(out_dir, result, logger)

        em = result["event_level_metrics"]
        seg_r2 = result["segment_level_metrics"]["row2_postproc_threshold_0_5"]
        comparison_rows.append({
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
        })
        logger.info("    TP=%d FP=%d FN=%d | Prec=%.4f Rec=%.4f F1=%.4f | FAR/hr=%.4f",
                    em["tp"], em["fp"], em["fn"],
                    em["precision"], em["recall"], em["f1"],
                    em["far_per_hour_event_CORRECTED"])

    write_comparison_csv(COMPARISON_CSV, comparison_rows, logger)
    plot_impact(comparison_rows, IMPACT_PLOT, logger)

    logger.info("=" * 65)
    logger.info("DONE")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
