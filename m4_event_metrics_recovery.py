"""
m4_event_metrics_recovery.py
============================
One-off recovery pipeline for M4 (MultiScaleTCNAttention). The M4 final
training run completed successfully (best F1 = 0.7629 at epoch 21, gate
fired at epoch 40) and weights were saved to
    OUTPUT_ROOT / "ms_attn_final_weights.pt"
but the post-training full-val pass crashed at AUROC with
ValueError: Input contains NaN. Root cause: FP16 attention softmax
overflow (exp(+inf)/sum(exp(...)) = inf/inf) -- a different mechanism
from the M3 input-overflow NaN, but the same symptom.

This script repeats the post-training full-val pass in FP32 (eval_utils
four-layer protection) and emits every val artefact that the training
script would have written, but routed to a dedicated subfolder so the
parent MODEL4 directory is preserved as-is:
    OUTPUT_ROOT / "event_metrics" / ...

Schema follows the new style: Row 1 (raw @ 0.5) + Row 2 (post-processed
@ 0.5) only -- Row 3 retired. FAR/hr reported in the corrected form
(denominator x step_sec=2.5) only -- the BROKEN_5s_denom column has been
dropped repo-wide per the migration decision.

The same FP32-final-eval fix has been applied to all four training
scripts (TCN.py, TCNTemporalAttention.py, MultiScaleTCN.py,
MultiScaleTCNAttention.py) so future training runs will not re-crash.
"""

import csv
import datetime
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

from MultiScaleTCNAttention import (
    OUTPUT_ROOT, WEIGHTS_PATH, SPLITS_PATH, ANNOT_DIR, MOUSE_METADATA_PATH,
    BACKBONE_PARAMS_PATH, ATTN_PARAMS_PATH, MODEL_NAME,
    SEGMENT_SEC, STEP_SEC, SMOOTHING_WIN, REFRACTORY_SEC, MIN_EVENT_SEC,
    SEED,
    set_seed, load_best_params, load_splits, build_model, count_parameters,
)
from eval_utils import (
    AMPLITUDE_THRESHOLD, SafeEEGSegmentDataset, make_safe_loader,
    evaluate_model_safe, verify_manifest_filtered,
    evaluate_event_level, build_classification_report,
)


EVENT_METRICS_DIR = OUTPUT_ROOT / "event_metrics"
LOG_DIR           = EVENT_METRICS_DIR / "logs"
LOG_PATH          = LOG_DIR / "m4_event_metrics_recovery.log"

VAL_PREDICTIONS_NPZ = EVENT_METRICS_DIR / "ms_attn_val_predictions_full.npz"
EVAL_REPORT_PATH    = EVENT_METRICS_DIR / "ms_attn_evaluation_report.json"
EVENT_DETAILS_CSV   = EVENT_METRICS_DIR / "ms_attn_event_details_row2.csv"
TWO_ROW_CSV         = EVENT_METRICS_DIR / "ms_attn_two_row_summary.csv"
ROW1_REPORT_PATH    = EVENT_METRICS_DIR / "ms_attn_classification_report_row1.json"
ROW2_REPORT_PATH    = EVENT_METRICS_DIR / "ms_attn_classification_report_row2.json"


def setup_logging():
    EVENT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("m4_event_metrics_recovery")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger


def main():
    logger = setup_logging()
    logger.info("=" * 65)
    logger.info("m4_event_metrics_recovery.py")
    logger.info("Timestamp        : %s", datetime.datetime.now().isoformat())
    logger.info("Weights          : %s", WEIGHTS_PATH)
    logger.info("Splits manifest  : %s", SPLITS_PATH)
    logger.info("Annotations dir  : %s", ANNOT_DIR)
    logger.info("Mouse metadata   : %s", MOUSE_METADATA_PATH)
    logger.info("Output dir       : %s", EVENT_METRICS_DIR)
    logger.info("Log              : %s", LOG_PATH)
    logger.info("=" * 65)

    set_seed(SEED)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logger.info("GPU  : %s", torch.cuda.get_device_name(0))
        logger.info("VRAM : %.2f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)
        logger.info("CUDA : %s", torch.version.cuda)
        try:
            _free_bytes, _total_bytes = torch.cuda.mem_get_info(0)
            _free_gb = _free_bytes / 1e9
            _total_gb = _total_bytes / 1e9
            logger.info("GPU memory free: %.2f / %.2f GB", _free_gb, _total_gb)
            if _free_gb < 8.0:
                logger.warning("GPU has only %.2f GB free (< 8 GB threshold). "
                               "Another process may be sharing this GPU, or VRAM "
                               "is fragmented. Inference may fail with CUDA OOM.",
                               _free_gb)
        except Exception as _e:
            logger.warning("Could not query GPU memory: %s", _e)
    else:
        logger.info("Device: CPU")
    logger.info("PyTorch: %s", torch.__version__)

    # Layer 1: refuse to run on an unfiltered manifest.
    manifest_filter = verify_manifest_filtered(SPLITS_PATH, partition_key="val", logger=logger)

    # Build model + load M4 final weights.
    backbone_config, backbone_hp, branch_dilations, attn_config, attn_hp = load_best_params(logger)
    model = build_model(backbone_hp, branch_dilations, attn_hp, DEVICE, logger)
    n_params = count_parameters(model)

    if not WEIGHTS_PATH.exists():
        logger.error("Final weights not found at %s. Train M4 first.", WEIGHTS_PATH)
        sys.exit(1)
    state = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    logger.info("Loaded M4 weights: %s (%.2f MB) | n_params=%d",
                WEIGHTS_PATH, WEIGHTS_PATH.stat().st_size / 1e6, n_params)

    # Val pairs + enriched val records (chrono_idx / t_start_sec per record).
    _train_pairs, val_pairs, val_records = load_splits(logger)
    del _train_pairs

    batch_size = int(attn_hp["batch_size"])
    SafeEEGSegmentDataset.n_sanitised = 0
    val_loader = make_safe_loader(val_pairs, batch_size, DEVICE)
    logger.info("Full val loader  : %d batches | batch_size=%d", len(val_loader), batch_size)

    # Layer 3 + Layer 4: FP32 forward with per-batch finiteness assertion.
    logger.info("Running full-val FP32 forward pass (eval_utils.evaluate_model_safe)...")
    t0 = time.time()
    val_f1_final, y_true, _y_pred_05, y_prob = evaluate_model_safe(
        model, val_loader, DEVICE, logger, use_amp=False)
    logger.info("Full-val F1 (raw @ 0.5): %.4f | wall time: %.0f s | sanitised: %d",
                val_f1_final, time.time() - t0, SafeEEGSegmentDataset.n_sanitised)

    # Chronology-aware per-mouse event-level evaluation.
    logger.info("Running per-mouse-chronological event-level evaluation "
                "(eval_utils.evaluate_event_level)...")
    if not MOUSE_METADATA_PATH.exists():
        logger.error("Mouse metadata not found at %s.", MOUSE_METADATA_PATH); sys.exit(1)
    mouse_metadata = json.loads(MOUSE_METADATA_PATH.read_text(encoding="utf-8"))
    val_eval_result = evaluate_event_level(
        val_records, y_true, y_prob, ANNOT_DIR, mouse_metadata, logger)

    reord = val_eval_result["reordered_arrays"]
    seg   = val_eval_result["segment_level_metrics"]
    em    = val_eval_result["event_level_metrics"]

    # Enriched val NPZ in chronological order.
    np.savez_compressed(
        VAL_PREDICTIONS_NPZ,
        y_true=reord["y_true"].astype(np.int8),
        y_prob=reord["y_prob"].astype(np.float32),
        y_pred_row1=reord["y_pred_row1"].astype(np.int8),
        y_pred_row2=reord["y_pred_row2"].astype(np.int8),
        mouse_id=reord["mouse_id"],
        chrono_idx=reord["chrono_idx"],
        t_start_sec=reord["t_start_sec"],
        n_segments=np.int64(len(reord["y_true"])),
        segment_sec=np.float32(SEGMENT_SEC),
        smoothing_win=np.int64(SMOOTHING_WIN),
        refractory_sec=np.float32(REFRACTORY_SEC),
        min_event_sec=np.float32(MIN_EVENT_SEC),
    )
    logger.info("Saved enriched val predictions bundle: %s (%.2f MB)",
                VAL_PREDICTIONS_NPZ, VAL_PREDICTIONS_NPZ.stat().st_size / 1e6)

    # Evaluation report JSON (corrected schema).
    corrected_report = {
        "model":     MODEL_NAME,
        "timestamp": datetime.datetime.now().isoformat(),
        "weights_path":   str(WEIGHTS_PATH),
        "evaluation_set": "validation",
        "recovery_source": "m4_event_metrics_recovery.py",
        "manifest_filter":         manifest_filter,
        "backbone_hyperparameters": backbone_hp,
        "branch_dilations":         branch_dilations,
        "attention_hyperparameters": attn_hp,
        "backbone_params_source":   str(BACKBONE_PARAMS_PATH),
        "attention_params_source":  str(ATTN_PARAMS_PATH),
        "post_processing_params": {
            "smoothing_window":       SMOOTHING_WIN,
            "refractory_period_sec":  REFRACTORY_SEC,
            "min_event_duration_sec": MIN_EVENT_SEC,
            "step_sec":               STEP_SEC,
            "matching_rule":          "any-overlap",
        },
        "totals":                val_eval_result["totals"],
        "segment_level_metrics": seg,
        "event_level_metrics":   em,
        "per_mouse":             val_eval_result["per_mouse_results"],
    }
    with open(EVAL_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(corrected_report, f, indent=2, default=str)
    logger.info("Saved evaluation report: %s", EVAL_REPORT_PATH)

    # Per-event details CSV (Row 2 post-processed events vs GT).
    fieldnames_evt = ["mouse_id", "is_true_alarm", "start_sec", "end_sec",
                      "duration_sec", "max_prob", "matched_gt_idx",
                      "matched_gt_start_sec", "matched_gt_end_sec"]
    with open(EVENT_DETAILS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_evt)
        writer.writeheader()
        for row in val_eval_result["all_event_details"]:
            writer.writerow(row)
    logger.info("Saved event-details CSV: %s (%d rows)",
                EVENT_DETAILS_CSV, len(val_eval_result["all_event_details"]))

    # Two-row summary CSV (Row 3 retired).
    fieldnames_2r = [
        "row", "threshold", "postprocessed",
        "accuracy", "precision", "recall", "specificity", "youden_j",
        "f1_macro", "auroc", "average_precision",
        "far_per_hour_seg", "far_per_hour_event",
        "n_true_alarms", "n_false_alarms", "n_total_events",
        "tp", "tn", "fp", "fn",
    ]
    rows_2r = [
        ("Row1_raw_0.5",      seg["row1_raw_threshold_0_5"],      "N/A"),
        ("Row2_postproc_0.5", seg["row2_postproc_threshold_0_5"], em["far_per_hour_event_CORRECTED"]),
    ]
    with open(TWO_ROW_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_2r)
        writer.writeheader()
        for label, m, far_evt in rows_2r:
            writer.writerow({
                "row":               label,
                "threshold":         m.get("threshold", 0.5),
                "postprocessed":     m.get("postprocessed", False),
                "accuracy":          m.get("accuracy", ""),
                "precision":         m.get("precision", ""),
                "recall":            m.get("recall", ""),
                "specificity":       m.get("specificity", ""),
                "youden_j":          m.get("youden_j", ""),
                "f1_macro":          m.get("f1_macro", ""),
                "auroc":             m.get("auroc", ""),
                "average_precision": m.get("average_precision", ""),
                "far_per_hour_seg":  m.get("far_per_hour_seg_CORRECTED_2_5s_denom", ""),
                "far_per_hour_event": far_evt,
                "n_true_alarms":     em["tp"]            if far_evt != "N/A" else "N/A",
                "n_false_alarms":    em["fp"]            if far_evt != "N/A" else "N/A",
                "n_total_events":    em["tp"] + em["fp"] if far_evt != "N/A" else "N/A",
                "tp": m.get("tp", ""), "tn": m.get("tn", ""),
                "fp": m.get("fp", ""), "fn": m.get("fn", ""),
            })
    logger.info("Saved two-row summary: %s", TWO_ROW_CSV)

    # Per-row sklearn classification reports (chronologically-aligned inputs).
    y_true_aligned = reord["y_true"]
    row1_report = build_classification_report(
        y_true_aligned, reord["y_pred_row1"], seg["row1_raw_threshold_0_5"])
    with open(ROW1_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(row1_report, f, indent=2, default=str)
    logger.info("Saved Row 1 classification report: %s", ROW1_REPORT_PATH)

    row2_report = build_classification_report(
        y_true_aligned, reord["y_pred_row2"], seg["row2_postproc_threshold_0_5"])
    with open(ROW2_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(row2_report, f, indent=2, default=str)
    logger.info("Saved Row 2 classification report: %s", ROW2_REPORT_PATH)

    # Final summary log block.
    logger.info("-" * 65)
    logger.info("M4 RECOVERY SUMMARY")
    logger.info("  N segments (val) : %d", val_eval_result["totals"]["n_segments_in_npz"])
    logger.info("  N mice           : %d", val_eval_result["totals"]["n_mice"])
    logger.info("  N GT seizures    : %d", val_eval_result["totals"]["n_ground_truth_seizures"])
    logger.info("  N pred events    : %d", val_eval_result["totals"]["n_predicted_events"])
    r1 = seg["row1_raw_threshold_0_5"]; r2 = seg["row2_postproc_threshold_0_5"]
    logger.info("  --- Segment Row 1 (raw @ 0.5) ---")
    logger.info("    F1 macro / AUROC / AP : %.4f / %.4f / %.4f",
                r1["f1_macro"], r1["auroc"], r1["average_precision"])
    logger.info("    Precision / Recall / Specificity : %.4f / %.4f / %.4f",
                r1["precision"], r1["recall"], r1["specificity"])
    logger.info("    FAR/hr seg : %.4f", r1["far_per_hour_seg_CORRECTED_2_5s_denom"])
    logger.info("  --- Segment Row 2 (post-processed @ 0.5) ---")
    logger.info("    F1 macro / AUROC / AP : %.4f / %.4f / %.4f",
                r2["f1_macro"], r2["auroc"], r2["average_precision"])
    logger.info("    Precision / Recall / Specificity : %.4f / %.4f / %.4f",
                r2["precision"], r2["recall"], r2["specificity"])
    logger.info("    FAR/hr seg : %.4f", r2["far_per_hour_seg_CORRECTED_2_5s_denom"])
    logger.info("  --- Event level (chronology-aware, any-overlap) ---")
    logger.info("    TP / FP / FN : %d / %d / %d", em["tp"], em["fp"], em["fn"])
    logger.info("    Precision / Recall / F1 : %.4f / %.4f / %.4f",
                em["precision"], em["recall"], em["f1"])
    logger.info("    FAR/hr event : %.4f (non-ictal hours = %.1f)",
                em["far_per_hour_event_CORRECTED"],
                val_eval_result["totals"]["non_ictal_hours_corrected"])
    logger.info("    Mean detection latency: %s s", em["mean_detection_latency_sec"])
    logger.info("-" * 65)
    logger.info("DONE")


if __name__ == "__main__":
    main()
