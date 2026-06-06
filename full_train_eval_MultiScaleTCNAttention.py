"""
full_train_eval_MultiScaleTCNAttention.py
=========================================
Final evaluation of the trained Multi-Scale TCN + Temporal Attention (M4)
model on the FULL UN-DOWNSAMPLED TRAIN partition (~28.7M segments at
~0.23% natural ictal prevalence). Mirrors
MultiScaleTCNAttention_evaluation.py in metrics, post-processing,
figures, and JSON/CSV outputs, so the full-train numbers are produced
through exactly the same computations as the test-set numbers reported
by the test-eval script.

Pipeline position
-----------------
After  : MultiScaleTCNAttention.py             (training; produces final weights)
After  : build_chronology.py --include-train   (per-mouse train chronology)
After  : enrich_manifest.py --include-train    (data_splits_full_train_enriched.json)
This   : full_train_eval_MultiScaleTCNAttention.py (full un-downsampled train eval)

Differences vs MultiScaleTCNAttention_evaluation.py
---------------------------------------------------
  * Manifest    : data_splits_full_train_enriched.json (un-downsampled)
                  rather than data_splits_nonictal_sampled_filtered_enriched.json
  * Partition   : train (rather than test)
  * Layer 1     : SKIPPED. The train partition was never subject to
                  apply_val_test_filter (that step is val/test-only). Layers
                  2-4 (SafeEEGSegmentDataset / FP32 / per-batch isfinite
                  assert) still apply.
  * Output root : OUTPUT_ROOT/full_train_evaluation/ (sibling of evaluation/)
  * File prefix : ms_attn_full_train_* (rather than ms_attn_*)
  * Attention saliency step is OMITTED here -- the attention-saliency
    figure has its own dedicated tool (event_attention_saliency.py) that
    runs only on the event-containing subset and is the correct way to
    produce that figure.

Inputs
------
  OUTPUT_ROOT / ms_attn_final_weights.pt
  best_multiscale_params.json + best_multiscale_attn_params.json
  data_splits_full_train_enriched.json
  mouse_recording_metadata.json
  /home/people/22206468/scratch/seizure_times_updated/{mouse}_xlsx.xlsx

Outputs (under OUTPUT_ROOT / full_train_evaluation/)
----------------------------------------------------
  ms_attn_full_train_evaluation_report.json
  ms_attn_full_train_three_row_summary.csv
  ms_attn_full_train_classification_report_row{1,2}.json
  ms_attn_full_train_event_details_row2.csv
  ms_attn_full_train_predictions_full.npz
  figures/                                           (Row 1 / Row 2 figures)
  Result_classReport/ms_attn_full_train_classreport_barplot.png
  ../logs/ms_attn_full_train_evaluation.log

Usage
-----
python full_train_eval_MultiScaleTCNAttention.py
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

import matplotlib
matplotlib.use("Agg")

from MultiScaleTCNAttention import (
    SEED, MODEL_NAME, OUTPUT_ROOT,
    WEIGHTS_PATH, ANNOT_DIR, MOUSE_METADATA_PATH,
    BACKBONE_PARAMS_PATH, ATTN_PARAMS_PATH,
    SEGMENT_SEC,
    MIN_EVENT_SEC, REFRACTORY_SEC, SMOOTHING_WIN,
    load_best_params, build_model,
    run_postprocessing_evaluations, save_all_results, plot_all_figures,
)

from eval_utils import (
    SafeEEGSegmentDataset,
    AMPLITUDE_THRESHOLD,
    make_safe_loader,
    evaluate_model_safe,
    evaluate_event_level,
    write_event_level_bundle,
    THRESHOLD,
    STEP_SEC,
)

from tcn_utils import (
    set_seed,
    count_parameters,
    make_classreport_barplot,
)


USE_AMP_FOR_EVAL    = False
NUM_DATA_WORKERS    = 4

FULL_TRAIN_SPLITS_PATH = Path(
    "/scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits_full_train_enriched.json"
)

FULL_TRAIN_EVAL_DIR              = OUTPUT_ROOT / "full_train_evaluation"
FULL_TRAIN_EVAL_FIGURE_DIR       = FULL_TRAIN_EVAL_DIR / "figures"
FULL_TRAIN_EVAL_CLASSREPORT_DIR  = FULL_TRAIN_EVAL_DIR / "Result_classReport"
FULL_TRAIN_EVAL_LOG_DIR          = OUTPUT_ROOT / "logs"
FULL_TRAIN_EVAL_LOG_PATH         = FULL_TRAIN_EVAL_LOG_DIR / "ms_attn_full_train_evaluation.log"

FULL_TRAIN_PREDICTIONS_NPZ       = FULL_TRAIN_EVAL_DIR / "ms_attn_full_train_predictions_full.npz"


def setup_logging():
    FULL_TRAIN_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    FULL_TRAIN_EVAL_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    FULL_TRAIN_EVAL_CLASSREPORT_DIR.mkdir(parents=True, exist_ok=True)
    FULL_TRAIN_EVAL_LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ms_attn_full_train_evaluation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(FULL_TRAIN_EVAL_LOG_PATH, mode="a", encoding="utf-8")
    fh.setFormatter(fmt); logger.addHandler(fh)
    return logger


def _redirect_outputs_to_full_train_eval(logger):
    """Rebind path globals on MultiScaleTCNAttention so save_all_results /
    plot_all_figures write into FULL_TRAIN_EVAL_DIR.
    """
    import MultiScaleTCNAttention as _m4

    redirections = {
        "OUTPUT_ROOT":      FULL_TRAIN_EVAL_DIR,
        "FIGURE_DIR":       FULL_TRAIN_EVAL_FIGURE_DIR,
        "LOG_DIR":          FULL_TRAIN_EVAL_LOG_DIR,
        "EVAL_REPORT_PATH": FULL_TRAIN_EVAL_DIR / "ms_attn_full_train_evaluation_report.json",
        "THREE_ROW_CSV":    FULL_TRAIN_EVAL_DIR / "ms_attn_full_train_three_row_summary.csv",
    }
    logger.info("Redirecting MultiScaleTCNAttention write targets to FULL_TRAIN_EVAL_DIR:")
    for name, new_path in redirections.items():
        original = getattr(_m4, name, None)
        setattr(_m4, name, new_path)
        logger.info("  %-18s : %s -> %s", name, original, new_path)


def load_full_train_split(logger):
    """Load TRAIN file-label pairs from the full-train enriched manifest."""
    if not FULL_TRAIN_SPLITS_PATH.exists():
        logger.error(
            "Full-train enriched manifest not found at %s. "
            "Run build_chronology.py --include-train and enrich_manifest.py "
            "--include-train first.", FULL_TRAIN_SPLITS_PATH)
        raise FileNotFoundError(str(FULL_TRAIN_SPLITS_PATH))

    logger.info("Loading full-train splits from: %s", FULL_TRAIN_SPLITS_PATH)
    with open(FULL_TRAIN_SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)

    if "train" not in splits or not splits["train"]:
        logger.error("Train partition is empty in manifest %s", FULL_TRAIN_SPLITS_PATH)
        raise RuntimeError("Empty train partition")

    train_pairs   = [(rec["filepath"], rec["label"]) for rec in splits["train"]]
    train_records = list(splits["train"])

    n_total = len(train_pairs)
    n_sz = sum(1 for _, l in train_pairs if l == 1)
    n_nsz = n_total - n_sz
    pct = n_sz / n_total * 100 if n_total > 0 else 0.0
    mouse_ids = sorted({Path(fp).stem.split("_")[0] for fp, _ in train_pairs})
    logger.info("FULL TRAIN: %d total | %d seizure | %d non-seizure | %.4f%% ictal | %d mice",
                n_total, n_sz, n_nsz, pct, len(mouse_ids))

    if not all("chrono_idx" in r for r in train_records[:5]):
        logger.error("Train records missing 'chrono_idx' -- manifest not enriched? "
                     "Expected enriched manifest; got: %s", FULL_TRAIN_SPLITS_PATH)
        raise RuntimeError("Manifest not enriched with chronology")

    return train_pairs, train_records


def main():
    logger = setup_logging()

    logger.info("=" * 65)
    logger.info("full_train_eval_MultiScaleTCNAttention.py")
    logger.info("Timestamp     : %s", datetime.datetime.now().isoformat())
    logger.info("Purpose       : Evaluate M4 on the FULL un-downsampled TRAIN set")
    logger.info("Model         : %s", MODEL_NAME)
    logger.info("Weights       : %s", WEIGHTS_PATH)
    logger.info("Read root     : %s (training artefacts; never overwritten)", OUTPUT_ROOT)
    logger.info("Write root    : %s", FULL_TRAIN_EVAL_DIR)
    logger.info("Manifest      : %s", FULL_TRAIN_SPLITS_PATH)
    logger.info("Eval log path : %s", FULL_TRAIN_EVAL_LOG_PATH)
    logger.info("Mode          : Row 1 + Row 2; FP32 forward; Layers 2-4 NaN protection "
                "(Layer 1 skipped -- train was never apply_val_test_filter'd)")
    logger.info("=" * 65)

    _redirect_outputs_to_full_train_eval(logger)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device        : %s", device)
    if torch.cuda.is_available():
        logger.info("GPU  : %s", torch.cuda.get_device_name(0))
        logger.info("VRAM : %.2f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)
        logger.info("CUDA : %s", torch.version.cuda)
        try:
            _free_bytes, _total_bytes = torch.cuda.mem_get_info(0)
            logger.info("GPU memory free: %.2f / %.2f GB", _free_bytes / 1e9, _total_bytes / 1e9)
        except Exception as _e:
            logger.warning("Could not query GPU memory: %s", _e)
    logger.info("PyTorch       : %s", torch.__version__)
    if device.type != "cuda":
        logger.warning("CUDA not available -- full-train eval on CPU would take weeks.")

    logger.info("-" * 65)
    logger.info("Step 1: Load best M4 hyperparameters (backbone + attention)")
    _b_cfg, backbone_hp, branch_dilations, _a_cfg, attn_hp = load_best_params(logger)

    logger.info("-" * 65)
    logger.info("Step 2: Verify trained weights file")
    if not WEIGHTS_PATH.exists():
        logger.error("Weights file not found: %s.", WEIGHTS_PATH); sys.exit(1)
    weights_size_mb = WEIGHTS_PATH.stat().st_size / 1e6
    logger.info("Weights OK: %s (%.2f MB)", WEIGHTS_PATH, weights_size_mb)

    logger.info("-" * 65)
    logger.info("Step 3: Layer 1 SKIPPED for train (train was never apply_val_test_filter'd). "
                "Layers 2-4 still active.")

    logger.info("-" * 65)
    logger.info("Step 4: Build MultiScaleTCNWithAttention and load final weights")
    model = build_model(backbone_hp, branch_dilations, attn_hp, device, logger)
    state_dict = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    n_params = count_parameters(model)
    logger.info("Loaded %s state_dict into model (%s trainable params).",
                WEIGHTS_PATH.name, "{:,}".format(n_params))

    logger.info("-" * 65)
    logger.info("Step 5: Load FULL TRAIN split and build SafeEEGSegmentDataset loader")
    train_pairs, train_records = load_full_train_split(logger)
    batch_size = int(attn_hp["batch_size"])
    train_loader = make_safe_loader(train_pairs, batch_size, device,
                                    num_workers=NUM_DATA_WORKERS)
    logger.info("Train loader: %d batches (batch_size=%d, num_workers=%d)",
                len(train_loader), batch_size, NUM_DATA_WORKERS)

    logger.info("-" * 65)
    logger.info("Step 6: FP32 forward pass with per-batch isfinite assert")
    SafeEEGSegmentDataset.n_sanitised = 0
    t0_eval = time.time()
    train_f1_raw, y_true, _y_pred_05, y_prob = evaluate_model_safe(
        model, train_loader, device, logger, use_amp=USE_AMP_FOR_EVAL)
    eval_seconds = time.time() - t0_eval
    logger.info("Forward pass complete: %d segments in %.1f s | "
                "raw t=0.5 macro F1 = %.4f",
                len(y_true), eval_seconds, train_f1_raw)
    sanitised = SafeEEGSegmentDataset.n_sanitised
    if sanitised:
        logger.warning("SafeEEGSegmentDataset had to sanitise %d segments. "
                       "Train data carried NaN/Inf or |x|>1000 segments.",
                       sanitised)
    else:
        logger.info("SafeEEGSegmentDataset sanitised 0 segments.")

    logger.info("-" * 65)
    logger.info("Step 7: Run post-processing evaluation (Row 1 + Row 2)")
    (row1_metrics, row2_metrics,
     post_row2, far_row2) = run_postprocessing_evaluations(
        y_true, y_prob, logger)

    logger.info("-" * 65)
    logger.info("Step 7b: Corrected per-mouse-chronological event-level evaluation")
    if not MOUSE_METADATA_PATH.exists():
        logger.error("Mouse metadata not found at %s.", MOUSE_METADATA_PATH)
        raise FileNotFoundError(str(MOUSE_METADATA_PATH))
    mouse_metadata = json.loads(MOUSE_METADATA_PATH.read_text(encoding="utf-8"))
    train_eval_result = evaluate_event_level(
        train_records, y_true, y_prob, ANNOT_DIR, mouse_metadata, logger)

    reord = train_eval_result["reordered_arrays"]
    y_pred_row1 = reord["y_pred_row1"]
    y_pred_row2 = reord["y_pred_row2"]
    logger.info("-" * 65)
    logger.info("Step 8: Cache enriched full-train predictions bundle")
    np.savez_compressed(
        FULL_TRAIN_PREDICTIONS_NPZ,
        y_true=reord["y_true"].astype(np.int8),
        y_prob=reord["y_prob"].astype(np.float32),
        y_pred_row1=y_pred_row1.astype(np.int8),
        y_pred_row2=y_pred_row2.astype(np.int8),
        mouse_id=reord["mouse_id"],
        chrono_idx=reord["chrono_idx"],
        t_start_sec=reord["t_start_sec"],
        n_segments=np.int64(len(reord["y_true"])),
        segment_sec=np.float32(SEGMENT_SEC),
        smoothing_win=np.int64(SMOOTHING_WIN),
        refractory_sec=np.float32(REFRACTORY_SEC),
        min_event_sec=np.float32(MIN_EVENT_SEC),
    )
    logger.info("Saved enriched full-train predictions bundle: %s (%.2f MB)",
                FULL_TRAIN_PREDICTIONS_NPZ,
                FULL_TRAIN_PREDICTIONS_NPZ.stat().st_size / 1e6)

    logger.info("-" * 65)
    logger.info("Step 9: Save evaluation report, two-row CSV, classification reports")
    history     = None
    best_epoch  = 0
    best_val_f1 = 0.0
    elapsed_dt  = datetime.timedelta(seconds=int(eval_seconds))
    save_all_results(
        history, row1_metrics, row2_metrics,
        far_row2, backbone_hp, branch_dilations, attn_hp,
        best_epoch, best_val_f1, elapsed_dt, device, n_params, y_true,
        (y_prob >= 0.5).astype(int), post_row2["smoothed_preds"], logger)

    eval_report_path = FULL_TRAIN_EVAL_DIR / "ms_attn_full_train_evaluation_report.json"
    corrected_report = {
        "model":     MODEL_NAME,
        "timestamp": datetime.datetime.now().isoformat(),
        "weights_path":   str(WEIGHTS_PATH),
        "evaluation_set": "full_train (un-downsampled)",
        "manifest":       str(FULL_TRAIN_SPLITS_PATH),
        "n_segments":     int(len(y_true)),
        "predictions_npz": str(FULL_TRAIN_PREDICTIONS_NPZ),
        "fp32_forward":   bool(not USE_AMP_FOR_EVAL),
        "forward_pass_seconds": round(float(eval_seconds), 1),
        "branch_dilations": branch_dilations,
        "fusion":           backbone_hp.get("fusion", "concat"),
        "attention_hyperparameters": attn_hp,
        "backbone_params_source":   str(BACKBONE_PARAMS_PATH),
        "attention_params_source":  str(ATTN_PARAMS_PATH),
        "nan_protection": {
            "layer_1_manifest_filter": "SKIPPED (train was never apply_val_test_filter'd)",
            "layer_2_dataset":         "SafeEEGSegmentDataset",
            "layer_3_precision":       "FP32",
            "layer_4_isfinite_assert": True,
            "amplitude_threshold":     AMPLITUDE_THRESHOLD,
            "segments_sanitised":      int(sanitised),
        },
        "post_processing_params": {
            "smoothing_window":       SMOOTHING_WIN,
            "refractory_period_sec":  REFRACTORY_SEC,
            "min_event_duration_sec": MIN_EVENT_SEC,
            "step_sec":               STEP_SEC,
            "matching_rule":          "any-overlap",
        },
        "totals":                train_eval_result["totals"],
        "segment_level_metrics": train_eval_result["segment_level_metrics"],
        "event_level_metrics":   train_eval_result["event_level_metrics"],
        "per_mouse":             train_eval_result["per_mouse_results"],
    }
    with open(eval_report_path, "w", encoding="utf-8") as f:
        json.dump(corrected_report, f, indent=2, default=str)
    logger.info("Overwrote eval report with corrected schema: %s", eval_report_path)

    event_details_path = FULL_TRAIN_EVAL_DIR / "ms_attn_full_train_event_details_row2.csv"
    fieldnames_evt = ["mouse_id", "is_true_alarm", "start_sec", "end_sec",
                      "duration_sec", "max_prob", "matched_gt_idx",
                      "matched_gt_start_sec", "matched_gt_end_sec"]
    with open(event_details_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_evt)
        writer.writeheader()
        for row in train_eval_result["all_event_details"]:
            writer.writerow(row)
    logger.info("Overwrote event-details CSV with corrected rows: %s (%d rows)",
                event_details_path, len(train_eval_result["all_event_details"]))

    seg = train_eval_result["segment_level_metrics"]
    em  = train_eval_result["event_level_metrics"]
    fieldnames_3r = [
        "row", "threshold", "postprocessed",
        "accuracy", "precision", "recall", "specificity", "youden_j",
        "f1_macro", "auroc", "average_precision",
        "far_per_hour_seg", "far_per_hour_event",
        "n_true_alarms", "n_false_alarms", "n_total_events",
        "tp", "tn", "fp", "fn",
    ]
    rows_3r = [
        ("Row1_raw_0.5",      seg["row1_raw_threshold_0_5"],      "N/A"),
        ("Row2_postproc_0.5", seg["row2_postproc_threshold_0_5"], em["far_per_hour_event_CORRECTED"]),
    ]
    three_row_path = FULL_TRAIN_EVAL_DIR / "ms_attn_full_train_three_row_summary.csv"
    with open(three_row_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_3r)
        writer.writeheader()
        for label, m, far_evt in rows_3r:
            writer.writerow({
                "row":               label,
                "threshold":         m.get("threshold", THRESHOLD),
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
    logger.info("Overwrote three-row summary with corrected values: %s", three_row_path)

    logger.info("=" * 65)
    logger.info("CORRECTED EVENT-LEVEL FULL-TRAIN METRICS")
    logger.info("  TP / FP / FN          : %d / %d / %d", em["tp"], em["fp"], em["fn"])
    logger.info("  Precision             : %.4f", em["precision"])
    logger.info("  Recall (sensitivity)  : %.4f", em["recall"])
    logger.info("  F1                    : %.4f", em["f1"])
    logger.info("  FAR/hr event CORRECTED: %.4f  (non-ictal hours = %.1f)",
                em["far_per_hour_event_CORRECTED"],
                train_eval_result["totals"]["non_ictal_hours_corrected"])
    logger.info("  Mean detection latency: %s s", em["mean_detection_latency_sec"])
    logger.info("=" * 65)

    try:
        write_event_level_bundle(
            FULL_TRAIN_EVAL_DIR, "train", "M4 (MS-TCN+Attention) [FULL TRAIN]",
            train_eval_result, logger,
            order="min_then_refractory",
            min_event_duration_sec=MIN_EVENT_SEC,
            refractory_period_sec=REFRACTORY_SEC,
            smoothing_window=SMOOTHING_WIN,
            threshold=0.5,
            step_sec=STEP_SEC,
        )
    except Exception as exc:
        logger.warning(
            "Canonical 4-file bundle emission failed (%s). Segment-level "
            "outputs and the bespoke per-mouse JSON/CSV above remain valid.",
            exc)

    logger.info("-" * 65)
    logger.info("Step 10: Plot all evaluation figures")
    plot_all_figures(
        history, best_epoch, best_val_f1, y_true, y_prob,
        y_pred_row1, y_pred_row2,
        row1_metrics, row2_metrics,
        post_row2, logger)

    logger.info("-" * 65)
    logger.info("Step 11: Result_classReport bar plot (Row 1 + Row 2)")
    make_classreport_barplot(
        y_true,
        (y_prob >= 0.5).astype(int),
        post_row2["smoothed_preds"],
        row1_metrics, row2_metrics,
        FULL_TRAIN_EVAL_CLASSREPORT_DIR / "ms_attn_full_train_classreport_barplot.png",
        title_prefix="MS-TCN + Attention (FULL TRAIN)", logger=logger)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared.")

    logger.info("=" * 65)
    logger.info("M4 FULL-TRAIN EVALUATION COMPLETE")
    logger.info("  Train segments        : %d", len(y_true))
    logger.info("  Forward pass time     : %.1f s", eval_seconds)
    logger.info("  Row 1 F1 (raw  @ 0.5) : %.4f",
                row1_metrics.get("f1_macro", float("nan")))
    logger.info("  Row 1 AUROC           : %.4f",
                row1_metrics.get("auroc", float("nan")))
    logger.info("  Row 1 PRAUC           : %.4f",
                row1_metrics.get("average_precision", float("nan")))
    logger.info("  Row 2 F1 (post @ 0.5) : %.4f",
                row2_metrics.get("f1_macro", float("nan")))
    logger.info("  Row 2 FAR/hr (event)  : %.4f",
                row2_metrics.get("far_per_hour_event", float("nan")))
    logger.info("=" * 65)

    pfx = "ms_attn_full_train"
    all_outputs = [
        FULL_TRAIN_EVAL_DIR / "ms_attn_full_train_evaluation_report.json",
        FULL_TRAIN_EVAL_DIR / "ms_attn_full_train_three_row_summary.csv",
        FULL_TRAIN_EVAL_DIR / "ms_attn_full_train_classification_report_row1.json",
        FULL_TRAIN_EVAL_DIR / "ms_attn_full_train_classification_report_row2.json",
        FULL_TRAIN_EVAL_DIR / "ms_attn_full_train_event_details_row2.csv",
        FULL_TRAIN_PREDICTIONS_NPZ,
        FULL_TRAIN_EVAL_FIGURE_DIR / ("%s_confusion_matrix_row1.png" % pfx),
        FULL_TRAIN_EVAL_FIGURE_DIR / ("%s_confusion_matrix_row2.png" % pfx),
        FULL_TRAIN_EVAL_FIGURE_DIR / ("%s_roc_curve.png" % pfx),
        FULL_TRAIN_EVAL_FIGURE_DIR / ("%s_metrics_comparison.png" % pfx),
        FULL_TRAIN_EVAL_FIGURE_DIR / ("%s_pr_curve.png" % pfx),
        FULL_TRAIN_EVAL_FIGURE_DIR / ("%s_calibration_curve.png" % pfx),
        FULL_TRAIN_EVAL_FIGURE_DIR / ("%s_far_comparison.png" % pfx),
        FULL_TRAIN_EVAL_FIGURE_DIR / ("%s_segment_length_analysis.png" % pfx),
        FULL_TRAIN_EVAL_CLASSREPORT_DIR / ("%s_classreport_barplot.png" % pfx),
        FULL_TRAIN_EVAL_LOG_PATH,
    ]
    for p in all_outputs:
        status = "OK     " if Path(p).exists() else "MISSING"
        logger.info("  [%s] %s", status, p)


if __name__ == "__main__":
    main()
