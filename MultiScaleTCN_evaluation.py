"""
MultiScaleTCN_evaluation.py
===========================
Final evaluation of the trained Multi-Scale TCN (M3) model on the
held-out TEST partition. Mirrors MultiScaleTCN.py's post-training pass
in metrics, post-processing, figures, and JSON/CSV outputs, so the
test-set numbers are produced through exactly the same computations as
the validation-set numbers reported by the training script.

Pipeline position
-----------------
After  : MultiScaleTCN.py      (training; produces final weights)
This   : MultiScaleTCN_evaluation.py (test-set evaluation)

What this script does
---------------------
1. Loads the final trained weights from
       OUTPUT_ROOT / multiscale_tcn_final_weights.pt
   (produced by MultiScaleTCN.py).
2. Loads the TEST partition from the filtered splits manifest
       data_splits_nonictal_sampled_filtered.json
   and verifies it was produced by apply_val_test_filter.py
   (Layer 1 NaN/extreme-amplitude protection).
3. Runs an FP32 forward pass on GPU (Layer 3) using
   SafeEEGSegmentDataset (Layer 2) and a per-batch isfinite assertion
   (Layer 4). All four layers mirror m3_post_eval.py.
4. Caches y_true and y_prob to
       evaluation/multiscale_tcn_test_predictions_raw.npz
   so the post-processing pass can be re-run later without inference.
5. Computes the same Row 1 (raw t=0.5) and Row 2 (post-processed t=0.5)
   metric set as MultiScaleTCN.py via the shared helpers
   compute_all_metrics() and run_postprocessing_evaluations().
6. Produces the same figures, JSON evaluation report, two-row summary
   CSV, per-row sklearn classification reports, event-details CSV,
   and Result_classReport bar plot.

Why this script reuses the training-script helpers
--------------------------------------------------
Importing MultiScaleTCN does not execute its main() (guarded by
__name__ == "__main__"), so all top-level constants and functions are
importable. By rebinding the path globals on the imported module before
calling save_all_results() / plot_all_figures(), every artefact those
helpers write lands inside OUTPUT_ROOT/evaluation/ instead of
OUTPUT_ROOT/. The training-script outputs are never overwritten.

Eval-folder hygiene: the eval folder MUST contain only test-side
artefacts. The reused helpers in MultiScaleTCN.py would normally also
emit four training-side artefacts (training_log.json, epoch_metrics.csv,
training_curves.png, lr_schedule.png). To prevent this, we call those
helpers with history=None; they detect the missing history and skip the
training-only writes. Result: the eval folder contains test outputs only,
and the training outputs continue to live in OUTPUT_ROOT/ (where the
training run wrote them).

Inputs
------
  OUTPUT_ROOT / multiscale_tcn_final_weights.pt
  best_multiscale_params.json                      (architecture metadata)
  data_splits_nonictal_sampled_filtered.json       (test partition)

Outputs (under OUTPUT_ROOT / evaluation/) -- TEST artefacts only
-----------------------------------------
  multiscale_tcn_evaluation_report.json    -- Row 1 + Row 2 metrics
  multiscale_tcn_three_row_summary.csv     -- two-row tabular form
  multiscale_tcn_classification_report_row{1,2}.json
  multiscale_tcn_event_details_row2.csv
  multiscale_tcn_test_predictions_raw.npz  -- cached y_true, y_prob
  Result_classReport/multiscale_tcn_classreport_barplot.png
  figures/                                 -- Row 1 / Row 2 figures
                                              (training_curves.png and
                                               lr_schedule.png are NOT
                                               written here -- they live
                                               in OUTPUT_ROOT/figures/
                                               from the training run)
  ../logs/multiscale_tcn_evaluation.log    -- persistent log (per user pref)

Usage
-----
python MultiScaleTCN_evaluation.py
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
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
matplotlib.use("Agg")                                  # non-interactive backend

# Reuse the M3 training script's helpers and constants. Importing the module
# does NOT execute its main() because that's guarded by __name__ == "__main__".
# All top-level constants and functions become importable as-is.
from MultiScaleTCN import (
    SEED, MODEL_NAME, OUTPUT_ROOT,
    SPLITS_PATH, WEIGHTS_PATH, ANNOT_DIR, MOUSE_METADATA_PATH,
    SEGMENT_SEC,
    MIN_EVENT_SEC, REFRACTORY_SEC, SMOOTHING_WIN,
    load_best_params, build_model,
    run_postprocessing_evaluations, save_all_results, plot_all_figures,
)

# Four-layer NaN protection workflow lives in eval_utils.py:
#   Layer 1 -- verify_manifest_filtered (manifest-level filter check)
#   Layer 2 -- SafeEEGSegmentDataset (sanitise NaN/Inf, clip |x|>1000)
#   Layer 3 -- FP32 forward (USE_AMP_FOR_EVAL=False)
#   Layer 4 -- per-batch torch.isfinite assertion in evaluate_model_safe
# m3_post_eval.py is a one-off recovery utility for the original M3 crash;
# it is NOT a shared library and must NOT be imported here.
from eval_utils import (
    SafeEEGSegmentDataset,
    AMPLITUDE_THRESHOLD,
    make_safe_loader,
    evaluate_model_safe,
    verify_manifest_filtered,
    evaluate_event_level,
    THRESHOLD,
    STEP_SEC,
)

from tcn_utils import (
    set_seed,
    count_parameters,
    make_classreport_barplot,
)


# ---------------------------------------------------------------------------
# Constants (script-local)
# ---------------------------------------------------------------------------
USE_AMP_FOR_EVAL    = False                            # Layer 3: FP32 forward
NUM_DATA_WORKERS    = 4                                # DataLoader workers

# Output layout. Per user spec, all evaluation artefacts land in a dedicated
# evaluation/ subdirectory under the existing M3 OUTPUT_ROOT, and the
# per-script log lives at OUTPUT_ROOT/logs/multiscale_tcn_evaluation.log.
EVALUATION_DIR             = OUTPUT_ROOT / "evaluation"
EVALUATION_FIGURE_DIR      = EVALUATION_DIR / "figures"
EVALUATION_CLASSREPORT_DIR = EVALUATION_DIR / "Result_classReport"
EVAL_LOG_DIR               = OUTPUT_ROOT / "logs"
EVAL_LOG_PATH              = EVAL_LOG_DIR / "multiscale_tcn_evaluation.log"

# Cached test-set predictions (insurance copy). Lets a future post-hoc pass
# recompute Row 1 / Row 2 without re-running the multi-hour FP32 forward.
# "full" schema: y_true + y_prob + y_pred_row1 (raw t=0.5) + y_pred_row2
# (post-processed smoothed preds) -- matches the val NPZ written by the
# training script.
TEST_PREDICTIONS_NPZ       = EVALUATION_DIR / "multiscale_tcn_test_predictions_full.npz"


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------
def setup_logging():
    """Console + persistent FileHandler at EVAL_LOG_PATH.

    Per user preference: every script writes its own .log via a dedicated
    FileHandler so SLURM stdout is not the only record. The save location
    was confirmed at script-write time -- see EVAL_LOG_PATH above.
    """
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_CLASSREPORT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("multiscale_tcn_evaluation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(EVAL_LOG_PATH, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# _redirect_outputs_to_evaluation
# ---------------------------------------------------------------------------
def _redirect_outputs_to_evaluation(logger):
    """Rebind path globals on the imported MultiScaleTCN module so
    save_all_results() and plot_all_figures() write into EVALUATION_DIR.

    Why this works: those helpers were defined in MultiScaleTCN.py and
    look up names like OUTPUT_ROOT, FIGURE_DIR, EVAL_REPORT_PATH at call
    time in *MultiScaleTCN's* module namespace -- not in the caller's. By
    rebinding those names on the MultiScaleTCN module before invoking
    the helpers from this script, every write target lands under
    evaluation/ for this process only. A separate Python process running
    MultiScaleTCN.py for training is not affected.

    The training-script directories are read for inputs (final weights,
    training history) but never overwritten by this script.
    """
    import MultiScaleTCN as _m3

    # TRAIN_LOG_PATH and EPOCH_CSV are intentionally NOT redirected: the
    # save_all_results() helper now skips writing them when called with
    # history=None (the test-eval path), so no redirection target is needed.
    # The eval folder therefore contains only test-side artefacts.
    redirections = {
        "OUTPUT_ROOT":      EVALUATION_DIR,
        "FIGURE_DIR":       EVALUATION_FIGURE_DIR,
        "LOG_DIR":          EVAL_LOG_DIR,
        "EVAL_REPORT_PATH": EVALUATION_DIR / "multiscale_tcn_evaluation_report.json",
        "THREE_ROW_CSV":    EVALUATION_DIR / "multiscale_tcn_three_row_summary.csv",
    }
    logger.info("Redirecting MultiScaleTCN write targets to EVALUATION_DIR:")
    for name, new_path in redirections.items():
        original = getattr(_m3, name, None)
        setattr(_m3, name, new_path)
        logger.info("  %-18s : %s -> %s", name, original, new_path)


# ---------------------------------------------------------------------------
# load_test_split
# ---------------------------------------------------------------------------
def load_test_split(logger):
    """Load TEST file-label pairs from the filtered splits manifest.

    Mirrors load_splits() in MultiScaleTCN.py but loads the test partition
    instead of train + val. The manifest is the single source of truth
    used across the pipeline (data_splits_nonictal_sampled_filtered.json).

    Returns
    -------
    list of (filepath, label) tuples for the held-out test partition.
    """
    if not SPLITS_PATH.exists():
        logger.error(
            "data_splits_nonictal_sampled_filtered.json not found at %s. "
            "Pipeline: generate_data_splits.py -> create_balanced_splits.py "
            "-> apply_val_test_filter.py.", SPLITS_PATH)
        raise FileNotFoundError(str(SPLITS_PATH))

    logger.info("Loading splits from: %s", SPLITS_PATH)
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)

    # Empty test partition is a hard error -- there is genuinely no data
    # to evaluate on. Check this BEFORE the test_status field so the
    # error message is accurate.
    if "test" not in splits or not splits["test"]:
        logger.error("Test partition is empty in manifest %s", SPLITS_PATH)
        raise RuntimeError("Empty test partition")

    # metadata.test_status guard: when "complete", proceed silently. When
    # missing or "pending", fall back to a positive-evidence check: if the
    # test partition is populated AND verify_manifest_test_filtered has
    # already confirmed apply_val_test_filter ran (Step 3 in main), the
    # data is genuinely ready -- the field was just dropped by an
    # intermediate pipeline step (create_balanced_splits.py or
    # apply_val_test_filter.py rebuilds the metadata block without
    # copying test_status across from generate_data_splits.py's output).
    # Warn and continue rather than blocking the run on a documentation
    # field that is decoupled from actual readiness.
    test_status = splits.get("metadata", {}).get("test_status", "pending")
    if test_status != "complete":
        logger.warning(
            "metadata.test_status = '%s' (expected 'complete'). "
            "Proceeding anyway because (a) the test partition contains "
            "%d records and (b) apply_val_test_filter is recorded in "
            "meta.filter_history (verified in Step 3). The test_status "
            "field is likely missing because an intermediate pipeline "
            "step rebuilt the metadata block.",
            test_status, len(splits["test"]))
    else:
        logger.info("metadata.test_status = 'complete'.")

    test_pairs   = [(rec["filepath"], rec["label"]) for rec in splits["test"]]
    test_records = list(splits["test"])     # full enriched records (mouse_id, chrono_idx, t_start_sec)

    n_total = len(test_pairs)
    n_sz = sum(1 for _, l in test_pairs if l == 1)
    n_nsz = n_total - n_sz
    pct = n_sz / n_total * 100 if n_total > 0 else 0.0
    mouse_ids = sorted({Path(fp).stem.split("_")[0] for fp, _ in test_pairs})
    logger.info("TEST: %d total | %d seizure | %d non-seizure | %.2f%% ictal | %d mice",
                n_total, n_sz, n_nsz, pct, len(mouse_ids))

    if not all("chrono_idx" in r for r in test_records[:5]):
        logger.error("Test records missing 'chrono_idx' -- pointing at the un-enriched manifest? "
                     "SPLITS_PATH must end in '_enriched.json'. Got: %s", SPLITS_PATH)
        raise RuntimeError("Manifest not enriched with chronology")

    return test_pairs, test_records




# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    logger = setup_logging()

    logger.info("=" * 65)
    logger.info("MultiScaleTCN_evaluation.py")
    logger.info("Timestamp     : %s", datetime.datetime.now().isoformat())
    logger.info("Purpose       : Final evaluation of M3 on TEST set")
    logger.info("Model         : %s", MODEL_NAME)
    logger.info("Weights       : %s", WEIGHTS_PATH)
    logger.info("Read root     : %s (training artefacts; never overwritten)", OUTPUT_ROOT)
    logger.info("Write root    : %s (everything this script emits lands here)", EVALUATION_DIR)
    logger.info("Eval log path : %s", EVAL_LOG_PATH)
    logger.info("Mode          : Row 1 + Row 2 (Row 3 retired); FP32 forward; "
                "four-layer NaN protection")
    logger.info("=" * 65)

    # Rebind MultiScaleTCN module path globals BEFORE invoking
    # save_all_results / plot_all_figures so all artefacts land in
    # evaluation/ rather than overwriting the training-run outputs.
    _redirect_outputs_to_evaluation(logger)

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
        logger.warning("CUDA not available -- evaluation will run on CPU and "
                       "may take significantly longer than on GPU.")

    # -- Step 1: Load M3 hyperparameters and architecture metadata ------------
    logger.info("-" * 65)
    logger.info("Step 1: Load best M3 hyperparameters")
    # load_best_params returns (config, hp, branch_dilations); only the
    # latter two are used downstream, so the full config dict is discarded.
    _config, hp, branch_dilations = load_best_params(logger)

    # -- Step 2: Verify weights file exists ----------------------------------
    logger.info("-" * 65)
    logger.info("Step 2: Verify trained weights file")
    if not WEIGHTS_PATH.exists():
        logger.error("Weights file not found: %s. Run MultiScaleTCN.py first "
                     "to produce final weights.", WEIGHTS_PATH)
        sys.exit(1)
    weights_size_mb = WEIGHTS_PATH.stat().st_size / 1e6
    logger.info("Weights OK: %s (%.2f MB)", WEIGHTS_PATH, weights_size_mb)

    # -- Step 3: Layer 1 NaN protection -- verify manifest filter -------------
    logger.info("-" * 65)
    logger.info("Step 3: Layer 1 -- verify apply_val_test_filter ran on test")
    manifest_filter = verify_manifest_filtered(SPLITS_PATH, partition_key="test", logger=logger)

    # -- Step 4: Build model and load trained weights -------------------------
    logger.info("-" * 65)
    logger.info("Step 4: Build MultiScaleTCN and load final weights")
    model = build_model(hp, branch_dilations, device, logger)
    state_dict = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    n_params = count_parameters(model)
    logger.info("Loaded %s state_dict into model (%s trainable params).",
                WEIGHTS_PATH.name, "{:,}".format(n_params))

    # -- Step 5: Load test split + build safe loader (Layer 2) ----------------
    logger.info("-" * 65)
    logger.info("Step 5: Load TEST split and build SafeEEGSegmentDataset loader")
    test_pairs, test_records = load_test_split(logger)
    batch_size = int(hp["batch_size"])
    test_loader = make_safe_loader(test_pairs, batch_size, device,
                                   num_workers=NUM_DATA_WORKERS)
    logger.info("Test loader: %d batches (batch_size=%d, num_workers=%d)",
                len(test_loader), batch_size, NUM_DATA_WORKERS)

    # -- Step 6: FP32 forward pass (Layer 3) + isfinite assert (Layer 4) ------
    logger.info("-" * 65)
    logger.info("Step 6: FP32 forward pass with per-batch isfinite assert")
    # Reset the class-level sanitisation counter so the post-eval log line
    # at line ~459 reflects only this script's invocation. Defensive against
    # any prior in-process instantiation of the dataset that left the counter
    # non-zero (not possible in a clean `python` invocation, but harmless).
    SafeEEGSegmentDataset.n_sanitised = 0
    t0_eval = time.time()
    # evaluate_model_safe returns (val_f1, y_true, y_pred, y_prob); the
    # per-batch y_pred at threshold 0.5 is recomputed below as y_pred_row1
    # (identical values), so we discard the helper's copy.
    test_f1_raw, y_true, _y_pred_05, y_prob = evaluate_model_safe(
        model, test_loader, device, logger, use_amp=USE_AMP_FOR_EVAL)
    eval_seconds = time.time() - t0_eval
    logger.info("Forward pass complete: %d segments in %.1f s | "
                "raw t=0.5 macro F1 = %.4f",
                len(y_true), eval_seconds, test_f1_raw)
    sanitised = SafeEEGSegmentDataset.n_sanitised
    if sanitised:
        logger.warning("SafeEEGSegmentDataset had to sanitise %d segments. "
                       "Layer 1 (manifest filter) likely missed something.",
                       sanitised)
    else:
        logger.info("SafeEEGSegmentDataset sanitised 0 segments (Layer 1 OK).")

    # -- Step 7: Run post-processing evaluation (Row 1 + Row 2) --------------
    # Legacy call -- kept for save_all_results' segment-level outputs (per-row
    # classification reports). The event-level FAR/hr it produces is
    # superseded by Step 7b and overwritten in the eval report.
    logger.info("-" * 65)
    logger.info("Step 7: Run post-processing evaluation (Row 1 + Row 2)")
    (row1_metrics, row2_metrics,
     post_row2, far_row2) = run_postprocessing_evaluations(
        y_true, y_prob, logger)

    # -- Step 7b: Corrected per-mouse-chronological event-level evaluation ----
    logger.info("-" * 65)
    logger.info("Step 7b: Corrected per-mouse-chronological event-level evaluation")
    if not MOUSE_METADATA_PATH.exists():
        logger.error("Mouse metadata not found at %s. Run extract_mouse_metadata.py first.",
                     MOUSE_METADATA_PATH)
        raise FileNotFoundError(str(MOUSE_METADATA_PATH))
    mouse_metadata = json.loads(MOUSE_METADATA_PATH.read_text(encoding="utf-8"))
    test_eval_result = evaluate_event_level(
        test_records, y_true, y_prob, ANNOT_DIR, mouse_metadata, logger)

    # -- Step 8: Cache test predictions bundle (enriched schema) -------------
    reord = test_eval_result["reordered_arrays"]
    y_pred_row1 = reord["y_pred_row1"]
    y_pred_row2 = reord["y_pred_row2"]
    logger.info("-" * 65)
    logger.info("Step 8: Cache enriched test predictions bundle")
    np.savez_compressed(
        TEST_PREDICTIONS_NPZ,
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
    logger.info("Saved enriched test predictions bundle: %s (%.2f MB)",
                TEST_PREDICTIONS_NPZ, TEST_PREDICTIONS_NPZ.stat().st_size / 1e6)

    # -- Step 9: Save structured results (JSON / CSV / per-row reports) -------
    logger.info("-" * 65)
    logger.info("Step 9: Save evaluation report, two-row CSV, classification reports")

    history     = None
    best_epoch  = 0
    best_val_f1 = 0.0
    elapsed_dt  = datetime.timedelta(seconds=int(eval_seconds))

    save_all_results(
        history, row1_metrics, row2_metrics,
        far_row2, hp, branch_dilations,
        best_epoch, best_val_f1, elapsed_dt, device, n_params, y_true,
        (y_prob >= 0.5).astype(int), post_row2["smoothed_preds"], logger)

    # -- Step 9b: Overwrite eval report with corrected event-level metrics ----
    eval_report_path = EVALUATION_DIR / "multiscale_tcn_evaluation_report.json"
    corrected_report = {
        "model":     MODEL_NAME,
        "timestamp": datetime.datetime.now().isoformat(),
        "weights_path":   str(WEIGHTS_PATH),
        "evaluation_set": "test",
        "n_test_segments": int(len(y_true)),
        "test_predictions_npz": str(TEST_PREDICTIONS_NPZ),
        "fp32_forward":   bool(not USE_AMP_FOR_EVAL),
        "forward_pass_seconds": round(float(eval_seconds), 1),
        "branch_dilations": branch_dilations,
        "fusion":           hp.get("fusion", "concat"),
        "tuning_source":    "best_multiscale_params.json",
        "nan_protection": {
            "layer_1_manifest_filter": manifest_filter,
            "layer_2_dataset":         "SafeEEGSegmentDataset",
            "layer_3_precision":       "FP32",
            "layer_4_isfinite_assert": True,
            "amplitude_threshold":     AMPLITUDE_THRESHOLD,
        },
        "post_processing_params": {
            "smoothing_window":       SMOOTHING_WIN,
            "refractory_period_sec":  REFRACTORY_SEC,
            "min_event_duration_sec": MIN_EVENT_SEC,
            "step_sec":               STEP_SEC,
            "matching_rule":          "any-overlap",
        },
        "totals":                test_eval_result["totals"],
        "segment_level_metrics": test_eval_result["segment_level_metrics"],
        "event_level_metrics":   test_eval_result["event_level_metrics"],
        "per_mouse":             test_eval_result["per_mouse_results"],
    }
    with open(eval_report_path, "w", encoding="utf-8") as f:
        json.dump(corrected_report, f, indent=2, default=str)
    logger.info("Overwrote eval report with corrected schema: %s", eval_report_path)

    # -- Step 9c: Overwrite event-details CSV with corrected per-event rows ---
    event_details_path = EVALUATION_DIR / "multiscale_tcn_event_details_row2.csv"
    fieldnames_evt = ["mouse_id", "is_true_alarm", "start_sec", "end_sec",
                      "duration_sec", "max_prob", "matched_gt_idx",
                      "matched_gt_start_sec", "matched_gt_end_sec"]
    with open(event_details_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_evt)
        writer.writeheader()
        for row in test_eval_result["all_event_details"]:
            writer.writerow(row)
    logger.info("Overwrote event-details CSV with corrected rows: %s (%d rows)",
                event_details_path, len(test_eval_result["all_event_details"]))

    # -- Step 9d: Overwrite three-row summary CSV with corrected values -------
    seg = test_eval_result["segment_level_metrics"]
    em  = test_eval_result["event_level_metrics"]
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
    three_row_path = EVALUATION_DIR / "multiscale_tcn_three_row_summary.csv"
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
    logger.info("CORRECTED EVENT-LEVEL TEST METRICS")
    logger.info("  TP / FP / FN          : %d / %d / %d", em["tp"], em["fp"], em["fn"])
    logger.info("  Precision             : %.4f", em["precision"])
    logger.info("  Recall (sensitivity)  : %.4f", em["recall"])
    logger.info("  F1                    : %.4f", em["f1"])
    logger.info("  FAR/hr event CORRECTED: %.4f  (non-ictal hours = %.1f)",
                em["far_per_hour_event_CORRECTED"],
                test_eval_result["totals"]["non_ictal_hours_corrected"])
    logger.info("  Mean detection latency: %s s", em["mean_detection_latency_sec"])
    logger.info("=" * 65)

    # -- Step 10: Plot all figures (Row 1 / Row 2; Row 3 retired) ------------
    logger.info("-" * 65)
    logger.info("Step 10: Plot all evaluation figures")
    plot_all_figures(
        history, best_epoch, best_val_f1, y_true, y_prob,
        y_pred_row1, y_pred_row2,
        row1_metrics, row2_metrics,
        post_row2,
        branch_dilations, hp, logger)

    # -- Step 11: Result_classReport bar plot -------------------------------
    # Macro-avg classification metrics (Row 1 + Row 2) plus FAR/hr in a
    # standalone two-panel figure -- identical layout across M1, M2, M3, M4
    # via the shared helper in tcn_utils. Writes under EVALUATION_DIR; the
    # bar plot the training script produced is not overwritten.
    logger.info("-" * 65)
    logger.info("Step 11: Result_classReport bar plot (Row 1 + Row 2)")
    make_classreport_barplot(
        y_true, y_pred_row1, y_pred_row2,
        row1_metrics, row2_metrics,
        EVALUATION_CLASSREPORT_DIR / "multiscale_tcn_classreport_barplot.png",
        title_prefix="Multi-Scale TCN", logger=logger)

    # -- Step 12: Final inventory and cleanup -------------------------------
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared.")

    logger.info("=" * 65)
    logger.info("M3 TEST EVALUATION COMPLETE")
    logger.info("  Test segments         : %d", len(y_true))
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
    logger.info("  Row 3                 : retired (threshold optimisation disabled)")
    logger.info("=" * 65)

    # Inventory check. Training-only artefacts (training_curves.png,
    # lr_schedule.png, training_log.json, epoch_metrics.csv) are intentionally
    # absent from this list -- they live in OUTPUT_ROOT/, written by the
    # training run, and must not appear under EVALUATION_DIR. The eval folder
    # is for test-side artefacts only.
    pfx = "multiscale_tcn"
    all_outputs = [
        EVALUATION_DIR / "multiscale_tcn_evaluation_report.json",
        EVALUATION_DIR / "multiscale_tcn_three_row_summary.csv",
        EVALUATION_DIR / "multiscale_tcn_classification_report_row1.json",
        EVALUATION_DIR / "multiscale_tcn_classification_report_row2.json",
        EVALUATION_DIR / "multiscale_tcn_event_details_row2.csv",
        TEST_PREDICTIONS_NPZ,
        EVALUATION_FIGURE_DIR / ("%s_confusion_matrix_row1.png" % pfx),
        EVALUATION_FIGURE_DIR / ("%s_confusion_matrix_row2.png" % pfx),
        EVALUATION_FIGURE_DIR / ("%s_roc_curve.png" % pfx),
        EVALUATION_FIGURE_DIR / ("%s_metrics_comparison.png" % pfx),
        EVALUATION_FIGURE_DIR / ("%s_pr_curve.png" % pfx),
        EVALUATION_FIGURE_DIR / ("%s_calibration_curve.png" % pfx),
        EVALUATION_FIGURE_DIR / ("%s_far_comparison.png" % pfx),
        EVALUATION_FIGURE_DIR / ("%s_segment_length_analysis.png" % pfx),
        EVALUATION_FIGURE_DIR / ("%s_branch_rf_diagram.png" % pfx),
        EVALUATION_CLASSREPORT_DIR / ("%s_classreport_barplot.png" % pfx),
        EVAL_LOG_PATH,
    ]
    for p in all_outputs:
        status = "OK     " if Path(p).exists() else "MISSING"
        logger.info("  [%s] %s", status, p)


if __name__ == "__main__":
    main()
