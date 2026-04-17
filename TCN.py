"""
TCN.py
======
Training script for Model 1: Supervised TCN baseline.

Trains for exactly 100 epochs with early stopping
on validation macro F1 (patience 10). Saves model
weights, checkpoints, logs, and a full evaluation
report on the validation set.

Three evaluation configurations are produced:
  Row 1: raw predictions at threshold 0.5
  Row 2: post-processed predictions at threshold 0.5
  Row 3: post-processed predictions at optimal threshold

The test set is never loaded in this script.
It is reserved for final_evaluation.py.

Pipeline position
-----------------
After  : tcn_HPT_binary.ipynb (produces best_params.json)
Before : final_evaluation.py (loads tcn_final_weights.pt)

Usage
-----
python TCN.py

Key outputs
-----------
outputs/TCN/tcn_final_weights.pt
outputs/TCN/tcn_evaluation_report.json
outputs/TCN/tcn_three_row_summary.csv
outputs/TCN/figures/  (12 figures)
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import json
import logging
import sys
import csv
import time
import datetime
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")                                  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve,
    average_precision_score,
    classification_report, calibration_curve,
)

# tcn_utils imports -- exact names confirmed from reading tcn_utils.py
# run_training() exists but is not used here; an explicit training loop
# provides full control over checkpointing, logging, and per-epoch CSV output.
from tcn_utils import (
    set_seed,
    TCN,
    make_loader,
    # filter_unpaired_subjects,  # handled offline by create_balanced_splits.py
    train_one_epoch,
    count_parameters,
    segment_predictions_to_events,
    compute_event_level_far,
    find_optimal_threshold,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED              = 42                                 # global reproducibility seed (Python, NumPy, PyTorch CPU+CUDA)
# 100 epochs provides a generous upper bound that lets the cosine
# annealing schedule complete a full cycle from lr to eta_min.
# Early stopping (patience=10) terminates training well before 100
# epochs if the model has converged, so the budget is rarely exhausted.
# This value is standard in EEG deep learning literature (Acharya et al.,
# 2018; Yildirim et al., 2020) and balances compute cost against the
# risk of premature termination when validation F1 plateaus temporarily.
MAX_EPOCHS        = 100                                # max training epochs (upper bound)
ES_PATIENCE       = 10                                 # epochs without val F1 improvement before stopping
CHECKPOINT_FREQ   = 5                                  # save periodic checkpoint every N epochs for crash recovery
KEEP_CKPTS        = 3                                  # disk-space cap: keep only the 3 most recent periodic checkpoints
FS                = 500                                # native EDF sampling rate (Hz)
SEGMENT_LEN       = 2500                               # samples per segment: 5 s * 500 Hz
SEGMENT_SEC       = 5.0                                # segment duration in seconds (= SEGMENT_LEN / FS)
STEP_SEC          = 2.5                                # step between segment starts: 50% overlap for temporal continuity
# Minimum event duration filter: genuine rodent seizures typically
# last at least 10 seconds (Luttjohann et al., 2009). Events shorter
# than this threshold are almost certainly artefacts or transient noise.
MIN_EVENT_SEC     = 10.0                               # discard detected events shorter than this (seconds)
# Refractory period: a single seizure can produce a brief mid-event
# probability dip below threshold. Merging events separated by < 30 s
# prevents one seizure from being counted as multiple alarms.
REFRACTORY_SEC    = 30.0                               # merge events separated by fewer than this (seconds)
# Smoothing window: a 3-segment uniform moving average suppresses
# isolated single-segment false-positive spikes caused by transient
# artefacts before binarisation. Wider windows risk blurring seizure
# onset boundaries; narrower windows provide insufficient suppression.
SMOOTHING_WIN     = 3                                  # number of segments in the probability smoothing kernel
MODEL_NAME        = "TCN"                              # model identifier for output filenames and JSON records

OUTPUT_ROOT       = Path("outputs") / "TCN"            # all M1 outputs under this directory
CKPT_DIR          = OUTPUT_ROOT / "checkpoints"        # periodic and best-model checkpoints
LOG_DIR           = OUTPUT_ROOT / "logs"               # training log (DEBUG-level detail)
FIGURE_DIR        = OUTPUT_ROOT / "figures"            # all 12 evaluation figures
WEIGHTS_PATH      = OUTPUT_ROOT / "tcn_final_weights.pt"       # final model weights (best epoch)
TRAIN_LOG_PATH    = OUTPUT_ROOT / "tcn_training_log.json"      # full training history as JSON
EVAL_REPORT_PATH  = OUTPUT_ROOT / "tcn_evaluation_report.json" # three-row evaluation report
THRESH_PATH       = OUTPUT_ROOT / "tcn_optimal_threshold.json" # Youden-optimal threshold + metadata
EPOCH_CSV         = OUTPUT_ROOT / "tcn_epoch_metrics.csv"      # per-epoch loss, F1, LR for plotting
THREE_ROW_CSV     = OUTPUT_ROOT / "tcn_three_row_summary.csv"  # paper Table 1 (M1 block)
# data_splits.json -- single source of truth (matches all other pipeline scripts)
# Previous (uniform downsampling): data_splits.json
# SPLITS_PATH         = Path("/scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits.json")
# Current (proximity-aware downsampling): data_splits_nonictal_sampled.json
SPLITS_PATH         = Path("/scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits_nonictal_sampled.json")
BEST_PARAMS_PATH    = Path("/home/people/22206468/scratch/OUTPUT/MODEL1_OUTPUT/TCNtuning_outputs") / "best_params.json"


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------
def setup_logging():
    """Create all output directories and configure the logger.

    Writes to both LOG_FILE (DEBUG+) and stdout (INFO+).

    Returns
    -------
    logger : logging.Logger
    """
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    log_file = LOG_DIR / "TCN_training.log"
    logger = logging.getLogger("TCN_training")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()                         # prevent duplicate handlers on resume

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")  # append to preserve logs on resume
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# load_best_params
# ---------------------------------------------------------------------------
def load_best_params(logger):
    """Load best TCN hyperparameters from best_params.json.

    Returns
    -------
    tuple of (full_config: dict, hp: dict)

    Raises
    ------
    FileNotFoundError if file is absent.
    KeyError if 'hyperparameters' key is missing.
    """
    if not BEST_PARAMS_PATH.exists():
        logger.error("best_params.json not found at %s. Run tcn_HPT_binary.ipynb first.",
                     BEST_PARAMS_PATH)
        raise FileNotFoundError(str(BEST_PARAMS_PATH))

    with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    if "hyperparameters" not in config:
        logger.error("'hyperparameters' key missing from %s.", BEST_PARAMS_PATH)
        raise KeyError("hyperparameters")

    hp = config["hyperparameters"]
    logger.info("TCN hyperparameters loaded from %s:", BEST_PARAMS_PATH)
    for k, v in hp.items():
        logger.info("  %-20s: %s", k, v)

    # Receptive field: RF = 2 * (2^L - 1) * (k - 1) + 1
    rf = 2 * (2 ** hp["num_layers"] - 1) * (hp["kernel_size"] - 1) + 1
    logger.info("Receptive field: %d samples (%.3f s at %d Hz)", rf, rf / FS, FS)
    if rf < FS:
        logger.warning("RF (%d) < FS (%d): receptive field is less than 1 second.", rf, FS)

    return config, hp


# ---------------------------------------------------------------------------
# load_splits
# ---------------------------------------------------------------------------
def load_splits(logger):
    """Load train and val file-label pairs from data_splits.json.

    Never loads test pairs. Returns (train_pairs, val_pairs).

    Raises
    ------
    FileNotFoundError if data_splits.json is absent.
    RuntimeError if either partition is empty.
    """
    if not SPLITS_PATH.exists():
        logger.error("data_splits.json not found at %s. "
                     "Run generate_data_splits.py first.", SPLITS_PATH)
        raise FileNotFoundError(str(SPLITS_PATH))

    logger.info("Loading splits from: %s", SPLITS_PATH)
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)

    # Convert records to (filepath, label) tuples
    train_pairs = [(rec["filepath"], rec["label"]) for rec in splits["train"]]
    val_pairs = [(rec["filepath"], rec["label"]) for rec in splits["val"]]

    if not train_pairs:
        logger.error("Train partition is empty.")
        raise RuntimeError("Empty train partition")
    if not val_pairs:
        logger.error("Val partition is empty.")
        raise RuntimeError("Empty val partition")

    for name, pairs in [("TRAIN", train_pairs), ("VAL", val_pairs)]:
        n_total = len(pairs)
        n_sz = sum(1 for _, l in pairs if l == 1)
        n_nsz = n_total - n_sz
        pct = n_sz / n_total * 100 if n_total > 0 else 0.0
        mouse_ids = sorted({Path(fp).stem.split("_")[0] for fp, _ in pairs})
        logger.info("%s: %d total | %d seizure | %d non-seizure | %.1f%% ictal | %d mice",
                    name, n_total, n_sz, n_nsz, pct, len(mouse_ids))

    # Check test status
    test_status = splits.get("metadata", {}).get("test_status", "pending")
    if test_status != "complete":
        logger.info("Test data not yet ready -- test split not loaded here.")

    return train_pairs, val_pairs


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------
def build_model(hp, device, logger):
    """Instantiate TCN using hyperparameters from hp.

    Parameters
    ----------
    hp     : dict -- hyperparameters (num_layers, num_filters, kernel_size, dropout)
    device : torch.device
    logger : logging.Logger

    Returns
    -------
    model : nn.Module
    """
    # Fix seed before weight init so every run starts from identical parameters.
    # This isolates the effect of hyperparameters from initialisation randomness.
    set_seed(SEED)
    # TCN.__init__(num_layers, num_filters, kernel_size, dropout, fs=500)
    # All four architecture params come from Optuna's best trial in best_params.json.
    # fs is used only for receptive-field logging inside TCN, not for computation.
    model = TCN(
        num_layers=int(hp["num_layers"]),     # L: depth and RF growth
        num_filters=int(hp["num_filters"]),   # channel width per block
        kernel_size=int(hp["kernel_size"]),   # local temporal resolution
        dropout=float(hp["dropout"]),         # spatial dropout rate (Dropout1d)
        fs=FS,                                # sampling rate for RF logging only
    )
    model = model.to(device)                  # move all parameters to GPU if available
    n_params = count_parameters(model)        # sum of requires_grad=True elements
    logger.info("Model: %s", MODEL_NAME)
    logger.info("Parameters: %s", "{:,}".format(n_params))
    logger.info("Device: %s", device)
    return model


# ---------------------------------------------------------------------------
# build_training_components
# ---------------------------------------------------------------------------
def build_training_components(model, hp, pos_weight, device, logger):
    """Build optimiser, scheduler, and loss function.

    pos_weight is passed in (set to 1.0 after offline downsampling).

    Returns (optimiser, scheduler, criterion).
    """
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=float(hp["learning_rate"]),
        weight_decay=float(hp["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max=MAX_EPOCHS,
        eta_min=float(hp["learning_rate"]) * 0.01)
    # pos_weight = 1.0: the 1:4 offline downsampling is the sole imbalance
    # correction. No additional pos_weight upweighting is applied.
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    logger.info("Optimiser: AdamW (lr=%.2e, wd=%.2e)", hp["learning_rate"], hp["weight_decay"])
    logger.info("Scheduler: CosineAnnealingLR (T_max=%d)", MAX_EPOCHS)
    logger.info("pos_weight: %.4f", pos_weight.item())
    return optimiser, scheduler, criterion


# ---------------------------------------------------------------------------
# save_checkpoint
# ---------------------------------------------------------------------------
def save_checkpoint(epoch, model, optimiser, scheduler, val_f1, train_loss,
                    hp, path, logger, best_model_state=None,
                    best_val_f1=None, best_epoch=None, epochs_no_imp=None):
    """Save a full training state checkpoint. Non-fatal on failure.

    When best_model_state is provided, the checkpoint contains everything
    needed for a clean resume: current training state + best-epoch weights.
    """
    try:
        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimiser_state": optimiser.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "val_f1": val_f1,
            "train_loss": train_loss,
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
            "epochs_no_imp": epochs_no_imp,
            "hyperparameters": hp,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        if best_model_state is not None:
            payload["best_model_state"] = best_model_state
        torch.save(payload, path)
        logger.debug("Checkpoint saved: %s", path)
    except Exception as exc:
        logger.error("Checkpoint save failed (%s): %s", path, exc)


# ---------------------------------------------------------------------------
# cleanup_checkpoints
# ---------------------------------------------------------------------------
def cleanup_checkpoints(ckpt_dir, keep_last_n, logger):
    """Delete old periodic checkpoints, keeping only the most recent keep_last_n.

    Never deletes tcn_best.pt.
    """
    pattern = "tcn_epoch_*.pt"
    ckpts = sorted(ckpt_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    to_remove = ckpts[:-keep_last_n] if len(ckpts) > keep_last_n else []
    for p in to_remove:
        p.unlink(missing_ok=True)
        logger.debug("Removed old checkpoint: %s", p.name)


# ---------------------------------------------------------------------------
# train_epoch
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimiser, criterion, device, scaler=None):
    """Run one training epoch. Delegates to train_one_epoch with optional AMP scaler.

    Returns mean training loss.
    """
    return train_one_epoch(model, loader, optimiser, criterion, device, scaler=scaler)


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------
def evaluate_model(model, loader, device, logger, use_amp=False):
    """Evaluate model in a single forward pass, collecting predictions and probabilities.

    Returns (val_f1, y_true, y_pred, y_prob).
    """
    model.eval()
    all_true = []
    all_pred = []
    all_probs = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
            # sigmoid + threshold outside autocast for FP32 precision at decision boundary
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            all_true.append(y.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_probs)
    val_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return val_f1, y_true, y_pred, y_prob


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------
def compute_all_metrics(y_true, y_pred, y_prob, segment_sec, logger, label=""):
    """Compute all evaluation metrics for one evaluation row.

    Returns a dict with all values for JSON serialisation.
    """
    # labels=[0,1] forces a 2x2 matrix even if one class is absent in y_pred
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)  # cast to Python int for JSON

    accuracy    = accuracy_score(y_true, y_pred)           # (TP+TN) / total
    prec        = precision_score(y_true, y_pred, pos_label=1, zero_division=0)  # TP / (TP+FP)
    recall      = recall_score(y_true, y_pred, pos_label=1, zero_division=0)     # TP / (TP+FN) = sensitivity
    f1_macro    = f1_score(y_true, y_pred, average="macro", zero_division=0)     # mean of per-class F1
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # TN / (TN+FP)
    # AUROC uses continuous probabilities, not binary predictions, so it is
    # threshold-invariant and identical across all three evaluation rows.
    auroc       = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    avg_prec    = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    # Youden J = sensitivity + specificity - 1. Ranges from -1 (anti-classifier)
    # to +1 (perfect). It is the vertical distance from the ROC diagonal to the
    # operating point. Used as the threshold selection objective because it
    # treats missed seizures and false alarms symmetrically.
    youden_j    = recall + specificity - 1.0

    # Segment-level FAR/hr: raw FP segments divided by non-ictal recording hours.
    # Only non-ictal segments contribute to the denominator because FAR measures
    # alarms during normal brain activity, not during seizures.
    n_non_ic   = tn + fp                                   # total non-ictal segments
    non_ic_hrs = (n_non_ic * segment_sec) / 3600.0         # non-ictal duration in hours
    far_seg    = fp / non_ic_hrs if non_ic_hrs > 0 else 0.0

    logger.info("-- Metrics [%s] --", label)
    logger.info("  Accuracy          : %.4f", accuracy)
    logger.info("  Precision         : %.4f", prec)
    logger.info("  Recall/Sensitivity: %.4f", recall)
    logger.info("  Specificity       : %.4f", specificity)
    logger.info("  Youden J          : %.4f", youden_j)
    logger.info("  F1-score (macro)  : %.4f", f1_macro)
    logger.info("  AUROC             : %.4f", auroc)
    logger.info("  Avg Precision     : %.4f", avg_prec)
    logger.info("  FAR/hr (segment)  : %.4f", far_seg)
    logger.info("  TP=%d FP=%d FN=%d TN=%d", tp, fp, fn, tn)

    return {
        "accuracy":          round(accuracy, 6),
        "precision":         round(prec, 6),
        "recall":            round(recall, 6),
        "specificity":       round(specificity, 6),
        "youden_j":          round(youden_j, 6),
        "f1_macro":          round(f1_macro, 6),
        "auroc":             round(auroc, 6),
        "average_precision": round(avg_prec, 6),
        "far_per_hour_seg":  round(far_seg, 6),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ---------------------------------------------------------------------------
# run_postprocessing_evaluations
# ---------------------------------------------------------------------------
def run_postprocessing_evaluations(y_true, y_prob, logger):
    """Run all three evaluation rows and compute optimal threshold.

    Returns (row1_metrics, row2_metrics, row3_metrics,
             post_row2, post_row3, far_row2, far_row3,
             thresh_result, optimal_threshold).
    """
    # -- Row 1: raw at 0.5 ----------------------------------------------------
    y_pred_row1 = (y_prob >= 0.5).astype(int)
    row1_metrics = compute_all_metrics(y_true, y_pred_row1, y_prob, SEGMENT_SEC, logger,
                                       label="Row1_raw_0.5")
    row1_metrics["threshold"] = 0.5
    row1_metrics["postprocessed"] = False

    # -- Find optimal threshold ------------------------------------------------
    # Youden J statistic (J = sensitivity + specificity - 1) is chosen as the
    # threshold selection objective for three reasons:
    #   1. Symmetry: it penalises missed seizures (low sensitivity) and false
    #      alarms (low specificity) equally, which matches the clinical priority
    #      of seizure detection where both under- and over-detection carry cost.
    #   2. Geometric meaning: J is the vertical distance from the ROC diagonal
    #      to the operating point, so maximising J selects the point on the ROC
    #      curve farthest from chance.
    #   3. Independence from prevalence: unlike F1, J does not depend on the
    #      positive predictive value, which is inflated or deflated by class
    #      imbalance. This makes it stable across datasets with different
    #      seizure-to-background ratios.
    # The alternative (F1) weights false negatives more heavily than false
    # positives via the precision term, which may not reflect the clinical
    # balance required in continuous EEG monitoring.
    # Always compute fresh from current model predictions — a cached threshold
    # from a previous run would be stale if the model weights changed.
    thresh_result = find_optimal_threshold(y_true, y_prob, objective="youden")
    optimal_threshold = thresh_result["optimal_threshold"]
    with open(THRESH_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "model": MODEL_NAME,
            "timestamp": datetime.datetime.now().isoformat(),
            "optimal_threshold": optimal_threshold,
            "objective": "youden",
            "optimal_score": thresh_result["optimal_score"],
            "sensitivity_at_opt": thresh_result["sensitivity_at_opt"],
            "specificity_at_opt": thresh_result["specificity_at_opt"],
            "post_processing": {
                "smoothing_window": SMOOTHING_WIN,
                "refractory_period_sec": REFRACTORY_SEC,
                "min_event_duration_sec": MIN_EVENT_SEC,
                "step_sec": STEP_SEC,
            },
        }, f, indent=2)
    logger.info("Optimal threshold computed and saved: %.4f", optimal_threshold)

    # -- Row 2: post-processed at 0.5 -----------------------------------------
    # segment_predictions_to_events(y_pred, y_prob, segment_len_sec, step_sec, ...)
    post_row2 = segment_predictions_to_events(
        y_pred=(y_prob >= 0.5).astype(int),
        y_prob=y_prob,
        segment_len_sec=SEGMENT_SEC,
        step_sec=STEP_SEC,
        min_event_duration_sec=MIN_EVENT_SEC,
        refractory_period_sec=REFRACTORY_SEC,
        smoothing_window=SMOOTHING_WIN,
        threshold=0.5)
    # compute_event_level_far(y_true_segments, post_processed, step_sec, segment_len_sec)
    far_row2 = compute_event_level_far(
        y_true_segments=y_true,
        post_processed=post_row2,
        step_sec=STEP_SEC,
        segment_len_sec=SEGMENT_SEC)

    y_pred_row2 = post_row2["smoothed_preds"]
    row2_metrics = compute_all_metrics(y_true, y_pred_row2, y_prob, SEGMENT_SEC, logger,
                                       label="Row2_postproc_0.5")
    row2_metrics["threshold"] = 0.5
    row2_metrics["postprocessed"] = True
    row2_metrics["far_per_hour_event"] = round(float(far_row2["far_per_hour"]), 6)
    row2_metrics["n_true_alarms"] = far_row2["n_true_alarms"]
    row2_metrics["n_false_alarms"] = far_row2["n_false_alarms"]
    row2_metrics["n_total_events"] = far_row2["n_total_events"]
    logger.info("Row2 events: %d total | %d true | %d false | FAR/hr=%.4f",
                far_row2["n_total_events"], far_row2["n_true_alarms"],
                far_row2["n_false_alarms"], far_row2["far_per_hour"])

    # -- Row 3: post-processed at optimal threshold ----------------------------
    post_row3 = segment_predictions_to_events(
        y_pred=(y_prob >= optimal_threshold).astype(int),
        y_prob=y_prob,
        segment_len_sec=SEGMENT_SEC,
        step_sec=STEP_SEC,
        min_event_duration_sec=MIN_EVENT_SEC,
        refractory_period_sec=REFRACTORY_SEC,
        smoothing_window=SMOOTHING_WIN,
        threshold=optimal_threshold)
    far_row3 = compute_event_level_far(
        y_true_segments=y_true,
        post_processed=post_row3,
        step_sec=STEP_SEC,
        segment_len_sec=SEGMENT_SEC)

    y_pred_row3 = post_row3["smoothed_preds"]
    row3_metrics = compute_all_metrics(y_true, y_pred_row3, y_prob, SEGMENT_SEC, logger,
                                       label="Row3_postproc_opt%.3f" % optimal_threshold)
    row3_metrics["threshold"] = optimal_threshold
    row3_metrics["postprocessed"] = True
    row3_metrics["far_per_hour_event"] = round(float(far_row3["far_per_hour"]), 6)
    row3_metrics["n_true_alarms"] = far_row3["n_true_alarms"]
    row3_metrics["n_false_alarms"] = far_row3["n_false_alarms"]
    row3_metrics["n_total_events"] = far_row3["n_total_events"]
    logger.info("Row3 events: %d total | %d true | %d false | FAR/hr=%.4f",
                far_row3["n_total_events"], far_row3["n_true_alarms"],
                far_row3["n_false_alarms"], far_row3["far_per_hour"])

    return (row1_metrics, row2_metrics, row3_metrics,
            post_row2, post_row3, far_row2, far_row3,
            thresh_result, optimal_threshold)


# ---------------------------------------------------------------------------
# save_all_results
# ---------------------------------------------------------------------------
def save_all_results(history, row1_metrics, row2_metrics, row3_metrics,
                     far_row2, far_row3, hp, best_epoch, best_val_f1,
                     elapsed, device, n_params, y_true,
                     y_pred_row1, y_pred_row2, y_pred_row3, logger):
    """Save all structured results to files (no figures)."""

    # -- a. Training log JSON --------------------------------------------------
    rf = 2 * (2 ** int(hp["num_layers"]) - 1) * (int(hp["kernel_size"]) - 1) + 1
    train_log = {
        "model": MODEL_NAME,
        "timestamp": datetime.datetime.now().isoformat(),
        "total_epochs": len(history["epoch"]),
        "best_epoch": best_epoch,
        "best_val_f1": round(best_val_f1, 6),
        "early_stopped": len(history["epoch"]) < MAX_EPOCHS,
        "duration_seconds": round(elapsed.total_seconds(), 1),
        "hyperparameters": hp,
        "receptive_field": {"samples": rf, "seconds": round(rf / FS, 4)},
        "trainable_params": n_params,
        "device": str(device),
        "history": history,
    }
    with open(TRAIN_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(train_log, f, indent=2)
    logger.info("Saved: %s", TRAIN_LOG_PATH)

    # -- b. Evaluation report JSON ---------------------------------------------
    eval_report = {
        "model": MODEL_NAME,
        "timestamp": datetime.datetime.now().isoformat(),
        "weights_path": str(WEIGHTS_PATH),
        "evaluation_set": "validation",
        "note": "Test set reserved for final_evaluation.py",
        "post_processing_params": {
            "smoothing_window": SMOOTHING_WIN,
            "refractory_period_sec": REFRACTORY_SEC,
            "min_event_duration_sec": MIN_EVENT_SEC,
            "step_sec": STEP_SEC,
        },
        "row1_raw_threshold_0_5": row1_metrics,
        "row2_postproc_threshold_0_5": row2_metrics,
        "row3_postproc_optimal_thresh": row3_metrics,
    }
    with open(EVAL_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)
    logger.info("Saved: %s", EVAL_REPORT_PATH)

    # -- c. Epoch metrics CSV --------------------------------------------------
    with open(EPOCH_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_f1", "lr"])
        writer.writeheader()
        for i in range(len(history["epoch"])):
            writer.writerow({
                "epoch": history["epoch"][i],
                "train_loss": round(history["train_loss"][i], 6),
                "val_f1": round(history["val_f1"][i], 6),
                "lr": history["lr"][i],
            })
    logger.info("Saved: %s", EPOCH_CSV)

    # -- d. Three-row summary CSV (paper Table 1 for M1) -----------------------
    fieldnames_3r = [
        "row", "threshold", "postprocessed",
        "accuracy", "precision", "recall", "specificity", "youden_j",
        "f1_macro", "auroc", "average_precision",
        "far_per_hour_seg", "far_per_hour_event",
        "n_true_alarms", "n_false_alarms", "n_total_events",
        "tp", "tn", "fp", "fn",
    ]
    with open(THREE_ROW_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_3r)
        writer.writeheader()
        for idx, (label, m) in enumerate([
            ("Row1_raw_0.5", row1_metrics),
            ("Row2_postproc_0.5", row2_metrics),
            ("Row3_postproc_opt", row3_metrics),
        ], 1):
            writer.writerow({
                "row": label,
                "threshold": m.get("threshold", ""),
                "postprocessed": m.get("postprocessed", False),
                "accuracy": m.get("accuracy", ""),
                "precision": m.get("precision", ""),
                "recall": m.get("recall", ""),
                "specificity": m.get("specificity", ""),
                "youden_j": m.get("youden_j", ""),
                "f1_macro": m.get("f1_macro", ""),
                "auroc": m.get("auroc", ""),
                "average_precision": m.get("average_precision", ""),
                "far_per_hour_seg": m.get("far_per_hour_seg", ""),
                "far_per_hour_event": m.get("far_per_hour_event", "N/A"),
                "n_true_alarms": m.get("n_true_alarms", "N/A"),
                "n_false_alarms": m.get("n_false_alarms", "N/A"),
                "n_total_events": m.get("n_total_events", "N/A"),
                "tp": m.get("tp", ""), "tn": m.get("tn", ""),
                "fp": m.get("fp", ""), "fn": m.get("fn", ""),
            })
    logger.info("Saved: %s", THREE_ROW_CSV)

    # -- e. Per-row sklearn classification reports -----------------------------
    for row_idx, (y_pred_row, row_label) in enumerate([
        (y_pred_row1, "row1"), (y_pred_row2, "row2"), (y_pred_row3, "row3"),
    ], 1):
        report_dict = classification_report(
            y_true, y_pred_row,
            target_names=["Non-ictal", "Ictal"],
            output_dict=True, zero_division=0)
        report_path = OUTPUT_ROOT / ("tcn_classification_report_%s.json" % row_label)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)
        logger.info("Saved: %s", report_path)

    # -- f. Event detail CSVs --------------------------------------------------
    for far_result, row_label in [(far_row2, "row2"), (far_row3, "row3")]:
        csv_path = OUTPUT_ROOT / ("tcn_event_details_%s.csv" % row_label)
        details = far_result.get("event_details", [])
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["start_sec", "end_sec", "duration_sec", "is_true_alarm"])
            writer.writeheader()
            for evt in details:
                writer.writerow({
                    "start_sec": evt["start_sec"],
                    "end_sec": evt["end_sec"],
                    "duration_sec": evt["duration_sec"],
                    "is_true_alarm": evt["is_true_alarm"],
                })
        logger.info("Saved: %s (%d events)", csv_path, len(details))


# ---------------------------------------------------------------------------
# plot_all_figures
# ---------------------------------------------------------------------------
def plot_all_figures(history, best_epoch, best_val_f1,
                    y_true, y_prob,
                    y_pred_row1, y_pred_row2, y_pred_row3,
                    row1_metrics, row2_metrics, row3_metrics,
                    post_row2, post_row3,
                    thresh_result, optimal_threshold, logger):
    """Produce and save all 12 figures at dpi=150."""

    epochs = history["epoch"]

    # -- Figure 1: Training curves ---------------------------------------------
    # PURPOSE: Demonstrates model convergence behaviour. The left panel shows
    # training loss declining over epochs, confirming the optimiser is reducing
    # the objective. The right panel shows validation macro F1, the metric that
    # drives early stopping. Together they reveal whether the model overfit
    # (loss drops but F1 plateaus/declines), underfit (both remain poor), or
    # converged healthily (loss drops and F1 rises then stabilises). The
    # vertical dashed line marks the epoch whose weights are saved as final.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(epochs, history["train_loss"], color="#5A7DC8", linewidth=1.2, label="Train loss")
    axes[0].axvline(best_epoch, linestyle="--", color="#C85A5A", alpha=0.7, label="Best epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCEWithLogitsLoss")
    axes[0].set_title("Training loss")
    axes[0].legend(fontsize=9)
    axes[1].plot(epochs, history["val_f1"], color="#5A7DC8", linewidth=1.2, label="Val F1")
    axes[1].axvline(best_epoch, linestyle="--", color="#C85A5A", alpha=0.7, label="Best epoch")
    axes[1].annotate("%.4f" % best_val_f1, xy=(best_epoch, best_val_f1),
                     xytext=(5, -15), textcoords="offset points", fontsize=9, color="#C85A5A")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1-score")
    axes[1].set_title("Validation macro F1")
    axes[1].legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "tcn_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", FIGURE_DIR / "tcn_training_curves.png")

    # -- Figure 2: LR schedule ------------------------------------------------
    # PURPOSE: Verifies the cosine annealing schedule behaved as described in
    # the Methods section. Reviewers check that the LR decay matches the
    # reported schedule. A smooth half-cosine from initial LR to eta_min
    # confirms correct configuration. Any flat segment at the end indicates
    # early stopping terminated before the full cosine cycle completed.
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(epochs, history["lr"], color="#5A7DC8", linewidth=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate (log scale)")
    ax.set_title("Cosine annealing learning rate schedule")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "tcn_lr_schedule.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", FIGURE_DIR / "tcn_lr_schedule.png")

    # -- Figures 3-5: Confusion matrices per row -------------------------------
    # PURPOSE: One confusion matrix per evaluation row shows TP, FP, FN, TN
    # counts. Comparing Row 1 to Row 2 isolates the effect of post-processing
    # (smoothing + merging + min-duration filter) at the same threshold.
    # Comparing Row 2 to Row 3 isolates the effect of threshold optimisation.
    # The title of each matrix includes sensitivity, specificity, F1, and
    # Youden J so the reader can assess all four quantities at a glance.
    for row_idx, (y_pred_row, m, row_label, thresh) in enumerate([
        (y_pred_row1, row1_metrics, "Row1: raw", 0.5),
        (y_pred_row2, row2_metrics, "Row2: post-proc", 0.5),
        (y_pred_row3, row3_metrics, "Row3: post-proc", optimal_threshold),
    ], 1):
        cm = confusion_matrix(y_true, y_pred_row, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Non-ictal", "Ictal"],
                    yticklabels=["Non-ictal", "Ictal"],
                    linewidths=0.5, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        pp_str = "post-proc" if m.get("postprocessed", False) else "raw"
        ax.set_title("%s | t=%.3f | Sens=%.3f Spec=%.3f F1=%.3f J=%.3f" % (
            row_label, thresh, m["recall"], m["specificity"], m["f1_macro"], m["youden_j"]))
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / ("tcn_confusion_matrix_row%d.png" % row_idx),
                    dpi=150, bbox_inches="tight")
        plt.close()
    logger.info("Saved: confusion matrices (3 figures)")

    # -- Figure 6: ROC curve ---------------------------------------------------
    # PURPOSE: Displays the trade-off between sensitivity and false positive
    # rate across all possible thresholds. AUROC (area under this curve) is a
    # threshold-invariant summary of discriminative ability. The three scatter
    # points (R1, R2, R3) mark the actual operating points chosen for each
    # evaluation row, showing where each configuration sits on the sensitivity-
    # specificity trade-off. The diagonal dashed line represents chance (AUROC=0.5).
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc_val = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="#5A7DC8", linewidth=1.5, label="TCN (AUROC = %.4f)" % auroc_val)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5, label="Chance")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#5A7DC8")
    # Operating points
    for m, lbl, marker in [
        (row1_metrics, "R1", "o"), (row2_metrics, "R2", "s"), (row3_metrics, "R3", "D"),
    ]:
        fpr_pt = 1 - m["specificity"]
        tpr_pt = m["recall"]
        ax.scatter([fpr_pt], [tpr_pt], marker=marker, s=60, zorder=5, label=lbl)
    ax.set_xlabel("False positive rate (1 - Specificity)")
    ax.set_ylabel("True positive rate (Sensitivity)")
    ax.set_title("ROC Curve -- Validation set")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "tcn_roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", FIGURE_DIR / "tcn_roc_curve.png")

    # -- Figure 7: Threshold curve ---------------------------------------------
    # PURPOSE: Directly answers the reviewer question "how was the threshold
    # selected?" by plotting Youden J as a function of threshold from 0.1 to
    # 0.9. The peak of this curve is the optimal threshold, marked with a
    # vertical dashed line. A broad, flat peak suggests the model is robust
    # to threshold choice; a narrow spike suggests sensitivity to the exact
    # value. This plot is computed on the validation set only -- the selected
    # threshold is applied without modification to the test set.
    curve = thresh_result.get("threshold_curve", {})
    if curve:
        thresholds_list = sorted(curve.keys())
        scores_list = [curve[t] for t in thresholds_list]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(thresholds_list, scores_list, color="#5A7DC8", linewidth=1.2)
        ax.axvline(optimal_threshold, linestyle="--", color="#C85A5A", linewidth=1.2)
        ax.annotate("Optimal t=%.3f" % optimal_threshold,
                    xy=(optimal_threshold, max(scores_list) if scores_list else 0),
                    xytext=(10, -10), textcoords="offset points", fontsize=9, color="#C85A5A")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Youden J statistic")
        ax.set_title("Threshold selection -- Youden J vs threshold")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "tcn_threshold_curve.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved: %s", FIGURE_DIR / "tcn_threshold_curve.png")

    # -- Figure 8: Metrics comparison ------------------------------------------
    # PURPOSE: A grouped bar chart providing a single-figure summary of all
    # seven classification metrics across all three evaluation rows. The top
    # panel lets the reader see at a glance whether post-processing and
    # threshold optimisation improved recall, specificity, and F1 relative
    # to the raw baseline. The bottom panel compares FAR/hr, the primary
    # clinical metric, across segment-level (Row 1) and event-level
    # (Rows 2, 3) computation. The transition from segment-level to
    # event-level FAR/hr typically shows a large reduction because post-
    # processing collapses multiple consecutive FP segments into one event.
    metric_names = ["accuracy", "precision", "recall", "specificity", "f1_macro", "auroc", "average_precision"]
    r1_vals = [row1_metrics.get(m, 0) for m in metric_names]
    r2_vals = [row2_metrics.get(m, 0) for m in metric_names]
    r3_vals = [row3_metrics.get(m, 0) for m in metric_names]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 7),
                                          gridspec_kw={"height_ratios": [7, 3]})
    x_pos = np.arange(len(metric_names))
    w = 0.25
    bars1 = ax_top.bar(x_pos - w, r1_vals, w, color="#E8A87C", label="Row1: raw t=0.5")
    bars2 = ax_top.bar(x_pos, r2_vals, w, color="#5A7DC8", label="Row2: post-proc t=0.5")
    bars3 = ax_top.bar(x_pos + w, r3_vals, w, color="#41B3A3", label="Row3: post-proc t=opt")
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax_top.annotate("%.2f" % h, xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", fontsize=6)
    ax_top.set_xticks(x_pos)
    ax_top.set_xticklabels(metric_names, fontsize=8)
    ax_top.set_ylabel("Score")
    ax_top.set_title("Evaluation metrics across three configurations")
    ax_top.legend(fontsize=8)
    ax_top.set_ylim(0, 1.15)

    # FAR comparison panel
    far_labels = ["Row1\nseg-level", "Row2\nevent-level", "Row3\nevent-level"]
    far_vals = [
        row1_metrics.get("far_per_hour_seg", 0),
        row2_metrics.get("far_per_hour_event", 0),
        row3_metrics.get("far_per_hour_event", 0),
    ]
    far_colors = ["#C85A5A", "#5A7DC8", "#41B3A3"]
    bars_far = ax_bot.bar(far_labels, far_vals, color=far_colors)
    for bar in bars_far:
        h = bar.get_height()
        ax_bot.annotate("%.2f" % h, xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8)
    ax_bot.set_ylabel("FAR/hr")
    ax_bot.set_title("False alarm rate per hour")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "tcn_metrics_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", FIGURE_DIR / "tcn_metrics_comparison.png")

    # -- Figure 9: PR curve ----------------------------------------------------
    # PURPOSE: The Precision-Recall curve is more informative than the ROC
    # curve for imbalanced datasets (Davis & Goadrich, 2006) because it is
    # sensitive to the positive class prevalence. AUROC can appear optimistic
    # when the negative class vastly outnumbers the positive class, because
    # large numbers of true negatives inflate the true negative rate. The
    # no-skill baseline (horizontal dashed line at prevalence) shows what a
    # random classifier would achieve. Average Precision (AP) summarises the
    # area under this curve. A model with high AUROC but low AP has poor
    # positive predictive value despite good discrimination.
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
    avg_prec_val = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    prevalence = np.mean(y_true)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rec_arr, prec_arr, color="#5A7DC8", linewidth=1.5,
            label="TCN (AP = %.4f)" % avg_prec_val)
    ax.axhline(prevalence, linestyle="--", color="gray", alpha=0.5, label="No-skill (%.3f)" % prevalence)
    ax.fill_between(rec_arr, prec_arr, alpha=0.08, color="#5A7DC8")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve -- Validation set")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "tcn_pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", FIGURE_DIR / "tcn_pr_curve.png")

    # -- Figure 10: Calibration curve ------------------------------------------
    # PURPOSE: Assesses whether predicted probabilities are well-calibrated,
    # i.e., whether a segment predicted at p=0.7 is truly ictal ~70% of the
    # time. Points above the diagonal indicate under-confidence (model is
    # better than it thinks); points below indicate over-confidence. Poor
    # calibration does not affect ranking metrics (AUROC, AP) but does affect
    # the clinical meaning of the predicted probability -- a clinician
    # interpreting p=0.8 as "80% chance of seizure" needs calibrated outputs.
    # If calibration is poor, Platt scaling or isotonic regression may be
    # applied as a post-hoc fix before deployment.
    fig, ax = plt.subplots(figsize=(5, 5))
    try:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
        ax.plot(mean_pred, frac_pos, "o-", color="#5A7DC8", label="TCN")
    except ValueError:
        pass
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve (reliability diagram)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "tcn_calibration_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", FIGURE_DIR / "tcn_calibration_curve.png")

    # -- Figure 11: FAR comparison bar chart -----------------------------------
    # PURPOSE: Highlights the distinction between segment-level and event-level
    # false alarm rates. Segment-level FAR (Row 1) counts every individual FP
    # segment as a separate alarm, which inflates the rate because a single
    # noise transient can trigger 2-3 consecutive FP segments due to 50%
    # overlap. Event-level FAR (Rows 2, 3) collapses consecutive FP segments
    # into a single alarm event after post-processing, yielding a clinically
    # realistic alarm rate. The visual drop from Row 1 to Rows 2/3 quantifies
    # the clinical benefit of the post-processing pipeline.
    fig, ax = plt.subplots(figsize=(7, 4))
    far_labels = ["Row1\nsegment-level", "Row2\nevent-level", "Row3\nevent-level"]
    far_vals = [
        row1_metrics.get("far_per_hour_seg", 0),
        row2_metrics.get("far_per_hour_event", 0),
        row3_metrics.get("far_per_hour_event", 0),
    ]
    far_colors = ["#C85A5A", "#5A7DC8", "#41B3A3"]
    bars = ax.bar(far_labels, far_vals, color=far_colors, edgecolor="white")
    for bar in bars:
        h = bar.get_height()
        ax.annotate("%.2f" % h, xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    ax.set_ylabel("False alarm rate per hour")
    ax.set_title("FAR/hr: segment-level vs event-level")
    ax.text(0.02, 0.95, "Segment-level: FP segments / non-ictal hours\n"
                         "Event-level: false alarm events / non-ictal hours",
            transform=ax.transAxes, fontsize=7, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "tcn_far_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", FIGURE_DIR / "tcn_far_comparison.png")

    # -- Figure 12: Segment length analysis ------------------------------------
    # PURPOSE: Justifies the MIN_EVENT_SEC and REFRACTORY_SEC post-processing
    # parameters. The left histogram shows the distribution of detected event
    # durations; events to the left of the MIN_EVENT_SEC dashed line were
    # discarded by the minimum-duration filter, confirming that the filter
    # removes only short-duration artefacts. The right scatter plot shows each
    # event at its start time vs its duration, colour-coded by true/false alarm
    # status. Clusters of short red (false alarm) events would suggest the
    # min-duration filter is not aggressive enough; long green (true alarm)
    # events confirm genuine seizure detections. This directly addresses
    # reviewer questions about post-processing parameter sensitivity.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Combine events from rows 2 and 3
    all_durations = []
    for post in [post_row2, post_row3]:
        for evt in post.get("events", []):
            all_durations.append(evt.get("duration_sec", 0))
    if all_durations:
        axes[0].hist(all_durations, bins=20, color="#5A7DC8", edgecolor="white", alpha=0.85)
        axes[0].axvline(MIN_EVENT_SEC, linestyle="--", color="#C85A5A", linewidth=1.2,
                        label="Min duration = %.0fs" % MIN_EVENT_SEC)
        axes[0].legend(fontsize=8)
    axes[0].set_xlabel("Event duration (s)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of detected event durations")

    # Row 3 event timeline
    events_r3 = far_row3_details = [evt for evt in
                                     compute_event_level_far(
                                         y_true, post_row3, STEP_SEC, SEGMENT_SEC
                                     ).get("event_details", [])]
    if events_r3:
        starts = [e["start_sec"] for e in events_r3]
        durations = [e["duration_sec"] for e in events_r3]
        colors = ["#41B3A3" if e["is_true_alarm"] else "#C85A5A" for e in events_r3]
        axes[1].scatter(starts, durations, c=colors, s=30, alpha=0.7)
        axes[1].axhline(MIN_EVENT_SEC, linestyle="--", color="gray", alpha=0.5)
        # Manual legend
        axes[1].scatter([], [], c="#41B3A3", label="True alarm")
        axes[1].scatter([], [], c="#C85A5A", label="False alarm")
        axes[1].legend(fontsize=8)
    axes[1].set_xlabel("Event start time (s)")
    axes[1].set_ylabel("Event duration (s)")
    axes[1].set_title("Event timeline -- Row 3 (optimal threshold)")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "tcn_segment_length_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", FIGURE_DIR / "tcn_segment_length_analysis.png")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    """Main entry point for TCN baseline training.

    Trains for MAX_EPOCHS=100 epochs with early stopping. Saves all outputs.
    Does not evaluate the test set.
    """
    # -- Step 1: Logging and setup ---------------------------------------------
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("TCN.py")
    logger.info("Timestamp       : %s", datetime.datetime.now().isoformat())
    logger.info("Ablation role   : M1 -- TCN baseline")
    logger.info("Params source   : best_params.json")
    logger.info("Purpose         : Final model training")
    logger.info("MAX_EPOCHS      : %d", MAX_EPOCHS)
    logger.info("ES_PATIENCE     : %d", ES_PATIENCE)
    logger.info("Test set        : NOT loaded")
    logger.info("=" * 60)

    set_seed(SEED)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logger.info("GPU  : %s", torch.cuda.get_device_name(0))
        logger.info("VRAM : %.2f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)
        logger.info("CUDA : %s", torch.version.cuda)
    else:
        logger.info("Device: CPU")
    logger.info("PyTorch: %s", torch.__version__)

    # -- Step 2: Load inputs ---------------------------------------------------
    config, hp = load_best_params(logger)
    train_pairs, val_pairs = load_splits(logger)

    # -- Corpus preparation ----------------------------------------------------
    # Downsampling and extreme-segment filtering are handled offline by
    # create_balanced_splits.py. The manifest is already clean.
    # Subject exclusion (m254), 1:4 downsampling, and extreme-segment filtering
    # are ALL handled offline by create_balanced_splits.py. The manifest is
    # already clean and balanced -- no further corpus preparation is needed here.
    # train_pairs = filter_unpaired_subjects(train_pairs, logger=logger)
    logger.info("Training corpus: %d segments (from balanced manifest)", len(train_pairs))
    pos_weight = torch.tensor([1.0], dtype=torch.float32)
    # -- End corpus preparation ------------------------------------------------

    # -- Step 3: Build model ---------------------------------------------------
    model = build_model(hp, DEVICE, logger)
    n_params = count_parameters(model)

    # -- Step 4: Build data loaders --------------------------------------------
    batch_size = int(hp["batch_size"])
    train_loader = make_loader(train_pairs, batch_size, True, DEVICE)
    val_loader = make_loader(val_pairs, batch_size, False, DEVICE)
    logger.info("Train loader: %d batches", len(train_loader))
    logger.info("Val loader  : %d batches", len(val_loader))

    # -- Step 5: Build training components -------------------------------------
    optimiser, scheduler, criterion = build_training_components(
        model, hp, pos_weight, DEVICE, logger)

    # -- Step 6: Initialise training state -------------------------------------
    history = {"epoch": [], "train_loss": [], "val_f1": [], "lr": []}
    best_val_f1 = 0.0
    epochs_no_imp = 0
    best_state = None                          # CPU copy of best-epoch weights
    best_epoch = 0
    start_epoch = 1
    training_start = datetime.datetime.now()

    # -- Resume from checkpoint if available -----------------------------------
    # latest.pt contains everything: current model/optimiser/scheduler state,
    # best-epoch weights, patience counter, and best_val_f1. A single file
    # load restores the exact training state. tcn_best.pt is a fallback if
    # latest.pt was corrupted (e.g., Slurm killed during torch.save).
    # To start fresh, delete the checkpoints/ directory.
    latest_path = CKPT_DIR / "tcn_latest.pt"
    best_ckpt_path = CKPT_DIR / "tcn_best.pt"
    resume_path = latest_path if latest_path.exists() else (
        best_ckpt_path if best_ckpt_path.exists() else None)

    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimiser.load_state_dict(ckpt["optimiser_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        best_val_f1 = ckpt.get("best_val_f1", ckpt.get("val_f1", 0.0))
        best_epoch = ckpt.get("best_epoch", ckpt.get("epoch", 0))
        start_epoch = ckpt["epoch"] + 1
        epochs_no_imp = ckpt.get("epochs_no_imp", 0)
        # Restore best-epoch weights (stored inside latest.pt)
        if "best_model_state" in ckpt:
            best_state = {k: v.cpu().clone() for k, v in ckpt["best_model_state"].items()}
        else:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        logger.info("RESUMED from %s: epoch %d, best_val_f1=%.4f (epoch %d), patience=%d/%d",
                    resume_path.name, ckpt["epoch"], best_val_f1,
                    best_epoch, epochs_no_imp, ES_PATIENCE)
    else:
        logger.info("No checkpoint found. Starting from epoch 1.")

    # -- Step 7: Training loop -------------------------------------------------
    # Mixed precision (AMP): use FP16 forward/backward on CUDA to leverage
    # Tensor Cores (V100, A100, L40S, T4). GradScaler dynamically adjusts
    # the loss scale to prevent FP16 gradient underflow. On CPU, use_amp is
    # False and all operations remain FP32 — no behavioural change.
    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    logger.info("=" * 65)
    logger.info("TRAINING STARTED")
    logger.info("  Start epoch : %d", start_epoch)
    logger.info("  Max epochs  : %d", MAX_EPOCHS)
    logger.info("  ES patience : %d", ES_PATIENCE)
    logger.info("  Ckpt freq   : every %d", CHECKPOINT_FREQ)
    logger.info("  Mixed prec  : %s", "AMP (FP16)" if use_amp else "FP32")
    logger.info("=" * 65)

    final_epoch = 0
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        final_epoch = epoch

        t0_train = time.time()
        train_loss = train_epoch(model, train_loader, optimiser, criterion, DEVICE, scaler=scaler)
        train_sec = time.time() - t0_train

        t0_val = time.time()
        val_f1, _, _, _ = evaluate_model(model, val_loader, DEVICE, logger, use_amp=use_amp)
        val_sec = time.time() - t0_val

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)

        # Lightweight logging: first, every 10th, early-stop, and new-best epochs
        if epoch == start_epoch or epoch % 10 == 0 or epochs_no_imp >= ES_PATIENCE or val_f1 > best_val_f1:
            logger.info(
                "Epoch %3d/%d | loss=%.4f | val_f1=%.4f | lr=%.2e | best=%.4f | patience=%d/%d"
                " | train %.0fs | val %.0fs",
                epoch, MAX_EPOCHS, train_loss, val_f1, current_lr,
                best_val_f1, epochs_no_imp, ES_PATIENCE, train_sec, val_sec)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_no_imp = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            # Save best checkpoint separately (insurance if latest.pt corrupts)
            save_checkpoint(epoch, model, optimiser, scheduler, val_f1, train_loss, hp,
                            best_ckpt_path, logger,
                            best_model_state=best_state,
                            best_val_f1=best_val_f1, best_epoch=best_epoch,
                            epochs_no_imp=epochs_no_imp)
            logger.info("  New best val F1: %.4f at epoch %d", best_val_f1, best_epoch)
        else:
            epochs_no_imp += 1

        # Save latest.pt every epoch — single file for clean resume
        save_checkpoint(epoch, model, optimiser, scheduler, val_f1, train_loss, hp,
                        latest_path, logger,
                        best_model_state=best_state,
                        best_val_f1=best_val_f1, best_epoch=best_epoch,
                        epochs_no_imp=epochs_no_imp)

        if epoch % CHECKPOINT_FREQ == 0:
            cleanup_checkpoints(CKPT_DIR, KEEP_CKPTS, logger)

        if epochs_no_imp >= ES_PATIENCE:
            logger.info("Early stopping at epoch %d.", epoch)
            break

    # -- Step 8: Restore best weights ------------------------------------------
    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    logger.info("Best weights restored from epoch %d", best_epoch)

    elapsed = datetime.datetime.now() - training_start
    logger.info("=" * 65)
    logger.info("TRAINING COMPLETE")
    logger.info("  Best val F1  : %.4f at epoch %d", best_val_f1, best_epoch)
    logger.info("  Total epochs : %d", final_epoch)
    logger.info("  Duration     : %s", str(elapsed).split(".")[0])
    logger.info("=" * 65)

    # -- Step 9: Save final weights --------------------------------------------
    # Move model to CPU before saving so the .pt file is device-agnostic --
    # it can be loaded on any machine regardless of GPU availability.
    torch.save(model.cpu().state_dict(), WEIGHTS_PATH)
    size_mb = WEIGHTS_PATH.stat().st_size / 1e6
    logger.info("Weights saved : %s (%.2f MB)", WEIGHTS_PATH, size_mb)
    model.to(DEVICE)  # move back to GPU for the final evaluation pass

    # -- Step 10: Final evaluation on validation set ---------------------------
    # Re-evaluate after restoring best weights. This produces the y_true and
    # y_prob arrays needed for post-processing (Rows 1-3) and all figures.
    logger.info("Running final validation evaluation...")
    val_f1_final, y_true, y_pred_05, y_prob = evaluate_model(model, val_loader, DEVICE, logger, use_amp=use_amp)

    # Consistency check: confirm F1 matches the best-epoch value. A mismatch
    # indicates weight restoration failed (e.g., device mismatch, state_dict
    # corruption). Tolerance of 1e-3 accounts for floating-point rounding.
    tol = 1e-3
    if abs(val_f1_final - best_val_f1) > tol:
        logger.warning("F1 mismatch: best=%.6f final=%.6f. Weight restoration may have failed.",
                       best_val_f1, val_f1_final)
    else:
        logger.info("F1 consistency check: PASS")

    # -- Step 11: Post-processing evaluations ----------------------------------
    (row1_metrics, row2_metrics, row3_metrics,
     post_row2, post_row3, far_row2, far_row3,
     thresh_result, optimal_threshold) = run_postprocessing_evaluations(y_true, y_prob, logger)

    # -- Step 12: Save all structured results ----------------------------------
    y_pred_row1 = (y_prob >= 0.5).astype(int)
    y_pred_row2 = post_row2["smoothed_preds"]
    y_pred_row3 = post_row3["smoothed_preds"]

    save_all_results(
        history, row1_metrics, row2_metrics, row3_metrics,
        far_row2, far_row3, hp, best_epoch, best_val_f1,
        elapsed, DEVICE, n_params, y_true,
        y_pred_row1, y_pred_row2, y_pred_row3, logger)

    # -- Step 13: Plot all figures ---------------------------------------------
    plot_all_figures(
        history, best_epoch, best_val_f1,
        y_true, y_prob,
        y_pred_row1, y_pred_row2, y_pred_row3,
        row1_metrics, row2_metrics, row3_metrics,
        post_row2, post_row3,
        thresh_result, optimal_threshold, logger)

    # -- Step 14: Final inventory and cleanup ----------------------------------
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared.")

    logger.info("=" * 65)
    logger.info("ALL OUTPUTS SAVED")
    all_outputs = [
        WEIGHTS_PATH,
        CKPT_DIR / "tcn_latest.pt",
        CKPT_DIR / "tcn_best.pt",
        TRAIN_LOG_PATH,
        EVAL_REPORT_PATH,
        THRESH_PATH,
        EPOCH_CSV,
        THREE_ROW_CSV,
        OUTPUT_ROOT / "tcn_classification_report_row1.json",
        OUTPUT_ROOT / "tcn_classification_report_row2.json",
        OUTPUT_ROOT / "tcn_classification_report_row3.json",
        OUTPUT_ROOT / "tcn_event_details_row2.csv",
        OUTPUT_ROOT / "tcn_event_details_row3.csv",
        FIGURE_DIR / "tcn_training_curves.png",
        FIGURE_DIR / "tcn_lr_schedule.png",
        FIGURE_DIR / "tcn_confusion_matrix_row1.png",
        FIGURE_DIR / "tcn_confusion_matrix_row2.png",
        FIGURE_DIR / "tcn_confusion_matrix_row3.png",
        FIGURE_DIR / "tcn_roc_curve.png",
        FIGURE_DIR / "tcn_threshold_curve.png",
        FIGURE_DIR / "tcn_metrics_comparison.png",
        FIGURE_DIR / "tcn_pr_curve.png",
        FIGURE_DIR / "tcn_calibration_curve.png",
        FIGURE_DIR / "tcn_far_comparison.png",
        FIGURE_DIR / "tcn_segment_length_analysis.png",
    ]
    for p in all_outputs:
        exists = Path(p).exists()
        status = "OK     " if exists else "MISSING"
        logger.info("  [%s] %s", status, p)

    logger.info("=" * 65)
    logger.info("NEXT: run final_evaluation.py when test data is ready.")


if __name__ == "__main__":
    main()


# ======================================================================
# RESEARCH REPORTING GUIDE -- TCN.py
# ======================================================================
#
# -- ARCHITECTURE (Methods) ------------------------------------------------
# "The TCN comprised L stacked dilated causal convolutional blocks with
# exponential dilation d_l = 2^l, yielding a receptive field of
# 2*(2^L - 1)*(k - 1) + 1 samples (X.XX seconds at 500 Hz). Each block
# contained two 1-D causal convolutions at the same dilation, layer
# normalisation, GELU activation, spatial dropout (Dropout1d), and a
# residual skip connection. Global average pooling collapsed the temporal
# dimension before a linear classification head."
# Reference: Bai et al. (2018) arXiv:1803.01271
#
# Parameters to report (from best_params.json):
#   num_layers L    -- determines RF
#   kernel_size k   -- temporal resolution per convolution
#   num_filters     -- model capacity (channel width)
#   dropout p       -- spatial dropout rate (Dropout1d)
#   Total trainable parameters (from count_parameters)
#   RF in samples and seconds at 500 Hz
#
# -- TRAINING (Methods) ----------------------------------------------------
# "The model was trained for up to 100 epochs using AdamW (Loshchilov &
# Hutter, 2019) with learning rate LR and weight decay WD, and a cosine
# annealing schedule (eta_min = LR * 0.01). Early stopping with patience
# 10 was applied, monitoring validation macro F1. Gradient clipping
# (max_norm=1.0) was applied at every step. Class imbalance was addressed
# by offline stratified downsampling of the non-ictal class to a 1:4
# ictal:non-ictal ratio, with pos_weight = 1.0 in BCEWithLogitsLoss."
#
# WHY 100 EPOCHS:
#   100 is a widely used upper bound in EEG deep learning literature
#   (Acharya et al., 2018; Yildirim et al., 2020). It provides enough
#   budget for the cosine annealing schedule to complete a full half-cycle
#   from lr to eta_min, ensuring the learning rate explores both rapid
#   and fine-grained optimisation phases. Early stopping (patience=10)
#   terminates training as soon as the model has converged, so the full
#   100 epochs are rarely exhausted. Increasing to 200 or 300 epochs
#   would not improve results because early stopping would still trigger
#   at approximately the same epoch, but would waste compute if the
#   patience window happened to see a transient improvement.
#
# WHY AdamW (not SGD or Adam):
#   AdamW decouples weight decay from the gradient update, so the
#   regularisation effect remains constant as the learning rate decays
#   under cosine annealing. Standard Adam conflates the two, causing
#   effective regularisation to shrink towards zero as LR approaches
#   eta_min, which can lead to overfitting in the final training phase.
#
# WHY COSINE ANNEALING (not step decay or constant LR):
#   Cosine annealing provides a smooth, monotonic LR decay without
#   abrupt drops that can destabilise training on small, noisy EEG
#   datasets. The gradual reduction allows the model to explore broadly
#   early in training and fine-tune late, without the need to manually
#   choose step-decay milestones.
#
# WHY OFFLINE DOWNSAMPLING + POS_WEIGHT=1.0:
#   The non-ictal class is downsampled offline to a 1:4 ratio, stratified
#   by recording. pos_weight is set to 1.0 because the downsampling is the
#   sole imbalance correction. See create_balanced_splits.py for full
#   methodological justification.
#
# -- POST-PROCESSING (Methods) ---------------------------------------------
# "Raw segment-level sigmoid probabilities were post-processed prior to
# computing event-level false alarm rate:
# (1) A 3-segment moving average was applied to suppress isolated
#     single-segment false positives caused by transient artefacts.
# (2) Consecutive positive segments separated by fewer than 30 seconds
#     were merged into a single event, preventing one seizure from being
#     counted as multiple alarms if the probability briefly dips mid-ictal.
# (3) Events shorter than 10 seconds were discarded, as genuine rodent
#     seizures typically last at least 10 seconds (Luttjohann et al., 2009)."
#
# WHY SMOOTHING_WIN = 3:
#   A 3-segment kernel is the minimum that suppresses isolated 1-segment
#   spikes while introducing only 1 segment of boundary uncertainty
#   (floor(3/2) * 2.5s = 2.5s). Wider windows (5, 7) risk blurring
#   seizure onset/offset boundaries, which would artificially inflate
#   event durations and reduce temporal localisation accuracy.
#
# WHY REFRACTORY_SEC = 30:
#   Rodent seizures can include brief inter-ictal pauses where the
#   probability dips below threshold for a few seconds before resuming.
#   A 30-second refractory period merges these fragmented detections into
#   a single event. This value was chosen to exceed the longest typical
#   inter-burst interval observed in the preprocessing analysis while
#   remaining short enough not to merge genuinely separate seizures.
#
# WHY MIN_EVENT_SEC = 10:
#   Genuine rodent seizures in the UNIQURE dataset last at least 10
#   seconds (Luttjohann et al., 2009). Events shorter than this are
#   overwhelmingly noise artefacts that survived smoothing. Discarding
#   them dramatically reduces the false alarm rate without sacrificing
#   true seizure detections.
#
# -- THRESHOLD SELECTION (Methods) -----------------------------------------
# "The classification threshold was selected by maximising the Youden J
# statistic (J = sensitivity + specificity - 1) on the validation set,
# evaluating thresholds from 0.1 to 0.9 in steps of 0.01. The optimal
# threshold was applied without modification to the test set."
#
# WHY YOUDEN J (not F1 or accuracy):
#   Youden J = sensitivity + specificity - 1. It has three advantages:
#   (1) Symmetry: it penalises missed seizures and false alarms equally,
#       matching the clinical requirement that both under-detection and
#       over-detection carry significant cost in continuous EEG monitoring.
#   (2) Geometric meaning: J is the vertical distance from the ROC
#       diagonal to the operating point, so maximising J selects the
#       point on the ROC curve farthest from chance.
#   (3) Prevalence independence: unlike F1, J does not depend on the
#       positive predictive value, which is inflated/deflated by class
#       imbalance. This makes J stable across datasets with different
#       seizure-to-background ratios.
#   F1, by contrast, penalises false negatives more heavily than false
#   positives through the precision term. Accuracy is dominated by the
#   majority class in imbalanced datasets and is therefore uninformative
#   for threshold selection in seizure detection.
#
# WHY SEARCH RANGE 0.1 to 0.9:
#   Thresholds below 0.1 produce near-all-positive predictions;
#   thresholds above 0.9 produce near-all-negative predictions. Both
#   extremes are clinically useless and numerically degenerate (Youden
#   J approaches -1 or 0). The 0.01 step size provides sufficient
#   granularity for a smooth threshold curve without excessive compute.
#
# -- THREE-ROW TABLE (Results) ---------------------------------------------
# Report tcn_three_row_summary.csv as Table 1 (M1 block):
#   Row 1: threshold=0.5, post-processing=No
#     Baseline raw classifier output at the standard threshold.
#     Allows comparison with prior work that uses 0.5 and no
#     post-processing. FAR/hr is segment-level only.
#   Row 2: threshold=0.5, post-processing=Yes
#     Isolates the contribution of post-processing alone.
#     Difference in FAR/hr between Row 1 and Row 2 quantifies
#     how much of the segment-level alarm burden was noise.
#   Row 3: threshold=optimal, post-processing=Yes
#     Best clinical operating point. Threshold selected on
#     validation set via Youden J. Apply to test set without
#     further adjustment.
# AUROC is identical across rows because it is computed from the
# continuous probability distribution, which is threshold-invariant.
# Report AUROC once per model with a footnote explaining this.
#
# -- FIGURES (Discussion) --------------------------------------------------
# tcn_training_curves.png     : convergence and overfitting assessment
# tcn_lr_schedule.png         : confirms cosine annealing behaved correctly
# tcn_confusion_matrix_*.png  : TP/FP/FN/TN counts per evaluation row
# tcn_roc_curve.png           : discrimination ability + operating points
# tcn_threshold_curve.png     : Youden J vs threshold, justifies selection
# tcn_metrics_comparison.png  : at-a-glance comparison across 3 rows
# tcn_pr_curve.png            : precision-recall for imbalanced assessment
# tcn_calibration_curve.png   : probability calibration (reliability)
# tcn_far_comparison.png      : segment vs event FAR/hr reduction
# tcn_segment_length_analysis : event duration and post-proc. justification
# ======================================================================
