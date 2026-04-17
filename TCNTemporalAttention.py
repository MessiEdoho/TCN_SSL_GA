"""
TCNTemporalAttention.py
=======================
Training script for Model 2: TCN with Temporal Attention.

Mirrors TCN.py in structure. Adapted for the
TCNWithAttention architecture from tcn_utils.py.

Trains for exactly 100 epochs with early stopping
on validation macro F1 (patience 10). ALL parameters
(backbone and attention) are trained jointly.

Hyperparameters loaded from TWO sources:
  outputs/best_params.json
    -> TCN backbone: num_layers, kernel_size,
                    num_filters, dropout
  outputs/best_attention_params.json
    -> Attention : attention_dim, attention_dropout
    -> Training  : learning_rate, weight_decay, batch_size
    -> Also contains tcn_hyperparameters for validation

Architecture note: TCNWithAttention uses a two-layer additive
temporal attention (tanh + linear scorer) with tunable
attention_dim and attention_dropout, matching
MultiScaleTCNWithAttention for consistent ablation.

Three evaluation rows:
  Row 1: raw predictions at threshold 0.5
  Row 2: post-processed at threshold 0.5
  Row 3: post-processed at optimal threshold

The test set is never loaded in this script.
It is reserved for final_evaluation.py.

Ablation position
-----------------
M1 (TCN.py) vs M2 (this script):
  Isolates the contribution of temporal attention
  over the single-branch TCN baseline.
  Both models share identical backbone architecture.

Pipeline position
-----------------
After  : tune_temporal_attention.py
Before : final_evaluation.py

Usage
-----
python TCNTemporalAttention.py

Key outputs
-----------
outputs/TCNAttention/tcn_attention_final_weights.pt
outputs/TCNAttention/tcn_attention_evaluation_report.json
outputs/TCNAttention/tcn_attention_three_row_summary.csv
outputs/TCNAttention/figures/  (13 figures)
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import json                          # read/write JSON config and result files
import logging                       # structured logging to file and stdout
import sys                           # stdout handle for StreamHandler
import csv                           # write epoch metrics and three-row summary CSVs
import time                          # per-epoch train/val timing
import datetime                      # ISO timestamps for JSON records and elapsed time
import shutil                        # reserved for potential file copy operations
from pathlib import Path             # cross-platform path handling throughout the script

import numpy as np                   # array operations for metrics, predictions, attention weights
import torch                         # deep learning framework (model, tensors, GPU)
import torch.nn as nn                # neural network modules (BCEWithLogitsLoss)

import matplotlib                    # plotting backend configuration
matplotlib.use("Agg")               # non-interactive backend -- must be set before importing pyplot
import matplotlib.pyplot as plt      # figure creation and saving for all 13 plots
import seaborn as sns                # heatmap rendering for confusion matrix figures

from sklearn.metrics import (        # scikit-learn metrics for comprehensive evaluation
    accuracy_score,                  # (TP+TN) / total
    precision_score,                 # TP / (TP+FP) -- positive predictive value
    recall_score,                    # TP / (TP+FN) -- sensitivity
    f1_score,                        # harmonic mean of precision and recall
    roc_auc_score,                   # area under ROC curve (threshold-invariant)
    confusion_matrix,                # 2x2 matrix: TN, FP, FN, TP
    roc_curve,                       # FPR vs TPR arrays for ROC plot
    precision_recall_curve,          # precision vs recall arrays for PR plot
    average_precision_score,         # area under PR curve (summary statistic)
    classification_report,           # per-class precision, recall, F1 as dict
    calibration_curve,               # reliability diagram: predicted prob vs true fraction
)

# tcn_utils imports -- exact names confirmed from reading tcn_utils.py.
# run_training() exists but is not used here; an explicit training loop
# provides full control over checkpointing, logging, and per-epoch CSV output.
from tcn_utils import (
    set_seed,                        # fix Python/NumPy/PyTorch seeds for reproducibility
    TCNWithAttention,                # M2 architecture: TCN backbone + two-layer additive attention
    make_loader,                     # build DataLoader (sequential for val, shuffled for train)
    # filter_unpaired_subjects,      # handled offline by create_balanced_splits.py
    train_one_epoch,                 # one epoch: forward + loss + backward + gradient clip + step
    count_parameters,                # sum of requires_grad=True parameter elements
    segment_predictions_to_events,   # post-processing: smooth -> merge -> min-duration filter
    compute_event_level_far,         # event-level false alarm rate per hour
    find_optimal_threshold,          # sweep thresholds 0.1-0.9 maximising Youden J
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED              = 42                                 # global seed (Python, NumPy, PyTorch CPU+CUDA)
# 100 epochs lets cosine annealing complete a full half-cycle. Early stopping
# (patience=10) terminates well before 100 if converged.
MAX_EPOCHS        = 100                                # max training epochs (upper bound)
ES_PATIENCE       = 10                                 # epochs without val F1 improvement before stopping
CHECKPOINT_FREQ   = 5                                  # save periodic checkpoint every N epochs
KEEP_CKPTS        = 3                                  # disk-space cap: keep only 3 most recent periodic ckpts
FS                = 500                                # native EDF sampling rate (Hz)
SEGMENT_LEN       = 2500                               # samples per segment: 5 s * 500 Hz
SEGMENT_SEC       = 5.0                                # segment duration in seconds
STEP_SEC          = 2.5                                # step between segment starts: 50% overlap
MIN_EVENT_SEC     = 10.0                               # discard events shorter than this (seconds)
REFRACTORY_SEC    = 30.0                               # merge events separated by fewer than this (seconds)
SMOOTHING_WIN     = 3                                  # segments in probability smoothing kernel
MODEL_NAME        = "TCNWithTemporalAttention"         # model identifier for filenames and JSON

OUTPUT_ROOT       = Path("/home/people/22206468/scratch/OUTPUT/MODEL2_OUTPUT") / "TCNAttention"             # all M2 outputs here
CKPT_DIR          = OUTPUT_ROOT / "checkpoints"                  # periodic and best checkpoints
LOG_DIR           = OUTPUT_ROOT / "logs"                         # training log
FIGURE_DIR        = OUTPUT_ROOT / "figures"                      # all 13 figures
WEIGHTS_PATH      = OUTPUT_ROOT / "tcn_attention_final_weights.pt"
TRAIN_LOG_PATH    = OUTPUT_ROOT / "tcn_attention_training_log.json"
EVAL_REPORT_PATH  = OUTPUT_ROOT / "tcn_attention_evaluation_report.json"
THRESH_PATH       = OUTPUT_ROOT / "tcn_attention_optimal_threshold.json"
EPOCH_CSV         = OUTPUT_ROOT / "tcn_attention_epoch_metrics.csv"
THREE_ROW_CSV     = OUTPUT_ROOT / "tcn_attention_three_row_summary.csv"

# Two JSON input files -- backbone and attention tuning results
BACKBONE_PARAMS_PATH = Path("/home/people/22206468/scratch/OUTPUT/MODEL1_OUTPUT/TCNtuning_outputs") / "best_params.json"            # from tcn_HPT_binary.ipynb
ATTN_PARAMS_PATH     = Path("/home/people/22206468/scratch/OUTPUT/MODEL2_OUTPUT") / "best_attention_params.json"  # from tune_temporal_attention.py

# data_splits.json -- single source of truth (matches all other pipeline scripts)
# Previous (uniform downsampling): data_splits.json
# SPLITS_PATH = Path("/scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits.json")
# Current (proximity-aware downsampling): data_splits_nonictal_sampled.json
SPLITS_PATH = Path("/scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits_nonictal_sampled.json")

# Backbone attribute prefix in TCNWithAttention (confirmed: self.tcn)
BACKBONE_ATTR = "tcn"                                  # for parameter counting


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------
def setup_logging():
    """Create all output directories and configure the logger.

    Mirror of TCN.py setup_logging() -- logger name and log file adapted.

    Returns
    -------
    logger : logging.Logger
    """
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)   # create outputs/TCNAttention/ if absent
    CKPT_DIR.mkdir(parents=True, exist_ok=True)     # create checkpoints/ subdirectory
    LOG_DIR.mkdir(parents=True, exist_ok=True)      # create logs/ subdirectory
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)   # create figures/ subdirectory

    log_file = LOG_DIR / "TCNTemporalAttention_training.log"  # full DEBUG-level log on disk
    logger = logging.getLogger("TCNAttention_training")       # named logger avoids root conflicts
    logger.setLevel(logging.DEBUG)                  # capture all severity levels
    logger.handlers.clear()                         # prevent duplicate handlers on re-import or resume

    fmt = logging.Formatter(                        # timestamp | level | message format
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")               # human-readable date without microseconds

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")  # append to preserve logs on resume
    fh.setLevel(logging.DEBUG)                      # file gets everything including debug
    fh.setFormatter(fmt)                            # apply the same format to file

    sh = logging.StreamHandler(sys.stdout)          # console output for real-time monitoring
    sh.setLevel(logging.INFO)                       # console gets INFO+ (skip debug noise)
    sh.setFormatter(fmt)                            # same format as file for consistency

    logger.addHandler(fh)                           # attach file handler to logger
    logger.addHandler(sh)                           # attach console handler to logger
    return logger                                   # caller stores this for all subsequent logging


# ---------------------------------------------------------------------------
# load_best_params
# ---------------------------------------------------------------------------
def load_best_params(logger):
    """Load hyperparameters from two JSON files.

    DIFFERENCE FROM TCN.py: loads from two files instead of one.

    1. outputs/best_params.json
       TCN backbone: num_layers, kernel_size, num_filters, dropout.

    2. outputs/best_attention_params.json
       Training HPs tuned on frozen backbone: learning_rate, weight_decay,
       batch_size, max_grad_norm.
       Also contains 'tcn_hyperparameters' for cross-validation.

    Note: TCNWithAttention uses two-layer additive attention with tunable
    attention_dim and attention_dropout from best_attention_params.json.

    Returns
    -------
    tuple of (backbone_config, backbone_hp, attn_config, attn_hp)
    """
    # -- Load backbone params (source 1: best_params.json) ----------------------
    if not BACKBONE_PARAMS_PATH.exists():              # fail fast if prerequisite missing
        logger.error("best_params.json not found at %s. "
                     "Run tcn_HPT_binary.ipynb first.", BACKBONE_PARAMS_PATH)
        raise FileNotFoundError(str(BACKBONE_PARAMS_PATH))

    with open(BACKBONE_PARAMS_PATH, "r", encoding="utf-8") as f:  # read full JSON config
        backbone_config = json.load(f)
    if "hyperparameters" not in backbone_config:       # validate expected schema
        logger.error("'hyperparameters' key missing from %s.", BACKBONE_PARAMS_PATH)
        raise KeyError("hyperparameters")
    backbone_hp = backbone_config["hyperparameters"]   # extract backbone HP sub-dict

    # -- Load attention tuning params (source 2: best_attention_params.json) ---
    if not ATTN_PARAMS_PATH.exists():                  # fail fast if prerequisite missing
        logger.error("best_attention_params.json not found at %s. "
                     "Run tune_temporal_attention.py first.", ATTN_PARAMS_PATH)
        raise FileNotFoundError(str(ATTN_PARAMS_PATH))

    with open(ATTN_PARAMS_PATH, "r", encoding="utf-8") as f:  # read full JSON config
        attn_config = json.load(f)
    if "hyperparameters" not in attn_config:           # validate expected schema
        logger.error("'hyperparameters' key missing from %s.", ATTN_PARAMS_PATH)
        raise KeyError("hyperparameters")
    attn_hp = attn_config["hyperparameters"]           # extract attention HP sub-dict

    # -- Consistency check: backbone in attention JSON must match backbone JSON -
    # tune_temporal_attention.py saves the backbone it used under "tcn_hyperparameters".
    # If these differ from best_params.json, the attention was tuned on a different
    # backbone, which invalidates the M1-vs-M2 ablation.
    saved_bb = attn_config.get("tcn_hyperparameters", None)  # may be absent in old runs
    if saved_bb is not None:
        for key in ["num_layers", "kernel_size", "num_filters", "dropout"]:
            if key in saved_bb and key in backbone_hp:
                if saved_bb[key] != backbone_hp[key]:  # mismatch detected
                    logger.warning(
                        "Backbone param mismatch: %s = %s in attention JSON vs %s in backbone JSON. "
                        "Attention was tuned on a different backbone. Proceeding with backbone JSON.",
                        key, saved_bb[key], backbone_hp[key])

    # -- Log backbone hyperparameters ------------------------------------------
    logger.info("-" * 55)
    logger.info("TCN backbone hyperparameters (from best_params.json):")
    for k, v in backbone_hp.items():
        logger.info("  %-20s: %s", k, v)

    # RF = 2 * (2^L - 1) * (k - 1) + 1
    rf = 2 * (2 ** int(backbone_hp["num_layers"]) - 1) * (int(backbone_hp["kernel_size"]) - 1) + 1
    logger.info("  Receptive field : %d samples (%.3f s at %d Hz)", rf, rf / FS, FS)
    if rf < FS:
        logger.warning("RF %d < %d -- below 1 second. Verify backbone configuration.", rf, FS)

    # -- Log attention/training hyperparameters --------------------------------
    logger.info("-" * 55)
    logger.info("Training hyperparameters (from best_attention_params.json):")
    for k, v in attn_hp.items():
        logger.info("  %-20s: %s", k, v)

    # -- Log tuning reference values -------------------------------------------
    bb_best_f1 = backbone_config.get("best_val_f1", "not recorded")
    attn_best_f1 = attn_config.get("best_val_f1", "not recorded")
    logger.info("-" * 55)
    logger.info("Backbone tuning best val F1 : %s", bb_best_f1)
    logger.info("Attention tuning best val F1: %s", attn_best_f1)
    logger.info("-" * 55)

    return backbone_config, backbone_hp, attn_config, attn_hp


# ---------------------------------------------------------------------------
# load_splits
# ---------------------------------------------------------------------------
def load_splits(logger):
    """Load train and val file-label pairs from data_splits.json.

    Mirror of TCN.py load_splits() -- no differences. Never loads test pairs.

    Returns
    -------
    tuple of (train_pairs, val_pairs)
    """
    if not SPLITS_PATH.exists():
        logger.error("data_splits.json not found at %s. "
                     "Run generate_data_splits.py first.", SPLITS_PATH)
        raise FileNotFoundError(str(SPLITS_PATH))

    logger.info("Loading splits from: %s", SPLITS_PATH)
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:  # read full JSON
        splits = json.load(f)

    # Convert list-of-dicts to list-of-tuples for make_loader() compatibility
    train_pairs = [(rec["filepath"], rec["label"]) for rec in splits["train"]]  # (path, 0/1)
    val_pairs = [(rec["filepath"], rec["label"]) for rec in splits["val"]]      # (path, 0/1)

    if not train_pairs:                                # guard against empty partitions
        logger.error("Train partition is empty.")
        raise RuntimeError("Empty train partition")
    if not val_pairs:                                  # guard against empty partitions
        logger.error("Val partition is empty.")
        raise RuntimeError("Empty val partition")

    # Log class composition for each partition
    for name, pairs in [("TRAIN", train_pairs), ("VAL", val_pairs)]:
        n_total = len(pairs)                           # total segment count
        n_sz = sum(1 for _, l in pairs if l == 1)      # ictal (seizure) segment count
        n_nsz = n_total - n_sz                         # non-ictal segment count
        pct = n_sz / n_total * 100 if n_total > 0 else 0.0  # ictal prevalence percentage
        mouse_ids = sorted({Path(fp).stem.split("_")[0] for fp, _ in pairs})  # unique mouse IDs
        logger.info("%s: %d total | %d seizure | %d non-seizure | %.1f%% ictal | %d mice",
                    name, n_total, n_sz, n_nsz, pct, len(mouse_ids))

    # Check whether test data is available (never loaded here -- reserved for final_evaluation.py)
    test_status = splits.get("metadata", {}).get("test_status", "pending")
    if test_status != "complete":
        logger.info("Test data not yet ready -- test split not loaded here.")

    return train_pairs, val_pairs                      # only train and val returned


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------
def build_model(backbone_hp, attn_hp, device, logger):
    """Instantiate TCNWithAttention using backbone + attention hyperparameters.

    DIFFERENCE FROM TCN.py: uses TCNWithAttention instead of TCN.
    TCNWithAttention.__init__(num_layers, num_filters, kernel_size, dropout,
                              attention_dim=64, attention_dropout=0.0,
                              return_embedding=False, fs=500)
    Backbone params from best_params.json, attention params from
    best_attention_params.json.

    ALL parameters are trainable (no freezing). This is logged explicitly.

    Returns
    -------
    model : nn.Module
    """
    # Fix seed before weight init for reproducibility
    set_seed(SEED)
    model = TCNWithAttention(
        num_layers=int(backbone_hp["num_layers"]),       # L: depth and RF growth
        num_filters=int(backbone_hp["num_filters"]),     # channel width
        kernel_size=int(backbone_hp["kernel_size"]),     # local temporal resolution
        dropout=float(backbone_hp["dropout"]),           # spatial dropout (Dropout1d)
        attention_dim=int(attn_hp["attention_dim"]),     # scoring projection dimension
        attention_dropout=float(attn_hp["attention_dropout"]),  # context vector dropout
        return_embedding=False,                          # classification mode (returns logits)
        fs=FS,                                           # sampling rate for RF logging only
    )
    model = model.to(device)

    # -- Confirm all parameters are trainable (not frozen) ---------------------
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.info("Parameter audit (all should be trainable):")
    logger.info("  Total trainable    : %s", "{:,}".format(n_trainable))
    logger.info("  Frozen (expected 0): %s", "{:,}".format(n_frozen))
    if n_frozen > 0:
        logger.warning("%s params frozen. For final training all should be trainable.",
                       "{:,}".format(n_frozen))

    # -- Backbone vs attention parameter counts --------------------------------
    # self.tcn is the backbone; everything else is attention + head
    backbone_module = getattr(model, BACKBONE_ATTR)
    n_backbone = sum(p.numel() for p in backbone_module.parameters() if p.requires_grad)
    n_attention = n_trainable - n_backbone
    logger.info("  Backbone params    : %s", "{:,}".format(n_backbone))
    logger.info("  Attention + head   : %s", "{:,}".format(n_attention))

    # -- Log RF ----------------------------------------------------------------
    rf = 2 * (2 ** int(backbone_hp["num_layers"]) - 1) * (int(backbone_hp["kernel_size"]) - 1) + 1
    logger.info("  Receptive field    : %d samples (%.3f s)", rf, rf / FS)
    logger.info("Model: %s", MODEL_NAME)
    logger.info("Device: %s", device)

    return model


# ---------------------------------------------------------------------------
# build_training_components
# ---------------------------------------------------------------------------
def build_training_components(model, pos_weight, attn_hp, device, logger):
    """Build optimiser, scheduler, and loss function.

    DIFFERENCE FROM TCN.py: learning_rate, weight_decay come from attn_hp
    (best_attention_params.json) because these were optimised during attention
    tuning. Optimiser covers ALL parameters (backbone + attention jointly).

    Imbalance strategy: offline stratified downsampling to 1:4 ratio with
    pos_weight=1.0.

    Returns (optimiser, scheduler, criterion).
    """
    # AdamW decouples weight decay from gradient update (Loshchilov & Hutter, 2019)
    optimiser = torch.optim.AdamW(
        model.parameters(),                               # ALL params -- no filtering
        lr=float(attn_hp["learning_rate"]),                # from attention tuning
        weight_decay=float(attn_hp["weight_decay"]))       # from attention tuning
    # Cosine annealing: smooth LR decay over T_max epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max=MAX_EPOCHS,
        eta_min=float(attn_hp["learning_rate"]) * 0.01)   # floor at 1% of initial LR
    # Imbalance handled by offline stratified downsampling to 1:4 ratio; pos_weight=1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    logger.info("Optimiser: AdamW (lr=%.2e, wd=%.2e) -- from attention tuning",
                attn_hp["learning_rate"], attn_hp["weight_decay"])
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

    Mirror of TCN.py -- pattern adapted for tcn_attention prefix.
    Never deletes tcn_attention_best.pt.
    """
    pattern = "tcn_attention_epoch_*.pt"
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

    Mirror of TCN.py compute_all_metrics() -- no differences.
    """
    # labels=[0,1] forces 2x2 matrix even if one class absent in y_pred
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)  # Python int for JSON serialisation

    accuracy    = accuracy_score(y_true, y_pred)           # (TP+TN) / total
    prec        = precision_score(y_true, y_pred, pos_label=1, zero_division=0)  # TP / (TP+FP)
    recall      = recall_score(y_true, y_pred, pos_label=1, zero_division=0)     # sensitivity = TP / (TP+FN)
    f1_macro    = f1_score(y_true, y_pred, average="macro", zero_division=0)     # mean of per-class F1
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # TN / (TN+FP)
    # AUROC uses continuous probabilities -- threshold-invariant, same across all 3 rows
    auroc       = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    avg_prec    = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    # Youden J = sensitivity + specificity - 1; ranges [-1, +1]; used for threshold selection
    youden_j    = recall + specificity - 1.0

    # Segment-level FAR/hr: FP segments / non-ictal recording hours
    n_non_ic   = tn + fp                                   # total non-ictal segments (denominator)
    non_ic_hrs = (n_non_ic * segment_sec) / 3600.0         # convert to hours
    far_seg    = fp / non_ic_hrs if non_ic_hrs > 0 else 0.0  # false alarms per hour

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

    Mirror of TCN.py -- paths adapted for TCNAttention output directory.

    Returns (row1_metrics, row2_metrics, row3_metrics,
             post_row2, post_row3, far_row2, far_row3,
             thresh_result, optimal_threshold).
    """
    # -- Row 1: raw predictions at standard threshold 0.5 -----------------------
    # No post-processing. Baseline for comparison with prior work.
    y_pred_row1 = (y_prob >= 0.5).astype(int)          # binarise at default threshold
    row1_metrics = compute_all_metrics(y_true, y_pred_row1, y_prob, SEGMENT_SEC, logger,
                                       label="Row1_raw_0.5")
    row1_metrics["threshold"] = 0.5                    # record threshold used
    row1_metrics["postprocessed"] = False               # flag: no post-processing applied

    # -- Find optimal threshold using Youden J statistic ----------------------
    # Youden J = sensitivity + specificity - 1. Chosen because it treats missed
    # seizures and false alarms symmetrically and is prevalence-independent.
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
    post_row2 = segment_predictions_to_events(
        y_pred=(y_prob >= 0.5).astype(int), y_prob=y_prob,
        segment_len_sec=SEGMENT_SEC, step_sec=STEP_SEC,
        min_event_duration_sec=MIN_EVENT_SEC, refractory_period_sec=REFRACTORY_SEC,
        smoothing_window=SMOOTHING_WIN, threshold=0.5)
    far_row2 = compute_event_level_far(
        y_true_segments=y_true, post_processed=post_row2,
        step_sec=STEP_SEC, segment_len_sec=SEGMENT_SEC)

    y_pred_row2 = post_row2["smoothed_preds"]
    row2_metrics = compute_all_metrics(y_true, y_pred_row2, y_prob, SEGMENT_SEC, logger,
                                       label="Row2_postproc_0.5")
    row2_metrics["threshold"] = 0.5
    row2_metrics["postprocessed"] = True
    row2_metrics["far_per_hour_event"] = round(float(far_row2["far_per_hour"]), 6)
    row2_metrics["n_true_alarms"] = far_row2["n_true_alarms"]
    row2_metrics["n_false_alarms"] = far_row2["n_false_alarms"]
    row2_metrics["n_total_events"] = far_row2["n_total_events"]

    # -- Row 3: post-processed at optimal threshold ----------------------------
    post_row3 = segment_predictions_to_events(
        y_pred=(y_prob >= optimal_threshold).astype(int), y_prob=y_prob,
        segment_len_sec=SEGMENT_SEC, step_sec=STEP_SEC,
        min_event_duration_sec=MIN_EVENT_SEC, refractory_period_sec=REFRACTORY_SEC,
        smoothing_window=SMOOTHING_WIN, threshold=optimal_threshold)
    far_row3 = compute_event_level_far(
        y_true_segments=y_true, post_processed=post_row3,
        step_sec=STEP_SEC, segment_len_sec=SEGMENT_SEC)

    y_pred_row3 = post_row3["smoothed_preds"]
    row3_metrics = compute_all_metrics(y_true, y_pred_row3, y_prob, SEGMENT_SEC, logger,
                                       label="Row3_postproc_opt%.3f" % optimal_threshold)
    row3_metrics["threshold"] = optimal_threshold
    row3_metrics["postprocessed"] = True
    row3_metrics["far_per_hour_event"] = round(float(far_row3["far_per_hour"]), 6)
    row3_metrics["n_true_alarms"] = far_row3["n_true_alarms"]
    row3_metrics["n_false_alarms"] = far_row3["n_false_alarms"]
    row3_metrics["n_total_events"] = far_row3["n_total_events"]

    return (row1_metrics, row2_metrics, row3_metrics,
            post_row2, post_row3, far_row2, far_row3,
            thresh_result, optimal_threshold)


# ---------------------------------------------------------------------------
# save_all_results
# ---------------------------------------------------------------------------
def save_all_results(history, row1_metrics, row2_metrics, row3_metrics,
                     far_row2, far_row3, backbone_hp, attn_hp,
                     best_epoch, best_val_f1, elapsed, device, n_params,
                     y_true, y_pred_row1, y_pred_row2, y_pred_row3, logger):
    """Save all structured results to files (no figures).

    DIFFERENCE FROM TCN.py: accepts backbone_hp and attn_hp separately.
    Includes both in JSON outputs. Adds training_note about joint training.
    """
    rf = 2 * (2 ** int(backbone_hp["num_layers"]) - 1) * (int(backbone_hp["kernel_size"]) - 1) + 1

    # -- a. Training log JSON --------------------------------------------------
    train_log = {
        "model": MODEL_NAME,
        "timestamp": datetime.datetime.now().isoformat(),
        "total_epochs": len(history["epoch"]),
        "best_epoch": best_epoch,
        "best_val_f1": round(best_val_f1, 6),
        "early_stopped": len(history["epoch"]) < MAX_EPOCHS,
        "duration_seconds": round(elapsed.total_seconds(), 1),
        "backbone_hyperparameters": backbone_hp,
        "attention_hyperparameters": attn_hp,
        "backbone_params_source": str(BACKBONE_PARAMS_PATH),
        "attention_params_source": str(ATTN_PARAMS_PATH),
        "training_note": ("All parameters (backbone + attention) trained jointly for 100 epochs. "
                          "Backbone was frozen only during attention tuning "
                          "(tune_temporal_attention.py), not during this final training run."),
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
        "backbone_hyperparameters": backbone_hp,
        "attention_hyperparameters": attn_hp,
        "backbone_params_source": str(BACKBONE_PARAMS_PATH),
        "attention_params_source": str(ATTN_PARAMS_PATH),
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

    # -- d. Three-row summary CSV (M2 in ablation table) -----------------------
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
        for label, m in [
            ("Row1_raw_0.5", row1_metrics),
            ("Row2_postproc_0.5", row2_metrics),
            ("Row3_postproc_opt", row3_metrics),
        ]:
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

    # -- e. Per-row classification reports -------------------------------------
    for y_pred_row, row_label in [
        (y_pred_row1, "row1"), (y_pred_row2, "row2"), (y_pred_row3, "row3"),
    ]:
        report_dict = classification_report(
            y_true, y_pred_row, target_names=["Non-ictal", "Ictal"],
            output_dict=True, zero_division=0)
        rpath = OUTPUT_ROOT / ("tcn_attention_classification_report_%s.json" % row_label)
        with open(rpath, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)
        logger.info("Saved: %s", rpath)

    # -- f. Event detail CSVs --------------------------------------------------
    for far_result, row_label in [(far_row2, "row2"), (far_row3, "row3")]:
        csv_path = OUTPUT_ROOT / ("tcn_attention_event_details_%s.csv" % row_label)
        details = far_result.get("event_details", [])
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["start_sec", "end_sec", "duration_sec", "is_true_alarm"])
            writer.writeheader()
            for evt in details:
                writer.writerow({
                    "start_sec": evt["start_sec"], "end_sec": evt["end_sec"],
                    "duration_sec": evt["duration_sec"], "is_true_alarm": evt["is_true_alarm"],
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
    """Produce and save all 12 standard figures at dpi=150.

    Mirror of TCN.py plot_all_figures() -- titles adapted for M2.
    """
    pfx = "tcn_attention"                              # filename prefix for all 12 figures
    epochs = history["epoch"]                          # x-axis values for training curve plots

    # -- Figure 1: Training curves ---------------------------------------------
    # PURPOSE: Shows convergence behaviour. Left: loss declining confirms
    # optimiser is working. Right: val F1 rising then stabilising confirms
    # generalisation. Vertical dashed line marks the epoch whose weights are saved.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(epochs, history["train_loss"], color="#5A7DC8", linewidth=1.2, label="Train loss")
    axes[0].axvline(best_epoch, linestyle="--", color="#C85A5A", alpha=0.7, label="Best epoch")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("BCEWithLogitsLoss")
    axes[0].set_title("TCN+Attention Training Loss"); axes[0].legend(fontsize=9)
    axes[1].plot(epochs, history["val_f1"], color="#5A7DC8", linewidth=1.2, label="Val F1")
    axes[1].axvline(best_epoch, linestyle="--", color="#C85A5A", alpha=0.7, label="Best epoch")
    axes[1].annotate("%.4f" % best_val_f1, xy=(best_epoch, best_val_f1),
                     xytext=(5, -15), textcoords="offset points", fontsize=9, color="#C85A5A")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Macro F1-score")
    axes[1].set_title("TCN+Attention Validation Macro F1"); axes[1].legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / ("%s_training_curves.png" % pfx), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s_training_curves.png", pfx)

    # -- Figure 2: LR schedule ------------------------------------------------
    # PURPOSE: Verifies cosine annealing behaved as described in Methods.
    # A flat tail indicates early stopping terminated before full cycle.
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(epochs, history["lr"], color="#5A7DC8", linewidth=1.2)
    ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("Learning rate (log scale)")
    ax.set_title("TCN+Attention Cosine Annealing LR")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / ("%s_lr_schedule.png" % pfx), dpi=150, bbox_inches="tight")
    plt.close()

    # -- Figures 3-5: Confusion matrices ---------------------------------------
    # PURPOSE: TP/FP/FN/TN counts per row. Row1-vs-Row2 isolates post-processing.
    # Row2-vs-Row3 isolates threshold optimisation. Titles show key metrics.
    for row_idx, (yp, m, rl, th) in enumerate([
        (y_pred_row1, row1_metrics, "Row1: raw", 0.5),
        (y_pred_row2, row2_metrics, "Row2: post-proc", 0.5),
        (y_pred_row3, row3_metrics, "Row3: post-proc", optimal_threshold),
    ], 1):
        cm = confusion_matrix(y_true, yp, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Non-ictal", "Ictal"], yticklabels=["Non-ictal", "Ictal"],
                    linewidths=0.5, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("M2 %s | t=%.3f | Sens=%.3f Spec=%.3f F1=%.3f J=%.3f" % (
            rl, th, m["recall"], m["specificity"], m["f1_macro"], m["youden_j"]))
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / ("%s_confusion_matrix_row%d.png" % (pfx, row_idx)),
                    dpi=150, bbox_inches="tight")
        plt.close()

    # -- Figure 6: ROC curve ---------------------------------------------------
    # PURPOSE: Sensitivity vs FPR trade-off. AUROC is threshold-invariant.
    # Three scatter points (R1, R2, R3) mark the actual operating points.
    fpr, tpr, _ = roc_curve(y_true, y_prob)            # arrays for the full ROC curve
    auroc_val = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="#5A7DC8", linewidth=1.5,
            label="TCN+Attention (AUROC = %.4f)" % auroc_val)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5, label="Chance")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#5A7DC8")
    for m, lbl, mk in [(row1_metrics, "R1", "o"), (row2_metrics, "R2", "s"), (row3_metrics, "R3", "D")]:
        ax.scatter([1 - m["specificity"]], [m["recall"]], marker=mk, s=60, zorder=5, label=lbl)
    ax.set_xlabel("FPR (1 - Specificity)"); ax.set_ylabel("TPR (Sensitivity)")
    ax.set_title("TCN+Attention ROC -- Validation"); ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / ("%s_roc_curve.png" % pfx), dpi=150, bbox_inches="tight")
    plt.close()

    # -- Figure 7: Threshold curve ---------------------------------------------
    # PURPOSE: Answers "how was the threshold selected?" Youden J vs threshold.
    # A broad peak = robust to threshold choice; narrow spike = sensitive.
    curve = thresh_result.get("threshold_curve", {})   # dict mapping threshold -> Youden J
    if curve:
        tlist = sorted(curve.keys()); slist = [curve[t] for t in tlist]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(tlist, slist, color="#5A7DC8", linewidth=1.2)
        ax.axvline(optimal_threshold, linestyle="--", color="#C85A5A", linewidth=1.2)
        ax.annotate("Optimal t=%.3f" % optimal_threshold,
                    xy=(optimal_threshold, max(slist) if slist else 0),
                    xytext=(10, -10), textcoords="offset points", fontsize=9, color="#C85A5A")
        ax.set_xlabel("Threshold"); ax.set_ylabel("Youden J")
        ax.set_title("TCN+Attention Threshold Selection -- Youden J")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / ("%s_threshold_curve.png" % pfx), dpi=150, bbox_inches="tight")
        plt.close()

    # -- Figure 8: Metrics comparison ------------------------------------------
    # PURPOSE: Single-figure summary of all 7 metrics across 3 rows.
    # Top panel: grouped bars. Bottom panel: FAR/hr (seg vs event).
    mnames = ["accuracy", "precision", "recall", "specificity", "f1_macro", "auroc", "average_precision"]
    r1 = [row1_metrics.get(m, 0) for m in mnames]
    r2 = [row2_metrics.get(m, 0) for m in mnames]
    r3 = [row3_metrics.get(m, 0) for m in mnames]
    fig, (ax_t, ax_b) = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [7, 3]})
    xp = np.arange(len(mnames)); w = 0.25
    b1 = ax_t.bar(xp - w, r1, w, color="#E8A87C", label="Row1: raw t=0.5")
    b2 = ax_t.bar(xp, r2, w, color="#5A7DC8", label="Row2: post-proc t=0.5")
    b3 = ax_t.bar(xp + w, r3, w, color="#41B3A3", label="Row3: post-proc t=opt")
    for bars in [b1, b2, b3]:
        for bar in bars:
            ax_t.annotate("%.2f" % bar.get_height(),
                          xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                          xytext=(0, 2), textcoords="offset points", ha="center", fontsize=6)
    ax_t.set_xticks(xp); ax_t.set_xticklabels(mnames, fontsize=8)
    ax_t.set_ylabel("Score"); ax_t.set_title("TCN+Attention Evaluation Metrics")
    ax_t.legend(fontsize=8); ax_t.set_ylim(0, 1.15)
    fv = [row1_metrics.get("far_per_hour_seg", 0),
          row2_metrics.get("far_per_hour_event", 0), row3_metrics.get("far_per_hour_event", 0)]
    fb = ax_b.bar(["Row1\nseg", "Row2\nevent", "Row3\nevent"], fv,
                  color=["#C85A5A", "#5A7DC8", "#41B3A3"])
    for bar in fb:
        ax_b.annotate("%.2f" % bar.get_height(),
                      xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8)
    ax_b.set_ylabel("FAR/hr"); ax_b.set_title("False Alarm Rate per Hour")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / ("%s_metrics_comparison.png" % pfx), dpi=150, bbox_inches="tight")
    plt.close()

    # -- Figure 9: PR curve ----------------------------------------------------
    # PURPOSE: More informative than ROC for imbalanced data. No-skill baseline
    # at prevalence shows what a random classifier achieves. AP summarises area.
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)  # arrays for full PR curve
    ap = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    prev = np.mean(y_true)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rec_arr, prec_arr, color="#5A7DC8", linewidth=1.5, label="TCN+Attn (AP=%.4f)" % ap)
    ax.axhline(prev, linestyle="--", color="gray", alpha=0.5, label="No-skill (%.3f)" % prev)
    ax.fill_between(rec_arr, prec_arr, alpha=0.08, color="#5A7DC8")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("TCN+Attention PR Curve -- Validation"); ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / ("%s_pr_curve.png" % pfx), dpi=150, bbox_inches="tight")
    plt.close()

    # -- Figure 10: Calibration curve ------------------------------------------
    # PURPOSE: Are predicted probabilities well-calibrated? Points above diagonal
    # = under-confidence; below = over-confidence. Poor calibration does not
    # affect AUROC but affects clinical interpretation of probability outputs.
    fig, ax = plt.subplots(figsize=(5, 5))
    try:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)  # bin probabilities
        ax.plot(mean_pred, frac_pos, "o-", color="#5A7DC8", label="TCN+Attention")
    except ValueError:                                 # can fail if too few positive samples
        pass
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5, label="Perfect")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction of positives")
    ax.set_title("TCN+Attention Calibration Curve"); ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / ("%s_calibration_curve.png" % pfx), dpi=150, bbox_inches="tight")
    plt.close()

    # -- Figure 11: FAR comparison ---------------------------------------------
    # PURPOSE: Highlights segment-level vs event-level FAR/hr. Post-processing
    # collapses consecutive FP segments into single events, drastically reducing
    # the clinically reported alarm rate. The visual drop quantifies the benefit.
    fig, ax = plt.subplots(figsize=(7, 4))
    fl = ["Row1\nsegment-level", "Row2\nevent-level", "Row3\nevent-level"]
    fvals = [row1_metrics.get("far_per_hour_seg", 0),
             row2_metrics.get("far_per_hour_event", 0), row3_metrics.get("far_per_hour_event", 0)]
    bars = ax.bar(fl, fvals, color=["#C85A5A", "#5A7DC8", "#41B3A3"], edgecolor="white")
    for bar in bars:
        ax.annotate("%.2f" % bar.get_height(),
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    ax.set_ylabel("FAR/hr"); ax.set_title("TCN+Attention FAR/hr: segment vs event")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / ("%s_far_comparison.png" % pfx), dpi=150, bbox_inches="tight")
    plt.close()

    # -- Figure 12: Segment length analysis ------------------------------------
    # PURPOSE: Justifies MIN_EVENT_SEC and REFRACTORY_SEC. Left: event duration
    # histogram shows the filter removes only short artefacts. Right: event
    # timeline colour-coded by true/false alarm validates post-processing.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    durations = []                                     # collect all event durations for histogram
    for post in [post_row2, post_row3]:
        for evt in post.get("events", []):
            durations.append(evt.get("duration_sec", 0))
    if durations:
        axes[0].hist(durations, bins=20, color="#5A7DC8", edgecolor="white", alpha=0.85)
        axes[0].axvline(MIN_EVENT_SEC, linestyle="--", color="#C85A5A", linewidth=1.2,
                        label="Min = %.0fs" % MIN_EVENT_SEC)
        axes[0].legend(fontsize=8)
    axes[0].set_xlabel("Event duration (s)"); axes[0].set_ylabel("Count")
    axes[0].set_title("TCN+Attention Event Duration Distribution")
    evts_r3 = compute_event_level_far(y_true, post_row3, STEP_SEC, SEGMENT_SEC).get("event_details", [])
    if evts_r3:
        axes[1].scatter([e["start_sec"] for e in evts_r3], [e["duration_sec"] for e in evts_r3],
                        c=["#41B3A3" if e["is_true_alarm"] else "#C85A5A" for e in evts_r3],
                        s=30, alpha=0.7)
        axes[1].axhline(MIN_EVENT_SEC, linestyle="--", color="gray", alpha=0.5)
        axes[1].scatter([], [], c="#41B3A3", label="True alarm")
        axes[1].scatter([], [], c="#C85A5A", label="False alarm")
        axes[1].legend(fontsize=8)
    axes[1].set_xlabel("Event start (s)"); axes[1].set_ylabel("Duration (s)")
    axes[1].set_title("Event Timeline -- Row 3 (optimal threshold)")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / ("%s_segment_length_analysis.png" % pfx), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: all 12 standard figures")


# ---------------------------------------------------------------------------
# plot_attention_saliency
# ---------------------------------------------------------------------------
def plot_attention_saliency(model, val_loader, y_true, device, logger):
    """Produce temporal attention saliency figure (Figure 13).

    Extracts attention weights for all validation segments using
    TCNWithAttention.get_attention_weights(). Plots mean saliency profiles
    averaged over ictal and non-ictal segments separately.

    File saved: outputs/TCNAttention/figures/tcn_attention_saliency_maps.png
    """
    # -- Collect attention weights alpha_t for every validation segment ----------
    # get_attention_weights() runs a forward pass through the TCN backbone
    # and attention layers, returning softmax-normalised weights per time step.
    # These weights are the model's learned temporal saliency map.
    model.eval()                                       # disable dropout for deterministic weights
    all_weights = []                                   # accumulate (batch, T) arrays
    all_labels = []                                    # accumulate ground truth labels
    with torch.no_grad():                              # no gradients needed for inference
        for x, y in val_loader:                        # iterate over all validation batches
            x = x.to(device, non_blocking=True)         # transfer input to GPU
            w = model.get_attention_weights(x)         # returns np.ndarray shape (batch, T)
            all_weights.extend(w)                      # extend list with individual arrays
            all_labels.extend(y.numpy())               # labels stay on CPU
    weights = np.array(all_weights)                    # (n_segments, T) -- all val attention maps
    labels = np.array(all_labels, dtype=int)           # (n_segments,) -- 0=non-ictal, 1=ictal

    ictal_weights = weights[labels == 1]               # attention maps for seizure segments
    nonictal_weights = weights[labels == 0]            # attention maps for normal segments
    logger.info("Attention saliency: %d ictal, %d non-ictal segments",
                ictal_weights.shape[0], nonictal_weights.shape[0])

    if ictal_weights.shape[0] == 0 or nonictal_weights.shape[0] == 0:
        logger.warning("Cannot plot saliency: one class has zero segments.")
        return

    mean_ictal = ictal_weights.mean(axis=0)             # average saliency profile for seizures
    mean_nonictal = nonictal_weights.mean(axis=0)      # average saliency profile for normals
    std_ictal = ictal_weights.std(axis=0)              # variability across ictal segments
    std_nonictal = nonictal_weights.std(axis=0)        # variability across non-ictal segments

    T = mean_ictal.shape[0]                            # number of time steps (= 2500 at 500 Hz)
    # Convert time-step indices to seconds for interpretable x-axis labels.
    # Causal trim in CausalConvBlock preserves the original sequence length.
    time_axis = np.arange(T) / FS                      # seconds: 0.000, 0.002, ..., 4.998

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9))  # three stacked subplots

    # -- Subplot 1: mean attention per class -----------------------------------
    # PURPOSE: Shows whether the model attends to different time steps for ictal
    # vs non-ictal segments. Higher ictal attention = model focuses on seizure.
    ax1.plot(time_axis, mean_ictal, color="#C85A5A", linewidth=1.2, label="Ictal (mean)")
    ax1.fill_between(time_axis, mean_ictal - std_ictal, mean_ictal + std_ictal,
                     color="#C85A5A", alpha=0.15, label="Ictal (std)")
    ax1.plot(time_axis, mean_nonictal, color="#5A7DC8", linewidth=1.2, label="Non-ictal (mean)")
    ax1.fill_between(time_axis, mean_nonictal - std_nonictal, mean_nonictal + std_nonictal,
                     color="#5A7DC8", alpha=0.15, label="Non-ictal (std)")
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Mean attention weight")
    ax1.set_title("Mean Temporal Attention Weights by Class")
    ax1.legend(fontsize=8)

    # -- Subplot 2: differential attention (ictal - non-ictal) -----------------
    # PURPOSE: Positive values indicate time steps where ictal segments receive
    # more attention than non-ictal ones. This directly shows whether the model
    # has learned to preferentially attend to seizure-relevant time steps.
    diff = mean_ictal - mean_nonictal                  # positive = ictal-focused
    ax2.fill_between(time_axis, 0, diff, where=(diff >= 0), color="#C85A5A", alpha=0.4,
                     label="Ictal > Non-ictal")
    ax2.fill_between(time_axis, 0, diff, where=(diff < 0), color="#5A7DC8", alpha=0.4,
                     label="Non-ictal > Ictal")
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Delta alpha (ictal - non-ictal)")
    ax2.set_title("Differential Attention: Ictal minus Non-ictal")
    ax2.legend(fontsize=8)

    # -- Subplot 3: example ictal segment with attention overlay ---------------
    # PURPOSE: Concrete visualisation of one seizure segment with the attention
    # weights overlaid as a colour bar. This provides interpretable evidence
    # that the model focuses on the ictal discharge rather than artefacts.
    # Select the ictal segment with the highest mean attention weight as the
    # most informative example for the figure.
    ictal_indices = np.where(labels == 1)[0]           # indices of all ictal segments
    mean_per_seg = weights[ictal_indices].mean(axis=1) # average attention across T per segment
    best_seg_local = np.argmax(mean_per_seg)           # index within ictal subset
    best_seg_global = ictal_indices[best_seg_local]    # index within full validation set

    # Load the raw EEG waveform for this example segment
    seg_path, seg_label = val_loader.dataset.pairs[best_seg_global]  # (filepath, label) tuple
    raw_eeg = np.load(seg_path).astype(np.float32)     # load .npy segment as float32
    # Normalise amplitude to [-1, 1] for visual display (does not affect attention weights)
    raw_max = np.abs(raw_eeg).max()                    # peak absolute amplitude
    if raw_max > 0:
        raw_eeg = raw_eeg / raw_max                    # scale to [-1, 1]
    eeg_time = np.arange(len(raw_eeg)) / FS            # time axis in seconds for raw EEG
    seg_weights = weights[best_seg_global]              # attention weights for this segment

    ax3.plot(eeg_time, raw_eeg, color="gray", linewidth=0.5, alpha=0.7, label="EEG")
    # Overlay attention as coloured scatter along the bottom
    # Map attention weights to time axis (same length as T)
    w_time = np.arange(len(seg_weights)) / FS
    scatter = ax3.scatter(w_time, np.full_like(seg_weights, raw_eeg.min() - 0.15),
                          c=seg_weights, cmap="Reds", s=4, vmin=0,
                          vmax=seg_weights.max() if seg_weights.max() > 0 else 1)
    fig.colorbar(scatter, ax=ax3, label="Attention weight", shrink=0.6)
    ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Normalised amplitude")
    ax3.set_title("Example Ictal Segment with Attention Overlay (seg #%d)" % best_seg_global)
    ax3.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "tcn_attention_saliency_maps.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: tcn_attention_saliency_maps.png")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    """Main entry point for TCN + Temporal Attention training (M2).

    Trains for MAX_EPOCHS=100 with early stopping. All params joint.
    Does not evaluate the test set.
    """
    # -- Step 1: Logging and setup ---------------------------------------------
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("TCNTemporalAttention.py")
    logger.info("Timestamp       : %s", datetime.datetime.now().isoformat())
    logger.info("Ablation role   : M2 -- TCN + Temporal Attention")
    logger.info("Params source   : best_params.json + best_attention_params.json")
    logger.info("Purpose         : Final model training (all params joint)")
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

    # -- Step 2: Load both parameter files -------------------------------------
    backbone_config, backbone_hp, attn_config, attn_hp = load_best_params(logger)

    # -- Step 3: Load data splits ----------------------------------------------
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

    # -- Step 4: Build model ---------------------------------------------------
    model = build_model(backbone_hp, attn_hp, DEVICE, logger)
    n_params = count_parameters(model)

    # -- Step 5: Build data loaders --------------------------------------------
    # batch_size comes from attention tuning (attn_hp), not backbone tuning
    batch_size = int(attn_hp["batch_size"])
    train_loader = make_loader(train_pairs, batch_size, True, DEVICE)
    val_loader = make_loader(val_pairs, batch_size, False, DEVICE)
    logger.info("Train loader: %d batches | batch_size=%d", len(train_loader), batch_size)
    logger.info("Val loader  : %d batches", len(val_loader))

    # -- Step 6: Build training components -------------------------------------
    # lr, wd come from attn_hp (optimised during attention tuning)
    optimiser, scheduler, criterion = build_training_components(
        model, pos_weight, attn_hp, DEVICE, logger)

    # -- Step 7: Initialise training state -------------------------------------
    history = {"epoch": [], "train_loss": [], "val_f1": [], "lr": []}
    best_val_f1 = 0.0
    epochs_no_imp = 0
    best_state = None
    best_epoch = 0
    start_epoch = 1
    training_start = datetime.datetime.now()

    # -- Resume from checkpoint if available -----------------------------------
    # latest.pt contains everything: current model/optimiser/scheduler state,
    # best-epoch weights, patience counter, and best_val_f1. A single file
    # load restores the exact training state. tcn_attention_best.pt is a
    # fallback if latest.pt was corrupted (e.g., Slurm killed during torch.save).
    # To start fresh, delete the checkpoints/ directory.
    latest_path = CKPT_DIR / "tcn_attention_latest.pt"
    best_ckpt_path = CKPT_DIR / "tcn_attention_best.pt"
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

    # -- Step 8: Training loop -------------------------------------------------
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
        final_epoch = epoch                            # update on each iteration for post-loop logging

        t0_train = time.time()
        train_loss = train_epoch(model, train_loader, optimiser, criterion, DEVICE, scaler=scaler)
        train_sec = time.time() - t0_train

        t0_val = time.time()
        val_f1, _, _, _ = evaluate_model(model, val_loader, DEVICE, logger, use_amp=use_amp)
        val_sec = time.time() - t0_val

        current_lr = scheduler.get_last_lr()[0]        # capture LR before scheduler.step() updates it
        scheduler.step()                               # advance cosine annealing by one epoch

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

        if val_f1 > best_val_f1:                        # new best model found
            best_val_f1 = val_f1                       # update best F1 tracker
            best_epoch = epoch                         # record which epoch produced the best
            epochs_no_imp = 0                          # reset patience counter
            # Clone weights to CPU to avoid consuming a second GPU copy in VRAM
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            # Save best checkpoint separately (insurance if latest.pt corrupts)
            save_checkpoint(epoch, model, optimiser, scheduler, val_f1, train_loss, attn_hp,
                            best_ckpt_path, logger,
                            best_model_state=best_state,
                            best_val_f1=best_val_f1, best_epoch=best_epoch,
                            epochs_no_imp=epochs_no_imp)
            logger.info("  New best val F1: %.4f at epoch %d", best_val_f1, best_epoch)
        else:
            epochs_no_imp += 1                         # no improvement -- increment patience

        # Save latest.pt every epoch — single file for clean resume
        save_checkpoint(epoch, model, optimiser, scheduler, val_f1, train_loss, attn_hp,
                        latest_path, logger,
                        best_model_state=best_state,
                        best_val_f1=best_val_f1, best_epoch=best_epoch,
                        epochs_no_imp=epochs_no_imp)

        if epoch % CHECKPOINT_FREQ == 0:
            cleanup_checkpoints(CKPT_DIR, KEEP_CKPTS, logger)

        if epochs_no_imp >= ES_PATIENCE:               # early stopping triggered
            logger.info("Early stopping at epoch %d.", epoch)
            break                                      # exit training loop

    # -- Step 9: Restore best weights ------------------------------------------
    # Load the CPU-stored best_state back to GPU for final evaluation.
    # This ensures all subsequent predictions use the best-epoch model.
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

    # -- Step 10: Save final weights -------------------------------------------
    # Move to CPU before saving so the .pt file is device-agnostic
    torch.save(model.cpu().state_dict(), WEIGHTS_PATH)  # save device-agnostic state_dict
    size_mb = WEIGHTS_PATH.stat().st_size / 1e6         # compute file size for logging
    logger.info("Weights saved : %s (%.2f MB)", WEIGHTS_PATH, size_mb)
    model.to(DEVICE)                                    # move back to GPU for final evaluation

    # -- Step 11: Final evaluation ---------------------------------------------
    # Re-evaluate to produce y_true, y_prob arrays needed for post-processing
    # and all 13 figures. Also serves as a consistency check on weight restoration.
    logger.info("Running final validation evaluation...")
    val_f1_final, y_true, y_pred_05, y_prob = evaluate_model(model, val_loader, DEVICE, logger, use_amp=use_amp)
    tol = 1e-3                                          # tolerance for floating-point rounding
    if abs(val_f1_final - best_val_f1) > tol:           # F1 should match best-epoch value
        logger.warning("F1 mismatch: best=%.6f final=%.6f.", best_val_f1, val_f1_final)
    else:
        logger.info("F1 consistency check: PASS")

    # -- Step 12: Post-processing evaluations ----------------------------------
    (row1_metrics, row2_metrics, row3_metrics,
     post_row2, post_row3, far_row2, far_row3,
     thresh_result, optimal_threshold) = run_postprocessing_evaluations(y_true, y_prob, logger)

    # -- Step 13: Save all results ---------------------------------------------
    y_pred_row1 = (y_prob >= 0.5).astype(int)          # raw binary preds at standard threshold
    y_pred_row2 = post_row2["smoothed_preds"]          # post-processed preds at threshold 0.5
    y_pred_row3 = post_row3["smoothed_preds"]          # post-processed preds at optimal threshold
    save_all_results(
        history, row1_metrics, row2_metrics, row3_metrics,
        far_row2, far_row3, backbone_hp, attn_hp,
        best_epoch, best_val_f1, elapsed, DEVICE, n_params, y_true,
        y_pred_row1, y_pred_row2, y_pred_row3, logger)

    # -- Step 14: Plot standard figures ----------------------------------------
    plot_all_figures(
        history, best_epoch, best_val_f1, y_true, y_prob,
        y_pred_row1, y_pred_row2, y_pred_row3,
        row1_metrics, row2_metrics, row3_metrics,
        post_row2, post_row3, thresh_result, optimal_threshold, logger)

    # -- Step 15: Plot attention saliency (Figure 13) --------------------------
    # TCNWithAttention.get_attention_weights() confirmed present in tcn_utils.py
    plot_attention_saliency(model, val_loader, y_true, DEVICE, logger)

    # -- Step 16: Final inventory and cleanup ----------------------------------
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared.")

    logger.info("=" * 65)
    logger.info("ALL OUTPUTS SAVED")
    pfx = "tcn_attention"
    all_outputs = [
        WEIGHTS_PATH,
        CKPT_DIR / "tcn_attention_latest.pt",
        CKPT_DIR / "tcn_attention_best.pt",
        TRAIN_LOG_PATH, EVAL_REPORT_PATH, THRESH_PATH, EPOCH_CSV, THREE_ROW_CSV,
        OUTPUT_ROOT / "tcn_attention_classification_report_row1.json",
        OUTPUT_ROOT / "tcn_attention_classification_report_row2.json",
        OUTPUT_ROOT / "tcn_attention_classification_report_row3.json",
        OUTPUT_ROOT / "tcn_attention_event_details_row2.csv",
        OUTPUT_ROOT / "tcn_attention_event_details_row3.csv",
        FIGURE_DIR / ("%s_training_curves.png" % pfx),
        FIGURE_DIR / ("%s_lr_schedule.png" % pfx),
        FIGURE_DIR / ("%s_confusion_matrix_row1.png" % pfx),
        FIGURE_DIR / ("%s_confusion_matrix_row2.png" % pfx),
        FIGURE_DIR / ("%s_confusion_matrix_row3.png" % pfx),
        FIGURE_DIR / ("%s_roc_curve.png" % pfx),
        FIGURE_DIR / ("%s_threshold_curve.png" % pfx),
        FIGURE_DIR / ("%s_metrics_comparison.png" % pfx),
        FIGURE_DIR / ("%s_pr_curve.png" % pfx),
        FIGURE_DIR / ("%s_calibration_curve.png" % pfx),
        FIGURE_DIR / ("%s_far_comparison.png" % pfx),
        FIGURE_DIR / ("%s_segment_length_analysis.png" % pfx),
        FIGURE_DIR / "tcn_attention_saliency_maps.png",
    ]
    for p in all_outputs:
        status = "OK     " if Path(p).exists() else "MISSING"
        logger.info("  [%s] %s", status, p)

    logger.info("=" * 65)
    logger.info("ABLATION NOTE: Compare outputs/TCN/tcn_three_row_summary.csv (M1) "
                "with outputs/TCNAttention/tcn_attention_three_row_summary.csv (M2) "
                "to isolate the contribution of temporal attention.")
    logger.info("NEXT: run final_evaluation.py when test data is ready.")


if __name__ == "__main__":
    main()


# ======================================================================
# RESEARCH REPORTING GUIDE -- TCNTemporalAttention.py
# ======================================================================
#
# -- ARCHITECTURE (Methods) ------------------------------------------------
# "Model 2 extended the TCN baseline (M1) by replacing global average
# pooling with two-layer additive temporal attention. For the TCN output
# H in R^(T x F), per-timestep energy scores were computed as
# e_t = tanh(W_a h_t + b_a) where W_a in R^(D_a x F), followed by
# scalar scores v^T e_t. Attention weights alpha_t = softmax({v^T e_t})
# formed a context vector c = sum_t alpha_t h_t, regularised by dropout
# before the linear classification head.
# Complexity is O(T * F * D_a), linear in sequence length."
#
# Attention uses two-layer additive scoring (tanh + linear) with
# tunable attention_dim and attention_dropout from best_attention_params.json.
# This matches MultiScaleTCNWithAttention for consistent ablation.
#
# Backbone params (from best_params.json):
#   num_layers L  -- determines RF = 2*(2^L - 1)*(k-1) + 1
#   kernel_size k
#   num_filters F -- also determines attention embed_dim
#   dropout p
#
# Attention params (from best_attention_params.json):
#   attention_dim D_a -- projection dimension of scoring function
#   attention_dropout p_a -- dropout on context vector
#   learning_rate, weight_decay, batch_size
#
# Total trainable parameters (all -- backbone + attention)
# Backbone parameter count (reported separately)
# Attention parameter count (= total - backbone)
#
# -- TRAINING (Methods) ----------------------------------------------------
# "Training hyperparameters (learning rate, weight decay, batch size) were
# tuned using Optuna TPE (tune_temporal_attention.py) with the TCN backbone
# frozen. For final training, all parameters (backbone and attention) were
# trained jointly for up to 100 epochs with early stopping (patience 10),
# allowing the backbone to adapt to the attention mechanism."
#
# WHY 100 EPOCHS: Standard EEG DL budget. Cosine annealing needs a full
#   half-cycle. Early stopping terminates when converged. Same as M1.
#
# WHY JOINT TRAINING (not frozen backbone):
#   During tuning, the backbone was frozen to isolate the attention
#   hyperparameter search. During final training, unfreezing the backbone
#   allows it to adapt slightly to the attention mechanism, producing
#   the best possible final model. The backbone architecture is still
#   identical to M1, ensuring the ablation comparison is valid.
#
# WHY TRAINING HPs FROM ATTENTION TUNING (not backbone tuning):
#   The learning rate and weight decay were optimised specifically for
#   training the attention-augmented model. Using backbone tuning HPs
#   would apply training settings optimised for a different architecture.
#
# -- ABLATION CONTEXT (Results/Discussion) ----------------------------------
# M1 (TCN) vs M2 (TCN + Temporal Attention):
#   Both share identical backbone (num_layers, kernel_size, num_filters,
#   dropout). The only difference is the attention pooling mechanism.
#   Compare: outputs/TCN/tcn_three_row_summary.csv (M1)
#       vs:  outputs/TCNAttention/tcn_attention_three_row_summary.csv (M2)
#
# -- INTERPRETABILITY (Results) --------------------------------------------
# "Temporal attention weights were extracted for all validation segments
# using get_attention_weights() and averaged over ictal and non-ictal
# classes separately. The saliency profiles demonstrated [higher/lower]
# attention during ictal periods, indicating preferential focus on
# seizure-relevant time steps without requiring post-hoc attribution."
# See: tcn_attention_saliency_maps.png (Figure 13)
#
# -- FIGURES ----------------------------------------------------------------
# Figures 1-12: same reviewer questions as TCN.py.
# Figure 13 (tcn_attention_saliency_maps.png):
#   Q: Does the attention learn clinically meaningful temporal focus?
#   Q: Is there a measurable class-dependent difference in saliency?
#   Q: Can predictions be explained without SHAP / GradCAM?
#   A: Three-panel figure: mean attention per class, differential
#      profile, and example segment with attention overlay.
# ======================================================================
