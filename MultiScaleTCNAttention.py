"""
MultiScaleTCNAttention.py
=========================
Training script for Model 4: Multi-Scale TCN with Temporal Attention.

Mirrors TCNTemporalAttention.py in structure. Adapted for the
MultiScaleTCNWithAttention architecture from tcn_utils.py.

Trains for exactly 100 epochs with early stopping
on validation macro F1 (patience 10). ALL parameters
(backbone and attention) are trained jointly.

Validation strategy
-------------------
Per-epoch monitoring uses a 5% stratified subset of
the validation set (preserving the natural ictal:
non-ictal prevalence). This keeps the early-stopping
signal cheap. The full validation set is used only
once after training, for the post-training pass:
three-row evaluation, smoothing/refractory/min-
duration post-processing, Youden's-J threshold
optimisation, and events/24h false-alarm rate.

Hyperparameters loaded from TWO sources:
  best_multiscale_params.json
    -> MultiScaleTCN backbone: num_filters, kernel_size,
                               dropout, fusion
    -> Branch dilations: branch1, branch2, branch3
  best_multiscale_attn_params.json
    -> Attention : attention_dim, attention_dropout
    -> Training  : learning_rate, weight_decay, batch_size
    -> Also contains backbone_hyperparameters for validation

Architecture: MultiScaleTCNWithAttention (tcn_utils.py)
  Backbone: three parallel CausalConvBlock branches
    Branch 1: dilations [1,  2,   4]   -- fine scale         (spike morphology)
    Branch 2: dilations [8,  16,  32]  -- intermediate scale (rhythmic bursts)
    Branch 3: dilations [32, 64, 128]  -- coarse scale       (seizure evolution)
  Attention: two-layer additive temporal attention
    (tanh + linear scorer) with tunable attention_dim
    and attention_dropout.

Three evaluation rows:
  Row 1: raw predictions at threshold 0.5
  Row 2: post-processed at threshold 0.5
  Row 3: post-processed at optimal threshold

The test set is never loaded in this script.
It is reserved for final_evaluation.py.

Ablation position
-----------------
M3 (MultiScaleTCN.py) vs M4 (this script):
  Isolates the contribution of temporal attention
  over the multi-scale TCN baseline.
  Both models share identical backbone architecture.

Pipeline position
-----------------
After  : tune_multiscale_attention.py
Before : final_evaluation.py

Usage
-----
python MultiScaleTCNAttention.py

Key outputs
-----------
{OUTPUT_ROOT}/ms_attn_final_weights.pt
{OUTPUT_ROOT}/ms_attn_evaluation_report.json
{OUTPUT_ROOT}/ms_attn_three_row_summary.csv
{OUTPUT_ROOT}/figures/  (13 figures)
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
import random                              # pre-flight file sampling
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve,
    average_precision_score,
    classification_report,
)
# calibration_curve lives in sklearn.calibration, not sklearn.metrics
# (since scikit-learn 0.21+). Prior versions placed it under metrics.
from sklearn.calibration import calibration_curve

from tcn_utils import (
    set_seed,
    MultiScaleTCNWithAttention,
    make_loader,
    # filter_unpaired_subjects,  # handled offline by create_balanced_splits.py
    train_one_epoch,
    count_parameters,
    segment_predictions_to_events,
    compute_event_level_far,
    find_optimal_threshold,
    make_classreport_barplot,
    ema_smooth,
)

from eval_utils import evaluate_event_level, THRESHOLD


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED              = 42
MAX_EPOCHS        = 100
ES_PATIENCE       = 15
# Min-epochs gate before ES can fire. The cosine schedule (T_max=100) starts
# almost flat -- by epoch 17 the LR has decayed only ~7% from its initial
# value. ES firing in this Regime A (high-LR exploration) phase produces
# under-converged models because the schedule's fine-tuning phase (Regime B,
# epochs ~30-80, where LR drops below ~50% of initial) is never reached.
# Setting MIN_EPOCHS_BEFORE_ES = 40 forces training through the high-LR
# phase unconditionally; ES can only fire from epoch 40 onward, when the
# LR is at ~36% of its initial value and the model is genuinely fine-tuning.
# After the gate, ES_PATIENCE = 15 applies as before.
MIN_EPOCHS_BEFORE_ES = 40    # ES disabled until this epoch
GRAD_CLIP         = 1.0      # max-norm gradient clipping (matches tuning protocol)
CHECKPOINT_FREQ   = 5
KEEP_CKPTS        = 3
# Per-epoch validation monitor fraction. The full val set has ~4.3M segments
# (~11h/epoch I/O bound) so per-epoch validation on the full set is
# prohibitive. We monitor on a 5% stratified subset (preserves natural
# ictal:non-ictal prevalence) for early-stopping decisions and re-evaluate
# on the full val set once after training for the three-row report,
# post-processing, and threshold optimisation.
VAL_MONITOR_FRACTION = 0.05      # 5% stratified per-epoch monitor
FS                = 500
SEGMENT_LEN       = 2500
SEGMENT_SEC       = 5.0
STEP_SEC          = 2.5
MIN_EVENT_SEC     = 10.0
REFRACTORY_SEC    = 30.0
SMOOTHING_WIN     = 3
MODEL_NAME        = "MultiScaleTCNWithAttention"

OUTPUT_ROOT       = Path("/home/people/22206468/scratch/OUTPUT/MODEL4_OUTPUT") / "MultiScaleTCNAttention"
CKPT_DIR          = OUTPUT_ROOT / "checkpoints"
RESULT_CLASSREPORT_DIR = OUTPUT_ROOT / "Result_classReport"      # macro-avg bar plot deliverable
LOG_DIR           = OUTPUT_ROOT / "logs"
FIGURE_DIR        = OUTPUT_ROOT / "figures"
WEIGHTS_PATH      = OUTPUT_ROOT / "ms_attn_final_weights.pt"
TRAIN_LOG_PATH    = OUTPUT_ROOT / "ms_attn_training_log.json"
EVAL_REPORT_PATH  = OUTPUT_ROOT / "ms_attn_evaluation_report.json"
THRESH_PATH       = OUTPUT_ROOT / "ms_attn_optimal_threshold.json"
EPOCH_CSV         = OUTPUT_ROOT / "ms_attn_epoch_metrics.csv"
THREE_ROW_CSV     = OUTPUT_ROOT / "ms_attn_three_row_summary.csv"
VAL_PREDICTIONS_NPZ = OUTPUT_ROOT / "ms_attn_val_predictions_full.npz"  # cached val predictions (Phase 3)

# Two JSON input files -- backbone and attention tuning results
BACKBONE_PARAMS_PATH = Path("/home/people/22206468/scratch/OUTPUT/MODEL3_OUTPUT/MultiScaleTCNtuning_outputs") / "best_multiscale_params.json"
ATTN_PARAMS_PATH     = Path("/home/people/22206468/scratch/OUTPUT/MODEL4_OUTPUT/multiscale_attention_tuning_outputs") / "best_multiscale_attn_params.json"

# Splits manifest -- single source of truth (matches all other pipeline scripts)
#   Previous (uniform downsampling)             : data_splits.json
#   Train-side proximity-aware downsampling     : data_splits_nonictal_sampled.json
#   + val/test filtered (apply_val_test_filter) : data_splits_nonictal_sampled_filtered.json
# Use the fully-filtered manifest so the |x|>1000 / NaN / Inf criterion is
# applied uniformly across train, val, and test partitions. Eliminates the
# FP16-overflow NaN failure mode observed in the M3 final-eval pass.
SPLITS_PATH = Path("/scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits_nonictal_sampled_filtered_enriched.json")
ANNOT_DIR           = Path("/home/people/22206468/scratch/seizure_times_updated")
MOUSE_METADATA_PATH = Path("/home/people/22206468/scratch/INPUT_DATA/Data_diagnostic/mouse_recording_metadata.json")

# Backbone attribute prefix in MultiScaleTCNWithAttention (confirmed: self.backbone)
BACKBONE_ATTR = "backbone"

# Fallback dilation schedules if branch_dilations not in JSON
# (match M3 backbone: tune_multiscale_tcn.py)
DEFAULT_BRANCH1 = [1, 2, 4]
DEFAULT_BRANCH2 = [8, 16, 32]
DEFAULT_BRANCH3 = [32, 64, 128]


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------
def setup_logging():
    """Create all output directories and configure the logger."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    log_file = LOG_DIR / "MultiScaleTCNAttention_training.log"
    logger = logging.getLogger("MSAttention_training")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
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
    """Load hyperparameters from two JSON files.

    1. best_multiscale_params.json
       MultiScaleTCN backbone: num_filters, kernel_size, dropout, fusion.
       Branch dilations: branch1, branch2, branch3.

    2. best_multiscale_attn_params.json
       Attention: attention_dim, attention_dropout.
       Training: learning_rate, weight_decay, batch_size.

    Returns
    -------
    tuple of (backbone_config, backbone_hp, branch_dilations, attn_config, attn_hp)
    """
    # -- Load backbone params --------------------------------------------------
    if not BACKBONE_PARAMS_PATH.exists():
        logger.error("best_multiscale_params.json not found at %s. "
                     "Run tune_multiscale_tcn.py first.", BACKBONE_PARAMS_PATH)
        raise FileNotFoundError(str(BACKBONE_PARAMS_PATH))

    with open(BACKBONE_PARAMS_PATH, "r", encoding="utf-8") as f:
        backbone_config = json.load(f)
    if "hyperparameters" not in backbone_config:
        logger.error("'hyperparameters' key missing from %s.", BACKBONE_PARAMS_PATH)
        raise KeyError("hyperparameters")
    backbone_hp = backbone_config["hyperparameters"]

    # Extract branch dilations from JSON or fall back to constants
    branch_dilations = backbone_config.get("branch_dilations", None)
    if branch_dilations is None:
        logger.warning("branch_dilations not found in JSON. Using defaults.")
        branch_dilations = {
            "branch1": DEFAULT_BRANCH1,
            "branch2": DEFAULT_BRANCH2,
            "branch3": DEFAULT_BRANCH3,
        }

    # -- Load attention tuning params ------------------------------------------
    if not ATTN_PARAMS_PATH.exists():
        logger.error("best_multiscale_attn_params.json not found at %s. "
                     "Run tune_multiscale_attention.py first.", ATTN_PARAMS_PATH)
        raise FileNotFoundError(str(ATTN_PARAMS_PATH))

    with open(ATTN_PARAMS_PATH, "r", encoding="utf-8") as f:
        attn_config = json.load(f)
    if "hyperparameters" not in attn_config:
        logger.error("'hyperparameters' key missing from %s.", ATTN_PARAMS_PATH)
        raise KeyError("hyperparameters")
    attn_hp = attn_config["hyperparameters"]

    # -- Consistency check: backbone in attention JSON must match backbone JSON -
    saved_bb = attn_config.get("backbone_hyperparameters", None)
    if saved_bb is not None:
        for key in ["num_filters", "kernel_size", "dropout", "fusion"]:
            if key in saved_bb and key in backbone_hp:
                if saved_bb[key] != backbone_hp[key]:
                    logger.warning(
                        "Backbone param mismatch: %s = %s in attention JSON vs %s in backbone JSON. "
                        "Attention was tuned on a different backbone. Proceeding with backbone JSON.",
                        key, saved_bb[key], backbone_hp[key])

    # -- Log backbone hyperparameters ------------------------------------------
    logger.info("-" * 55)
    logger.info("MultiScaleTCN backbone hyperparameters:")
    for k, v in backbone_hp.items():
        logger.info("  %-20s: %s", k, v)
    for bname, dils in branch_dilations.items():
        logger.info("  %-20s: %s", bname, dils)

    # Per-branch receptive field: RF = 1 + 2 * sum((k-1)*d for d in dilations)
    ks = int(backbone_hp["kernel_size"])
    for bname, dils in branch_dilations.items():
        rf = 1 + 2 * sum((ks - 1) * d for d in dils)
        logger.info("  %s RF: %d samples (%.3f s at %d Hz)", bname, rf, rf / FS, FS)

    # -- Log attention/training hyperparameters --------------------------------
    logger.info("-" * 55)
    logger.info("Attention/training hyperparameters:")
    for k, v in attn_hp.items():
        logger.info("  %-20s: %s", k, v)

    # -- Log tuning reference values -------------------------------------------
    bb_best_f1 = backbone_config.get("best_val_f1", "not recorded")
    attn_best_f1 = attn_config.get("best_val_f1", "not recorded")
    logger.info("-" * 55)
    logger.info("Backbone tuning best val F1 : %s", bb_best_f1)
    logger.info("Attention tuning best val F1: %s", attn_best_f1)
    logger.info("-" * 55)

    return backbone_config, backbone_hp, branch_dilations, attn_config, attn_hp


# ---------------------------------------------------------------------------
# load_splits
# ---------------------------------------------------------------------------
def load_splits(logger):
    """Load train and val file-label pairs from data_splits_nonictal_sampled_filtered.json.

    Never loads test pairs. Returns (train_pairs, val_pairs).
    """
    if not SPLITS_PATH.exists():
        logger.error("data_splits_nonictal_sampled_filtered.json not found at %s. "
                     "Pipeline: generate_data_splits.py -> create_balanced_splits.py -> "
                     "apply_val_test_filter.py. Run apply_val_test_filter.py to produce "
                     "this manifest.", SPLITS_PATH)
        raise FileNotFoundError(str(SPLITS_PATH))

    logger.info("Loading splits from: %s", SPLITS_PATH)
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_pairs = [(rec["filepath"], rec["label"]) for rec in splits["train"]]
    val_pairs   = [(rec["filepath"], rec["label"]) for rec in splits["val"]]
    val_records = list(splits["val"])     # full enriched records (mouse_id, chrono_idx, t_start_sec)

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

    if not all("chrono_idx" in r for r in val_records[:5]):
        logger.error("Val records missing 'chrono_idx' -- pointing at the un-enriched manifest? "
                     "SPLITS_PATH must end in '_enriched.json'. Got: %s", SPLITS_PATH)
        raise RuntimeError("Manifest not enriched with chronology")

    test_status = splits.get("metadata", {}).get("test_status", "pending")
    if test_status != "complete":
        logger.info("Test data not yet ready -- test split not loaded here.")

    return train_pairs, val_pairs, val_records


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------
def build_model(backbone_hp, branch_dilations, attn_hp, device, logger):
    """Instantiate MultiScaleTCNWithAttention.

    Backbone params from best_multiscale_params.json, attention params from
    best_multiscale_attn_params.json.

    ALL parameters are trainable (no freezing). This is logged explicitly.

    Returns
    -------
    model : nn.Module
    """
    set_seed(SEED)
    model = MultiScaleTCNWithAttention(
        num_filters=int(backbone_hp["num_filters"]),
        kernel_size=int(backbone_hp["kernel_size"]),
        dropout=float(backbone_hp["dropout"]),
        fusion=str(backbone_hp["fusion"]),
        attention_dim=int(attn_hp["attention_dim"]),
        attention_dropout=float(attn_hp["attention_dropout"]),
        branch1_dilations=branch_dilations["branch1"],
        branch2_dilations=branch_dilations["branch2"],
        branch3_dilations=branch_dilations["branch3"],
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
    backbone_module = getattr(model, BACKBONE_ATTR)
    n_backbone = sum(p.numel() for p in backbone_module.parameters() if p.requires_grad)
    n_attention = n_trainable - n_backbone
    logger.info("  Backbone params    : %s", "{:,}".format(n_backbone))
    logger.info("  Attention + head   : %s", "{:,}".format(n_attention))

    # -- Log per-branch RFs ----------------------------------------------------
    ks = int(backbone_hp["kernel_size"])
    for bname, dils in branch_dilations.items():
        rf = 1 + 2 * sum((ks - 1) * d for d in dils)
        logger.info("  %s RF: %d samples (%.3f s)", bname, rf, rf / FS)
    logger.info("Model      : %s", MODEL_NAME)
    logger.info("Fusion     : %s", backbone_hp["fusion"])
    logger.info("Device     : %s", device)

    return model


# ---------------------------------------------------------------------------
# build_training_components
# ---------------------------------------------------------------------------
def build_training_components(model, pos_weight, attn_hp, device, logger):
    """Build optimiser, scheduler, and loss function.

    learning_rate, weight_decay come from attn_hp (optimised during attention
    tuning). Optimiser covers ALL parameters (backbone + attention jointly).

    Returns (optimiser, scheduler, criterion).
    """
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=float(attn_hp["learning_rate"]),
        weight_decay=float(attn_hp["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max=MAX_EPOCHS,
        eta_min=float(attn_hp["learning_rate"]) * 0.01)
    # Class imbalance is already handled offline via proximity-aware non-ictal
    # downsampling (1:2.37 ictal:non-ictal in train manifest), so pos_weight=1.0
    # at the loss layer. See create_balanced_splits.py for downsampling logic.
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

    Never deletes ms_attn_best.pt or ms_attn_latest.pt.
    """
    pattern = "ms_attn_epoch_*.pt"
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

    Pass max_grad_norm=GRAD_CLIP explicitly so the protocol matches the tuning
    pipeline (1.0 max-norm clipping). Returns mean training loss.
    """
    return train_one_epoch(model, loader, optimiser, criterion, device,
                           max_grad_norm=GRAD_CLIP, scaler=scaler)


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
    """Compute all evaluation metrics for one evaluation row."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

    accuracy    = accuracy_score(y_true, y_pred)
    prec        = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall      = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_macro    = f1_score(y_true, y_pred, average="macro", zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auroc       = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    avg_prec    = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    youden_j    = recall + specificity - 1.0

    n_non_ic   = tn + fp
    non_ic_hrs = (n_non_ic * segment_sec) / 3600.0
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
    """Run Row 1 and Row 2 evaluations (Row 3 retired).

    Threshold optimisation is intentionally disabled: Row 3 (post-processed
    at an Optuna-tuned threshold) consistently produced poor recall on this
    dataset's ~0.27% ictal prevalence, so the report is restricted to:
        Row 1 -- raw predictions at threshold 0.5
        Row 2 -- post-processed predictions at threshold 0.5
    The threshold-optimisation block and Row 3 computation block are kept
    in the source as commented-out scaffolding for future re-enablement.

    Returns (row1_metrics, row2_metrics, post_row2, far_row2).
    """
    # -- Row 1: raw at 0.5 ----------------------------------------------------
    y_pred_row1 = (y_prob >= 0.5).astype(int)
    row1_metrics = compute_all_metrics(y_true, y_pred_row1, y_prob, SEGMENT_SEC, logger,
                                       label="Row1_raw_0.5")
    row1_metrics["threshold"] = 0.5
    row1_metrics["postprocessed"] = False

    # -- (Retired) Find optimal threshold -- Row 3 was disabled --------------
    # Threshold optimisation produced operating points with poor recall on
    # this dataset (Youden tau*=0.10 -> F1=0.56 / precision=7.4% vs
    # F1=0.83 / precision=62% at tau=0.5). The report now ends at Row 2
    # (post-processing at tau=0.5). Re-enable the block below to restore
    # Row 3.
    #
    # thresh_result = find_optimal_threshold(y_true, y_prob, objective="f1")
    # optimal_threshold = thresh_result["optimal_threshold"]
    # with open(THRESH_PATH, "w", encoding="utf-8") as f:
    #     json.dump({
    #         "model": MODEL_NAME,
    #         "timestamp": datetime.datetime.now().isoformat(),
    #         "optimal_threshold": optimal_threshold,
    #         "objective": "f1",
    #         "optimal_score": thresh_result["optimal_score"],
    #         "sensitivity_at_opt": thresh_result["sensitivity_at_opt"],
    #         "specificity_at_opt": thresh_result["specificity_at_opt"],
    #         "post_processing": {
    #             "smoothing_window": SMOOTHING_WIN,
    #             "refractory_period_sec": REFRACTORY_SEC,
    #             "min_event_duration_sec": MIN_EVENT_SEC,
    #             "step_sec": STEP_SEC,
    #         },
    #     }, f, indent=2)
    # logger.info("Optimal threshold computed and saved: %.4f", optimal_threshold)

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
    # Pass smoothed_probs so AUROC and PRAUC reflect the post-processed pipeline.
    # Stage 1 of post-processing applies a moving-average smoothing kernel to
    # y_prob (window = SMOOTHING_WIN); the smoothed continuous probabilities
    # are what the threshold is applied to. Reporting AUROC/PRAUC on raw
    # y_prob would understate the effective discrimination of the deployed
    # system because the smoothing reduces single-segment noise spikes.
    row2_metrics = compute_all_metrics(
        y_true, y_pred_row2, post_row2["smoothed_probs"], SEGMENT_SEC, logger,
        label="Row2_postproc_0.5")
    row2_metrics["threshold"] = 0.5
    row2_metrics["postprocessed"] = True
    row2_metrics["far_per_hour_event"] = round(float(far_row2["far_per_hour"]), 6)
    row2_metrics["n_true_alarms"] = far_row2["n_true_alarms"]
    row2_metrics["n_false_alarms"] = far_row2["n_false_alarms"]
    row2_metrics["n_total_events"] = far_row2["n_total_events"]

    # -- (Retired) Row 3: post-processed at optimal threshold -----------------
    # Re-enable together with the threshold-optimisation block above.
    #
    # post_row3 = segment_predictions_to_events(
    #     y_pred=(y_prob >= optimal_threshold).astype(int), y_prob=y_prob,
    #     segment_len_sec=SEGMENT_SEC, step_sec=STEP_SEC,
    #     min_event_duration_sec=MIN_EVENT_SEC, refractory_period_sec=REFRACTORY_SEC,
    #     smoothing_window=SMOOTHING_WIN, threshold=optimal_threshold)
    # far_row3 = compute_event_level_far(
    #     y_true_segments=y_true, post_processed=post_row3,
    #     step_sec=STEP_SEC, segment_len_sec=SEGMENT_SEC)
    # y_pred_row3 = post_row3["smoothed_preds"]
    # row3_metrics = compute_all_metrics(
    #     y_true, y_pred_row3, post_row3["smoothed_probs"], SEGMENT_SEC, logger,
    #     label="Row3_postproc_opt%.3f" % optimal_threshold)
    # row3_metrics["threshold"] = optimal_threshold
    # row3_metrics["postprocessed"] = True
    # row3_metrics["far_per_hour_event"] = round(float(far_row3["far_per_hour"]), 6)
    # row3_metrics["n_true_alarms"] = far_row3["n_true_alarms"]
    # row3_metrics["n_false_alarms"] = far_row3["n_false_alarms"]
    # row3_metrics["n_total_events"] = far_row3["n_total_events"]

    return (row1_metrics, row2_metrics, post_row2, far_row2)


# ---------------------------------------------------------------------------
# save_all_results
# ---------------------------------------------------------------------------
def save_all_results(history, row1_metrics, row2_metrics,
                     far_row2, backbone_hp, branch_dilations, attn_hp,
                     best_epoch, best_val_f1, elapsed, device, n_params,
                     y_true, y_pred_row1, y_pred_row2, logger):
    """Save all structured results to files (no figures).

    Row 3 (threshold-optimised post-processing) has been retired due to
    poor recall on the ~0.27% ictal prevalence. Only Row 1 (raw t=0.5)
    and Row 2 (post-processed t=0.5) are persisted.
    """
    ks = int(backbone_hp["kernel_size"])
    branch_rfs = {}
    for bname, dils in branch_dilations.items():
        rf = 1 + 2 * sum((ks - 1) * d for d in dils)
        branch_rfs[bname] = {"samples": rf, "seconds": round(rf / FS, 4)}

    # -- a. Training log JSON --------------------------------------------------
    # DO NOT REMOVE THIS GUARD -- required by MultiScaleTCNAttention_evaluation.py,
    # which calls save_all_results with history=None. Removing it will
    # cause that script to crash with TypeError on history["epoch"].
    if history and history.get("epoch"):
        train_log = {
            "model": MODEL_NAME,
            "timestamp": datetime.datetime.now().isoformat(),
            "total_epochs": len(history["epoch"]),
            "best_epoch": best_epoch,
            "best_val_f1": round(best_val_f1, 6),
            "early_stopped": len(history["epoch"]) < MAX_EPOCHS,
            "duration_seconds": round(elapsed.total_seconds(), 1),
            "backbone_hyperparameters": backbone_hp,
            "branch_dilations": branch_dilations,
            "attention_hyperparameters": attn_hp,
            "backbone_params_source": str(BACKBONE_PARAMS_PATH),
            "attention_params_source": str(ATTN_PARAMS_PATH),
            "training_note": ("All parameters (backbone + attention) trained jointly end-to-end for "
                              "100 epochs. Tuning (tune_multiscale_attention.py) is also end-to-end "
                              "with backbone architecture HPs fixed from best_multiscale_params.json."),
            "branch_receptive_fields": branch_rfs,
            "trainable_params": n_params,
            "device": str(device),
            "history": history,
        }
        with open(TRAIN_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(train_log, f, indent=2)
        logger.info("Saved: %s", TRAIN_LOG_PATH)
    else:
        logger.info("Skipping training log JSON (no training history -- "
                    "test-eval invocation).")

    # -- b. Evaluation report JSON ---------------------------------------------
    eval_report = {
        "model": MODEL_NAME,
        "timestamp": datetime.datetime.now().isoformat(),
        "weights_path": str(WEIGHTS_PATH),
        "evaluation_set": "validation",
        "note": "Test set reserved for final_evaluation.py",
        "backbone_hyperparameters": backbone_hp,
        "branch_dilations": branch_dilations,
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
    }
    with open(EVAL_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)
    logger.info("Saved: %s", EVAL_REPORT_PATH)

    # -- c. Epoch metrics CSV --------------------------------------------------
    # DO NOT REMOVE THIS GUARD -- required by MultiScaleTCNAttention_evaluation.py,
    # which calls save_all_results with history=None. Removing it will
    # cause that script to crash with TypeError on history["epoch"].
    if history and history.get("epoch"):
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
    else:
        logger.info("Skipping per-epoch metrics CSV (no training history).")

    # -- d. Three-row summary CSV (M4 in ablation table) -----------------------
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
    # The base sklearn report covers per-class precision/recall/F1/support and
    # macro/weighted averages. We augment it with three top-level keys --
    # auroc, prauc, specificity -- so each row's JSON is self-contained and a
    # reader does not need to cross-reference the three-row CSV. For Row 2 and
    # Row 3, AUROC/PRAUC reflect the post-processing-smoothed continuous score
    # (see run_postprocessing_evaluations comment).
    for y_pred_row, row_metrics, row_label in [
        (y_pred_row1, row1_metrics, "row1"),
        (y_pred_row2, row2_metrics, "row2"),
    ]:
        report_dict = classification_report(
            y_true, y_pred_row, target_names=["Non-ictal", "Ictal"],
            output_dict=True, zero_division=0)
        report_dict["auroc"]       = row_metrics.get("auroc")
        report_dict["prauc"]       = row_metrics.get("average_precision")
        report_dict["specificity"] = row_metrics.get("specificity")
        rpath = OUTPUT_ROOT / ("ms_attn_classification_report_%s.json" % row_label)
        with open(rpath, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)
        logger.info("Saved: %s", rpath)

    # -- f. Event detail CSVs --------------------------------------------------
    for far_result, row_label in [(far_row2, "row2")]:
        csv_path = OUTPUT_ROOT / ("ms_attn_event_details_%s.csv" % row_label)
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
                    y_pred_row1, y_pred_row2,
                    row1_metrics, row2_metrics,
                    post_row2, logger):
    """Produce and save figures at dpi=150.

    Row 3 / threshold-optimisation figures (confusion matrix row 3,
    threshold curve, Row 3 contributions to ROC/metrics/FAR/timeline)
    have been retired. The remaining figures cover Row 1 + Row 2 only.
    """
    pfx = "ms_attn"

    # -- Figures 1 & 2: training-curve and LR-schedule -------------------------
    # Both figures depend on per-epoch training history. They are emitted on
    # every training run (history is always populated at the end of the loop)
    # and suppressed for test-evaluation invocations that pass history=None
    # or {}, so the eval folder contains only test-side artefacts.
    # DO NOT REMOVE THIS GUARD -- required by MultiScaleTCNAttention_evaluation.py,
    # which calls plot_all_figures with history=None. Removing it will
    # cause that script to crash with TypeError on history["epoch"].
    if history and history.get("epoch"):
        epochs = history["epoch"]

        # -- Figure 1: Training curves ----------------------------------------
        # Each panel overlays the raw per-epoch trace (faint) with an EMA-smoothed
        # trend (alpha=0.6, TensorBoard-default) -- val F1 is noisy at this dataset
        # prevalence and the smoothed line makes the convergence trajectory legible.
        EMA_ALPHA = 0.6
        train_loss_smoothed = ema_smooth(history["train_loss"], alpha=EMA_ALPHA)
        val_f1_smoothed     = ema_smooth(history["val_f1"], alpha=EMA_ALPHA)

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        axes[0].plot(epochs, history["train_loss"], color="#5A7DC8",
                     linewidth=1.0, alpha=0.25, label="Train loss (raw)")
        axes[0].plot(epochs, train_loss_smoothed, color="#5A7DC8",
                     linewidth=1.6, label="Train loss (EMA, α=0.6)")
        axes[0].axvline(best_epoch, linestyle="--", color="#C85A5A", alpha=0.7, label="Best epoch")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("BCEWithLogitsLoss")
        axes[0].set_title("MS-TCN+Attention Training Loss"); axes[0].legend(fontsize=9)
        axes[1].plot(epochs, history["val_f1"], color="#5A7DC8",
                     linewidth=1.0, alpha=0.25, label="Val F1 (raw)")
        axes[1].plot(epochs, val_f1_smoothed, color="#5A7DC8",
                     linewidth=1.6, label="Val F1 (EMA, α=0.6)")
        axes[1].axvline(best_epoch, linestyle="--", color="#C85A5A", alpha=0.7, label="Best epoch")
        axes[1].annotate("%.4f" % best_val_f1, xy=(best_epoch, best_val_f1),
                         xytext=(5, -15), textcoords="offset points", fontsize=9, color="#C85A5A")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Macro F1-score")
        axes[1].set_title("MS-TCN+Attention Validation Macro F1"); axes[1].legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / ("%s_training_curves.png" % pfx), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved: %s_training_curves.png", pfx)

        # -- Figure 2: LR schedule --------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(epochs, history["lr"], color="#5A7DC8", linewidth=1.2)
        ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("Learning rate (log scale)")
        ax.set_title("MS-TCN+Attention Cosine Annealing LR")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / ("%s_lr_schedule.png" % pfx), dpi=150, bbox_inches="tight")
        plt.close()
    else:
        logger.info("Skipping training-curve and LR-schedule figures "
                    "(no training history -- test-eval invocation).")

    # -- Figures 3-4: Confusion matrices ---------------------------------------
    # Row 3 (threshold-optimised post-processing) has been retired; see
    # run_postprocessing_evaluations docstring.
    for row_idx, (yp, m, rl, th) in enumerate([
        (y_pred_row1, row1_metrics, "Raw", 0.5),
        (y_pred_row2, row2_metrics, "Post-processed", 0.5),
    ], 1):
        cm = confusion_matrix(y_true, yp, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Non-ictal", "Ictal"], yticklabels=["Non-ictal", "Ictal"],
                    linewidths=0.5, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title("M4 %s | t=%.3f | Sens=%.3f Spec=%.3f F1=%.3f J=%.3f" % (
            rl, th, m["recall"], m["specificity"], m["f1_macro"], m["youden_j"]))
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / ("%s_confusion_matrix_row%d.png" % (pfx, row_idx)),
                    dpi=150, bbox_inches="tight")
        plt.close()

    # -- Figure 6: ROC curve ---------------------------------------------------
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc_val = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="#5A7DC8", linewidth=1.5,
            label="MS-TCN+Attn (AUROC = %.4f)" % auroc_val)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5, label="Chance")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#5A7DC8")
    for m, lbl, mk in [(row1_metrics, "Raw", "o"), (row2_metrics, "Post-processed", "s")]:
        ax.scatter([1 - m["specificity"]], [m["recall"]], marker=mk, s=60, zorder=5, label=lbl)
    ax.set_xlabel("FPR (1 - Specificity)"); ax.set_ylabel("TPR (Sensitivity)")
    ax.set_title("MS-TCN+Attention ROC -- Validation"); ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / ("%s_roc_curve.png" % pfx), dpi=150, bbox_inches="tight")
    plt.close()

    # -- Figure 7: Threshold curve (RETIRED) -----------------------------------
    # Threshold optimisation is disabled (see run_postprocessing_evaluations);
    # without an optimal threshold there is nothing to plot here. The figure
    # is intentionally left out of the output set.

    # -- Figure 8: Metrics comparison ------------------------------------------
    mnames = ["accuracy", "precision", "recall", "specificity", "f1_macro", "auroc", "average_precision"]
    r1 = [row1_metrics.get(m, 0) for m in mnames]
    r2 = [row2_metrics.get(m, 0) for m in mnames]
    fig, (ax_t, ax_b) = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [7, 3]})
    xp = np.arange(len(mnames)); w = 0.35
    b1 = ax_t.bar(xp - w / 2, r1, w, color="#E8A87C", label="Raw")
    b2 = ax_t.bar(xp + w / 2, r2, w, color="#5A7DC8", label="Post-processed")
    for bars in [b1, b2]:
        for bar in bars:
            ax_t.annotate("%.2f" % bar.get_height(),
                          xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                          xytext=(0, 2), textcoords="offset points", ha="center", fontsize=6)
    ax_t.set_xticks(xp); ax_t.set_xticklabels(mnames, fontsize=8)
    ax_t.set_ylabel("Score"); ax_t.set_title("MS-TCN+Attention Evaluation Metrics")
    ax_t.legend(fontsize=8); ax_t.set_ylim(0, 1.15)
    fv = [row1_metrics.get("far_per_hour_seg", 0),
          row2_metrics.get("far_per_hour_event", 0)]
    fb = ax_b.bar(["Raw\nseg", "Post-processed\nevent"], fv,
                  color=["#C85A5A", "#5A7DC8"])
    for bar in fb:
        ax_b.annotate("%.2f" % bar.get_height(),
                      xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8)
    ax_b.set_ylabel("FAR/hr"); ax_b.set_title("False Alarm Rate per Hour")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / ("%s_metrics_comparison.png" % pfx), dpi=150, bbox_inches="tight")
    plt.close()

    # -- Figure 9: PR curve ----------------------------------------------------
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    prev = np.mean(y_true)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rec_arr, prec_arr, color="#5A7DC8", linewidth=1.5,
            label="PR Curve (AP=%.4f)" % ap)
    ax.axhline(prev, linestyle="--", color="gray", alpha=0.5, label="No-skill (%.3f)" % prev)
    ax.fill_between(rec_arr, prec_arr, alpha=0.08, color="#5A7DC8")
    # Overlay Row 1 (raw) and Row 2 (post-processed) operating points.
    # Row 1 sits on the curve (single threshold of y_prob); Row 2 typically
    # sits off the curve because post-processing is non-monotone.
    r1_recall = float(row1_metrics.get("recall", 0))
    r1_precision = float(row1_metrics.get("precision", 0))
    r2_recall = float(row2_metrics.get("recall", 0))
    r2_precision = float(row2_metrics.get("precision", 0))
    ax.scatter([r1_recall], [r1_precision],
               color="#5A7DC8", s=70, zorder=5, edgecolor="black", linewidth=0.6)
    ax.annotate("Raw (τ=0.5)", xy=(r1_recall, r1_precision),
                xytext=(8, -4), textcoords="offset points", fontsize=9)
    ax.scatter([r2_recall], [r2_precision],
               color="#E8A87C", s=70, zorder=5, edgecolor="black", linewidth=0.6)
    ax.annotate("Post-proc (τ=0.5)", xy=(r2_recall, r2_precision),
                xytext=(8, -4), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("MS-TCN+Attention PR Curve with Operating Points")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / ("%s_pr_curve.png" % pfx), dpi=150, bbox_inches="tight")
    plt.close()

    # -- Figure 10: Calibration curve ------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 5))
    try:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
        ax.plot(mean_pred, frac_pos, "o-", color="#5A7DC8", label="MS-TCN+Attention")
    except ValueError:
        pass
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5, label="Perfect")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction of positives")
    ax.set_title("MS-TCN+Attention Calibration Curve"); ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / ("%s_calibration_curve.png" % pfx), dpi=150, bbox_inches="tight")
    plt.close()

    # -- Figure 11: FAR comparison ---------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    fl = ["Raw\nsegment-level", "Post-processed\nevent-level"]
    fvals = [row1_metrics.get("far_per_hour_seg", 0),
             row2_metrics.get("far_per_hour_event", 0)]
    bars = ax.bar(fl, fvals, color=["#C85A5A", "#5A7DC8"], edgecolor="white")
    for bar in bars:
        ax.annotate("%.2f" % bar.get_height(),
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    ax.set_ylabel("FAR/hr"); ax.set_title("MS-TCN+Attention FAR/hr: segment vs event")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / ("%s_far_comparison.png" % pfx), dpi=150, bbox_inches="tight")
    plt.close()

    # -- Figure 12: Segment length analysis ------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    durations = []
    for post in [post_row2]:
        for evt in post.get("events", []):
            durations.append(evt.get("duration_sec", 0))
    if durations:
        axes[0].hist(durations, bins=20, color="#5A7DC8", edgecolor="white", alpha=0.85)
        axes[0].axvline(MIN_EVENT_SEC, linestyle="--", color="#C85A5A", linewidth=1.2,
                        label="Min = %.0fs" % MIN_EVENT_SEC)
        axes[0].legend(fontsize=8)
    axes[0].set_xlabel("Event duration (s)"); axes[0].set_ylabel("Count")
    axes[0].set_title("MS-TCN+Attention Event Duration Distribution")
    evts_r2 = compute_event_level_far(y_true, post_row2, STEP_SEC, SEGMENT_SEC).get("event_details", [])
    if evts_r2:
        axes[1].scatter([e["start_sec"] for e in evts_r2], [e["duration_sec"] for e in evts_r2],
                        c=["#41B3A3" if e["is_true_alarm"] else "#C85A5A" for e in evts_r2],
                        s=30, alpha=0.7)
        axes[1].axhline(MIN_EVENT_SEC, linestyle="--", color="gray", alpha=0.5)
        axes[1].scatter([], [], c="#41B3A3", label="True alarm")
        axes[1].scatter([], [], c="#C85A5A", label="False alarm")
        axes[1].legend(fontsize=8)
    axes[1].set_xlabel("Event start (s)"); axes[1].set_ylabel("Duration (s)")
    axes[1].set_title("Event Timeline -- Post-processed")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / ("%s_segment_length_analysis.png" % pfx), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: all standard figures (Row 1 + Row 2 only)")


# ---------------------------------------------------------------------------
# plot_attention_saliency
# ---------------------------------------------------------------------------
def plot_attention_saliency(model, val_loader, y_true, device, logger):
    """Produce temporal attention saliency figure (Figure 13).

    Extracts attention weights for all validation segments using
    MultiScaleTCNWithAttention.get_attention_weights(). Plots mean saliency
    profiles averaged over ictal and non-ictal segments separately.
    """
    model.eval()
    all_weights = []
    all_labels = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            w = model.get_attention_weights(x)
            all_weights.extend(w)
            all_labels.extend(y.numpy())
    weights = np.array(all_weights)
    labels = np.array(all_labels, dtype=int)

    ictal_weights = weights[labels == 1]
    nonictal_weights = weights[labels == 0]
    logger.info("Attention saliency: %d ictal, %d non-ictal segments",
                ictal_weights.shape[0], nonictal_weights.shape[0])

    if ictal_weights.shape[0] == 0 or nonictal_weights.shape[0] == 0:
        logger.warning("Cannot plot saliency: one class has zero segments.")
        return

    mean_ictal = ictal_weights.mean(axis=0)
    mean_nonictal = nonictal_weights.mean(axis=0)
    std_ictal = ictal_weights.std(axis=0)
    std_nonictal = nonictal_weights.std(axis=0)

    T = mean_ictal.shape[0]
    time_axis = np.arange(T) / FS

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9))

    # -- Subplot 1: mean attention per class -----------------------------------
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
    diff = mean_ictal - mean_nonictal
    ax2.fill_between(time_axis, 0, diff, where=(diff >= 0), color="#C85A5A", alpha=0.4,
                     label="Ictal > Non-ictal")
    ax2.fill_between(time_axis, 0, diff, where=(diff < 0), color="#5A7DC8", alpha=0.4,
                     label="Non-ictal > Ictal")
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Delta alpha (ictal - non-ictal)")
    ax2.set_title("Differential Attention: Ictal minus Non-ictal")
    ax2.legend(fontsize=8)

    # -- Subplot 3: example ictal segment with attention overlay ---------------
    ictal_indices = np.where(labels == 1)[0]
    mean_per_seg = weights[ictal_indices].mean(axis=1)
    best_seg_local = np.argmax(mean_per_seg)
    best_seg_global = ictal_indices[best_seg_local]

    seg_path, seg_label = val_loader.dataset.pairs[best_seg_global]
    raw_eeg = np.load(seg_path).astype(np.float32)
    raw_max = np.abs(raw_eeg).max()
    if raw_max > 0:
        raw_eeg = raw_eeg / raw_max
    eeg_time = np.arange(len(raw_eeg)) / FS
    seg_weights = weights[best_seg_global]

    ax3.plot(eeg_time, raw_eeg, color="gray", linewidth=0.5, alpha=0.7, label="EEG")
    w_time = np.arange(len(seg_weights)) / FS
    scatter = ax3.scatter(w_time, np.full_like(seg_weights, raw_eeg.min() - 0.15),
                          c=seg_weights, cmap="Reds", s=4, vmin=0,
                          vmax=seg_weights.max() if seg_weights.max() > 0 else 1)
    fig.colorbar(scatter, ax=ax3, label="Attention weight", shrink=0.6)
    ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Normalised amplitude")
    ax3.set_title("Example Ictal Segment with Attention Overlay (seg #%d)" % best_seg_global)
    ax3.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "ms_attn_saliency_maps.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: ms_attn_saliency_maps.png")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    """Main entry point for MultiScaleTCN + Temporal Attention training (M4).

    Trains for MAX_EPOCHS=100 with early stopping. All params joint.
    Does not evaluate the test set.
    """
    # -- Step 1: Logging and setup ---------------------------------------------
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("MultiScaleTCNAttention.py")
    logger.info("Timestamp       : %s", datetime.datetime.now().isoformat())
    logger.info("Ablation role   : M4 -- Multi-Scale TCN + Temporal Attention")
    logger.info("Params source   : best_multiscale_params.json + best_multiscale_attn_params.json")
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
        # Pre-flight GPU memory check: warn if another job is already on this
        # GPU or if VRAM is fragmented. Does not abort -- the user may still
        # want to run on a partially-occupied GPU at reduced batch size.
        try:
            _free_bytes, _total_bytes = torch.cuda.mem_get_info(0)
            _free_gb = _free_bytes / 1e9
            _total_gb = _total_bytes / 1e9
            logger.info("GPU memory free: %.2f / %.2f GB", _free_gb, _total_gb)
            if _free_gb < 8.0:
                logger.warning("GPU has only %.2f GB free (< 8 GB threshold). "
                               "Another process may be sharing this GPU, or VRAM "
                               "is fragmented. Training may fail with CUDA OOM.",
                               _free_gb)
        except Exception as _e:
            logger.warning("Could not query GPU memory: %s", _e)
    else:
        logger.info("Device: CPU")
    logger.info("PyTorch: %s", torch.__version__)

    # -- Step 2: Load both parameter files -------------------------------------
    backbone_config, backbone_hp, branch_dilations, attn_config, attn_hp = load_best_params(logger)

    # -- Step 3: Load data splits ----------------------------------------------
    train_pairs, val_pairs, val_records = load_splits(logger)

    # -- Corpus preparation ----------------------------------------------------
    # Subject exclusion (m254), proximity-aware non-ictal downsampling to a
    # 1:2.37 ictal-to-non-ictal ratio, and extreme-segment filtering are ALL
    # handled offline by create_balanced_splits.py. The manifest is already
    # clean and balanced -- no further corpus preparation is needed here.
    # train_pairs = filter_unpaired_subjects(train_pairs, logger=logger)
    logger.info("Training corpus: %d segments (from balanced manifest)", len(train_pairs))
    pos_weight = torch.tensor([1.0], dtype=torch.float32)
    # -- End corpus preparation ------------------------------------------------

    # -- Step 4: Build model ---------------------------------------------------
    model = build_model(backbone_hp, branch_dilations, attn_hp, DEVICE, logger)
    n_params = count_parameters(model)

    # -- Pre-flight check: verify a small sample of .npy files is readable -----
    # Fail fast if the manifest's paths are stale or /scratch is inaccessible,
    # rather than discovering this N retries per file into the first epoch.
    # Probe 5 files per partition (train + val) to cover both mounts in case
    # the partitions live on different storage (e.g. TRAIN_DATA_* vs VAL_DATA).
    _preflight_rng = random.Random(SEED)
    _probe_paths = [_preflight_rng.choice(train_pairs)[0] for _ in range(5)]
    _probe_paths += [_preflight_rng.choice(val_pairs)[0] for _ in range(5)]
    for _probe_path in _probe_paths:
        try:
            _ = np.load(_probe_path)
        except Exception as _e:
            raise RuntimeError(
                "Pre-flight check failed: cannot read %s. "
                "Check that /scratch is accessible and the manifest is current. "
                "Underlying error: %s" % (_probe_path, _e))
    logger.info("Pre-flight .npy read OK (%d probes, train+val): %s, ...",
                len(_probe_paths), _probe_paths[0])
    del _probe_path, _probe_paths, _preflight_rng

    # -- Build 5% stratified val subset for per-epoch monitoring ---------------
    # Validating on the full val set every epoch is I/O-bound (~11h/epoch on
    # 4.3M segments). For early-stopping decisions we only need a representative
    # F1 estimate, so we monitor on a stratified subset that preserves the
    # natural ictal:non-ictal prevalence. The full val set is reserved for the
    # post-training pass (three-row evaluation, smoothing/refractory/min-
    # duration post-processing, Youden's-J threshold optimisation, events/24h
    # false-alarm rate, and attention saliency visualisation). Seeded from
    # SEED+1 so the monitor subset is identical across SLURM resubmissions,
    # keeping the early-stop signal stable.
    _monitor_rng = random.Random(SEED + 1)
    _ictal_val = [p for p in val_pairs if p[1] == 1]
    _nonictal_val = [p for p in val_pairs if p[1] == 0]
    _n_ictal_mon = max(1, int(round(len(_ictal_val) * VAL_MONITOR_FRACTION)))
    _n_nonictal_mon = max(1, int(round(len(_nonictal_val) * VAL_MONITOR_FRACTION)))
    val_monitor_pairs = (_monitor_rng.sample(_ictal_val, _n_ictal_mon) +
                         _monitor_rng.sample(_nonictal_val, _n_nonictal_mon))
    _monitor_rng.shuffle(val_monitor_pairs)
    logger.info("Val monitor subset (%.0f%% stratified): %d segments | "
                "%d ictal + %d non-ictal | full val reserved for final eval",
                VAL_MONITOR_FRACTION * 100, len(val_monitor_pairs),
                _n_ictal_mon, _n_nonictal_mon)
    del _ictal_val, _nonictal_val, _n_ictal_mon, _n_nonictal_mon, _monitor_rng

    # -- Step 5: Build data loaders --------------------------------------------
    batch_size = int(attn_hp["batch_size"])
    train_loader = make_loader(train_pairs, batch_size, True, DEVICE)
    val_monitor_loader = make_loader(val_monitor_pairs, batch_size, False, DEVICE)
    val_loader = make_loader(val_pairs, batch_size, False, DEVICE)
    logger.info("Train loader       : %d batches | batch_size=%d", len(train_loader), batch_size)
    logger.info("Val monitor loader : %d batches (per-epoch early-stop)", len(val_monitor_loader))
    logger.info("Full val loader    : %d batches (post-training eval only)", len(val_loader))

    # -- Step 6: Build training components -------------------------------------
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
    latest_path = CKPT_DIR / "ms_attn_latest.pt"
    best_ckpt_path = CKPT_DIR / "ms_attn_best.pt"
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
        if "best_model_state" in ckpt:
            best_state = {k: v.cpu().clone() for k, v in ckpt["best_model_state"].items()}
        else:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        logger.info("RESUMED from %s: epoch %d, best_val_f1=%.4f (epoch %d), patience=%d/%d",
                    resume_path.name, ckpt["epoch"], best_val_f1,
                    best_epoch, min(epochs_no_imp, ES_PATIENCE), ES_PATIENCE)
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
    logger.info("  ES gate     : disabled until epoch %d (Regime B gate)", MIN_EPOCHS_BEFORE_ES)
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
        # Per-epoch monitor on the 5% stratified subset (early-stopping signal).
        # Full val is run only once after training, in the final eval step.
        val_f1, _, _, _ = evaluate_model(model, val_monitor_loader, DEVICE, logger, use_amp=use_amp)
        val_sec = time.time() - t0_val

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)

        # Log every epoch -- 100-epoch budget is short enough for full visibility
        # in the .log file. Per-epoch CSV (..._epoch_metrics.csv) remains the
        # primary source for downstream analysis.
        logger.info(
            "Epoch %3d/%d | loss=%.4f | val_f1=%.4f | lr=%.2e | best=%.4f | patience=%d/%d"
            " | train %.0fs | val %.0fs",
            epoch, MAX_EPOCHS, train_loss, val_f1, current_lr,
            best_val_f1, min(epochs_no_imp, ES_PATIENCE), ES_PATIENCE, train_sec, val_sec)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_no_imp = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            save_checkpoint(epoch, model, optimiser, scheduler, val_f1, train_loss, attn_hp,
                            best_ckpt_path, logger,
                            best_model_state=best_state,
                            best_val_f1=best_val_f1, best_epoch=best_epoch,
                            epochs_no_imp=epochs_no_imp)
            logger.info("  New best val F1: %.4f at epoch %d", best_val_f1, best_epoch)
        else:
            epochs_no_imp += 1

        # Save latest.pt every epoch — single file for clean resume
        save_checkpoint(epoch, model, optimiser, scheduler, val_f1, train_loss, attn_hp,
                        latest_path, logger,
                        best_model_state=best_state,
                        best_val_f1=best_val_f1, best_epoch=best_epoch,
                        epochs_no_imp=epochs_no_imp)

        if epoch % CHECKPOINT_FREQ == 0:
            cleanup_checkpoints(CKPT_DIR, KEEP_CKPTS, logger)

        # Regime B gate: ES is disabled until epoch >= MIN_EPOCHS_BEFORE_ES so
        # that the cosine LR schedule reaches the fine-tuning phase before
        # early termination becomes possible.
        if epoch >= MIN_EPOCHS_BEFORE_ES and epochs_no_imp >= ES_PATIENCE:
            logger.info("Early stopping at epoch %d (ES gate open since epoch %d).",
                        epoch, MIN_EPOCHS_BEFORE_ES)
            break

    # -- Step 9: Restore best weights ------------------------------------------
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
    torch.save(model.cpu().state_dict(), WEIGHTS_PATH)
    size_mb = WEIGHTS_PATH.stat().st_size / 1e6
    logger.info("Weights saved : %s (%.2f MB)", WEIGHTS_PATH, size_mb)
    model.to(DEVICE)

    # -- Step 11: Final evaluation ---------------------------------------------
    # Two passes after weight restoration:
    #   (a) Monitor-subset re-evaluation -- a sanity check that compares the
    #       restored model's F1 on the same 5% subset against best_val_f1.
    #       Both numbers come from the identical subset, so a mismatch beyond
    #       floating-point tolerance signals a weight-restoration failure.
    #   (b) Full-val pass -- produces the y_true and y_prob arrays consumed by
    #       run_postprocessing_evaluations() for the three-row report,
    #       smoothing/refractory/min-duration filtering, Youden's-J threshold
    #       optimisation, events/24h FAR, and attention saliency. This is the
    #       only pass that uses the full 4.3M-segment val set.
    logger.info("Sanity-check on monitor subset (verifies weight restoration)...")
    val_f1_monitor, _, _, _ = evaluate_model(model, val_monitor_loader, DEVICE, logger, use_amp=use_amp)
    tol = 1e-3
    if abs(val_f1_monitor - best_val_f1) > tol:
        logger.warning("F1 mismatch on monitor subset: best=%.6f restored=%.6f. "
                       "Weight restoration may have failed.", best_val_f1, val_f1_monitor)
    else:
        logger.info("F1 consistency check (monitor subset): PASS (%.4f)", val_f1_monitor)

    logger.info("Running full-val final evaluation (post-processing inputs)...")
    val_f1_final, y_true, y_pred_05, y_prob = evaluate_model(model, val_loader, DEVICE, logger, use_amp=False)
    logger.info("Full-val F1 (raw @ 0.5): %.4f", val_f1_final)

    # -- Step 12: Post-processing evaluations (Row 3 retired) ------------------
    # Legacy call -- still used by save_all_results for the per-row sklearn
    # classification reports + epoch CSV (segment-level outputs that are
    # order-invariant). The event-level FAR/hr it produces (far_row2) is
    # superseded by Step 12b below and overwritten in the eval report.
    (row1_metrics, row2_metrics,
     post_row2, far_row2) = run_postprocessing_evaluations(y_true, y_prob, logger)

    # -- Step 12b: Corrected per-mouse-chronological event-level evaluation ----
    logger.info("Running per-mouse-chronological event-level evaluation "
                "(eval_utils.evaluate_event_level)...")
    if not MOUSE_METADATA_PATH.exists():
        logger.error("Mouse metadata not found at %s. Run extract_mouse_metadata.py first.",
                     MOUSE_METADATA_PATH)
        raise FileNotFoundError(str(MOUSE_METADATA_PATH))
    mouse_metadata = json.loads(MOUSE_METADATA_PATH.read_text(encoding="utf-8"))
    val_eval_result = evaluate_event_level(
        val_records, y_true, y_prob, ANNOT_DIR, mouse_metadata, logger)

    # -- Step 13: Save all results ---------------------------------------------
    # y_pred_row1 / y_pred_row2 come from the corrected per-mouse-chronological
    # pipeline. The val NPZ written below is in chronological order with the
    # enriched schema (mouse_id, chrono_idx, t_start_sec) per Phase B B1a.
    reord = val_eval_result["reordered_arrays"]
    y_pred_row1 = reord["y_pred_row1"]
    y_pred_row2 = reord["y_pred_row2"]

    np.savez_compressed(
        VAL_PREDICTIONS_NPZ,
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
    logger.info("Saved enriched val predictions bundle: %s (%.2f MB)",
                VAL_PREDICTIONS_NPZ, VAL_PREDICTIONS_NPZ.stat().st_size / 1e6)

    # save_all_results writes legacy artefacts (training log, classification
    # reports, epoch CSV). The eval_report.json it produces contains the
    # broken event-level FAR/hr; Step 13b overwrites it with the corrected
    # schema. Pass manifest-order y_true with manifest-order y_pred so the
    # legacy per-row classification reports are internally consistent.
    save_all_results(
        history, row1_metrics, row2_metrics,
        far_row2, backbone_hp, branch_dilations, attn_hp,
        best_epoch, best_val_f1, elapsed, DEVICE, n_params, y_true,
        (y_prob >= 0.5).astype(int), post_row2["smoothed_preds"], logger)

    # -- Step 13b: Overwrite eval report with corrected event-level metrics ----
    corrected_report = {
        "model":     MODEL_NAME,
        "timestamp": datetime.datetime.now().isoformat(),
        "weights_path":   str(WEIGHTS_PATH),
        "evaluation_set": "validation",
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
        "segment_level_metrics": val_eval_result["segment_level_metrics"],
        "event_level_metrics":   val_eval_result["event_level_metrics"],
        "per_mouse":             val_eval_result["per_mouse_results"],
    }
    with open(EVAL_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(corrected_report, f, indent=2, default=str)
    logger.info("Overwrote eval report with corrected schema: %s", EVAL_REPORT_PATH)

    # -- Step 13c: Overwrite event-details CSV with corrected per-event rows ---
    event_details_path = OUTPUT_ROOT / "ms_attn_event_details_row2.csv"
    fieldnames_evt = ["mouse_id", "is_true_alarm", "start_sec", "end_sec",
                      "duration_sec", "max_prob", "matched_gt_idx",
                      "matched_gt_start_sec", "matched_gt_end_sec"]
    with open(event_details_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_evt)
        writer.writeheader()
        for row in val_eval_result["all_event_details"]:
            writer.writerow(row)
    logger.info("Overwrote event-details CSV with corrected rows: %s (%d rows)",
                event_details_path, len(val_eval_result["all_event_details"]))

    # -- Step 13d: Overwrite three-row summary CSV with corrected values -------
    seg = val_eval_result["segment_level_metrics"]
    em  = val_eval_result["event_level_metrics"]
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
    with open(THREE_ROW_CSV, "w", newline="", encoding="utf-8") as f:
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
    logger.info("Overwrote three-row summary with corrected values: %s", THREE_ROW_CSV)

    em = val_eval_result["event_level_metrics"]
    logger.info("=" * 65)
    logger.info("CORRECTED EVENT-LEVEL VAL METRICS")
    logger.info("  TP / FP / FN          : %d / %d / %d", em["tp"], em["fp"], em["fn"])
    logger.info("  Precision             : %.4f", em["precision"])
    logger.info("  Recall (sensitivity)  : %.4f", em["recall"])
    logger.info("  F1                    : %.4f", em["f1"])
    logger.info("  FAR/hr event CORRECTED: %.4f  (non-ictal hours = %.1f)",
                em["far_per_hour_event_CORRECTED"],
                val_eval_result["totals"]["non_ictal_hours_corrected"])
    logger.info("  Mean detection latency: %s s", em["mean_detection_latency_sec"])
    logger.info("=" * 65)

    # -- Step 14: Plot standard figures ----------------------------------------
    plot_all_figures(
        history, best_epoch, best_val_f1, y_true, y_prob,
        y_pred_row1, y_pred_row2,
        row1_metrics, row2_metrics,
        post_row2, logger)

    # -- Step 14b: Result_classReport bar plot ---------------------------------
    # Macro-avg classification metrics (Row 1 + Row 2) plus FAR/hr in a
    # standalone two-panel figure under OUTPUT_ROOT/Result_classReport/.
    # Same protocol as M1, M2, M3 -- shared helper in tcn_utils.py.
    make_classreport_barplot(
        y_true, y_pred_row1, y_pred_row2,
        row1_metrics, row2_metrics,
        RESULT_CLASSREPORT_DIR / "ms_attn_classreport_barplot.png",
        title_prefix="MS-TCN + Attention", logger=logger)

    # -- Step 15: Plot attention saliency (Figure 13) --------------------------
    plot_attention_saliency(model, val_loader, y_true, DEVICE, logger)

    # -- Step 16: Final inventory and cleanup ----------------------------------
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared.")

    logger.info("=" * 65)
    logger.info("ALL OUTPUTS SAVED")
    pfx = "ms_attn"
    all_outputs = [
        WEIGHTS_PATH,
        CKPT_DIR / "ms_attn_latest.pt",
        CKPT_DIR / "ms_attn_best.pt",
        TRAIN_LOG_PATH, EVAL_REPORT_PATH, EPOCH_CSV, THREE_ROW_CSV,
        OUTPUT_ROOT / "ms_attn_classification_report_row1.json",
        OUTPUT_ROOT / "ms_attn_classification_report_row2.json",
        OUTPUT_ROOT / "ms_attn_event_details_row2.csv",
        FIGURE_DIR / ("%s_training_curves.png" % pfx),
        FIGURE_DIR / ("%s_lr_schedule.png" % pfx),
        FIGURE_DIR / ("%s_confusion_matrix_row1.png" % pfx),
        FIGURE_DIR / ("%s_confusion_matrix_row2.png" % pfx),
        FIGURE_DIR / ("%s_roc_curve.png" % pfx),
        FIGURE_DIR / ("%s_metrics_comparison.png" % pfx),
        FIGURE_DIR / ("%s_pr_curve.png" % pfx),
        FIGURE_DIR / ("%s_calibration_curve.png" % pfx),
        FIGURE_DIR / ("%s_far_comparison.png" % pfx),
        FIGURE_DIR / ("%s_segment_length_analysis.png" % pfx),
        FIGURE_DIR / "ms_attn_saliency_maps.png",
        RESULT_CLASSREPORT_DIR / ("%s_classreport_barplot.png" % pfx),
    ]
    for p in all_outputs:
        status = "OK     " if Path(p).exists() else "MISSING"
        logger.info("  [%s] %s", status, p)

    logger.info("=" * 65)
    logger.info("ABLATION NOTE: Compare M3 (MultiScaleTCN) three_row_summary.csv "
                "with M4 (this script) ms_attn_three_row_summary.csv "
                "to isolate the contribution of temporal attention over multi-scale TCN.")
    logger.info("NEXT: run final_evaluation.py when test data is ready.")


if __name__ == "__main__":
    main()
