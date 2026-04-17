"""
tune_temporal_attention.py
==========================
Tunes the temporal attention hyperparameters of
TCNWithAttention from tcn_utils.py using Optuna.

Tuning protocol
----------------
N_TRIALS    = 50   total Optuna trials
MAX_EPOCHS  = 20   max epochs per trial (cosine annealing half-cycle)
ES_PATIENCE = 5    early stopping patience (epochs without val F1 improvement)

Early stopping at patience=5 within a 20-epoch budget means most trials
terminate between epochs 8-15. The reduced budget accelerates the search
while retaining enough epochs for the cosine schedule to differentiate
good from bad hyperparameter configurations (Li et al., 2017).

Validation subset: a stratified 10% subset of the validation partition
is used during tuning to reduce per-epoch evaluation cost. The subset
preserves the original class ratio and is fixed across all trials
(Falkner et al., 2018).

DataLoader optimisations: num_workers=4, persistent_workers=True,
prefetch_factor=4 overlap CPU-side data loading with GPU compute,
reducing per-epoch wall time on multi-core nodes (Mattson et al., 2020).

References
----------
Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A.
    (2017). Hyperband: A Novel Bandit-Based Approach to Hyperparameter
    Optimization. JMLR, 18(185), 1-52.
Falkner, S., Klein, A., & Hutter, F. (2018). BOHB: Robust and Efficient
    Hyperparameter Optimization at Scale. ICML 2018.
Mattson, P. et al. (2020). MLPerf Training Benchmark. MLSys 2020.

The TCN backbone is frozen. Only temporal attention
parameters (TemporalAttention scorer, LayerNorm, and
classification head) receive gradient updates during
each trial.

This script produces tuning outputs only.
It does NOT train a final model.
Final model training is performed by
train_model2_tcn_attention.ipynb, which loads
outputs/best_attention_params.json produced here.

Architecture note
-----------------
TCNWithAttention uses a TemporalAttention module that
comprises a single nn.Linear(embed_dim, 1) scorer.
Its architecture is fully determined by embed_dim
(= num_filters from the TCN backbone). There are no
configurable architectural attention hyperparameters
(no attention_dim, no num_heads, no attention_dropout).

The Optuna search space therefore consists of:
  - attention_dim   : projection dimension of scoring function
  - attention_dropout : dropout on context vector
  - learning_rate   : AdamW lr for attention params
  - weight_decay    : L2 regularisation coefficient
  - batch_size      : segments per gradient step

Pipeline position
-----------------
1. generate_data_splits.py --no-test
2. tcn_HPT_binary.ipynb         -> best_params.json
3. tune_temporal_attention.py   -> best_attention_params.json
4. train_model2_tcn_attention.ipynb
...

Usage
-----
python tune_temporal_attention.py

Outputs (no model weights are saved)
-------------------------------------
outputs/best_attention_params.json
outputs/attention_study_results.csv
outputs/attention_tuning_summary.json
outputs/logs/tune_temporal_attention.log
outputs/figures/attention_f1_history.png
outputs/figures/attention_importance.png
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
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")                                  # non-interactive backend for script use
import matplotlib.pyplot as plt

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from tcn_utils import (
    set_seed,
    make_loader,
    # filter_unpaired_subjects,  # handled offline by create_balanced_splits.py
    downsample_val_stratified,
    TCNWithAttention,
    count_parameters,
    train_one_epoch,
    evaluate,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED            = 42                                   # global reproducibility seed
MAX_EPOCHS      = 20                                   # max epochs per trial (ES fires before 20)
ES_PATIENCE     = 5                                    # early stopping patience (epochs)
N_TRIALS        = 50                                   # total Optuna trials
N_STARTUP       = 10                                   # random startup trials before TPE kicks in
FS              = 500                                  # EEG sampling rate (Hz)
SEGMENT_LEN     = 2500                                 # samples per segment (5 s at 500 Hz)
SEGMENT_SEC     = 5.0                                  # segment duration in seconds

OUTPUT_DIR      = Path("outputs")                      # all pipeline outputs
LOG_DIR         = OUTPUT_DIR / "logs"                   # log file directory
FIGURE_DIR      = OUTPUT_DIR / "figures"                # figure output directory
BEST_TCN_PATH   = OUTPUT_DIR / "best_params.json"       # fixed TCN backbone hyperparameters
# Previous (uniform downsampling): data_splits.json
# SPLITS_PATH     = Path("/scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits.json")
# Current (proximity-aware downsampling): data_splits_nonictal_sampled.json
SPLITS_PATH     = Path("/scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits_nonictal_sampled.json")
BEST_ATTN_PATH  = OUTPUT_DIR / "best_attention_params.json"
STUDY_CSV       = OUTPUT_DIR / "attention_study_results.csv"
SUMMARY_PATH    = OUTPUT_DIR / "attention_tuning_summary.json"
LOG_FILE        = LOG_DIR    / "tune_temporal_attention.log"
FIG_F1          = FIGURE_DIR / "attention_f1_history.png"
FIG_IMP         = FIGURE_DIR / "attention_importance.png"

# Temporal attention search ranges
# TCNWithAttention now uses a two-layer additive attention (tanh + linear)
# with attention_dim and attention_dropout as tunable architectural params,
# matching MultiScaleTCNWithAttention for consistent ablation design.
ATTN_DIM_CHOICES  = [32, 64, 128]                      # attention projection dimension
ATTN_DROP_MIN     = 0.0                                # attention dropout lower bound
ATTN_DROP_MAX     = 0.4                                # attention dropout upper bound
ATTN_DROP_STEP    = 0.05                               # attention dropout step size
LR_MIN            = 1e-4                               # AdamW learning rate lower bound
LR_MAX            = 1e-2                               # AdamW learning rate upper bound
WD_MIN            = 1e-5                               # weight decay lower bound
WD_MAX            = 1e-3                               # weight decay upper bound
BATCH_CHOICES     = [16, 32, 64]                       # batch size candidates (powers of 2)

# TCN backbone attribute prefix in TCNWithAttention (confirmed from tcn_utils.py)
# TCNWithAttention stores its backbone as self.tcn = nn.Sequential(...)
BACKBONE_PREFIX = "tcn."


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------
def setup_logging():
    """Create output directories and configure the module logger.

    Returns
    -------
    logger : logging.Logger
        Configured logger with FileHandler (DEBUG+) and
        StreamHandler (INFO+ to stdout).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("temporal_attention_tuning")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# load_best_tcn_params
# ---------------------------------------------------------------------------
def load_best_tcn_params(logger):
    """Load the best TCN hyperparameters from outputs/best_params.json.

    These parameters define the frozen TCN backbone and must not be
    included in the Optuna search space.

    Parameters
    ----------
    logger : logging.Logger

    Returns
    -------
    tuple of (full_config: dict, hp: dict)
        full_config : entire JSON content
        hp          : the 'hyperparameters' sub-dict containing TCN
                      backbone params (num_layers, num_filters,
                      kernel_size, dropout)

    Raises
    ------
    FileNotFoundError
        If best_params.json is absent.
    KeyError
        If 'hyperparameters' key is missing from the JSON.
    """
    if not BEST_TCN_PATH.exists():
        logger.error(
            "best_params.json not found at %s. "
            "Run tcn_HPT_binary.ipynb first.", BEST_TCN_PATH)
        raise FileNotFoundError(str(BEST_TCN_PATH))

    with open(BEST_TCN_PATH, "r", encoding="utf-8") as f:
        full_config = json.load(f)

    if "hyperparameters" not in full_config:
        logger.error(
            "'hyperparameters' key missing from %s. "
            "Expected keys: num_layers, num_filters, kernel_size, dropout.",
            BEST_TCN_PATH)
        raise KeyError("hyperparameters")

    hp = full_config["hyperparameters"]

    logger.info("Fixed TCN backbone hyperparameters:")
    for k, v in hp.items():
        logger.info("  %-20s: %s", k, v)

    return full_config, hp


# ---------------------------------------------------------------------------
# load_splits
# ---------------------------------------------------------------------------
def load_splits(logger):
    """Load train and val file-label pairs from data_splits.json.

    Parameters
    ----------
    logger : logging.Logger

    Returns
    -------
    tuple of (train_pairs, val_pairs)
        Each is a list of (filepath: str, label: int) tuples
        compatible with make_loader().

    Raises
    ------
    FileNotFoundError
        If data_splits.json is absent.
    RuntimeError
        If train or val lists are empty.
    """
    if not SPLITS_PATH.exists():
        logger.error(
            "data_splits.json not found at %s. "
            "Run generate_data_splits.py --no-test first.", SPLITS_PATH)
        raise FileNotFoundError(str(SPLITS_PATH))

    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)

    # Convert list-of-dicts to list-of-tuples: (filepath, label)
    train_pairs = [
        (rec["filepath"], rec["label"])
        for rec in splits["train"]
    ]
    val_pairs = [
        (rec["filepath"], rec["label"])
        for rec in splits["val"]
    ]

    if not train_pairs:
        logger.error("Train partition is empty in %s.", SPLITS_PATH)
        raise RuntimeError("Empty train partition")
    if not val_pairs:
        logger.error("Val partition is empty in %s.", SPLITS_PATH)
        raise RuntimeError("Empty val partition")

    # Log partition sizes and class composition
    for name, pairs in [("train", train_pairs), ("val", val_pairs)]:
        n_total = len(pairs)
        n_sz = sum(1 for _, l in pairs if l == 1)
        n_nsz = n_total - n_sz
        pct = n_sz / n_total * 100 if n_total > 0 else 0.0
        logger.info(
            "%s partition: %d total | %d seizure | %d non-seizure | %.1f%% ictal",
            name.upper(), n_total, n_sz, n_nsz, pct)

    return train_pairs, val_pairs


# ---------------------------------------------------------------------------
# optuna_objective
# ---------------------------------------------------------------------------
def optuna_objective(trial, train_pairs, val_pairs, tcn_hp, device, logger):
    """Optuna objective function for temporal attention tuning.

    The TCN backbone is instantiated with fixed parameters from tcn_hp
    and its weights are frozen immediately after the model is moved to
    device. Only the temporal attention parameters (attention_fc,
    attention_v, attention_drop, classifier head) receive gradient updates.

    Sampled hyperparameters:
      Architecture (attention-specific):
        attention_dim     : categorical [32, 64, 128]
        attention_dropout : float [0.0, 0.4] step 0.05
      Training (for attention + head params only):
        learning_rate     : float [1e-4, 1e-2] log scale
        weight_decay      : float [1e-5, 1e-3] log scale
        batch_size        : categorical [16, 32, 64]

    Fixed parameters (TCN backbone -- not sampled):
        num_layers, num_filters, kernel_size, dropout

    Parameters
    ----------
    trial       : optuna.Trial
    train_pairs : list of (filepath, label) tuples
    val_pairs   : list of (filepath, label) tuples
    tcn_hp      : dict -- fixed TCN backbone hyperparameters
    device      : torch.device
    logger      : logging.Logger

    Returns
    -------
    float -- best validation macro F1 achieved in this trial
    """
    # -- a. Sample attention architecture + training hyperparameters -----------
    attention_dim = trial.suggest_categorical(
        "attention_dim", ATTN_DIM_CHOICES)              # projection dimension in scoring function
    attention_dropout = trial.suggest_float(
        "attention_dropout",
        ATTN_DROP_MIN, ATTN_DROP_MAX,
        step=ATTN_DROP_STEP)                            # dropout on context vector
    learning_rate = trial.suggest_float(
        "learning_rate", LR_MIN, LR_MAX, log=True)     # log scale spans orders of magnitude
    weight_decay = trial.suggest_float(
        "weight_decay", WD_MIN, WD_MAX, log=True)      # L2 regularisation on attention params
    batch_size = trial.suggest_categorical(
        "batch_size", BATCH_CHOICES)                    # powers of 2 for GPU efficiency

    # -- b. Set seed for reproducible weight initialisation --------------------
    set_seed(SEED)

    # -- c. Instantiate model with fixed backbone + sampled attention params ---
    # TCNWithAttention.__init__(num_layers, num_filters, kernel_size, dropout,
    #   attention_dim=64, attention_dropout=0.0, return_embedding=False, fs=500)
    model = TCNWithAttention(
        num_layers=int(tcn_hp["num_layers"]),
        num_filters=int(tcn_hp["num_filters"]),
        kernel_size=int(tcn_hp["kernel_size"]),
        dropout=float(tcn_hp["dropout"]),
        attention_dim=attention_dim,                     # sampled attention architecture param
        attention_dropout=attention_dropout,             # sampled attention architecture param
        return_embedding=False,                         # classification mode (returns logits)
        fs=FS,
    )
    model = model.to(device)

    # -- d. Freeze TCN backbone parameters -------------------------------------
    # TCNWithAttention stores the backbone as self.tcn (nn.Sequential)
    # All parameters with names starting with "tcn." belong to the backbone
    for name, param in model.named_parameters():
        if name.startswith(BACKBONE_PREFIX):
            param.requires_grad = False                 # freeze backbone weights

    # Log parameter audit on first trial to confirm freezing worked
    if trial.number == 0:
        n_frozen = sum(
            p.numel() for p in model.parameters()
            if not p.requires_grad)
        n_trainable = sum(
            p.numel() for p in model.parameters()
            if p.requires_grad)
        logger.info("Trial 0 -- parameter audit:")
        logger.info("  Frozen    : %s", f"{n_frozen:,}")
        logger.info("  Trainable : %s", f"{n_trainable:,}")

    # -- e. Log trial header with all sampled hyperparameters --------------------
    n_params = count_parameters(model)
    logger.info(
        "Trial %d: attn_dim=%d attn_drop=%.2f lr=%.2e wd=%.2e bs=%d params=%d",
        trial.number, attention_dim, attention_dropout,
        learning_rate, weight_decay, batch_size, n_params)

    # -- f. Build data loaders -------------------------------------------------
    train_loader = make_loader(
        train_pairs, batch_size=batch_size,
        train=True, device=device)
    val_loader = make_loader(
        val_pairs, batch_size=batch_size,
        train=False, device=device)

    # -- f. Build optimiser on trainable parameters only -----------------------
    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad]                             # excludes frozen backbone
    optimiser = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max=MAX_EPOCHS,
        eta_min=learning_rate * 0.01)                   # decay to 1% of initial lr

    # -- g. Build loss -- imbalance handled by offline downsampling; pos_weight=1.0
    pos_weight = torch.tensor([1.0], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # -- h. Training loop with early stopping ----------------------------------
    best_val_f1 = 0.0
    epochs_no_imp = 0

    # Mixed precision (AMP): use FP16 forward/backward on CUDA to leverage
    # Tensor Cores (V100, A100, L40S, T4). GradScaler dynamically adjusts
    # the loss scale to prevent FP16 gradient underflow. On CPU, use_amp is
    # False and all operations remain FP32 — no behavioural change.
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    for epoch in range(1, MAX_EPOCHS + 1):

        t0_train = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimiser, criterion, device, scaler=scaler)
        train_sec = time.time() - t0_train

        # evaluate() returns (macro_f1, y_true, y_pred)
        t0_val = time.time()
        val_f1, _, _ = evaluate(model, val_loader, device, use_amp=use_amp)
        val_sec = time.time() - t0_val

        scheduler.step()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_imp = 0
        else:
            epochs_no_imp += 1

        # Log every epoch (20-epoch budget is short enough for full visibility)
        if True:
            logger.info(
                "  T%d ep %3d/%d | loss=%.4f | f1=%.4f | best=%.4f | pat=%d/%d"
                " | train %.0fs | val %.0fs",
                trial.number, epoch, MAX_EPOCHS, train_loss, val_f1,
                best_val_f1, epochs_no_imp, ES_PATIENCE, train_sec, val_sec)

        # Report to Optuna for pruner to evaluate
        trial.report(val_f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if epochs_no_imp >= ES_PATIENCE:
            break

    # -- i. One-line trial summary ---------------------------------------------
    logger.info(
        "Trial %3d | val_F1=%.4f | attn_dim=%d | attn_drop=%.2f | "
        "lr=%.2e | wd=%.2e | bs=%d",
        trial.number, best_val_f1, attention_dim, attention_dropout,
        learning_rate, weight_decay, batch_size)

    # -- j. Return best validation F1 ------------------------------------------
    return best_val_f1


# ---------------------------------------------------------------------------
# save_study_results
# ---------------------------------------------------------------------------
def save_study_results(study, tcn_hp, tcn_config, device, logger):
    """Save all tuning outputs to files.

    Does NOT train a final model. Does NOT save weights.

    Saves:
        outputs/best_attention_params.json
        outputs/attention_study_results.csv
        outputs/attention_tuning_summary.json

    Parameters
    ----------
    study      : optuna.Study -- completed Optuna study
    tcn_hp     : dict -- fixed TCN backbone hyperparameters
    tcn_config : dict -- full content of best_params.json
    device     : torch.device -- device used during tuning
    logger     : logging.Logger

    Returns
    -------
    dict -- the best_attention_params record
    """
    # -- a. Extract study results ----------------------------------------------
    best_trial = study.best_trial
    best_val_f1 = best_trial.value
    best_params = best_trial.params
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.PRUNED]

    logger.info("=" * 60)
    logger.info("TUNING COMPLETE")
    logger.info("  Completed       : %d", len(completed))
    logger.info("  Pruned          : %d", len(pruned))
    logger.info("  Best trial      : #%d", best_trial.number)
    logger.info("  Best val F1     : %.4f", best_val_f1)
    logger.info("  Best parameters :")
    logger.info("    attention_dim   : %d", best_params["attention_dim"])
    logger.info("    attention_drop  : %.2f", best_params["attention_dropout"])
    logger.info("    learning_rate   : %.2e", best_params["learning_rate"])
    logger.info("    weight_decay    : %.2e", best_params["weight_decay"])
    logger.info("    batch_size      : %d", best_params["batch_size"])
    logger.info("    Device          : %s", device)

    # -- b. Save best_attention_params.json ------------------------------------
    # Compute receptive field from config or from backbone params
    rf_samples = tcn_config.get(
        "receptive_field_samples",
        2 * (2 ** int(tcn_hp["num_layers"]) - 1) * (int(tcn_hp["kernel_size"]) - 1) + 1)
    rf_seconds = tcn_config.get(
        "receptive_field_seconds",
        rf_samples / FS)

    record = {
        "model":              "M2_TCN_TemporalAttention",
        "timestamp":          datetime.datetime.now().isoformat(),
        "note":               (
            "Hyperparameter tuning only. "
            "No final model trained. "
            "Use train_model2_tcn_attention.ipynb "
            "to train the final model."),
        "best_trial_number":  best_trial.number,
        "best_val_f1":        round(best_val_f1, 6),
        "n_trials_completed": len(completed),
        "n_trials_pruned":    len(pruned),
        "training_device":    str(device),
        "receptive_field": {
            "samples": rf_samples,
            "seconds": round(rf_seconds, 4),
        },
        "tcn_hyperparameters": tcn_hp,
        "hyperparameters":     best_params,
    }

    with open(BEST_ATTN_PATH, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    logger.info("Saved: %s", BEST_ATTN_PATH)

    # -- c. Save attention_study_results.csv -----------------------------------
    # One row per trial; columns: trial_number, val_f1, state, <params>, duration_seconds
    param_names = sorted(best_params.keys())            # deterministic column order
    fieldnames = ["trial_number", "val_f1", "state"] + param_names + ["duration_seconds"]

    with open(STUDY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in study.trials:
            # Use last reported value for pruned trials, 0.0 if nothing reported
            if t.state == optuna.trial.TrialState.COMPLETE:
                val_f1_row = t.value
            elif t.intermediate_values:
                val_f1_row = t.intermediate_values[max(t.intermediate_values.keys())]
            else:
                val_f1_row = 0.0

            duration = (
                (t.datetime_complete - t.datetime_start).total_seconds()
                if t.datetime_start and t.datetime_complete else 0.0)

            row = {
                "trial_number": t.number,
                "val_f1":       round(val_f1_row, 6),
                "state":        t.state.name,
                "duration_seconds": round(duration, 1),
            }
            for pn in param_names:
                row[pn] = t.params.get(pn, "")          # blank if param wasn't sampled (pruned early)
            writer.writerow(row)

    logger.info("Saved: %s", STUDY_CSV)

    # -- d. Save attention_tuning_summary.json ---------------------------------
    summary = {
        "timestamp":          datetime.datetime.now().isoformat(),
        "study_name":         "temporal_attention_tuning",
        "n_trials_requested": N_TRIALS,
        "n_trials_completed": len(completed),
        "n_trials_pruned":    len(pruned),
        "best_trial_number":  best_trial.number,
        "best_val_f1":        round(best_val_f1, 6),
        "training_device":    str(device),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else "cpu"),
        "fs_hz":               FS,
        "segment_len_samples": SEGMENT_LEN,
        "segment_len_seconds": SEGMENT_SEC,
        "search_ranges": {
            "attention_dim":     ATTN_DIM_CHOICES,
            "attention_dropout":
                "%s to %s step %s" % (ATTN_DROP_MIN, ATTN_DROP_MAX, ATTN_DROP_STEP),
            "learning_rate":
                "%s to %s (log scale)" % (LR_MIN, LR_MAX),
            "weight_decay":
                "%s to %s (log scale)" % (WD_MIN, WD_MAX),
            "batch_size":       BATCH_CHOICES,
        },
        "architecture_note": (
            "TCNWithAttention uses two-layer additive attention "
            "(tanh + linear scorer) with tunable attention_dim and "
            "attention_dropout, matching MultiScaleTCNWithAttention."),
        "fixed_tcn_params": tcn_hp,
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved: %s", SUMMARY_PATH)

    # -- e. Return the record for downstream use --------------------------------
    return record


# ---------------------------------------------------------------------------
# plot_tuning_figures
# ---------------------------------------------------------------------------
def plot_tuning_figures(study, best_val_f1, logger):
    """Generate and save two tuning visualisation figures.

    Both figures are saved before plt.close() -- required when running
    as a non-interactive script with Agg backend.

    Saves:
        outputs/figures/attention_f1_history.png
        outputs/figures/attention_importance.png

    Parameters
    ----------
    study       : optuna.Study -- completed study
    best_val_f1 : float -- best validation F1 from the study
    logger      : logging.Logger
    """
    # -- a. F1 history figure --------------------------------------------------
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE]
    trial_nums = [t.number for t in completed]
    f1_vals = [t.value for t in completed]

    # Compute running best for overlay
    running_best = []
    current_best = 0.0
    for v in f1_vals:
        current_best = max(current_best, v)
        running_best.append(current_best)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Subplot 0: F1 per trial with running best
    axes[0].scatter(
        trial_nums, f1_vals,
        s=18, alpha=0.6, color="#5A7DC8",
        label="Trial F1")
    axes[0].plot(
        trial_nums, running_best,
        color="#C85A5A", linewidth=1.8,
        label="Running best")
    axes[0].axhline(
        best_val_f1, linestyle="--",
        color="#C85A5A", linewidth=1.0, alpha=0.5)
    if running_best:
        best_idx = running_best.index(best_val_f1)
        axes[0].annotate(
            "Best: %.4f" % best_val_f1,
            xy=(trial_nums[best_idx], best_val_f1),
            xytext=(5, -15),
            textcoords="offset points",
            fontsize=9, color="#C85A5A")
    axes[0].set_xlabel("Trial number")
    axes[0].set_ylabel("Validation macro F1")
    axes[0].set_title("F1 across Optuna trials")
    axes[0].legend(fontsize=9)

    # Subplot 1: F1 distribution histogram
    axes[1].hist(
        f1_vals, bins=12,
        color="#5A7DC8", edgecolor="white", alpha=0.85)
    axes[1].axvline(
        best_val_f1, color="#C85A5A", linewidth=1.8,
        linestyle="--",
        label="Best: %.4f" % best_val_f1)
    axes[1].set_xlabel("Validation macro F1")
    axes[1].set_ylabel("Count")
    axes[1].set_title("F1 distribution")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_F1, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", FIG_F1)

    # -- b. Hyperparameter importance figure -----------------------------------
    try:
        importance = optuna.importance.get_param_importances(study)
        names = list(importance.keys())
        values = list(importance.values())
        max_v = max(values) if values else 1.0

        colours = [
            "#C85A5A" if v == max_v else "#5A7DC8"
            for v in values]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(
            names[::-1], values[::-1],
            color=colours[::-1], edgecolor="white")
        ax.set_xlabel("Relative importance (fANOVA)")
        ax.set_title("Temporal attention hyperparameter importance")
        plt.tight_layout()
        plt.savefig(FIG_IMP, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved: %s", FIG_IMP)
        logger.info("Importance scores:")
        for n, v in importance.items():
            logger.info("  %-25s: %.4f", n, v)
    except Exception as exc:
        logger.warning(
            "Importance plot skipped: %s. "
            "Requires >= 2 completed trials.", exc)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    """Entry point. Runs temporal attention tuning and saves all outputs.

    Does not train a final model. Does not save model weights.
    """
    # -- Step 1: setup logging and device --------------------------------------
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("tune_temporal_attention.py")
    logger.info("Timestamp       : %s", datetime.datetime.now().isoformat())
    logger.info("Ablation role   : M2 -- TCN + Temporal Attention")
    logger.info("Params source   : best_params.json (backbone, frozen)")
    logger.info("Purpose         : Tune attention HPs (no final model)")
    logger.info("N_TRIALS        : %d", N_TRIALS)
    logger.info("MAX_EPOCHS      : %d", MAX_EPOCHS)
    logger.info("ES_PATIENCE     : %d", ES_PATIENCE)
    logger.info("Test set        : NOT loaded")
    logger.info("=" * 60)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        logger.info("GPU  : %s", torch.cuda.get_device_name(0))
        logger.info("VRAM : %.2f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)
        logger.info("CUDA : %s", torch.version.cuda)
    else:
        logger.info("Device: CPU")
    logger.info("PyTorch: %s", torch.__version__)

    # -- Step 2: load inputs ---------------------------------------------------
    tcn_config, tcn_hp = load_best_tcn_params(logger)
    train_pairs, val_pairs = load_splits(logger)

    # -- Corpus preparation ----------------------------------------------------
    # Downsampling and extreme-segment filtering are handled offline by
    # create_balanced_splits.py. The manifest is already clean.
    # Subject exclusion (m254), 1:4 downsampling, and extreme-segment filtering
    # are ALL handled offline by create_balanced_splits.py. The manifest is
    # already clean and balanced -- no further corpus preparation is needed here.
    # train_pairs = filter_unpaired_subjects(train_pairs, logger=logger)
    logger.info("Training corpus: %d segments (from balanced manifest)", len(train_pairs))

    # Stratified 10% validation subset for tuning speed.
    val_pairs = downsample_val_stratified(val_pairs, fraction=0.10, seed=42)
    logger.info("Val subset for tuning: %d segments (10%% stratified)", len(val_pairs))
    # -- End corpus preparation ------------------------------------------------

    # -- Step 3: configure and run Optuna study --------------------------------
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # SQLite storage enables resume after crash. load_if_exists=True loads
    # the existing study so completed trials are not re-run.
    study_db = OUTPUT_DIR / "tune_temporal_attention.db"
    study = optuna.create_study(
        study_name="temporal_attention_tuning",
        direction="maximize",
        sampler=TPESampler(seed=SEED, n_startup_trials=N_STARTUP),
        pruner=MedianPruner(n_startup_trials=N_STARTUP, n_warmup_steps=3),
        storage="sqlite:///" + str(study_db.resolve()),
        load_if_exists=True,
    )
    logger.info("Optuna storage: %s | completed trials so far: %d",
                study_db, len(study.trials))

    logger.info(
        "Starting Optuna study | %d trials | %d random startup",
        N_TRIALS, N_STARTUP)

    study.optimize(
        lambda trial: optuna_objective(
            trial, train_pairs, val_pairs,
            tcn_hp, device, logger),
        n_trials=N_TRIALS,
        catch=(OSError,),          # transient I/O errors fail the trial, not the study
    )

    # -- Step 4: save all results ----------------------------------------------
    best_record = save_study_results(
        study, tcn_hp, tcn_config, device, logger)

    # -- Step 5: plot figures --------------------------------------------------
    plot_tuning_figures(
        study, best_record["best_val_f1"], logger)

    # -- Step 6: GPU cleanup and final output inventory ------------------------
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared.")

    logger.info("=" * 60)
    logger.info("ALL OUTPUTS SAVED")
    logger.info("No model weights were saved (by design)")
    output_files = [
        BEST_ATTN_PATH, STUDY_CSV,
        SUMMARY_PATH, LOG_FILE,
        FIG_F1, FIG_IMP,
    ]
    for p in output_files:
        status = "OK     " if Path(p).exists() else "MISSING"
        logger.info("  [%s] %s", status, p)
    logger.info("=" * 60)
    logger.info(
        "NEXT: train_model2_tcn_attention.ipynb will load "
        "best_attention_params.json and train the final model.")


if __name__ == "__main__":
    main()


# ======================================================================
# HOW TO REPORT TEMPORAL ATTENTION TUNING IN PAPER
# ======================================================================
#
# Architecture (Methods)
# ----------------------
# TCNWithAttention (tcn_utils.py) appends a TemporalAttention module
# after the TCN stack. TemporalAttention uses a single nn.Linear(D, 1)
# to score each time step, followed by softmax over T to produce
# attention weights alpha_t. The attended context vector is:
#   c = sum_t alpha_t * h_t
# where h_t is the TCN feature vector at time step t (dimension D =
# num_filters). LayerNorm is applied to c before the classification
# head. Complexity is O(T * D), linear in sequence length.
#
# Parameters to report in Methods table
# --------------------------------------
# All parameters from best_attention_params.json under "hyperparameters":
#   attention_dim     : projection dimension of scoring function
#   attention_dropout : dropout on context vector
#   learning_rate     : AdamW lr (attention + head params only)
#   weight_decay      : L2 coefficient
#   batch_size        : segments per gradient step
#
# Also report:
#   N_TRIALS    = 50  (Optuna trials)
#   N_STARTUP   = 10  (random startup trials)
#   ES_PATIENCE = 5   (early stopping patience)
#   Tuning metric: validation macro F1-score
#   Optimiser: AdamW with cosine annealing
#   Backbone status: frozen during attention tuning
#
# Note on search space:
#   TCNWithAttention uses two-layer additive attention (tanh + linear)
#   with attention_dim and attention_dropout as tunable architectural
#   parameters. This matches MultiScaleTCNWithAttention, ensuring
#   consistent ablation comparison between M1/M2 and M3/M4.
#
# Freezing rationale (Methods)
# ----------------------------
# "The TCN backbone weights were frozen during temporal attention
# tuning so that only the attention mechanism parameters received
# gradient updates. This ensures the comparison between M1
# (TCN alone) and M2 (TCN with temporal attention) is controlled
# -- both models share identical backbone weights, isolating the
# contribution of the temporal attention module."
#
# Why temporal attention (Discussion)
# ------------------------------------
# TemporalAttention replaces multi-head self-attention (Q=K=V).
# Complexity: O(T * D) vs O(T^2 * D * H).
# At T=2500, D=64, H=4: MHA processes 6.25M pairwise interactions
# per sample; TemporalAttention processes 160K scalar scores.
# Temporal attention produces directly interpretable per-timestep
# saliency weights (sum-to-one, non-negative).
#
# Attention weights as interpretability (Results)
# ------------------------------------------------
# TCNWithAttention.forward() returns logits by default; however
# TemporalAttention.forward() returns (context, weights). To
# extract attention weights at inference, call model.attn(out)
# directly on the transposed TCN output, or modify forward()
# to return weights as a second output.
#
# WARNING -- confirm before writing the paper
# --------------------------------------------
# Read the actual TCNWithAttention.forward() in tcn_utils.py and
# describe what is implemented there, not what this comment assumes.
# ======================================================================
