"""
tune_multiscale_tcn.py
======================
Tunes all hyperparameters of MultiScaleTCN from
scratch using Optuna TPE.

Tuning protocol
----------------
N_TRIALS    = 50   total Optuna trials
MAX_EPOCHS  = 20   max epochs per trial (cosine annealing half-cycle)
ES_PATIENCE = 5    early stopping patience (epochs without val F1 improvement)

Early stopping at patience=5 within a 20-epoch budget means most trials
terminate between epochs 8-15. The reduced budget accelerates the search
while retaining enough epochs for the cosine schedule to differentiate
good from bad hyperparameter configurations (Li et al., 2017).

Validation subset: a stratified 1% subset of the validation partition
is used during tuning to reduce per-epoch evaluation cost. The subset
preserves the original class ratio and is fixed across all trials
(Falkner et al., 2018). See STUDY_REPORT.txt "Validation subset during
tuning" for the timing justification.

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

No parameters are transferred from the single-branch
TCN tuning (best_params.json is not loaded here).
Independent tuning ensures MultiScaleTCN is assessed
at its true optimum.

Architecture: MultiScaleTCN (from tcn_utils.py)

Branch depth and dilation schedules -- fixed, not tuned
-------------------------------------------------------
Each branch has exactly 3 CausalConvBlocks with fixed dilation schedules:
  Branch 1: [1,  2,   4]    -- fine scale         (spike morphology)
  Branch 2: [8,  16,  32]   -- intermediate scale (rhythmic structure)
  Branch 3: [32, 64, 128]   -- coarse scale       (seizure evolution)

Per-branch receptive field (kernel_size k=7, two dilated convs per block,
RF = 1 + 2(k-1) * Sum(d)):
  Branch 1:   85 samples =   170 ms  (at 500 Hz)
  Branch 2:  673 samples =  1.35 s
  Branch 3: 2689 samples =  5.38 s (spans the full 5-second segment)

Unlike the single-branch TCN (M1), where num_layers is tuned to control
depth and receptive field, MultiScaleTCN replaces depth with breadth:
three shallow branches operating at complementary temporal resolutions
provide receptive field diversity without deep stacking. This follows
the multi-scale temporal convolution design in Lea et al. (2017) and
Farha & Gall (2019), where branch structure is an architectural prior
fixed before hyperparameter search.

Branch depth is not tuned for three reasons:
  1. The multi-scale design derives its expressive power from the
     dilation spread across branches, not from per-branch depth.
     The three branches jointly cover sub-second spike morphology
     through full-segment seizure evolution, matching the relevant
     EEG temporal scales for rodent seizure detection (Luttjohann
     et al., 2009).
  2. Tuning per-branch depth would create a combinatorial explosion
     (e.g. 3-6 blocks x 3 branches = 64 combinations) that 50 trials
     cannot explore meaningfully. Fixing depth focuses the search
     budget on parameters with higher sensitivity: num_filters,
     dropout, and learning_rate (Bergstra & Bengio, 2012).
  3. The dilation schedules are selected to give each branch a
     receptive field that targets a distinct physiological time
     scale of rodent EEG seizures: ~50-200 ms spike complexes,
     ~0.4-1.4 s rhythmic bursts, and ~1.8-5.4 s ictal envelopes.
     The coarse branch's 5.38 s receptive field ensures full
     segment-level context is available at fusion.

References (architecture):
  Lea, C., Flynn, M. D., Vidal, R., Reiter, A., & Hager, G. D. (2017).
      Temporal Convolutional Networks for Action Segmentation and
      Detection. CVPR 2017.
  Farha, Y. A. & Gall, J. (2019). MS-TCN: Multi-Stage Temporal
      Convolutional Network for Action Segmentation. CVPR 2019.
  Bergstra, J. & Bengio, Y. (2012). Random Search for Hyper-Parameter
      Optimization. JMLR, 13, 281-305.
  Luttjohann, A., Fabene, P. F., & van Luijtelaar, G. (2009). A revised
      Racine's scale for PTZ-induced seizures in rats. Physiology &
      Behavior, 98(5), 579-586.

Hyperparameters tuned
---------------------
num_filters   : branch channel width
kernel_size   : convolutional kernel (odd: 3, 5, 7)
dropout       : spatial dropout rate
fusion        : concat or average
learning_rate : AdamW step size
weight_decay  : L2 regularisation
batch_size    : segments per gradient step

Outputs (no model weights saved)
---------------------------------
{OUTPUT_DIR}/best_multiscale_params.json
{OUTPUT_DIR}/multiscale_study_results.csv
{OUTPUT_DIR}/multiscale_tuning_summary.json
{OUTPUT_DIR}/logs/tune_multiscale_tcn.log
{OUTPUT_DIR}/figures/multiscale_f1_history.png
{OUTPUT_DIR}/figures/multiscale_importance.png
{OUTPUT_DIR}/tune_multiscale_tcn.db  (Optuna SQLite for resume)

Usage
-----
python tune_multiscale_tcn.py

Pipeline position
-----------------
1. generate_data_splits.py --no-test
2. create_balanced_splits.py
3. tcn_HPT_binary.py
4. tune_multiscale_tcn.py      <- this script
5. tune_multiscale_attention.py
6. Training scripts (TCN.py, TCNTemporalAttention.py,
                     MultiScaleTCN.py, MultiScaleTCNAttention.py)
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import json
import logging
import sys
import csv
import time
import gc                               # explicit garbage collection between trials
import os                               # environment variable access (dry-run mode)
import random                           # pre-flight file sampling
import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")                                  # non-interactive backend
import matplotlib.pyplot as plt

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from tcn_utils import (
    set_seed,
    make_loader,
    # filter_unpaired_subjects,  # handled offline by create_balanced_splits.py
    downsample_val_stratified,
    MultiScaleTCN,
    count_parameters,
    train_one_epoch,
    evaluate,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED              = 42                                 # global reproducibility seed
MAX_EPOCHS        = 20                                 # max epochs per trial (ES fires before 20)
ES_PATIENCE       = 5                                  # early stopping patience (epochs)
GRAD_CLIP         = 1.0                                # maximum gradient norm for clipping
N_TRIALS          = 50                                 # total Optuna trials
N_STARTUP         = 10                                 # random startup before TPE
FS                = 500                                # EEG sampling rate (Hz)
SEGMENT_LEN       = 2500                               # samples per segment (5 s at 500 Hz)
SEGMENT_SEC       = 5.0                                # segment duration in seconds

# -- Dry-run mode: set TCN_HPT_DRY_RUN=1 to cap N_TRIALS at 2 for cluster
# job shakedowns (verifies the script starts, loads data, reaches the
# training loop, and writes outputs without consuming a full tuning budget).
# Example: TCN_HPT_DRY_RUN=1 python tune_multiscale_tcn.py
DRY_RUN = os.environ.get("TCN_HPT_DRY_RUN", "").lower() in ("1", "true", "yes")
if DRY_RUN:
    N_TRIALS = 2
    N_STARTUP = 1

OUTPUT_DIR        = Path("/home/people/22206468/scratch/OUTPUT/MODEL3_OUTPUT/MultiScaleTCNtuning_outputs")
LOG_DIR           = OUTPUT_DIR / "logs"
FIGURE_DIR        = OUTPUT_DIR / "figures"
# Previous (uniform downsampling): data_splits.json
# SPLITS_PATH       = Path("/scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits.json")
# Current (proximity-aware downsampling): data_splits_nonictal_sampled.json
SPLITS_PATH       = Path("/scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits_nonictal_sampled.json")
BEST_MS_PATH      = OUTPUT_DIR / "best_multiscale_params.json"
STUDY_CSV         = OUTPUT_DIR / "multiscale_study_results.csv"
SUMMARY_PATH      = OUTPUT_DIR / "multiscale_tuning_summary.json"
LOG_FILE          = LOG_DIR / "tune_multiscale_tcn.log"
FIG_F1            = FIGURE_DIR / "multiscale_f1_history.png"
FIG_IMP           = FIGURE_DIR / "multiscale_importance.png"

# Hyperparameter search ranges
NUM_FILTERS_CHOICES = [32, 64, 128]                    # branch channel width candidates
KERNEL_SIZE_CHOICES = [3, 5, 7]                        # odd kernels for symmetric causal padding
DROPOUT_MIN         = 0.1                              # spatial dropout lower bound
DROPOUT_MAX         = 0.4                              # spatial dropout upper bound
DROPOUT_STEP        = 0.05                             # dropout step size
FUSION_CHOICES      = ["concat", "average"]            # branch fusion strategies
LR_MIN              = 1e-4                             # AdamW lr lower bound
LR_MAX              = 5e-4                             # AdamW lr upper bound
WD_MIN              = 5e-5                             # weight decay lower bound
WD_MAX              = 1e-3                             # weight decay upper bound
BATCH_CHOICES       = [32, 64]                     # batch size candidates

# Dilation schedules -- fixed by architectural design, not tuned.
# Per-branch RF (k=7, 3 blocks, RF = 1 + 2(k-1)*sum(d)):
#   Branch 1:   85 samples =   170 ms  at 500 Hz
#   Branch 2:  673 samples =  1.35 s
#   Branch 3: 2689 samples =  5.38 s   (covers full 5-s segment)
BRANCH1_DILATIONS   = [1, 2, 4]                        # fine:         spike morphology
BRANCH2_DILATIONS   = [8, 16, 32]                      # intermediate: rhythmic bursts
BRANCH3_DILATIONS   = [32, 64, 128]                    # coarse:       seizure evolution


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------
def setup_logging():
    """Create output directories and configure the module logger.

    FileHandler: LOG_FILE mode='a' level DEBUG (append for resume).
    StreamHandler: sys.stdout level INFO.

    Returns
    -------
    logger : logging.Logger
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("multiscale_tcn_tuning")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()                # prevent duplicate handlers on re-import or re-run

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
# load_splits
# ---------------------------------------------------------------------------
def load_splits(logger):
    """Load train and val file-label pairs from data_splits_nonictal_sampled.json.

    Parameters
    ----------
    logger : logging.Logger

    Returns
    -------
    tuple of (train_pairs, val_pairs)
        Each is a list of (filepath: str, label: int) tuples.

    Raises
    ------
    FileNotFoundError if data_splits_nonictal_sampled.json is absent.
    RuntimeError if either partition is empty.
    """
    if not SPLITS_PATH.exists():
        logger.error(
            "data_splits_nonictal_sampled.json not found at %s. "
            "Run create_balanced_splits.py first (which itself requires "
            "data_splits.json from generate_data_splits.py).", SPLITS_PATH)
        raise FileNotFoundError(str(SPLITS_PATH))

    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)

    # Convert list-of-dicts to list-of-tuples: (filepath, label)
    train_pairs = [(rec["filepath"], rec["label"]) for rec in splits["train"]]
    val_pairs   = [(rec["filepath"], rec["label"]) for rec in splits["val"]]

    if not train_pairs:
        logger.error("Train partition is empty in %s.", SPLITS_PATH)
        raise RuntimeError("Empty train partition")
    if not val_pairs:
        logger.error("Val partition is empty in %s.", SPLITS_PATH)
        raise RuntimeError("Empty val partition")

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
def optuna_objective(trial, train_pairs, val_pairs, device, logger):
    """Optuna objective for MultiScaleTCN tuning.

    Samples all seven hyperparameters from scratch.
    Dilation schedules are fixed -- not sampled.
    Returns best validation macro F1.

    Parameters
    ----------
    trial       : optuna.Trial
    train_pairs : list of (filepath, label) tuples
    val_pairs   : list of (filepath, label) tuples
    device      : torch.device
    logger      : logging.Logger

    Returns
    -------
    float -- best validation macro F1 achieved in this trial
    """
    # -- a. Sample hyperparameters ---------------------------------------------
    num_filters   = trial.suggest_categorical("num_filters", NUM_FILTERS_CHOICES)
    kernel_size   = trial.suggest_categorical("kernel_size", KERNEL_SIZE_CHOICES)
    dropout       = trial.suggest_float("dropout", DROPOUT_MIN, DROPOUT_MAX, step=DROPOUT_STEP)
    fusion        = trial.suggest_categorical("fusion", FUSION_CHOICES)
    learning_rate = trial.suggest_float("learning_rate", LR_MIN, LR_MAX, log=True)
    weight_decay  = trial.suggest_float("weight_decay", WD_MIN, WD_MAX, log=True)
    batch_size    = trial.suggest_categorical("batch_size", BATCH_CHOICES)

    # -- Resource cleanup: release model and loaders on exit, even on exception
    # (TrialPruned mid-epoch, OSError from catch=..., or normal completion).
    # Prevents GPU VRAM and DataLoader worker accumulation across 50 trials.
    model = None
    train_loader = None
    val_loader = None
    try:
        # -- b. Set seed for reproducible weight initialisation ----------------
        set_seed(SEED)

        # -- c. Instantiate model ----------------------------------------------
        model = MultiScaleTCN(
            num_filters=num_filters,
            kernel_size=kernel_size,
            dropout=dropout,
            branch1_dilations=BRANCH1_DILATIONS,
            branch2_dilations=BRANCH2_DILATIONS,
            branch3_dilations=BRANCH3_DILATIONS,
            fusion=fusion,
        ).to(device)

        # -- Log trial header with all sampled hyperparameters -----------------
        n_params = count_parameters(model)
        logger.info(
            "Trial %d: f=%d k=%d drop=%.2f fusion=%s lr=%.2e wd=%.2e bs=%d params=%d",
            trial.number, num_filters, kernel_size, dropout, fusion,
            learning_rate, weight_decay, batch_size, n_params)

        # -- d. Build data loaders ---------------------------------------------
        train_loader = make_loader(train_pairs, batch_size=batch_size, train=True, device=device)
        val_loader   = make_loader(val_pairs, batch_size=batch_size, train=False, device=device)

        # -- e. Build optimiser and scheduler ----------------------------------
        optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=MAX_EPOCHS, eta_min=learning_rate * 0.01)

        # -- f. Build loss -- imbalance handled by offline downsampling; pos_weight=1.0
        pos_weight = torch.tensor([1.0], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # -- g. Training loop with early stopping ------------------------------
        best_val_f1   = 0.0
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
                model, train_loader, optimiser, criterion, device,
                max_grad_norm=GRAD_CLIP, scaler=scaler)
            train_sec = time.time() - t0_train

            t0_val = time.time()
            val_f1, _, _ = evaluate(model, val_loader, device, use_amp=use_amp)
            val_sec = time.time() - t0_val

            scheduler.step()

            if val_f1 > best_val_f1:
                best_val_f1   = val_f1
                epochs_no_imp = 0
            else:
                epochs_no_imp += 1

            # Log every epoch (20-epoch budget is short enough for full visibility)
            logger.info(
                "  T%d ep %3d/%d | loss=%.4f | f1=%.4f | best=%.4f | pat=%d/%d"
                " | train %.0fs | val %.0fs",
                trial.number, epoch, MAX_EPOCHS, train_loss, val_f1,
                best_val_f1, epochs_no_imp, ES_PATIENCE, train_sec, val_sec)

            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if epochs_no_imp >= ES_PATIENCE:
                break

        # -- h. One-line trial summary -----------------------------------------
        logger.info(
            "Trial %3d | F1=%.4f | filters=%d | k=%d | drop=%.2f | "
            "fusion=%s | lr=%.2e | bs=%d",
            trial.number, best_val_f1, num_filters, kernel_size,
            dropout, fusion, learning_rate, batch_size)

        return best_val_f1
    finally:
        # Release GPU memory and DataLoader workers between trials. Without
        # this, persistent_workers=True leaves worker processes alive across
        # trials and cached GPU allocations accumulate, eventually causing OOM.
        del model, train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# save_results
# ---------------------------------------------------------------------------
def save_results(study, device, logger):
    """Save all tuning outputs. No model weights saved.

    Saves:
        {OUTPUT_DIR}/best_multiscale_params.json   (only if any trial completed)
        {OUTPUT_DIR}/multiscale_study_results.csv
        {OUTPUT_DIR}/multiscale_tuning_summary.json

    Parameters
    ----------
    study  : optuna.Study -- completed Optuna study
    device : torch.device
    logger : logging.Logger

    Returns
    -------
    dict -- the best_multiscale_params record, or a minimal placeholder
            with {"best_val_f1": None} if no trial completed.
    """
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    # Guard: study.best_trial raises ValueError if no trial completed.
    # This can happen if every trial raised TrialPruned or hit the OSError catch.
    if len(completed) == 0:
        logger.error("No trials completed successfully. Check earlier log lines "
                     "for per-trial errors. best_multiscale_params.json will NOT "
                     "be written; CSV and summary will record the empty study.")
        best_trial  = None
        best_val_f1 = None
        best_params = None
    else:
        best_trial  = study.best_trial
        best_val_f1 = best_trial.value
        best_params = best_trial.params

    logger.info("=" * 60)
    logger.info("TUNING COMPLETE")
    logger.info("  Completed       : %d", len(completed))
    logger.info("  Pruned          : %d", len(pruned))
    if best_trial is not None:
        logger.info("  Best trial      : #%d", best_trial.number)
        logger.info("  Best val F1     : %.4f", best_val_f1)
        logger.info("  Best parameters :")
        logger.info("    num_filters     : %d", best_params["num_filters"])
        logger.info("    kernel_size     : %d", best_params["kernel_size"])
        logger.info("    dropout         : %.2f", best_params["dropout"])
        logger.info("    fusion          : %s", best_params["fusion"])
        logger.info("    learning_rate   : %.2e", best_params["learning_rate"])
        logger.info("    weight_decay    : %.2e", best_params["weight_decay"])
        logger.info("    batch_size      : %d", best_params["batch_size"])
        logger.info("    Device          : %s", device)

    # -- a. Save best_multiscale_params.json (only if we have a best) ----------
    if best_trial is not None:
        record = {
            "model":              "MultiScaleTCN",
            "timestamp":          datetime.datetime.now().isoformat(),
            "note":               ("All params tuned from scratch. No parameter transfer. "
                                   "No model weights saved here. "
                                   "Use train notebook to train the final model."),
            "best_trial_number":  best_trial.number,
            "best_val_f1":        round(best_val_f1, 6),
            "n_trials_completed": len(completed),
            "n_trials_pruned":    len(pruned),
            "training_device":    str(device),
            "branch_dilations": {
                "branch1": BRANCH1_DILATIONS,
                "branch2": BRANCH2_DILATIONS,
                "branch3": BRANCH3_DILATIONS,
            },
            "hyperparameters":    best_params,
        }
        with open(BEST_MS_PATH, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        logger.info("Saved: %s", BEST_MS_PATH)
    else:
        logger.warning("Skipping %s: no completed trials.", BEST_MS_PATH.name)
        record = {"best_val_f1": None}

    # -- b. Save multiscale_study_results.csv ----------------------------------
    # Derive param names from any trial that has them (best trial if available,
    # else the first trial in the study). If no trials exist, fall back to
    # the known search-space keys so the CSV still has a header.
    if best_params is not None:
        param_names = sorted(best_params.keys())
    elif study.trials and study.trials[0].params:
        param_names = sorted(study.trials[0].params.keys())
    else:
        param_names = ["num_filters", "kernel_size", "dropout", "fusion",
                       "learning_rate", "weight_decay", "batch_size"]
    fieldnames = ["trial_number", "val_f1", "state"] + param_names + ["duration_seconds"]

    with open(STUDY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in study.trials:
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
                "trial_number":    t.number,
                "val_f1":          round(val_f1_row, 6),
                "state":           t.state.name,
                "duration_seconds": round(duration, 1),
            }
            for pn in param_names:
                row[pn] = t.params.get(pn, "")
            writer.writerow(row)

    logger.info("Saved: %s", STUDY_CSV)

    # -- c. Save multiscale_tuning_summary.json --------------------------------
    summary = {
        "timestamp":           datetime.datetime.now().isoformat(),
        "study_name":          "multiscale_tcn_tuning",
        "n_trials_requested":  N_TRIALS,
        "n_trials_completed":  len(completed),
        "n_trials_pruned":     len(pruned),
        # best_trial_number / best_val_f1 are None if no trial completed.
        "best_trial_number":   (best_trial.number if best_trial is not None else None),
        "best_val_f1":         (round(best_val_f1, 6) if best_val_f1 is not None else None),
        "training_device":     str(device),
        "gpu_name":            (torch.cuda.get_device_name(0)
                                if torch.cuda.is_available() else "cpu"),
        "fs_hz":               FS,
        "segment_len_samples": SEGMENT_LEN,
        "segment_len_seconds": SEGMENT_SEC,
        "search_ranges": {
            "num_filters":   NUM_FILTERS_CHOICES,
            "kernel_size":   KERNEL_SIZE_CHOICES,
            "dropout":       "%s to %s step %s" % (DROPOUT_MIN, DROPOUT_MAX, DROPOUT_STEP),
            "fusion":        FUSION_CHOICES,
            "learning_rate": "%s to %s (log scale)" % (LR_MIN, LR_MAX),
            "weight_decay":  "%s to %s (log scale)" % (WD_MIN, WD_MAX),
            "batch_size":    BATCH_CHOICES,
        },
        "fixed_dilation_schedules": {
            "branch1": BRANCH1_DILATIONS,
            "branch2": BRANCH2_DILATIONS,
            "branch3": BRANCH3_DILATIONS,
        },
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved: %s", SUMMARY_PATH)

    return record


# ---------------------------------------------------------------------------
# plot_figures
# ---------------------------------------------------------------------------
def plot_figures(study, best_val_f1, logger):
    """Generate and save two tuning visualisation figures.

    Saves:
        {FIGURE_DIR}/multiscale_f1_history.png
        {FIGURE_DIR}/multiscale_importance.png

    Parameters
    ----------
    study       : optuna.Study
    best_val_f1 : float or None -- None if no trials completed
    logger      : logging.Logger
    """
    # -- a. F1 history figure --------------------------------------------------
    completed  = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    trial_nums = [t.number for t in completed]
    f1_vals    = [t.value for t in completed]

    # Skip plotting entirely if no trials completed -- axhline/hist/index would
    # crash with None or empty lists.
    if not f1_vals or best_val_f1 is None:
        logger.warning("No completed trials -- skipping tuning plots.")
        return

    running_best = []
    current_best = 0.0
    for v in f1_vals:
        current_best = max(current_best, v)
        running_best.append(current_best)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Subplot 0: F1 per trial with running best
    axes[0].scatter(trial_nums, f1_vals, s=18, alpha=0.6, color="#5A7DC8", label="Trial F1")
    axes[0].plot(trial_nums, running_best, color="#C85A5A", linewidth=1.8, label="Running best")
    axes[0].axhline(best_val_f1, linestyle="--", color="#C85A5A", linewidth=1.0, alpha=0.5)
    if running_best:
        # Use argmax on the raw running_best list. best_val_f1 is rounded
        # to 6 dp upstream, so running_best.index(best_val_f1) would raise
        # ValueError when the rounded and raw floats differ.
        best_idx = max(range(len(running_best)), key=lambda i: running_best[i])
        axes[0].annotate(
            "Best: %.4f" % best_val_f1,
            xy=(trial_nums[best_idx], running_best[best_idx]),
            xytext=(5, -15), textcoords="offset points",
            fontsize=9, color="#C85A5A")
    axes[0].set_xlabel("Trial number")
    axes[0].set_ylabel("Validation macro F1")
    axes[0].set_title("MultiScaleTCN F1 across trials")
    axes[0].legend(fontsize=9)

    # Subplot 1: F1 distribution histogram
    axes[1].hist(f1_vals, bins=12, color="#5A7DC8", edgecolor="white", alpha=0.85)
    axes[1].axvline(best_val_f1, color="#C85A5A", linewidth=1.8, linestyle="--",
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
        names  = list(importance.keys())
        values = list(importance.values())
        max_v  = max(values) if values else 1.0
        colours = ["#C85A5A" if v == max_v else "#5A7DC8" for v in values]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(names[::-1], values[::-1], color=colours[::-1], edgecolor="white")
        ax.set_xlabel("Relative importance (fANOVA)")
        ax.set_title("MultiScaleTCN hyperparameter importance")
        plt.tight_layout()
        plt.savefig(FIG_IMP, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved: %s", FIG_IMP)
        logger.info("Importance scores:")
        for n, v in importance.items():
            logger.info("  %-25s: %.4f", n, v)
    except Exception as exc:
        logger.warning(
            "Importance plot skipped: %s. Requires >= 2 completed trials.", exc)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    """Entry point. Runs MultiScaleTCN tuning and saves all outputs.

    Does not train a final model. Does not save model weights.
    """
    # -- Step 1: setup logging, seed, device -----------------------------------
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("tune_multiscale_tcn.py")
    logger.info("Timestamp       : %s", datetime.datetime.now().isoformat())
    logger.info("Ablation role   : M3 -- Multi-Scale TCN")
    logger.info("Params source   : Optuna TPE search (this script)")
    logger.info("Purpose         : Tune all MultiScaleTCN HPs (no final model)")
    logger.info("N_TRIALS        : %d%s", N_TRIALS, "  [DRY-RUN]" if DRY_RUN else "")
    logger.info("MAX_EPOCHS      : %d", MAX_EPOCHS)
    logger.info("ES_PATIENCE     : %d", ES_PATIENCE)
    logger.info("Test set        : NOT loaded")
    if DRY_RUN:
        logger.warning("DRY-RUN MODE: N_TRIALS capped at %d for shakedown. Unset "
                       "TCN_HPT_DRY_RUN environment variable for a full run.", N_TRIALS)
    logger.info("=" * 60)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        logger.info("GPU  : %s", torch.cuda.get_device_name(0))
        logger.info("VRAM : %.2f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)
        logger.info("CUDA : %s", torch.version.cuda)
        # Pre-flight GPU memory check: warn if another job is already on this GPU
        # or if VRAM is fragmented. Does not abort -- the user may still want to
        # run on a partially-occupied GPU at reduced batch size.
        try:
            _free_bytes, _total_bytes = torch.cuda.mem_get_info(0)
            _free_gb = _free_bytes / 1e9
            _total_gb = _total_bytes / 1e9
            logger.info("GPU memory free: %.2f / %.2f GB", _free_gb, _total_gb)
            if _free_gb < 8.0:
                logger.warning("GPU has only %.2f GB free (< 8 GB threshold). "
                               "Another process may be sharing this GPU, or VRAM "
                               "is fragmented. Trials may fail with CUDA OOM.",
                               _free_gb)
        except Exception as _e:
            logger.warning("Could not query GPU memory: %s", _e)
    else:
        logger.info("Device: CPU")
    logger.info("PyTorch: %s", torch.__version__)

    # -- Step 2: load data splits ----------------------------------------------
    train_pairs, val_pairs = load_splits(logger)

    # -- Corpus preparation ----------------------------------------------------
    # Subject exclusion (m254), 1:4 downsampling, and extreme-segment filtering
    # are ALL handled offline by create_balanced_splits.py. The manifest is
    # already clean and balanced -- no further corpus preparation is needed here.
    # train_pairs = filter_unpaired_subjects(train_pairs, logger=logger)
    logger.info("Training corpus: %d segments (from balanced manifest)", len(train_pairs))

    # Stratified 10% validation subset for tuning speed.
    val_pairs = downsample_val_stratified(val_pairs, fraction=0.01, seed=SEED)
    logger.info("Val subset for tuning: %d segments (1%% stratified)", len(val_pairs))
    # -- End corpus preparation ------------------------------------------------

    # -- Pre-flight check: verify a small sample of .npy files is readable -----
    # Fail fast if the manifest's paths are stale or /scratch is inaccessible,
    # rather than discovering this N retries per file into the first trial.
    # Probe 5 files to cover heterogeneous storage layouts across ~72 mice.
    _preflight_rng = random.Random(SEED)
    _probe_paths = [_preflight_rng.choice(train_pairs)[0] for _ in range(5)]
    for _probe_path in _probe_paths:
        try:
            _ = np.load(_probe_path)
        except Exception as _e:
            raise RuntimeError(
                "Pre-flight check failed: cannot read %s. "
                "Check that /scratch is accessible and the manifest is current. "
                "Underlying error: %s" % (_probe_path, _e))
    logger.info("Pre-flight .npy read OK (%d probes): %s, ...",
                len(_probe_paths), _probe_paths[0])
    del _probe_path, _probe_paths, _preflight_rng

    # -- Step 3: create and run Optuna study -----------------------------------
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # SQLite storage enables resume after crash. load_if_exists=True loads
    # the existing study so completed trials are not re-run.
    study_db = OUTPUT_DIR / "tune_multiscale_tcn.db"
    study = optuna.create_study(
        study_name="multiscale_tcn_tuning",
        direction="maximize",
        sampler=TPESampler(seed=SEED, n_startup_trials=N_STARTUP),
        pruner=MedianPruner(n_startup_trials=N_STARTUP, n_warmup_steps=3),
        storage="sqlite:///" + str(study_db.resolve()),
        load_if_exists=True,
    )
    _prior_completed = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    logger.info("Optuna storage: %s | trials in study: %d (completed: %d)",
                study_db, len(study.trials), _prior_completed)

    logger.info("Starting Optuna study | %d trials | %d random startup", N_TRIALS, N_STARTUP)

    # Wrap study.optimize() in try/except so that a catastrophic error
    # (e.g., CUDA runtime error, MemoryError, unhandled library exception)
    # does not prevent post-analysis from running on completed trials. The
    # Optuna SQLite database preserves all completed-trial data across crashes.
    try:
        study.optimize(
            lambda trial: optuna_objective(trial, train_pairs, val_pairs, device, logger),
            n_trials=N_TRIALS,
            catch=(OSError,),      # trial-level: transient I/O fails trial, not study
        )
    except KeyboardInterrupt:
        logger.warning("Study interrupted by user (Ctrl+C or SIGINT). "
                       "Proceeding with post-analysis on completed trials.")
    except BaseException as _study_err:
        logger.error("Study crashed with unhandled exception: %s: %s",
                     type(_study_err).__name__, _study_err, exc_info=True)
        logger.error("Proceeding with post-analysis on completed trials. "
                     "The Optuna SQLite database is preserved for resume.")

    # -- Step 4: save all results ----------------------------------------------
    record = save_results(study, device, logger)

    # -- Step 5: plot figures --------------------------------------------------
    plot_figures(study, record["best_val_f1"], logger)

    # -- Step 6: GPU cleanup and final output inventory ------------------------
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU cache cleared.")

    logger.info("=" * 60)
    logger.info("ALL OUTPUTS SAVED")
    logger.info("No model weights saved (by design)")
    output_files = [BEST_MS_PATH, STUDY_CSV, SUMMARY_PATH, LOG_FILE, FIG_F1, FIG_IMP]
    for p in output_files:
        status = "OK     " if Path(p).exists() else "MISSING"
        logger.info("  [%s] %s", status, p)
    logger.info("=" * 60)
    logger.info("NEXT: tune_multiscale_attention.py")


if __name__ == "__main__":
    main()


# -- REPORTING: tune_multiscale_tcn.py -----------------------------------------
# Methods section template:
#   "MultiScaleTCN hyperparameters were tuned independently from scratch
#   using Optuna TPE with 50 trials and 10 random startup trials. The tuning
#   objective was validation macro F1-score. Dilation schedules [1,2,4],
#   [8,16,32], [32,64,128] were fixed by architectural design. Early
#   stopping with patience 5 was applied within each trial."
#
# All tuned values are in best_multiscale_params.json under "hyperparameters".
# Report every key in the Methods table. Also report N_TRIALS, N_STARTUP,
# ES_PATIENCE, optimiser (AdamW + cosine annealing), and tuning metric
# (validation macro F1).
# -----------------------------------------------------------------------------
