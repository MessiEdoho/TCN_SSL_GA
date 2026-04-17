"""
tcn_HPT_binary.py -- TCN hyperparameter tuning via Optuna TPE.

Tuning protocol
---------------
N_TRIALS    = 50   total Optuna trials
MAX_EPOCHS  = 20   max epochs per trial (cosine annealing half-cycle)
ES_PATIENCE = 5    early stopping patience (epochs without val F1 improvement)

Early stopping at patience=5 within a 20-epoch budget means most trials
terminate between epochs 8-15. The reduced budget accelerates the search
while retaining enough epochs for the cosine schedule to differentiate
good from bad hyperparameter configurations (Li et al., 2017).

Validation subset: a stratified 1% subset of the validation partition
is used during tuning to reduce per-epoch evaluation cost from 67K
batches to ~6.7K batches. The subset preserves the original class ratio
and is fixed across all trials (Falkner et al., 2018).
"""

# -- Section 2: Install dependencies (skip if already installed) ---------------
# !pip install torch optuna numpy scikit-learn matplotlib seaborn  # uncomment if needed
# -*- coding: utf-8 -*-
# -- Section 3: Imports, reproducibility, device detection ---------------------

import json                          # save hyperparameters and summary as JSON
import csv                           # write study results to CSV
import gc                            # explicit garbage collection between trials
import os                            # environment variable access (dry-run mode)
import logging                       # structured logging to file and console
from pathlib import Path             # cross-platform file path handling
from datetime import datetime        # ISO 8601 timestamp for summary

import numpy as np                   # numerical operations on arrays
import torch                         # deep learning framework

import optuna                        # hyperparameter optimisation framework
from optuna.samplers import TPESampler  # Tree-structured Parzen Estimator
from optuna.pruners import MedianPruner  # prune underperforming trials

import matplotlib                    # plotting backend configuration
matplotlib.use('Agg')               # non-interactive backend for cluster use
import matplotlib.pyplot as plt      # plotting API

optuna.logging.set_verbosity(optuna.logging.WARNING)  # suppress Optuna's verbose output

# -- Import shared utilities from tcn_utils.py ---------------------------------
from tcn_utils import (
    set_seed,
    make_loader,
    # filter_unpaired_subjects,  # handled offline by create_balanced_splits.py
    downsample_val_stratified,
    TCN,
    count_parameters,
    run_training,
)

# -- Reproducibility: define SEED before first set_seed() so all seeding uses
# the same constant. Changing SEED here updates all seeding points in this
# script (module-level init, pre-flight probe, per-trial model init).
SEED = 42
set_seed(SEED)  # set global seed immediately

# -- Device detection ----------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GPU info logged after logger setup (Section 4)
# -- Section 4: Configuration --------------------------------------------------

# -- Data splits ---------------------------------------------------------------
# All training and tuning scripts load data from data_splits.json, produced by
# generate_data_splits.py. This ensures consistent train/val partitions across
# the entire pipeline and inherits the mouse-level leakage check.
# Previous (uniform downsampling): data_splits.json
# SPLITS_PATH = Path("/scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits.json")
# Current (proximity-aware downsampling): data_splits_nonictal_sampled.json
SPLITS_PATH = Path("/scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits_nonictal_sampled.json")

# -- Signal parameters ---------------------------------------------------------
FS            = 500    # sampling rate in Hz
SEGMENT_LEN   = 2500   # samples per segment: 5 s * 500 Hz

# -- Training protocol ---------------------------------------------------------
MAX_EPOCHS    = 20     # maximum epochs per trial (early stopping typically fires before 20)
ES_PATIENCE   = 5      # early stopping patience: epochs without val F1 improvement
GRAD_CLIP     = 1.0    # maximum gradient norm for gradient clipping
# SEED is defined near the top of the script (before set_seed on import)
# to ensure the module-level and per-trial seeds match.

# -- Optuna configuration ------------------------------------------------------
N_TRIALS      = 50     # total number of Optuna trials
N_STARTUP     = 10     # random exploration trials before TPE kicks in
STUDY_NAME    = "tcn_HPT_binary_optuna"  # Optuna study name

# -- Dry-run mode: set TCN_HPT_DRY_RUN=1 to cap N_TRIALS at 2 for cluster
# job shakedowns (verifies the script starts, loads data, reaches the
# training loop, and writes outputs without consuming a full tuning budget).
# Example: TCN_HPT_DRY_RUN=1 python tcn_HPT_binary.py
DRY_RUN = os.environ.get("TCN_HPT_DRY_RUN", "").lower() in ("1", "true", "yes")
if DRY_RUN:
    N_TRIALS = 2
    N_STARTUP = 1

# -- Output --------------------------------------------------------------------
OUTPUT_DIR    = Path("/home/people/22206468/scratch/OUTPUT/MODEL1_OUTPUT/TCNtuning_outputs")       # directory for all saved outputs
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # create full path if absent

# -- Logging setup -------------------------------------------------------------
log = logging.getLogger("tcn_hpt")    # named logger for this notebook
log.setLevel(logging.INFO)            # minimum log level
log.handlers.clear()                  # clear handlers from previous runs

fh = logging.FileHandler(OUTPUT_DIR / "tcn_HPT_binary.log", mode="a", encoding="utf-8")  # file handler
fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))  # timestamp format

ch = logging.StreamHandler()          # console handler
ch.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))  # same format

log.addHandler(fh)                    # attach file handler
log.addHandler(ch)                    # attach console handler

log.info("=" * 60)
log.info("tcn_HPT_binary.py")
log.info("Timestamp       : %s", datetime.now().isoformat())
log.info("Ablation role   : M1 -- TCN baseline")
log.info("Params source   : Optuna TPE search (this script)")
log.info("Purpose         : Hyperparameter tuning (no final model)")
log.info("N_TRIALS        : %d%s", N_TRIALS, "  [DRY-RUN]" if DRY_RUN else "")
log.info("MAX_EPOCHS      : %d", MAX_EPOCHS)
log.info("ES_PATIENCE     : %d", ES_PATIENCE)
log.info("Test set        : NOT loaded")
if DRY_RUN:
    log.warning("DRY-RUN MODE: N_TRIALS capped at %d for shakedown. Unset "
                "TCN_HPT_DRY_RUN environment variable for a full run.", N_TRIALS)
log.info("=" * 60)
if torch.cuda.is_available():
    log.info("GPU  : %s", torch.cuda.get_device_name(0))
    log.info("VRAM : %.2f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)
    log.info("CUDA : %s", torch.version.cuda)
    # Pre-flight GPU memory check: warn if another job is already on this GPU
    # or if VRAM is fragmented. Does not abort -- the user may still want to
    # run on a partially-occupied GPU at reduced batch size.
    try:
        _free_bytes, _total_bytes = torch.cuda.mem_get_info(0)
        _free_gb = _free_bytes / 1e9
        _total_gb = _total_bytes / 1e9
        log.info("GPU memory free: %.2f / %.2f GB", _free_gb, _total_gb)
        if _free_gb < 8.0:
            log.warning("GPU has only %.2f GB free (< 8 GB threshold). "
                        "Another process may be sharing this GPU, or VRAM "
                        "is fragmented. Trials may fail with CUDA OOM.",
                        _free_gb)
        del _free_bytes, _total_bytes, _free_gb, _total_gb
    except Exception as _e:
        log.warning("Could not query GPU memory: %s", _e)
else:
    log.info("Device: CPU")
log.info("PyTorch: %s", torch.__version__)
log.info("Optuna : %s", optuna.__version__)
log.info("Output directory: %s", OUTPUT_DIR.resolve())
# -- Section 5: Dataset and DataLoader -----------------------------------------
# Load train/val file-label pairs from data_splits_nonictal_sampled.json (produced
# by create_balanced_splits.py). All pipeline scripts load the same balanced
# manifest to guarantee consistent partitions and inherit the mouse-level
# leakage check from generate_data_splits.py (upstream).

if not SPLITS_PATH.exists():
    raise FileNotFoundError(
        f"data_splits_nonictal_sampled.json not found at {SPLITS_PATH}. "
        f"Run create_balanced_splits.py first (which itself requires "
        f"data_splits.json from generate_data_splits.py).")

log.info(f"Loading splits from: {SPLITS_PATH}")

with open(SPLITS_PATH, "r", encoding="utf-8") as _f:
    _splits = json.load(_f)

# -- Convert records to (filepath, label) tuples for make_loader() -------------
train_pairs = [(rec["filepath"], rec["label"]) for rec in _splits["train"]]
val_pairs   = [(rec["filepath"], rec["label"]) for rec in _splits["val"]]

if not train_pairs:
    raise RuntimeError("Train partition is empty in data_splits_nonictal_sampled.json.")
if not val_pairs:
    raise RuntimeError("Val partition is empty in data_splits_nonictal_sampled.json.")

# -- Corpus preparation --------------------------------------------------------
# Subject exclusion (m254), 1:4 downsampling, and extreme-segment filtering
# are ALL handled offline by create_balanced_splits.py. The manifest is
# already clean and balanced -- no further corpus preparation is needed here.
# train_pairs = filter_unpaired_subjects(train_pairs, logger=log)
log.info(f"Training corpus: {len(train_pairs)} segments (from balanced manifest)")

# Stratified 1% validation subset for tuning speed. See STUDY_REPORT.txt
# Section on "Validation subset during tuning" for timing justification.
val_pairs = downsample_val_stratified(val_pairs, fraction=0.01, seed=SEED)
log.info(f"Val subset for tuning: {len(val_pairs)} segments (1% stratified)")
# -- End corpus preparation ----------------------------------------------------

# -- Class statistics (from pre-balanced manifest) -----------------------------
n_train_ictal     = sum(1 for _, l in train_pairs if l == 1)
n_train_non_ictal = sum(1 for _, l in train_pairs if l == 0)
n_val_ictal       = sum(1 for _, l in val_pairs if l == 1)
n_val_non_ictal   = sum(1 for _, l in val_pairs if l == 0)

log.info(f"Train: {n_train_ictal} ictal + {n_train_non_ictal} non-ictal "
         f"= {len(train_pairs)} total ({100*n_train_ictal/max(len(train_pairs),1):.1f}% ictal)")
log.info(f"Val:   {n_val_ictal} ictal + {n_val_non_ictal} non-ictal "
         f"= {len(val_pairs)} total ({100*n_val_ictal/max(len(val_pairs),1):.1f}% ictal)")
log.info("pos_weight = 1.0 (set inside run_training)")

# -- Pre-flight check: verify at least one .npy file is readable ---------------
# Fail fast if the manifest's paths are stale or /scratch is inaccessible,
# rather than discovering this 50 * 3 retries per file into the first trial.
import random as _preflight_random
_preflight_random.seed(SEED)
_probe_path = _preflight_random.choice(train_pairs)[0]
try:
    _ = np.load(_probe_path)
    log.info(f"Pre-flight .npy read OK: {_probe_path}")
except Exception as _e:
    raise RuntimeError(
        f"Pre-flight check failed: cannot read {_probe_path}. "
        f"Check that /scratch is accessible and the manifest is current. "
        f"Underlying error: {_e}")
del _probe_path, _preflight_random
# -- Section 8: Optuna Objective Function --------------------------------------

def optuna_objective(trial):
    """Optuna objective: train a TCN with proposed hyperparameters, return val F1.

    Parameters
    ----------
    trial : optuna.trial.Trial

    Returns
    -------
    best_val_f1 : float
    """
    # -- Sample hyperparameters ------------------------------------------------
    # num_layers lower bound is 6 because L=5 produces RF < 500 samples at every
    # kernel size (RF_max = 373 at k=7), which is insufficient for seizure
    # detection over a 5-second window. Restricting to [6, 9] eliminates all
    # three L=5 configurations upfront so TPE's startup trials are not wasted.
    num_layers = trial.suggest_int("num_layers", 6, 9)                 # depth of the TCN
    kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])  # odd kernel sizes only
    num_filters = trial.suggest_categorical("num_filters", [32, 64, 128])  # channel width
    dropout = trial.suggest_float("dropout", 0.10, 0.50, step=0.05)   # spatial dropout rate
    lr = trial.suggest_float("learning_rate", 1e-4, 5e-4, log=True)   # AdamW learning rate
    wd = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)    # L2 regularisation
    batch_size = trial.suggest_categorical("batch_size", [32, 64]) # segments per batch

    # -- Check receptive field constraint --------------------------------------
    # RF = 2*(2^L - 1)*(k - 1) + 1: two convolutions per block (Bai et al., 2018).
    # With num_layers in [6, 9], only (L=6, k=3) produces RF < 500 (=253). All
    # other 11 of 12 (L, k) combinations pass -- a 92% acceptance rate vs 73%
    # with the previous [5, 9] range.
    rf = 2 * (2 ** num_layers - 1) * (kernel_size - 1) + 1
    if rf < 500:                                 # must cover at least 1 second at 500 Hz
        raise optuna.exceptions.TrialPruned()    # reject this configuration immediately

    # -- Resource cleanup: release model and loaders on exit, even on exception
    # (TrialPruned mid-epoch, OSError from catch=..., or normal completion).
    # Prevents GPU VRAM and DataLoader worker accumulation across 50 trials.
    model = None
    train_loader = None
    val_loader = None
    try:
        # -- Build model -------------------------------------------------------
        set_seed(SEED)                           # ensure reproducible weight initialisation
        model = TCN(num_layers, num_filters, kernel_size, dropout).to(DEVICE)

        log.info(f"Trial {trial.number}: L={num_layers} k={kernel_size} f={num_filters} "
                 f"drop={dropout:.2f} lr={lr:.2e} wd={wd:.2e} bs={batch_size} "
                 f"RF={rf} params={count_parameters(model)}")

        # -- Build data loaders ------------------------------------------------
        # Class imbalance handled by offline downsampling in create_balanced_splits.py.
        # make_loader(train=True) shuffles the pre-balanced corpus.
        train_loader = make_loader(train_pairs, batch_size, train=True, device=DEVICE)
        val_loader = make_loader(val_pairs, batch_size, train=False, device=DEVICE)

        # -- Train and evaluate (pos_weight=1.0 set inside run_training) -------
        best_val_f1 = run_training(
            model, train_loader, val_loader,
            lr=lr, weight_decay=wd,
            max_epochs=MAX_EPOCHS, patience=ES_PATIENCE,
            device=DEVICE,
            max_grad_norm=GRAD_CLIP,       # explicitly pass configured gradient clip
            trial=trial, logger=log
        )

        return best_val_f1
    finally:
        # Release GPU memory and DataLoader workers between trials. Without
        # this, persistent_workers=True leaves worker processes alive across
        # trials and cached GPU allocations accumulate, eventually causing OOM.
        del model, train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
# -- Section 9: Run the Hyperparameter Search ----------------------------------

def trial_callback(study, trial):
    """Callback executed after each completed trial."""
    if trial.state == optuna.trial.TrialState.COMPLETE:
        p = trial.params
        log.info(f"  [Trial {trial.number:3d}] F1={trial.value:.4f} "
                 f"L={p['num_layers']} k={p['kernel_size']} f={p['num_filters']} "
                 f"drop={p['dropout']:.2f} lr={p['learning_rate']:.2e} "
                 f"wd={p['weight_decay']:.2e} bs={p['batch_size']} "
                 f"device={DEVICE.type}")


# -- Create study --------------------------------------------------------------
sampler = TPESampler(seed=SEED, n_startup_trials=N_STARTUP)  # TPE with N_STARTUP random starts
pruner  = MedianPruner(n_startup_trials=N_STARTUP, n_warmup_steps=3)   # prune below median after epoch 3

# SQLite storage enables resume after crash: re-running the script picks up
# from the last completed trial. load_if_exists=True loads the existing study
# if the database already contains one with the same study_name.
STUDY_DB = OUTPUT_DIR / "tcn_hpt.db"
study = optuna.create_study(
    study_name=STUDY_NAME,
    direction="maximize",        # maximise validation macro F1
    sampler=sampler,
    pruner=pruner,
    storage="sqlite:///" + str(STUDY_DB.resolve()),
    load_if_exists=True,
)
log.info(f"Optuna storage: {STUDY_DB} | completed trials so far: {len(study.trials)}")

log.info(f"Starting Optuna study: {N_TRIALS} trials, TPE sampler, MedianPruner")
log.info(f"Training device: {DEVICE.type}")

# Wrap study.optimize() in try/except so that a catastrophic error
# (e.g., CUDA runtime error, MemoryError, unhandled library exception)
# does not prevent post-analysis from running on completed trials. The
# Optuna SQLite database preserves all completed-trial data across crashes,
# so plots, CSV, and summary JSON can still be generated from whatever
# trials finished before the failure. The user can then investigate the
# error in the log and decide whether to resume or abort.
try:
    study.optimize(
        optuna_objective,
        n_trials=N_TRIALS,
        callbacks=[trial_callback],
        catch=(OSError,),        # trial-level: transient I/O fails trial, not study
        show_progress_bar=False  # disabled for cluster/log compatibility
    )
except KeyboardInterrupt:
    log.warning("Study interrupted by user (Ctrl+C or SIGINT). "
                "Proceeding with post-analysis on completed trials.")
except BaseException as _study_err:
    log.error("Study crashed with unhandled exception: %s: %s",
              type(_study_err).__name__, _study_err, exc_info=True)
    log.error("Proceeding with post-analysis on completed trials. "
              "The Optuna SQLite database is preserved for resume.")

# -- Print results -------------------------------------------------------------
completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

log.info(f"\n{'='*60}")
log.info(f"Study complete: {len(completed)} completed, {len(pruned)} pruned "
         f"out of {len(study.trials)} total")

# Guard: study.best_trial raises ValueError if no trial completed successfully.
# This can happen if every trial raised TrialPruned or hit the OSError catch.
# Without this guard, all downstream sections (log summary, plots, JSON, CSV)
# would crash and no output would be saved.
if len(completed) == 0:
    log.error("No trials completed successfully. Check earlier log lines "
              "for per-trial errors. No best-model outputs will be saved.")
    best = None
    bp = None
    best_rf = None
else:
    best = study.best_trial
    bp = best.params
    # RF = 2*(2^L - 1)*(k - 1) + 1: two convolutions per block (Bai et al., 2018)
    best_rf = 2 * (2 ** bp["num_layers"] - 1) * (bp["kernel_size"] - 1) + 1
    rf_check = "PASS" if best_rf >= 500 else "WARNING: RF < 500"

    log.info(f"Best trial     : {best.number}")
    log.info(f"Best val F1    : {best.value:.6f}")
    log.info(f"  num_layers   : {bp['num_layers']}")
    log.info(f"  kernel_size  : {bp['kernel_size']}")
    log.info(f"  num_filters  : {bp['num_filters']}")
    log.info(f"  dropout      : {bp['dropout']:.2f}")
    log.info(f"  learning_rate: {bp['learning_rate']:.2e}")
    log.info(f"  weight_decay : {bp['weight_decay']:.2e}")
    log.info(f"  batch_size   : {bp['batch_size']}")
    log.info(f"  RF           : {best_rf} samples ({best_rf/FS:.2f} s) [{rf_check}]")
    log.info(f"  Device       : {DEVICE.type}")

# Note: GPU cache is already cleared inside optuna_objective()'s finally block
# after every trial, so no study-level empty_cache is needed here.
# -- Section 10: Visualise Tuning Results --------------------------------------

completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
trial_nums = [t.number for t in completed_trials]    # trial indices
trial_f1s  = [t.value for t in completed_trials]     # corresponding val F1 values

# All plotting is skipped if no trials completed. argmax/max on an empty
# list would crash; importance computation requires at least two trials.
if len(completed_trials) == 0:
    log.warning("No completed trials -- skipping all tuning plots.")
else:
    # -- Plot 1: F1 history ----------------------------------------------------
    running_best = np.maximum.accumulate(trial_f1s)      # running best F1 up to each trial
    best_idx = int(np.argmax(trial_f1s))                 # index of the best trial in the list

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(trial_nums, trial_f1s, alpha=0.6, s=30, label="Trial F1")  # scatter of all F1 values
    ax.plot(trial_nums, running_best, color="red", linewidth=2, label="Running best")  # best-so-far line
    ax.scatter([trial_nums[best_idx]], [trial_f1s[best_idx]],
               color="gold", s=150, zorder=5, edgecolors="black", marker="*",
               label=f"Best: {trial_f1s[best_idx]:.4f}")  # highlight best trial
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Validation macro F1")
    ax.set_title("Hyperparameter Tuning -- F1 History")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tuning_f1_history.png", dpi=150)
    plt.close(fig)
    log.info(f"Saved: {OUTPUT_DIR / 'tuning_f1_history.png'}")

    # -- Plot 2: F1 distribution -----------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(trial_f1s, bins=20, edgecolor="black", alpha=0.7)
    best_f1 = max(trial_f1s)
    ax.axvline(best_f1, color="red", linestyle="--", linewidth=2,
               label=f"Best: {best_f1:.4f}")
    ax.set_xlabel("Validation macro F1")
    ax.set_ylabel("Count")
    ax.set_title("Hyperparameter Tuning -- F1 Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tuning_f1_distribution.png", dpi=150)
    plt.close(fig)
    log.info(f"Saved: {OUTPUT_DIR / 'tuning_f1_distribution.png'}")

    # -- Plot 3: Hyperparameter importance -------------------------------------
    try:
        importances = optuna.importance.get_param_importances(study)
        params_sorted = list(importances.keys())
        values_sorted = list(importances.values())

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["tomato" if i == 0 else "steelblue" for i in range(len(params_sorted))]
        ax.barh(params_sorted[::-1], values_sorted[::-1], color=colors[::-1])
        ax.set_xlabel("Importance")
        ax.set_title("Hyperparameter Importance")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "hyperparameter_importance.png", dpi=150)
        plt.close(fig)
        log.info(f"Saved: {OUTPUT_DIR / 'hyperparameter_importance.png'}")
    except Exception as e:
        log.warning(f"Could not compute hyperparameter importance: {e}")

    # -- Plot 4: Parallel coordinate plot --------------------------------------
    try:
        from optuna.visualization.matplotlib import plot_parallel_coordinate
        fig = plot_parallel_coordinate(study)
        fig.figure.tight_layout()
        fig.figure.savefig(OUTPUT_DIR / "parallel_coordinates.png", dpi=150)
        plt.close(fig.figure)
        log.info(f"Saved: {OUTPUT_DIR / 'parallel_coordinates.png'}")
    except Exception as e:
        log.warning(f"Could not create parallel coordinate plot: {e}")
# -- Section 11a: Save best hyperparameters ------------------------------------
# bp and best_rf already computed above after study.optimize().
# Only save best_params.json if at least one trial completed successfully.
params_path = OUTPUT_DIR / "best_params.json"
if best is None:
    log.warning(f"Skipping {params_path.name}: no completed trials to record.")
else:
    best_params = {
        "best_trial_number": best.number,
        "best_val_f1": round(best.value, 6),
        "receptive_field_samples": best_rf,
        "receptive_field_seconds": round(best_rf / FS, 4),
        "training_device": DEVICE.type,
        "hyperparameters": {
            "num_layers":    bp["num_layers"],
            "kernel_size":   bp["kernel_size"],
            "num_filters":   bp["num_filters"],
            "dropout":       bp["dropout"],
            "learning_rate": bp["learning_rate"],
            "weight_decay":  bp["weight_decay"],
            "batch_size":    bp["batch_size"]
        }
    }
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    log.info(f"Saved: {params_path.resolve()}")
# -- Section 11b: Save full study results as CSV -------------------------------

csv_path = OUTPUT_DIR / "study_results.csv"
fieldnames = ["trial_number", "val_f1", "num_layers", "kernel_size", "num_filters",
              "dropout", "learning_rate", "weight_decay", "batch_size",
              "duration_seconds", "device"]

# Reuse the `completed` list computed after study.optimize() (line 309).
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for t in completed:
        dur = (t.datetime_complete - t.datetime_start).total_seconds() if t.datetime_complete else 0
        writer.writerow({
            "trial_number":    t.number,
            "val_f1":          round(t.value, 6),
            "num_layers":      t.params["num_layers"],
            "kernel_size":     t.params["kernel_size"],
            "num_filters":     t.params["num_filters"],
            "dropout":         t.params["dropout"],
            "learning_rate":   t.params["learning_rate"],
            "weight_decay":    t.params["weight_decay"],
            "batch_size":      t.params["batch_size"],
            "duration_seconds": round(dur, 1),
            "device":          DEVICE.type
        })

log.info(f"Saved: {csv_path.resolve()}")
# -- Section 11c: Save tuning summary -----------------------------------------

summary = {
    "timestamp":              datetime.now().isoformat(),
    "study_name":             STUDY_NAME,
    "n_trials_requested":     N_TRIALS,
    "n_trials_completed":     len([t for t in study.trials
                                   if t.state == optuna.trial.TrialState.COMPLETE]),
    "n_trials_pruned":        len([t for t in study.trials
                                   if t.state == optuna.trial.TrialState.PRUNED]),
    # best_trial_number / best_val_f1 are None if no trial completed.
    "best_trial_number":      (best.number if best is not None else None),
    "best_val_f1":            (round(best.value, 6) if best is not None else None),
    "training_device":        DEVICE.type,
    "gpu_name":               torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "fs_hz":                  FS,
    "segment_len_samples":    SEGMENT_LEN,
    "segment_len_seconds":    SEGMENT_LEN / FS
}

summary_path = OUTPUT_DIR / "tuning_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
log.info(f"Saved: {summary_path.resolve()}")

# -- Confirm all outputs -------------------------------------------------------
# Enumerate only the files actually written. best_params.json is skipped
# when no trials completed, so checking each path's existence avoids
# misleading "saved successfully" messages for files that don't exist.
log.info("")
log.info("Tuning outputs:")
for _out_path in [params_path, csv_path, summary_path]:
    _status = "OK     " if _out_path.exists() else "MISSING"
    log.info(f"  [{_status}] {_out_path.resolve()}")
log.info("TCN hyperparameter tuning execution finished.")
