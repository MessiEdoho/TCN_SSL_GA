# -- Section 2: Install dependencies (skip if already installed) ---------------
# !pip install torch optuna numpy scikit-learn matplotlib seaborn  # uncomment if needed
# -*- coding: utf-8 -*-
# -- Section 3: Imports, reproducibility, device detection ---------------------

import json                          # save hyperparameters and summary as JSON
import csv                           # write study results to CSV
import time                          # measure trial duration
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
    filter_unpaired_subjects,
    downsample_non_ictal,
    TCN,
    count_parameters,
    evaluate,
    run_training,
)

set_seed(42)  # set global seed immediately

# -- Device detection ----------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    cuda_ver = torch.version.cuda
    print(f"Device        : {gpu_name}")
    print(f"VRAM total    : {vram_gb:.2f} GB")
    print(f"CUDA version  : {cuda_ver}")
else:
    gpu_name = "cpu"
    print("No GPU detected -- training will run on CPU.")
    print("Tuning 60 trials on CPU may take considerably longer.")

print(f"PyTorch       : {torch.__version__}")
print(f"Optuna        : {optuna.__version__}")
print(f"Using device  : {DEVICE}")
# -- Section 4: Configuration --------------------------------------------------

# -- Data splits ---------------------------------------------------------------
# All training and tuning scripts load data from data_splits.json, produced by
# generate_data_splits.py. This ensures consistent train/val partitions across
# the entire pipeline and inherits the mouse-level leakage check.
SPLITS_PATH = Path("/scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits.json")

# -- Signal parameters ---------------------------------------------------------
FS            = 500    # sampling rate in Hz
SEGMENT_LEN   = 2500   # samples per segment: 5 s * 500 Hz

# -- Training protocol ---------------------------------------------------------
MAX_EPOCHS    = 100    # maximum epochs per trial before early stopping
ES_PATIENCE   = 10     # early stopping patience: epochs without val F1 improvement
GRAD_CLIP     = 1.0    # maximum gradient norm for gradient clipping
SEED          = 42     # random seed for reproducibility

# -- Optuna configuration ------------------------------------------------------
N_TRIALS      = 60     # total number of Optuna trials
N_STARTUP     = 15     # random exploration trials before TPE kicks in
STUDY_NAME    = "tcn_HPT_binary_optuna"  # Optuna study name

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

log.info(f"Configuration loaded. Device: {DEVICE}")
log.info(f"Output directory: {OUTPUT_DIR.resolve()}")
# -- Section 5: Dataset and DataLoader -----------------------------------------
# Load train/val file-label pairs from data_splits.json (single source of truth).
# All pipeline scripts and this notebook use the same JSON to guarantee consistent
# partitions and inherit the mouse-level leakage check from generate_data_splits.py.

if not SPLITS_PATH.exists():
    raise FileNotFoundError(
        f"data_splits.json not found at {SPLITS_PATH}. "
        f"Run generate_data_splits.py --no-test first.")

log.info(f"Loading splits from: {SPLITS_PATH}")

with open(SPLITS_PATH, "r", encoding="utf-8") as _f:
    _splits = json.load(_f)

# -- Convert records to (filepath, label) tuples for make_loader() -------------
train_pairs = [(rec["filepath"], rec["label"]) for rec in _splits["train"]]
val_pairs   = [(rec["filepath"], rec["label"]) for rec in _splits["val"]]

if not train_pairs:
    raise RuntimeError("Train partition is empty in data_splits.json.")
if not val_pairs:
    raise RuntimeError("Val partition is empty in data_splits.json.")

# -- Corpus preparation --------------------------------------------------------
# Step 1: remove subjects with no ictal segments.
train_pairs = filter_unpaired_subjects(train_pairs, logger=log)
# Step 2: downsample non-ictal to 1:4 ratio, stratified by recording.
train_pairs = downsample_non_ictal(train_pairs, ratio=4, seed=42)
# pos_weight = 1.0 is set inside run_training() (downsampling is sole correction).
log.info(f"Post-downsampling corpus: {len(train_pairs)} segments")
# -- End corpus preparation ----------------------------------------------------

# -- Class statistics (post-downsampling) --------------------------------------
n_train_ictal     = sum(1 for _, l in train_pairs if l == 1)
n_train_non_ictal = sum(1 for _, l in train_pairs if l == 0)
n_val_ictal       = sum(1 for _, l in val_pairs if l == 1)
n_val_non_ictal   = sum(1 for _, l in val_pairs if l == 0)

log.info(f"Train: {n_train_ictal} ictal + {n_train_non_ictal} non-ictal "
         f"= {len(train_pairs)} total ({100*n_train_ictal/max(len(train_pairs),1):.1f}% ictal)")
log.info(f"Val:   {n_val_ictal} ictal + {n_val_non_ictal} non-ictal "
         f"= {len(val_pairs)} total ({100*n_val_ictal/max(len(val_pairs),1):.1f}% ictal)")
log.info("pos_weight = 1.0 (set inside run_training)")
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
    num_layers = trial.suggest_int("num_layers", 5, 9)                 # depth of the TCN
    kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])  # odd kernel sizes only
    num_filters = trial.suggest_categorical("num_filters", [32, 64, 128])  # channel width
    dropout = trial.suggest_float("dropout", 0.10, 0.50, step=0.05)   # spatial dropout rate
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)   # AdamW learning rate
    wd = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)    # L2 regularisation
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64]) # segments per batch

    # -- Check receptive field constraint --------------------------------------
    # RF = 2*(2^L - 1)*(k - 1) + 1: two convolutions per block (Bai et al., 2018)
    rf = 2 * (2 ** num_layers - 1) * (kernel_size - 1) + 1
    if rf < 500:                                 # must cover at least 1 second at 500 Hz
        raise optuna.exceptions.TrialPruned()    # reject this configuration immediately

    # -- Build model -----------------------------------------------------------
    set_seed(SEED)                               # ensure reproducible weight initialisation
    model = TCN(num_layers, num_filters, kernel_size, dropout).to(DEVICE)  # move model to GPU/CPU

    log.info(f"Trial {trial.number}: L={num_layers} k={kernel_size} f={num_filters} "
             f"drop={dropout:.2f} lr={lr:.2e} wd={wd:.2e} bs={batch_size} "
             f"RF={rf} params={count_parameters(model)}")

    # -- Build data loaders ----------------------------------------------------
    # Class imbalance handled by offline downsampling (applied once before
    # Optuna loop). make_loader(train=True) shuffles the downsampled corpus.
    train_loader = make_loader(train_pairs, batch_size, train=True, device=DEVICE)
    val_loader = make_loader(val_pairs, batch_size, train=False, device=DEVICE)

    # -- Train and evaluate (pos_weight=1.0 set inside run_training) -------------
    best_val_f1 = run_training(
        model, train_loader, val_loader,
        lr=lr, weight_decay=wd,
        max_epochs=MAX_EPOCHS, patience=ES_PATIENCE,
        device=DEVICE,
        trial=trial
    )

    # -- Save checkpoint for this trial ----------------------------------------
    ckpt_path = OUTPUT_DIR / f"trial_{trial.number:03d}.pt"
    torch.save(model.cpu().state_dict(), ckpt_path)  # save on CPU for device-agnostic loading

    return best_val_f1
# -- Section 9: Run the Hyperparameter Search ----------------------------------

def trial_callback(study, trial):
    """Callback executed after each completed trial."""
    if trial.state == optuna.trial.TrialState.COMPLETE:
        p = trial.params
        log.info(f"  [Trial {trial.number:3d}] F1={trial.value:.4f} "
                 f"L={p['num_layers']} k={p['kernel_size']} f={p['num_filters']} "
                 f"lr={p['learning_rate']:.2e} device={DEVICE.type}")


# -- Create study --------------------------------------------------------------
sampler = TPESampler(seed=SEED, n_startup_trials=N_STARTUP)  # TPE with 15 random starts
pruner  = MedianPruner(n_startup_trials=N_STARTUP, n_warmup_steps=15)  # prune below median

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

study.optimize(
    optuna_objective,
    n_trials=N_TRIALS,
    callbacks=[trial_callback],
    show_progress_bar=False      # disabled for cluster/log compatibility
)

# -- Print results -------------------------------------------------------------
completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

log.info(f"\n{'='*60}")
log.info(f"Study complete: {len(completed)} completed, {len(pruned)} pruned "
         f"out of {len(study.trials)} total")

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

# -- Release GPU memory --------------------------------------------------------
if torch.cuda.is_available():
    torch.cuda.empty_cache()     # free cached GPU memory
    log.info("GPU cache cleared.")
# -- Section 10: Visualise Tuning Results --------------------------------------

completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
trial_nums = [t.number for t in completed_trials]    # trial indices
trial_f1s  = [t.value for t in completed_trials]     # corresponding val F1 values

# -- Plot 1: F1 history -------------------------------------------------------
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
fig.savefig(OUTPUT_DIR / "tuning_f1_history.png", dpi=150)  # save before showing
plt.close(fig)
log.info(f"Saved: {OUTPUT_DIR / 'tuning_f1_history.png'}")

# -- Plot 2: F1 distribution --------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(trial_f1s, bins=20, edgecolor="black", alpha=0.7)  # histogram of F1 values
best_f1 = max(trial_f1s)
ax.axvline(best_f1, color="red", linestyle="--", linewidth=2,
           label=f"Best: {best_f1:.4f}")  # vertical line at best F1
ax.set_xlabel("Validation macro F1")
ax.set_ylabel("Count")
ax.set_title("Hyperparameter Tuning -- F1 Distribution")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "tuning_f1_distribution.png", dpi=150)
plt.close(fig)
log.info(f"Saved: {OUTPUT_DIR / 'tuning_f1_distribution.png'}")

# -- Plot 3: Hyperparameter importance ----------------------------------------
try:
    importances = optuna.importance.get_param_importances(study)  # compute importance scores
    params_sorted = list(importances.keys())      # parameter names sorted by importance
    values_sorted = list(importances.values())    # corresponding importance values

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["tomato" if i == 0 else "steelblue" for i in range(len(params_sorted))]  # highlight top
    ax.barh(params_sorted[::-1], values_sorted[::-1], color=colors[::-1])  # horizontal bars
    ax.set_xlabel("Importance")
    ax.set_title("Hyperparameter Importance")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "hyperparameter_importance.png", dpi=150)
    plt.close(fig)
    log.info(f"Saved: {OUTPUT_DIR / 'hyperparameter_importance.png'}")
except Exception as e:
    log.warning(f"Could not compute hyperparameter importance: {e}")

# -- Plot 4: Parallel coordinate plot -----------------------------------------
try:
    from optuna.visualization.matplotlib import plot_parallel_coordinate
    fig = plot_parallel_coordinate(study)          # Optuna's built-in parallel coordinate plot
    fig.figure.tight_layout()
    fig.figure.savefig(OUTPUT_DIR / "parallel_coordinates.png", dpi=150)
    plt.close(fig.figure)
    log.info(f"Saved: {OUTPUT_DIR / 'parallel_coordinates.png'}")
except Exception as e:
    log.warning(f"Could not create parallel coordinate plot: {e}")
# -- Section 11a: Save best hyperparameters ------------------------------------
# bp and best_rf already computed above after study.optimize()

best_params = {
    "best_trial_number": study.best_trial.number,
    "best_val_f1": round(study.best_trial.value, 6),
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

params_path = OUTPUT_DIR / "best_params.json"
with open(params_path, "w") as f:
    json.dump(best_params, f, indent=2)
log.info(f"Saved: {params_path.resolve()}")
# -- Section 11b: Save full study results as CSV -------------------------------

csv_path = OUTPUT_DIR / "study_results.csv"
fieldnames = ["trial_number", "val_f1", "num_layers", "kernel_size", "num_filters",
              "dropout", "learning_rate", "weight_decay", "batch_size",
              "duration_seconds", "device"]

completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

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
    "best_trial_number":      study.best_trial.number,
    "best_val_f1":            round(study.best_trial.value, 6),
    "training_device":        DEVICE.type,
    "gpu_name":               gpu_name,
    "fs_hz":                  FS,
    "segment_len_samples":    SEGMENT_LEN,
    "segment_len_seconds":    SEGMENT_LEN / FS
}

summary_path = OUTPUT_DIR / "tuning_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
log.info(f"Saved: {summary_path.resolve()}")

# -- Confirm all outputs -------------------------------------------------------
log.info("")
log.info("All tuning outputs saved successfully:")
log.info(f"  1. {params_path.resolve()}")
log.info(f"  2. {csv_path.resolve()}")
log.info(f"  3. {summary_path.resolve()}")
log.info("Tuning notebook complete.")
# -- End of notebook -----------------------------------------------------------
log.info("TCN hyperparameter tuning execution finished.")
