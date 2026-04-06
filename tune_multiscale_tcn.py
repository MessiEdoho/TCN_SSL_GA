"""
tune_multiscale_tcn.py
======================
Tunes all hyperparameters of MultiScaleTCN from
scratch using Optuna TPE.

No parameters are transferred from the single-branch
TCN tuning (best_params.json is not loaded here).
Independent tuning ensures MultiScaleTCN is assessed
at its true optimum.

Architecture: MultiScaleTCN (from tcn_utils.py)
Dilation schedules fixed by design:
  Branch 1: [1, 2, 4]
  Branch 2: [2, 4, 8]
  Branch 3: [4, 8, 16]

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
outputs/best_multiscale_params.json
outputs/multiscale_study_results.csv
outputs/multiscale_tuning_summary.json
outputs/logs/tune_multiscale_tcn.log
outputs/figures/multiscale_f1_history.png
outputs/figures/multiscale_importance.png

Usage
-----
python tune_multiscale_tcn.py

Pipeline position
-----------------
1. generate_data_splits.py --no-test
2. tcn_HPT_binary.ipynb
3. tune_multiscale_tcn.py      <- this script
4. tune_multiscale_attention.py
5. Training notebooks
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import json
import logging
import sys
import csv
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
    filter_unpaired_subjects,
    downsample_non_ictal,
    MultiScaleTCN,
    count_parameters,
    train_one_epoch,
    evaluate,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED              = 42                                 # global reproducibility seed
MAX_EPOCHS        = 100                                # max training epochs per trial
ES_PATIENCE       = 10                                 # early stopping patience (epochs)
N_TRIALS          = 50                                 # total Optuna trials
N_STARTUP         = 15                                 # random startup before TPE
FS                = 500                                # EEG sampling rate (Hz)
SEGMENT_LEN       = 2500                               # samples per segment (5 s at 500 Hz)
SEGMENT_SEC       = 5.0                                # segment duration in seconds

OUTPUT_DIR        = Path("outputs")
LOG_DIR           = OUTPUT_DIR / "logs"
FIGURE_DIR        = OUTPUT_DIR / "figures"
SPLITS_PATH       = Path("data_splits_outputs") / "data_splits.json"
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
DROPOUT_MAX         = 0.5                              # spatial dropout upper bound
DROPOUT_STEP        = 0.05                             # dropout step size
FUSION_CHOICES      = ["concat", "average"]            # branch fusion strategies
LR_MIN              = 1e-4                             # AdamW lr lower bound
LR_MAX              = 1e-2                             # AdamW lr upper bound
WD_MIN              = 1e-5                             # weight decay lower bound
WD_MAX              = 1e-3                             # weight decay upper bound
BATCH_CHOICES       = [16, 32, 64]                     # batch size candidates

# Dilation schedules -- fixed by architectural design, not tuned
BRANCH1_DILATIONS   = [1, 2, 4]                        # fine scale
BRANCH2_DILATIONS   = [2, 4, 8]                        # medium scale
BRANCH3_DILATIONS   = [4, 8, 16]                       # coarse scale


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------
def setup_logging():
    """Create output directories and configure the module logger.

    FileHandler: LOG_FILE mode='w' level DEBUG.
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

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
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
    """Load train and val file-label pairs from data_splits.json.

    Parameters
    ----------
    logger : logging.Logger

    Returns
    -------
    tuple of (train_pairs, val_pairs)
        Each is a list of (filepath: str, label: int) tuples.

    Raises
    ------
    FileNotFoundError if data_splits.json is absent.
    RuntimeError if either partition is empty.
    """
    if not SPLITS_PATH.exists():
        logger.error(
            "data_splits.json not found at %s. "
            "Run generate_data_splits.py --no-test first.", SPLITS_PATH)
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

    # -- b. Set seed for reproducible weight initialisation --------------------
    set_seed(SEED)

    # -- c. Instantiate model --------------------------------------------------
    model = MultiScaleTCN(
        num_filters=num_filters,
        kernel_size=kernel_size,
        dropout=dropout,
        branch1_dilations=BRANCH1_DILATIONS,
        branch2_dilations=BRANCH2_DILATIONS,
        branch3_dilations=BRANCH3_DILATIONS,
        fusion=fusion,
    ).to(device)

    # Log parameter count on first trial
    if trial.number == 0:
        n_params = count_parameters(model)
        logger.info("Trial 0 -- total trainable parameters: %s", "{:,}".format(n_params))

    # -- d. Build data loaders -------------------------------------------------
    train_loader = make_loader(train_pairs, batch_size=batch_size, train=True, device=device)
    val_loader   = make_loader(val_pairs, batch_size=batch_size, train=False, device=device)

    # -- e. Build optimiser and scheduler --------------------------------------
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=MAX_EPOCHS, eta_min=learning_rate * 0.01)

    # -- f. Build loss -- imbalance handled by offline downsampling; pos_weight=1.0
    pos_weight = torch.tensor([1.0], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # -- g. Training loop with early stopping ----------------------------------
    best_val_f1   = 0.0
    epochs_no_imp = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimiser, criterion, device)
        val_f1, _, _ = evaluate(model, val_loader, device)
        scheduler.step()

        if val_f1 > best_val_f1:
            best_val_f1   = val_f1
            epochs_no_imp = 0
        else:
            epochs_no_imp += 1

        trial.report(val_f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if epochs_no_imp >= ES_PATIENCE:
            logger.debug(
                "Trial %d: early stop at epoch %d | best_f1=%.4f",
                trial.number, epoch, best_val_f1)
            break

    # -- h. One-line trial summary ---------------------------------------------
    logger.info(
        "Trial %3d | F1=%.4f | filters=%d | k=%d | drop=%.2f | "
        "fusion=%s | lr=%.2e | bs=%d",
        trial.number, best_val_f1, num_filters, kernel_size,
        dropout, fusion, learning_rate, batch_size)

    return best_val_f1


# ---------------------------------------------------------------------------
# save_results
# ---------------------------------------------------------------------------
def save_results(study, device, logger):
    """Save all tuning outputs. No model weights saved.

    Saves:
        outputs/best_multiscale_params.json
        outputs/multiscale_study_results.csv
        outputs/multiscale_tuning_summary.json

    Parameters
    ----------
    study  : optuna.Study -- completed Optuna study
    device : torch.device
    logger : logging.Logger

    Returns
    -------
    dict -- the best_multiscale_params record
    """
    best_trial  = study.best_trial
    best_val_f1 = best_trial.value
    best_params = best_trial.params
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    logger.info("=" * 60)
    logger.info("TUNING COMPLETE")
    logger.info("  Best trial      : #%d", best_trial.number)
    logger.info("  Best val F1     : %.4f", best_val_f1)
    logger.info("  Completed       : %d", len(completed))
    logger.info("  Pruned          : %d", len(pruned))
    logger.info("  Best parameters :")
    for k, v in best_params.items():
        logger.info("    %-25s: %s", k, v)

    # -- a. Save best_multiscale_params.json -----------------------------------
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

    # -- b. Save multiscale_study_results.csv ----------------------------------
    param_names = sorted(best_params.keys())
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
        "best_trial_number":   best_trial.number,
        "best_val_f1":         round(best_val_f1, 6),
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
        outputs/figures/multiscale_f1_history.png
        outputs/figures/multiscale_importance.png

    Parameters
    ----------
    study       : optuna.Study
    best_val_f1 : float
    logger      : logging.Logger
    """
    # -- a. F1 history figure --------------------------------------------------
    completed  = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    trial_nums = [t.number for t in completed]
    f1_vals    = [t.value for t in completed]

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
        best_idx = running_best.index(best_val_f1)
        axes[0].annotate(
            "Best: %.4f" % best_val_f1,
            xy=(trial_nums[best_idx], best_val_f1),
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
    logger.info("Timestamp: %s", datetime.datetime.now().isoformat())
    logger.info("No parameter transfer from single-branch TCN")
    logger.info("No final model trained in this script")
    logger.info("=" * 60)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        logger.info("GPU  : %s", torch.cuda.get_device_name(0))
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("VRAM : %.2f GB", vram)
        logger.info("CUDA : %s", torch.version.cuda)
    else:
        logger.info("Device: CPU")
    logger.info("PyTorch: %s", torch.__version__)

    # -- Step 2: load data splits ----------------------------------------------
    train_pairs, val_pairs = load_splits(logger)

    # -- Corpus preparation ----------------------------------------------------
    # Step 1: remove subjects with no ictal segments.
    train_pairs = filter_unpaired_subjects(train_pairs, logger=logger)
    # Step 2: downsample non-ictal to 1:4 ratio, stratified by recording.
    train_pairs = downsample_non_ictal(train_pairs, ratio=4, seed=42)
    # Step 3: pos_weight = 1.0 (downsampling is the sole imbalance correction).
    pos_weight = torch.tensor([1.0], dtype=torch.float32)
    logger.info("Post-downsampling corpus: %d segments", len(train_pairs))
    # -- End corpus preparation ------------------------------------------------

    # -- Step 3: create and run Optuna study -----------------------------------
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name="multiscale_tcn_tuning",
        direction="maximize",
        sampler=TPESampler(seed=SEED, n_startup_trials=N_STARTUP),
        pruner=MedianPruner(n_startup_trials=N_STARTUP, n_warmup_steps=10),
    )

    logger.info("Starting Optuna study | %d trials | %d random startup", N_TRIALS, N_STARTUP)

    study.optimize(
        lambda trial: optuna_objective(trial, train_pairs, val_pairs, device, logger),
        n_trials=N_TRIALS,
    )

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
#   using Optuna TPE with 50 trials and 15 random startup trials. The tuning
#   objective was validation macro F1-score. Dilation schedules [1,2,4],
#   [2,4,8], [4,8,16] were fixed by architectural design. Early stopping
#   with patience 10 was applied within each trial."
#
# All tuned values are in best_multiscale_params.json under "hyperparameters".
# Report every key in the Methods table. Also report N_TRIALS, N_STARTUP,
# ES_PATIENCE, optimiser (AdamW + cosine annealing), and tuning metric
# (validation macro F1).
# -----------------------------------------------------------------------------
