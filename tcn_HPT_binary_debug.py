"""
tcn_HPT_binary_debug.py
========================
Debug script: runs ONE Optuna trial with a data quality filter applied
between downsample_non_ictal() and make_loader().

Purpose
-------
Diagnose whether the loss=2,020,323,589 explosion in tcn_HPT_binary.py
is caused by extreme-amplitude segments that were never normalised during
preprocessing (raw ADC values up to 4.36e+17 found in the corpus).

Three-step data quality approach tested here
---------------------------------------------
Step A -- filter_unpaired_subjects(): remove ictal-free subjects (existing).
Step B -- downsample_non_ictal(): 1:4 ratio (existing).
Step C -- filter_extreme_segments() (NEW): scan each .npy file and remove
          any segment whose max(|x|) exceeds AMPLITUDE_THRESHOLD.
          This is a hard filter, not a clipping operation. Segments with
          catastrophic values (10^15+) are unsalvageable -- they were never
          z-scored. Segments with mild outliers (100-400) are retained
          because they are legitimate z-scored EEG with artefact spikes.

What to check in the output
----------------------------
1. How many segments were removed by filter_extreme_segments().
2. Whether epoch 1 loss is in the normal range (0.3-1.0).
3. Whether loss decreases over epochs (convergence).
4. Whether F1 rises above 0.5 (learning signal).

If this works, filter_extreme_segments() can be integrated into the main
pipeline (tcn_utils.py) at your approval.

Usage
-----
python tcn_HPT_binary_debug.py

Output directory
----------------
{OUTPUT_DIR}/debug/   (separate from the main tuning outputs)
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

import matplotlib
matplotlib.use('Agg')

from tcn_utils import (
    set_seed,
    make_loader,
    filter_unpaired_subjects,
    downsample_non_ictal,
    TCN,
    count_parameters,
    train_one_epoch,
    evaluate,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED              = 42
FS                = 500
SEGMENT_LEN       = 2500
MAX_EPOCHS        = 100
ES_PATIENCE       = 10
N_TRIALS          = 1      # ONE trial only -- debug run
N_STARTUP         = 0      # no random startup needed for 1 trial

# Amplitude threshold for filtering extreme segments.
# Z-scored EEG should have values in the range [-10, +10] for >99.99% of samples.
# Values above 1000 indicate raw unscaled data (preprocessing failure).
# Values in the 100-400 range are legitimate artefact spikes in z-scored EEG.
AMPLITUDE_THRESHOLD = 1000.0

SPLITS_PATH = Path("/scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits.json")
OUTPUT_DIR  = Path("/home/people/22206468/scratch/OUTPUT/MODEL1_OUTPUT/TCNtuning_outputs")
DEBUG_DIR   = OUTPUT_DIR / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Logging -- writes to debug/ subdirectory
# ---------------------------------------------------------------------------
log = logging.getLogger("tcn_hpt_debug")
log.setLevel(logging.DEBUG)
log.handlers.clear()

fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

fh = logging.FileHandler(DEBUG_DIR / "debug_trial.log", mode="w", encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(fmt)

sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.INFO)
sh.setFormatter(fmt)

log.addHandler(fh)
log.addHandler(sh)


# ---------------------------------------------------------------------------
# filter_extreme_segments -- the NEW data quality function under test
# ---------------------------------------------------------------------------
def filter_extreme_segments(pairs, threshold, logger):
    """Remove segments whose max absolute amplitude exceeds threshold.

    This is a hard filter, not a clipping operation. Segments with
    catastrophic values (raw unscaled EEG from preprocessing failures)
    are removed entirely because their waveform morphology is not
    recoverable by clipping.

    Parameters
    ----------
    pairs : list of (str or Path, int) tuples
        File-label pairs after downsampling.
    threshold : float
        Maximum allowed max(|x|) per segment. Segments exceeding
        this value are removed. Recommended: 1000.0 for z-scored EEG.
    logger : logging.Logger
        Logs the number of removed segments, affected subjects, and
        the worst value found.

    Returns
    -------
    list of (str or Path, int) tuples
        Pairs with extreme segments removed.
    """
    t0 = time.time()
    clean = []
    removed = []
    worst_val = 0.0
    worst_file = ""

    for i, (fp, label) in enumerate(pairs):
        x = np.load(fp)
        mx = float(np.abs(x).max())

        if mx > threshold:
            removed.append((fp, label, mx))
            if mx > worst_val:
                worst_val = mx
                worst_file = fp
        else:
            clean.append((fp, label))

        if (i + 1) % 50000 == 0:
            logger.info("  filter_extreme_segments: scanned %d/%d | removed so far: %d",
                        i + 1, len(pairs), len(removed))

    elapsed = time.time() - t0

    # Log summary
    logger.info("=" * 60)
    logger.info("filter_extreme_segments summary:")
    logger.info("  Threshold          : %.1f", threshold)
    logger.info("  Total scanned      : %d", len(pairs))
    logger.info("  Removed            : %d (%.2f%%)",
                len(removed), 100 * len(removed) / max(len(pairs), 1))
    logger.info("  Retained           : %d", len(clean))
    logger.info("  Scan time          : %.1f s", elapsed)

    if removed:
        logger.info("  Worst value        : %.4e in %s", worst_val, worst_file)

        # Per-subject breakdown of removed segments
        subject_counts = {}
        for fp, label, mx in removed:
            subj = Path(fp).stem.split("_", 1)[0]
            if subj not in subject_counts:
                subject_counts[subj] = {"count": 0, "worst": 0.0}
            subject_counts[subj]["count"] += 1
            subject_counts[subj]["worst"] = max(subject_counts[subj]["worst"], mx)

        logger.info("  Affected subjects  : %d", len(subject_counts))
        for subj in sorted(subject_counts):
            info = subject_counts[subj]
            logger.info("    %s: %d segments removed (worst=%.2e)",
                        subj, info["count"], info["worst"])

        # Breakdown by severity
        catastrophic = sum(1 for _, _, mx in removed if mx > 1e6)
        mild = len(removed) - catastrophic
        logger.info("  Catastrophic (>1e6): %d", catastrophic)
        logger.info("  Mild (%.0f-1e6)     : %d", threshold, mild)

    logger.info("=" * 60)

    # Save removed segments list to JSON for audit
    removed_log = [{"filepath": fp, "label": label, "max_abs": mx}
                   for fp, label, mx in removed]
    with open(DEBUG_DIR / "removed_segments.json", "w", encoding="utf-8") as f:
        json.dump(removed_log, f, indent=2)
    logger.info("Saved: %s (%d entries)", DEBUG_DIR / "removed_segments.json", len(removed_log))

    return clean


# ---------------------------------------------------------------------------
# Main debug run
# ---------------------------------------------------------------------------
log.info("=" * 65)
log.info("tcn_HPT_binary_debug.py -- ONE-TRIAL DEBUG RUN")
log.info("Timestamp: %s", datetime.now().isoformat())
log.info("Purpose: test filter_extreme_segments() before integrating into main pipeline")
log.info("Amplitude threshold: %.1f", AMPLITUDE_THRESHOLD)
log.info("=" * 65)

set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Device: %s", DEVICE)
if torch.cuda.is_available():
    log.info("GPU: %s | VRAM: %.2f GB",
             torch.cuda.get_device_name(0),
             torch.cuda.get_device_properties(0).total_memory / 1e9)

# -- Load data splits ----------------------------------------------------------
if not SPLITS_PATH.exists():
    raise FileNotFoundError("data_splits.json not found at %s" % SPLITS_PATH)

with open(SPLITS_PATH, "r", encoding="utf-8") as f:
    splits = json.load(f)

train_pairs = [(rec["filepath"], rec["label"]) for rec in splits["train"]]
val_pairs   = [(rec["filepath"], rec["label"]) for rec in splits["val"]]
log.info("Raw train pairs: %d | Val pairs: %d", len(train_pairs), len(val_pairs))

# -- Step A: filter unpaired subjects -----------------------------------------
train_pairs = filter_unpaired_subjects(train_pairs, logger=log)

# -- Step B: downsample non-ictal to 1:4 ratio --------------------------------
train_pairs = downsample_non_ictal(train_pairs, ratio=4, seed=SEED)
n_ic = sum(1 for _, l in train_pairs if l == 1)
n_nic = sum(1 for _, l in train_pairs if l == 0)
log.info("Post-downsampling: %d total | %d ictal | %d non-ictal | ratio=%.2f",
         len(train_pairs), n_ic, n_nic, n_nic / max(n_ic, 1))

# -- Step C: filter extreme segments (NEW -- under test) -----------------------
log.info("Scanning all %d segments for extreme amplitudes (threshold=%.1f)...",
         len(train_pairs), AMPLITUDE_THRESHOLD)
train_pairs = filter_extreme_segments(train_pairs, AMPLITUDE_THRESHOLD, log)

# -- Post-filter statistics ----------------------------------------------------
n_ic_after = sum(1 for _, l in train_pairs if l == 1)
n_nic_after = sum(1 for _, l in train_pairs if l == 0)
log.info("Post-filter corpus: %d total | %d ictal | %d non-ictal",
         len(train_pairs), n_ic_after, n_nic_after)

# -- Run ONE Optuna trial with fixed hyperparameters ---------------------------
# Use a fixed configuration so the debug run is reproducible and comparable
# to the original Trial 1 that produced loss=2e9.
log.info("-" * 65)
log.info("Running ONE trial with fixed hyperparameters:")
log.info("  num_layers=6, kernel_size=7, num_filters=32, dropout=0.35")
log.info("  lr=1.90e-04, wd=3.84e-05, batch_size=64")
log.info("-" * 65)

set_seed(SEED)
model = TCN(
    num_layers=6,
    num_filters=32,
    kernel_size=7,
    dropout=0.35
).to(DEVICE)

log.info("Model parameters: %s", "{:,}".format(count_parameters(model)))
rf = 2 * (2 ** 6 - 1) * (7 - 1) + 1
log.info("Receptive field: %d samples (%.3f s)", rf, rf / FS)

# Build loaders
train_loader = make_loader(train_pairs, batch_size=64, train=True, device=DEVICE)
val_loader   = make_loader(val_pairs, batch_size=64, train=False, device=DEVICE)
log.info("Train batches: %d | Val batches: %d", len(train_loader), len(val_loader))

# Build training components (matching run_training internals)
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

pw = torch.tensor([1.0], dtype=torch.float32).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
optimiser = AdamW(model.parameters(), lr=1.90e-04, weight_decay=3.84e-05)
scheduler = CosineAnnealingLR(optimiser, T_max=MAX_EPOCHS, eta_min=1.90e-04 * 0.01)

log.info("pos_weight: %.4f", pw.item())
log.info("Optimiser: AdamW (lr=1.90e-04, wd=3.84e-05)")
log.info("Scheduler: CosineAnnealingLR (T_max=%d)", MAX_EPOCHS)

# -- Training loop with FULL per-epoch logging (debug mode) --------------------
log.info("=" * 65)
log.info("TRAINING STARTED (debug: logging EVERY epoch)")
log.info("=" * 65)

best_val_f1 = 0.0
epochs_no_imp = 0
best_state = None

for epoch in range(1, MAX_EPOCHS + 1):
    t_start = time.time()

    train_loss = train_one_epoch(model, train_loader, optimiser, criterion, DEVICE)
    val_f1, y_true, y_pred = evaluate(model, val_loader, DEVICE)
    scheduler.step()

    epoch_time = time.time() - t_start

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        epochs_no_imp = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        epochs_no_imp += 1

    # Log EVERY epoch in debug mode (not every 10th)
    log.info("Epoch %3d/%d | loss=%.6f | val_f1=%.4f | best=%.4f | pat=%d/%d | %.1fs",
             epoch, MAX_EPOCHS, train_loss, val_f1,
             best_val_f1, epochs_no_imp, ES_PATIENCE, epoch_time)

    # Early check: if loss is still >1e6 after epoch 1, something is still wrong
    if epoch == 1 and train_loss > 1e6:
        log.error("LOSS STILL EXPLOSIVE after filtering (%.4e). "
                  "filter_extreme_segments did not fix the problem. "
                  "Check threshold or investigate further.", train_loss)
        log.error("Aborting debug run.")
        break

    if epochs_no_imp >= ES_PATIENCE:
        log.info("Early stopping at epoch %d.", epoch)
        break

# -- Results -------------------------------------------------------------------
log.info("=" * 65)
log.info("DEBUG TRIAL COMPLETE")
log.info("  Best val F1      : %.6f", best_val_f1)
log.info("  Final epoch      : %d", epoch)
log.info("  Early stopped    : %s", "Yes" if epochs_no_imp >= ES_PATIENCE else "No")
log.info("=" * 65)

# -- Diagnostic: check if loss was in normal range ----------------------------
if train_loss < 2.0:
    log.info("VERDICT: Loss is in normal range (%.4f). "
             "filter_extreme_segments() fixed the problem.", train_loss)
    log.info("NEXT: Integrate filter_extreme_segments() into tcn_utils.py "
             "and re-run tcn_HPT_binary.py with the full 60 trials.")
elif train_loss < 1e6:
    log.info("VERDICT: Loss is elevated but not explosive (%.4f). "
             "Consider lowering AMPLITUDE_THRESHOLD or adding clipping.", train_loss)
else:
    log.info("VERDICT: Loss is still explosive (%.4e). "
             "filter_extreme_segments() did not fix the problem. "
             "Investigate further.", train_loss)

# Save debug results
debug_results = {
    "timestamp": datetime.now().isoformat(),
    "amplitude_threshold": AMPLITUDE_THRESHOLD,
    "corpus_size_before_filter": n_ic + n_nic,
    "corpus_size_after_filter": len(train_pairs),
    "segments_removed": (n_ic + n_nic) - len(train_pairs),
    "ictal_after_filter": n_ic_after,
    "nonictal_after_filter": n_nic_after,
    "best_val_f1": round(best_val_f1, 6),
    "final_train_loss": round(float(train_loss), 6),
    "final_epoch": epoch,
    "early_stopped": epochs_no_imp >= ES_PATIENCE,
    "hyperparameters": {
        "num_layers": 6, "kernel_size": 7, "num_filters": 32,
        "dropout": 0.35, "learning_rate": 1.90e-04,
        "weight_decay": 3.84e-05, "batch_size": 64
    },
}
with open(DEBUG_DIR / "debug_results.json", "w", encoding="utf-8") as f:
    json.dump(debug_results, f, indent=2)
log.info("Saved: %s", DEBUG_DIR / "debug_results.json")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    log.info("GPU cache cleared.")

log.info("Debug run finished.")
