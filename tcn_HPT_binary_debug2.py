"""
tcn_HPT_binary_debug2.py
=========================
Debug script #2: same as debug1 but with:
  - AMPLITUDE_THRESHOLD = 100 (catches mild outliers too)
  - Inline training loop (mirrors run_training exactly)
    with tensor assertions injected before each forward pass
  - try/except guard around model(x) to catch cuDNN/CUDA
    backend failures without crashing
  - Non-finite check in filter_extreme_segments

Tuning protocol: MAX_EPOCHS=20, ES_PATIENCE=5 (one trial only).

Purpose
-------
Determine whether lowering the threshold from 1000 to 100 removes
the mild outlier segments (100-400 range) and whether the remaining
corpus trains cleanly. The inline assertions and forward-pass guard
catch any NaN/inf, shape mismatch, or backend failure at the tensor
level, providing diagnostic evidence if the loss still explodes.

Usage
-----
python tcn_HPT_binary_debug2.py
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
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib
matplotlib.use('Agg')

from tcn_utils import (
    set_seed,
    make_loader,
    filter_unpaired_subjects,
    downsample_non_ictal,
    TCN,
    count_parameters,
    evaluate,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED              = 42
FS                = 500
SEGMENT_LEN       = 2500
MAX_EPOCHS        = 20                                 # max epochs per trial (ES fires before 20)
ES_PATIENCE       = 5                                  # early stopping patience (epochs)
MAX_GRAD_NORM     = 1.0

# Lower threshold: removes both catastrophic (10^15+) and mild outliers (100-400)
AMPLITUDE_THRESHOLD = 100.0

SPLITS_PATH = Path("/scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits.json")
OUTPUT_DIR  = Path("/home/people/22206468/scratch/OUTPUT/MODEL1_OUTPUT/TCNtuning_outputs")
DEBUG_DIR   = OUTPUT_DIR / "debug2"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("tcn_hpt_debug2")
log.setLevel(logging.DEBUG)
log.handlers.clear()

fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

fh = logging.FileHandler(DEBUG_DIR / "debug2_trial.log", mode="w", encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(fmt)

sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.INFO)
sh.setFormatter(fmt)

log.addHandler(fh)
log.addHandler(sh)


# ---------------------------------------------------------------------------
# filter_extreme_segments -- hardened with non-finite check
# ---------------------------------------------------------------------------
def filter_extreme_segments(pairs, threshold, logger):
    """Remove segments whose max absolute amplitude exceeds threshold
    or that contain non-finite (NaN/inf) values.

    Parameters
    ----------
    pairs : list of (str or Path, int) tuples
    threshold : float
    logger : logging.Logger

    Returns
    -------
    list of (str or Path, int) tuples
    """
    t0 = time.time()
    clean = []
    removed = []
    n_nonfinite = 0
    worst_val = 0.0
    worst_file = ""

    for i, (fp, label) in enumerate(pairs):
        x = np.load(fp)

        # Check for non-finite values first
        if not np.isfinite(x).all():
            removed.append((fp, label, float("inf")))
            n_nonfinite += 1
            if (i + 1) % 50000 == 0:
                logger.info("  filter_extreme_segments: scanned %d/%d | removed so far: %d",
                            i + 1, len(pairs), len(removed))
            continue

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

    logger.info("=" * 60)
    logger.info("filter_extreme_segments summary:")
    logger.info("  Threshold          : %.1f", threshold)
    logger.info("  Total scanned      : %d", len(pairs))
    logger.info("  Removed            : %d (%.2f%%)",
                len(removed), 100 * len(removed) / max(len(pairs), 1))
    logger.info("  Retained           : %d", len(clean))
    logger.info("  Scan time          : %.1f s", elapsed)
    logger.info("  Non-finite removed : %d", n_nonfinite)

    if removed:
        if worst_val > 0:
            logger.info("  Worst finite value : %.4e in %s", worst_val, worst_file)

        subject_counts = {}
        for fp, label, mx in removed:
            subj = Path(fp).stem.split("_", 1)[0]
            if subj not in subject_counts:
                subject_counts[subj] = {"count": 0, "worst": 0.0}
            subject_counts[subj]["count"] += 1
            if mx != float("inf"):
                subject_counts[subj]["worst"] = max(subject_counts[subj]["worst"], mx)

        logger.info("  Affected subjects  : %d", len(subject_counts))
        for subj in sorted(subject_counts):
            info = subject_counts[subj]
            logger.info("    %s: %d segments removed (worst=%.2e)",
                        subj, info["count"], info["worst"])

        # Breakdown by severity -- only over finite removed segments
        finite_removed = [(fp, l, mx) for fp, l, mx in removed if mx != float("inf")]
        catastrophic = sum(1 for _, _, mx in finite_removed if mx > 1e6)
        mild = sum(1 for _, _, mx in finite_removed if mx <= 1e6)
        logger.info("  Non-finite         : %d", n_nonfinite)
        logger.info("  Catastrophic (>1e6): %d", catastrophic)
        logger.info("  Mild (%.0f-1e6)     : %d", threshold, mild)

    logger.info("=" * 60)

    removed_log = [{"filepath": fp, "label": label,
                    "max_abs": mx if mx != float("inf") else "non-finite"}
                   for fp, label, mx in removed]
    with open(DEBUG_DIR / "removed_segments_debug2.json", "w", encoding="utf-8") as f:
        json.dump(removed_log, f, indent=2)
    logger.info("Saved: %s (%d entries)", DEBUG_DIR / "removed_segments_debug2.json", len(removed_log))

    return clean


# ---------------------------------------------------------------------------
# Main debug run
# ---------------------------------------------------------------------------
log.info("=" * 65)
log.info("tcn_HPT_binary_debug2.py -- ONE-TRIAL DEBUG RUN #2")
log.info("Timestamp: %s", datetime.now().isoformat())
log.info("Purpose: threshold=100 + inline loop with assertions + forward-pass guard")
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

# -- Step C: filter extreme segments (threshold=100) ---------------------------
log.info("Scanning all %d segments for extreme amplitudes (threshold=%.1f)...",
         len(train_pairs), AMPLITUDE_THRESHOLD)
train_pairs = filter_extreme_segments(train_pairs, AMPLITUDE_THRESHOLD, log)

# -- Post-filter statistics ----------------------------------------------------
n_ic_after = sum(1 for _, l in train_pairs if l == 1)
n_nic_after = sum(1 for _, l in train_pairs if l == 0)
log.info("Post-filter corpus: %d total | %d ictal | %d non-ictal",
         len(train_pairs), n_ic_after, n_nic_after)

# -- Build model ---------------------------------------------------------------
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

# -- Build loaders -------------------------------------------------------------
train_loader = make_loader(train_pairs, batch_size=64, train=True, device=DEVICE)
val_loader   = make_loader(val_pairs, batch_size=64, train=False, device=DEVICE)
log.info("Train batches: %d | Val batches: %d", len(train_loader), len(val_loader))

# -- Build training components (mirrors run_training exactly) ------------------
pw = torch.tensor([1.0], dtype=torch.float32).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
optimiser = AdamW(model.parameters(), lr=1.90e-04, weight_decay=3.84e-05)
scheduler = CosineAnnealingLR(optimiser, T_max=MAX_EPOCHS, eta_min=1.90e-04 * 0.01)

log.info("pos_weight: %.4f", pw.item())
log.info("Optimiser: AdamW (lr=1.90e-04, wd=3.84e-05)")
log.info("Scheduler: CosineAnnealingLR (T_max=%d)", MAX_EPOCHS)

# -- Inline training loop (mirrors run_training with assertions + guard) -------
log.info("=" * 65)
log.info("TRAINING STARTED (debug2: every epoch + assertions + forward guard)")
log.info("=" * 65)

best_val_f1 = 0.0
epochs_no_improve = 0
best_state = None
assertion_failures = 0
forward_failures = 0

for epoch in range(1, MAX_EPOCHS + 1):
    t_epoch = time.time()

    # ---- TRAIN ONE EPOCH (mirrors train_one_epoch with assertions) -----------
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (x_raw, y_raw) in enumerate(train_loader):

        # -- Explicit, deterministic tensor transfer ---------------------------
        x = x_raw.to(device=DEVICE, dtype=torch.float32, non_blocking=True).contiguous()
        y = y_raw.to(device=DEVICE, dtype=torch.float32, non_blocking=True)

        # -- Finite-value assertions on inputs ---------------------------------
        if not torch.isfinite(x).all():
            n_bad = (~torch.isfinite(x)).sum().item()
            log.error("ASSERTION FAIL: x has %d non-finite values | "
                      "epoch=%d batch=%d shape=%s dtype=%s device=%s "
                      "min=N/A max=N/A",
                      n_bad, epoch, batch_idx,
                      tuple(x.shape), x.dtype, x.device)
            assertion_failures += 1
            continue

        if not torch.isfinite(y).all():
            n_bad = (~torch.isfinite(y)).sum().item()
            log.error("ASSERTION FAIL: y has %d non-finite values | "
                      "epoch=%d batch=%d shape=%s dtype=%s device=%s",
                      n_bad, epoch, batch_idx,
                      tuple(y.shape), y.dtype, y.device)
            assertion_failures += 1
            continue

        # -- Forward pass with try/except guard --------------------------------
        optimiser.zero_grad(set_to_none=True)

        try:
            logits = model(x)
        except Exception:
            forward_failures += 1
            # Gather model parameter info for diagnostics
            try:
                p0 = next(model.parameters())
                p_dtype = p0.dtype
                p_device = p0.device
            except StopIteration:
                p_dtype = "unknown"
                p_device = "unknown"
            log.exception(
                "FORWARD PASS FAILED | epoch=%d batch=%d "
                "x.shape=%s x.dtype=%s x.device=%s x.contiguous=%s "
                "x.isfinite=%s x.min=%.4e x.max=%.4e x.abs_max=%.4e "
                "model_param_dtype=%s model_param_device=%s",
                epoch, batch_idx,
                tuple(x.shape), x.dtype, x.device, x.is_contiguous(),
                torch.isfinite(x).all().item(),
                x.min().item(), x.max().item(), x.abs().max().item(),
                p_dtype, p_device)
            assertion_failures += 1
            continue

        # -- Shape assertion ---------------------------------------------------
        if logits.shape != y.shape:
            log.error("ASSERTION FAIL: logits.shape=%s != y.shape=%s | "
                      "epoch=%d batch=%d",
                      tuple(logits.shape), tuple(y.shape),
                      epoch, batch_idx)
            assertion_failures += 1
            continue

        # -- Finite-value assertion on logits ----------------------------------
        if not torch.isfinite(logits).all():
            n_bad = (~torch.isfinite(logits)).sum().item()
            log.error("ASSERTION FAIL: logits has %d non-finite values | "
                      "epoch=%d batch=%d shape=%s "
                      "min=%.4e max=%.4e",
                      n_bad, epoch, batch_idx,
                      tuple(logits.shape),
                      logits.min().item(), logits.max().item())
            assertion_failures += 1
            continue

        # -- Loss, backward, clip, step (mirrors train_one_epoch exactly) ------
        loss = criterion(logits, y)

        if not torch.isfinite(loss):
            log.error("ASSERTION FAIL: loss is non-finite (%.4e) | "
                      "epoch=%d batch=%d "
                      "logits: min=%.4e max=%.4e | "
                      "y: min=%.1f max=%.1f",
                      loss.item(), epoch, batch_idx,
                      logits.min().item(), logits.max().item(),
                      y.min().item(), y.max().item())
            assertion_failures += 1
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimiser.step()
        total_loss += loss.item()
        n_batches += 1

        # Log first batch of first epoch with full diagnostics
        if epoch == 1 and batch_idx == 0:
            log.info("  [batch 0] loss=%.6f | "
                     "logits: shape=%s min=%.4f max=%.4f mean=%.4f | "
                     "x: shape=%s dtype=%s device=%s contiguous=%s min=%.4f max=%.4f | "
                     "y: shape=%s min=%.1f max=%.1f",
                     loss.item(),
                     tuple(logits.shape),
                     logits.min().item(), logits.max().item(), logits.mean().item(),
                     tuple(x.shape), x.dtype, x.device, x.is_contiguous(),
                     x.min().item(), x.max().item(),
                     tuple(y.shape),
                     y.min().item(), y.max().item())

    train_loss = total_loss / max(n_batches, 1)

    # ---- EVALUATE (uses tcn_utils.evaluate as normal) ------------------------
    val_f1, _, _ = evaluate(model, val_loader, DEVICE)

    # ---- SCHEDULER STEP (mirrors run_training) -------------------------------
    scheduler.step()

    # ---- EARLY STOPPING LOGIC (mirrors run_training) -------------------------
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        epochs_no_improve = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        epochs_no_improve += 1

    epoch_time = time.time() - t_epoch

    # Log every epoch (debug mode)
    log.info("Epoch %3d/%d | loss=%.6f | val_f1=%.4f | best=%.4f | pat=%d/%d | "
             "assert_fail=%d fwd_fail=%d | %.1fs",
             epoch, MAX_EPOCHS, train_loss, val_f1,
             best_val_f1, epochs_no_improve, ES_PATIENCE,
             assertion_failures, forward_failures, epoch_time)

    # Early abort if loss is still explosive after epoch 1
    if epoch == 1 and train_loss > 1e6:
        log.error("LOSS STILL EXPLOSIVE (%.4e) after filtering at threshold=%.1f. Aborting.",
                  train_loss, AMPLITUDE_THRESHOLD)
        break

    if epochs_no_improve >= ES_PATIENCE:
        log.info("Early stopping at epoch %d.", epoch)
        break

# -- Restore best weights (mirrors run_training) ------------------------------
if best_state is not None:
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

# -- Results -------------------------------------------------------------------
log.info("=" * 65)
log.info("DEBUG2 TRIAL COMPLETE")
log.info("  Best val F1          : %.6f", best_val_f1)
log.info("  Final epoch          : %d", epoch)
log.info("  Final train loss     : %.6f", train_loss)
log.info("  Early stopped        : %s", "Yes" if epochs_no_improve >= ES_PATIENCE else "No")
log.info("  Assertion failures   : %d", assertion_failures)
log.info("  Forward-pass failures: %d", forward_failures)
log.info("  Amplitude threshold  : %.1f", AMPLITUDE_THRESHOLD)
log.info("  Segments removed     : %d", (n_ic + n_nic) - len(train_pairs))
log.info("=" * 65)

# Verdict
if train_loss < 2.0 and assertion_failures == 0:
    log.info("VERDICT: Loss normal (%.4f), zero assertion failures. "
             "filter_extreme_segments(threshold=100) fixed the problem.", train_loss)
elif train_loss < 2.0 and assertion_failures > 0:
    log.info("VERDICT: Loss normal (%.4f) but %d assertion failures "
             "(%d forward-pass). Investigate the failed batches.",
             train_loss, assertion_failures, forward_failures)
elif train_loss >= 2.0 and assertion_failures > 0:
    log.info("VERDICT: Loss elevated (%.4f) with %d assertion failures "
             "(%d forward-pass). Data quality or backend issues remain.",
             train_loss, assertion_failures, forward_failures)
else:
    log.info("VERDICT: Loss elevated (%.4f) but zero assertion failures. "
             "The issue is not data quality -- investigate model or optimiser.",
             train_loss)

# Save results
debug_results = {
    "timestamp": datetime.now().isoformat(),
    "script": "tcn_HPT_binary_debug2.py",
    "amplitude_threshold": AMPLITUDE_THRESHOLD,
    "corpus_before_filter": n_ic + n_nic,
    "corpus_after_filter": len(train_pairs),
    "segments_removed": (n_ic + n_nic) - len(train_pairs),
    "ictal_after": n_ic_after,
    "nonictal_after": n_nic_after,
    "best_val_f1": round(best_val_f1, 6),
    "final_train_loss": round(float(train_loss), 6),
    "final_epoch": epoch,
    "early_stopped": epochs_no_improve >= ES_PATIENCE,
    "assertion_failures": assertion_failures,
    "forward_failures": forward_failures,
    "hyperparameters": {
        "num_layers": 6, "kernel_size": 7, "num_filters": 32,
        "dropout": 0.35, "learning_rate": 1.90e-04,
        "weight_decay": 3.84e-05, "batch_size": 64
    },
}
with open(DEBUG_DIR / "debug2_results.json", "w", encoding="utf-8") as f:
    json.dump(debug_results, f, indent=2)
log.info("Saved: %s", DEBUG_DIR / "debug2_results.json")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    log.info("GPU cache cleared.")

log.info("Debug2 run finished.")
