"""
m3_post_eval.py
===============
Standalone post-training evaluation for the M3 (MultiScaleTCN) model,
recovering the three-row report after the original training run aborted
during post-processing due to NaN values in y_prob.

The original full-validation forward pass crashed at roc_auc_score on
NaN-valued probabilities. The diagnostic scan
(scan_val_test_extreme.py) then established that:

  - Zero NaN/Inf values exist in the source .npy files.
  - 3,201 val segments contain extreme-but-finite amplitudes
    (worst |x| = 1.65 x 10^19, all label=0).
  - The FP16 ceiling is 65,504, so any input |x| > 65,504 overflows
    in the AMP forward pass to +inf, which then triggers NaN inside
    LayerNorm via inf - inf in the channel-wise reduction.

This script re-runs the full-validation pass and the entire post-
processing pipeline using the saved final weights, with FOUR layered
protections so the failure cannot recur:

  Layer 1 -- input filter:
    filter_extreme_segments(val_pairs, threshold=1000.0) is applied
    after load_splits() so that no segment with |x| > 1000 reaches
    the model.

  Layer 2 -- dataset hardening (defence-in-depth):
    SafeEEGSegmentDataset replaces NaN/Inf with 0.0 via np.nan_to_num
    and clips to +- 1000 via np.clip in __getitem__, even though
    Layer 1 should have already removed such segments. Tracks how
    many samples needed sanitisation for transparency.

  Layer 3 -- FP32 forward pass:
    Autocast is disabled (use_amp=False), giving 5+ orders of
    magnitude more headroom than FP16. Increases wall time by ~30 %
    compared to AMP but eliminates the FP16 overflow path entirely.

  Layer 4 -- finiteness assertion:
    After every forward pass we assert torch.isfinite(logits).all().
    If ANY logit is non-finite despite the first three layers, the
    script crashes with a localised diagnostic (batch index, sample
    indices, sample logit values) instead of writing a NaN-poisoned
    report to disk.

Inputs (paths inherited from MultiScaleTCN.py)
----------------------------------------------
  WEIGHTS_PATH         multiscale_tcn_final_weights.pt
  BACKBONE_PARAMS_PATH best_multiscale_params.json
  SPLITS_PATH          data_splits_nonictal_sampled.json

Outputs (under OUTPUT_ROOT = /scratch/.../MODEL3_OUTPUT/MultiScaleTCN/)
----------------------------------------------------------------------
  multiscale_tcn_evaluation_report.json   -- three-row metrics + meta
  multiscale_tcn_three_row_summary.csv    -- compact tabular form
  multiscale_tcn_optimal_threshold.json   -- Youden's-J threshold
  figures/                                -- 13 figures (ROC/PR/...)
  logs/m3_post_eval.log                   -- persistent log

Usage
-----
python m3_post_eval.py
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import datetime
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")                                  # non-interactive backend

from sklearn.metrics import f1_score

# Reuse the M3 training script's helpers and constants. Importing the module
# does NOT execute its main() because that's guarded by `if __name__`. All
# top-level constants and functions become importable as-is.
from MultiScaleTCN import (
    SEED, MODEL_NAME, OUTPUT_ROOT, LOG_DIR, FIGURE_DIR,
    WEIGHTS_PATH, EVAL_REPORT_PATH, THRESH_PATH, EPOCH_CSV, THREE_ROW_CSV,
    SPLITS_PATH,
    SEGMENT_LEN, SEGMENT_SEC, FS,
    MIN_EVENT_SEC, REFRACTORY_SEC, SMOOTHING_WIN,
    load_splits, load_best_params, build_model,
    run_postprocessing_evaluations, save_all_results, plot_all_figures,
)
from tcn_utils import (
    set_seed, count_parameters, filter_extreme_segments,
    EEGSegmentDataset,
)


# ---------------------------------------------------------------------------
# Constants (script-local)
# ---------------------------------------------------------------------------
EVAL_LOG_PATH       = LOG_DIR / "m3_post_eval.log"     # persistent .log
AMPLITUDE_THRESHOLD = 1000.0                           # matches train-side filter
USE_AMP_FOR_EVAL    = False                            # Layer 3: FP32 forward
NUM_DATA_WORKERS    = 4                                # DataLoader workers


# ---------------------------------------------------------------------------
# setup_logging  (console + persistent file handler)
# ---------------------------------------------------------------------------
def setup_logging():
    """Mirror the MultiScaleTCN.py log style; route to logs/m3_post_eval.log."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("m3_post_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(EVAL_LOG_PATH, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# SafeEEGSegmentDataset  (Layer 2: dataset hardening)
# ---------------------------------------------------------------------------
class SafeEEGSegmentDataset(EEGSegmentDataset):
    """Defensive subclass that sanitises every loaded segment.

    Even after Layer 1 (filter_extreme_segments), a stray bad file could
    in principle make it to the loader (e.g., race conditions, corrupt
    write between scan and eval). This class catches anything that
    slips through:

      - non-finite values  -> replaced by 0.0   via np.nan_to_num
      - amplitude > 1000   -> clipped to +-1000 via np.clip

    Maintains a class-level counter `n_sanitised` so the eval report
    can record how many samples needed defensive cleanup. If the counter
    is non-zero after a run, the upstream filter likely missed something.
    """

    n_sanitised = 0

    def __getitem__(self, idx):
        path, label = self.pairs[idx]
        # Same retry behaviour as the parent class (Lustre/GPFS resilience)
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                x = np.load(path).astype(np.float32)
                break
            except OSError as exc:
                if attempt < max_retries:
                    time.sleep(5)                       # transient I/O retry
                else:
                    raise OSError(
                        f"Failed to load {path} after {max_retries} retries: {exc}"
                    ) from exc

        # Defensive sanitisation
        if not np.isfinite(x).all():
            SafeEEGSegmentDataset.n_sanitised += 1
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if np.abs(x).max() > AMPLITUDE_THRESHOLD:
            SafeEEGSegmentDataset.n_sanitised += 1
            np.clip(x, -AMPLITUDE_THRESHOLD, AMPLITUDE_THRESHOLD, out=x)

        x = torch.from_numpy(x).unsqueeze(0)            # (1, segment_len)
        y = torch.tensor(label, dtype=torch.float32)
        return x, y


def make_safe_loader(file_label_pairs, batch_size, device,
                     num_workers=NUM_DATA_WORKERS):
    """Eval-only DataLoader using the hardened dataset (no shuffle)."""
    dataset = SafeEEGSegmentDataset(file_label_pairs)
    pin = (device.type == "cuda")
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )


# ---------------------------------------------------------------------------
# evaluate_model_safe  (Layer 3: FP32 + Layer 4: isfinite assert)
# ---------------------------------------------------------------------------
def evaluate_model_safe(model, loader, device, logger, use_amp=False):
    """FP32 forward pass with per-batch finiteness assertion.

    Mirrors MultiScaleTCN.evaluate_model in interface (returns the same
    4-tuple) but adds:
      - use_amp default False -> FP32 forward (Layer 3)
      - torch.isfinite(logits).all() assertion at every batch (Layer 4)
      - per-50-batch progress logging (long-running pass)

    If the Layer-4 assertion fires, the script raises with a localised
    diagnostic (batch index, sample indices, first 8 logit values) so we
    know precisely where to investigate. This is intentional fail-loud
    behaviour -- the alternative would be writing a NaN-corrupted report.
    """
    model.eval()
    n_batches = len(loader)
    log_every = max(1, n_batches // 50)                # ~50 progress lines

    all_true = []
    all_pred = []
    all_probs = []
    n_segments = 0
    t0 = time.time()

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)

            # Layer 4: hard assert. If this fires, all four protections failed.
            if not torch.isfinite(logits).all():
                bad_mask = ~torch.isfinite(logits)
                n_bad = int(bad_mask.sum().item())
                bad_idx = torch.nonzero(bad_mask).flatten().tolist()
                # Examine the corresponding inputs to localise the cause
                logger.error(
                    "NON-FINITE LOGIT in batch %d/%d: %d/%d samples bad. "
                    "First bad sample indices in batch: %s. Sample logits: %s",
                    batch_idx, n_batches, n_bad, logits.numel(),
                    bad_idx[:5],
                    logits.flatten()[:8].cpu().tolist())
                if bad_idx:
                    bad_input = x[bad_idx[0]].cpu().numpy()
                    logger.error(
                        "Bad sample-0 input stats: shape=%s | finite=%s | "
                        "min=%.4e | max=%.4e | abs_max=%.4e",
                        tuple(bad_input.shape), bool(np.isfinite(bad_input).all()),
                        float(bad_input.min()), float(bad_input.max()),
                        float(np.abs(bad_input).max()))
                raise AssertionError(
                    f"Forward pass produced non-finite logits in batch "
                    f"{batch_idx} despite four-layer protection (FP32="
                    f"{not use_amp}). Investigate before re-running."
                )

            # Sigmoid in same precision as logits; downstream metrics consume probs
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            all_true.append(y.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            n_segments += x.size(0)

            if (batch_idx + 1) % log_every == 0 or batch_idx == n_batches - 1:
                elapsed = time.time() - t0
                rate = n_segments / max(elapsed, 1e-3)
                eta = (n_batches - batch_idx - 1) * elapsed / max(batch_idx + 1, 1)
                logger.info(
                    "  eval %d/%d batches (%.0f%%) | %d segments | "
                    "%.0f seg/s | eta %.0f s",
                    batch_idx + 1, n_batches,
                    100 * (batch_idx + 1) / n_batches,
                    n_segments, rate, eta)

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_probs)

    # Belt-and-braces: assert the NumPy-side outputs are also finite. With
    # Layer 4 in place this should never fire, but it would catch any
    # downstream NaN introduced during torch->numpy conversion edge cases.
    assert np.isfinite(y_prob).all(), \
        "y_prob contains NaN/Inf despite the per-batch finiteness assert"

    val_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return val_f1, y_true, y_pred, y_prob


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    logger = setup_logging()

    logger.info("=" * 65)
    logger.info("m3_post_eval.py")
    logger.info("Timestamp       : %s", datetime.datetime.now().isoformat())
    logger.info("Purpose         : Post-training eval with 4-layer NaN protection")
    logger.info("Model           : %s", MODEL_NAME)
    logger.info("Weights         : %s", WEIGHTS_PATH)
    logger.info("Splits          : %s", SPLITS_PATH)
    logger.info("Output dir      : %s", OUTPUT_ROOT)
    logger.info("Eval log path   : %s", EVAL_LOG_PATH)
    logger.info("Layer 1 (filter): threshold=%.1f", AMPLITUDE_THRESHOLD)
    logger.info("Layer 2 (dataset): SafeEEGSegmentDataset (nan_to_num + clip)")
    logger.info("Layer 3 (FP32)  : use_amp=%s", USE_AMP_FOR_EVAL)
    logger.info("Layer 4 (assert): torch.isfinite(logits).all() per batch")
    logger.info("=" * 65)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device          : %s", device)
    if torch.cuda.is_available():
        logger.info("GPU             : %s", torch.cuda.get_device_name(0))
        logger.info("VRAM            : %.2f GB",
                    torch.cuda.get_device_properties(0).total_memory / 1e9)
        logger.info("CUDA            : %s", torch.version.cuda)
    logger.info("PyTorch         : %s", torch.__version__)

    # -- Step 1: Load M3 hyperparameters from JSON --------------------------
    logger.info("-" * 65)
    logger.info("Step 1: Load M3 hyperparameters")
    config, hp, branch_dilations = load_best_params(logger)

    # -- Step 2: Build model and load saved weights -------------------------
    logger.info("-" * 65)
    logger.info("Step 2: Build model and load saved weights")
    model = build_model(hp, branch_dilations, device, logger)
    if not WEIGHTS_PATH.exists():
        logger.error("Weights file not found: %s", WEIGHTS_PATH)
        sys.exit(1)
    state_dict = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state_dict)
    n_params = count_parameters(model)
    logger.info("Loaded weights : %d params from %s",
                n_params, WEIGHTS_PATH.name)

    # -- Step 3: Load val pairs ---------------------------------------------
    logger.info("-" * 65)
    logger.info("Step 3: Load val pairs from manifest")
    train_pairs, val_pairs = load_splits(logger)
    n_val_raw = len(val_pairs)
    logger.info("Raw val pairs   : %d", n_val_raw)

    # -- Step 4: Layer 1 -- filter extreme segments -------------------------
    logger.info("-" * 65)
    logger.info("Step 4: Layer 1 -- filter_extreme_segments(threshold=%.1f)",
                AMPLITUDE_THRESHOLD)
    t0 = time.time()
    val_pairs = filter_extreme_segments(
        val_pairs, threshold=AMPLITUDE_THRESHOLD, logger=logger)
    n_val_filtered = len(val_pairs)
    n_filter_removed = n_val_raw - n_val_filtered
    logger.info("Filtered val    : %d  (%d removed in %.1f s | %.4f%%)",
                n_val_filtered, n_filter_removed, time.time() - t0,
                100 * n_filter_removed / max(n_val_raw, 1))

    # -- Step 5: Layer 2 -- hardened DataLoader -----------------------------
    logger.info("-" * 65)
    logger.info("Step 5: Layer 2 -- SafeEEGSegmentDataset + DataLoader")
    batch_size = int(hp["batch_size"])
    val_loader = make_safe_loader(val_pairs, batch_size, device,
                                  num_workers=NUM_DATA_WORKERS)
    logger.info("Val loader      : %d batches | batch_size=%d | workers=%d",
                len(val_loader), batch_size, NUM_DATA_WORKERS)

    # -- Step 6: Layers 3 & 4 -- FP32 forward + per-batch isfinite ----------
    logger.info("-" * 65)
    logger.info("Step 6: Layers 3+4 -- FP32 forward pass with finiteness asserts")
    t0_eval = time.time()
    val_f1, y_true, y_pred_05, y_prob = evaluate_model_safe(
        model, val_loader, device, logger, use_amp=USE_AMP_FOR_EVAL)
    eval_seconds = time.time() - t0_eval
    logger.info("Eval pass done  : %.1f min  | val F1 (raw @ 0.5) = %.4f",
                eval_seconds / 60, val_f1)
    logger.info("Layer 2 caught  : %d segments needed in-loader sanitisation",
                SafeEEGSegmentDataset.n_sanitised)

    # -- Step 7: Three-row post-processing ----------------------------------
    logger.info("-" * 65)
    logger.info("Step 7: Run three-row post-processing evaluation")
    (row1_metrics, row2_metrics, row3_metrics,
     post_row2, post_row3, far_row2, far_row3,
     thresh_result, optimal_threshold) = run_postprocessing_evaluations(
         y_true, y_prob, logger)

    # -- Step 8: Save report and figures ------------------------------------
    logger.info("-" * 65)
    logger.info("Step 8: Save evaluation report, three-row CSV, and figures")
    y_pred_row1 = (y_prob >= 0.5).astype(int)
    y_pred_row2 = post_row2["smoothed_preds"]
    y_pred_row3 = post_row3["smoothed_preds"]

    # The original training run's per-epoch history was lost when the post-
    # processing crashed. The SLURM .out file documents the trajectory; here
    # we record only the best-epoch summary so save_all_results / plotting
    # have a valid history dict.
    history = {
        "epoch":      [8],
        "train_loss": [0.1871],
        "val_f1":     [0.7923],
        "lr":         [2.87e-4],
    }
    elapsed_dt = datetime.timedelta(seconds=int(eval_seconds))

    save_all_results(
        history, row1_metrics, row2_metrics, row3_metrics,
        far_row2, far_row3, hp, branch_dilations,
        8, 0.7923, elapsed_dt, device, n_params, y_true,
        y_pred_row1, y_pred_row2, y_pred_row3, logger)

    plot_all_figures(
        history, 8, 0.7923, y_true, y_prob,
        y_pred_row1, y_pred_row2, y_pred_row3,
        row1_metrics, row2_metrics, row3_metrics,
        post_row2, post_row3, thresh_result, optimal_threshold, logger)

    # -- Final summary ------------------------------------------------------
    logger.info("=" * 65)
    logger.info("M3 POST-EVAL COMPLETE")
    logger.info("  Val pairs (raw)        : %d", n_val_raw)
    logger.info("  Val pairs (post-filter): %d", n_val_filtered)
    logger.info("  Layer 1 removed        : %d (%.4f%%)",
                n_filter_removed, 100 * n_filter_removed / max(n_val_raw, 1))
    logger.info("  Layer 2 sanitised      : %d (in-loader, post-filter)",
                SafeEEGSegmentDataset.n_sanitised)
    logger.info("  Layer 3 (FP32 eval)    : enabled (use_amp=%s)", USE_AMP_FOR_EVAL)
    logger.info("  Layer 4 (assert)       : never fired => all logits finite")
    logger.info("  Wall time              : %.1f min", eval_seconds / 60)
    logger.info("  Row 1 F1 (raw @ 0.5)   : %.4f",
                row1_metrics.get("f1_macro", float("nan")))
    logger.info("  Row 2 F1 (post @ 0.5)  : %.4f",
                row2_metrics.get("f1_macro", float("nan")))
    logger.info("  Row 3 F1 (post @ tau*) : %.4f",
                row3_metrics.get("f1_macro", float("nan")))
    logger.info("  Optimal threshold tau* : %.4f", optimal_threshold)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
