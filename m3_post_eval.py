"""
m3_post_eval.py
===============
Post-training evaluation for the M3 (MultiScaleTCN) model that operates
entirely on the cached predictions saved by an earlier weights-loaded
run -- no model weights are reloaded and no inference is re-executed.

This script reads y_true and y_prob from
    OUTPUT_ROOT / "multiscale_tcn_predictions_raw.npz"
recomputes the post-processing rows, and writes the resulting figures,
JSONs, and CSVs. Because the >5-hour FP32 forward pass is replaced by a
~1-second .npz load, the script can run on CPU without the
multiscale_tcn_final_weights.pt file.

What changed (relative to the earlier inference-based design)
-------------------------------------------------------------
1. Threshold optimisation is disabled. Row 3 (post-processed at the
   F1-optimal threshold) consistently produced poor recall on this
   dataset's ~0.27 % ictal prevalence and is no longer reported.
   Only Row 1 (raw t = 0.5) and Row 2 (post-processed t = 0.5)
   remain. The threshold/Row-3 code is left in place as commented
   scaffolding for future re-enablement.
2. The four-layer NaN-protection workflow (manifest verification,
   SafeEEGSegmentDataset, FP32 forward, per-batch finiteness assert)
   has been retired in this pass. The cached predictions already
   encode the result of that protected pass from the original M3
   training run. Restore the commented-out Steps 2-6 in main() if a
   fresh inference run is required.
3. A new bar plot summarising the Row 1 / Row 2 macro-average
   classification report is written to
       OUTPUT_ROOT / "Result_classReport"
   alongside the rest of the evaluation outputs.

Inputs
------
  PREDICTIONS_RAW_NPZ   multiscale_tcn_predictions_raw.npz
                        (must already exist; produced by a prior
                         weights-loaded run of this script)
  best_multiscale_params.json
                        (architecture metadata for save_all_results)

Outputs (under OUTPUT_ROOT = /scratch/.../MODEL3_OUTPUT/MultiScaleTCN/)
----------------------------------------------------------------------
  multiscale_tcn_evaluation_report.json    -- Row 1 + Row 2 metrics + meta
  multiscale_tcn_three_row_summary.csv     -- two-row tabular form
  multiscale_tcn_classification_report_row{1,2}.json
  multiscale_tcn_event_details_row2.csv
  Result_classReport/multiscale_tcn_classreport_barplot.png
  figures/                                 -- Row 1 / Row 2 figures
  logs/m3_post_eval.log                    -- persistent log

Usage
-----
python m3_post_eval.py     # CPU is sufficient; no GPU required
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import csv
import datetime
import json
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
    SEED, MODEL_NAME, OUTPUT_ROOT,
    WEIGHTS_PATH,
    SEGMENT_LEN, SEGMENT_SEC, FS,
    MIN_EVENT_SEC, REFRACTORY_SEC, SMOOTHING_WIN,
    load_splits, load_best_params, build_model,
    run_postprocessing_evaluations, save_all_results, plot_all_figures,
)
from tcn_utils import (
    set_seed, count_parameters, EEGSegmentDataset,
    make_classreport_barplot,
)


# ---------------------------------------------------------------------------
# Constants (script-local)
# ---------------------------------------------------------------------------
AMPLITUDE_THRESHOLD = 1000.0                           # matches train-side filter
USE_AMP_FOR_EVAL    = False                            # Layer 3: FP32 forward
NUM_DATA_WORKERS    = 4                                # DataLoader workers

# Cached-prediction path: m3_post_eval.py loads y_true and y_prob from this
# file produced by a prior weights-based run, eliminating the need to
# reload the model and re-run the >5-hour FP32 forward pass. Read from the
# original training-output directory, never overwritten.
PREDICTIONS_RAW_NPZ = OUTPUT_ROOT / "multiscale_tcn_predictions_raw.npz"

# Per-epoch history sources written by MultiScaleTCN.py at training time.
# Read from the original training-output directory, never overwritten.
TRAINING_LOG_JSON_PATH = OUTPUT_ROOT / "multiscale_tcn_training_log.json"
TRAINING_EPOCH_CSV_PATH = OUTPUT_ROOT / "multiscale_tcn_epoch_metrics.csv"

# Dedicated output folder for everything m3_post_eval.py writes. By design
# this script never overwrites artefacts the training script produced --
# all of its outputs (figures, JSONs, CSVs, logs, bar plot) land here.
POST_EVAL_DIR          = OUTPUT_ROOT / "post_eval_result"
POST_EVAL_FIGURE_DIR   = POST_EVAL_DIR / "figures"
POST_EVAL_LOG_DIR      = POST_EVAL_DIR / "logs"
RESULT_CLASSREPORT_DIR = POST_EVAL_DIR / "Result_classReport"
EVAL_LOG_PATH          = POST_EVAL_LOG_DIR / "m3_post_eval.log"   # persistent .log


# ---------------------------------------------------------------------------
# _redirect_outputs_to_post_eval
# ---------------------------------------------------------------------------
def _redirect_outputs_to_post_eval(logger):
    """Rebind the path globals on the imported MultiScaleTCN module so
    save_all_results() and plot_all_figures() write into POST_EVAL_DIR.

    Why this works: those helpers were defined in MultiScaleTCN.py and
    look up names like OUTPUT_ROOT, FIGURE_DIR, EVAL_REPORT_PATH at call
    time in *MultiScaleTCN's* module namespace -- not in the caller's. By
    rebinding those names on the MultiScaleTCN module before invoking
    the helpers from this script, every write target lands under
    post_eval_result/ for this process only. A separate Python process
    running MultiScaleTCN.py for training is not affected (it never
    imports m3_post_eval and never sees the rebinding).

    The training-script directories (the original OUTPUT_ROOT, FIGURE_DIR,
    LOG_DIR) are read for inputs (predictions_raw.npz, training history)
    but never overwritten by this script.
    """
    import MultiScaleTCN as _m3

    redirections = {
        "OUTPUT_ROOT":      POST_EVAL_DIR,
        "FIGURE_DIR":       POST_EVAL_FIGURE_DIR,
        "LOG_DIR":          POST_EVAL_LOG_DIR,
        "TRAIN_LOG_PATH":   POST_EVAL_DIR / "multiscale_tcn_training_log.json",
        "EVAL_REPORT_PATH": POST_EVAL_DIR / "multiscale_tcn_evaluation_report.json",
        "EPOCH_CSV":        POST_EVAL_DIR / "multiscale_tcn_epoch_metrics.csv",
        "THREE_ROW_CSV":    POST_EVAL_DIR / "multiscale_tcn_three_row_summary.csv",
    }
    logger.info("Redirecting MultiScaleTCN write targets to POST_EVAL_DIR:")
    for name, new_path in redirections.items():
        original = getattr(_m3, name, None)
        setattr(_m3, name, new_path)
        logger.info("  %-18s : %s -> %s", name, original, new_path)


# ---------------------------------------------------------------------------
# _load_history
# ---------------------------------------------------------------------------
def _load_history(logger):
    """Load per-epoch training history for the EMA-smoothed training-curves
    figure.

    Tries, in order:
      1. multiscale_tcn_training_log.json -- preferred; full schema.
      2. multiscale_tcn_epoch_metrics.csv -- fallback if JSON is missing.
      3. Single-point stub -- last resort, matches the legacy behaviour
         from when the original training run aborted before either
         artefact was written.

    Returns a dict with keys epoch, train_loss, val_f1, lr (lists).
    """
    if TRAINING_LOG_JSON_PATH.exists():
        with open(TRAINING_LOG_JSON_PATH, "r", encoding="utf-8") as f:
            log = json.load(f)
        history = log.get("history", {})
        if history.get("epoch"):
            logger.info("Loaded per-epoch history from %s (%d epochs)",
                        TRAINING_LOG_JSON_PATH.name, len(history["epoch"]))
            return history

    if TRAINING_EPOCH_CSV_PATH.exists():
        history = {"epoch": [], "train_loss": [], "val_f1": [], "lr": []}
        with open(TRAINING_EPOCH_CSV_PATH, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                history["epoch"].append(int(row["epoch"]))
                history["train_loss"].append(float(row["train_loss"]))
                history["val_f1"].append(float(row["val_f1"]))
                history["lr"].append(float(row["lr"]))
        if history["epoch"]:
            logger.info("Loaded per-epoch history from %s (%d epochs)",
                        TRAINING_EPOCH_CSV_PATH.name, len(history["epoch"]))
            return history

    logger.warning(
        "No per-epoch history artefact found. Tried %s and %s. Falling back "
        "to single best-epoch stub -- training curves will collapse to one "
        "point. Re-run MultiScaleTCN.py to produce a full history.",
        TRAINING_LOG_JSON_PATH.name, TRAINING_EPOCH_CSV_PATH.name)
    return {
        "epoch":      [8],
        "train_loss": [0.1871],
        "val_f1":     [0.7923],
        "lr":         [2.87e-4],
    }


# ---------------------------------------------------------------------------
# setup_logging  (console + persistent file handler)
# ---------------------------------------------------------------------------
def setup_logging():
    """Mirror the MultiScaleTCN.py log style; route to POST_EVAL_DIR/logs/.

    Creates the full POST_EVAL_DIR subtree on first use:
        post_eval_result/
        post_eval_result/figures/
        post_eval_result/logs/
        post_eval_result/Result_classReport/
    The training-script directories (LOG_DIR, FIGURE_DIR) are not touched.
    """
    POST_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    POST_EVAL_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    POST_EVAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_CLASSREPORT_DIR.mkdir(parents=True, exist_ok=True)

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


def verify_manifest_filtered(splits_path, logger):
    """Verify the manifest's meta block records an apply_val_test_filter step.

    Layer 1 protection lives at the manifest level: apply_val_test_filter.py
    is the single upstream pipeline step that scrubs the val and test
    partitions of segments with NaN/Inf values or |x| > 1000. This function
    reads the manifest's meta.filter_history block and confirms that step
    was run, returning the recorded statistics for downstream logging and
    npz audit trail.

    Fails loudly (sys.exit(1)) if the entry is missing -- this prevents
    silently running on an unfiltered manifest, which would re-introduce
    the FP16 overflow / NaN failure mode the four-layer protection was
    designed to eliminate.

    Returns
    -------
    dict
        Keys: 'step', 'timestamp', 'threshold', 'val_after', 'val_removed'.
        Values come from the matching entry in meta.filter_history.
    """
    with open(splits_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    meta = manifest.get("meta", {}) or {}
    filter_history = meta.get("filter_history", []) or []

    apply_step = None
    for step in filter_history:
        if step.get("step") == "apply_val_test_filter":
            apply_step = step
            break

    if apply_step is None:
        logger.error(
            "Layer 1 verification FAILED: manifest %s does not record an "
            "'apply_val_test_filter' step in meta.filter_history. Run "
            "apply_val_test_filter.py first to produce a properly filtered "
            "manifest, or point SPLITS_PATH at a manifest that records this "
            "filtering step.", splits_path)
        sys.exit(1)

    val_block = apply_step.get("val") or {}
    info = {
        "step": apply_step.get("step", "apply_val_test_filter"),
        "timestamp": apply_step.get("timestamp", "unknown"),
        "threshold": float(apply_step.get("threshold", AMPLITUDE_THRESHOLD)),
        "val_after": int(val_block.get("after", 0)),
        "val_removed": int(val_block.get("removed", 0)),
    }
    logger.info(
        "Layer 1 verification PASSED: manifest filtered by %s at %s "
        "(threshold=%.1f). Val: %d retained, %d removed.",
        info["step"], info["timestamp"], info["threshold"],
        info["val_after"], info["val_removed"])
    return info


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
    logger.info("Purpose         : Post-training eval from cached predictions")
    logger.info("Model           : %s", MODEL_NAME)
    logger.info("Cached preds    : %s", PREDICTIONS_RAW_NPZ)
    logger.info("Read root       : %s (training-script artefacts; never overwritten)", OUTPUT_ROOT)
    logger.info("Write root      : %s (everything this script emits lands here)", POST_EVAL_DIR)
    logger.info("Eval log path   : %s", EVAL_LOG_PATH)
    logger.info("Mode            : Row 1 + Row 2 (Row 3 retired); no model "
                "weights loaded")
    logger.info("=" * 65)

    # Rebind path globals on the imported MultiScaleTCN module BEFORE any
    # call to save_all_results / plot_all_figures, so every artefact those
    # helpers write lands inside POST_EVAL_DIR. Reads (predictions_raw.npz,
    # training history) still happen against the original OUTPUT_ROOT.
    _redirect_outputs_to_post_eval(logger)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device          : %s (CPU is sufficient -- no inference run)", device)
    logger.info("PyTorch         : %s", torch.__version__)

    # -- Step 1: Load M3 hyperparameters from JSON --------------------------
    # Only used for branch_dilations / batch_size logging; no weights loaded.
    logger.info("-" * 65)
    logger.info("Step 1: Load M3 hyperparameters (architecture metadata only)")
    config, hp, branch_dilations = load_best_params(logger)
    n_params = 0  # placeholder; actual count was logged by the original training run

    # -- (Retired) Steps 2-6: model load + manifest verify + FP32 forward ---
    # The four-layer protection workflow (build_model -> load_state_dict ->
    # SafeEEGSegmentDataset -> evaluate_model_safe) has been retired in this
    # post-hoc pass because the cached predictions in PREDICTIONS_RAW_NPZ
    # already encode the result of that protected forward pass from the
    # original M3 training run. To re-enable a fresh inference pass (e.g.,
    # after retraining), restore the block below.
    #
    # if not WEIGHTS_PATH.exists():
    #     logger.error("Weights file not found: %s", WEIGHTS_PATH)
    #     sys.exit(1)
    # model = build_model(hp, branch_dilations, device, logger)
    # state_dict = torch.load(WEIGHTS_PATH, map_location=device)
    # model.load_state_dict(state_dict)
    # n_params = count_parameters(model)
    # train_pairs, val_pairs = load_splits(logger)
    # manifest_filter = verify_manifest_filtered(SPLITS_PATH, logger)
    # batch_size = int(hp["batch_size"])
    # val_loader = make_safe_loader(val_pairs, batch_size, device,
    #                               num_workers=NUM_DATA_WORKERS)
    # t0_eval = time.time()
    # val_f1, y_true, y_pred_05, y_prob = evaluate_model_safe(
    #     model, val_loader, device, logger, use_amp=USE_AMP_FOR_EVAL)
    # eval_seconds = time.time() - t0_eval
    # raw_npz_path = OUTPUT_ROOT / "multiscale_tcn_predictions_raw.npz"
    # np.savez_compressed(raw_npz_path, y_true=y_true.astype(np.int8),
    #                     y_prob=y_prob.astype(np.float32), ...)

    # -- Step 2 (revised): Load cached predictions from the prior run -------
    logger.info("-" * 65)
    logger.info("Step 2: Load cached y_true, y_prob from predictions_raw.npz")
    if not PREDICTIONS_RAW_NPZ.exists():
        logger.error(
            "Cached predictions file not found: %s\n"
            "Run a weights-loaded m3_post_eval.py pass first (see retired "
            "Steps 2-6 in the source) to produce this file, or restore the "
            "inference block to recompute predictions in-place.",
            PREDICTIONS_RAW_NPZ)
        sys.exit(1)
    t0_load = time.time()
    cached = np.load(PREDICTIONS_RAW_NPZ, allow_pickle=False)
    y_true = cached["y_true"].astype(np.int64)
    y_prob = cached["y_prob"].astype(np.float32)
    n_val_segments = int(cached.get("n_segments", len(y_true)))
    eval_seconds = time.time() - t0_load
    logger.info("Loaded predictions: %d segments | %.1f s | %s",
                n_val_segments, eval_seconds, PREDICTIONS_RAW_NPZ.name)
    logger.info("  y_true  : shape=%s, sum=%d (positives)",
                y_true.shape, int(y_true.sum()))
    logger.info("  y_prob  : shape=%s, range=[%.4f, %.4f]",
                y_prob.shape, float(y_prob.min()), float(y_prob.max()))

    # -- Step 7: Three-row post-processing ----------------------------------
    logger.info("-" * 65)
    logger.info("Step 7: Run post-processing evaluation (Row 1 + Row 2; Row 3 retired)")
    (row1_metrics, row2_metrics,
     post_row2, far_row2) = run_postprocessing_evaluations(
         y_true, y_prob, logger)

    # -- Step 8: Save report and figures ------------------------------------
    logger.info("-" * 65)
    logger.info("Step 8: Save evaluation report, two-row CSV, and figures")
    y_pred_row1 = (y_prob >= 0.5).astype(int)
    y_pred_row2 = post_row2["smoothed_preds"]

    # -- (Retired) Step 8a: Save comprehensive predictions bundle ----------
    # The Row 3 / optimal-threshold fields no longer have meaning, and the
    # raw insurance copy already exists at PREDICTIONS_RAW_NPZ. Rather than
    # overwrite that file with a Row-3-less variant, this step is left as
    # commented scaffolding. Re-enable together with the threshold and
    # Row 3 blocks if the optimal-threshold workflow is ever restored.
    #
    # bundle_npz_path = OUTPUT_ROOT / "multiscale_tcn_predictions.npz"
    # np.savez_compressed(
    #     bundle_npz_path,
    #     y_true=y_true.astype(np.int8),
    #     y_prob=y_prob.astype(np.float32),
    #     y_pred_row1=y_pred_row1.astype(np.int8),
    #     y_pred_row2=y_pred_row2.astype(np.int8),
    #     segment_sec=np.float32(SEGMENT_SEC),
    #     smoothing_win=np.int64(SMOOTHING_WIN),
    #     refractory_sec=np.float32(REFRACTORY_SEC),
    #     min_event_sec=np.float32(MIN_EVENT_SEC),
    #     n_segments=np.int64(len(y_true)),
    # )
    # logger.info("Saved predictions bundle: %s", bundle_npz_path)

    # Load the per-epoch training history from MultiScaleTCN.py's persisted
    # artefacts (training_log.json preferred, epoch_metrics.csv as fallback,
    # single-point stub as last resort). The full multi-epoch history is
    # what makes the EMA-smoothed training curves in plot_all_figures
    # actually informative -- a single best-epoch point would render as a
    # dot, not a trajectory.
    logger.info("-" * 65)
    logger.info("Step 7b: Load per-epoch training history")
    history = _load_history(logger)
    best_epoch = int(history["epoch"][history["val_f1"].index(max(history["val_f1"]))])
    best_val_f1 = float(max(history["val_f1"]))
    logger.info("  Best epoch / val F1 from history : %d / %.4f", best_epoch, best_val_f1)
    elapsed_dt = datetime.timedelta(seconds=int(eval_seconds))

    save_all_results(
        history, row1_metrics, row2_metrics,
        far_row2, hp, branch_dilations,
        best_epoch, best_val_f1, elapsed_dt, device, n_params, y_true,
        y_pred_row1, y_pred_row2, logger)

    plot_all_figures(
        history, best_epoch, best_val_f1, y_true, y_prob,
        y_pred_row1, y_pred_row2,
        row1_metrics, row2_metrics,
        post_row2,
        branch_dilations, hp, logger)

    # -- Step 9: Result_classReport bar plot --------------------------------
    # Macro-avg classification metrics (Row 1 + Row 2) plus FAR/hr in a
    # standalone two-panel figure -- identical layout across M1, M2, M3, M4
    # via the shared helper in tcn_utils. Writes under POST_EVAL_DIR (its
    # own Result_classReport folder); the bar plot the training script
    # produced under OUTPUT_ROOT/Result_classReport/ is not overwritten.
    logger.info("-" * 65)
    logger.info("Step 9: Result_classReport bar plot (Row 1 + Row 2)")
    make_classreport_barplot(
        row1_metrics, row2_metrics,
        RESULT_CLASSREPORT_DIR / "multiscale_tcn_classreport_barplot.png",
        title_prefix="Multi-Scale TCN", logger=logger)

    # -- Final summary ------------------------------------------------------
    logger.info("=" * 65)
    logger.info("M3 POST-EVAL COMPLETE")
    logger.info("  Val segments           : %d (cached predictions)", n_val_segments)
    logger.info("  Source                 : %s", PREDICTIONS_RAW_NPZ.name)
    logger.info("  Wall time (load only)  : %.2f s", eval_seconds)
    logger.info("  Row 1 F1 (raw @ 0.5)   : %.4f",
                row1_metrics.get("f1_macro", float("nan")))
    logger.info("  Row 2 F1 (post @ 0.5)  : %.4f",
                row2_metrics.get("f1_macro", float("nan")))
    logger.info("  Row 3                  : retired (threshold optimisation disabled)")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
