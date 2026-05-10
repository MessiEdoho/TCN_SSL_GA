"""
eval_utils.py
=============
Shared four-layer NaN-protection plumbing for test-set evaluation scripts.

Background
----------
Earlier in the project, FP16 forward passes on the M3 model produced NaN
logits when fed segments that survived the upstream filters but still had
extreme amplitudes or non-finite values. The fix was a four-layer
defence-in-depth pattern, originally implemented inline in
m3_post_eval.py during the M3 recovery pass:

  Layer 1 : Manifest-level filter -- apply_val_test_filter.py removed
            segments with NaN/Inf or |x| > 1000 from the val and test
            partitions, recording the operation in meta.filter_history.
            verify_manifest_filtered() reads that record before any
            inference begins and aborts if it is missing.

  Layer 2 : Dataset hardening    -- SafeEEGSegmentDataset subclasses
            EEGSegmentDataset and sanitises every loaded segment in
            __getitem__: non-finite -> 0.0, |x| > 1000 -> clipped. The
            class-level counter `n_sanitised` records how many segments
            needed defensive cleanup (should be 0 when Layer 1 ran).

  Layer 3 : FP32 forward         -- evaluate_model_safe defaults
            use_amp=False, eliminating the FP16 overflow failure mode
            entirely.

  Layer 4 : Per-batch isfinite   -- evaluate_model_safe asserts
            torch.isfinite(logits).all() at every batch and raises with
            a localised diagnostic if any logit is non-finite.

The same four-layer pattern is required by every test-set evaluation
script in the repo (M1/M2/M3/M4). Centralising it here means a single
source of truth -- bug fixes apply uniformly to all eval scripts.

m3_post_eval.py is NOT a dependency of this module and must NOT be
imported anywhere except for backward-compat reads of its legacy NPZ.
It remains as a one-off recovery utility for the original M3 crash and
should be left untouched.

Usage
-----
    from eval_utils import (
        AMPLITUDE_THRESHOLD,
        SafeEEGSegmentDataset,
        make_safe_loader,
        evaluate_model_safe,
        verify_manifest_filtered,
    )

    # Layer 1
    manifest_filter = verify_manifest_filtered(SPLITS_PATH, partition_key="test", logger=logger)

    # Layer 2 (DataLoader uses SafeEEGSegmentDataset internally)
    loader = make_safe_loader(test_pairs, batch_size, device)

    # Layer 3 + Layer 4
    f1, y_true, y_pred, y_prob = evaluate_model_safe(
        model, loader, device, logger, use_amp=False)
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import json
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score

from tcn_utils import EEGSegmentDataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AMPLITUDE_THRESHOLD = 1000.0     # matches train-side filter (apply_val_test_filter.py)
NUM_DATA_WORKERS    = 4          # default DataLoader workers for the safe loader


# ---------------------------------------------------------------------------
# SafeEEGSegmentDataset  (Layer 2: dataset hardening)
# ---------------------------------------------------------------------------
class SafeEEGSegmentDataset(EEGSegmentDataset):
    """Defensive subclass that sanitises every loaded segment.

    Even after Layer 1 (apply_val_test_filter), a stray bad file could in
    principle make it to the loader (e.g., race conditions, corrupt write
    between scan and eval). This class catches anything that slips through:

      - non-finite values  -> replaced by 0.0   via np.nan_to_num
      - amplitude > 1000   -> clipped to +-1000 via np.clip

    Maintains a class-level counter `n_sanitised` so the eval report can
    record how many samples needed defensive cleanup. If the counter is
    non-zero after a run, the upstream filter likely missed something.
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


# ---------------------------------------------------------------------------
# verify_manifest_filtered  (Layer 1: manifest-level filter check)
# ---------------------------------------------------------------------------
def verify_manifest_filtered(splits_path, partition_key, logger):
    """Verify the manifest's meta block records an apply_val_test_filter step
    and return the recorded statistics for the requested partition.

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

    Parameters
    ----------
    splits_path : Path or str
        Path to the splits manifest JSON.
    partition_key : str
        Which partition's stats to surface in the returned dict. Typically
        "test" for test-eval scripts, "val" for any val-eval recovery
        scripts. Must match the key inside the apply_val_test_filter step.
    logger : logging.Logger
        For PASS/FAIL log lines.

    Returns
    -------
    dict
        Keys: 'step', 'timestamp', 'threshold',
              '<partition_key>_after', '<partition_key>_removed'.
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
            "manifest.", splits_path)
        sys.exit(1)

    partition_block = apply_step.get(partition_key) or {}
    info = {
        "step":       apply_step.get("step", "apply_val_test_filter"),
        "timestamp":  apply_step.get("timestamp", "unknown"),
        "threshold":  float(apply_step.get("threshold", AMPLITUDE_THRESHOLD)),
        f"{partition_key}_after":   int(partition_block.get("after", 0)),
        f"{partition_key}_removed": int(partition_block.get("removed", 0)),
    }
    logger.info(
        "Layer 1 verification PASSED: manifest filtered by %s at %s "
        "(threshold=%.1f). %s: %d retained, %d removed.",
        info["step"], info["timestamp"], info["threshold"],
        partition_key.upper(),
        info[f"{partition_key}_after"],
        info[f"{partition_key}_removed"])
    return info


# ---------------------------------------------------------------------------
# make_safe_loader
# ---------------------------------------------------------------------------
def make_safe_loader(file_label_pairs, batch_size, device,
                     num_workers=NUM_DATA_WORKERS):
    """Eval-only DataLoader using the hardened SafeEEGSegmentDataset.

    No shuffling (test/val eval is order-independent), drop_last=False so
    every segment is evaluated, persistent_workers=True to amortise worker
    spawn cost over the multi-hour pass.
    """
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

    Mirrors the training-script `evaluate_model` interface (returns the
    same 4-tuple) but adds:
      - use_amp default False -> FP32 forward (Layer 3)
      - torch.isfinite(logits).all() assertion at every batch (Layer 4)
      - per-50-batch progress logging (long-running pass)

    If the Layer-4 assertion fires, the function raises with a localised
    diagnostic (batch index, sample indices, first 8 logit values) so we
    know precisely where to investigate. This is intentional fail-loud
    behaviour -- the alternative would be writing a NaN-corrupted report.

    Returns
    -------
    (val_f1, y_true, y_pred, y_prob) -- same shape as evaluate_model in
    the training scripts. y_pred is the raw t=0.5 thresholded prediction.
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
