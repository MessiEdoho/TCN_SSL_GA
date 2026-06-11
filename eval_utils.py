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
import csv
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, matthews_corrcoef,
)

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


# ===========================================================================
# Chronologically-correct event detection and metrics
# ===========================================================================
# Verified pieces lifted from recover_event_metrics.py (proven on local M3
# val + test). These are the deployment-quality replacements for the
# broken event-detection logic in the production pipeline that:
#   (1) treated the entire partition as one concatenated stream regardless
#       of mouse boundaries, and
#   (2) used segment_len_sec as the FAR/hr denominator (over-counts by 2x
#       under 50% segment overlap).
#
# All functions here operate on per-mouse, chronologically-ordered prediction
# streams. The chronology is rebuilt deterministically by replaying the
# preprocessing grid walk against the same Excel annotations preprocessing
# consumed -- no EDF re-loading required.


# ---------------------------------------------------------------------------
# Post-processing constants (single source of truth across all eval scripts)
# ---------------------------------------------------------------------------
FS              = 500       # Hz
WIN_LEN         = 2500      # samples (= 5 s)
STEP            = 1250      # samples (= 2.5 s)
SEGMENT_SEC     = WIN_LEN / FS    # 5.0
STEP_SEC        = STEP / FS       # 2.5

THRESHOLD       = 0.5
SMOOTHING_WIN   = 3
REFRACTORY_SEC  = 30.0
MIN_EVENT_SEC   = 25.0
MAX_EVENT_SEC   = 120.0     # drop events longer than 2 min (artefact / stuck-model cleanup;
                            # applied as the FINAL step in both orderings of
                            # detect_events_in_chunk -- see STUDY_REPORT 7.6.9)


# ---------------------------------------------------------------------------
# load_annotations -- mirrors preprocessing_binary.load_annotations
# ---------------------------------------------------------------------------
def load_annotations(xlsx_path, recording_start_dt):
    """Read the Excel annotation file and return seizure intervals as a list
    of (start_sec, end_sec) tuples relative to recording start.
    """
    df = pd.read_excel(xlsx_path)
    df["start_time"] = pd.to_datetime(df["start_time"], dayfirst=True)
    df["end_time"]   = pd.to_datetime(df["end_time"],   dayfirst=True)

    intervals = []
    for _, row in df.iterrows():
        s = row["start_time"].to_pydatetime().replace(tzinfo=None)
        e = row["end_time"].to_pydatetime().replace(tzinfo=None)
        intervals.append(((s - recording_start_dt).total_seconds(),
                          (e - recording_start_dt).total_seconds()))
    return intervals


# ---------------------------------------------------------------------------
# is_ictal -- mirrors preprocessing_binary.is_ictal
# ---------------------------------------------------------------------------
def is_ictal(seg_start_sec, seg_end_sec, seizure_intervals):
    for sz_start, sz_end in seizure_intervals:
        if seg_start_sec < sz_end and seg_end_sec > sz_start:
            return True
    return False


# ---------------------------------------------------------------------------
# build_mouse_chronology
# ---------------------------------------------------------------------------
def build_mouse_chronology(mouse_id, n_samples, seizure_intervals, logger):
    """Replay the preprocessing grid walk to produce a mapping from each
    .npy filename to its true chronological position in the recording.

    Returns
    -------
    fname_to_chrono : dict
        Keys are .npy basenames (e.g. 'm1_ictal_00001.npy'); values are
        dicts with keys 'chrono_idx', 't_start_sec', 'label' (1=ictal).
    n_ictal, n_nonictal : int
        Total counts on the reconstructed grid (filtered or not).
    """
    last_valid = n_samples - WIN_LEN
    grid = np.arange(0, last_valid + 1, STEP, dtype=np.int64)

    fname_to_chrono = {}
    ictal_count    = 0
    nonictal_count = 0
    for chrono_idx, seg_start_idx in enumerate(grid):
        seg_start_sec = float(seg_start_idx) / FS
        seg_end_sec   = float(seg_start_idx + WIN_LEN) / FS

        if is_ictal(seg_start_sec, seg_end_sec, seizure_intervals):
            ictal_count += 1
            fname = f"{mouse_id}_ictal_{ictal_count:05d}.npy"
            label = 1
        else:
            nonictal_count += 1
            fname = f"{mouse_id}_nonictal_{nonictal_count:05d}.npy"
            label = 0

        fname_to_chrono[fname] = {
            "chrono_idx":  chrono_idx,
            "t_start_sec": seg_start_sec,
            "label":       label,
        }

    logger.info("  %-8s : grid=%d | ictal=%d | nonictal=%d",
                mouse_id, len(grid), ictal_count, nonictal_count)
    return fname_to_chrono, ictal_count, nonictal_count


# ---------------------------------------------------------------------------
# reorder_to_chronology
# ---------------------------------------------------------------------------
def reorder_to_chronology(mouse_id, mouse_block, fname_to_chrono,
                          y_true_all, y_prob_all, logger):
    """Reorder one mouse's segments into chronological order. mouse_block
    is a dict with 'fnames', 'labels', 'npz_indices' lists in manifest order.
    Returns parallel arrays sorted by chrono_idx, or None on label mismatch.
    """
    chrono_idx_list = []
    t_start_list    = []
    label_list      = []
    npz_idx_list    = []
    unmatched       = 0

    for fname, label, npz_idx in zip(mouse_block["fnames"],
                                     mouse_block["labels"],
                                     mouse_block["npz_indices"]):
        info = fname_to_chrono.get(fname)
        if info is None:
            unmatched += 1
            continue
        if info["label"] != label:
            logger.warning("  %s : manifest label %d disagrees with replayed "
                           "label %d for %s", mouse_id, label, info["label"], fname)
        chrono_idx_list.append(info["chrono_idx"])
        t_start_list.append(info["t_start_sec"])
        label_list.append(label)
        npz_idx_list.append(npz_idx)

    if unmatched:
        logger.warning("  %s : %d manifest segments not found in replayed "
                       "chronology (likely filename convention mismatch).",
                       mouse_id, unmatched)

    order = np.argsort(np.asarray(chrono_idx_list, dtype=np.int64))
    chrono_idx_arr = np.asarray(chrono_idx_list, dtype=np.int64)[order]
    t_start_arr    = np.asarray(t_start_list,    dtype=np.float64)[order]
    label_arr      = np.asarray(label_list,      dtype=np.int64)[order]
    npz_idx_arr    = np.asarray(npz_idx_list,    dtype=np.int64)[order]

    y_true_mouse = y_true_all[npz_idx_arr]
    y_prob_mouse = y_prob_all[npz_idx_arr]

    if not np.array_equal(y_true_mouse, label_arr):
        n_diff = int(np.sum(y_true_mouse != label_arr))
        logger.error("  %s : %d/%d y_true mismatches between NPZ and "
                     "manifest labels after reordering. Aborting this mouse.",
                     mouse_id, n_diff, len(label_arr))
        return None

    return {
        "chrono_idx":  chrono_idx_arr,
        "t_start_sec": t_start_arr,
        "y_true":      y_true_mouse,
        "y_prob":      y_prob_mouse,
    }


# ---------------------------------------------------------------------------
# split_into_chunks
# ---------------------------------------------------------------------------
def split_into_chunks(chrono_idx_arr):
    """Split a sorted chrono_idx array into contiguous chunks. Two
    consecutive chrono_idx values that differ by > 1 are separated by
    at least one filtered-out segment, so they break the chunk.
    Returns a list of (start_pos, end_pos) slice indices into the array.
    """
    if len(chrono_idx_arr) == 0:
        return []
    deltas = np.diff(chrono_idx_arr)
    breaks = np.where(deltas != 1)[0] + 1
    starts = np.concatenate(([0], breaks))
    ends   = np.concatenate((breaks, [len(chrono_idx_arr)]))
    return list(zip(starts.tolist(), ends.tolist()))


# ---------------------------------------------------------------------------
# detect_events_in_chunk
# ---------------------------------------------------------------------------
def _refractory_merge(events):
    """Merge consecutive events whose inter-event gap is < REFRACTORY_SEC."""
    merged = []
    for evt in events:
        if merged and (evt["start_sec"] - merged[-1]["end_sec"]) < REFRACTORY_SEC:
            merged[-1]["end_sec"] = evt["end_sec"]
            merged[-1]["seg_indices"].extend(evt["seg_indices"])
        else:
            merged.append({
                "start_sec":   evt["start_sec"],
                "end_sec":     evt["end_sec"],
                "seg_indices": list(evt["seg_indices"]),
            })
    return merged


def _min_duration_filter(events, min_sec=None):
    """Drop events shorter than min_sec (default: MIN_EVENT_SEC)."""
    if min_sec is None:
        min_sec = MIN_EVENT_SEC
    return [evt for evt in events
            if (evt["end_sec"] - evt["start_sec"]) >= min_sec]


def _max_duration_filter(events, max_sec=None):
    """Drop events longer than max_sec (default: MAX_EVENT_SEC). Catches
    artefact-driven super-long events and refractory-merged super-events
    that should not be considered single seizures."""
    if max_sec is None:
        max_sec = MAX_EVENT_SEC
    return [evt for evt in events
            if (evt["end_sec"] - evt["start_sec"]) <= max_sec]


def detect_events_in_chunk(t_start, y_prob, mouse_id, chunk_id,
                           order="min_then_refractory",
                           min_event_duration_sec=None,
                           max_event_duration_sec=None):
    """Run smoothing + threshold + run-detection, then apply min-duration,
    refractory-merge (in the order specified), and finally max-duration.

    Parameters
    ----------
    order : {"refractory_then_min", "min_then_refractory"}, default
        "min_then_refractory" (selected by the 2 x 5 grid sweep on val;
        see STUDY_REPORT.txt Section 7.6.10).
        "min_then_refractory" drops short candidate events first, then
        merges surviving ones; matches the standard order in the clinical
        seizure-detection literature and Pareto-dominates the alternative
        on this dataset.
        "refractory_then_min" (legacy) merges first, then drops surviving
        short events; can rescue fragmented true detections at the cost
        of also rescuing fragmented false alarms.
    min_event_duration_sec : float or None
        Minimum event length. None -> use module-level MIN_EVENT_SEC.
    max_event_duration_sec : float or None
        Maximum event length. Applied as the FINAL step in both orderings.
        None -> use module-level MAX_EVENT_SEC. Events exceeding this are
        dropped (artefact / model-stuck cleanup).

    Returns
    -------
    (events, smoothed_probs, smoothed_preds)
        Chunks shorter than SMOOTHING_WIN bypass smoothing so the smoothed
        array length always equals the input length.
    """
    n = len(y_prob)
    if n == 0:
        return [], np.array([], dtype=np.float64), np.array([], dtype=np.int64)

    if n < SMOOTHING_WIN:
        smoothed = y_prob.astype(np.float64).copy()
    else:
        kernel = np.ones(SMOOTHING_WIN) / SMOOTHING_WIN
        smoothed = np.convolve(y_prob, kernel, mode="same")
    preds = (smoothed >= THRESHOLD).astype(np.int64)

    raw_events = []
    in_event = False
    seg_idx_in_event = []
    for i in range(n):
        if preds[i] == 1 and not in_event:
            in_event = True
            seg_idx_in_event = [i]
        elif preds[i] == 1 and in_event:
            seg_idx_in_event.append(i)
        elif preds[i] == 0 and in_event:
            in_event = False
            raw_events.append({
                "start_sec":   float(t_start[seg_idx_in_event[0]]),
                "end_sec":     float(t_start[seg_idx_in_event[-1]]) + SEGMENT_SEC,
                "seg_indices": list(seg_idx_in_event),
            })
    if in_event:
        raw_events.append({
            "start_sec":   float(t_start[seg_idx_in_event[0]]),
            "end_sec":     float(t_start[seg_idx_in_event[-1]]) + SEGMENT_SEC,
            "seg_indices": list(seg_idx_in_event),
        })

    if order == "refractory_then_min":
        processed = _min_duration_filter(
            _refractory_merge(raw_events), min_event_duration_sec)
    elif order == "min_then_refractory":
        processed = _refractory_merge(
            _min_duration_filter(raw_events, min_event_duration_sec))
    else:
        raise ValueError(
            "Unknown post-processing order %r; expected 'refractory_then_min' "
            "or 'min_then_refractory'." % order)

    # Max-duration filter: applied as the FINAL step in both orderings so
    # it catches both raw long runs and refractory-merged super-events.
    # Events with duration > max_event_duration_sec are dropped (treated as
    # artefact / model-stuck cleanup; see STUDY_REPORT 7.6.9).
    processed = _max_duration_filter(processed, max_event_duration_sec)

    final = []
    for evt in processed:
        seg_idx = evt["seg_indices"]
        final.append({
            "mouse_id":     mouse_id,
            "chunk_id":     chunk_id,
            "start_sec":    round(evt["start_sec"], 4),
            "end_sec":      round(evt["end_sec"], 4),
            "duration_sec": round(evt["end_sec"] - evt["start_sec"], 4),
            "mean_prob":    round(float(np.mean(smoothed[seg_idx])), 6),
            "max_prob":     round(float(np.max(smoothed[seg_idx])), 6),
        })
    return final, smoothed, preds


# ---------------------------------------------------------------------------
# match_events_to_ground_truth
# ---------------------------------------------------------------------------
def match_events_to_ground_truth(predicted_events, gt_intervals):
    """Match predicted events against ground-truth seizure_intervals using
    the any-overlap rule. Returns (tp, fp, fn, latencies) where:
      tp -- list of (gt_idx, predicted_event) tuples (one per matched GT)
      fp -- list of predicted events with no GT overlap
      fn -- list of (gt_idx, gt_interval) for missed seizures
      latencies -- list of (matched_pred.start_sec - gt.start_sec) per TP
    A single predicted event may overlap multiple GT seizures; in that
    case each GT counts as a separate TP.
    """
    matched_gt        = set()
    matched_pred_idxs = set()
    latencies         = []
    tp                = []

    for gt_idx, (gs, ge) in enumerate(gt_intervals):
        first_pred = None
        for p_idx, p in enumerate(predicted_events):
            if p["start_sec"] < ge and p["end_sec"] > gs:
                matched_gt.add(gt_idx)
                matched_pred_idxs.add(p_idx)
                if first_pred is None or p["start_sec"] < first_pred["start_sec"]:
                    first_pred = p
        if first_pred is not None:
            tp.append((gt_idx, first_pred))
            latencies.append(first_pred["start_sec"] - gs)

    fp = [p for p_idx, p in enumerate(predicted_events) if p_idx not in matched_pred_idxs]
    fn = [(gt_idx, gt_intervals[gt_idx]) for gt_idx in range(len(gt_intervals))
          if gt_idx not in matched_gt]
    return tp, fp, fn, latencies


# ---------------------------------------------------------------------------
# safe_div -- helper
# ---------------------------------------------------------------------------
def safe_div(num, den):
    return float(num) / float(den) if den > 0 else 0.0


# ---------------------------------------------------------------------------
# compute_segment_level_metrics
# ---------------------------------------------------------------------------
def compute_segment_level_metrics(y_true, y_pred, y_prob_for_auc, n_non_ictal_segs):
    """Per-segment classification metrics + corrected FAR/hr (denominator
    x step_sec=2.5, the unique recording time each segment contributes
    under 50% segment overlap).
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

    accuracy = accuracy_score(y_true, y_pred)
    prec     = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall   = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1m      = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    spec     = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auroc    = roc_auc_score(y_true, y_prob_for_auc) if len(np.unique(y_true)) > 1 else 0.0
    ap       = average_precision_score(y_true, y_prob_for_auc) if len(np.unique(y_true)) > 1 else 0.0
    mcc      = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
    youden_j = recall + spec - 1.0

    non_ic_hrs = (n_non_ictal_segs * STEP_SEC) / 3600.0
    far_seg    = fp / non_ic_hrs if non_ic_hrs > 0 else 0.0

    return {
        "accuracy":          round(accuracy, 6),
        "precision":         round(prec, 6),
        "recall":            round(recall, 6),
        "sensitivity":       round(recall, 6),
        "specificity":       round(spec, 6),
        "youden_j":          round(youden_j, 6),
        "f1_macro":          round(f1m, 6),
        "precision_macro":   round(prec_macro, 6),
        "recall_macro":      round(rec_macro, 6),
        "mcc":               round(mcc, 6),
        "auroc":             round(auroc, 6),
        "average_precision": round(ap, 6),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "far_per_hour_seg_CORRECTED_2_5s_denom": round(far_seg, 6),
    }


# ---------------------------------------------------------------------------
# build_classification_report
# ---------------------------------------------------------------------------
def build_classification_report(y_true, y_pred, row_metrics):
    """sklearn classification_report dict with target_names=['Non-ictal',
    'Ictal'], augmented with auroc / prauc / specificity / sensitivity /
    corrected FAR/hr so each per-row JSON is self-contained.
    """
    report = classification_report(
        y_true, y_pred, target_names=["Non-ictal", "Ictal"],
        output_dict=True, zero_division=0)
    report["auroc"]                                = row_metrics.get("auroc")
    report["prauc"]                                = row_metrics.get("average_precision")
    report["specificity"]                          = row_metrics.get("specificity")
    report["sensitivity"]                          = row_metrics.get("sensitivity")
    report["precision_macro"]                      = row_metrics.get("precision_macro")
    report["recall_macro"]                         = row_metrics.get("recall_macro")
    report["mcc"]                                  = row_metrics.get("mcc")
    report["far_per_hour_seg_CORRECTED_2_5s_denom"] = row_metrics.get("far_per_hour_seg_CORRECTED_2_5s_denom")
    return report


# ---------------------------------------------------------------------------
# evaluate_event_level -- the production orchestrator
# ---------------------------------------------------------------------------
def evaluate_event_level(partition_records, y_true_all, y_prob_all,
                         annotations_dir, mouse_metadata, logger,
                         order="min_then_refractory",
                         min_event_duration_sec=None,
                         max_event_duration_sec=None):
    """End-to-end per-mouse-chronological event-level evaluation.

    Parameters
    ----------
    partition_records : list of dict
        Enriched manifest records (must have keys filepath, label,
        mouse_id, chrono_idx, t_start_sec). Order matches y_true_all
        and y_prob_all (i.e. the order the loader processed them).
    y_true_all, y_prob_all : np.ndarray, length N
        Arrays from the model's val/test forward pass, in manifest order.
    annotations_dir : pathlib.Path
        Directory containing {mouse_id}_xlsx.xlsx seizure annotation files.
    mouse_metadata : dict
        Loaded mouse_recording_metadata.json (per-mouse n_samples, fs_hz,
        recording_start_dt).
    logger : logging.Logger
    order : {"refractory_then_min", "min_then_refractory"}, default
        "min_then_refractory". Controls the post-processing step order
        in detect_events_in_chunk -- see that function's docstring.
    min_event_duration_sec, max_event_duration_sec : float or None
        Pass-through to detect_events_in_chunk. None -> use module-level
        MIN_EVENT_SEC / MAX_EVENT_SEC.

    Returns
    -------
    dict with keys:
        "totals"               -- partition-level counts
        "event_level_metrics"  -- TP/FP/FN/Precision/Recall/F1/FAR/hr/latency
        "segment_level_metrics"-- {row1_raw_threshold_0_5, row2_postproc_threshold_0_5}
                                  each with both BROKEN and CORRECTED FAR/hr
        "per_mouse_results"    -- {mouse_id: {...}}
        "all_event_details"    -- list of per-event diagnostic rows
        "reordered_arrays"     -- {y_true, y_prob, y_pred_row1, y_pred_row2,
                                   mouse_id, chrono_idx, t_start_sec}
                                  All sorted per-mouse-chronologically; ready
                                  to dump into an enriched val/test NPZ.
    """
    # Step 1: Group records by mouse, preserve original NPZ index for lookup.
    per_mouse = {}
    for npz_idx, rec in enumerate(partition_records):
        mid = rec.get("mouse_id") or Path(rec["filepath"]).stem.split("_")[0]
        per_mouse.setdefault(mid, {
            "fnames": [], "labels": [], "npz_indices": [],
            "chrono_idxs": [], "t_starts": [],
        })
        per_mouse[mid]["fnames"].append(Path(rec["filepath"]).name)
        per_mouse[mid]["labels"].append(int(rec["label"]))
        per_mouse[mid]["npz_indices"].append(npz_idx)
        per_mouse[mid]["chrono_idxs"].append(int(rec["chrono_idx"]))
        per_mouse[mid]["t_starts"].append(float(rec["t_start_sec"]))

    # Step 2: Per-mouse processing.
    all_predicted     = []
    all_event_details = []
    per_mouse_results = {}

    seg_y_true_chunks    = []
    seg_y_prob_chunks    = []
    seg_smoothed_probs   = []
    seg_smoothed_preds   = []

    reord_y_true     = []
    reord_y_prob     = []
    reord_y_pred_row1= []
    reord_y_pred_row2= []
    reord_mouse_id   = []
    reord_chrono_idx = []
    reord_t_start    = []

    total_tp = total_fp = total_fn = 0
    total_n_gt = 0
    total_non_ictal_segs = 0
    total_latencies = []

    for mouse_id in sorted(per_mouse):
        mouse_block = per_mouse[mouse_id]
        if mouse_id not in mouse_metadata:
            logger.error("  %s : no metadata record. Skipping.", mouse_id)
            continue
        meta = mouse_metadata[mouse_id]
        xlsx_path = annotations_dir / f"{mouse_id}_xlsx.xlsx"
        if not xlsx_path.exists():
            logger.error("  %s : annotation Excel not found at %s. Skipping.",
                         mouse_id, xlsx_path)
            continue

        rec_start = datetime.datetime.fromisoformat(meta["recording_start_dt"])
        seizure_intervals = load_annotations(xlsx_path, rec_start)
        logger.info("  %s : %d ground-truth seizure(s) from %s",
                    mouse_id, len(seizure_intervals), xlsx_path.name)

        sort_idx = np.argsort(np.asarray(mouse_block["chrono_idxs"], dtype=np.int64))
        chrono_idx_arr = np.asarray(mouse_block["chrono_idxs"], dtype=np.int64)[sort_idx]
        t_start_arr    = np.asarray(mouse_block["t_starts"],    dtype=np.float64)[sort_idx]
        npz_idx_arr    = np.asarray(mouse_block["npz_indices"], dtype=np.int64)[sort_idx]
        label_arr      = np.asarray(mouse_block["labels"],      dtype=np.int64)[sort_idx]

        y_true_mouse = y_true_all[npz_idx_arr]
        y_prob_mouse = y_prob_all[npz_idx_arr]

        if not np.array_equal(y_true_mouse, label_arr):
            n_diff = int(np.sum(y_true_mouse != label_arr))
            logger.error("  %s : %d/%d y_true mismatches between NPZ and "
                         "manifest labels. Skipping.", mouse_id, n_diff, len(label_arr))
            continue

        chunks = split_into_chunks(chrono_idx_arr)
        logger.info("  %s : %d surviving segments in %d contiguous chunk(s)",
                    mouse_id, len(chrono_idx_arr), len(chunks))

        mouse_predicted = []
        mouse_smoothed_probs_full = np.empty(len(chrono_idx_arr), dtype=np.float64)
        mouse_smoothed_preds_full = np.empty(len(chrono_idx_arr), dtype=np.int64)
        for chunk_id, (s, e) in enumerate(chunks):
            evts, smoothed_probs_chunk, smoothed_preds_chunk = detect_events_in_chunk(
                t_start_arr[s:e], y_prob_mouse[s:e], mouse_id, chunk_id,
                order=order,
                min_event_duration_sec=min_event_duration_sec,
                max_event_duration_sec=max_event_duration_sec)
            mouse_predicted.extend(evts)
            mouse_smoothed_probs_full[s:e] = smoothed_probs_chunk
            mouse_smoothed_preds_full[s:e] = smoothed_preds_chunk
            seg_y_true_chunks.append(y_true_mouse[s:e])
            seg_y_prob_chunks.append(y_prob_mouse[s:e])
            seg_smoothed_probs.append(smoothed_probs_chunk)
            seg_smoothed_preds.append(smoothed_preds_chunk)

        tp, fp, fn, latencies = match_events_to_ground_truth(
            mouse_predicted, seizure_intervals)
        n_non_ictal = int(np.sum(y_true_mouse == 0))

        per_mouse_results[mouse_id] = {
            "n_ground_truth_seizures":   len(seizure_intervals),
            "n_predicted_events":        len(mouse_predicted),
            "tp":                        len(tp),
            "fp":                        len(fp),
            "fn":                        len(fn),
            "precision":                 round(safe_div(len(tp), len(tp) + len(fp)), 6),
            "recall":                    round(safe_div(len(tp), len(tp) + len(fn)), 6),
            "f1":                        round(safe_div(2 * len(tp),
                                                        2 * len(tp) + len(fp) + len(fn)), 6),
            "non_ictal_segments":        n_non_ictal,
            "non_ictal_hours":           round(n_non_ictal * STEP_SEC / 3600.0, 4),
            "far_per_hour":              round(safe_div(len(fp),
                                                        n_non_ictal * STEP_SEC / 3600.0), 6),
            "mean_detection_latency_sec":(round(float(np.mean(latencies)), 4)
                                          if latencies else None),
            "n_chunks":                  len(chunks),
        }

        # Format seconds-from-recording-start as HHhMMmSS.s with overflow
        # hours (e.g. 73h14m22.5 for a 3-day recording). The "h"/"m"
        # separators block Excel from auto-parsing the cell as a duration
        # and silently truncating to mm:ss when sub-seconds are present.
        def _to_hms(sec):
            sec = float(sec)
            h = int(sec // 3600)
            m = int((sec % 3600) // 60)
            s = sec - h * 3600 - m * 60
            return f"{h:02d}h{m:02d}m{s:04.1f}"

        gt_lookup = {gt_idx: seizure_intervals[gt_idx] for gt_idx, _ in tp}
        for gt_idx, pred in tp:
            gt_start_sec, gt_end_sec = gt_lookup[gt_idx]
            latency = round(float(pred["start_sec"]) - float(gt_start_sec), 4)
            all_event_details.append({
                "mouse_id":                    mouse_id,
                "is_true_alarm":               True,
                "start_sec":                   pred["start_sec"],
                "end_sec":                     pred["end_sec"],
                "duration_sec":                pred["duration_sec"],
                "start_recording_time":        _to_hms(pred["start_sec"]),
                "end_recording_time":          _to_hms(pred["end_sec"]),
                "mean_prob":                   pred.get("mean_prob"),
                "max_prob":                    pred["max_prob"],
                "matched_gt_idx":              gt_idx,
                "matched_gt_start_sec":        round(gt_start_sec, 4),
                "matched_gt_end_sec":          round(gt_end_sec, 4),
                "matched_gt_start_recording_time": _to_hms(gt_start_sec),
                "matched_gt_end_recording_time":   _to_hms(gt_end_sec),
                "detection_latency_sec":       latency,
            })
        for pred in fp:
            all_event_details.append({
                "mouse_id":                    mouse_id,
                "is_true_alarm":               False,
                "start_sec":                   pred["start_sec"],
                "end_sec":                     pred["end_sec"],
                "duration_sec":                pred["duration_sec"],
                "start_recording_time":        _to_hms(pred["start_sec"]),
                "end_recording_time":          _to_hms(pred["end_sec"]),
                "mean_prob":                   pred.get("mean_prob"),
                "max_prob":                    pred["max_prob"],
                "matched_gt_idx":              None,
                "matched_gt_start_sec":        None,
                "matched_gt_end_sec":          None,
                "matched_gt_start_recording_time": None,
                "matched_gt_end_recording_time":   None,
                "detection_latency_sec":       None,
            })

        reord_y_true.append(y_true_mouse)
        reord_y_prob.append(y_prob_mouse)
        reord_y_pred_row1.append((y_prob_mouse >= THRESHOLD).astype(np.int64))
        reord_y_pred_row2.append(mouse_smoothed_preds_full)
        reord_mouse_id.extend([mouse_id] * len(chrono_idx_arr))
        reord_chrono_idx.append(chrono_idx_arr)
        reord_t_start.append(t_start_arr)

        all_predicted.extend(mouse_predicted)
        total_tp        += len(tp)
        total_fp        += len(fp)
        total_fn        += len(fn)
        total_n_gt      += len(seizure_intervals)
        total_non_ictal_segs += n_non_ictal
        total_latencies.extend(latencies)

        logger.info("  %s : GT=%d | Pred=%d | TP=%d FP=%d FN=%d | "
                    "Prec=%.3f Rec=%.3f F1=%.3f | FAR/hr=%.4f",
                    mouse_id, len(seizure_intervals), len(mouse_predicted),
                    len(tp), len(fp), len(fn),
                    per_mouse_results[mouse_id]["precision"],
                    per_mouse_results[mouse_id]["recall"],
                    per_mouse_results[mouse_id]["f1"],
                    per_mouse_results[mouse_id]["far_per_hour"])

    total_non_ictal_hours = total_non_ictal_segs * STEP_SEC / 3600.0

    seg_y_true_concat       = (np.concatenate(seg_y_true_chunks)    if seg_y_true_chunks    else np.array([], dtype=np.int64))
    seg_y_prob_concat       = (np.concatenate(seg_y_prob_chunks)    if seg_y_prob_chunks    else np.array([], dtype=np.float64))
    seg_smoothed_probs_cat  = (np.concatenate(seg_smoothed_probs)   if seg_smoothed_probs   else np.array([], dtype=np.float64))
    seg_smoothed_preds_cat  = (np.concatenate(seg_smoothed_preds)   if seg_smoothed_preds   else np.array([], dtype=np.int64))
    n_non_ic_concat = int(np.sum(seg_y_true_concat == 0))

    row1_pred = (seg_y_prob_concat >= THRESHOLD).astype(np.int64)
    row1_metrics = compute_segment_level_metrics(
        seg_y_true_concat, row1_pred, seg_y_prob_concat, n_non_ic_concat)
    row1_metrics["threshold"]     = THRESHOLD
    row1_metrics["postprocessed"] = False

    row2_metrics = compute_segment_level_metrics(
        seg_y_true_concat, seg_smoothed_preds_cat, seg_smoothed_probs_cat, n_non_ic_concat)
    row2_metrics["threshold"]     = THRESHOLD
    row2_metrics["postprocessed"] = True

    reordered = {
        "y_true":      np.concatenate(reord_y_true)     if reord_y_true     else np.array([], dtype=np.int64),
        "y_prob":      np.concatenate(reord_y_prob)     if reord_y_prob     else np.array([], dtype=np.float64),
        "y_pred_row1": np.concatenate(reord_y_pred_row1)if reord_y_pred_row1 else np.array([], dtype=np.int64),
        "y_pred_row2": np.concatenate(reord_y_pred_row2)if reord_y_pred_row2 else np.array([], dtype=np.int64),
        "mouse_id":    np.array(reord_mouse_id, dtype=object),
        "chrono_idx":  np.concatenate(reord_chrono_idx) if reord_chrono_idx else np.array([], dtype=np.int64),
        "t_start_sec": np.concatenate(reord_t_start)    if reord_t_start    else np.array([], dtype=np.float64),
    }

    return {
        "totals": {
            "n_mice":                       len(per_mouse_results),
            "n_segments_in_npz":             int(len(y_true_all)),
            "n_segments_in_chronology_used": int(len(seg_y_true_concat)),
            "n_ground_truth_seizures":       int(total_n_gt),
            "n_predicted_events":            int(len(all_predicted)),
            "non_ictal_segments_corrected":  int(total_non_ictal_segs),
            "non_ictal_hours_corrected":     round(total_non_ictal_hours, 4),
        },
        "event_level_metrics": {
            "tp":           int(total_tp),
            "fp":           int(total_fp),
            "fn":           int(total_fn),
            "precision":    round(safe_div(total_tp, total_tp + total_fp), 6),
            "recall":       round(safe_div(total_tp, total_tp + total_fn), 6),
            "sensitivity":  round(safe_div(total_tp, total_tp + total_fn), 6),
            "f1":           round(safe_div(2 * total_tp,
                                           2 * total_tp + total_fp + total_fn), 6),
            "far_per_hour_event_CORRECTED": round(safe_div(total_fp, total_non_ictal_hours), 6),
            "mean_detection_latency_sec":   (round(float(np.mean(total_latencies)), 4)
                                             if total_latencies else None),
        },
        "segment_level_metrics": {
            "row1_raw_threshold_0_5":      row1_metrics,
            "row2_postproc_threshold_0_5": row2_metrics,
        },
        "per_mouse_results": per_mouse_results,
        "all_event_details": all_event_details,
        "reordered_arrays":  reordered,
    }


# ===========================================================================
# Canonical event-level output bundle
# ---------------------------------------------------------------------------
# Single source of truth for the 4-file (event_details + summary +
# classification_report_row1 + classification_report_row2) bundle emitted
# by every event-level evaluation in the repo. postproc_sweep.py,
# recover_event_metrics.py, m4_event_metrics_recovery.py, the four
# training scripts, the two _evaluation scripts, and m3_post_eval.py
# all funnel through the helpers below so any schema change is one edit.
#
# deploy_inference.py uses the 2-file deploy variant (no ground truth,
# so no classification reports and a reduced summary.json) -- same
# event_details.csv schema with GT cells left empty.
# ===========================================================================

EVENT_DETAILS_FIELDS = [
    "event_idx",
    "mouse_id",
    "partition",
    "is_true_alarm",
    "start_sec",
    "end_sec",
    "duration_sec",
    "start_recording_time",
    "end_recording_time",
    "mean_prob",
    "max_prob",
    "matched_gt_idx",
    "matched_gt_start_sec",
    "matched_gt_end_sec",
    "matched_gt_start_recording_time",
    "matched_gt_end_recording_time",
    "detection_latency_sec",
]


def write_event_details_csv(out_path, events, partition, logger):
    """Write the canonical 17-column event_details.csv.

    Any field absent or equal to None is rendered as an empty cell, so the
    schema is identical between GT-having (training/eval/recovery/sweep)
    and GT-less (deploy_inference) contexts. Rows are written in the
    order supplied; an `event_idx` column is added automatically.

    Parameters
    ----------
    out_path : Path or str
    events : list of dict
        Each dict may carry any subset of EVENT_DETAILS_FIELDS keys
        (excluding event_idx, which is generated here, and `partition`,
        which is injected from the caller).
    partition : str
        Value written into the `partition` column for every row
        (e.g. "val", "test", "deploy").
    logger : logging.Logger
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVENT_DETAILS_FIELDS)
        writer.writeheader()
        for i, evt in enumerate(events):
            row = {k: "" for k in EVENT_DETAILS_FIELDS}
            row["event_idx"] = i
            row["partition"] = partition
            for k, v in evt.items():
                if k in EVENT_DETAILS_FIELDS and v is not None:
                    row[k] = v
            writer.writerow(row)
    logger.info("Saved: %s (%d events)", out_path, len(events))


def write_summary_json(out_path, payload, logger):
    """Write a JSON summary (any dict) with deterministic ordering."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8")
    logger.info("Saved: %s", out_path)


def write_classification_reports(out_dir, partition, evaluator_result, logger):
    """Write {partition}_classification_report_row1.json and _row2.json
    using the reordered arrays + segment-level metrics from
    evaluate_event_level's return dict. Delegates to the existing
    build_classification_report helper for schema consistency."""
    seg = evaluator_result["segment_level_metrics"]
    reord = evaluator_result["reordered_arrays"]
    y_true = reord["y_true"]
    r1 = build_classification_report(
        y_true, reord["y_pred_row1"], seg["row1_raw_threshold_0_5"])
    r2 = build_classification_report(
        y_true, reord["y_pred_row2"], seg["row2_postproc_threshold_0_5"])
    p1 = Path(out_dir) / f"{partition}_classification_report_row1.json"
    p2 = Path(out_dir) / f"{partition}_classification_report_row2.json"
    write_summary_json(p1, r1, logger)
    write_summary_json(p2, r2, logger)


def write_event_level_bundle(out_dir, partition, model_label,
                              evaluator_result, logger,
                              order="min_then_refractory",
                              min_event_duration_sec=25.0,
                              refractory_period_sec=30.0,
                              smoothing_window=3,
                              threshold=0.5,
                              step_sec=2.5,
                              max_event_duration_sec=120.0,
                              extra_summary_fields=None):
    """Emit the canonical 4-file bundle:

        {partition}_summary.json
        {partition}_event_details.csv
        {partition}_classification_report_row1.json
        {partition}_classification_report_row2.json

    Parameters
    ----------
    out_dir : Path
    partition : str
        "val" / "test" / "deploy" -- becomes the filename prefix and the
        partition column inside event_details.csv.
    model_label : str
        Descriptive label written into summary.json's "model" field,
        e.g. "M3 (MultiScaleTCN)".
    evaluator_result : dict
        Return value of evaluate_event_level. Required keys:
        totals, event_level_metrics, segment_level_metrics,
        per_mouse_results, all_event_details, reordered_arrays.
    extra_summary_fields : dict, optional
        Additional top-level keys to merge into summary.json (e.g. a
        timestamp from the caller).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model":     model_label,
        "partition": partition,
        "timestamp": datetime.datetime.now().isoformat(),
        "post_processing_params": {
            "matching_rule":           "any-overlap",
            "order":                   order,
            "min_event_duration_sec":  min_event_duration_sec,
            "max_event_duration_sec":  max_event_duration_sec,
            "refractory_period_sec":   refractory_period_sec,
            "smoothing_window":        smoothing_window,
            "step_sec":                step_sec,
            "threshold":               threshold,
        },
        "event_level_metrics":   evaluator_result["event_level_metrics"],
        "segment_level_metrics": evaluator_result["segment_level_metrics"],
        "per_mouse":             evaluator_result["per_mouse_results"],
        "totals":                evaluator_result["totals"],
    }
    if extra_summary_fields:
        summary.update(extra_summary_fields)

    write_summary_json(
        out_dir / f"{partition}_summary.json", summary, logger)
    write_event_details_csv(
        out_dir / f"{partition}_event_details.csv",
        evaluator_result["all_event_details"], partition, logger)
    write_classification_reports(out_dir, partition, evaluator_result, logger)


def emit_event_level_bundle(out_dir, partition, model_label,
                             y_true, y_prob, partition_records,
                             annotations_dir, mouse_metadata, logger,
                             order="min_then_refractory",
                             min_event_duration_sec=None,
                             max_event_duration_sec=None,
                             extra_summary_fields=None):
    """One-call convenience wrapper: per-mouse evaluation + 4-file bundle.

    Calls evaluate_event_level (which executes the post-processing
    pipeline at the locked operating point) then write_event_level_bundle
    to emit the four files. Returns the evaluator_result dict in case the
    caller wants to inspect or log values from it.
    """
    result = evaluate_event_level(
        partition_records, y_true, y_prob,
        annotations_dir, mouse_metadata, logger, order=order,
        min_event_duration_sec=min_event_duration_sec,
        max_event_duration_sec=max_event_duration_sec)
    # Resolve None defaults so the summary records the actual values used.
    eff_min = MIN_EVENT_SEC if min_event_duration_sec is None else min_event_duration_sec
    eff_max = MAX_EVENT_SEC if max_event_duration_sec is None else max_event_duration_sec
    write_event_level_bundle(
        out_dir, partition, model_label, result, logger,
        order=order,
        min_event_duration_sec=eff_min,
        max_event_duration_sec=eff_max,
        extra_summary_fields=extra_summary_fields)
    return result


def write_deploy_event_bundle(out_dir, stem, events,
                               recording_metadata, model_label, logger,
                               order="min_then_refractory",
                               min_event_duration_sec=25.0,
                               refractory_period_sec=30.0,
                               smoothing_window=3,
                               threshold=0.5,
                               step_sec=2.5,
                               max_event_duration_sec=120.0):
    """Two-file deploy bundle for inference contexts without ground truth.

        {stem}_event_details.csv  (canonical schema, GT cells empty)
        {stem}_summary.json       (metadata + post-processing params +
                                   predicted event counts; no event_level
                                   or per_mouse blocks since no GT)

    Parameters
    ----------
    out_dir : Path
    stem : str
        Filename prefix (typically the EDF stem); written into
        event_details.csv's mouse_id column for every row.
    events : list of dict
        Output of segment_predictions_to_events (or equivalent). Each
        event dict must carry start_sec, end_sec, duration_sec,
        mean_prob, max_prob, and may carry start_recording_time /
        end_recording_time for the elapsed-time columns (HH:MM:SS.s since
        recording start, overflow hours).
    recording_metadata : dict
        Must contain at least edf_path, recording_start_datetime
        (ISO 8601 string), recording_duration_sec.
    model_label : str
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stamp mouse_id (= stem) on every event before writing.
    deploy_events = []
    for evt in events:
        e = dict(evt)
        e.setdefault("mouse_id", stem)
        e.setdefault("is_true_alarm", "")          # unknown in deploy
        deploy_events.append(e)

    total_pred_duration = round(
        sum(float(e.get("duration_sec", 0.0)) for e in deploy_events), 4)

    summary = {
        "model":     model_label,
        "partition": "deploy",
        "timestamp": datetime.datetime.now().isoformat(),
        "recording": {
            "stem":                    stem,
            "edf_path":                str(recording_metadata.get("edf_path", "")),
            "recording_start_datetime": recording_metadata.get(
                "recording_start_datetime"),
            "recording_duration_sec":  recording_metadata.get(
                "recording_duration_sec"),
        },
        "post_processing_params": {
            "matching_rule":           "any-overlap",
            "order":                   order,
            "min_event_duration_sec":  min_event_duration_sec,
            "max_event_duration_sec":  max_event_duration_sec,
            "refractory_period_sec":   refractory_period_sec,
            "smoothing_window":        smoothing_window,
            "step_sec":                step_sec,
            "threshold":               threshold,
        },
        "predicted_events": {
            "n_predicted_events":                  len(deploy_events),
            "total_predicted_event_duration_sec":  total_pred_duration,
            "predicted_event_starts_recording_time": [
                e.get("start_recording_time", "") for e in deploy_events
            ],
        },
    }

    write_event_details_csv(
        out_dir / f"{stem}_event_details.csv",
        deploy_events, partition="deploy", logger=logger)
    write_summary_json(
        out_dir / f"{stem}_summary.json", summary, logger)


def emit_deploy_event_bundle(out_dir, stem, events,
                              recording_metadata, model_label, logger,
                              **post_proc_kwargs):
    """Alias for write_deploy_event_bundle to mirror emit_event_level_bundle
    naming. Same behaviour."""
    write_deploy_event_bundle(
        out_dir, stem, events, recording_metadata, model_label, logger,
        **post_proc_kwargs)

