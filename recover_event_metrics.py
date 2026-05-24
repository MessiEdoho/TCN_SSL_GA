"""
recover_event_metrics.py
========================
Local recovery pipeline that recomputes clinically-meaningful event-level
metrics for the M3 (MultiScaleTCN) val and test sets, using cached
predictions and per-mouse seizure annotations. Replaces the broken
event-level FAR/hr produced by the production pipeline (which concatenates
all-ictal-then-all-nonictal segments across mice and uses an inflated
denominator).

Pipeline (per partition)
------------------------
1. Per mouse, replay the preprocessing grid walk (compute_segment_grid +
   is_ictal) using EDF metadata + Excel annotations to assign each
   chronological grid position to the .npy filename it produced.
2. Reorder the cached y_true / y_prob arrays into per-mouse,
   chronologically-sorted blocks; surviving segments only (filtered
   segments leave gaps that the chunk logic respects).
3. Within each mouse, split surviving segments into contiguous chunks
   (consecutive chrono_idx). Each chunk is processed independently:
   smoothing window=3, threshold tau=0.5, run-detection, min-duration
   filter 25 s, refractory merge 30 s. The min-then-refractory order
   and 25 s minimum are the locked operating point (postproc_sweep.py;
   STUDY_REPORT.txt §7.6.10).
4. Match each predicted event against the mouse's Excel seizure_intervals
   using the any-overlap rule. Aggregate event TP / FP / FN per mouse,
   then across mice.
5. Compute event-level Precision, Recall, F1, FAR/hr, and mean detection
   latency. Write JSON + per-event CSV.

Inputs (all local)
------------------
  manifest          data_splits_nonictal_sampled_filtered.json
  metadata          mouse_recording_metadata.json
  annotations       {val,test}_seizure_annot_updated/{mouse}_xlsx.xlsx
  cached val NPZ    MultiScaleTCN/multiscale_tcn_predictions_raw.npz
  cached test NPZ   MultiScaleTCN/evaluation/multiscale_tcn_test_predictions_raw.npz

Outputs (under OUTPUT_ROOT)
---------------------------
  recover_event_metrics_val.json     event-level summary for val
  recover_event_metrics_test.json    event-level summary for test
  recover_event_details_val.csv      per-event detail rows for val
  recover_event_details_test.csv     per-event detail rows for test
  recover_event_metrics.log          persistent log
"""

import csv
import datetime
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report,
)

# Canonical 4-file output bundle is emitted via the shared writer in
# eval_utils so postproc_sweep / recover_event_metrics /
# m4_event_metrics_recovery / training+evaluation scripts all produce
# identically-named files with identical schemas. Imported under an
# aliased name to avoid clashing with the script's older inline writers.
from eval_utils import write_event_level_bundle as eval_utils_write_event_level_bundle


# ---------------------------------------------------------------------------
# Local paths
# ---------------------------------------------------------------------------
LOCAL_ROOT       = Path(r"C:\Users\messi\OneDrive\Desktop\Desktop\TCN_UNIQURE_PROJECT")
MANIFEST_PATH    = LOCAL_ROOT / "data_splits_nonictal_sampled_filtered.json"
METADATA_PATH    = LOCAL_ROOT / "mouse_recording_metadata.json"
ANNOT_DIRS       = {
    "val":  LOCAL_ROOT / "val_seizure_annot_updated",
    "test": LOCAL_ROOT / "test_seizure_annot_updated",
}
NPZ_PATHS        = {
    "val":  LOCAL_ROOT / "MultiScaleTCN" / "multiscale_tcn_predictions_raw.npz",
    "test": LOCAL_ROOT / "MultiScaleTCN" / "evaluation" / "multiscale_tcn_test_predictions_raw.npz",
}

OUTPUT_ROOT      = LOCAL_ROOT / "MultiScaleTCN" / "event_metrics_recovery"
LOG_PATH         = OUTPUT_ROOT / "recover_event_metrics.log"


# ---------------------------------------------------------------------------
# Constants (must match production preprocessing + post-processing)
# ---------------------------------------------------------------------------
FS                  = 500            # Hz
WIN_LEN             = 2500           # samples (= 5 s)
STEP                = 1250           # samples (= 2.5 s)
SEGMENT_SEC         = WIN_LEN / FS   # 5.0
STEP_SEC            = STEP / FS      # 2.5

THRESHOLD           = 0.5
SMOOTHING_WIN       = 3
REFRACTORY_SEC      = 30.0
MIN_EVENT_SEC       = 25.0  # locked operating point (Pareto-knee, postproc_sweep.py); STUDY_REPORT.txt 7.6.10

MODEL_NAME          = "MultiScaleTCN"


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------
def setup_logging():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("recover_event_metrics")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger


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
# build_chronology
# ---------------------------------------------------------------------------
def build_chronology(mouse_id, n_samples, seizure_intervals, logger):
    """Replay the preprocessing grid walk to produce a mapping from each
    .npy filename to its true chronological position in the recording.

    Returns
    -------
    fname_to_chrono : dict
        Keys are .npy basenames (e.g. 'm1_ictal_00001.npy'); values are
        dicts with keys 'chrono_idx', 't_start_sec', 'label' (1=ictal).
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
# load_partition_inputs
# ---------------------------------------------------------------------------
def load_partition_inputs(partition, manifest, metadata, logger):
    """Slice the manifest to one partition, load the cached NPZ, return
    a per-mouse dict of {mouse_id -> {manifest_indices, fnames, labels,
    npz_indices_in_partition}}.

    Each manifest record's position in splits[partition] is its index in
    the cached NPZ (because the eval scripts iterate the manifest in
    order with shuffle=False).
    """
    records = manifest.get(partition, [])
    if not records:
        logger.error("Partition '%s' is empty in manifest.", partition)
        sys.exit(1)

    npz = np.load(NPZ_PATHS[partition], allow_pickle=False)
    y_true_all = npz["y_true"].astype(np.int64)
    y_prob_all = npz["y_prob"].astype(np.float64)

    if len(y_true_all) != len(records):
        logger.error("NPZ length %d != manifest length %d for %s. "
                     "Check that the cached NPZ matches the current manifest.",
                     len(y_true_all), len(records), partition)
        sys.exit(1)

    per_mouse = {}
    for npz_idx, rec in enumerate(records):
        fname = Path(rec["filepath"]).name
        mouse = fname.split("_")[0]
        per_mouse.setdefault(mouse, {
            "fnames":      [],
            "labels":      [],
            "npz_indices": [],
        })
        per_mouse[mouse]["fnames"].append(fname)
        per_mouse[mouse]["labels"].append(int(rec["label"]))
        per_mouse[mouse]["npz_indices"].append(npz_idx)

    logger.info("Loaded %s NPZ: %d segments | %d unique mice",
                partition, len(y_true_all), len(per_mouse))

    expected = sorted(m for m, v in metadata.items() if partition in v["partitions"])
    found    = sorted(per_mouse.keys())
    if expected != found:
        logger.warning("Manifest mice for %s (%d) differ from metadata.partitions (%d). "
                       "Manifest extra: %s | Metadata extra: %s",
                       partition, len(found), len(expected),
                       sorted(set(found) - set(expected)),
                       sorted(set(expected) - set(found)))

    return per_mouse, y_true_all, y_prob_all


# ---------------------------------------------------------------------------
# reorder_to_chronology
# ---------------------------------------------------------------------------
def reorder_to_chronology(mouse_id, mouse_block, fname_to_chrono,
                          y_true_all, y_prob_all, logger):
    """Reorder one mouse's segments into chronological order. Returns
    parallel arrays sorted by chrono_idx (containing only surviving,
    matched-against-chronology segments).
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
# compute_segment_metrics
# ---------------------------------------------------------------------------
def compute_segment_metrics(y_true, y_pred, y_prob_for_auc, n_non_ictal_segs):
    """Same metric set as production compute_all_metrics. Reports BOTH the
    broken FAR/hr (denominator multiplied by segment_len_sec) and the
    corrected FAR/hr (multiplied by step_sec).
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

    accuracy = accuracy_score(y_true, y_pred)
    prec     = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall   = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1m      = f1_score(y_true, y_pred, average="macro", zero_division=0)
    spec     = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auroc    = roc_auc_score(y_true, y_prob_for_auc) if len(np.unique(y_true)) > 1 else 0.0
    ap       = average_precision_score(y_true, y_prob_for_auc) if len(np.unique(y_true)) > 1 else 0.0
    youden_j = recall + spec - 1.0

    non_ic_hrs_broken    = (n_non_ictal_segs * SEGMENT_SEC) / 3600.0
    non_ic_hrs_corrected = (n_non_ictal_segs * STEP_SEC)    / 3600.0
    far_seg_broken    = fp / non_ic_hrs_broken    if non_ic_hrs_broken    > 0 else 0.0
    far_seg_corrected = fp / non_ic_hrs_corrected if non_ic_hrs_corrected > 0 else 0.0

    return {
        "accuracy":          round(accuracy, 6),
        "precision":         round(prec, 6),
        "recall":            round(recall, 6),
        "sensitivity":       round(recall, 6),
        "specificity":       round(spec, 6),
        "youden_j":          round(youden_j, 6),
        "f1_macro":          round(f1m, 6),
        "auroc":             round(auroc, 6),
        "average_precision": round(ap, 6),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "far_per_hour_seg_BROKEN_5s_denom":    round(far_seg_broken, 6),
        "far_per_hour_seg_CORRECTED_2_5s_denom": round(far_seg_corrected, 6),
    }


# ---------------------------------------------------------------------------
# build_classification_report
# ---------------------------------------------------------------------------
def build_classification_report(y_true, y_pred, row_metrics):
    """Mirror of the production per-row classification report: sklearn's
    classification_report dict with target_names=['Non-ictal', 'Ictal'],
    augmented with auroc / prauc / specificity / FAR/hr fields so each
    file is self-contained.
    """
    report = classification_report(
        y_true, y_pred, target_names=["Non-ictal", "Ictal"],
        output_dict=True, zero_division=0)
    report["auroc"]                              = row_metrics.get("auroc")
    report["prauc"]                              = row_metrics.get("average_precision")
    report["specificity"]                        = row_metrics.get("specificity")
    report["sensitivity"]                        = row_metrics.get("sensitivity")
    report["far_per_hour_seg_BROKEN_5s_denom"]   = row_metrics.get("far_per_hour_seg_BROKEN_5s_denom")
    report["far_per_hour_seg_CORRECTED_2_5s_denom"] = row_metrics.get("far_per_hour_seg_CORRECTED_2_5s_denom")
    return report


# ---------------------------------------------------------------------------
# detect_events_in_chunk
# ---------------------------------------------------------------------------
def detect_events_in_chunk(t_start, y_prob, mouse_id, chunk_id):
    """Run smoothing + threshold + run-detection + min-duration + refractory
    on one chunk's prediction subarray (min-then-refractory order, locked
    by the Pareto-knee sweep -- see STUDY_REPORT.txt 7.6.10). Returns
    (events, smoothed_probs, smoothed_preds) so the caller can accumulate
    per-chunk smoothed arrays for Row 2 segment-level metrics.
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

    # min-duration filter applied BEFORE refractory merge so short artefacts
    # cannot be rescued by being merged with a neighbour.
    filtered = [
        evt for evt in raw_events
        if (evt["end_sec"] - evt["start_sec"]) >= MIN_EVENT_SEC
    ]

    merged = []
    for evt in filtered:
        if merged and (evt["start_sec"] - merged[-1]["end_sec"]) < REFRACTORY_SEC:
            merged[-1]["end_sec"] = evt["end_sec"]
            merged[-1]["seg_indices"].extend(evt["seg_indices"])
        else:
            merged.append({
                "start_sec":   evt["start_sec"],
                "end_sec":     evt["end_sec"],
                "seg_indices": list(evt["seg_indices"]),
            })

    final = []
    for evt in merged:
        duration = evt["end_sec"] - evt["start_sec"]
        seg_idx = evt["seg_indices"]
        final.append({
            "mouse_id":     mouse_id,
            "chunk_id":     chunk_id,
            "start_sec":    round(evt["start_sec"], 4),
            "end_sec":      round(evt["end_sec"], 4),
            "duration_sec": round(duration, 4),
            "mean_prob":    round(float(np.mean(smoothed[seg_idx])), 6),
            "max_prob":     round(float(np.max(smoothed[seg_idx])), 6),
        })
    return final, smoothed, preds


# ---------------------------------------------------------------------------
# match_events
# ---------------------------------------------------------------------------
def match_events(predicted_events, gt_intervals):
    """Match predicted events against ground-truth seizure_intervals using
    the any-overlap rule. Returns:
      tp -- list of (gt_idx, predicted_event) tuples (one per matched GT)
      fp -- list of predicted events with no GT overlap
      fn -- list of (gt_idx, gt_interval) for missed seizures
      latencies -- list of (matched_pred.start_sec - gt.start_sec) per TP
    A single predicted event may overlap multiple GT seizures; in that
    case each GT counts as a separate TP and the event contributes to
    none of them as FP.
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
# safe_div
# ---------------------------------------------------------------------------
def safe_div(num, den):
    return float(num) / float(den) if den > 0 else 0.0


# ---------------------------------------------------------------------------
# process_partition
# ---------------------------------------------------------------------------
def process_partition(partition, manifest, metadata, logger):
    logger.info("=" * 65)
    logger.info("Processing partition: %s", partition.upper())
    logger.info("=" * 65)

    per_mouse, y_true_all, y_prob_all = load_partition_inputs(
        partition, manifest, metadata, logger)
    annot_dir = ANNOT_DIRS[partition]

    per_mouse_results   = {}
    all_predicted       = []
    all_event_details   = []
    total_tp = total_fp = total_fn = 0
    total_n_gt = 0
    total_non_ictal_segs = 0
    total_latencies = []

    # Accumulators for Row 1 (raw) and Row 2 (post-processed) segment-level
    # metrics. All entries are in true per-mouse chronological order so the
    # AUROC/PRAUC/F1 computed at the end matches what the production pipeline
    # would report on the same predictions but is correctly aligned per mouse.
    seg_y_true_all = []
    seg_y_prob_raw = []
    seg_smoothed_probs = []
    seg_smoothed_preds = []

    for mouse_id in sorted(per_mouse.keys()):
        if mouse_id not in metadata:
            logger.error("  %s : no metadata record. Skipping.", mouse_id)
            continue

        meta = metadata[mouse_id]
        n_samples = int(meta["n_samples"])
        rec_start = datetime.datetime.fromisoformat(meta["recording_start_dt"])
        xlsx_path = annot_dir / f"{mouse_id}_xlsx.xlsx"
        if not xlsx_path.exists():
            logger.error("  %s : annotation Excel not found at %s. Skipping.",
                         mouse_id, xlsx_path)
            continue

        seizure_intervals = load_annotations(xlsx_path, rec_start)
        logger.info("  %s : %d ground-truth seizures from %s",
                    mouse_id, len(seizure_intervals), xlsx_path.name)

        fname_to_chrono, n_ictal_grid, n_nonictal_grid = build_chronology(
            mouse_id, n_samples, seizure_intervals, logger)

        ordered = reorder_to_chronology(
            mouse_id, per_mouse[mouse_id], fname_to_chrono,
            y_true_all, y_prob_all, logger)
        if ordered is None:
            continue

        chunks = split_into_chunks(ordered["chrono_idx"])
        logger.info("  %s : %d surviving segments in %d contiguous chunk(s)",
                    mouse_id, len(ordered["chrono_idx"]), len(chunks))

        mouse_predicted = []
        for chunk_id, (s, e) in enumerate(chunks):
            evts, smoothed_probs_chunk, smoothed_preds_chunk = detect_events_in_chunk(
                ordered["t_start_sec"][s:e], ordered["y_prob"][s:e],
                mouse_id, chunk_id)
            mouse_predicted.extend(evts)
            seg_y_true_all.append(ordered["y_true"][s:e])
            seg_y_prob_raw.append(ordered["y_prob"][s:e])
            seg_smoothed_probs.append(smoothed_probs_chunk)
            seg_smoothed_preds.append(smoothed_preds_chunk)

        tp, fp, fn, latencies = match_events(mouse_predicted, seizure_intervals)
        n_non_ictal = int(np.sum(ordered["y_true"] == 0))

        per_mouse_results[mouse_id] = {
            "n_ground_truth_seizures":     len(seizure_intervals),
            "n_predicted_events":          len(mouse_predicted),
            "tp":                          len(tp),
            "fp":                          len(fp),
            "fn":                          len(fn),
            "precision":                   round(safe_div(len(tp), len(tp) + len(fp)), 6),
            "recall":                      round(safe_div(len(tp), len(tp) + len(fn)), 6),
            "f1":                          round(safe_div(2 * len(tp),
                                                          2 * len(tp) + len(fp) + len(fn)), 6),
            "non_ictal_segments":          n_non_ictal,
            "non_ictal_hours":             round(n_non_ictal * STEP_SEC / 3600.0, 4),
            "far_per_hour":                round(safe_div(len(fp),
                                                          n_non_ictal * STEP_SEC / 3600.0), 6),
            "mean_detection_latency_sec":  (round(float(np.mean(latencies)), 4)
                                            if latencies else None),
            "n_chunks":                    len(chunks),
        }

        # Format seconds-from-recording-start as HHhMMmSS.s with overflow
        # hours (e.g. 73h14m22.5). Matches eval_utils._to_hms.
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
                "partition":                   partition,
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
                "partition":                   partition,
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

    # Concatenate per-mouse-per-chunk arrays for partition-level segment metrics.
    seg_y_true_concat   = np.concatenate(seg_y_true_all)     if seg_y_true_all     else np.array([], dtype=np.int64)
    seg_y_prob_concat   = np.concatenate(seg_y_prob_raw)     if seg_y_prob_raw     else np.array([], dtype=np.float64)
    seg_smoothed_probs_concat = np.concatenate(seg_smoothed_probs) if seg_smoothed_probs else np.array([], dtype=np.float64)
    seg_smoothed_preds_concat = np.concatenate(seg_smoothed_preds) if seg_smoothed_preds else np.array([], dtype=np.int64)
    n_non_ic_concat = int(np.sum(seg_y_true_concat == 0))

    row1_pred = (seg_y_prob_concat >= THRESHOLD).astype(np.int64)
    row1_metrics = compute_segment_metrics(
        seg_y_true_concat, row1_pred, seg_y_prob_concat, n_non_ic_concat)
    row1_metrics["threshold"]      = THRESHOLD
    row1_metrics["postprocessed"]  = False

    row2_metrics = compute_segment_metrics(
        seg_y_true_concat, seg_smoothed_preds_concat, seg_smoothed_probs_concat,
        n_non_ic_concat)
    row2_metrics["threshold"]      = THRESHOLD
    row2_metrics["postprocessed"]  = True

    summary = {
        "model":     MODEL_NAME,
        "partition": partition,
        "timestamp": datetime.datetime.now().isoformat(),
        "post_processing": {
            "threshold":              THRESHOLD,
            "smoothing_window":       SMOOTHING_WIN,
            "refractory_period_sec":  REFRACTORY_SEC,
            "min_event_duration_sec": MIN_EVENT_SEC,
            "matching_rule":          "any-overlap",
        },
        "totals": {
            "n_mice":                       len(per_mouse_results),
            "n_segments_in_npz":             int(len(y_true_all)),
            "n_segments_in_chronology_used": int(len(seg_y_true_concat)),
            "n_ground_truth_seizures":       int(total_n_gt),
            "n_predicted_events":            int(len(all_predicted)),
            "non_ictal_segments_corrected":  int(total_non_ictal_segs),
            "non_ictal_hours_corrected":     round(total_non_ictal_hours, 4),
        },
        "segment_level_metrics": {
            "row1_raw_threshold_0_5":         row1_metrics,
            "row2_postproc_threshold_0_5":    row2_metrics,
            "note": ("Per-segment classification metrics (accuracy / precision / "
                     "recall / specificity / F1 / AUROC / AP / TP / FP / FN / TN) "
                     "are order-invariant and match the production eval logs. "
                     "FAR/hr is reported in BOTH the broken form (denominator "
                     "× segment_len_sec=5.0, matching what the production "
                     "pipeline wrote) and the corrected form (× step_sec=2.5, "
                     "the unique recording time each segment contributes under "
                     "50%% overlap). Multiply broken by 0.5 to recover corrected."),
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
            "mean_detection_latency_sec":  (round(float(np.mean(total_latencies)), 4)
                                            if total_latencies else None),
        },
        "per_mouse": per_mouse_results,
    }

    # Emit the canonical 4-file bundle via the shared writer in eval_utils
    # so postproc_sweep, recover_event_metrics, m4_event_metrics_recovery,
    # the training scripts, and the evaluation scripts all produce
    # identically-named files with identical schemas.
    evaluator_result = {
        "totals":                summary["totals"],
        "event_level_metrics":   summary["event_level_metrics"],
        "segment_level_metrics": summary["segment_level_metrics"],
        "per_mouse_results":     per_mouse_results,
        "all_event_details":     all_event_details,
        "reordered_arrays": {
            "y_true":      seg_y_true_concat,
            "y_pred_row1": row1_pred,
            "y_pred_row2": seg_smoothed_preds_concat,
            "y_prob":      seg_y_prob_concat,
        },
    }
    eval_utils_write_event_level_bundle(
        OUTPUT_ROOT, partition, MODEL_NAME, evaluator_result, logger,
        order="min_then_refractory",
        min_event_duration_sec=MIN_EVENT_SEC,
        refractory_period_sec=REFRACTORY_SEC,
        smoothing_window=SMOOTHING_WIN,
        threshold=THRESHOLD,
        step_sec=STEP_SEC,
    )

    logger.info("-" * 65)
    em = summary["event_level_metrics"]
    tot = summary["totals"]
    logger.info("PARTITION %s SUMMARY", partition.upper())
    logger.info("  Mice                  : %d", tot["n_mice"])
    logger.info("  GT seizures           : %d", tot["n_ground_truth_seizures"])
    logger.info("  Predicted events      : %d", tot["n_predicted_events"])
    r1 = summary["segment_level_metrics"]["row1_raw_threshold_0_5"]
    r2 = summary["segment_level_metrics"]["row2_postproc_threshold_0_5"]
    logger.info("  --- Segment-level (Row 1: raw @ 0.5) ---")
    logger.info("    Accuracy / Precision / Recall / Specificity : "
                "%.4f / %.4f / %.4f / %.4f",
                r1["accuracy"], r1["precision"], r1["recall"], r1["specificity"])
    logger.info("    F1 macro / AUROC / AP : %.4f / %.4f / %.4f",
                r1["f1_macro"], r1["auroc"], r1["average_precision"])
    logger.info("    FAR/hr seg BROKEN (×5s)    : %.4f", r1["far_per_hour_seg_BROKEN_5s_denom"])
    logger.info("    FAR/hr seg CORRECTED (×2.5s): %.4f", r1["far_per_hour_seg_CORRECTED_2_5s_denom"])
    logger.info("  --- Segment-level (Row 2: post-processed @ 0.5) ---")
    logger.info("    Accuracy / Precision / Recall / Specificity : "
                "%.4f / %.4f / %.4f / %.4f",
                r2["accuracy"], r2["precision"], r2["recall"], r2["specificity"])
    logger.info("    F1 macro / AUROC / AP : %.4f / %.4f / %.4f",
                r2["f1_macro"], r2["auroc"], r2["average_precision"])
    logger.info("    FAR/hr seg BROKEN (×5s)    : %.4f", r2["far_per_hour_seg_BROKEN_5s_denom"])
    logger.info("    FAR/hr seg CORRECTED (×2.5s): %.4f", r2["far_per_hour_seg_CORRECTED_2_5s_denom"])
    logger.info("  --- Event-level (this is what was previously broken) ---")
    logger.info("    TP / FP / FN          : %d / %d / %d", em["tp"], em["fp"], em["fn"])
    logger.info("    Precision             : %.4f", em["precision"])
    logger.info("    Recall (sensitivity)  : %.4f", em["recall"])
    logger.info("    F1                    : %.4f", em["f1"])
    logger.info("    FAR/hr event CORRECTED: %.4f  (non-ictal hours = %.1f)",
                em["far_per_hour_event_CORRECTED"], tot["non_ictal_hours_corrected"])
    logger.info("    Mean detection latency: %s s",
                em["mean_detection_latency_sec"])
    logger.info("-" * 65)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    logger = setup_logging()
    logger.info("=" * 65)
    logger.info("recover_event_metrics.py")
    logger.info("Local root      : %s", LOCAL_ROOT)
    logger.info("Manifest        : %s", MANIFEST_PATH)
    logger.info("Metadata        : %s", METADATA_PATH)
    logger.info("Output          : %s", OUTPUT_ROOT)
    logger.info("Log             : %s", LOG_PATH)
    logger.info("=" * 65)

    for p in [MANIFEST_PATH, METADATA_PATH] + list(NPZ_PATHS.values()):
        if not p.exists():
            logger.error("Required input missing: %s", p); sys.exit(1)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    for partition in ("val", "test"):
        process_partition(partition, manifest, metadata, logger)

    logger.info("=" * 65)
    logger.info("DONE")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
