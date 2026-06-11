"""
false_negative_audit.py
=======================
Per-mouse audit of FALSE NEGATIVES (GT seizures the model missed) for
M3 or M4, on val / test / full-train. CPU-only, no model forward; reads
the variant's cached predictions NPZ + the Excel annotations and writes
one diagnostic CSV row per missed seizure.

Why this exists
---------------
The canonical event_details_row2.csv schema is "one row per predicted
event". An FN is a GT seizure with no matching predicted event, so it
has no entry in that CSV. This script re-derives the FN set at runtime
(annotations + match_events_to_ground_truth) and attaches diagnostic
fields explaining each miss.

Per-FN diagnostic fields
------------------------
  mouse_id, gt_idx
  gt_start_sec, gt_end_sec, gt_duration_sec
  gt_start_hms, gt_end_hms
  n_segments_in_gt
  max_prob_raw, mean_prob_raw                  raw model output inside GT
  max_prob_smoothed, mean_prob_smoothed        post W=3 moving average
  frac_above_0.5_raw, frac_above_0.5_smoothed  fraction of segments crossing 0.5
  longest_pred_run_sec                         longest contiguous smoothed_pred=1
                                               run inside GT (sec)
  brief_firing_dropped_by_min_dur              True iff 0 < longest_pred_run_sec
                                               < MIN_EVENT_SEC
  nearest_pred_event_distance_sec              distance to closest TP or FP event
  nearest_pred_event_label                     "TP" / "FP" / "none"
  near_recording_edge                          True iff GT is within 30 s of
                                               recording start or end
  crosses_filter_gap                           True iff the GT interval spans a
                                               filter-gap break in the chronology
  diagnosis_hint                               one of:
       short_gt_below_minimum
       edge_of_recording
       near_filter_gap
       brief_firing_dropped_by_min_dur
       subthreshold_signal
       non_contiguous_above_threshold
       other

CLI
---
    python false_negative_audit.py --variant MultiScaleTCN              --partition val
    python false_negative_audit.py --variant MultiScaleTCN              --partition test
    python false_negative_audit.py --variant MultiScaleTCN              --partition train
    python false_negative_audit.py --variant MultiScaleTCNWithAttention --partition val
    ...

Outputs (under <variant>/<evaluation or full_train_evaluation>/)
---------------------------------------------------------------
  false_negative_audit_<partition>.csv
  false_negative_audit_<partition>.json   summary (counts per diagnosis bucket)
  false_negative_audit_<partition>.log
"""

import argparse
import csv
import datetime
import json
import logging
import sys
from pathlib import Path

import numpy as np

from eval_utils import (
    FS, WIN_LEN, STEP, SEGMENT_SEC, STEP_SEC,
    THRESHOLD, SMOOTHING_WIN, MIN_EVENT_SEC, MAX_EVENT_SEC, REFRACTORY_SEC,
    load_annotations,
    detect_events_in_chunk,
    match_events_to_ground_truth,
    split_into_chunks,
)
from postproc_sweep import load_records_and_arrays


# ---------------------------------------------------------------------------
# Variant + partition path resolution
# ---------------------------------------------------------------------------
CLUSTER_OUTPUT  = Path("/home/people/22206468/scratch/OUTPUT")
CLUSTER_SCRATCH = Path("/home/people/22206468/scratch")
ANNOT_DIR       = CLUSTER_SCRATCH / "seizure_times_updated"
METADATA_PATH   = CLUSTER_SCRATCH / "INPUT_DATA" / "Data_diagnostic" / "mouse_recording_metadata.json"
MANIFEST_STD    = CLUSTER_SCRATCH / "INPUT_DATA" / "data_splits_outputs" / "data_splits_nonictal_sampled_filtered_enriched.json"
MANIFEST_TRAIN  = CLUSTER_SCRATCH / "INPUT_DATA" / "data_splits_outputs" / "data_splits_full_train_enriched.json"

NEAR_EDGE_SEC   = 30.0     # GT within this distance of recording start/end is "edge"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True,
                   choices=["MultiScaleTCN", "MultiScaleTCNWithAttention"])
    p.add_argument("--partition", required=True,
                   choices=["val", "test", "train"])
    p.add_argument("--min-event-sec", type=float, default=MIN_EVENT_SEC,
                   help=f"Min event duration in seconds (default: {MIN_EVENT_SEC}). "
                        f"Match the postproc-sweep cell you want to audit.")
    p.add_argument("--max-event-sec", type=float, default=MAX_EVENT_SEC,
                   help=f"Max event duration in seconds (default: {MAX_EVENT_SEC}). "
                        f"Events longer than this are dropped before matching.")
    p.add_argument("--order", choices=["min_then_refractory", "refractory_then_min"],
                   default="min_then_refractory",
                   help="Post-processing order. Default: min_then_refractory "
                        "(canonical).")
    return p.parse_args()


def variant_paths(variant, partition):
    if variant == "MultiScaleTCN":
        base = CLUSTER_OUTPUT / "MODEL3_OUTPUT" / "MultiScaleTCN"
        npz_map = {
            "val":   base / "multiscale_tcn_predictions_raw.npz",
            "test":  base / "evaluation" / "multiscale_tcn_test_predictions_raw.npz",
            "train": base / "full_train_evaluation" / "multiscale_tcn_full_train_predictions_full.npz",
        }
    else:   # MultiScaleTCNWithAttention
        base = CLUSTER_OUTPUT / "MODEL4_OUTPUT" / "MultiScaleTCNAttention"
        npz_map = {
            "val":   base / "val_event_metrics" / "ms_attn_val_predictions_full.npz",
            "test":  base / "evaluation" / "ms_attn_test_predictions_full.npz",
            "train": base / "full_train_evaluation" / "ms_attn_full_train_predictions_full.npz",
        }
    out_dir_map = {
        "val":   base / "evaluation",
        "test":  base / "evaluation",
        "train": base / "full_train_evaluation",
    }
    manifest_map = {
        "val":   MANIFEST_STD,
        "test":  MANIFEST_STD,
        "train": MANIFEST_TRAIN,
    }
    return {
        "npz":      npz_map[partition],
        "out_dir":  out_dir_map[partition],
        "manifest": manifest_map[partition],
    }


# ---------------------------------------------------------------------------
def _op_suffix(min_event_sec, max_event_sec, order):
    """Return a filename suffix encoding the operating point so audits at
    different sweep points don't clobber each other."""
    return (f"min{int(round(min_event_sec))}s"
            f"_max{int(round(max_event_sec))}s"
            f"_{order}")


def setup_logging(out_dir, partition, op_suffix):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"false_negative_audit_{partition}_{op_suffix}.log"
    logger = logging.getLogger("false_negative_audit")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8"); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger, log_path


def _to_hms(sec):
    sec = float(sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec - h * 3600 - m * 60
    return f"{h:02d}h{m:02d}m{s:04.1f}"


def longest_consecutive_run(binary):
    """Return the longest run of 1s in a binary numpy array."""
    if binary.size == 0:
        return 0
    runs = []
    cur = 0
    for v in binary:
        if v == 1:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    return max(runs) if runs else 0


# ---------------------------------------------------------------------------
def diagnose(row, min_event_sec):
    """Pick the most actionable diagnosis bucket for one FN. min_event_sec
    is the operating-point value used for this audit run (not the module
    constant) so the 'short_gt_below_minimum' bucket reflects the actual
    threshold applied."""
    if row["gt_duration_sec"] < min_event_sec:
        return "short_gt_below_minimum"
    if row["near_recording_edge"]:
        return "edge_of_recording"
    if row["crosses_filter_gap"]:
        return "near_filter_gap"
    if row["brief_firing_dropped_by_min_dur"]:
        return "brief_firing_dropped_by_min_dur"
    if row["max_prob_raw"] < 0.5:
        return "subthreshold_signal"
    # Raw probability did exceed 0.5 somewhere in the GT, but the smoothed
    # path never produced a contiguous run -- the threshold crossings were
    # too sparse for the moving average to hold them together.
    if row["longest_pred_run_sec"] == 0.0:
        return "non_contiguous_above_threshold"
    return "other"


# ---------------------------------------------------------------------------
def audit_partition(variant, partition, min_event_sec, max_event_sec, order, logger):
    paths = variant_paths(variant, partition)
    logger.info("Variant       : %s", variant)
    logger.info("Partition     : %s", partition)
    logger.info("Order         : %s", order)
    logger.info("MIN_EVENT_SEC : %.1f s", min_event_sec)
    logger.info("MAX_EVENT_SEC : %.1f s", max_event_sec)
    logger.info("NPZ           : %s", paths["npz"])
    logger.info("Manifest      : %s", paths["manifest"])
    logger.info("Out dir       : %s", paths["out_dir"])
    logger.info("Annot dir     : %s", ANNOT_DIR)
    logger.info("Metadata      : %s", METADATA_PATH)

    if not METADATA_PATH.exists():
        logger.error("Mouse metadata missing: %s", METADATA_PATH); sys.exit(1)
    mouse_metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    if not ANNOT_DIR.exists():
        logger.error("Annotations dir missing: %s", ANNOT_DIR); sys.exit(1)
    if not paths["npz"].exists():
        logger.error("Predictions NPZ missing: %s", paths["npz"]); sys.exit(1)

    # Load NPZ (handles _full and _raw formats transparently via postproc_sweep)
    y_true_all, y_prob_all, records = load_records_and_arrays(
        paths["npz"], paths["manifest"], partition, ANNOT_DIR,
        mouse_metadata, logger)

    # Group records by mouse and preserve original index
    per_mouse = {}
    for i, rec in enumerate(records):
        per_mouse.setdefault(rec["mouse_id"], []).append((i, rec))

    fn_rows = []
    n_total_gt = 0
    n_total_fn = 0
    for mouse_id in sorted(per_mouse):
        rec_list = per_mouse[mouse_id]
        if mouse_id not in mouse_metadata:
            logger.warning("  %s : no metadata, skipping.", mouse_id); continue
        meta = mouse_metadata[mouse_id]
        xlsx = ANNOT_DIR / f"{mouse_id}_xlsx.xlsx"
        if not xlsx.exists():
            logger.warning("  %s : annotation %s missing, skipping.", mouse_id, xlsx); continue

        rec_start = datetime.datetime.fromisoformat(meta["recording_start_dt"])
        recording_duration_sec = float(meta["n_samples"]) / float(meta["fs_hz"])
        gt_intervals = load_annotations(xlsx, rec_start)
        n_total_gt += len(gt_intervals)

        # Chronological reorder
        idxs   = np.array([i for i, _ in rec_list], dtype=np.int64)
        cidxs  = np.array([r["chrono_idx"]  for _, r in rec_list], dtype=np.int64)
        starts = np.array([r["t_start_sec"] for _, r in rec_list], dtype=np.float64)
        order  = np.argsort(cidxs)
        chrono_idx_arr = cidxs[order]
        t_start_arr    = starts[order]
        y_true_mouse   = y_true_all[idxs[order]]
        y_prob_mouse   = y_prob_all[idxs[order]]

        chunks = split_into_chunks(chrono_idx_arr)

        # Post-process to get predicted events + smoothed probs/preds aligned
        # with t_start_arr (chunk-aware: chunks each get their own smoothing
        # so gaps in the chronology don't bleed across).
        smoothed_probs = np.empty(len(chrono_idx_arr), dtype=np.float64)
        smoothed_preds = np.empty(len(chrono_idx_arr), dtype=np.int64)
        predicted = []
        for chunk_id, (s, e) in enumerate(chunks):
            evts, sp_chunk, spred_chunk = detect_events_in_chunk(
                t_start_arr[s:e], y_prob_mouse[s:e], mouse_id, chunk_id,
                order=order,
                min_event_duration_sec=min_event_sec,
                max_event_duration_sec=max_event_sec)
            smoothed_probs[s:e] = sp_chunk
            smoothed_preds[s:e] = spred_chunk
            predicted.extend(evts)

        tp_pairs, fp_list, fn_list, _lat = match_events_to_ground_truth(
            predicted, gt_intervals)

        # Build set of predicted-event start_sec for "nearest pred event"
        # plus a TP/FP label lookup.
        pred_label = []     # list of (start_sec, end_sec, "TP"/"FP")
        tp_pred_set = {id(p) for _, p in tp_pairs}
        for p in predicted:
            lbl = "TP" if id(p) in tp_pred_set else "FP"
            pred_label.append((float(p["start_sec"]), float(p["end_sec"]), lbl))

        # Filter-gap chunk boundaries on the absolute time axis
        chunk_breaks = []
        for cid, (s, e) in enumerate(chunks):
            if cid == 0:
                continue
            chunk_breaks.append(float(t_start_arr[s]))   # start of new chunk

        for gt_idx, (gs, ge) in fn_list:
            in_gt = (t_start_arr >= gs - SEGMENT_SEC) & (t_start_arr < ge)
            n_in_gt = int(in_gt.sum())
            if n_in_gt == 0:
                # Pathological: GT lies entirely in a filter gap.
                row = {
                    "mouse_id":                          mouse_id,
                    "gt_idx":                            int(gt_idx),
                    "gt_start_sec":                      round(float(gs), 4),
                    "gt_end_sec":                        round(float(ge), 4),
                    "gt_duration_sec":                   round(float(ge - gs), 4),
                    "gt_start_hms":                      _to_hms(gs),
                    "gt_end_hms":                        _to_hms(ge),
                    "n_segments_in_gt":                  0,
                    "max_prob_raw":                      None,
                    "mean_prob_raw":                     None,
                    "max_prob_smoothed":                 None,
                    "mean_prob_smoothed":                None,
                    "frac_above_0.5_raw":                None,
                    "frac_above_0.5_smoothed":           None,
                    "longest_pred_run_sec":              0.0,
                    "brief_firing_dropped_by_min_dur":   False,
                    "nearest_pred_event_distance_sec":   None,
                    "nearest_pred_event_label":          "none",
                    "near_recording_edge":               (gs < NEAR_EDGE_SEC or
                                                          (recording_duration_sec - ge) < NEAR_EDGE_SEC),
                    "crosses_filter_gap":                True,
                    "diagnosis_hint":                    "near_filter_gap",
                }
                # near_filter_gap is set directly; no need to consult diagnose()
                fn_rows.append(row); continue

            raw_in_gt = y_prob_mouse[in_gt]
            sm_in_gt  = smoothed_probs[in_gt]
            pred_in_gt = smoothed_preds[in_gt].astype(np.int64)

            longest_run_segs = longest_consecutive_run(pred_in_gt)
            longest_run_sec  = float(longest_run_segs) * STEP_SEC

            # Crosses a filter gap?
            gt_crosses_gap = any(gs < cb < ge for cb in chunk_breaks)

            # Nearest predicted event by start_sec
            if pred_label:
                dists = [(min(abs(ps - gs), abs(pe - ge), abs(ps - ge), abs(pe - gs)), lbl)
                         for ps, pe, lbl in pred_label]
                nearest_dist, nearest_lbl = min(dists, key=lambda x: x[0])
            else:
                nearest_dist, nearest_lbl = None, "none"

            row = {
                "mouse_id":                          mouse_id,
                "gt_idx":                            int(gt_idx),
                "gt_start_sec":                      round(float(gs), 4),
                "gt_end_sec":                        round(float(ge), 4),
                "gt_duration_sec":                   round(float(ge - gs), 4),
                "gt_start_hms":                      _to_hms(gs),
                "gt_end_hms":                        _to_hms(ge),
                "n_segments_in_gt":                  n_in_gt,
                "max_prob_raw":                      round(float(raw_in_gt.max()), 6),
                "mean_prob_raw":                     round(float(raw_in_gt.mean()), 6),
                "max_prob_smoothed":                 round(float(sm_in_gt.max()), 6),
                "mean_prob_smoothed":                round(float(sm_in_gt.mean()), 6),
                "frac_above_0.5_raw":                round(float((raw_in_gt >= THRESHOLD).mean()), 6),
                "frac_above_0.5_smoothed":           round(float((sm_in_gt >= THRESHOLD).mean()), 6),
                "longest_pred_run_sec":              round(longest_run_sec, 4),
                "brief_firing_dropped_by_min_dur":   bool(0 < longest_run_sec < min_event_sec),
                "nearest_pred_event_distance_sec":   None if nearest_dist is None else round(float(nearest_dist), 4),
                "nearest_pred_event_label":          nearest_lbl,
                "near_recording_edge":               (gs < NEAR_EDGE_SEC or
                                                      (recording_duration_sec - ge) < NEAR_EDGE_SEC),
                "crosses_filter_gap":                bool(gt_crosses_gap),
            }
            row["diagnosis_hint"] = diagnose(row, min_event_sec)
            fn_rows.append(row)

        n_total_fn += len(fn_list)
        logger.info("  %-8s : GT=%d | predicted=%d | TP-pairs=%d | FP=%d | FN=%d",
                    mouse_id, len(gt_intervals), len(predicted),
                    len(tp_pairs), len(fp_list), len(fn_list))

    logger.info("-" * 65)
    logger.info("Total GT seizures across partition : %d", n_total_gt)
    logger.info("Total FN seizures across partition : %d", n_total_fn)
    return fn_rows


# ---------------------------------------------------------------------------
def write_csv(out_path, rows, logger):
    fieldnames = [
        "mouse_id", "gt_idx",
        "gt_start_sec", "gt_end_sec", "gt_duration_sec",
        "gt_start_hms", "gt_end_hms",
        "n_segments_in_gt",
        "max_prob_raw", "mean_prob_raw",
        "max_prob_smoothed", "mean_prob_smoothed",
        "frac_above_0.5_raw", "frac_above_0.5_smoothed",
        "longest_pred_run_sec", "brief_firing_dropped_by_min_dur",
        "nearest_pred_event_distance_sec", "nearest_pred_event_label",
        "near_recording_edge", "crosses_filter_gap",
        "diagnosis_hint",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    logger.info("Saved FN audit CSV: %s (%d rows)", out_path, len(rows))


def write_summary(out_path, rows, variant, partition,
                  min_event_sec, max_event_sec, order, logger):
    buckets = {}
    per_mouse = {}
    for r in rows:
        buckets[r["diagnosis_hint"]] = buckets.get(r["diagnosis_hint"], 0) + 1
        per_mouse.setdefault(r["mouse_id"], 0)
        per_mouse[r["mouse_id"]] += 1
    payload = {
        "variant":           variant,
        "partition":         partition,
        "n_fn_total":        len(rows),
        "diagnosis_buckets": buckets,
        "fns_per_mouse":     per_mouse,
        "thresholds_used": {
            "min_event_sec":   min_event_sec,
            "max_event_sec":   max_event_sec,
            "order":           order,
            "SMOOTHING_WIN":   SMOOTHING_WIN,
            "THRESHOLD":       THRESHOLD,
            "REFRACTORY_SEC":  REFRACTORY_SEC,
            "NEAR_EDGE_SEC":   NEAR_EDGE_SEC,
            "STEP_SEC":        STEP_SEC,
            "SEGMENT_SEC":     SEGMENT_SEC,
        },
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Saved FN audit summary: %s", out_path)
    logger.info("Per-bucket counts:")
    for k, v in sorted(buckets.items(), key=lambda kv: -kv[1]):
        logger.info("  %-40s : %d", k, v)


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    paths = variant_paths(args.variant, args.partition)
    suffix = _op_suffix(args.min_event_sec, args.max_event_sec, args.order)
    logger, log_path = setup_logging(paths["out_dir"], args.partition, suffix)

    logger.info("=" * 65)
    logger.info("false_negative_audit.py")
    logger.info("Log file : %s", log_path)
    logger.info("=" * 65)

    rows = audit_partition(
        args.variant, args.partition,
        args.min_event_sec, args.max_event_sec, args.order, logger)
    csv_path  = paths["out_dir"] / f"false_negative_audit_{args.partition}_{suffix}.csv"
    json_path = paths["out_dir"] / f"false_negative_audit_{args.partition}_{suffix}.json"
    write_csv(csv_path, rows, logger)
    write_summary(json_path, rows, args.variant, args.partition,
                  args.min_event_sec, args.max_event_sec, args.order, logger)
    logger.info("=" * 65)
    logger.info("DONE")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
