"""
event_attention_saliency.py
===========================
Event-level temporal-attention saliency for M4 (MultiScaleTCNWithAttention).

For a given partition (val or test), produces three mean +- SEM saliency
curves -- one each for true positives, false positives, and false negatives
-- aligned at event onset and truncated to MIN_EVENT_SEC seconds.

Pipeline
--------
  1. Read the canonical event_details_row2.csv (post-processed events):
       - rows with is_true_alarm=True   -> TP candidates
       - rows with is_true_alarm=False  -> FP candidates
     CSV path:
       val  : OUTPUT_ROOT/val_event_details.csv
       test : OUTPUT_ROOT/evaluation/test_event_details.csv

  2. Per mouse, re-derive FNs via match_events_to_ground_truth: GT seizures
     with no overlapping predicted event. Not stored in event_details.csv.

  3. Per mouse, identify the small subset of windows in the enriched
     manifest whose [t_start, t_start+5s) span overlaps any event interval.
     Run only those windows through the model on GPU (typical: ~20k
     windows for the full test set vs 9.75M total -- ~0.2 %).

  4. Per event, run an overlap-add of the per-window attention vectors onto
     a sample-resolution local timeline of length MIN_EVENT_SEC*FS=12500:
         S[t] += alpha[k]
         C[t] += 1                              for each (window, position)
         saliency_continuous[t] = S[t] / max(C[t], 1)
     With STEP=1250 and WIN_LEN=2500 (50 % overlap) each interior sample
     receives contributions from two windows; division by C is the mean,
     not a sum, so the overlapping contributions are not double counted.

  5. Pool crops across mice within each class. Aggregate to mean +- SEM
     per sample. Save figure, NPZ of raw crops, summary JSON.

Memory pattern: streaming per mouse. Peak working set per mouse is the
attention cache for that mouse's relevant windows (~few MB) plus the
running list of per-class crops (~MB). No bulk materialisation.

Attention temporal resolution
-----------------------------
MultiScaleTCN backbone uses dilated causal convolutions only (no stride,
no pooling). Fusion is a 1x1 conv. So the attention vector length T
equals the input window length 2500 -- attention is already at raw EEG
sample resolution. No upsampling required.

Inputs
------
  --partition {val, test}

Outputs (under OUTPUT_ROOT/saliency_interprete/{partition}/)
  event_attention_saliency.log
  event_saliency_curves.png        TP / FP / FN mean +- SEM, onset aligned
  tp_fp_fn_difference.png          (TP-FP) and (TP-FN) difference curves
  per_event_saliency.npz           stacked crops per class + metadata
  event_saliency_summary.json      counts, peak time/amplitude per class
"""

import argparse
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_utils import (
    SafeEEGSegmentDataset,
    FS, WIN_LEN, STEP, SEGMENT_SEC, STEP_SEC,
    MIN_EVENT_SEC, REFRACTORY_SEC, SMOOTHING_WIN, THRESHOLD,
    load_annotations,
    match_events_to_ground_truth,
)
from tcn_utils import MultiScaleTCNWithAttention


# ---------------------------------------------------------------------------
# Paths -- mirror MultiScaleTCNAttention.py
# ---------------------------------------------------------------------------
OUTPUT_ROOT          = Path("/home/people/22206468/scratch/OUTPUT/MODEL4_OUTPUT") / "MultiScaleTCNAttention"
EVALUATION_DIR       = OUTPUT_ROOT / "evaluation"
WEIGHTS_PATH         = OUTPUT_ROOT / "ms_attn_final_weights.pt"
BACKBONE_PARAMS_PATH = Path("/home/people/22206468/scratch/OUTPUT/MODEL3_OUTPUT/MultiScaleTCNtuning_outputs") / "best_multiscale_params.json"
ATTN_PARAMS_PATH     = Path("/home/people/22206468/scratch/OUTPUT/MODEL4_OUTPUT/multiscale_attention_tuning_outputs") / "best_multiscale_attn_params.json"
SPLITS_PATH          = Path("/scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits_nonictal_sampled_filtered_enriched.json")
ANNOT_DIR            = Path("/home/people/22206468/scratch/seizure_times_updated")
MOUSE_METADATA_PATH  = Path("/home/people/22206468/scratch/INPUT_DATA/Data_diagnostic/mouse_recording_metadata.json")

SALIENCY_ROOT        = OUTPUT_ROOT / "saliency_interprete"

DEFAULT_BRANCH = {
    "branch1": [1, 2, 4],
    "branch2": [8, 16, 32],
    "branch3": [32, 64, 128],
}

CROP_LEN_SAMPLES = int(MIN_EVENT_SEC * FS)   # 25.0 s * 500 Hz = 12500


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--partition", choices=["val", "test"], required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def event_details_path_for(partition):
    if partition == "val":
        return OUTPUT_ROOT / "val_event_details.csv"
    return EVALUATION_DIR / "test_event_details.csv"


def setup_logging(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "event_attention_saliency.log"
    logger = logging.getLogger("event_attn_saliency")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8"); fh.setFormatter(fmt); logger.addHandler(fh)
    logger.info("Saliency log file: %s", log_file)
    return logger


# ---------------------------------------------------------------------------
def build_model(device, logger):
    with open(BACKBONE_PARAMS_PATH, "r", encoding="utf-8") as f:
        bb_cfg = json.load(f)
    backbone_hp = bb_cfg["hyperparameters"]
    branch_dilations = bb_cfg.get("branch_dilations", DEFAULT_BRANCH)

    with open(ATTN_PARAMS_PATH, "r", encoding="utf-8") as f:
        attn_hp = json.load(f)["hyperparameters"]

    model = MultiScaleTCNWithAttention(
        num_filters       =int(backbone_hp["num_filters"]),
        kernel_size       =int(backbone_hp["kernel_size"]),
        dropout           =float(backbone_hp["dropout"]),
        fusion            =str(backbone_hp["fusion"]),
        attention_dim     =int(attn_hp["attention_dim"]),
        attention_dropout =float(attn_hp["attention_dropout"]),
        branch1_dilations =branch_dilations["branch1"],
        branch2_dilations =branch_dilations["branch2"],
        branch3_dilations =branch_dilations["branch3"],
    ).to(device)

    state = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    logger.info("Loaded M4 weights from %s", WEIGHTS_PATH)
    logger.info("Backbone HP: num_filters=%d kernel=%d dropout=%.3f fusion=%s",
                backbone_hp["num_filters"], backbone_hp["kernel_size"],
                backbone_hp["dropout"], backbone_hp["fusion"])
    logger.info("Attention HP: attention_dim=%d attention_dropout=%.3f",
                attn_hp["attention_dim"], attn_hp["attention_dropout"])
    return model


# ---------------------------------------------------------------------------
def load_predicted_events(csv_path, logger):
    """Group TP+FP rows from event_details.csv by mouse_id."""
    if not csv_path.exists():
        logger.error("event_details.csv not found at %s", csv_path)
        sys.exit(1)
    per_mouse = {}
    n_tp = n_fp = 0
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            mid = row["mouse_id"]
            is_tp = str(row["is_true_alarm"]).strip().lower() in ("true", "1")
            evt = {
                "start_sec":   float(row["start_sec"]),
                "end_sec":     float(row["end_sec"]),
                "is_true_alarm": is_tp,
            }
            per_mouse.setdefault(mid, []).append(evt)
            n_tp += int(is_tp); n_fp += int(not is_tp)
    logger.info("Loaded %d predicted events (%d TP, %d FP) from %s",
                n_tp + n_fp, n_tp, n_fp, csv_path.name)
    return per_mouse


def derive_fn_events(predicted_events, gt_intervals):
    _, _, fn, _ = match_events_to_ground_truth(predicted_events, gt_intervals)
    return [{"start_sec": float(gs), "end_sec": float(ge)} for _, (gs, ge) in fn]


# ---------------------------------------------------------------------------
def select_windows_overlapping_events(records, events):
    """Return the manifest indices of windows whose [t_start, t_start+5s)
    overlaps any event interval. Each event is treated as spanning the
    CROP_LEN region (MIN_EVENT_SEC s from start), since that's the region
    we crop.
    """
    if not events:
        return []
    starts = np.array([e["start_sec"] for e in events], dtype=np.float64)
    crop_ends = starts + MIN_EVENT_SEC

    selected = []
    for i, rec in enumerate(records):
        wt = float(rec["t_start_sec"])
        we = wt + SEGMENT_SEC
        # Overlaps any event-crop interval if there exists k with
        #   wt < crop_ends[k] AND we > starts[k]
        if np.any((wt < crop_ends) & (we > starts)):
            selected.append(i)
    return selected


def extract_attention(model, file_label_pairs, device, batch_size, num_workers, logger):
    """Run forward + get_attention_weights on the supplied file pairs.
    Returns (B, WIN_LEN) numpy float32 array in the loader's iteration
    order. Loader is non-shuffled, drop_last=False, so order matches the
    input list.
    """
    dataset = SafeEEGSegmentDataset(file_label_pairs)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    chunks = []
    n_done = 0
    t0 = time.time()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            w = model.get_attention_weights(x)            # (B, WIN_LEN) float32
            chunks.append(w.astype(np.float32, copy=False))
            n_done += x.size(0)
    if not chunks:
        return np.zeros((0, WIN_LEN), dtype=np.float32)
    out = np.concatenate(chunks, axis=0)
    elapsed = time.time() - t0
    logger.info("    attention extracted: %d windows in %.1fs (%.0f win/s)",
                n_done, elapsed, n_done / max(elapsed, 1e-3))
    return out


def overlap_add_crop(event, window_starts_samp, attention_subset):
    """Crop continuous saliency over [event_start, event_start+CROP_LEN)
    using per-window overlap-add. Returns (crop, coverage) both length
    CROP_LEN_SAMPLES; coverage[t] is the number of windows that contributed
    to sample t (used to mask sparsely-covered tails before averaging).
    """
    event_start_samp = int(round(event["start_sec"] * FS))
    S = np.zeros(CROP_LEN_SAMPLES, dtype=np.float64)
    C = np.zeros(CROP_LEN_SAMPLES, dtype=np.int32)
    for ws, alpha in zip(window_starts_samp, attention_subset):
        # Window covers absolute samples [ws, ws+WIN_LEN). In event-relative
        # coords, that's [ws - event_start, ws - event_start + WIN_LEN).
        rel_start = ws - event_start_samp
        src_lo = max(0, -rel_start)
        src_hi = min(WIN_LEN, CROP_LEN_SAMPLES - rel_start)
        if src_hi <= src_lo:
            continue
        tgt_lo = rel_start + src_lo
        tgt_hi = rel_start + src_hi
        S[tgt_lo:tgt_hi] += alpha[src_lo:src_hi]
        C[tgt_lo:tgt_hi] += 1
    crop = np.where(C > 0, S / np.maximum(C, 1), np.nan).astype(np.float32)
    return crop, C


# ---------------------------------------------------------------------------
def process_partition(partition, args, logger):
    SALIENCY_DIR = SALIENCY_ROOT / partition
    SALIENCY_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Output dir: %s", SALIENCY_DIR)

    # -- Device + GPU diagnostics ----------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logger.info("GPU  : %s", torch.cuda.get_device_name(0))
        logger.info("VRAM : %.2f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)
        logger.info("CUDA : %s", torch.version.cuda)
        try:
            _free_bytes, _total_bytes = torch.cuda.mem_get_info(0)
            _free_gb = _free_bytes / 1e9
            _total_gb = _total_bytes / 1e9
            logger.info("GPU memory free: %.2f / %.2f GB", _free_gb, _total_gb)
            if _free_gb < 8.0:
                logger.warning("GPU has only %.2f GB free (< 8 GB threshold). "
                               "Another process may be sharing this GPU, or VRAM "
                               "is fragmented.", _free_gb)
        except Exception as _e:
            logger.warning("Could not query GPU memory: %s", _e)
    else:
        logger.info("Device: CPU")
    logger.info("PyTorch: %s", torch.__version__)

    # -- Inputs ----------------------------------------------------------------
    model = build_model(device, logger)

    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)
    if partition not in splits or not splits[partition]:
        logger.error("Partition %s is empty in manifest", partition); sys.exit(1)
    part_records = list(splits[partition])
    logger.info("Manifest: %d %s records", len(part_records), partition)

    mouse_metadata = json.loads(MOUSE_METADATA_PATH.read_text(encoding="utf-8"))

    predicted_per_mouse = load_predicted_events(event_details_path_for(partition), logger)

    # Group manifest records by mouse_id (preserves manifest order within mouse)
    records_per_mouse = {}
    for rec in part_records:
        records_per_mouse.setdefault(rec["mouse_id"], []).append(rec)
    logger.info("Mice in partition: %d", len(records_per_mouse))

    # -- Per-mouse processing --------------------------------------------------
    crops = {"TP": [], "FP": [], "FN": []}
    crops_meta = {"TP": [], "FP": [], "FN": []}     # list of (mouse_id, start_sec, end_sec) per crop

    n_windows_total = 0
    for mouse_id in sorted(records_per_mouse):
        mouse_records = records_per_mouse[mouse_id]
        predicted = predicted_per_mouse.get(mouse_id, [])

        if mouse_id not in mouse_metadata:
            logger.warning("  %s : no mouse metadata, skipping", mouse_id); continue
        meta = mouse_metadata[mouse_id]
        xlsx = ANNOT_DIR / f"{mouse_id}_xlsx.xlsx"
        if not xlsx.exists():
            logger.warning("  %s : annotations missing at %s, skipping", mouse_id, xlsx); continue
        rec_start = datetime.datetime.fromisoformat(meta["recording_start_dt"])
        gt_intervals = load_annotations(xlsx, rec_start)

        tp_events = [e for e in predicted if e["is_true_alarm"]]
        fp_events = [e for e in predicted if not e["is_true_alarm"]]
        fn_events = derive_fn_events(predicted, gt_intervals)

        all_events = tp_events + fp_events + fn_events
        if not all_events:
            logger.info("  %s : no events (TP/FP/FN) -- skipping", mouse_id); continue

        sel_idx = select_windows_overlapping_events(mouse_records, all_events)
        if not sel_idx:
            logger.info("  %s : 0 overlapping windows for %d events -- skipping",
                        mouse_id, len(all_events))
            continue

        sel_records = [mouse_records[i] for i in sel_idx]
        pairs = [(r["filepath"], r["label"]) for r in sel_records]
        window_starts_samp = np.array(
            [int(round(float(r["t_start_sec"]) * FS)) for r in sel_records],
            dtype=np.int64)

        logger.info("  %s : GT=%d TP=%d FP=%d FN=%d | windows selected=%d",
                    mouse_id, len(gt_intervals),
                    len(tp_events), len(fp_events), len(fn_events), len(pairs))
        n_windows_total += len(pairs)

        attention = extract_attention(model, pairs, device,
                                      args.batch_size, args.num_workers, logger)

        for evt in tp_events:
            crop, _ = overlap_add_crop(evt, window_starts_samp, attention)
            crops["TP"].append(crop)
            crops_meta["TP"].append((mouse_id, evt["start_sec"], evt["end_sec"]))
        for evt in fp_events:
            crop, _ = overlap_add_crop(evt, window_starts_samp, attention)
            crops["FP"].append(crop)
            crops_meta["FP"].append((mouse_id, evt["start_sec"], evt["end_sec"]))
        for evt in fn_events:
            crop, _ = overlap_add_crop(evt, window_starts_samp, attention)
            crops["FN"].append(crop)
            crops_meta["FN"].append((mouse_id, evt["start_sec"], evt["end_sec"]))

        # Free per-mouse attention cache
        del attention

    logger.info("Total windows forward-passed: %d (vs %d in partition = %.3f%%)",
                n_windows_total, len(part_records),
                100.0 * n_windows_total / max(len(part_records), 1))

    return crops, crops_meta, SALIENCY_DIR


# ---------------------------------------------------------------------------
def aggregate_class(stack):
    """stack: (n_events, CROP_LEN_SAMPLES) float32 with NaN for uncovered.
    Returns (mean, sem, n_per_sample). NaN-aware so partially-covered
    samples are excluded from both mean and SEM.
    """
    if stack.shape[0] == 0:
        zeros = np.zeros(CROP_LEN_SAMPLES, dtype=np.float64)
        return zeros, zeros, np.zeros(CROP_LEN_SAMPLES, dtype=np.int64)
    mask = ~np.isnan(stack)
    n_per_sample = mask.sum(axis=0)
    mean = np.nanmean(stack, axis=0)
    std  = np.nanstd(stack, axis=0, ddof=1)
    sem  = np.where(n_per_sample > 1, std / np.sqrt(n_per_sample), 0.0)
    return mean, sem, n_per_sample


def plot_curves(class_stats, out_path, partition, logger):
    """class_stats: dict class -> (mean, sem, n_per_sample, n_events)."""
    t = np.arange(CROP_LEN_SAMPLES) / FS
    colors = {"TP": "#1F8E5A", "FP": "#C03A2B", "FN": "#5A7DC8"}

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    for cls in ["TP", "FP", "FN"]:
        mean, sem, _, n_ev = class_stats[cls]
        if n_ev == 0:
            continue
        ax.plot(t, mean, color=colors[cls], lw=1.5,
                label=f"{cls} (n={n_ev})")
        ax.fill_between(t, mean - sem, mean + sem, color=colors[cls], alpha=0.25)
    ax.set_ylabel("Mean attention weight")
    ax.set_title(f"M4 Event-Level Attention Saliency ({partition.upper()})")
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="upper right")

    axn = axes[1]
    for cls in ["TP", "FP", "FN"]:
        _, _, n_per_samp, n_ev = class_stats[cls]
        if n_ev == 0:
            continue
        axn.plot(t, n_per_samp, color=colors[cls], lw=1.0)
    axn.set_xlabel("Time from event onset (s)")
    axn.set_ylabel("Events contributing")
    axn.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


def plot_differences(class_stats, out_path, partition, logger):
    t = np.arange(CROP_LEN_SAMPLES) / FS
    tp_mean = class_stats["TP"][0]
    fp_mean = class_stats["FP"][0]
    fn_mean = class_stats["FN"][0]

    fig, ax = plt.subplots(figsize=(10, 5))
    if class_stats["TP"][3] and class_stats["FP"][3]:
        ax.plot(t, tp_mean - fp_mean, color="#9534A0", lw=1.5,
                label=f"TP - FP (n_TP={class_stats['TP'][3]}, n_FP={class_stats['FP'][3]})")
    if class_stats["TP"][3] and class_stats["FN"][3]:
        ax.plot(t, tp_mean - fn_mean, color="#D9893F", lw=1.5,
                label=f"TP - FN (n_TP={class_stats['TP'][3]}, n_FN={class_stats['FN'][3]})")
    ax.axhline(0.0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Time from event onset (s)")
    ax.set_ylabel("Mean attention difference")
    ax.set_title(f"M4 Differential Attention ({partition.upper()})")
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


def summarise(class_stats, partition):
    """Top-line stats per class for the summary JSON."""
    summary = {
        "partition":           partition,
        "crop_length_sec":     MIN_EVENT_SEC,
        "crop_length_samples": CROP_LEN_SAMPLES,
        "fs_hz":               FS,
        "win_len_samples":     WIN_LEN,
        "step_samples":        STEP,
        "model":               "MultiScaleTCNWithAttention",
        "classes":             {},
    }
    for cls in ["TP", "FP", "FN"]:
        mean, sem, n_per, n_ev = class_stats[cls]
        if n_ev == 0:
            summary["classes"][cls] = {"n_events": 0}
            continue
        # Peak across the valid (non-NaN) part of the curve
        valid = ~np.isnan(mean)
        if not valid.any():
            summary["classes"][cls] = {"n_events": int(n_ev), "peak_amplitude": None,
                                       "peak_time_sec": None}
            continue
        idx_valid = np.where(valid)[0]
        peak_local = int(np.nanargmax(mean[valid]))
        peak_idx = int(idx_valid[peak_local])
        summary["classes"][cls] = {
            "n_events":             int(n_ev),
            "peak_amplitude":       float(mean[peak_idx]),
            "peak_time_sec":        float(peak_idx / FS),
            "mean_amplitude":       float(np.nanmean(mean)),
            "min_samples_covered":  int(n_per.min()),
            "max_samples_covered":  int(n_per.max()),
        }
    return summary


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    out_dir = SALIENCY_ROOT / args.partition
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir)
    logger.info("=" * 70)
    logger.info("event_attention_saliency.py")
    logger.info("Partition  : %s", args.partition)
    logger.info("Crop len   : %.1f s (%d samples @ %d Hz)",
                MIN_EVENT_SEC, CROP_LEN_SAMPLES, FS)
    logger.info("Constants  : WIN_LEN=%d  STEP=%d  THRESHOLD=%.2f  SMOOTHING_WIN=%d  "
                "MIN_EVENT_SEC=%.1f  REFRACTORY_SEC=%.1f",
                WIN_LEN, STEP, THRESHOLD, SMOOTHING_WIN,
                MIN_EVENT_SEC, REFRACTORY_SEC)
    logger.info("=" * 70)

    t_start = time.time()
    crops, crops_meta, out_dir = process_partition(args.partition, args, logger)

    # Stack to (n_events, CROP_LEN) per class, then aggregate
    class_stats = {}
    for cls in ["TP", "FP", "FN"]:
        if crops[cls]:
            stack = np.stack(crops[cls], axis=0).astype(np.float32)
        else:
            stack = np.zeros((0, CROP_LEN_SAMPLES), dtype=np.float32)
        mean, sem, n_per = aggregate_class(stack)
        class_stats[cls] = (mean, sem, n_per, stack.shape[0])
        logger.info("Class %s : %d events stacked", cls, stack.shape[0])

    # -- Save figures ----------------------------------------------------------
    plot_curves(class_stats, out_dir / "event_saliency_curves.png", args.partition, logger)
    plot_differences(class_stats, out_dir / "tp_fp_fn_difference.png", args.partition, logger)

    # -- Save NPZ of raw crops + per-class mean / sem --------------------------
    npz_payload = {}
    for cls in ["TP", "FP", "FN"]:
        if crops[cls]:
            npz_payload[f"{cls}_crops"] = np.stack(crops[cls], axis=0).astype(np.float32)
            npz_payload[f"{cls}_meta"]  = np.array(crops_meta[cls], dtype=object)
        npz_payload[f"{cls}_mean"]      = class_stats[cls][0].astype(np.float32)
        npz_payload[f"{cls}_sem"]       = class_stats[cls][1].astype(np.float32)
        npz_payload[f"{cls}_n_per_sample"] = class_stats[cls][2].astype(np.int64)
    npz_payload["time_sec"] = (np.arange(CROP_LEN_SAMPLES) / FS).astype(np.float32)
    npz_path = out_dir / "per_event_saliency.npz"
    np.savez_compressed(npz_path, **npz_payload)
    logger.info("Saved: %s", npz_path)

    # -- Save summary JSON -----------------------------------------------------
    summary = summarise(class_stats, args.partition)
    summary["walltime_seconds"] = round(time.time() - t_start, 2)
    summary["weights_path"]     = str(WEIGHTS_PATH)
    summary["splits_path"]      = str(SPLITS_PATH)
    summary["event_details_csv"]= str(event_details_path_for(args.partition))
    json_path = out_dir / "event_saliency_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info("Saved: %s", json_path)

    logger.info("=" * 70)
    logger.info("Done in %.1f s.", time.time() - t_start)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
