"""
deploy_inference.py
===================
Run a trained seizure-detection model on a single EDF recording and
emit a clinician-facing event list with absolute wall-clock timestamps.

Detection only. No evaluation, no ground-truth comparison, no metric
computation. Evaluation against annotated seizure intervals is a
separate operation handled by the *_evaluation.py scripts.

The script is variant-agnostic: it loads any of the four trained
architectures (TCN, TCNWithAttention, MultiScaleTCN,
MultiScaleTCNWithAttention) by passing --variant on the command line.

Pipeline
--------
1. Load EDF header (mne, preload=False).
2. Compute the segment grid identical to preprocessing_binary.py.
3. Iterate the grid in chunks; for each chunk:
     a. Read raw samples from EDF.
     b. Apply notch filter -> DC removal -> robust z-score (matches
        the preprocessing pipeline used at training time).
     c. Extract individual 5-s segments.
     d. Sanitise (NaN/Inf -> 0, clip |x|>1000) -- Layer 2.
     e. Batch + FP32 forward pass -- Layer 3.
     f. Per-batch torch.isfinite assertion -- Layer 4.
4. Apply smoothing + threshold + run-detection + refractory merge +
   minimum-duration filter (single-mouse, single-chunk pipeline).
5. Augment each event with absolute datetime stamps
   (recording_start_dt + t_start_sec).
6. Write events.csv, events.json, predictions.npz, and a persistent log.

Usage
-----
  python deploy_inference.py \\
      --edf            /path/to/m_new.edf \\
      --variant        MultiScaleTCN \\
      --weights        /path/to/multiscale_tcn_final_weights.pt \\
      --params         /path/to/best_multiscale_params.json \\
      --output-dir     /path/to/deploy_outputs/m_new \\
     [--attn-params    /path/to/best_multiscale_attn_params.json]   # only for *Attention variants
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import mne
import numpy as np
import torch
from scipy.signal import iirnotch, filtfilt
from scipy.stats import median_abs_deviation

import tcn_utils
from eval_utils import (
    detect_events_in_chunk,
    emit_deploy_event_bundle,
    AMPLITUDE_THRESHOLD,
    FS, WIN_LEN, STEP, STEP_SEC,
    THRESHOLD, SMOOTHING_WIN, REFRACTORY_SEC, MIN_EVENT_SEC,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NOTCH_FREQ          = 50.0     # powerline interference frequency (EU)
NOTCH_Q             = 30       # notch filter Q-factor
SEGMENTS_PER_CHUNK  = 200      # ~500 s of data per chunk; balances RAM and EDF reads
USE_AMP_FOR_EVAL    = False    # FP32 forward pass


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--edf",         required=True, type=Path,
                   help="Path to the EDF file to detect seizures in.")
    p.add_argument("--variant",     required=True,
                   choices=["TCN", "TCNWithAttention",
                            "MultiScaleTCN", "MultiScaleTCNWithAttention"],
                   help="Trained model architecture variant.")
    p.add_argument("--weights",     required=True, type=Path,
                   help="Path to the trained .pt weights file.")
    p.add_argument("--params",      required=True, type=Path,
                   help="best_*_params.json with hyperparameters. For "
                        "*Attention variants this is the BACKBONE params.")
    p.add_argument("--attn-params", type=Path, default=None,
                   help="Optional best_*_attn_params.json (required for "
                        "MultiScaleTCNWithAttention).")
    p.add_argument("--output-dir",  required=True, type=Path,
                   help="Directory for events.csv, events.json, log, etc.")
    p.add_argument("--batch-size",  type=int, default=32,
                   help="Inference batch size (default 32).")
    p.add_argument("--device",      default=None,
                   help="cuda or cpu. Default: cuda if available, else cpu.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------
def setup_logging(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "deploy_inference.log"
    logger = logging.getLogger("deploy_inference")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8"); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger, log_path


# ---------------------------------------------------------------------------
# Preprocessing -- mirrors preprocessing_binary.py exactly
# ---------------------------------------------------------------------------
def apply_notch_filter(signal, fs, freq, Q):
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, signal)


def remove_dc(signal):
    return signal - np.mean(signal)


def robust_zscore(signal):
    median = np.median(signal)
    mad    = median_abs_deviation(signal, scale="normal")
    if mad <= 0:
        return signal - median
    return (signal - median) / mad


def preprocess_chunk(chunk, fs):
    s = apply_notch_filter(chunk, fs, NOTCH_FREQ, NOTCH_Q)
    s = remove_dc(s)
    s = robust_zscore(s)
    return s


def sanitise_segment(x):
    """Inline equivalent of SafeEEGSegmentDataset's defensive cleanup."""
    n_san = 0
    if not np.isfinite(x).all():
        n_san += 1
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if np.abs(x).max() > AMPLITUDE_THRESHOLD:
        n_san += 1
        np.clip(x, -AMPLITUDE_THRESHOLD, AMPLITUDE_THRESHOLD, out=x)
    return x, n_san


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def build_model(variant, hp, attn_hp, branch_dilations, device, logger):
    """Construct the appropriate model from the variant name and HPs."""
    if variant == "TCN":
        model = tcn_utils.TCN(
            num_layers=int(hp["num_layers"]),
            num_filters=int(hp["num_filters"]),
            kernel_size=int(hp["kernel_size"]),
            dropout=float(hp["dropout"]),
        )
    elif variant == "TCNWithAttention":
        ahp = attn_hp if attn_hp is not None else hp
        model = tcn_utils.TCNWithAttention(
            num_layers=int(hp["num_layers"]),
            num_filters=int(hp["num_filters"]),
            kernel_size=int(hp["kernel_size"]),
            dropout=float(hp["dropout"]),
            attention_dim=int(ahp.get("attention_dim", 64)),
            attention_dropout=float(ahp.get("attention_dropout", 0.0)),
        )
    elif variant == "MultiScaleTCN":
        bd = branch_dilations or {"branch1": [1,2,4], "branch2": [8,16,32], "branch3": [32,64,128]}
        model = tcn_utils.MultiScaleTCN(
            num_filters=int(hp["num_filters"]),
            kernel_size=int(hp["kernel_size"]),
            dropout=float(hp["dropout"]),
            branch1_dilations=bd["branch1"],
            branch2_dilations=bd["branch2"],
            branch3_dilations=bd["branch3"],
            fusion=str(hp.get("fusion", "concat")),
        )
    elif variant == "MultiScaleTCNWithAttention":
        if attn_hp is None:
            raise ValueError("MultiScaleTCNWithAttention requires --attn-params")
        bd = branch_dilations or {"branch1": [1,2,4], "branch2": [8,16,32], "branch3": [32,64,128]}
        model = tcn_utils.MultiScaleTCNWithAttention(
            num_filters=int(hp["num_filters"]),
            kernel_size=int(hp["kernel_size"]),
            dropout=float(hp["dropout"]),
            fusion=str(hp.get("fusion", "concat")),
            attention_dim=int(attn_hp["attention_dim"]),
            attention_dropout=float(attn_hp["attention_dropout"]),
            branch1_dilations=bd["branch1"],
            branch2_dilations=bd["branch2"],
            branch3_dilations=bd["branch3"],
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Built %s model (%s trainable params)", variant, "{:,}".format(n_params))
    return model


# ---------------------------------------------------------------------------
# infer_on_edf
# ---------------------------------------------------------------------------
def infer_on_edf(model, edf_path, batch_size, device, logger):
    """Forward-pass the model over an entire EDF in chunks. Returns
    (y_prob, n_sanitised, n_samples, fs, recording_start_dt).
    """
    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
    n_samples = int(raw.n_times)
    fs        = float(raw.info["sfreq"])
    meas_date = raw.info["meas_date"]
    if meas_date is None:
        raise ValueError(f"EDF {edf_path} has no meas_date in header.")
    rec_start = meas_date.replace(tzinfo=None)
    duration_h = n_samples / fs / 3600

    if abs(fs - FS) > 0.5:
        logger.warning("EDF sampling rate %g Hz differs from training FS=%d -- "
                       "predictions may be invalid.", fs, FS)

    last_valid     = n_samples - WIN_LEN
    segment_starts = np.arange(0, last_valid + 1, STEP, dtype=np.int64)
    n_segs         = len(segment_starts)
    logger.info("EDF: %s | n_samples=%d | fs=%g | duration=%.2f h | n_segs=%d",
                edf_path.name, n_samples, fs, duration_h, n_segs)

    chunk_groups = [segment_starts[i:i + SEGMENTS_PER_CHUNK]
                    for i in range(0, n_segs, SEGMENTS_PER_CHUNK)]
    logger.info("Processing in %d chunks of up to %d segments each.",
                len(chunk_groups), SEGMENTS_PER_CHUNK)

    y_prob_all = np.empty(n_segs, dtype=np.float64)
    n_sanitised_total = 0
    write_idx = 0
    batch_buffer = []

    def flush_batch():
        nonlocal write_idx, batch_buffer
        if not batch_buffer:
            return
        x = torch.from_numpy(np.stack(batch_buffer)).unsqueeze(1).to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=USE_AMP_FOR_EVAL):
            logits = model(x)
        if not torch.isfinite(logits).all():
            raise AssertionError(
                f"Non-finite logit at batch ending segment index "
                f"{write_idx + len(batch_buffer) - 1}. Investigate before re-running.")
        probs = torch.sigmoid(logits).cpu().numpy()
        y_prob_all[write_idx:write_idx + len(probs)] = probs
        write_idx += len(probs)
        batch_buffer = []

    log_every = max(1, len(chunk_groups) // 50)
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        for chunk_idx, chunk_seg_starts in enumerate(chunk_groups):
            chunk_start = int(chunk_seg_starts[0])
            chunk_end   = int(chunk_seg_starts[-1]) + WIN_LEN

            raw_block = raw.get_data(start=chunk_start, stop=chunk_end).squeeze()
            block = preprocess_chunk(raw_block.astype(np.float64), fs)

            for seg_start_idx in chunk_seg_starts:
                local_start = int(seg_start_idx) - chunk_start
                segment = block[local_start:local_start + WIN_LEN].astype(np.float32)
                segment, n_san = sanitise_segment(segment)
                n_sanitised_total += n_san
                batch_buffer.append(segment)
                if len(batch_buffer) >= batch_size:
                    flush_batch()

            if (chunk_idx + 1) % log_every == 0 or chunk_idx == len(chunk_groups) - 1:
                elapsed = time.time() - t0
                rate = (write_idx + len(batch_buffer)) / max(elapsed, 1e-3)
                eta = (n_segs - write_idx - len(batch_buffer)) / max(rate, 1e-3)
                logger.info("  chunk %d/%d (%.0f%%) | %d segs done | %.0f seg/s | eta %.0f s",
                            chunk_idx + 1, len(chunk_groups),
                            100 * (chunk_idx + 1) / len(chunk_groups),
                            write_idx + len(batch_buffer), rate, eta)

        flush_batch()

    if n_sanitised_total:
        logger.warning("Sanitised %d segments (NaN/Inf or |x|>1000).",
                       n_sanitised_total)
    return y_prob_all, n_sanitised_total, n_samples, fs, rec_start


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    logger, log_path = setup_logging(args.output_dir)
    logger.info("=" * 65)
    logger.info("deploy_inference.py")
    logger.info("EDF        : %s", args.edf)
    logger.info("Variant    : %s", args.variant)
    logger.info("Weights    : %s", args.weights)
    logger.info("Params     : %s", args.params)
    logger.info("Attn params: %s", args.attn_params)
    logger.info("Output dir : %s", args.output_dir)
    logger.info("Log path   : %s", log_path)
    logger.info("Mode       : detection only (no evaluation, no GT comparison)")
    logger.info("=" * 65)

    for path in [args.edf, args.weights, args.params]:
        if not Path(path).exists():
            logger.error("Missing required file: %s", path); sys.exit(1)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Device        : %s", device)
    if torch.cuda.is_available() and device.type == "cuda":
        logger.info("GPU  : %s", torch.cuda.get_device_name(0))
        logger.info("VRAM : %.2f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)
        logger.info("CUDA : %s", torch.version.cuda)
        try:
            _free, _total = torch.cuda.mem_get_info(0)
            logger.info("GPU memory free: %.2f / %.2f GB", _free / 1e9, _total / 1e9)
        except Exception as _e:
            logger.warning("Could not query GPU memory: %s", _e)
    logger.info("PyTorch       : %s", torch.__version__)

    # -- Load HPs ------------------------------------------------------------
    backbone_cfg = json.loads(args.params.read_text(encoding="utf-8"))
    hp = backbone_cfg.get("hyperparameters", backbone_cfg)
    branch_dilations = backbone_cfg.get("branch_dilations", None)
    attn_hp = None
    if args.attn_params and args.attn_params.exists():
        attn_cfg = json.loads(args.attn_params.read_text(encoding="utf-8"))
        attn_hp  = attn_cfg.get("hyperparameters", attn_cfg)

    # -- Build model + load weights -----------------------------------------
    model = build_model(args.variant, hp, attn_hp, branch_dilations, device, logger)
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    logger.info("Loaded weights from %s", args.weights)

    # -- Inference -----------------------------------------------------------
    t0 = time.time()
    y_prob, n_sanitised, n_samples, fs, rec_start = infer_on_edf(
        model, args.edf, args.batch_size, device, logger)
    eval_seconds = time.time() - t0
    logger.info("Inference complete: %d segments in %.1f s", len(y_prob), eval_seconds)

    # -- Event detection (single mouse) -------------------------------------
    mouse_id = args.edf.stem
    t_start_sec = np.arange(len(y_prob)) * STEP_SEC
    events, _, _ = detect_events_in_chunk(t_start_sec, y_prob, mouse_id, chunk_id=0)
    logger.info("Detected %d events after smoothing/refractory/min-duration filtering.",
                len(events))

    # Augment events with recording-relative HH:MM:SS.s timestamps so they
    # can be located directly on the EDF viewer's elapsed-time axis.
    def _to_hms(sec):
        sec = float(sec)
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = sec - h * 3600 - m * 60
        return f"{h:02d}:{m:02d}:{s:04.1f}"
    for evt in events:
        evt["start_recording_time"] = _to_hms(evt["start_sec"])
        evt["end_recording_time"]   = _to_hms(evt["end_sec"])

    # -- Write outputs (canonical 2-file deploy bundle) ----------------------
    # Use the shared writer in eval_utils so the deploy event_details CSV
    # uses the exact same 17-column schema as training / evaluation /
    # recovery / sweep outputs (GT-related cells left empty in deploy).
    recording_metadata = {
        "edf_path":                 str(args.edf),
        "recording_start_datetime": rec_start.isoformat(),
        "recording_duration_sec":   round(n_samples / fs, 4),
        "n_samples":                int(n_samples),
        "fs_hz":                    float(fs),
        "n_segments":               int(len(y_prob)),
        "n_segments_sanitised":     int(n_sanitised),
        "inference_seconds":        round(float(eval_seconds), 1),
        "variant":                  args.variant,
        "weights_path":             str(args.weights),
    }
    emit_deploy_event_bundle(
        args.output_dir, mouse_id, events,
        recording_metadata=recording_metadata,
        model_label=args.variant,
        logger=logger,
        order="min_then_refractory",
        min_event_duration_sec=MIN_EVENT_SEC,
        refractory_period_sec=REFRACTORY_SEC,
        smoothing_window=SMOOTHING_WIN,
        threshold=THRESHOLD,
        step_sec=STEP_SEC,
    )

    # Cache predictions NPZ for downstream re-analysis
    npz_path = args.output_dir / "predictions.npz"
    np.savez_compressed(
        npz_path,
        y_prob=y_prob.astype(np.float32),
        t_start_sec=t_start_sec.astype(np.float64),
        n_samples=np.int64(n_samples),
        fs_hz=np.float64(fs),
        recording_start_dt=np.array(rec_start.isoformat()),
    )
    logger.info("Saved predictions NPZ: %s (%.2f MB)",
                npz_path, npz_path.stat().st_size / 1e6)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("DONE")


if __name__ == "__main__":
    main()
