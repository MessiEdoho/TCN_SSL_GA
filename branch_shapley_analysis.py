"""
branch_shapley_analysis.py
==========================
Per-prediction Shapley value attribution for the three temporal-scale branches
of MultiScaleTCN (M3) and MultiScaleTCNWithAttention (M4).

For each segment-level prediction the script computes the Shapley value of every
branch (B1 fine, B2 medium, B3 coarse) with respect to the model's pre-sigmoid
logit. With K=3 branches all 2^K = 8 coalitions are enumerated exactly -- no
sampling approximation. Branch convolutions are evaluated once per batch and
their outputs are reused across the 8 fusion masks, so the wall-time cost is
~2x a standard test-set inference pass rather than 8x.

Use case
--------
Aggregate ablation (run_branch_ablation in interpretability_analysis.py) reports
one F1 number per branch -- "which branch matters most on average." This script
answers the per-prediction question: "which temporal scale drove THIS seizure
call, and which scale drove THIS false alarm?" The output is a streaming CSV
with one row per segment, suitable for downstream stratification by TP / FP /
TN / FN or by mouse.

Outputs
-------
outputs/interpretability/<partition>/branch_shapley/
    <model>_shapley_<partition>.csv      streaming per-segment CSV
    <model>_shapley_<partition>_summary.json  population-level aggregates
    logs/branch_shapley_<model>_<partition>.log  FileHandler log

Usage
-----
python branch_shapley_analysis.py --partition test --model M3
python branch_shapley_analysis.py --partition val  --model M4
python branch_shapley_analysis.py --partition test --model all

Prerequisites
-------------
Trained weights at the paths inherited from interpretability_analysis.py:
    outputs/MultiScaleTCN/multiscale_tcn_final_weights.pt           (M3)
    outputs/MultiScaleTCNAttention/multiscale_tcn_attention_final_weights.pt  (M4)
"""

import argparse
import csv
import datetime
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import torch

from tcn_utils import set_seed
from interpretability_analysis import (
    SEED,
    INTERP_BASE,
    M3_WEIGHTS,
    M4_WEIGHTS,
    BEST_TCN_PATH,
    BEST_MS_PATH,
    BEST_MS_ATTN_PATH,
    DEFAULT_BRANCH1,
    DEFAULT_BRANCH2,
    DEFAULT_BRANCH3,
    M4_AVAILABLE,
    load_model_weights,
    load_partition_data,
    compute_branch_outputs,
    forward_with_branch_mask,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DECISION_THRESHOLD = 0.5     # raw-output threshold matching run_branch_ablation
BATCH_SIZE = 32              # inherits the existing interpretability batch size
FNAME_REGEX = re.compile(r"^(m\d+)_(ictal|nonictal)_(\d+)")


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-prediction branch Shapley attribution for M3 / M4.")
    parser.add_argument("--partition", choices=["val", "test"], required=True,
                        help="val: development (carries optimisation bias). "
                             "test: paper reporting.")
    parser.add_argument("--model", choices=["M3", "M4", "all"], default="all",
                        help="Which multi-scale model to attribute (default: all).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------
def setup_logging(interp_root, partition, model_name):
    out_dir = interp_root / "branch_shapley"
    log_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / ("branch_shapley_%s_%s.log" % (model_name.lower(), partition))
    logger = logging.getLogger("branch_shapley_%s_%s" % (model_name, partition))
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)

    logger.info("=" * 70)
    logger.info("branch_shapley_analysis.py -- %s / partition: %s",
                model_name, partition.upper())
    logger.info("Timestamp: %s", datetime.datetime.now().isoformat())
    logger.info("Log file:  %s", log_file)
    if partition == "val":
        logger.warning("PARTITION: VALIDATION -- carries threshold-selection bias. "
                       "Use --partition test for paper-quality attributions.")
    logger.info("=" * 70)
    return logger, out_dir


# ---------------------------------------------------------------------------
# parse_filepath_metadata
# ---------------------------------------------------------------------------
def parse_filepath_metadata(filepath):
    """Extract (mouse_id, label_str, seg_index, basename) from a segment .npy path.

    Filenames follow the preprocessing_binary*.py convention:
        {mouse_id}_ictal_{index:05d}.npy   or   {mouse_id}_nonictal_{index:05d}.npy
    """
    basename = Path(filepath).stem
    m = FNAME_REGEX.match(basename)
    if m is None:
        return ("unknown", "unknown", -1, basename)
    return (m.group(1), m.group(2), int(m.group(3)), basename)


# ---------------------------------------------------------------------------
# compute_shapley_batch
# ---------------------------------------------------------------------------
def compute_shapley_batch(model, x):
    """Return per-sample Shapley values phi_B1, phi_B2, phi_B3 and the full logit.

    Enumerates all 8 coalitions exactly. Branch outputs are computed once and
    reused across the 8 fusion masks via forward_with_branch_mask(branch_outputs=...).

    Parameters
    ----------
    model : MultiScaleTCN or MultiScaleTCNWithAttention, in eval mode on `device`
    x     : torch.Tensor, shape (batch, 1, T) on the same device as model

    Returns
    -------
    dict with float32 numpy arrays of shape (batch,):
        logit_full, phi_B1, phi_B2, phi_B3, logit_empty
    """
    bo = compute_branch_outputs(model, x)              # (o1, o2, o3), computed once

    # Coalition value v(S) = logit when branches IN S are ACTIVE (others zeroed).
    # forward_with_branch_mask zeroes the branches passed in the mask argument.
    v_empty = forward_with_branch_mask(model, x, [1, 2, 3], branch_outputs=bo)
    v_1     = forward_with_branch_mask(model, x, [2, 3],    branch_outputs=bo)
    v_2     = forward_with_branch_mask(model, x, [1, 3],    branch_outputs=bo)
    v_3     = forward_with_branch_mask(model, x, [1, 2],    branch_outputs=bo)
    v_12    = forward_with_branch_mask(model, x, [3],       branch_outputs=bo)
    v_13    = forward_with_branch_mask(model, x, [2],       branch_outputs=bo)
    v_23    = forward_with_branch_mask(model, x, [1],       branch_outputs=bo)
    v_full  = forward_with_branch_mask(model, x, [],        branch_outputs=bo)

    # Shapley value for K=3:
    #   weights:  |S|=0 -> 1/3,  |S|=1 -> 1/6,  |S|=2 -> 1/3
    # phi_k = (1/3)(v({k})-v(empty)) + (1/6)(v({k,j})-v({j})) + (1/6)(v({k,l})-v({l}))
    #       + (1/3)(v(full)-v(\{j,l}))
    one_third, one_sixth = 1.0 / 3.0, 1.0 / 6.0
    phi_b1 = (one_third * (v_1   - v_empty)
              + one_sixth * (v_12 - v_2)
              + one_sixth * (v_13 - v_3)
              + one_third * (v_full - v_23))
    phi_b2 = (one_third * (v_2   - v_empty)
              + one_sixth * (v_12 - v_1)
              + one_sixth * (v_23 - v_3)
              + one_third * (v_full - v_13))
    phi_b3 = (one_third * (v_3   - v_empty)
              + one_sixth * (v_13 - v_1)
              + one_sixth * (v_23 - v_2)
              + one_third * (v_full - v_12))

    return {
        "logit_full": v_full.detach().cpu().numpy().astype(np.float32),
        "logit_empty": v_empty.detach().cpu().numpy().astype(np.float32),
        "phi_B1": phi_b1.detach().cpu().numpy().astype(np.float32),
        "phi_B2": phi_b2.detach().cpu().numpy().astype(np.float32),
        "phi_B3": phi_b3.detach().cpu().numpy().astype(np.float32),
    }


# ---------------------------------------------------------------------------
# categorise_prediction
# ---------------------------------------------------------------------------
def categorise_prediction(y_true_i, y_pred_i):
    if y_true_i == 1 and y_pred_i == 1:
        return "TP"
    if y_true_i == 0 and y_pred_i == 1:
        return "FP"
    if y_true_i == 0 and y_pred_i == 0:
        return "TN"
    return "FN"


# ---------------------------------------------------------------------------
# run_for_model
# ---------------------------------------------------------------------------
def run_for_model(model, model_name, pairs, loader, partition, out_dir,
                  device, logger):
    """Stream per-segment Shapley rows to CSV; return a population summary dict.

    Efficiency-axiom sanity check is logged: per sample,
        phi_B1 + phi_B2 + phi_B3 == logit_full - logit_empty (within fp32 tolerance).
    """
    csv_path = out_dir / ("%s_shapley_%s.csv" % (model_name.lower(), partition))
    summary_path = out_dir / ("%s_shapley_%s_summary.json" % (model_name.lower(), partition))

    counts = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    # population-level aggregates for the JSON summary
    abs_sum = {"TP": np.zeros(3, dtype=np.float64),
               "FP": np.zeros(3, dtype=np.float64),
               "TN": np.zeros(3, dtype=np.float64),
               "FN": np.zeros(3, dtype=np.float64)}
    signed_sum = {k: np.zeros(3, dtype=np.float64) for k in counts}
    winner_counts = {k: {"B1": 0, "B2": 0, "B3": 0} for k in counts}
    efficiency_residuals = []

    model.eval()
    n_seen = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "segment_id", "mouse_id", "label", "seg_index", "file_basename",
            "y_true", "y_pred", "p_seizure", "logit_full", "logit_empty",
            "phi_B1", "phi_B2", "phi_B3",
            "winner_branch_abs", "dominance_abs", "category",
        ])
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                out = compute_shapley_batch(model, x)
                p = 1.0 / (1.0 + np.exp(-out["logit_full"]))
                y_pred = (p >= DECISION_THRESHOLD).astype(np.int64)
                y_true = y.numpy().astype(np.int64)
                phis = np.stack([out["phi_B1"], out["phi_B2"], out["phi_B3"]], axis=1)  # (B, 3)
                abs_phis = np.abs(phis)
                winners = np.argmax(abs_phis, axis=1)               # 0/1/2 -> B1/B2/B3
                dominance = abs_phis[np.arange(len(winners)), winners]
                efficiency_residuals.extend(
                    (phis.sum(axis=1) - (out["logit_full"] - out["logit_empty"])).tolist())

                for j in range(x.shape[0]):
                    fp_idx = n_seen + j
                    filepath, _label = pairs[fp_idx]
                    mouse_id, label_str, seg_idx, basename = parse_filepath_metadata(filepath)
                    cat = categorise_prediction(int(y_true[j]), int(y_pred[j]))
                    counts[cat] += 1
                    abs_sum[cat] += abs_phis[j]
                    signed_sum[cat] += phis[j]
                    winner_counts[cat]["B%d" % (winners[j] + 1)] += 1
                    writer.writerow([
                        fp_idx, mouse_id, label_str, seg_idx, basename,
                        int(y_true[j]), int(y_pred[j]),
                        "%.6f" % float(p[j]),
                        "%.6f" % float(out["logit_full"][j]),
                        "%.6f" % float(out["logit_empty"][j]),
                        "%.6f" % float(out["phi_B1"][j]),
                        "%.6f" % float(out["phi_B2"][j]),
                        "%.6f" % float(out["phi_B3"][j]),
                        "B%d" % (winners[j] + 1),
                        "%.6f" % float(dominance[j]),
                        cat,
                    ])
                n_seen += x.shape[0]
                if n_seen % (BATCH_SIZE * 500) == 0:
                    logger.info("  ...processed %d / %d segments", n_seen, len(pairs))

    if n_seen != len(pairs):
        logger.error("Sample count mismatch: processed %d, expected %d.", n_seen, len(pairs))
    logger.info("Wrote %d rows -> %s", n_seen, csv_path)

    residuals = np.asarray(efficiency_residuals, dtype=np.float64)
    logger.info("Efficiency axiom residual (|phi1+phi2+phi3 - (logit_full - logit_empty)|): "
                "max=%.3e, mean=%.3e", float(np.max(np.abs(residuals))),
                float(np.mean(np.abs(residuals))))

    summary = {
        "model_name": model_name,
        "partition": partition,
        "decision_threshold": DECISION_THRESHOLD,
        "n_segments": int(n_seen),
        "counts": counts,
        "mean_abs_phi": {cat: (abs_sum[cat] / max(counts[cat], 1)).tolist()
                         for cat in counts},
        "mean_signed_phi": {cat: (signed_sum[cat] / max(counts[cat], 1)).tolist()
                            for cat in counts},
        "winner_branch_counts": winner_counts,
        "efficiency_residual_max": float(np.max(np.abs(residuals))),
        "efficiency_residual_mean": float(np.mean(np.abs(residuals))),
        "csv_path": str(csv_path),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote summary -> %s", summary_path)

    for cat in ["TP", "FP", "TN", "FN"]:
        wc = winner_counts[cat]
        ma = summary["mean_abs_phi"][cat]
        logger.info("  %s n=%d | winner B1/B2/B3 = %d/%d/%d | mean|phi| = "
                    "B1 %.4f, B2 %.4f, B3 %.4f",
                    cat, counts[cat], wc["B1"], wc["B2"], wc["B3"],
                    ma[0], ma[1], ma[2])
    return summary


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    partition = args.partition
    interp_root = INTERP_BASE / partition

    requested = ["M3", "M4"] if args.model == "all" else [args.model]
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up a top-level logger for the data-loading phase shared across models.
    top_logger, out_dir = setup_logging(interp_root, partition, "shared")
    if torch.cuda.is_available():
        top_logger.info("GPU  : %s", torch.cuda.get_device_name(0))
        top_logger.info("VRAM : %.2f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)
        top_logger.info("CUDA : %s", torch.version.cuda)
        try:
            _free_bytes, _total_bytes = torch.cuda.mem_get_info(0)
            _free_gb = _free_bytes / 1e9
            _total_gb = _total_bytes / 1e9
            top_logger.info("GPU memory free: %.2f / %.2f GB", _free_gb, _total_gb)
            if _free_gb < 8.0:
                top_logger.warning("GPU has only %.2f GB free (< 8 GB threshold). "
                                   "Another process may be sharing this GPU, or VRAM "
                                   "is fragmented. Inference may fail with CUDA OOM.",
                                   _free_gb)
        except Exception as _e:
            top_logger.warning("Could not query GPU memory: %s", _e)
    else:
        top_logger.info("Device: CPU (Shapley analysis will be slow)")
    top_logger.info("PyTorch: %s", torch.__version__)

    # Hyperparameters (mirror interpretability_analysis.main()).
    with open(BEST_TCN_PATH, "r", encoding="utf-8") as f:
        _ = json.load(f)["hyperparameters"]  # not used directly, but kept for parity

    ms_hp = None
    branch_dilations = {"branch1": DEFAULT_BRANCH1,
                        "branch2": DEFAULT_BRANCH2,
                        "branch3": DEFAULT_BRANCH3}
    if BEST_MS_PATH.exists():
        with open(BEST_MS_PATH, "r", encoding="utf-8") as f:
            ms_cfg = json.load(f)
        ms_hp = ms_cfg["hyperparameters"]
        branch_dilations = ms_cfg.get("branch_dilations", branch_dilations)
        top_logger.info("Loaded multiscale HP from %s", BEST_MS_PATH)
    else:
        top_logger.error("best_multiscale_params.json missing: %s", BEST_MS_PATH)
        sys.exit(1)

    ms_attn_hp = None
    if BEST_MS_ATTN_PATH.exists():
        with open(BEST_MS_ATTN_PATH, "r", encoding="utf-8") as f:
            ms_attn_hp = json.load(f)["hyperparameters"]
        top_logger.info("Loaded multiscale attention HP from %s", BEST_MS_ATTN_PATH)

    # Shared data load (so the test set is read once for both models).
    pairs, loader, _y_true, _x_all, n_ictal, n_nonictal = load_partition_data(
        partition, batch_size=BATCH_SIZE, device=device, logger=top_logger)
    top_logger.info("Loaded %s partition: %d segments (%d ictal, %d non-ictal)",
                    partition, len(pairs), n_ictal, n_nonictal)

    summaries = {}
    for mname in requested:
        if mname == "M3":
            if not M3_WEIGHTS.exists():
                top_logger.error("M3 weights not found: %s. Skipping.", M3_WEIGHTS)
                continue
            mlogger, m_out_dir = setup_logging(interp_root, partition, "M3")
            model = load_model_weights("M3", M3_WEIGHTS, ms_hp, device, mlogger,
                                       branch_dilations=branch_dilations)
            summaries["M3"] = run_for_model(
                model, "M3", pairs, loader, partition, m_out_dir, device, mlogger)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif mname == "M4":
            if not M4_AVAILABLE:
                top_logger.error("MultiScaleTCNWithAttention class unavailable. Skipping M4.")
                continue
            if not M4_WEIGHTS.exists():
                top_logger.error("M4 weights not found: %s. Skipping.", M4_WEIGHTS)
                continue
            if ms_attn_hp is None:
                top_logger.error("best_multiscale_attn_params.json missing. Skipping M4.")
                continue
            mlogger, m_out_dir = setup_logging(interp_root, partition, "M4")
            model = load_model_weights("M4", M4_WEIGHTS, ms_hp, device, mlogger,
                                       attn_hp=ms_attn_hp,
                                       branch_dilations=branch_dilations)
            summaries["M4"] = run_for_model(
                model, "M4", pairs, loader, partition, m_out_dir, device, mlogger)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    top_logger.info("=" * 70)
    top_logger.info("Branch Shapley analysis complete.")
    top_logger.info("Models processed: %s", list(summaries.keys()))
    top_logger.info("Outputs under: %s", out_dir)


if __name__ == "__main__":
    main()
