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

Outputs (under the canonical per-model OUTPUT_ROOTs used by the training
and evaluation scripts)
------------------------------------------------------------------------
/home/people/22206468/scratch/OUTPUT/MODEL3_OUTPUT/MultiScaleTCN/
    interpret_branch_ablation/<partition>/
        m3_shapley_<partition>.csv               streaming per-segment CSV
        m3_shapley_<partition>_summary.json      population-level aggregates
        logs/branch_shapley_m3_<partition>.log   FileHandler log

/home/people/22206468/scratch/OUTPUT/MODEL4_OUTPUT/MultiScaleTCNAttention/
    interpret_branch_ablation/<partition>/
        m4_shapley_<partition>.csv
        m4_shapley_<partition>_summary.json
        logs/branch_shapley_m4_<partition>.log

Usage
-----
python branch_shapley_analysis.py --partition test --model M3
python branch_shapley_analysis.py --partition val  --model M4
python branch_shapley_analysis.py --partition test --model all

Prerequisites
-------------
Trained weights at the canonical per-model OUTPUT_ROOTs:
    MODEL3_OUTPUT/MultiScaleTCN/multiscale_tcn_final_weights.pt           (M3)
    MODEL4_OUTPUT/MultiScaleTCNAttention/ms_attn_final_weights.pt          (M4)
"""

import argparse
import csv
import datetime
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from tcn_utils import set_seed, make_loader
from interpretability_analysis import (
    SEED,
    DEFAULT_BRANCH1,
    DEFAULT_BRANCH2,
    DEFAULT_BRANCH3,
    M4_AVAILABLE,
    load_model_weights,
    compute_branch_outputs,
    forward_with_branch_mask,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DECISION_THRESHOLD = 0.5     # raw-output threshold matching run_branch_ablation
BATCH_SIZE = 32              # inherits the existing interpretability batch size

# Canonical splits manifest used by all training and evaluation scripts
# (TCN.py, TCNTemporalAttention.py, MultiScaleTCN.py, MultiScaleTCNAttention.py,
# MultiScaleTCN_evaluation.py, MultiScaleTCNAttention_evaluation.py). The
# enriched suffix indicates that each val/test record carries the
# chronology fields {mouse_id, chrono_idx, t_start_sec} added by
# enrich_manifest.py. Override with --splits-path if needed.
DEFAULT_SPLITS_PATH = Path(
    "/scratch/22206468/INPUT_DATA/data_splits_outputs/"
    "data_splits_nonictal_sampled_filtered_enriched.json")

# Per-model OUTPUT_ROOT, trained-weights paths, and tuning-output (best-
# hyperparameter) paths. These mirror the constants used by the training
# scripts (MultiScaleTCN.py:168,173,188; MultiScaleTCNAttention.py:160,174,175)
# and the evaluation scripts, so artefacts produced here land in the same
# per-model hierarchy on the cluster.
CLUSTER_OUTPUT = Path("/home/people/22206468/scratch/OUTPUT")
OUTPUT_ROOTS = {
    "M3": CLUSTER_OUTPUT / "MODEL3_OUTPUT" / "MultiScaleTCN",
    "M4": CLUSTER_OUTPUT / "MODEL4_OUTPUT" / "MultiScaleTCNAttention",
}
WEIGHTS_PATHS = {
    "M3": OUTPUT_ROOTS["M3"] / "multiscale_tcn_final_weights.pt",
    "M4": OUTPUT_ROOTS["M4"] / "ms_attn_final_weights.pt",
}
# Best-hyperparameter JSONs live in sibling tuning-output directories under
# each MODEL*_OUTPUT root, NOT inside the per-model OUTPUT_ROOT.
BEST_MS_PATH = (CLUSTER_OUTPUT / "MODEL3_OUTPUT"
                / "MultiScaleTCNtuning_outputs" / "best_multiscale_params.json")
BEST_MS_ATTN_PATH = (CLUSTER_OUTPUT / "MODEL4_OUTPUT"
                     / "multiscale_attention_tuning_outputs"
                     / "best_multiscale_attn_params.json")

# Subdirectory under each model's OUTPUT_ROOT for branch-ablation /
# Shapley artefacts. Partition (val|test) is appended as a sibling
# subdirectory so val and test artefacts are never mixed.
BRANCH_ABLATION_SUBDIR = "interpret_branch_ablation"


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
    parser.add_argument("--splits-path", type=Path, default=DEFAULT_SPLITS_PATH,
                        help="Path to the enriched splits manifest. Must match "
                             "the manifest used by the training and evaluation "
                             "scripts. Default: %(default)s")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------
def setup_logging(model_name, partition):
    """Create the per-model output directory under the model's OUTPUT_ROOT and
    configure a logger that writes to a FileHandler inside it.

    Layout:
        {OUTPUT_ROOTS[model_name]}/{BRANCH_ABLATION_SUBDIR}/{partition}/
            logs/branch_shapley_{model}_{partition}.log
            {model}_shapley_{partition}.csv
            {model}_shapley_{partition}_summary.json
    """
    out_dir = OUTPUT_ROOTS[model_name] / BRANCH_ABLATION_SUBDIR / partition
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
    logger.info("Output dir: %s", out_dir)
    logger.info("Log file:   %s", log_file)
    if partition == "val":
        logger.warning("PARTITION: VALIDATION -- carries threshold-selection bias. "
                       "Use --partition test for paper-quality attributions.")
    logger.info("=" * 70)
    return logger, out_dir


# ---------------------------------------------------------------------------
# load_partition_records
# ---------------------------------------------------------------------------
def load_partition_records(splits_path, partition, batch_size, device, logger):
    """Load the enriched manifest and build a sequential DataLoader.

    Asserts that the manifest is the enriched variant (the canonical one used
    by all training and evaluation scripts) by requiring (i) the path ends in
    '_enriched.json' and (ii) the val/test records carry the 'chrono_idx' /
    'mouse_id' / 't_start_sec' fields added by enrich_manifest.py.

    Returns
    -------
    records : list of dict -- full enriched records (mouse_id, chrono_idx,
              t_start_sec, filepath, label), aligned 1:1 with the loader.
    loader  : DataLoader -- shuffle=False so order matches `records`.
    """
    if not str(splits_path).endswith("_enriched.json"):
        logger.error("Splits path must end in '_enriched.json' "
                     "(canonical manifest used by training and evaluation). "
                     "Got: %s", splits_path)
        sys.exit(1)
    if not splits_path.exists():
        logger.error("Splits manifest not found: %s", splits_path)
        sys.exit(1)

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    records = list(splits.get(partition, []))
    if not records:
        logger.error("Partition '%s' is empty in %s", partition, splits_path)
        sys.exit(1)
    required = ("mouse_id", "chrono_idx", "t_start_sec", "filepath", "label")
    missing = [k for k in required if k not in records[0]]
    if missing:
        logger.error("Enriched fields missing on first record (%s). "
                     "Re-run enrich_manifest.py.", missing)
        sys.exit(1)

    pairs = [(r["filepath"], r["label"]) for r in records]
    loader = make_loader(pairs, batch_size, False, device)
    n_ictal = sum(1 for _, l in pairs if l == 1)
    n_nonictal = len(pairs) - n_ictal
    logger.info("Splits manifest: %s", splits_path)
    logger.info("Partition: %s | Total: %d | Ictal: %d (%.2f%%) | Non-ictal: %d",
                partition.upper(), len(pairs), n_ictal,
                100.0 * n_ictal / len(pairs), n_nonictal)
    return records, loader


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
def run_for_model(model, model_name, records, loader, partition, out_dir,
                  device, logger):
    """Stream per-segment Shapley rows to CSV; return a population summary dict.

    Efficiency-axiom sanity check is logged: per sample,
        phi_B1 + phi_B2 + phi_B3 == logit_full - logit_empty (within fp32 tolerance).

    `records` is the list of enriched manifest entries (one per segment, in
    loader order) -- mouse_id / chrono_idx / t_start_sec are read from each
    entry rather than parsed from the filename.
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
            "segment_id", "mouse_id", "chrono_idx", "t_start_sec",
            "file_basename", "y_true", "y_pred", "p_seizure",
            "logit_full", "logit_empty",
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
                    rec = records[fp_idx]
                    basename = Path(rec["filepath"]).stem
                    cat = categorise_prediction(int(y_true[j]), int(y_pred[j]))
                    counts[cat] += 1
                    abs_sum[cat] += abs_phis[j]
                    signed_sum[cat] += phis[j]
                    winner_counts[cat]["B%d" % (winners[j] + 1)] += 1
                    writer.writerow([
                        fp_idx,
                        rec["mouse_id"],
                        int(rec["chrono_idx"]),
                        "%.4f" % float(rec["t_start_sec"]),
                        basename,
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
                    logger.info("  ...processed %d / %d segments", n_seen, len(records))

    if n_seen != len(records):
        logger.error("Sample count mismatch: processed %d, expected %d.", n_seen, len(records))
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
# log_gpu_diagnostics
# ---------------------------------------------------------------------------
def log_gpu_diagnostics(logger):
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
                               "is fragmented. Inference may fail with CUDA OOM.",
                               _free_gb)
        except Exception as _e:
            logger.warning("Could not query GPU memory: %s", _e)
    else:
        logger.info("Device: CPU (Shapley analysis will be slow)")
    logger.info("PyTorch: %s", torch.__version__)


# ---------------------------------------------------------------------------
# load_hps
# ---------------------------------------------------------------------------
def load_hps(model_name, logger):
    """Load the hyperparameter JSONs needed to instantiate `model_name`.

    Returns (ms_hp, ms_attn_hp_or_None, branch_dilations). For M3, ms_attn_hp
    is always None. For M4, ms_attn_hp must be present or the caller should
    skip M4.
    """
    branch_dilations = {"branch1": DEFAULT_BRANCH1,
                        "branch2": DEFAULT_BRANCH2,
                        "branch3": DEFAULT_BRANCH3}
    if not BEST_MS_PATH.exists():
        logger.error("best_multiscale_params.json missing: %s", BEST_MS_PATH)
        sys.exit(1)
    with open(BEST_MS_PATH, "r", encoding="utf-8") as f:
        ms_cfg = json.load(f)
    ms_hp = ms_cfg["hyperparameters"]
    branch_dilations = ms_cfg.get("branch_dilations", branch_dilations)
    logger.info("Loaded multiscale HP from %s", BEST_MS_PATH)

    ms_attn_hp = None
    if model_name == "M4":
        if not BEST_MS_ATTN_PATH.exists():
            logger.error("best_multiscale_attn_params.json missing: %s. "
                         "Cannot instantiate M4.", BEST_MS_ATTN_PATH)
            return ms_hp, None, branch_dilations
        with open(BEST_MS_ATTN_PATH, "r", encoding="utf-8") as f:
            ms_attn_hp = json.load(f)["hyperparameters"]
        logger.info("Loaded multiscale attention HP from %s", BEST_MS_ATTN_PATH)
    return ms_hp, ms_attn_hp, branch_dilations


# ---------------------------------------------------------------------------
# run_session -- one complete model session: log -> data -> model -> shapley
# ---------------------------------------------------------------------------
def run_session(model_name, partition, splits_path, device):
    """Self-contained session for a single model. Each session writes its
    own complete log to {OUTPUT_ROOT}/interpret_branch_ablation/{partition}/logs/.
    """
    if model_name not in OUTPUT_ROOTS:
        raise ValueError("Unknown model: %s" % model_name)
    logger, out_dir = setup_logging(model_name, partition)
    log_gpu_diagnostics(logger)

    # Hyperparameters
    ms_hp, ms_attn_hp, branch_dilations = load_hps(model_name, logger)
    if model_name == "M4" and ms_attn_hp is None:
        logger.error("Skipping M4 (attention HPs unavailable).")
        return None
    if model_name == "M4" and not M4_AVAILABLE:
        logger.error("MultiScaleTCNWithAttention class unavailable. Skipping M4.")
        return None

    weights_path = WEIGHTS_PATHS[model_name]
    if not weights_path.exists():
        logger.error("%s weights not found: %s. Skipping.", model_name, weights_path)
        return None

    # Data (read once per session; the per-session cost is just JSON parse
    # plus DataLoader setup, no segment-file reads).
    records, loader = load_partition_records(
        splits_path, partition, batch_size=BATCH_SIZE, device=device, logger=logger)

    # Model
    model = load_model_weights(model_name, weights_path, ms_hp, device, logger,
                               attn_hp=ms_attn_hp,
                               branch_dilations=branch_dilations)

    summary = run_for_model(
        model, model_name, records, loader, partition, out_dir, device, logger)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    partition = args.partition
    requested = ["M3", "M4"] if args.model == "all" else [args.model]

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for mname in requested:
        run_session(mname, partition, args.splits_path, device)


if __name__ == "__main__":
    main()
