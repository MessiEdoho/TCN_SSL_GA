"""
raw_segment_level_3partitions.py
================================
Cluster script. Produces a raw segment-level (Row 1) cross-partition
comparison -- train vs val vs test -- as a single bar-plot and matching
CSV / JSON. Eight metrics are reported per partition, side by side:

  Accuracy, Recall (macro), Precision (macro), Specificity,
  AUROC, AP (PRAUC), F1 (macro), MCC.

Train metrics are computed live by running an FP32 inference pass of
the trained model on splits["train"] (proximity-aware downsampled,
~30% ictal prevalence). Val and test metrics are read from the
already-cached *_classification_report_row1.json files written by
the canonical *_evaluation.py runs (post-processing-invariant, so
unaffected by MIN_EVENT_SEC / ordering choices).

Why "raw segment level" / "Row 1"
---------------------------------
The training partition is proximity-aware downsampled, so its segments
are not chronologically contiguous. Smoothing (Row 2) and event-level
detection both assume time contiguity and are therefore not meaningful
for train. Row 1 (raw classifier at tau = 0.5, scored per segment) is
the only stage that is meaningful for all three partitions and is the
right comparison target for cross-partition diagnosis. Post-processed
(Row 2) and event-level metrics for val and test are reported separately
(plot_segment_metrics_barplot.py).

Note on prevalence-sensitive metrics
------------------------------------
Among the eight metrics reported, Accuracy and Specificity are
heavily prevalence-sensitive -- train is downsampled to ~30% ictal
while val and test are at the natural ~0.27% prevalence, so these
two metrics will look very different across partitions for reasons
unrelated to model behaviour. They are kept in the figure for
completeness of Row-1 reporting; for genuine cross-partition
generalisation diagnosis rely on the prevalence-robust metrics
(macro Recall, macro Precision, AUROC, AP, macro F1, MCC).

Pipeline
--------
1. Layer 2 dataset hardening (SafeEEGSegmentDataset).
2. FP32 forward pass on splits["train"] with per-batch isfinite
   assertion (Layer 4).
3. compute_segment_level_metrics on Row 1.
4. Build sklearn classification report (with macro precision / recall /
   f1 + MCC + AUROC + AP + specificity surfaced at top level).
5. Save train_evaluation_report.json + train_classification_report.json
   + train_predictions_raw.npz.
6. Read val and test eval reports from the cluster (handles both old
   and new schemas; Row 1 values are identical across schemas).
7. Build train-vs-val-vs-test comparison (8 metrics x 3 partitions).
8. Save partition_comparison.{json,csv,png}.

Usage
-----
  python raw_segment_level_3partitions.py --variant MultiScaleTCN
  python raw_segment_level_3partitions.py --variant MultiScaleTCNWithAttention
  python raw_segment_level_3partitions.py --variant TCN                       # future
  python raw_segment_level_3partitions.py --variant TCNWithAttention          # future

Override CLI flags only if defaults don't match (see VARIANT_CONFIG below).
"""

import argparse
import csv
import datetime
import json
import logging
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import tcn_utils
from eval_utils import (
    SafeEEGSegmentDataset,
    AMPLITUDE_THRESHOLD,
    make_safe_loader,
    evaluate_model_safe,
    compute_segment_level_metrics,
    build_classification_report,
    THRESHOLD,
)


# ---------------------------------------------------------------------------
# Per-variant defaults (override via CLI flags if a path differs)
# ---------------------------------------------------------------------------
SCRATCH = Path("/home/people/22206468/scratch")
SPLITS_PATH = SCRATCH / "INPUT_DATA" / "data_splits_outputs" / "data_splits_nonictal_sampled_filtered.json"

VARIANT_CONFIG = {
    "TCN": {
        "output_root":  SCRATCH / "OUTPUT" / "MODEL1_OUTPUT" / "TCN",
        "prefix":       "tcn",
        "weights":      "tcn_final_weights.pt",
        "params":       SCRATCH / "OUTPUT" / "MODEL1_OUTPUT" / "TCNtuning_outputs" / "best_params.json",
        "attn_params":  None,
        "val_report":   "tcn_evaluation_report.json",
        "test_report":  "evaluation/tcn_evaluation_report.json",
    },
    "TCNWithAttention": {
        "output_root":  SCRATCH / "OUTPUT" / "MODEL2_OUTPUT" / "TCNAttention",
        "prefix":       "tcn_attention",
        "weights":      "tcn_attention_final_weights.pt",
        "params":       SCRATCH / "OUTPUT" / "MODEL1_OUTPUT" / "TCNtuning_outputs" / "best_params.json",
        "attn_params":  SCRATCH / "OUTPUT" / "MODEL2_OUTPUT" / "best_attention_params.json",
        "val_report":   "tcn_attention_evaluation_report.json",
        "test_report":  "evaluation/tcn_attention_evaluation_report.json",
    },
    "MultiScaleTCN": {
        "output_root":  SCRATCH / "OUTPUT" / "MODEL3_OUTPUT" / "MultiScaleTCN",
        "prefix":       "multiscale_tcn",
        "weights":      "multiscale_tcn_final_weights.pt",
        "params":       SCRATCH / "OUTPUT" / "MODEL3_OUTPUT" / "MultiScaleTCNtuning_outputs" / "best_multiscale_params.json",
        "attn_params":  None,
        "val_report":   "multiscale_tcn_evaluation_report.json",
        "test_report":  "evaluation/multiscale_tcn_evaluation_report.json",
    },
    "MultiScaleTCNWithAttention": {
        "output_root":  SCRATCH / "OUTPUT" / "MODEL4_OUTPUT" / "MultiScaleTCNAttention",
        "prefix":       "ms_attn",
        "weights":      "ms_attn_final_weights.pt",
        "params":       SCRATCH / "OUTPUT" / "MODEL3_OUTPUT" / "MultiScaleTCNtuning_outputs" / "best_multiscale_params.json",
        "attn_params":  SCRATCH / "OUTPUT" / "MODEL4_OUTPUT" / "multiscale_attention_tuning_outputs" / "best_multiscale_attn_params.json",
        # User renamed M4 val report folder to reflect content:
        "val_report":   "val_event_metrics/ms_attn_evaluation_report.json",
        "test_report":  "evaluation/ms_attn_evaluation_report.json",
    },
}


COMPARISON_METRICS = [
    ("Accuracy",    "accuracy"),
    ("Recall",      "recall_macro"),
    ("Precision",   "precision_macro"),
    ("Specificity", "specificity"),
    ("AUROC",       "auroc"),
    ("AP (PRAUC)",  "average_precision"),
    ("F1",          "f1_macro"),
    ("MCC",         "mcc"),
]


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant",     required=True, choices=list(VARIANT_CONFIG.keys()))
    p.add_argument("--weights",     type=Path, default=None,
                   help="Override default weights path.")
    p.add_argument("--params",      type=Path, default=None,
                   help="Override default best_*_params.json path.")
    p.add_argument("--attn-params", type=Path, default=None,
                   help="Override default best_*_attn_params.json path (for *Attention variants).")
    p.add_argument("--output-dir",  type=Path, default=None,
                   help="Override default {OUTPUT_ROOT}/train_evaluation/.")
    p.add_argument("--val-report",  type=Path, default=None,
                   help="Override default val eval-report path.")
    p.add_argument("--test-report", type=Path, default=None,
                   help="Override default test eval-report path.")
    p.add_argument("--splits-path", type=Path, default=SPLITS_PATH)
    p.add_argument("--batch-size",  type=int,  default=32)
    p.add_argument("--device",      default=None)
    return p.parse_args()


def resolve_paths(args):
    cfg = VARIANT_CONFIG[args.variant]
    output_root = cfg["output_root"]
    return {
        "prefix":      cfg["prefix"],
        "weights":     args.weights     or (output_root / cfg["weights"]),
        "params":      args.params      or cfg["params"],
        "attn_params": args.attn_params or cfg["attn_params"],
        "output_dir":  args.output_dir  or (output_root / "train_evaluation"),
        "val_report":  args.val_report  or (output_root / cfg["val_report"]),
        "test_report": args.test_report or (output_root / cfg["test_report"]),
        "splits_path": args.splits_path,
    }


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------
def setup_logging(output_dir, prefix):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{prefix}_raw_segment_level_3partitions.log"
    logger = logging.getLogger("raw_segment_level_3partitions")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8"); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger, log_path


# ---------------------------------------------------------------------------
# build_model -- mirrors deploy_inference.build_model for the 4 variants
# ---------------------------------------------------------------------------
def build_model(variant, hp, attn_hp, branch_dilations, device, logger):
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
        bd = branch_dilations or {"branch1":[1,2,4],"branch2":[8,16,32],"branch3":[32,64,128]}
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
        bd = branch_dilations or {"branch1":[1,2,4],"branch2":[8,16,32],"branch3":[32,64,128]}
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
# extract_row1 -- handle both old and new eval-report schemas
# ---------------------------------------------------------------------------
def extract_row1(report_path, prefix, logger):
    """Returns the Row 1 (raw at tau=0.5) segment-level metric dict from
    either the OLD schema (top-level row1_raw_threshold_0_5) or the
    NEW schema (segment_level_metrics.row1_raw_threshold_0_5).

    Back-fills `precision_macro` and `recall_macro` from the sibling
    classification-report JSON's "macro avg" block if those keys are
    missing (cached reports written before eval_utils added them).
    """
    if not report_path.exists():
        logger.warning("Eval report not found: %s -- comparison cell will be NA.", report_path)
        return None
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    if "segment_level_metrics" in rep:
        row1 = rep["segment_level_metrics"].get("row1_raw_threshold_0_5") or {}
    else:
        row1 = rep.get("row1_raw_threshold_0_5") or {}
    row1 = dict(row1)

    if "precision_macro" not in row1 or "recall_macro" not in row1:
        sibling = report_path.parent / f"{prefix}_classification_report_row1.json"
        if sibling.exists():
            cr = json.loads(sibling.read_text(encoding="utf-8"))
            ma = cr.get("macro avg") or {}
            row1.setdefault("precision_macro", ma.get("precision"))
            row1.setdefault("recall_macro",    ma.get("recall"))

    if "mcc" not in row1:
        tp = float(row1.get("tp", 0)); fp = float(row1.get("fp", 0))
        fn = float(row1.get("fn", 0)); tn = float(row1.get("tn", 0))
        denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        row1["mcc"] = ((tp * tn - fp * fn) / denom) if denom > 0 else 0.0
    return row1


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    args  = parse_args()
    paths = resolve_paths(args)
    logger, log_path = setup_logging(paths["output_dir"], paths["prefix"])

    logger.info("=" * 65)
    logger.info("raw_segment_level_3partitions.py")
    logger.info("Variant     : %s", args.variant)
    logger.info("Weights     : %s", paths["weights"])
    logger.info("Params      : %s", paths["params"])
    logger.info("Attn params : %s", paths["attn_params"])
    logger.info("Splits      : %s", paths["splits_path"])
    logger.info("Output dir  : %s", paths["output_dir"])
    logger.info("Val  report : %s", paths["val_report"])
    logger.info("Test report : %s", paths["test_report"])
    logger.info("Log         : %s", log_path)
    logger.info("=" * 65)

    for p in [paths["weights"], paths["params"], paths["splits_path"]]:
        if not Path(p).exists():
            logger.error("Missing required file: %s", p); sys.exit(1)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Device      : %s", device)
    if torch.cuda.is_available() and device.type == "cuda":
        logger.info("GPU  : %s", torch.cuda.get_device_name(0))
        logger.info("VRAM : %.2f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)
        logger.info("CUDA : %s", torch.version.cuda)
        try:
            _free, _total = torch.cuda.mem_get_info(0)
            logger.info("GPU memory free: %.2f / %.2f GB", _free / 1e9, _total / 1e9)
        except Exception as _e:
            logger.warning("Could not query GPU memory: %s", _e)
    logger.info("PyTorch     : %s", torch.__version__)

    # -- Load HPs ------------------------------------------------------------
    backbone_cfg = json.loads(Path(paths["params"]).read_text(encoding="utf-8"))
    hp = backbone_cfg.get("hyperparameters", backbone_cfg)
    branch_dilations = backbone_cfg.get("branch_dilations", None)
    attn_hp = None
    if paths["attn_params"] and Path(paths["attn_params"]).exists():
        attn_cfg = json.loads(Path(paths["attn_params"]).read_text(encoding="utf-8"))
        attn_hp  = attn_cfg.get("hyperparameters", attn_cfg)

    # -- Load splits manifest, take train ------------------------------------
    splits = json.loads(Path(paths["splits_path"]).read_text(encoding="utf-8"))
    train_records = splits.get("train", [])
    if not train_records:
        logger.error("Train partition is empty in manifest %s", paths["splits_path"]); sys.exit(1)
    train_pairs = [(rec["filepath"], int(rec["label"])) for rec in train_records]
    n_total = len(train_pairs)
    n_sz    = sum(1 for _, l in train_pairs if l == 1)
    logger.info("TRAIN: %d total | %d ictal | %d non-ictal | %.1f%% ictal",
                n_total, n_sz, n_total - n_sz, 100 * n_sz / n_total)

    # -- Build model + load weights -----------------------------------------
    model = build_model(args.variant, hp, attn_hp, branch_dilations, device, logger)
    state_dict = torch.load(paths["weights"], map_location=device)
    model.load_state_dict(state_dict)
    logger.info("Loaded weights from %s", paths["weights"])

    # -- FP32 forward pass on train ------------------------------------------
    SafeEEGSegmentDataset.n_sanitised = 0
    loader = make_safe_loader(train_pairs, args.batch_size, device, num_workers=4)
    logger.info("Train loader: %d batches (batch_size=%d)", len(loader), args.batch_size)

    t0 = time.time()
    train_f1, y_true, _y_pred, y_prob = evaluate_model_safe(
        model, loader, device, logger, use_amp=False)
    eval_seconds = time.time() - t0
    logger.info("Forward pass complete: %d segments in %.1f s | "
                "raw t=0.5 macro F1 = %.4f", len(y_true), eval_seconds, train_f1)
    sanitised = SafeEEGSegmentDataset.n_sanitised
    if sanitised:
        logger.warning("SafeEEGSegmentDataset sanitised %d train segments.", sanitised)

    # -- Compute segment-level Row 1 metrics --------------------------------
    n_non_ic = int(np.sum(y_true == 0))
    y_pred_row1  = (y_prob >= THRESHOLD).astype(np.int64)
    train_metrics = compute_segment_level_metrics(y_true, y_pred_row1, y_prob, n_non_ic)
    train_metrics["threshold"]     = THRESHOLD
    train_metrics["postprocessed"] = False

    train_report = build_classification_report(y_true, y_pred_row1, train_metrics)

    # -- Save train artefacts ------------------------------------------------
    out = paths["output_dir"]
    pfx = paths["prefix"]
    train_eval_report_path = out / f"{pfx}_train_evaluation_report.json"
    train_classrep_path    = out / f"{pfx}_train_classification_report.json"
    train_npz_path         = out / f"{pfx}_train_predictions_raw.npz"

    train_eval_report = {
        "model":                args.variant,
        "timestamp":            datetime.datetime.now().isoformat(),
        "weights_path":         str(paths["weights"]),
        "evaluation_set":       "train",
        "evaluation_set_note":  ("Train partition was proximity-aware downsampled "
                                 "(~30%% ictal). Only segment-level Row 1 metrics "
                                 "are computed -- Row 2 (post-processing) and "
                                 "event-level metrics require time-contiguous "
                                 "segments which the downsampled train does not "
                                 "provide. Use Sensitivity / AUROC / AP / F1 for "
                                 "fair train-vs-val-vs-test comparison."),
        "n_segments":           int(len(y_true)),
        "n_segments_sanitised": int(sanitised),
        "n_ictal":              int(n_sz),
        "n_non_ictal":          int(n_total - n_sz),
        "ictal_prevalence":     round(n_sz / n_total, 6),
        "fp32_forward":         True,
        "forward_pass_seconds": round(float(eval_seconds), 1),
        "post_processing_params": {
            "threshold":         THRESHOLD,
            "amplitude_threshold": AMPLITUDE_THRESHOLD,
        },
        "row1_raw_threshold_0_5": train_metrics,
    }
    train_eval_report_path.write_text(
        json.dumps(train_eval_report, indent=2, default=str), encoding="utf-8")
    logger.info("Saved: %s", train_eval_report_path)

    train_classrep_path.write_text(
        json.dumps(train_report, indent=2, default=str), encoding="utf-8")
    logger.info("Saved: %s", train_classrep_path)

    np.savez_compressed(
        train_npz_path,
        y_true=y_true.astype(np.int8),
        y_prob=y_prob.astype(np.float32),
        n_segments=np.int64(len(y_true)),
    )
    logger.info("Saved: %s (%.2f MB)", train_npz_path, train_npz_path.stat().st_size / 1e6)

    # -- Read val + test reports and build comparison ------------------------
    val_row1  = extract_row1(Path(paths["val_report"]),  paths["prefix"], logger)
    test_row1 = extract_row1(Path(paths["test_report"]), paths["prefix"], logger)

    comparison = {"metric": [], "train": [], "val": [], "test": []}
    for label, key in COMPARISON_METRICS:
        comparison["metric"].append(label)
        comparison["train"].append(round(float(train_metrics.get(key, 0.0)), 6))
        comparison["val"].append(round(float(val_row1.get(key, 0.0)), 6) if val_row1 else None)
        comparison["test"].append(round(float(test_row1.get(key, 0.0)), 6) if test_row1 else None)

    cmp_json_path = out / f"{pfx}_partition_comparison.json"
    cmp_csv_path  = out / f"{pfx}_partition_comparison.csv"
    cmp_png_path  = out / f"{pfx}_partition_comparison.png"

    cmp_json_path.write_text(json.dumps({
        "model":                  args.variant,
        "timestamp":              datetime.datetime.now().isoformat(),
        "metrics_compared":       [m for m, _ in COMPARISON_METRICS],
        "metrics_excluded_note":  ("Class-1 precision and FAR-hr are excluded "
                                   "from the train-vs-val-vs-test comparison "
                                   "because they are heavily prevalence-sensitive "
                                   "(train ~30%% ictal; val ~0.4%%; test ~0.24%%). "
                                   "Accuracy is included for completeness as "
                                   "Row-1 reporting, but it is also prevalence-"
                                   "sensitive -- the prevalence-robust signals "
                                   "are macro Recall, macro Precision, "
                                   "Specificity, AUROC, AP, macro F1, and MCC."),
        "comparison":             comparison,
        "sources": {
            "train_report": str(train_eval_report_path),
            "val_report":   str(paths["val_report"]),
            "test_report":  str(paths["test_report"]),
        },
    }, indent=2, default=str), encoding="utf-8")
    logger.info("Saved: %s", cmp_json_path)

    with open(cmp_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "train", "val", "test"])
        for i in range(len(comparison["metric"])):
            w.writerow([comparison["metric"][i],
                        comparison["train"][i],
                        comparison["val"][i] if comparison["val"][i] is not None else "NA",
                        comparison["test"][i] if comparison["test"][i] is not None else "NA"])
    logger.info("Saved: %s", cmp_csv_path)

    # -- Plot ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(comparison["metric"]))
    w = 0.25
    train_vals = comparison["train"]
    val_vals   = [v if v is not None else 0 for v in comparison["val"]]
    test_vals  = [v if v is not None else 0 for v in comparison["test"]]
    b_tr = ax.bar(x - w, train_vals, w, color="#5A7DC8", label="Train")
    b_v  = ax.bar(x,     val_vals,   w, color="#E8A87C", label="Validation")
    b_t  = ax.bar(x + w, test_vals,  w, color="#41B3A3", label="Test")
    for bars in (b_tr, b_v, b_t):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate("%.3f" % h,
                            xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(comparison["metric"], fontsize=10)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_title("%s -- Train vs Validation vs Test (segment-level Row 1)" % args.variant)
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig(cmp_png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", cmp_png_path)

    # -- Print summary -------------------------------------------------------
    logger.info("=" * 65)
    logger.info("PARTITION COMPARISON (%s)", args.variant)
    logger.info("  %-15s %10s %10s %10s", "metric", "train", "val", "test")
    for i, m in enumerate(comparison["metric"]):
        v = comparison["val"][i]
        t = comparison["test"][i]
        logger.info("  %-15s %10.4f %10s %10s",
                    m,
                    comparison["train"][i],
                    "%.4f" % v if v is not None else "NA",
                    "%.4f" % t if t is not None else "NA")
    logger.info("=" * 65)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("DONE")


if __name__ == "__main__":
    main()
