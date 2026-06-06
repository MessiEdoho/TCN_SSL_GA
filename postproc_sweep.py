"""
postproc_sweep.py
=================
Sweep post-processing parameters (refractory-vs-min-duration order;
MIN_EVENT_SEC) for any of the four model variants. By default sweeps
val + test from cached _full predictions NPZ; pass --include-train to
additionally sweep the full un-downsampled train partition (separate
cached NPZ produced by full_train_eval_*.py). CPU-only, no model
forward, no GPU.

Sweep grid per partition:
    order ∈ {refractory_then_min, min_then_refractory}
    MIN_EVENT_SEC ∈ {10, 15, 20, 25, 30}    (seconds)
= 10 configurations per partition × 2 partitions = 20 evaluations
  (40 if --include-train).

Smoothing window (W=3), threshold (tau=0.5), refractory period (30 s),
and FAR/hr denominator (step_sec = 2.5 s) are held at canonical values;
only the post-processing order and MIN_EVENT_SEC vary.

Two orderings under comparison:
  - "refractory_then_min" (legacy default): refractory merge, then drop
    survivors shorter than MIN_EVENT_SEC. Can rescue fragmented true
    detections (and fragmented false alarms).
  - "min_then_refractory" (clinical-standard order; Saab 2020, Tang 2022,
    Persyst, Encevis): drop short candidate events first, then merge any
    survivors that are < 30 s apart. Strictly more conservative.

Variant + partition + order sweep is the methodological ablation the
paper reports as "Section X.Y: Post-processing parameter ablation".

Dual-mode execution
-------------------
By default reads/writes the local Windows-mirror layout under
TCN_UNIQURE_PROJECT/. Pass --cluster to switch to the cluster paths under
/home/people/22206468/scratch/OUTPUT/. Override individual paths via
--val-npz / --test-npz / --val-annot-dir / --test-annot-dir / --metadata
/ --output-dir.

Variant flag
------------
    python postproc_sweep.py --variant MultiScaleTCN              # M3
    python postproc_sweep.py --variant MultiScaleTCNWithAttention # M4
    python postproc_sweep.py --variant TCN                        # M1 (when NPZs exist)
    python postproc_sweep.py --variant TCNWithAttention           # M2 (when NPZs exist)

Outputs (per variant, under <output-dir>)
    val/
        refractory_then_min/  MIN_EVENT_SEC_10s/  15s/  20s/  30s/
        min_then_refractory/  MIN_EVENT_SEC_10s/  15s/  20s/  30s/
    test/
        refractory_then_min/  MIN_EVENT_SEC_10s/  15s/  20s/  30s/
        min_then_refractory/  MIN_EVENT_SEC_10s/  15s/  20s/  30s/
    comparison_val_vs_test.csv         (20 rows: partition x order x sec)
    impact_val_vs_test.png             (2x3 panels: rows = partitions)
    postproc_sweep.log

Train sweep (with --include-train) is written to a SIBLING output root
post_process_varing_sec_train/ under each variant, so it never mixes
with the val/test artefacts:
    full_train_evaluation/post_process_varing_sec_train/
        train/  refractory_then_min/  MIN_EVENT_SEC_*s/
                min_then_refractory/  MIN_EVENT_SEC_*s/
        comparison_train.csv           (10 rows)
        impact_train.png               (1x3 panels)
        postproc_sweep.log
"""

import argparse
import csv
import datetime
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import eval_utils
from eval_utils import (
    evaluate_event_level,
    write_event_level_bundle,
)


CLUSTER_OUTPUT  = Path("/home/people/22206468/scratch/OUTPUT")
CLUSTER_SCRATCH = Path("/home/people/22206468/scratch")
LOCAL_DEFAULT   = Path(r"C:\Users\messi\OneDrive\Desktop\Desktop\TCN_UNIQURE_PROJECT")

PARTITIONS = ["val", "test"]
ORDERINGS  = ["refractory_then_min", "min_then_refractory"]
SWEEP_SECS = [10, 15, 20, 25, 30]

ORDER_LABEL = {
    "refractory_then_min": "Refractory -> Min-dur",
    "min_then_refractory": "Min-dur -> Refractory",
}
ORDER_COLOR = {
    "refractory_then_min": "#5A7DC8",
    "min_then_refractory": "#C85A5A",
}
ORDER_LINESTYLE = {
    "refractory_then_min": "-",
    "min_then_refractory": "--",
}
ORDER_MARKER = {
    "refractory_then_min": "o",
    "min_then_refractory": "^",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", required=True,
                   choices=["TCN", "TCNWithAttention",
                            "MultiScaleTCN", "MultiScaleTCNWithAttention"])
    p.add_argument("--cluster", action="store_true",
                   help="Use cluster paths (/home/people/22206468/scratch/...). "
                        "Default: local Windows-mirror paths.")
    p.add_argument("--local-root",   type=Path, default=LOCAL_DEFAULT)
    p.add_argument("--cluster-root", type=Path, default=CLUSTER_OUTPUT,
                   help="Override the cluster OUTPUT root (default /home/people/22206468/scratch/OUTPUT).")
    p.add_argument("--val-npz",        type=Path, default=None, help="Override val NPZ path.")
    p.add_argument("--test-npz",       type=Path, default=None, help="Override test NPZ path.")
    p.add_argument("--train-npz",      type=Path, default=None,
                   help="Override train NPZ path (only used with --include-train).")
    p.add_argument("--metadata",       type=Path, default=None, help="Override mouse metadata JSON path.")
    p.add_argument("--val-annot-dir",  type=Path, default=None, help="Override val annotations dir.")
    p.add_argument("--test-annot-dir", type=Path, default=None, help="Override test annotations dir.")
    p.add_argument("--train-annot-dir",type=Path, default=None,
                   help="Override train annotations dir (only used with --include-train).")
    p.add_argument("--manifest",       type=Path, default=None,
                   help="Override manifest path (enriched preferred). Used only "
                        "for _raw-format NPZs that lack embedded chronology.")
    p.add_argument("--train-manifest", type=Path, default=None,
                   help="Override train manifest path (only used with --include-train). "
                        "Default points at data_splits_full_train_enriched.json.")
    p.add_argument("--output-dir",     type=Path, default=None, help="Override output root (val/test).")
    p.add_argument("--train-output-dir", type=Path, default=None,
                   help="Override train output root (only used with --include-train). "
                        "Default = <variant>/full_train_evaluation/post_process_varing_sec_train.")
    p.add_argument("--include-train", action="store_true",
                   help="Additionally sweep the full un-downsampled train partition. "
                        "Writes to a sibling output root, untouched val/test outputs.")
    return p.parse_args()


def variant_config(local_root, cluster_root, cluster_mode):
    """Return per-variant input/output paths plus partition-level annotation
    dirs and the global mouse_metadata path. Cluster mode collapses val and
    test annotations into the single shared folder (build_chronology.py
    convention); local mirror keeps them in separate folders.
    """
    if cluster_mode:
        m1 = cluster_root / "MODEL1_OUTPUT" / "TCN"
        m2 = cluster_root / "MODEL2_OUTPUT" / "TCNAttention"
        m3 = cluster_root / "MODEL3_OUTPUT" / "MultiScaleTCN"
        m4 = cluster_root / "MODEL4_OUTPUT" / "MultiScaleTCNAttention"
        val_annot   = CLUSTER_SCRATCH / "seizure_times_updated"
        test_annot  = CLUSTER_SCRATCH / "seizure_times_updated"
        train_annot = CLUSTER_SCRATCH / "seizure_times_updated"
        metadata    = CLUSTER_SCRATCH / "INPUT_DATA" / "Data_diagnostic" / "mouse_recording_metadata.json"
        manifest    = CLUSTER_SCRATCH / "INPUT_DATA" / "data_splits_outputs" / "data_splits_nonictal_sampled_filtered_enriched.json"
        train_manifest = CLUSTER_SCRATCH / "INPUT_DATA" / "data_splits_outputs" / "data_splits_full_train_enriched.json"
    else:
        m1 = local_root / "TCN"
        m2 = local_root / "TCNAttention"
        m3 = local_root / "MultiScaleTCN"
        m4 = local_root / "MultiScaleTCNAttention"
        val_annot   = local_root / "val_seizure_annot_updated"
        test_annot  = local_root / "test_seizure_annot_updated"
        train_annot = local_root / "train_seizure_annot_updated"
        metadata    = local_root / "mouse_recording_metadata.json"
        manifest    = local_root / "data_splits_nonictal_sampled_filtered_enriched.json"
        train_manifest = local_root / "data_splits_full_train_enriched.json"

    variants = {
        "TCN": {
            "model_label": "M1 (TCN)",
            "val_npz":   m1 / "tcn_val_predictions_full.npz",
            "test_npz":  m1 / "evaluation" / "tcn_test_predictions_full.npz",
            "train_npz": m1 / "full_train_evaluation" / "tcn_full_train_predictions_full.npz",
            "output":         m1 / "evaluation" / "post_process_varing_sec",
            "output_train":   m1 / "full_train_evaluation" / "post_process_varing_sec_train",
        },
        "TCNWithAttention": {
            "model_label": "M2 (TCNAttention)",
            "val_npz":   m2 / "tcn_attention_val_predictions_full.npz",
            "test_npz":  m2 / "evaluation" / "tcn_attention_test_predictions_full.npz",
            "train_npz": m2 / "full_train_evaluation" / "tcn_attention_full_train_predictions_full.npz",
            "output":         m2 / "evaluation" / "post_process_varing_sec",
            "output_train":   m2 / "full_train_evaluation" / "post_process_varing_sec_train",
        },
        "MultiScaleTCN": {
            "model_label": "M3 (MultiScaleTCN)",
            # Cluster has the older _raw NPZ format (manifest-order, no
            # chronology fields). load_records_and_arrays auto-detects this
            # and falls back to manifest-based enrichment.
            "val_npz":   m3 / "multiscale_tcn_predictions_raw.npz",
            "test_npz":  m3 / "evaluation" / "multiscale_tcn_test_predictions_raw.npz",
            "train_npz": m3 / "full_train_evaluation" / "multiscale_tcn_full_train_predictions_full.npz",
            "output":         m3 / "evaluation" / "post_process_varing_sec",
            "output_train":   m3 / "full_train_evaluation" / "post_process_varing_sec_train",
        },
        "MultiScaleTCNWithAttention": {
            "model_label": "M4 (MultiScaleTCNAttention)",
            "val_npz":   m4 / "val_event_metrics" / "ms_attn_val_predictions_full.npz",
            "test_npz":  m4 / "evaluation" / "ms_attn_test_predictions_full.npz",
            "train_npz": m4 / "full_train_evaluation" / "ms_attn_full_train_predictions_full.npz",
            "output":         m4 / "evaluation" / "post_process_varing_sec",
            "output_train":   m4 / "full_train_evaluation" / "post_process_varing_sec_train",
        },
    }
    return variants, val_annot, test_annot, train_annot, metadata, manifest, train_manifest


def resolve_paths(args):
    variants, val_annot, test_annot, train_annot, metadata, manifest, train_manifest = variant_config(
        args.local_root, args.cluster_root, args.cluster)
    vcfg = variants[args.variant]
    return {
        "model_label":     vcfg["model_label"],
        "val_npz":         args.val_npz         or vcfg["val_npz"],
        "test_npz":        args.test_npz        or vcfg["test_npz"],
        "train_npz":       args.train_npz       or vcfg["train_npz"],
        "val_annot_dir":   args.val_annot_dir   or val_annot,
        "test_annot_dir":  args.test_annot_dir  or test_annot,
        "train_annot_dir": args.train_annot_dir or train_annot,
        "metadata":        args.metadata        or metadata,
        "manifest":        args.manifest        or manifest,
        "train_manifest":  args.train_manifest  or train_manifest,
        "output":          args.output_dir      or vcfg["output"],
        "output_train":    args.train_output_dir or vcfg["output_train"],
    }


def setup_logging(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "postproc_sweep.log"
    logger = logging.getLogger("postproc_sweep")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8"); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger, log_path


def load_records_and_arrays(npz_path, manifest_path, partition, annot_dir,
                            mouse_metadata, logger):
    """Load (y_true, y_prob, records) from cached predictions with NPZ-format
    auto-detection. Returns arrays aligned with records[i].

    Supports three input shapes:

      1. **_full NPZ** -- chronologically reordered, has mouse_id /
         chrono_idx / t_start_sec arrays embedded. Records built directly
         from NPZ; manifest is not consulted. This is what
         *_evaluation.py and m4_event_metrics_recovery.py write.

      2. **_raw NPZ + enriched manifest** -- NPZ in manifest order with
         only y_true/y_prob; manifest records have chrono_idx and
         t_start_sec. Records built from manifest, paired with NPZ by
         index. This is the M3 cluster case today.

      3. **_raw NPZ + unenriched manifest** -- last-resort fallback. The
         chronology is rebuilt in-process via
         eval_utils.build_mouse_chronology from EDF metadata + Excel
         annotations (~1 min for the full partition). Requires no
         precomputed chronology cache.
    """
    npz = np.load(npz_path, allow_pickle=True)
    if "y_true" not in npz.files or "y_prob" not in npz.files:
        raise KeyError(
            "NPZ %s missing y_true/y_prob. Available: %s"
            % (npz_path, list(npz.files)))
    y_true = np.asarray(npz["y_true"]).astype(np.int64)
    y_prob = np.asarray(npz["y_prob"]).astype(np.float64)

    has_chrono_in_npz = all(
        k in npz.files for k in ("mouse_id", "chrono_idx", "t_start_sec"))

    # Case 1: _full NPZ
    if has_chrono_in_npz:
        logger.info("NPZ format: _full (chronology fields embedded; "
                    "manifest not needed)")
        mouse_id_arr    = np.asarray(npz["mouse_id"])
        chrono_idx_arr  = np.asarray(npz["chrono_idx"]).astype(np.int64)
        t_start_sec_arr = np.asarray(npz["t_start_sec"]).astype(np.float64)
        records = []
        for i in range(len(y_true)):
            records.append({
                "mouse_id":    str(mouse_id_arr[i]),
                "chrono_idx":  int(chrono_idx_arr[i]),
                "t_start_sec": float(t_start_sec_arr[i]),
                "label":       int(y_true[i]),
                "filepath":    "",
            })
        logger.info("Built %d records from NPZ (%d unique mice)",
                    len(records), len({r["mouse_id"] for r in records}))
        return y_true, y_prob, records

    # Cases 2 / 3: _raw NPZ -- need manifest
    logger.info("NPZ format: _raw (manifest order; chronology not embedded)")
    if manifest_path is None or not manifest_path.exists():
        raise FileNotFoundError(
            "NPZ %s is _raw format and requires a manifest, but manifest "
            "path %s does not exist. Pass --manifest <PATH>."
            % (npz_path, manifest_path))
    logger.info("Loading manifest at %s", manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_records = manifest.get(partition, []) or []
    if len(manifest_records) != len(y_true):
        raise ValueError(
            "NPZ length %d != manifest partition '%s' length %d."
            % (len(y_true), partition, len(manifest_records)))

    sample = manifest_records[:5]
    manifest_enriched = bool(sample) and all(
        ("chrono_idx" in r and "t_start_sec" in r) for r in sample)

    kept_idx, records = [], []
    if manifest_enriched:
        logger.info("Manifest is enriched -- using its chronology fields directly.")
        for i, rec in enumerate(manifest_records):
            ci = rec.get("chrono_idx", -1)
            if ci is None or int(ci) < 0:
                continue
            r2 = dict(rec)
            r2.setdefault("mouse_id", Path(rec["filepath"]).stem.split("_")[0])
            r2["chrono_idx"]  = int(ci)
            r2["t_start_sec"] = float(rec.get("t_start_sec", 0.0))
            records.append(r2)
            kept_idx.append(i)
    else:
        logger.info("Manifest is unenriched -- rebuilding chronology in-process "
                    "via eval_utils.build_mouse_chronology.")
        from eval_utils import build_mouse_chronology as _bmc
        from eval_utils import load_annotations as _load_annot
        mice = sorted({Path(r["filepath"]).stem.split("_")[0] for r in manifest_records})
        chrono_maps = {}
        for mouse_id in mice:
            if mouse_id not in mouse_metadata:
                logger.warning("  %s : no metadata; dropping its records.", mouse_id)
                chrono_maps[mouse_id] = None
                continue
            xlsx = annot_dir / f"{mouse_id}_xlsx.xlsx"
            if not xlsx.exists():
                logger.warning("  %s : no annotation at %s; dropping its records.",
                               mouse_id, xlsx)
                chrono_maps[mouse_id] = None
                continue
            meta = mouse_metadata[mouse_id]
            rec_start = datetime.datetime.fromisoformat(meta["recording_start_dt"])
            seizure_intervals = _load_annot(xlsx, rec_start)
            cmap, _, _ = _bmc(mouse_id, int(meta["n_samples"]),
                              seizure_intervals, logger)
            chrono_maps[mouse_id] = cmap
        for i, rec in enumerate(manifest_records):
            fname = Path(rec["filepath"]).name
            mouse = fname.split("_")[0]
            cmap = chrono_maps.get(mouse)
            if cmap is None:
                continue
            info = cmap.get(fname)
            if info is None:
                continue
            r2 = dict(rec)
            r2["mouse_id"]    = mouse
            r2["chrono_idx"]  = int(info["chrono_idx"])
            r2["t_start_sec"] = float(info["t_start_sec"])
            records.append(r2)
            kept_idx.append(i)

    y_true_kept = y_true[kept_idx]
    y_prob_kept = y_prob[kept_idx]
    logger.info("Built %d records from manifest (%d unique mice; %d dropped)",
                len(records), len({r["mouse_id"] for r in records}),
                len(manifest_records) - len(records))
    return y_true_kept, y_prob_kept, records


# Inline writers retired -- the four-file bundle (summary.json,
# event_details.csv, classification_report_row{1,2}.json) is now emitted
# by eval_utils.write_event_level_bundle, the single source of truth used
# by postproc_sweep, recover_event_metrics, m4_event_metrics_recovery,
# the four training scripts, the two _evaluation scripts, and
# m3_post_eval. See eval_utils.py for the canonical 17-column
# event_details schema + summary.json shape.


def write_comparison_csv(out_path, rows, logger):
    fieldnames = [
        "partition", "order", "min_event_sec",
        "n_predicted_events", "tp", "fp", "fn",
        "event_precision", "event_recall", "event_f1",
        "event_far_per_hour", "event_mean_latency_sec",
        "seg_row2_f1_macro", "seg_row2_auroc", "seg_row2_precision",
        "seg_row2_recall", "seg_row2_specificity", "seg_row2_far_per_hour",
        "seg_row2_mcc",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    logger.info("Saved comparison CSV: %s (%d rows)", out_path, len(rows))


def plot_impact(rows, out_path, model_label, logger):
    """2 rows (val/test) x 3 cols (P/R/F1 lines, FAR line, Pareto)."""
    def _subset(partition, order):
        sub = [r for r in rows if r["partition"] == partition and r["order"] == order]
        sub.sort(key=lambda r: r["min_event_sec"])
        return sub

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for row_i, partition in enumerate(PARTITIONS):
        ax = axes[row_i, 0]
        for order in ORDERINGS:
            sub = _subset(partition, order)
            if not sub:
                continue
            secs = [r["min_event_sec"]   for r in sub]
            prec = [r["event_precision"] for r in sub]
            rec  = [r["event_recall"]    for r in sub]
            f1v  = [r["event_f1"]        for r in sub]
            ls = ORDER_LINESTYLE[order]; mk = ORDER_MARKER[order]
            lbl = ORDER_LABEL[order]
            ax.plot(secs, prec, linestyle=ls, marker=mk, color="#5A7DC8",
                    linewidth=1.6, label=f"Precision ({lbl})")
            ax.plot(secs, rec,  linestyle=ls, marker=mk, color="#5AC880",
                    linewidth=1.6, label=f"Recall ({lbl})")
            ax.plot(secs, f1v,  linestyle=ls, marker=mk, color="#C8A05A",
                    linewidth=1.6, label=f"F1 ({lbl})")
        ax.set_xlabel("MIN_EVENT_SEC (s)"); ax.set_ylabel("Score")
        ax.set_title(f"[{partition.upper()}] Event Precision / Recall / F1")
        ax.set_xticks(SWEEP_SECS); ax.grid(True, alpha=0.3)
        # Legend BELOW the axes so the 6 entries never overlap with the
        # P/R/F1 curves (loc="best" picked the data-covered lower-left
        # corner for this dataset).
        ax.legend(fontsize=7, ncol=3, loc="upper center",
                  bbox_to_anchor=(0.5, -0.18), frameon=True)

        ax = axes[row_i, 1]
        for order in ORDERINGS:
            sub = _subset(partition, order)
            if not sub:
                continue
            secs = [r["min_event_sec"]      for r in sub]
            fars = [r["event_far_per_hour"] for r in sub]
            ax.plot(secs, fars,
                    linestyle=ORDER_LINESTYLE[order],
                    marker=ORDER_MARKER[order],
                    color=ORDER_COLOR[order],
                    linewidth=1.6,
                    label=ORDER_LABEL[order])
        ax.set_xlabel("MIN_EVENT_SEC (s)"); ax.set_ylabel("Event-level FAR/hr")
        ax.set_title(f"[{partition.upper()}] Event-level FAR/hr")
        ax.set_xticks(SWEEP_SECS); ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

        ax = axes[row_i, 2]
        for order in ORDERINGS:
            sub = _subset(partition, order)
            if not sub:
                continue
            recs = [r["event_recall"]       for r in sub]
            fars = [r["event_far_per_hour"] for r in sub]
            secs = [r["min_event_sec"]      for r in sub]
            ax.scatter(recs, fars, s=80, color=ORDER_COLOR[order],
                       marker=ORDER_MARKER[order], zorder=3,
                       label=ORDER_LABEL[order])
            for s, x_, y_ in zip(secs, recs, fars):
                ax.annotate(f"{s}s", xy=(x_, y_), xytext=(6, 4),
                            textcoords="offset points", fontsize=9,
                            color=ORDER_COLOR[order])
        ax.set_xlabel("Event recall (sensitivity)")
        ax.set_ylabel("Event-level FAR/hr")
        ax.set_title(f"[{partition.upper()}] Recall vs FAR/hr Pareto\n(bottom-right = ideal)")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="best")

    plt.suptitle("Impact of post-processing order x MIN_EVENT_SEC on %s "
                 "-- val (top) vs test (bottom)" % model_label,
                 fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    # Add vertical breathing room between the two rows so the [VAL] P/R/F1
    # legend (sitting below its axes) doesn't crowd the [TEST] row's titles.
    fig.subplots_adjust(hspace=0.55)
    # bbox_inches="tight" expands the saved figure to fit the below-axis
    # legend on the bottom row (which would otherwise sit outside the
    # default canvas).
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved impact figure: %s", out_path)


def plot_impact_train_only(rows, out_path, model_label, logger):
    """Single-row variant of plot_impact: 1 row (train) x 3 cols. Used when
    the train sweep is run on its own (no val/test in the row set)."""
    def _subset(order):
        sub = [r for r in rows if r["partition"] == "train" and r["order"] == order]
        sub.sort(key=lambda r: r["min_event_sec"])
        return sub

    _, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    ax = axes[0]
    for order in ORDERINGS:
        sub = _subset(order)
        if not sub:
            continue
        secs = [r["min_event_sec"]   for r in sub]
        prec = [r["event_precision"] for r in sub]
        rec  = [r["event_recall"]    for r in sub]
        f1v  = [r["event_f1"]        for r in sub]
        ls = ORDER_LINESTYLE[order]; mk = ORDER_MARKER[order]
        lbl = ORDER_LABEL[order]
        ax.plot(secs, prec, linestyle=ls, marker=mk, color="#5A7DC8",
                linewidth=1.6, label=f"Precision ({lbl})")
        ax.plot(secs, rec,  linestyle=ls, marker=mk, color="#5AC880",
                linewidth=1.6, label=f"Recall ({lbl})")
        ax.plot(secs, f1v,  linestyle=ls, marker=mk, color="#C8A05A",
                linewidth=1.6, label=f"F1 ({lbl})")
    ax.set_xlabel("MIN_EVENT_SEC (s)"); ax.set_ylabel("Score")
    ax.set_title("[TRAIN] Event Precision / Recall / F1")
    ax.set_xticks(SWEEP_SECS); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), frameon=True)

    ax = axes[1]
    for order in ORDERINGS:
        sub = _subset(order)
        if not sub:
            continue
        secs = [r["min_event_sec"]      for r in sub]
        fars = [r["event_far_per_hour"] for r in sub]
        ax.plot(secs, fars,
                linestyle=ORDER_LINESTYLE[order],
                marker=ORDER_MARKER[order],
                color=ORDER_COLOR[order],
                linewidth=1.6,
                label=ORDER_LABEL[order])
    ax.set_xlabel("MIN_EVENT_SEC (s)"); ax.set_ylabel("Event-level FAR/hr")
    ax.set_title("[TRAIN] Event-level FAR/hr")
    ax.set_xticks(SWEEP_SECS); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    ax = axes[2]
    for order in ORDERINGS:
        sub = _subset(order)
        if not sub:
            continue
        recs = [r["event_recall"]       for r in sub]
        fars = [r["event_far_per_hour"] for r in sub]
        secs = [r["min_event_sec"]      for r in sub]
        ax.scatter(recs, fars, s=80, color=ORDER_COLOR[order],
                   marker=ORDER_MARKER[order], zorder=3,
                   label=ORDER_LABEL[order])
        for s, x_, y_ in zip(secs, recs, fars):
            ax.annotate(f"{s}s", xy=(x_, y_), xytext=(6, 4),
                        textcoords="offset points", fontsize=9,
                        color=ORDER_COLOR[order])
    ax.set_xlabel("Event recall (sensitivity)")
    ax.set_ylabel("Event-level FAR/hr")
    ax.set_title("[TRAIN] Recall vs FAR/hr Pareto\n(bottom-right = ideal)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="best")

    plt.suptitle("Impact of post-processing order x MIN_EVENT_SEC on %s "
                 "-- FULL TRAIN (un-downsampled)" % model_label,
                 fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved impact figure (train-only): %s", out_path)


def sweep_partition(partition, npz_path, manifest_path, annot_dir,
                    mouse_metadata, output_root, model_label, logger):
    """Run the 2-order x N-sec sweep (N = len(SWEEP_SECS)) for one
    partition. Returns list of comparison-row dicts (one per (order, sec))."""
    if not npz_path.exists():
        logger.warning("[%s] NPZ missing at %s -- skipping this partition.",
                       partition, npz_path)
        return []
    if not annot_dir.exists():
        logger.warning("[%s] Annotations dir missing at %s -- skipping this partition.",
                       partition, annot_dir)
        return []

    try:
        y_true_kept, y_prob_kept, records = load_records_and_arrays(
            npz_path, manifest_path, partition, annot_dir, mouse_metadata, logger)
    except (KeyError, FileNotFoundError, ValueError) as exc:
        logger.error("[%s] %s", partition, exc)
        return []
    logger.info("[%s] NPZ loaded: %d segments | %d positive (%.3f%%)",
                partition, len(y_true_kept), int(y_true_kept.sum()),
                100.0 * float(y_true_kept.sum()) / max(1, len(y_true_kept)))

    rows = []
    for order in ORDERINGS:
        for sec in SWEEP_SECS:
            logger.info("-" * 65)
            logger.info("[%s] order=%s | MIN_EVENT_SEC=%d s", partition, order, sec)
            eval_utils.MIN_EVENT_SEC = float(sec)
            result = evaluate_event_level(
                records, y_true_kept, y_prob_kept,
                annot_dir, mouse_metadata, logger, order=order)

            out_dir = output_root / partition / order / f"MIN_EVENT_SEC_{sec}s"
            out_dir.mkdir(parents=True, exist_ok=True)

            write_event_level_bundle(
                out_dir, partition, model_label, result, logger,
                order=order,
                min_event_duration_sec=float(sec),
                refractory_period_sec=eval_utils.REFRACTORY_SEC,
                smoothing_window=eval_utils.SMOOTHING_WIN,
                threshold=eval_utils.THRESHOLD,
                step_sec=eval_utils.STEP_SEC,
            )

            em = result["event_level_metrics"]
            seg_r2 = result["segment_level_metrics"]["row2_postproc_threshold_0_5"]
            rows.append({
                "partition":              partition,
                "order":                  order,
                "min_event_sec":          sec,
                "n_predicted_events":     result["totals"]["n_predicted_events"],
                "tp":                     em["tp"],
                "fp":                     em["fp"],
                "fn":                     em["fn"],
                "event_precision":        em["precision"],
                "event_recall":           em["recall"],
                "event_f1":               em["f1"],
                "event_far_per_hour":     em["far_per_hour_event_CORRECTED"],
                "event_mean_latency_sec": em["mean_detection_latency_sec"],
                "seg_row2_f1_macro":      seg_r2["f1_macro"],
                "seg_row2_auroc":         seg_r2["auroc"],
                "seg_row2_precision":     seg_r2["precision"],
                "seg_row2_recall":        seg_r2["recall"],
                "seg_row2_specificity":   seg_r2["specificity"],
                "seg_row2_far_per_hour":  seg_r2["far_per_hour_seg_CORRECTED_2_5s_denom"],
                "seg_row2_mcc":           seg_r2.get("mcc"),
            })
            logger.info("      TP=%d FP=%d FN=%d | Prec=%.4f Rec=%.4f F1=%.4f | FAR/hr=%.4f",
                        em["tp"], em["fp"], em["fn"],
                        em["precision"], em["recall"], em["f1"],
                        em["far_per_hour_event_CORRECTED"])
    return rows


def main():
    args   = parse_args()
    paths  = resolve_paths(args)
    logger, log_path = setup_logging(paths["output"])

    output_root    = paths["output"]
    comparison_csv = output_root / "comparison_val_vs_test.csv"
    impact_plot    = output_root / "impact_val_vs_test.png"

    logger.info("=" * 65)
    logger.info("postproc_sweep.py")
    logger.info("Variant         : %s (%s)", args.variant, paths["model_label"])
    logger.info("Mode            : %s", "cluster" if args.cluster else "local")
    logger.info("Val NPZ         : %s", paths["val_npz"])
    logger.info("Test NPZ        : %s", paths["test_npz"])
    if args.include_train:
        logger.info("Train NPZ       : %s", paths["train_npz"])
        logger.info("Train manifest  : %s", paths["train_manifest"])
        logger.info("Train annot dir : %s", paths["train_annot_dir"])
        logger.info("Train output    : %s", paths["output_train"])
    logger.info("Metadata        : %s", paths["metadata"])
    logger.info("Manifest        : %s", paths["manifest"])
    logger.info("Val annot dir   : %s", paths["val_annot_dir"])
    logger.info("Test annot dir  : %s", paths["test_annot_dir"])
    logger.info("Output root     : %s", output_root)
    logger.info("Log             : %s", log_path)
    logger.info("Orderings       : %s", ORDERINGS)
    logger.info("Sweep secs      : %s", SWEEP_SECS)
    logger.info("Include train   : %s", args.include_train)
    logger.info("=" * 65)

    if not paths["metadata"].exists():
        logger.error("Required input missing: %s", paths["metadata"]); sys.exit(1)
    mouse_metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))

    npz_map        = {"val": paths["val_npz"],       "test": paths["test_npz"]}
    annot_dir_map  = {"val": paths["val_annot_dir"], "test": paths["test_annot_dir"]}

    all_rows = []
    for partition in PARTITIONS:
        logger.info("#" * 65)
        logger.info("PARTITION: %s", partition.upper())
        logger.info("#" * 65)
        rows = sweep_partition(
            partition, npz_map[partition], paths["manifest"],
            annot_dir_map[partition],
            mouse_metadata, output_root, paths["model_label"], logger)
        all_rows.extend(rows)

    if not all_rows:
        logger.error("No val/test sweep rows produced -- comparison CSV / plot will not be written.")
    else:
        write_comparison_csv(comparison_csv, all_rows, logger)
        plot_impact(all_rows, impact_plot, paths["model_label"], logger)

    if args.include_train:
        train_output_root  = paths["output_train"]
        train_log, _ = setup_logging(train_output_root)
        logger.info("#" * 65)
        logger.info("PARTITION: TRAIN (sibling output root: %s)", train_output_root)
        logger.info("#" * 65)
        train_rows = sweep_partition(
            "train", paths["train_npz"], paths["train_manifest"],
            paths["train_annot_dir"],
            mouse_metadata, train_output_root, paths["model_label"], train_log)
        if not train_rows:
            logger.error("No train sweep rows produced -- train comparison CSV / plot "
                         "will not be written.")
        else:
            train_comparison_csv = train_output_root / "comparison_train.csv"
            train_impact_plot    = train_output_root / "impact_train.png"
            write_comparison_csv(train_comparison_csv, train_rows, train_log)
            plot_impact_train_only(train_rows, train_impact_plot,
                                   paths["model_label"], train_log)

    if not all_rows and not (args.include_train):
        sys.exit(1)

    logger.info("=" * 65)
    logger.info("DONE")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
