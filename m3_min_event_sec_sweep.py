"""
m3_min_event_sec_sweep.py
=========================
Dual-partition (val + test) sweep over the post-processing order and
MIN_EVENT_SEC for M3 (MultiScaleTCN). Reads cached per-partition
predictions (no model forward pass, no GPU) and recomputes the
chronology-aware event-level evaluation for each combination of:

    order in {refractory_then_min, min_then_refractory}
    MIN_EVENT_SEC in {10, 15, 20, 30}    (seconds)

8 configurations per partition x 2 partitions = 16 evaluations total.
Smoothing window (W=3), threshold (tau=0.5), refractory period (30 s),
and FAR/hr denominator (step_sec = 2.5 s) are held at their canonical
values; only the ordering and MIN_EVENT_SEC vary.

The two orderings are:
  - "refractory_then_min" (legacy, today's production default): refractory
    merge first, then drop survivors shorter than MIN_EVENT_SEC. Can
    rescue fragmented true detections (and fragmented false alarms).
  - "min_then_refractory" (clinical-standard order; Saab 2020, Tang 2022,
    Persyst, Encevis): drop short candidate events first, then merge any
    survivors that are < 30 s apart. Strictly more conservative.

Dual-mode execution (cluster or local)
--------------------------------------
By default reads/writes the canonical cluster paths under
/home/people/22206468/scratch/. Pass --local-root to switch to the
Windows local-mirror layout (TCN_UNIQURE_PROJECT/). Individual paths
can be overridden via flags (see parse_args).

Manifest can be either the enriched (data_splits_nonictal_sampled_
filtered_enriched.json) or the unenriched filtered variant. If records
already carry chrono_idx + t_start_sec the in-process chronology rebuild
is skipped; otherwise it runs once per partition.

Outputs (under --output-dir, default <root>/MultiScaleTCN/evaluation/post_process_varing_sec)
    val/
        refractory_then_min/  MIN_EVENT_SEC_10s/  15s/  20s/  30s/
        min_then_refractory/  MIN_EVENT_SEC_10s/  15s/  20s/  30s/
    test/
        refractory_then_min/  MIN_EVENT_SEC_10s/  15s/  20s/  30s/
        min_then_refractory/  MIN_EVENT_SEC_10s/  15s/  20s/  30s/
    comparison_val_vs_test.csv         (16 rows: partition x order x sec)
    impact_val_vs_test.png             (2x3 panels: rows = partitions)
    m3_min_event_sec_sweep.log
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
    load_annotations, build_mouse_chronology, evaluate_event_level,
    build_classification_report,
)


CLUSTER_SCRATCH = Path("/home/people/22206468/scratch")
LOCAL_DEFAULT   = Path(r"C:\Users\messi\OneDrive\Desktop\Desktop\TCN_UNIQURE_PROJECT")

MODEL_NAME = "MultiScaleTCN"
PARTITIONS = ["val", "test"]
ORDERINGS  = ["refractory_then_min", "min_then_refractory"]
SWEEP_SECS = [10, 15, 20, 30]

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
    p.add_argument("--local-root", type=Path, default=None,
                   help="Switch to local Windows-mirror layout rooted here. "
                        "Omit on the cluster to use default /home/people/22206468/scratch/... paths.")
    p.add_argument("--val-npz",        type=Path, default=None, help="Override val NPZ path.")
    p.add_argument("--test-npz",       type=Path, default=None, help="Override test NPZ path.")
    p.add_argument("--manifest",       type=Path, default=None, help="Override manifest path (enriched preferred).")
    p.add_argument("--metadata",       type=Path, default=None, help="Override mouse metadata JSON path.")
    p.add_argument("--val-annot-dir",  type=Path, default=None, help="Override val annotations dir.")
    p.add_argument("--test-annot-dir", type=Path, default=None, help="Override test annotations dir.")
    p.add_argument("--output-dir",     type=Path, default=None, help="Override output root.")
    return p.parse_args()


def resolve_paths(args):
    """Return dict of all resolved input/output paths. Defaults switch on
    --local-root. Individual flags override one-at-a-time. The
    'manifest_fallback' key gives the unenriched manifest path used only
    if the preferred (enriched) file is missing.
    """
    if args.local_root is not None:
        root = args.local_root
        defaults = {
            "val_npz":        root / "MultiScaleTCN" / "multiscale_tcn_predictions_raw.npz",
            "test_npz":       root / "MultiScaleTCN" / "evaluation" / "multiscale_tcn_test_predictions_raw.npz",
            "manifest":       root / "data_splits_nonictal_sampled_filtered_enriched.json",
            "manifest_fallback": root / "data_splits_nonictal_sampled_filtered.json",
            "metadata":       root / "mouse_recording_metadata.json",
            "val_annot_dir":  root / "val_seizure_annot_updated",
            "test_annot_dir": root / "test_seizure_annot_updated",
            "output":         root / "MultiScaleTCN" / "evaluation" / "post_process_varing_sec",
        }
    else:
        defaults = {
            "val_npz":        CLUSTER_SCRATCH / "OUTPUT" / "MODEL3_OUTPUT" / "MultiScaleTCN" / "multiscale_tcn_predictions_raw.npz",
            "test_npz":       CLUSTER_SCRATCH / "OUTPUT" / "MODEL3_OUTPUT" / "MultiScaleTCN" / "evaluation" / "multiscale_tcn_test_predictions_raw.npz",
            "manifest":       CLUSTER_SCRATCH / "INPUT_DATA" / "data_splits_outputs" / "data_splits_nonictal_sampled_filtered_enriched.json",
            "manifest_fallback": CLUSTER_SCRATCH / "INPUT_DATA" / "data_splits_outputs" / "data_splits_nonictal_sampled_filtered.json",
            "metadata":       CLUSTER_SCRATCH / "INPUT_DATA" / "Data_diagnostic" / "mouse_recording_metadata.json",
            # On cluster val+test annotations live in a single shared folder.
            "val_annot_dir":  CLUSTER_SCRATCH / "seizure_times_updated",
            "test_annot_dir": CLUSTER_SCRATCH / "seizure_times_updated",
            "output":         CLUSTER_SCRATCH / "OUTPUT" / "MODEL3_OUTPUT" / "MultiScaleTCN" / "evaluation" / "post_process_varing_sec",
        }

    manifest = args.manifest or defaults["manifest"]
    if args.manifest is None and not manifest.exists() and defaults["manifest_fallback"].exists():
        manifest = defaults["manifest_fallback"]

    return {
        "val_npz":        args.val_npz        or defaults["val_npz"],
        "test_npz":       args.test_npz       or defaults["test_npz"],
        "manifest":       manifest,
        "metadata":       args.metadata       or defaults["metadata"],
        "val_annot_dir":  args.val_annot_dir  or defaults["val_annot_dir"],
        "test_annot_dir": args.test_annot_dir or defaults["test_annot_dir"],
        "output":         args.output_dir     or defaults["output"],
    }


def setup_logging(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "m3_min_event_sec_sweep.log"
    logger = logging.getLogger("m3_min_event_sec_sweep")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8"); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger, log_path


def enrich_partition_records(partition, records, mouse_metadata, annotations_dir, logger):
    """Ensure mouse_id / chrono_idx / t_start_sec are set on every record.

    Idempotent. If records came from the enriched manifest (chrono_idx +
    t_start_sec already present), pass them through after dropping sentinel
    -1 rows. Otherwise rebuild chronology in-process via
    build_mouse_chronology (one-pass per mouse).
    """
    sample = records[:5]
    already_enriched = bool(sample) and all(
        ("chrono_idx" in r and "t_start_sec" in r) for r in sample)

    if already_enriched:
        logger.info("[%s] Manifest already enriched; skipping in-process rebuild.", partition)
        kept_recs, kept_idx = [], []
        for npz_idx, r in enumerate(records):
            ci = r.get("chrono_idx", -1)
            if ci is None or int(ci) < 0:
                continue
            r2 = dict(r)
            r2.setdefault("mouse_id", Path(r["filepath"]).stem.split("_")[0])
            r2["chrono_idx"]  = int(ci)
            r2["t_start_sec"] = float(r.get("t_start_sec", 0.0))
            kept_recs.append(r2)
            kept_idx.append(npz_idx)
        logger.info("[%s] Already-enriched: %d retained | %d dropped (sentinel chrono_idx=-1)",
                    partition, len(kept_recs), len(records) - len(kept_recs))
        return kept_recs, kept_idx

    logger.info("[%s] Manifest unenriched; rebuilding chronology in-process.", partition)
    mice = sorted({Path(r["filepath"]).stem.split("_")[0] for r in records})
    logger.info("[%s] %d records, %d unique mice", partition, len(records), len(mice))

    chrono_maps = {}
    for mouse_id in mice:
        if mouse_id not in mouse_metadata:
            logger.warning("  [%s] %s : no metadata; dropping all its records.",
                           partition, mouse_id)
            chrono_maps[mouse_id] = None
            continue
        xlsx = annotations_dir / f"{mouse_id}_xlsx.xlsx"
        if not xlsx.exists():
            logger.warning("  [%s] %s : no annotation at %s; dropping all its records.",
                           partition, mouse_id, xlsx)
            chrono_maps[mouse_id] = None
            continue
        meta = mouse_metadata[mouse_id]
        rec_start = datetime.datetime.fromisoformat(meta["recording_start_dt"])
        seizure_intervals = load_annotations(xlsx, rec_start)
        cmap, _, _ = build_mouse_chronology(
            mouse_id, int(meta["n_samples"]), seizure_intervals, logger)
        chrono_maps[mouse_id] = cmap

    kept_recs, kept_idx = [], []
    for npz_idx, r in enumerate(records):
        fname = Path(r["filepath"]).name
        mouse = fname.split("_")[0]
        cmap = chrono_maps.get(mouse)
        if cmap is None:
            continue
        info = cmap.get(fname)
        if info is None:
            continue
        r2 = dict(r)
        r2["mouse_id"]    = mouse
        r2["chrono_idx"]  = info["chrono_idx"]
        r2["t_start_sec"] = info["t_start_sec"]
        kept_recs.append(r2)
        kept_idx.append(npz_idx)
    logger.info("[%s] In-process enriched: %d retained | %d dropped (no chronology)",
                partition, len(kept_recs), len(records) - len(kept_recs))
    return kept_recs, kept_idx


def write_summary_json(out_path, model, partition, sec, order,
                       manifest_filter_state, result, logger):
    payload = {
        "model":     model,
        "partition": partition,
        "timestamp": datetime.datetime.now().isoformat(),
        "post_processing_params": {
            "order":                  order,
            "smoothing_window":       eval_utils.SMOOTHING_WIN,
            "refractory_period_sec":  eval_utils.REFRACTORY_SEC,
            "min_event_duration_sec": sec,
            "threshold":              eval_utils.THRESHOLD,
            "step_sec":               eval_utils.STEP_SEC,
            "matching_rule":          "any-overlap",
        },
        "totals":                result["totals"],
        "segment_level_metrics": result["segment_level_metrics"],
        "event_level_metrics":   result["event_level_metrics"],
        "per_mouse":             result["per_mouse_results"],
    }
    if manifest_filter_state is not None:
        payload["manifest_filter_history"] = manifest_filter_state
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str),
                        encoding="utf-8")
    logger.info("      summary -> %s", out_path)


def write_event_details_csv(out_path, partition, result, logger):
    fieldnames = ["mouse_id", "partition", "is_true_alarm", "start_sec", "end_sec",
                  "duration_sec", "max_prob", "matched_gt_idx",
                  "matched_gt_start_sec", "matched_gt_end_sec"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["all_event_details"]:
            r = dict(row)
            r["partition"] = partition
            writer.writerow(r)
    logger.info("      event details -> %s (%d rows)",
                out_path, len(result["all_event_details"]))


def write_classification_reports(out_dir, partition, result, logger):
    seg = result["segment_level_metrics"]
    reord = result["reordered_arrays"]
    y_true = reord["y_true"]
    r1 = build_classification_report(y_true, reord["y_pred_row1"],
                                     seg["row1_raw_threshold_0_5"])
    r2 = build_classification_report(y_true, reord["y_pred_row2"],
                                     seg["row2_postproc_threshold_0_5"])
    p1 = out_dir / f"{partition}_classification_report_row1.json"
    p2 = out_dir / f"{partition}_classification_report_row2.json"
    p1.write_text(json.dumps(r1, indent=2, sort_keys=True, default=str), encoding="utf-8")
    p2.write_text(json.dumps(r2, indent=2, sort_keys=True, default=str), encoding="utf-8")
    logger.info("      row1 report -> %s", p1)
    logger.info("      row2 report -> %s", p2)


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


def plot_impact(rows, out_path, logger):
    """2 rows (val/test) x 3 cols (P/R/F1 lines, FAR line, Pareto)."""
    def _subset(partition, order):
        sub = [r for r in rows if r["partition"] == partition and r["order"] == order]
        sub.sort(key=lambda r: r["min_event_sec"])
        return sub

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for row_i, partition in enumerate(PARTITIONS):
        # ---- Col 0: Event Precision / Recall / F1 vs MIN_EVENT_SEC ----------
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
        ax.legend(fontsize=7, ncol=2, loc="best")

        # ---- Col 1: Event-level FAR/hr vs MIN_EVENT_SEC ---------------------
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
        ax.set_title(f"[{partition.upper()}] Event-level FAR/hr (corrected)")
        ax.set_xticks(SWEEP_SECS); ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

        # ---- Col 2: Recall vs FAR/hr Pareto, 8 points per panel -------------
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

    plt.suptitle("Impact of post-processing order x MIN_EVENT_SEC on M3 "
                 "(MultiScaleTCN) -- val (top) vs test (bottom)",
                 fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved impact figure: %s", out_path)


def sweep_partition(partition, npz_path, annot_dir, test_records,
                    mouse_metadata, manifest_filter_state, output_root, logger):
    """Run the 2-order x 4-sec sweep for one partition. Returns list of
    comparison-row dicts (one per (order, sec))."""
    if not npz_path.exists():
        logger.warning("[%s] NPZ missing at %s -- skipping this partition.",
                       partition, npz_path)
        return []
    if not annot_dir.exists():
        logger.warning("[%s] Annotations dir missing at %s -- skipping this partition.",
                       partition, annot_dir)
        return []

    npz = np.load(npz_path, allow_pickle=False)
    y_true_all = npz["y_true"].astype(np.int64)
    y_prob_all = npz["y_prob"].astype(np.float64)
    if len(y_true_all) != len(test_records):
        logger.error("[%s] NPZ length %d != manifest partition length %d -- skipping.",
                     partition, len(y_true_all), len(test_records))
        return []
    logger.info("[%s] NPZ loaded: %d segments | %d positive (%.3f%%)",
                partition, len(y_true_all), int(y_true_all.sum()),
                100.0 * float(y_true_all.sum()) / max(1, len(y_true_all)))

    enriched, kept_idx = enrich_partition_records(
        partition, test_records, mouse_metadata, annot_dir, logger)
    y_true_kept = y_true_all[kept_idx]
    y_prob_kept = y_prob_all[kept_idx]

    rows = []
    for order in ORDERINGS:
        for sec in SWEEP_SECS:
            logger.info("-" * 65)
            logger.info("[%s] order=%s | MIN_EVENT_SEC=%d s", partition, order, sec)
            eval_utils.MIN_EVENT_SEC = float(sec)
            result = evaluate_event_level(
                enriched, y_true_kept, y_prob_kept,
                annot_dir, mouse_metadata, logger, order=order)

            out_dir = output_root / partition / order / f"MIN_EVENT_SEC_{sec}s"
            out_dir.mkdir(parents=True, exist_ok=True)

            write_summary_json(out_dir / f"{partition}_summary.json",
                               MODEL_NAME, partition, sec, order,
                               manifest_filter_state, result, logger)
            write_event_details_csv(out_dir / f"{partition}_event_details.csv",
                                    partition, result, logger)
            write_classification_reports(out_dir, partition, result, logger)

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
    args = parse_args()
    paths = resolve_paths(args)
    logger, log_path = setup_logging(paths["output"])

    output_root    = paths["output"]
    comparison_csv = output_root / "comparison_val_vs_test.csv"
    impact_plot    = output_root / "impact_val_vs_test.png"

    logger.info("=" * 65)
    logger.info("m3_min_event_sec_sweep.py")
    logger.info("Mode            : %s", "local" if args.local_root else "cluster")
    logger.info("Val NPZ         : %s", paths["val_npz"])
    logger.info("Test NPZ        : %s", paths["test_npz"])
    logger.info("Manifest        : %s", paths["manifest"])
    logger.info("Metadata        : %s", paths["metadata"])
    logger.info("Val annot dir   : %s", paths["val_annot_dir"])
    logger.info("Test annot dir  : %s", paths["test_annot_dir"])
    logger.info("Output root     : %s", output_root)
    logger.info("Log             : %s", log_path)
    logger.info("Orderings       : %s", ORDERINGS)
    logger.info("Sweep secs      : %s", SWEEP_SECS)
    logger.info("=" * 65)

    for p in [paths["manifest"], paths["metadata"]]:
        if not p.exists():
            logger.error("Required input missing: %s", p); sys.exit(1)

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    mouse_metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    manifest_filter_state = (manifest.get("meta", {}) or {}).get("filter_history", None)

    all_rows = []
    npz_map        = {"val": paths["val_npz"],       "test": paths["test_npz"]}
    annot_dir_map  = {"val": paths["val_annot_dir"], "test": paths["test_annot_dir"]}

    for partition in PARTITIONS:
        logger.info("#" * 65)
        logger.info("PARTITION: %s", partition.upper())
        logger.info("#" * 65)
        records = manifest.get(partition) or []
        if not records:
            logger.warning("[%s] empty partition in manifest -- skipping.", partition)
            continue
        rows = sweep_partition(
            partition, npz_map[partition], annot_dir_map[partition],
            records, mouse_metadata, manifest_filter_state, output_root, logger)
        all_rows.extend(rows)

    if not all_rows:
        logger.error("No sweep rows produced -- comparison CSV / plot will not be written.")
        sys.exit(1)

    write_comparison_csv(comparison_csv, all_rows, logger)
    plot_impact(all_rows, impact_plot, logger)

    logger.info("=" * 65)
    logger.info("DONE")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
