"""
m3_min_event_sec_sweep.py
=========================
Sweep over MIN_EVENT_SEC for M3 (MultiScaleTCN) on the test partition.
Reads cached test predictions (no model forward pass, no GPU required)
and recomputes the chronology-aware event-level evaluation for each
candidate value of the post-processing min-event-duration filter, to
quantify the precision / recall / FAR-per-hour tradeoff of raising the
filter.

Smoothing window (W=3), threshold (tau=0.5), refractory period (30 s),
and FAR/hr denominator (step_sec = 2.5 s) are held at their canonical
values; only MIN_EVENT_SEC varies.

Sweep values: MIN_EVENT_SEC in [10, 15, 20, 30] seconds (10 s is the
current production baseline; 15 / 20 / 30 are FP-reduction candidates).

Dual-mode execution (cluster or local)
--------------------------------------
By default the script reads/writes the canonical cluster paths under
/home/people/22206468/scratch/. Pass --local-root to switch to the
Windows local-mirror layout (TCN_UNIQURE_PROJECT/). Individual paths
can also be overridden one-at-a-time via --npz / --manifest /
--metadata / --annot-dir / --output-dir.

The manifest can be either the enriched (data_splits_nonictal_sampled_
filtered_enriched.json) or the unenriched filtered variant. If records
already carry chrono_idx + t_start_sec, the in-process chronology
rebuild is skipped; otherwise it runs once (~1 min) from EDF metadata +
Excel annotations.

Outputs (under --output-dir)
    MIN_EVENT_SEC_<sec>s/
        test_summary.json
        test_classification_report_row1.json
        test_classification_report_row2.json
        test_event_details.csv
    test_postproc_min_event_sec_comparison.csv
    test_postproc_min_event_sec_impact.png
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
PARTITION  = "test"
SWEEP_SECS = [10, 15, 20, 30]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--local-root", type=Path, default=None,
                   help="Switch to local Windows-mirror layout rooted here. "
                        "Omit on the cluster to use the default /home/people/22206468/scratch/... paths.")
    p.add_argument("--npz",        type=Path, default=None, help="Override NPZ path.")
    p.add_argument("--manifest",   type=Path, default=None, help="Override manifest path (enriched preferred).")
    p.add_argument("--metadata",   type=Path, default=None, help="Override mouse metadata JSON path.")
    p.add_argument("--annot-dir",  type=Path, default=None, help="Override test annotations dir.")
    p.add_argument("--output-dir", type=Path, default=None, help="Override output root.")
    return p.parse_args()


def resolve_paths(args):
    """Return a dict of all resolved input/output paths.

    Defaults switch on whether --local-root was passed. Individual flags
    override the defaults one-at-a-time. The "manifest_fallback" key gives
    the unenriched manifest path used only if the preferred (enriched)
    file is missing.
    """
    if args.local_root is not None:
        root = args.local_root
        defaults = {
            "npz":       root / "MultiScaleTCN" / "evaluation" / "multiscale_tcn_test_predictions_raw.npz",
            "manifest":  root / "data_splits_nonictal_sampled_filtered_enriched.json",
            "manifest_fallback": root / "data_splits_nonictal_sampled_filtered.json",
            "metadata":  root / "mouse_recording_metadata.json",
            "annot_dir": root / "test_seizure_annot_updated",
            "output":    root / "MultiScaleTCN" / "evaluation" / "post_process_varing_sec",
        }
    else:
        defaults = {
            "npz":       CLUSTER_SCRATCH / "OUTPUT" / "MODEL3_OUTPUT" / "MultiScaleTCN" / "evaluation" / "multiscale_tcn_test_predictions_raw.npz",
            "manifest":  CLUSTER_SCRATCH / "INPUT_DATA" / "data_splits_outputs" / "data_splits_nonictal_sampled_filtered_enriched.json",
            "manifest_fallback": CLUSTER_SCRATCH / "INPUT_DATA" / "data_splits_outputs" / "data_splits_nonictal_sampled_filtered.json",
            "metadata":  CLUSTER_SCRATCH / "INPUT_DATA" / "Data_diagnostic" / "mouse_recording_metadata.json",
            "annot_dir": CLUSTER_SCRATCH / "seizure_times_updated",
            "output":    CLUSTER_SCRATCH / "OUTPUT" / "MODEL3_OUTPUT" / "MultiScaleTCN" / "evaluation" / "post_process_varing_sec",
        }

    npz      = args.npz       or defaults["npz"]
    manifest = args.manifest  or defaults["manifest"]
    if args.manifest is None and not manifest.exists() and defaults["manifest_fallback"].exists():
        manifest = defaults["manifest_fallback"]
    metadata = args.metadata  or defaults["metadata"]
    annot    = args.annot_dir or defaults["annot_dir"]
    output   = args.output_dir or defaults["output"]
    return {
        "npz":       npz,
        "manifest":  manifest,
        "metadata":  metadata,
        "annot_dir": annot,
        "output":    output,
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


def enrich_test_records(records, mouse_metadata, annotations_dir, logger):
    """Ensure mouse_id / chrono_idx / t_start_sec are set on every record.

    Idempotent: if the records came from the enriched manifest (chrono_idx
    + t_start_sec already present), pass them through after dropping any
    sentinel -1 rows that enrich_manifest.py couldn't match. Otherwise
    rebuild chronology in-process from EDF metadata + Excel annotations
    via build_mouse_chronology (one-pass, per-mouse).

    Returns (records_kept, npz_indices_kept) -- npz_indices_kept is the
    ordered list of original positions so the caller can trim the NPZ
    arrays in lockstep.
    """
    sample = records[:5]
    already_enriched = bool(sample) and all(
        ("chrono_idx" in r and "t_start_sec" in r) for r in sample)

    if already_enriched:
        logger.info("Manifest already enriched -- skipping in-process chronology rebuild.")
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
        logger.info("Already-enriched records: %d retained | %d dropped (sentinel chrono_idx=-1)",
                    len(kept_recs), len(records) - len(kept_recs))
        return kept_recs, kept_idx

    logger.info("Manifest is unenriched -- rebuilding chronology in-process.")
    mice = sorted({Path(r["filepath"]).stem.split("_")[0] for r in records})
    logger.info("Test partition: %d records, %d unique mice", len(records), len(mice))

    chrono_maps = {}
    for mouse_id in mice:
        if mouse_id not in mouse_metadata:
            logger.warning("  %s : no metadata; dropping all its records.", mouse_id)
            chrono_maps[mouse_id] = None
            continue
        xlsx = annotations_dir / f"{mouse_id}_xlsx.xlsx"
        if not xlsx.exists():
            logger.warning("  %s : no annotation at %s; dropping all its records.",
                           mouse_id, xlsx)
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
    logger.info("In-process enriched: %d retained | %d dropped (no chronology)",
                len(kept_recs), len(records) - len(kept_recs))
    return kept_recs, kept_idx


def write_summary_json(out_path, model, partition, sec,
                       manifest_filter_state, result, logger):
    payload = {
        "model":     model,
        "partition": partition,
        "timestamp": datetime.datetime.now().isoformat(),
        "post_processing_params": {
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
    logger.info("    summary -> %s", out_path)


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
    logger.info("    event details -> %s (%d rows)",
                out_path, len(result["all_event_details"]))


def write_classification_reports(out_dir, result, logger):
    seg = result["segment_level_metrics"]
    reord = result["reordered_arrays"]
    y_true = reord["y_true"]
    r1 = build_classification_report(y_true, reord["y_pred_row1"],
                                     seg["row1_raw_threshold_0_5"])
    r2 = build_classification_report(y_true, reord["y_pred_row2"],
                                     seg["row2_postproc_threshold_0_5"])
    p1 = out_dir / "test_classification_report_row1.json"
    p2 = out_dir / "test_classification_report_row2.json"
    p1.write_text(json.dumps(r1, indent=2, sort_keys=True, default=str), encoding="utf-8")
    p2.write_text(json.dumps(r2, indent=2, sort_keys=True, default=str), encoding="utf-8")
    logger.info("    row1 report -> %s", p1)
    logger.info("    row2 report -> %s", p2)


def write_comparison_csv(out_path, rows, logger):
    fieldnames = [
        "min_event_sec",
        "n_predicted_events", "tp", "fp", "fn",
        "event_precision", "event_recall", "event_f1",
        "event_far_per_hour", "event_mean_latency_sec",
        "seg_row2_f1_macro", "seg_row2_auroc", "seg_row2_precision",
        "seg_row2_recall", "seg_row2_specificity", "seg_row2_far_per_hour",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    logger.info("Saved comparison CSV: %s", out_path)


def plot_impact(rows, out_path, logger):
    """2x2 panel.
      (0,0) Event Precision / Recall / F1 vs MIN_EVENT_SEC (line+marker)
      (0,1) Event TP / FP / FN grouped bars
      (1,0) Event-level FAR/hr vs MIN_EVENT_SEC (line+marker)
      (1,1) Recall vs FAR/hr Pareto scatter (each point labelled with sec)
    """
    secs   = [r["min_event_sec"]      for r in rows]
    prec   = [r["event_precision"]    for r in rows]
    rec    = [r["event_recall"]       for r in rows]
    f1     = [r["event_f1"]           for r in rows]
    tps    = [r["tp"]                 for r in rows]
    fps    = [r["fp"]                 for r in rows]
    fns    = [r["fn"]                 for r in rows]
    fars   = [r["event_far_per_hour"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(secs, prec, "o-", color="#5A7DC8", linewidth=1.6, label="Precision")
    ax.plot(secs, rec,  "s-", color="#5AC880", linewidth=1.6, label="Recall")
    ax.plot(secs, f1,   "^-", color="#C8A05A", linewidth=1.6, label="F1")
    ax.set_xlabel("MIN_EVENT_SEC (s)"); ax.set_ylabel("Score")
    ax.set_title("Event-level Precision / Recall / F1")
    ax.set_xticks(secs); ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    ax = axes[0, 1]
    x = np.arange(len(secs)); w = 0.27
    ax.bar(x - w, tps, width=w, color="#5AC880", label="TP")
    ax.bar(x,     fps, width=w, color="#C85A5A", label="FP")
    ax.bar(x + w, fns, width=w, color="#5A7DC8", label="FN")
    ax.set_xticks(x); ax.set_xticklabels([str(s) for s in secs])
    ax.set_xlabel("MIN_EVENT_SEC (s)"); ax.set_ylabel("Event count")
    ax.set_title("Event-level TP / FP / FN counts")
    ax.grid(True, alpha=0.3, axis="y"); ax.legend(fontsize=9)

    ax = axes[1, 0]
    ax.plot(secs, fars, "o-", color="#C85A5A", linewidth=1.6)
    ax.set_xlabel("MIN_EVENT_SEC (s)"); ax.set_ylabel("Event-level FAR/hr (corrected)")
    ax.set_title("Event-level FAR/hr")
    ax.set_xticks(secs); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.scatter(rec, fars, s=90, color="#5A7DC8", zorder=3)
    for s, x_, y_ in zip(secs, rec, fars):
        ax.annotate(f"{s}s", xy=(x_, y_), xytext=(6, 4),
                    textcoords="offset points", fontsize=10)
    ax.set_xlabel("Event recall (sensitivity)")
    ax.set_ylabel("Event-level FAR/hr")
    ax.set_title("Recall vs FAR/hr Pareto\n(label = MIN_EVENT_SEC, bottom-right = ideal)")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Impact of MIN_EVENT_SEC on M3 (MultiScaleTCN) test partition",
                 fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved impact figure: %s", out_path)


def main():
    args = parse_args()
    paths = resolve_paths(args)
    logger, log_path = setup_logging(paths["output"])

    output_root    = paths["output"]
    comparison_csv = output_root / "test_postproc_min_event_sec_comparison.csv"
    impact_plot    = output_root / "test_postproc_min_event_sec_impact.png"

    logger.info("=" * 65)
    logger.info("m3_min_event_sec_sweep.py")
    logger.info("Mode            : %s", "local" if args.local_root else "cluster")
    logger.info("NPZ             : %s", paths["npz"])
    logger.info("Manifest        : %s", paths["manifest"])
    logger.info("Metadata        : %s", paths["metadata"])
    logger.info("Annotations dir : %s", paths["annot_dir"])
    logger.info("Output root     : %s", output_root)
    logger.info("Log             : %s", log_path)
    logger.info("Sweep values    : %s", SWEEP_SECS)
    logger.info("=" * 65)

    for p in [paths["npz"], paths["manifest"], paths["metadata"], paths["annot_dir"]]:
        if not p.exists():
            logger.error("Required input missing: %s", p); sys.exit(1)

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    test_records = manifest.get(PARTITION) or []
    if not test_records:
        logger.error("Empty test partition in %s.", paths["manifest"]); sys.exit(1)
    manifest_filter_state = (manifest.get("meta", {}) or {}).get("filter_history", None)

    mouse_metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))

    npz = np.load(paths["npz"], allow_pickle=False)
    y_true_all = npz["y_true"].astype(np.int64)
    y_prob_all = npz["y_prob"].astype(np.float64)
    if len(y_true_all) != len(test_records):
        logger.error("NPZ length %d != manifest test length %d",
                     len(y_true_all), len(test_records)); sys.exit(1)
    logger.info("NPZ loaded: %d segments | %d positive (%.3f%%)",
                len(y_true_all), int(y_true_all.sum()),
                100.0 * float(y_true_all.sum()) / max(1, len(y_true_all)))

    logger.info("Enriching test records (one-time)...")
    enriched, kept_idx = enrich_test_records(
        test_records, mouse_metadata, paths["annot_dir"], logger)
    y_true_kept = y_true_all[kept_idx]
    y_prob_kept = y_prob_all[kept_idx]

    comparison_rows = []
    for sec in SWEEP_SECS:
        logger.info("-" * 65)
        logger.info("Sweep value: MIN_EVENT_SEC = %d s", sec)
        eval_utils.MIN_EVENT_SEC = float(sec)
        result = evaluate_event_level(
            enriched, y_true_kept, y_prob_kept,
            paths["annot_dir"], mouse_metadata, logger)

        out_dir = output_root / f"MIN_EVENT_SEC_{sec}s"
        out_dir.mkdir(parents=True, exist_ok=True)

        write_summary_json(out_dir / "test_summary.json",
                           MODEL_NAME, PARTITION, sec, manifest_filter_state,
                           result, logger)
        write_event_details_csv(out_dir / "test_event_details.csv",
                                PARTITION, result, logger)
        write_classification_reports(out_dir, result, logger)

        em = result["event_level_metrics"]
        seg_r2 = result["segment_level_metrics"]["row2_postproc_threshold_0_5"]
        comparison_rows.append({
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
        })
        logger.info("    TP=%d FP=%d FN=%d | Prec=%.4f Rec=%.4f F1=%.4f | FAR/hr=%.4f",
                    em["tp"], em["fp"], em["fn"],
                    em["precision"], em["recall"], em["f1"],
                    em["far_per_hour_event_CORRECTED"])

    write_comparison_csv(comparison_csv, comparison_rows, logger)
    plot_impact(comparison_rows, impact_plot, logger)

    logger.info("=" * 65)
    logger.info("DONE")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
