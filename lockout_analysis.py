"""
lockout_analysis.py
===================
Analyse the cost / benefit trade-off of adding an alarm-level *lockout*
window on top of the existing 30 s intra-seizure refractory merge.

Goal
----
Decide what additional lockout L (L > 30 s) can be safely applied to the
final event stream emitted by a model's test-set pipeline at the locked
operating point (min_then_refractory, MIN_EVENT_SEC = 25 s). The 30 s
refractory already merges fragments of the same seizure that briefly dip
below threshold. The lockout proposed here is a downstream alarm-
scheduling policy: after an event fires, suppress any subsequent event
whose start falls within L seconds of the previous (non-suppressed)
event's start.

Usage
-----
    python lockout_analysis.py --model MultiScaleTCN          --partition test
    python lockout_analysis.py --model MultiScaleTCN          --partition val
    python lockout_analysis.py --model MultiScaleTCNAttention --partition test
    python lockout_analysis.py --model MultiScaleTCNAttention --partition val

Inputs (per model x partition)
------------------------------
  Event details : {model}/evaluation/post_process_varing_sec/{partition}/
                  min_then_refractory/MIN_EVENT_SEC_25s/{partition}_event_details.csv
  Summary       : same dir / {partition}_summary.json   (per-mouse non-ictal
                                                         hours used in FAR/hr)
  Annotations   : seizure_times_updated/{mouse_id}_xlsx.xlsx
                                                  (one Excel per mouse in
                                                  the partition; columns
                                                  start_time, end_time)

Outputs (under {model}/evaluation/lockout_analysis/{partition}/)
----------------------------------------------------------------
  gt_isi_per_mouse.csv          one row per consecutive GT-seizure pair
                                (mouse_id, isi_sec, prev_gt_start, this_gt_start)
  lockout_sweep_per_mouse.csv   per-mouse cost/benefit at each L
  lockout_sweep_pooled.csv      pooled summary table at each L
  gt_isi_ecdf.png               ECDF of GT ISIs (per-mouse + pooled)
  pareto_lockout.png            % TP preserved vs % FP removed (per-mouse +
                                pooled, labelled by L)
  lockout_analysis.log          persistent run log

Methodology
-----------
1. GT inter-seizure interval (ISI) is the time between consecutive
   ground-truth seizure starts within a mouse (sorted chronologically).
   ISIs <= 30 s are already implicitly handled by the in-pipeline 30 s
   refractory merge; they are documented but excluded from the
   additional-lockout cost calculation.
2. Lockout candidates: L in {60, 90, 120, 180, 240, 300} s.
3. For each L, walk predicted events chronologically per mouse and
   suppress any event whose start is within L of the previous non-
   suppressed event's start.
4. TP losses are decomposed by suppressor identity:
     - tp_lost_after_tp : genuine recurrent-seizure suppression
                          (biological cost, irreducible)
     - tp_lost_after_fp : FP gated a real seizure
                          (reducible by improving the model)
5. Suppressed FPs are pure benefit.
6. Suppressing a TP increases FN by 1 (its matched GT seizure is now
   unmatched).
"""

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Common paths + per-model registry
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(r"C:\Users\messi\OneDrive\Desktop\Desktop\TCN_UNIQURE_PROJECT")
ANNOT_DIR    = PROJECT_ROOT / "seizure_times_updated"

# CLI --model value -> display label used in log headers
MODEL_LABELS = {
    "MultiScaleTCN":          "M3 (MultiScaleTCN)",
    "MultiScaleTCNAttention": "M4 (MS-TCN+Attention)",
}

# Lockout candidates (seconds). All > 30 s; the in-pipeline 30 s refractory
# already collapses ISIs <= 30 s and is not re-tested here.
LOCKOUT_CANDIDATES      = [60, 90, 120, 180, 240, 300]
EXISTING_REFRACTORY_SEC = 30   # documented, used only for ECDF annotation


def build_paths(model_name, partition):
    """Return a dict of per-(model, partition) input + output paths."""
    eval_root      = PROJECT_ROOT / model_name / "evaluation"
    event_details  = (eval_root / "post_process_varing_sec" / partition
                      / "min_then_refractory" / "MIN_EVENT_SEC_25s"
                      / f"{partition}_event_details.csv")
    summary_json   = event_details.parent / f"{partition}_summary.json"
    out_dir        = eval_root / "lockout_analysis" / partition
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "event_details":       event_details,
        "summary_json":        summary_json,
        "out_dir":             out_dir,
        "log_path":            out_dir / "lockout_analysis.log",
        "gt_isi_csv":          out_dir / "gt_isi_per_mouse.csv",
        "sweep_per_mouse_csv": out_dir / "lockout_sweep_per_mouse.csv",
        "sweep_pooled_csv":    out_dir / "lockout_sweep_pooled.csv",
        "ecdf_png":            out_dir / "gt_isi_ecdf.png",
        "pareto_png":          out_dir / "pareto_lockout.png",
    }


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(log_path, model_name, partition):
    logger = logging.getLogger(f"lockout_analysis.{model_name}.{partition}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8"); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Step 1 -- GT inter-seizure intervals from annotation Excel files
# ---------------------------------------------------------------------------
def compute_gt_isi(mouse_ids, logger):
    """Return DataFrame of per-mouse GT ISIs (mouse_id, isi_sec,
    prev_gt_start_iso, this_gt_start_iso)."""
    rows = []
    n_total_seizures = 0
    for mouse_id in sorted(mouse_ids):
        xlsx_path = ANNOT_DIR / f"{mouse_id}_xlsx.xlsx"
        if not xlsx_path.exists():
            logger.warning("  %s : annotation file missing (%s) -- skipped",
                           mouse_id, xlsx_path)
            continue
        df = pd.read_excel(xlsx_path)
        df["start_time"] = pd.to_datetime(df["start_time"], dayfirst=True)
        df["end_time"]   = pd.to_datetime(df["end_time"],   dayfirst=True)
        df = df.sort_values("start_time").reset_index(drop=True)
        n_total_seizures += len(df)

        for i in range(1, len(df)):
            isi = (df.loc[i, "start_time"] - df.loc[i - 1, "start_time"]).total_seconds()
            rows.append({
                "mouse_id":           mouse_id,
                "isi_sec":            float(isi),
                "prev_gt_start_iso":  df.loc[i - 1, "start_time"].isoformat(),
                "this_gt_start_iso":  df.loc[i,     "start_time"].isoformat(),
            })
        logger.info("  %-6s : %3d GT seizures -> %3d ISIs",
                    mouse_id, len(df), max(0, len(df) - 1))

    gt_isi_df = pd.DataFrame(rows)
    logger.info("Total GT seizures across %d mice : %d (-> %d ISIs)",
                len(mouse_ids), n_total_seizures, len(gt_isi_df))
    return gt_isi_df, n_total_seizures


# ---------------------------------------------------------------------------
# Step 2 -- per-mouse lockout sweep over predicted events
# ---------------------------------------------------------------------------
def apply_lockout(events_df, lockout_sec):
    """Walk the events of a single mouse sorted by start_sec and tag each
    event as kept / suppressed-after-tp / suppressed-after-fp.

    A subsequent event is suppressed iff its start_sec is within
    `lockout_sec` of the previous (non-suppressed) event's start_sec.

    Returns the input DataFrame with two extra columns:
        kept                : bool
        suppressed_by_label : "" (kept) | "TP" | "FP" (the suppressor's label)
    """
    df = events_df.sort_values("start_sec").reset_index(drop=True)
    kept = [False] * len(df)
    suppressor_label = [""] * len(df)

    last_kept_idx = None
    for i in range(len(df)):
        if last_kept_idx is None:
            kept[i] = True
            last_kept_idx = i
            continue
        gap = float(df.loc[i, "start_sec"]) - float(df.loc[last_kept_idx, "start_sec"])
        if gap > lockout_sec:
            kept[i] = True
            last_kept_idx = i
        else:
            kept[i] = False
            suppressor_label[i] = "TP" if bool(df.loc[last_kept_idx, "is_true_alarm"]) else "FP"

    df["kept"] = kept
    df["suppressed_by_label"] = suppressor_label
    return df


def sweep_lockouts(events_df, per_mouse_meta, logger):
    """Apply each lockout candidate per mouse, recompute event-level metrics,
    return per-mouse and pooled DataFrames.

    per_mouse_meta : dict mouse_id -> dict with keys
                     non_ictal_hours, n_ground_truth_seizures, fn (from
                     test_summary.json per-mouse block).
    """
    per_mouse_rows = []
    pooled_rows    = []

    for L in LOCKOUT_CANDIDATES:
        pooled = {
            "lockout_sec":         L,
            "tp_kept":             0,
            "fp_kept":             0,
            "tp_lost_after_tp":    0,
            "tp_lost_after_fp":    0,
            "fp_removed":          0,
            "fn":                  0,
            "non_ictal_hours":     0.0,
            "n_predicted_events":  0,
            "n_ground_truth":      0,
        }
        for mouse_id, mouse_df in events_df.groupby("mouse_id"):
            tagged = apply_lockout(mouse_df, L)
            tp_kept_m          = int(((tagged["kept"]) & (tagged["is_true_alarm"])).sum())
            fp_kept_m          = int(((tagged["kept"]) & (~tagged["is_true_alarm"])).sum())
            tp_lost_after_tp_m = int(((~tagged["kept"]) & (tagged["is_true_alarm"])
                                      & (tagged["suppressed_by_label"] == "TP")).sum())
            tp_lost_after_fp_m = int(((~tagged["kept"]) & (tagged["is_true_alarm"])
                                      & (tagged["suppressed_by_label"] == "FP")).sum())
            fp_removed_m       = int(((~tagged["kept"]) & (~tagged["is_true_alarm"])).sum())
            tp_lost_m          = tp_lost_after_tp_m + tp_lost_after_fp_m

            meta = per_mouse_meta.get(mouse_id, {})
            non_ictal_hr   = float(meta.get("non_ictal_hours", 0.0))
            fn_orig        = int(meta.get("fn", 0))
            n_gt           = int(meta.get("n_ground_truth_seizures", 0))
            fn_new         = fn_orig + tp_lost_m  # each suppressed TP turns a GT into FN

            prec = tp_kept_m / (tp_kept_m + fp_kept_m) if (tp_kept_m + fp_kept_m) > 0 else 0.0
            rec  = tp_kept_m / (tp_kept_m + fn_new)    if (tp_kept_m + fn_new)    > 0 else 0.0
            f1   = (2 * prec * rec / (prec + rec))     if (prec + rec)            > 0 else 0.0
            far  = fp_kept_m / non_ictal_hr            if non_ictal_hr             > 0 else 0.0

            per_mouse_rows.append({
                "mouse_id":          mouse_id,
                "lockout_sec":       L,
                "n_predicted_events":int(len(tagged)),
                "n_ground_truth":    n_gt,
                "tp_kept":           tp_kept_m,
                "fp_kept":           fp_kept_m,
                "tp_lost_after_tp":  tp_lost_after_tp_m,
                "tp_lost_after_fp":  tp_lost_after_fp_m,
                "fp_removed":        fp_removed_m,
                "fn":                fn_new,
                "non_ictal_hours":   round(non_ictal_hr, 4),
                "precision":         round(prec, 6),
                "recall":            round(rec, 6),
                "f1":                round(f1, 6),
                "far_per_hour":      round(far, 6),
            })

            pooled["tp_kept"]            += tp_kept_m
            pooled["fp_kept"]            += fp_kept_m
            pooled["tp_lost_after_tp"]   += tp_lost_after_tp_m
            pooled["tp_lost_after_fp"]   += tp_lost_after_fp_m
            pooled["fp_removed"]         += fp_removed_m
            pooled["fn"]                 += fn_new
            pooled["non_ictal_hours"]    += non_ictal_hr
            pooled["n_predicted_events"] += int(len(tagged))
            pooled["n_ground_truth"]     += n_gt

        tp_kept = pooled["tp_kept"]; fp_kept = pooled["fp_kept"]; fn = pooled["fn"]
        prec = tp_kept / (tp_kept + fp_kept) if (tp_kept + fp_kept) > 0 else 0.0
        rec  = tp_kept / (tp_kept + fn)      if (tp_kept + fn)      > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec)      > 0 else 0.0
        far  = fp_kept / pooled["non_ictal_hours"] if pooled["non_ictal_hours"] > 0 else 0.0
        pooled["precision"]    = round(prec, 6)
        pooled["recall"]       = round(rec, 6)
        pooled["f1"]           = round(f1, 6)
        pooled["far_per_hour"] = round(far, 6)
        pooled_rows.append(pooled)

        logger.info("  L=%3ds : pooled TP_kept=%4d FP_kept=%4d "
                    "TP_lost(after TP=%d, after FP=%d) FP_removed=%4d "
                    "P=%.3f R=%.3f F1=%.3f FAR/hr=%.4f",
                    L, tp_kept, fp_kept,
                    pooled["tp_lost_after_tp"], pooled["tp_lost_after_fp"],
                    pooled["fp_removed"], prec, rec, f1, far)

    return pd.DataFrame(per_mouse_rows), pd.DataFrame(pooled_rows)


# ---------------------------------------------------------------------------
# Step 3 -- plots
# ---------------------------------------------------------------------------
def plot_ecdf(gt_isi_df, ecdf_png, model_label, partition, logger):
    """ECDF of GT inter-seizure intervals. Translucent per-mouse curves +
    bold pooled curve. Annotations for the existing 30 s refractory and for
    each lockout candidate."""
    if gt_isi_df.empty:
        logger.warning("ECDF: no GT ISI rows -- skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for mouse_id, sub in gt_isi_df.groupby("mouse_id"):
        vals = np.sort(sub["isi_sec"].to_numpy())
        if len(vals) < 1:
            continue
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, y, where="post", color="#5A7DC8", alpha=0.18, linewidth=1.0)

    vals = np.sort(gt_isi_df["isi_sec"].to_numpy())
    y = np.arange(1, len(vals) + 1) / len(vals)
    ax.step(vals, y, where="post", color="#1f3a78", linewidth=2.2, label="Pooled GT ISI")

    ax.axvline(EXISTING_REFRACTORY_SEC, color="gray", linestyle=":", linewidth=1.0)
    ax.text(EXISTING_REFRACTORY_SEC, 1.04, "30 s\n(refractory)",
            ha="center", va="bottom", fontsize=8, color="gray")
    label_heights = [1.08, 1.13, 1.08, 1.13, 1.08, 1.13]
    for L, h in zip(LOCKOUT_CANDIDATES, label_heights):
        ax.axvline(L, color="#C85A5A", linestyle="--", linewidth=0.8, alpha=0.55)
        ax.text(L, h, f"{L}s", ha="center", va="bottom",
                fontsize=8, color="#C85A5A")

    ax.set_xscale("log")
    ax.set_xlim(1, max(gt_isi_df["isi_sec"].max() * 1.1, 3600))
    ax.set_ylim(0, 1.17)
    ax.set_xlabel("Inter-seizure interval (s, log scale)")
    ax.set_ylabel("Cumulative fraction of GT seizures")
    ax.set_title(f"Ground-truth ISI ECDF -- {model_label} {partition} partition (per-mouse + pooled)")
    ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(ecdf_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", ecdf_png)


def plot_pareto(per_mouse_df, pooled_df, n_orig_tp, n_orig_fp,
                pareto_png, model_label, partition, logger):
    """% TP preserved vs % FP removed, points labelled with L.
    Translucent per-mouse traces + bold pooled trace."""
    if pooled_df.empty:
        logger.warning("Pareto: pooled sweep is empty -- skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for mouse_id, sub in per_mouse_df.groupby("mouse_id"):
        sub = sub.sort_values("lockout_sec")
        orig_tp_m = sub["tp_kept"] + sub["tp_lost_after_tp"] + sub["tp_lost_after_fp"]
        orig_fp_m = sub["fp_kept"] + sub["fp_removed"]
        otp = float(orig_tp_m.max()); ofp = float(orig_fp_m.max())
        if otp == 0 or ofp == 0:
            continue
        pct_tp_pres = 100.0 * sub["tp_kept"] / otp
        pct_fp_rem  = 100.0 * sub["fp_removed"] / ofp
        ax.plot(pct_fp_rem, pct_tp_pres, color="#5A7DC8", alpha=0.18, linewidth=1.0)

    pooled_df = pooled_df.sort_values("lockout_sec").reset_index(drop=True)
    pct_tp_pres_pooled = 100.0 * pooled_df["tp_kept"] / n_orig_tp
    pct_fp_rem_pooled  = 100.0 * pooled_df["fp_removed"] / n_orig_fp
    ax.plot(pct_fp_rem_pooled, pct_tp_pres_pooled,
            color="#1f3a78", linewidth=2.2, marker="o",
            markersize=8, label="Pooled")

    for x, y, L in zip(pct_fp_rem_pooled, pct_tp_pres_pooled, pooled_df["lockout_sec"]):
        ax.annotate(f"{int(L)} s",
                    xy=(x, y),
                    xytext=(7, -3), textcoords="offset points",
                    fontsize=9, color="#1f3a78", fontweight="bold")

    ax.set_xlabel("% FPs removed (benefit)")
    ax.set_ylabel("% TPs preserved (cost: 100% = no recall loss)")
    ax.set_title(f"Pareto: alarm-level lockout above 30 s refractory -- {model_label} ({partition})")
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_ylim(min(0, pct_tp_pres_pooled.min() - 2), 102)

    # Zoomed inset
    ax_in = fig.add_axes([0.55, 0.30, 0.35, 0.32])
    for mouse_id, sub in per_mouse_df.groupby("mouse_id"):
        sub = sub.sort_values("lockout_sec")
        otp = float((sub["tp_kept"] + sub["tp_lost_after_tp"]
                     + sub["tp_lost_after_fp"]).max())
        ofp = float((sub["fp_kept"] + sub["fp_removed"]).max())
        if otp == 0 or ofp == 0:
            continue
        ax_in.plot(100.0 * sub["fp_removed"] / ofp,
                   100.0 * sub["tp_kept"] / otp,
                   color="#5A7DC8", alpha=0.18, linewidth=0.9)
    ax_in.plot(pct_fp_rem_pooled, pct_tp_pres_pooled,
               color="#1f3a78", linewidth=2.0, marker="o", markersize=6)
    for x, y, L in zip(pct_fp_rem_pooled, pct_tp_pres_pooled,
                       pooled_df["lockout_sec"]):
        ax_in.annotate(f"{int(L)}s",
                       xy=(x, y), xytext=(5, -2),
                       textcoords="offset points",
                       fontsize=8, color="#1f3a78")
    ax_in.set_xlabel("% FPs removed", fontsize=8)
    ax_in.set_ylabel("% TPs preserved", fontsize=8)
    ax_in.set_title("Zoom: 95-100 % TP band", fontsize=8)
    ax_in.set_ylim(95, 100.5)
    ax_in.grid(True, linestyle=":", linewidth=0.3, alpha=0.6)
    ax_in.tick_params(labelsize=7)

    plt.savefig(pareto_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", pareto_png)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=list(MODEL_LABELS.keys()),
                        help="Which model's predictions to analyse.")
    parser.add_argument("--partition", required=True, choices=["val", "test"],
                        help="Which partition to analyse (val or test).")
    args = parser.parse_args()

    model_name  = args.model
    partition   = args.partition
    model_label = MODEL_LABELS[model_name]
    paths       = build_paths(model_name, partition)

    logger = setup_logging(paths["log_path"], model_name, partition)
    logger.info("=" * 70)
    logger.info("lockout_analysis.py  --  %s %s partition, locked operating point",
                model_label, partition)
    logger.info("Timestamp     : %s", datetime.datetime.now().isoformat())
    logger.info("Model         : %s", model_label)
    logger.info("Partition     : %s", partition)
    logger.info("Event details : %s", paths["event_details"])
    logger.info("Annotations   : %s", ANNOT_DIR)
    logger.info("Output dir    : %s", paths["out_dir"])
    logger.info("Lockout cands : %s s (all > 30 s; 30 s already implemented "
                "as refractory merge)", LOCKOUT_CANDIDATES)
    logger.info("=" * 70)

    # ----- Load event_details + per-mouse meta -----------------------------
    events_df = pd.read_csv(paths["event_details"])
    events_df["is_true_alarm"] = events_df["is_true_alarm"].astype(str).str.lower() \
                                  .map({"true": True, "1": True, "false": False, "0": False})
    events_df = events_df.dropna(subset=["start_sec", "mouse_id"])
    logger.info("Loaded %d predicted events across %d mice",
                len(events_df), events_df["mouse_id"].nunique())

    with open(paths["summary_json"], "r", encoding="utf-8") as f:
        summary = json.load(f)
    per_mouse_meta = summary.get("per_mouse", {})
    n_orig_tp = int(summary["event_level_metrics"]["tp"])
    n_orig_fp = int(summary["event_level_metrics"]["fp"])
    n_orig_fn = int(summary["event_level_metrics"]["fn"])
    logger.info("Original event-level metrics: TP=%d FP=%d FN=%d  "
                "(precision=%.4f recall=%.4f F1=%.4f FAR/hr=%.4f)",
                n_orig_tp, n_orig_fp, n_orig_fn,
                summary["event_level_metrics"]["precision"],
                summary["event_level_metrics"]["recall"],
                summary["event_level_metrics"]["f1"],
                summary["event_level_metrics"]["far_per_hour_event_CORRECTED"])

    # ----- Step 1: GT ISI distribution -------------------------------------
    logger.info("-" * 70)
    logger.info("Step 1: GT inter-seizure intervals from Excel annotations")
    gt_isi_df, n_gt_total = compute_gt_isi(events_df["mouse_id"].unique(), logger)
    gt_isi_df.to_csv(paths["gt_isi_csv"], index=False)
    logger.info("Saved: %s (%d rows)", paths["gt_isi_csv"], len(gt_isi_df))

    if not gt_isi_df.empty:
        isi_vals = gt_isi_df["isi_sec"].to_numpy()
        within_30 = int((isi_vals <= 30).sum())
        logger.info("GT ISI <= 30 s (already handled by refractory): %d (%.2f%% of %d ISIs)",
                    within_30, 100.0 * within_30 / len(isi_vals), len(isi_vals))
        for L in LOCKOUT_CANDIDATES:
            n_in_30_L = int(((isi_vals > 30) & (isi_vals <= L)).sum())
            pct_in    = 100.0 * n_in_30_L / len(isi_vals)
            logger.info("  GT seizures in (30, %d] s : %3d  (%.2f%% of GT ISIs)",
                        L, n_in_30_L, pct_in)

    # ----- Step 2: lockout sweep over predicted events ---------------------
    logger.info("-" * 70)
    logger.info("Step 2: per-mouse lockout sweep over predicted events")
    per_mouse_sweep_df, pooled_sweep_df = sweep_lockouts(events_df, per_mouse_meta, logger)
    per_mouse_sweep_df.to_csv(paths["sweep_per_mouse_csv"], index=False)
    pooled_sweep_df.to_csv(paths["sweep_pooled_csv"], index=False)
    logger.info("Saved: %s (%d rows)", paths["sweep_per_mouse_csv"], len(per_mouse_sweep_df))
    logger.info("Saved: %s (%d rows)", paths["sweep_pooled_csv"], len(pooled_sweep_df))

    # ----- Step 3: plots ---------------------------------------------------
    logger.info("-" * 70)
    logger.info("Step 3: plots")
    plot_ecdf(gt_isi_df, paths["ecdf_png"], model_label, partition, logger)
    plot_pareto(per_mouse_sweep_df, pooled_sweep_df, n_orig_tp, n_orig_fp,
                paths["pareto_png"], model_label, partition, logger)

    # ----- Final summary --------------------------------------------------
    logger.info("=" * 70)
    logger.info("LOCKOUT ANALYSIS COMPLETE  (%s, %s)", model_label, partition)
    logger.info("  Original TP / FP / FN     : %d / %d / %d", n_orig_tp, n_orig_fp, n_orig_fn)
    logger.info("  GT seizures (%s)        : %d across %d mice", partition,
                n_gt_total, events_df["mouse_id"].nunique())
    logger.info("  GT ISIs computed          : %d", len(gt_isi_df))
    logger.info("  Lockout candidates        : %s s", LOCKOUT_CANDIDATES)
    logger.info("Outputs under %s :", paths["out_dir"])
    for key in ["gt_isi_csv", "sweep_per_mouse_csv", "sweep_pooled_csv",
                "ecdf_png", "pareto_png", "log_path"]:
        p = paths[key]
        logger.info("  [%s] %s", "OK" if p.exists() else "MISSING", p.name)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
