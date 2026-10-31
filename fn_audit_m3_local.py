"""
fn_audit_m3_local.py
====================
Local runner for the M3 (MultiScaleTCN) false-negative audit on the TEST
partition, writing to <MultiScaleTCN>/FN_audit/.

false_negative_audit.py is hardcoded to the cluster paths and is NOT edited
here (per request). This driver imports it and monkey-patches only the path
resolution (variant_paths) and the two module-level path globals (ANNOT_DIR,
METADATA_PATH) to the local Windows layout, then calls its unchanged
audit_partition / write_csv / write_summary functions. All diagnostic logic
-- the FN set derivation and the per-seizure `diagnosis_hint` reason -- is
exactly false_negative_audit.py's.

Operating point (fixed to the reported canonical cell):
    partition      = test
    order          = min_then_refractory
    MIN_EVENT_SEC  = 25 s
    MAX_EVENT_SEC  = 120 s
"""

from pathlib import Path

import false_negative_audit as fna

LOCAL_ROOT = Path(r"C:\Users\messi\OneDrive\Desktop\Desktop\TCN_UNIQURE_PROJECT")
M3_BASE    = LOCAL_ROOT / "MultiScaleTCN"
OUT_DIR    = M3_BASE / "FN_audit"

PARTITION     = "test"
ORDER         = "min_then_refractory"
MIN_EVENT_SEC = 25.0
MAX_EVENT_SEC = 120.0


def local_variant_paths(variant, partition):
    """Local-layout replacement for false_negative_audit.variant_paths.
    Only MultiScaleTCN (M3) is wired up; output is redirected to FN_audit."""
    npz_map = {
        "val":   M3_BASE / "multiscale_tcn_predictions_raw.npz",
        "test":  M3_BASE / "evaluation" / "multiscale_tcn_test_predictions_raw.npz",
        "train": M3_BASE / "full_train_evaluation" / "multiscale_tcn_full_train_predictions_full.npz",
    }
    manifest_map = {
        "val":   LOCAL_ROOT / "data_splits_nonictal_sampled_filtered_enriched.json",
        "test":  LOCAL_ROOT / "data_splits_nonictal_sampled_filtered_enriched.json",
        "train": LOCAL_ROOT / "data_splits_full_train_enriched.json",
    }
    return {
        "npz":      npz_map[partition],
        "out_dir":  OUT_DIR,
        "manifest": manifest_map[partition],
    }


def main():
    # ---- redirect the audit to the local layout (no edits to fna) ----
    fna.variant_paths = local_variant_paths
    fna.ANNOT_DIR     = LOCAL_ROOT / "seizure_times_updated"
    fna.METADATA_PATH = LOCAL_ROOT / "mouse_recording_metadata.json"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = fna._op_suffix(MIN_EVENT_SEC, MAX_EVENT_SEC, ORDER)
    logger, log_path = fna.setup_logging(OUT_DIR, PARTITION, suffix)

    logger.info("=" * 65)
    logger.info("fn_audit_m3_local.py  (local runner; fna unedited)")
    logger.info("Log file : %s", log_path)
    logger.info("=" * 65)

    rows = fna.audit_partition(
        "MultiScaleTCN", PARTITION,
        MIN_EVENT_SEC, MAX_EVENT_SEC, ORDER, logger)

    csv_path  = OUT_DIR / f"false_negative_audit_{PARTITION}_{suffix}.csv"
    json_path = OUT_DIR / f"false_negative_audit_{PARTITION}_{suffix}.json"
    fna.write_csv(csv_path, rows, logger)
    fna.write_summary(json_path, rows, "MultiScaleTCN", PARTITION,
                      MIN_EVENT_SEC, MAX_EVENT_SEC, ORDER, logger)

    # ---- concise console summary ----
    buckets = {}
    for r in rows:
        buckets[r["diagnosis_hint"]] = buckets.get(r["diagnosis_hint"], 0) + 1
    print("\n" + "=" * 55)
    print("M3 (MultiScaleTCN) FALSE-NEGATIVE AUDIT -- %s" % PARTITION.upper())
    print("Operating point: %s, MIN=%.0fs, MAX=%.0fs"
          % (ORDER, MIN_EVENT_SEC, MAX_EVENT_SEC))
    print("-" * 55)
    print("Total missed seizures (FN): %d" % len(rows))
    print("Reasons (diagnosis_hint):")
    for k, v in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print("  %-34s %d" % (k, v))
    print("=" * 55)
    print("Outputs -> %s" % OUT_DIR)


if __name__ == "__main__":
    main()
