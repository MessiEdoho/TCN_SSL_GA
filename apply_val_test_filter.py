"""
apply_val_test_filter.py
========================
One-shot data-prep step: take the existing balanced manifest produced by
create_balanced_splits.py (which filters TRAIN only, leaves VAL and TEST
untouched), apply the same |x| > 1000 / NaN / Inf filter to VAL and TEST,
and write a new manifest with all three partitions cleaned.

Pipeline position
-----------------
After  : create_balanced_splits.py  (produces data_splits_nonictal_sampled.json)
Before : TCN.py / TCNTemporalAttention.py / MultiScaleTCN.py /
         MultiScaleTCNAttention.py / final_evaluation.py  (consume the
         filtered manifest produced here)

Why this exists
---------------
The full-validation pass on M3 crashed mid-post-processing because
~3,201 / 4,298,154 val segments contained extreme amplitudes (worst
|x| = 1.65e19) that overflow FP16 in the AMP forward pass and
propagate to NaN via the LayerNorm reduction. Running this filter
once at the manifest level avoids re-doing the ~2-hour scan inside
every training script's main(), and ensures all four ablation models
see an identical val/test partition during their final evaluations.

Inputs
------
  data_splits_nonictal_sampled.json  -- existing manifest from
                                        create_balanced_splits.py

Outputs
-------
  data_splits_nonictal_sampled_filtered.json  -- new manifest with
                                                 filtered val + test
  Data_diagnostic/apply_val_test_filter.log    -- persistent log

Usage
-----
python apply_val_test_filter.py
    [--input  /path/to/data_splits_nonictal_sampled.json]
    [--output /path/to/data_splits_nonictal_sampled_filtered.json]
    [--threshold 1000.0]
    [--log-path /path/to/apply_val_test_filter.log]
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import argparse
import datetime
import json
import logging
import sys
import time
from pathlib import Path

# Reuse the well-tested, single-source-of-truth filter
from tcn_utils import filter_extreme_segments


# ---------------------------------------------------------------------------
# Defaults (override via CLI args)
# ---------------------------------------------------------------------------
DEFAULT_INPUT     = "/scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits_nonictal_sampled.json"
DEFAULT_OUTPUT    = "/scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits_nonictal_sampled_filtered.json"
DEFAULT_THRESHOLD = 1000.0
# Persistent log: same Data_diagnostic convention as scan_val_test_extreme.py
DEFAULT_LOG_PATH  = "/home/people/22206468/scratch/INPUT_DATA/Data_diagnostic/apply_val_test_filter.log"


# ---------------------------------------------------------------------------
# Logger setup (console + persistent file, append mode)
# ---------------------------------------------------------------------------
def setup_logging(log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger("apply_val_test_filter")
    log.setLevel(logging.INFO)
    log.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)

    log.info("Log file: %s", log_path.resolve())
    return log


# ---------------------------------------------------------------------------
# Filter one partition (records list -> filtered records list + stats)
# ---------------------------------------------------------------------------
def filter_partition(name, records, threshold, logger):
    """Apply filter_extreme_segments to a partition's record list.

    Records are dicts {"filepath": ..., "label": ...}. The filter operates
    on (filepath, label) tuples, so we round-trip through that representation.
    """
    logger.info("=" * 65)
    logger.info("Filtering %s: %d records (threshold=%.1f)",
                name, len(records), threshold)
    logger.info("=" * 65)

    # records -> tuple list (input format expected by filter_extreme_segments)
    pairs = [(rec["filepath"], rec["label"]) for rec in records]
    n_before = len(pairs)

    t0 = time.time()
    kept = filter_extreme_segments(pairs, threshold=threshold, logger=logger)
    elapsed = time.time() - t0

    n_after = len(kept)
    n_removed = n_before - n_after
    logger.info("%s filter complete: %d retained / %d removed (%.4f%%) in %.1f s",
                name, n_after, n_removed,
                100 * n_removed / max(n_before, 1), elapsed)

    # Build the new record list, preserving dict shape and any extra keys.
    # filter_extreme_segments returns the filtered (filepath, label) tuples
    # in the same order as the input, so we can re-attach by exact match.
    kept_set = set((str(fp), int(label)) for fp, label in kept)
    filtered_records = [
        rec for rec in records
        if (str(rec["filepath"]), int(rec["label"])) in kept_set
    ]
    assert len(filtered_records) == n_after, (
        f"Record reconstruction mismatch on {name}: "
        f"{len(filtered_records)} != {n_after}")

    return filtered_records, {
        "before": n_before,
        "after": n_after,
        "removed": n_removed,
        "removed_pct": 100 * n_removed / max(n_before, 1),
        "elapsed_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help="Existing manifest from create_balanced_splits.py")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Where to write the filtered manifest")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Amplitude threshold (matches train-side filter)")
    parser.add_argument("--log-path", default=DEFAULT_LOG_PATH,
                        help="Persistent .log file path")
    args = parser.parse_args()

    logger = setup_logging(args.log_path)
    logger.info("=" * 65)
    logger.info("apply_val_test_filter.py")
    logger.info("Timestamp   : %s", datetime.datetime.now().isoformat())
    logger.info("Input       : %s", args.input)
    logger.info("Output      : %s", args.output)
    logger.info("Threshold   : %.1f (matches train-side filter)", args.threshold)
    logger.info("=" * 65)

    in_path = Path(args.input)
    if not in_path.exists():
        logger.error("Input manifest not found: %s", in_path)
        sys.exit(1)

    # -- Load existing manifest --------------------------------------------
    logger.info("Loading: %s", in_path)
    with open(in_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    logger.info("Loaded keys: %s", sorted(splits.keys()))

    # -- Filter VAL --------------------------------------------------------
    val_stats = None
    if "val" in splits and splits["val"]:
        filtered_val, val_stats = filter_partition(
            "VAL", splits["val"], args.threshold, logger)
        splits["val"] = filtered_val
    else:
        logger.warning("No 'val' partition found -- skipping val filter")

    # -- Filter TEST -------------------------------------------------------
    test_stats = None
    if "test" in splits and splits["test"]:
        filtered_test, test_stats = filter_partition(
            "TEST", splits["test"], args.threshold, logger)
        splits["test"] = filtered_test
    else:
        logger.warning("No 'test' partition found -- skipping test filter")

    # -- Annotate the meta block so the manifest documents what we did -----
    meta = splits.get("meta", {})
    meta.setdefault("filter_history", [])
    meta["filter_history"].append({
        "step": "apply_val_test_filter",
        "timestamp": datetime.datetime.now().isoformat(),
        "threshold": args.threshold,
        "input_manifest": str(in_path),
        "val": val_stats,
        "test": test_stats,
    })
    # Also surface compact totals at the top level for quick inspection
    if val_stats is not None:
        meta["n_val_after_filter"] = val_stats["after"]
    if test_stats is not None:
        meta["n_test_after_filter"] = test_stats["after"]
    splits["meta"] = meta

    # -- Write the filtered manifest ---------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing filtered manifest: %s", out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    logger.info("Manifest written: %.1f MB",
                out_path.stat().st_size / 1e6)

    # -- Final summary ------------------------------------------------------
    logger.info("=" * 65)
    logger.info("DONE")
    if val_stats is not None:
        logger.info("  VAL : %d -> %d  (%d removed | %.4f%% | %.1f min)",
                    val_stats["before"], val_stats["after"],
                    val_stats["removed"], val_stats["removed_pct"],
                    val_stats["elapsed_seconds"] / 60)
    if test_stats is not None:
        logger.info("  TEST: %d -> %d  (%d removed | %.4f%% | %.1f min)",
                    test_stats["before"], test_stats["after"],
                    test_stats["removed"], test_stats["removed_pct"],
                    test_stats["elapsed_seconds"] / 60)
    logger.info("Use this manifest in the training scripts by setting")
    logger.info("SPLITS_PATH = %s", out_path)


if __name__ == "__main__":
    main()
