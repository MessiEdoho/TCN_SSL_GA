"""
generate_data_splits.py
=======================
Builds data_splits.json -- the single source of truth
for all pipeline scripts (tuning, training, evaluation).

Run this script ONCE before any tuning or training.
Re-run it (with --include-test) when test data is ready.

Expected folder structure
-------------------------
Training data is distributed across five roots from
parallel preprocessing. Each root has the same layout:

    /scratch/22206468/TRAIN_DATA/
        seizure/             .npy ictal segments
        non_seizure/         .npy non-ictal segments
    /scratch/22206468/TRAIN_DATA_2/
        seizure/
        non_seizure/
    /scratch/22206468/TRAIN_DATA_3/
        seizure/
        non_seizure/
    /scratch/22206468/TRAIN_DATA_4/
        seizure/
        non_seizure/
    /scratch/22206468/TRAIN_DATA_5/
        seizure/
        non_seizure/

    /scratch/22206468/VAL_DATA/
        seizure/
        non_seizure/

    /scratch/22206468/TEST_DATA/     (only with --include-test)
        seizure/
        non_seizure/

All five training roots are scanned and concatenated
into one train partition. No file copying is needed --
data_splits.json stores absolute paths to originals.

Filename convention: {mouse_id}_{ictal|nonictal}_{index:05d}.npy
  e.g. m1_ictal_00001.npy, m1_nonictal_00003.npy
Mouse ID is extracted by splitting on the first underscore.

Usage
-----
Before test data is ready:
    python generate_data_splits.py --no-test

When test data is fully preprocessed and ready:
    python generate_data_splits.py --include-test

The --include-test flag must be provided explicitly.
The script never detects or ingests test data
automatically. This prevents partially preprocessed
test files from contaminating the split.

Outputs
-------
/scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits.json
/scratch/22206468/INPUT_DATA/data_splits_outputs/splits_generation.log
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import argparse
import json
import logging
import sys
import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Training data is distributed across five roots from parallel preprocessing.
# generate_data_splits.py scans all five and concatenates the records.
# No file copying is needed -- data_splits.json stores absolute paths.
TRAIN_DIRS  = [
    Path("/scratch/22206468/TRAIN_DATA"),
    Path("/scratch/22206468/TRAIN_DATA_2"),
    Path("/scratch/22206468/TRAIN_DATA_3"),
    Path("/scratch/22206468/TRAIN_DATA_4"),
    Path("/scratch/22206468/TRAIN_DATA_5"),
]
VAL_DIR     = Path("/scratch/22206468/VAL_DATA")   # validation partition root
TEST_DIR    = Path("/scratch/22206468/TEST_DATA")  # test partition root (--include-test only)
OUTPUT_DIR  = Path("/scratch/22206468/INPUT_DATA/data_splits_outputs")           # all pipeline outputs live here
OUTPUT_FILE = OUTPUT_DIR / "data_splits.json"
LOG_FILE    = OUTPUT_DIR / "splits_generation.log"

SEIZURE_SUBFOLDER     = "seizure"       # subfolder name for ictal segments
NON_SEIZURE_SUBFOLDER = "non_seizure"   # subfolder name for non-ictal segments

LABEL_SEIZURE     = 1   # ictal -- matches make_loader() in tcn_utils.py
LABEL_NON_SEIZURE = 0   # non-ictal -- matches make_loader() in tcn_utils.py

VALID_EXTENSION = ".npy"

# Label convention:
# 1 = seizure (ictal)         -- subfolder: seizure/
# 0 = non_seizure (non-ictal) -- subfolder: non_seizure/
# This convention must match tcn_utils.py make_loader().
# It is also recorded in data_splits.json under
# metadata.label_convention for documentation purposes.


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------
def parse_args():
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace with attributes:
        no_test      : bool -- True if --no-test was passed
        include_test : bool -- True if --include-test was passed
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build outputs/data_splits.json for the EEG "
            "seizure detection pipeline."
        )
    )

    # Mutually exclusive: user must pick exactly one mode
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--no-test",
        action="store_true",
        dest="no_test",
        help=(
            "Build train and val splits only. Use this when "
            "test data has not yet been fully preprocessed. "
            "test key in JSON will be [] with "
            "test_status='pending'."
        )
    )

    group.add_argument(
        "--include-test",
        action="store_true",
        dest="include_test",
        help=(
            "Build train, val, and test splits. Use this "
            "ONLY when TEST_DATA is fully preprocessed and "
            "ready. Overwrites any existing data_splits.json."
        )
    )

    args = parser.parse_args()
    return args


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------
def setup_logging():
    """Configure and return the module logger.

    Writes to both stdout (StreamHandler) and
    outputs/splits_generation.log (FileHandler, mode='w').
    outputs/ is created here if it does not exist.

    Returns
    -------
    logging.Logger named 'data_splits'
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("data_splits")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(LOG_FILE, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# validate_folder
# ---------------------------------------------------------------------------
def validate_folder(folder, partition_name, logger):
    """Validate that a data folder exists and contains the expected
    subfolder structure with at least one .npy file.

    Parameters
    ----------
    folder         : Path -- partition root directory
    partition_name : str  -- human-readable name for logging
    logger         : logging.Logger

    Returns
    -------
    bool : True if valid. False if any check fails.
           Failures are logged at ERROR level so the caller
           can decide whether to abort or continue.
    """
    # Check root folder exists
    if not folder.exists():
        logger.error(
            "%s not found at '%s'. "
            "Verify the folder name and location.",
            partition_name, folder.resolve())
        return False

    # Check seizure subfolder exists
    seizure_dir = folder / SEIZURE_SUBFOLDER
    if not seizure_dir.exists():
        logger.error(
            "%s: subfolder '%s' not found inside '%s'. "
            "Expected path: %s",
            partition_name, SEIZURE_SUBFOLDER,
            folder, seizure_dir)
        return False

    # Check non_seizure subfolder exists
    non_sz_dir = folder / NON_SEIZURE_SUBFOLDER
    if not non_sz_dir.exists():
        logger.error(
            "%s: subfolder '%s' not found inside '%s'. "
            "Expected path: %s",
            partition_name, NON_SEIZURE_SUBFOLDER,
            folder, non_sz_dir)
        return False

    # Count .npy files to catch empty preprocessing output
    n_sz = len(list(seizure_dir.glob("*" + VALID_EXTENSION)))
    n_nsz = len(list(non_sz_dir.glob("*" + VALID_EXTENSION)))
    n_tot = n_sz + n_nsz

    if n_tot == 0:
        logger.error(
            "%s: no %s files found in either subfolder. "
            "Check that preprocessing is complete.",
            partition_name, VALID_EXTENSION)
        return False

    logger.info(
        "%s validated: %d seizure + %d non-seizure "
        "= %d total %s files",
        partition_name, n_sz, n_nsz, n_tot,
        VALID_EXTENSION)
    return True


# ---------------------------------------------------------------------------
# build_pairs
# ---------------------------------------------------------------------------
def build_pairs(root_dir, logger):
    """Scan a partition folder and build a list of records for every
    .npy segment file found.

    Parameters
    ----------
    root_dir : Path -- partition root (e.g. Path('TRAIN_DATA'))
    logger   : logging.Logger

    Returns
    -------
    list of dicts, each with keys:
        filepath : str  -- absolute resolved path
        label    : int  -- 1 (seizure) or 0 (non_seizure)
        mouse_id : str  -- parsed from filename stem
        filename : str  -- filename only, for reference

    Mouse ID parsing
    ----------------
    Mouse ID is extracted using stem.split('_', 1)[0].
    Splits on the FIRST underscore from the left.
    Assumes filenames follow: <mouseID>_<ictal|nonictal>_<index>.npy
    Examples:
        'm1_ictal_00001.npy'    -> mouse_id = 'm1'
        'm1_nonictal_00001.npy' -> mouse_id = 'm1'
        'm330_ictal_00042.npy'  -> mouse_id = 'm330'

    If no underscore is present in the stem, the full stem
    is used as mouse_id and a WARNING is logged. This allows
    the script to complete but the mouse_id may not be
    meaningful for leakage checking.

    Files are processed in sorted order for deterministic
    output across operating systems and Python versions.
    """
    records = []

    for subfolder, label_val in [
            (SEIZURE_SUBFOLDER,     LABEL_SEIZURE),
            (NON_SEIZURE_SUBFOLDER, LABEL_NON_SEIZURE)]:

        target_dir = root_dir / subfolder
        # Sort by name so output is deterministic regardless of
        # filesystem traversal order
        files = sorted(
            target_dir.glob("*" + VALID_EXTENSION),
            key=lambda p: p.name
        )

        for fp in files:
            stem = fp.stem

            if "_" in stem:
                # Split on the FIRST underscore from the left.
                # 'm1_ictal_00001' -> ['m1', 'ictal_00001'] -> mouse_id = 'm1'
                # 'm1_nonictal_00001' -> ['m1', 'nonictal_00001'] -> mouse_id = 'm1'
                mouse_id = stem.split("_", 1)[0]
            else:
                mouse_id = stem
                logger.warning(
                    "No underscore in '%s'. "
                    "Using full stem '%s' as mouse_id. "
                    "Verify this matches your naming "
                    "convention -- mouse_id is used for "
                    "the leakage check.",
                    fp.name, stem)

            records.append({
                "filepath": str(fp.resolve()),
                "label":    label_val,
                "mouse_id": mouse_id,
                "filename": fp.name
            })

    logger.info("  %s: %d records built", root_dir, len(records))
    return records


# ---------------------------------------------------------------------------
# log_partition_stats
# ---------------------------------------------------------------------------
def log_partition_stats(partition_name, records, logger):
    """Compute and log descriptive statistics for one partition.

    Parameters
    ----------
    partition_name : str  -- 'train', 'val', or 'test'
    records        : list of dicts from build_pairs()
    logger         : logging.Logger

    Returns
    -------
    dict with keys:
        n_total, n_seizure, n_non_seizure,
        ictal_pct, n_mice, mouse_ids
    """
    n_total = len(records)
    n_sz = sum(1 for r in records if r["label"] == LABEL_SEIZURE)
    n_nsz = n_total - n_sz
    # Zero-division guard for empty partitions
    ictal_pct = (n_sz / n_total * 100 if n_total > 0 else 0.0)
    mouse_ids = sorted({r["mouse_id"] for r in records})
    n_mice = len(mouse_ids)

    logger.info(
        "-- %s %s", partition_name.upper(),
        "-" * (40 - len(partition_name)))
    logger.info("  Total segments  : %d", n_total)
    logger.info("  Seizure         : %d", n_sz)
    logger.info("  Non-seizure     : %d", n_nsz)
    logger.info("  Ictal %%         : %.1f", ictal_pct)
    logger.info("  Unique mice     : %d", n_mice)
    logger.info("  Mouse IDs       : %s", mouse_ids)

    if n_sz == 0:
        logger.warning(
            "%s: zero seizure segments. "
            "Confirm '%s/' is not empty.",
            partition_name, SEIZURE_SUBFOLDER)
    if n_nsz == 0:
        logger.warning(
            "%s: zero non-seizure segments. "
            "Confirm '%s/' is not empty.",
            partition_name, NON_SEIZURE_SUBFOLDER)

    return {
        "n_total":       n_total,
        "n_seizure":     n_sz,
        "n_non_seizure": n_nsz,
        "ictal_pct":     round(ictal_pct, 2),
        "n_mice":        n_mice,
        "mouse_ids":     mouse_ids
    }


# ---------------------------------------------------------------------------
# run_leakage_check
# ---------------------------------------------------------------------------
def run_leakage_check(train_records, val_records, test_records, logger):
    """Verify that no mouse appears in more than one partition.

    Data leakage occurs when the same mouse contributes
    segments to both training and validation (or test).
    Because EEG recordings from the same animal share
    subject-specific spectral characteristics, leakage
    allows the model to recognise the animal rather than
    seizure morphology, producing inflated metrics that
    do not generalise. This check must pass before any
    split is saved.

    Parameters
    ----------
    train_records : list -- records from build_pairs() across all TRAIN_DIRS
    val_records   : list -- records from build_pairs(VAL_DIR)
    test_records  : list -- records from build_pairs(TEST_DIR),
                           pass [] if test not included
    logger        : logging.Logger

    Returns
    -------
    bool : True if no leakage detected. False otherwise.
           If False, the caller must abort without saving
           data_splits.json.
    """
    train_mice = set(r["mouse_id"] for r in train_records)
    val_mice = set(r["mouse_id"] for r in val_records)
    # test_mice is empty set when test_records is []
    test_mice = set(r["mouse_id"] for r in test_records)

    tv = train_mice & val_mice
    tt = train_mice & test_mice
    vt = val_mice & test_mice

    leakage = False

    if tv:
        logger.error(
            "LEAKAGE: mice in BOTH train and val: %s. "
            "Move these mice to a single partition.",
            sorted(tv))
        leakage = True
    if tt:
        logger.error(
            "LEAKAGE: mice in BOTH train and test: %s. "
            "Move these mice to a single partition.",
            sorted(tt))
        leakage = True
    if vt:
        logger.error(
            "LEAKAGE: mice in BOTH val and test: %s. "
            "Move these mice to a single partition.",
            sorted(vt))
        leakage = True

    if leakage:
        logger.error(
            "Leakage check FAILED. data_splits.json was "
            "NOT saved. Correct the folder assignments "
            "and re-run the script.")
        return False

    logger.info(
        "Leakage check PASS: no mouse appears in more "
        "than one partition. Split is valid.")
    return True


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    """Main entry point. Parses arguments, validates folders,
    builds records, runs the leakage check, and saves
    outputs/data_splits.json.
    """
    # -- Step 1: parse arguments and configure logging -------------------------
    args = parse_args()
    logger = setup_logging()

    logger.info("=" * 55)
    logger.info("generate_data_splits.py")
    logger.info("Timestamp : %s", datetime.datetime.now().isoformat())
    logger.info("=" * 55)

    include_test = args.include_test
    mode = "include-test" if include_test else "no-test"
    logger.info("Mode      : --%s", mode)

    if include_test:
        logger.info(
            "Test partition WILL be included. Ensure "
            "TEST_DATA is fully preprocessed before "
            "proceeding.")
    else:
        logger.info(
            "Test partition will NOT be included. "
            "test key in JSON will be [] with "
            "test_status='pending'. "
            "Re-run with --include-test when ready.")

    # -- Step 2: validate required partitions ----------------------------------
    # Validate all five training roots and the single validation root.
    all_train_ok = True
    for td in TRAIN_DIRS:
        if not validate_folder(td, td.name, logger):
            all_train_ok = False
    val_ok = validate_folder(VAL_DIR, "VAL_DATA", logger)

    if not all_train_ok or not val_ok:
        logger.error(
            "One or more required partitions failed "
            "validation. Fix the folder structure and "
            "re-run. Aborting.")
        sys.exit(1)

    # -- Step 3: validate test partition if requested --------------------------
    if include_test:
        test_ok = validate_folder(TEST_DIR, "TEST_DATA", logger)
        if not test_ok:
            logger.error(
                "TEST_DATA failed validation. Ensure the "
                "folder exists and preprocessing is complete "
                "before using --include-test. Aborting.")
            sys.exit(1)

    # -- Step 4: build records -------------------------------------------------
    # Scan all five training roots and concatenate into one train partition.
    # Each root has the same seizure/ and non_seizure/ structure.
    logger.info("Building segment records...")
    train_records = []
    for td in TRAIN_DIRS:
        train_records.extend(build_pairs(td, logger))
    logger.info("  Total train records across %d roots: %d", len(TRAIN_DIRS), len(train_records))
    val_records = build_pairs(VAL_DIR, logger)

    if include_test:
        test_records = build_pairs(TEST_DIR, logger)
        test_status = "complete"
    else:
        test_records = []
        test_status = "pending"

    # -- Step 5: log partition statistics --------------------------------------
    logger.info("Partition statistics:")
    train_stats = log_partition_stats("train", train_records, logger)
    val_stats = log_partition_stats("val", val_records, logger)

    if include_test:
        test_stats = log_partition_stats("test", test_records, logger)
    else:
        test_stats = {}
        logger.info("-- TEST -----------------------------------------")
        logger.info("  Status: pending (--no-test flag used)")

    # -- Step 6: run leakage check ---------------------------------------------
    logger.info("Running leakage check...")
    leakage_ok = run_leakage_check(
        train_records, val_records, test_records, logger)

    if not leakage_ok:
        # Exit without saving -- a corrupted split file is more
        # dangerous than no file at all
        sys.exit(1)

    # -- Step 7: build and save data_splits.json -------------------------------
    splits = {
        "metadata": {
            "timestamp":   datetime.datetime.now().isoformat(),
            "mode":        mode,
            "test_status": test_status,
            "script":      "generate_data_splits.py",
            "rerun_note": (
                "Re-run with --include-test when TEST_DATA "
                "is fully preprocessed to update this file."
            ),
            "label_convention": {
                "1": "seizure (ictal)",
                "0": "non_seizure (non-ictal)"
            },
            "folder_paths": {
                "train": [str(td.resolve()) for td in TRAIN_DIRS],
                "val":   str(VAL_DIR.resolve()),
                "test":  (str(TEST_DIR.resolve())
                          if include_test else "not included")
            },
            "statistics": {
                "train": train_stats,
                "val":   val_stats,
                "test":  test_stats
            }
        },
        "train": train_records,
        "val":   val_records,
        "test":  test_records
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(splits, f, indent=2)

    size_kb = OUTPUT_FILE.stat().st_size / 1024
    logger.info("Saved: %s (%.1f KB)", OUTPUT_FILE, size_kb)

    # -- Step 8: final summary -------------------------------------------------
    logger.info("=" * 55)
    logger.info("COMPLETE")
    logger.info("  Mode         : --%s", mode)
    logger.info("  Test status  : %s", test_status)
    logger.info("  Train segs   : %d", train_stats["n_total"])
    logger.info("  Val segs     : %d", val_stats["n_total"])
    if include_test:
        logger.info("  Test segs    : %d", test_stats["n_total"])
    else:
        logger.info("  Test segs    : pending")
    logger.info("  Output file  : %s", OUTPUT_FILE)
    logger.info("  Log file     : %s", LOG_FILE)
    logger.info("=" * 55)

    if not include_test:
        logger.info(
            "NEXT: run tcn_HPT_binary.ipynb and all "
            "training notebooks. When TEST_DATA is fully "
            "preprocessed, re-run this script with "
            "--include-test before running "
            "final_evaluation.ipynb.")
    else:
        logger.info(
            "NEXT: run final_evaluation.ipynb to evaluate "
            "all trained models on the test set.")


if __name__ == "__main__":
    main()


# =====================================================
# HOW TO LOAD data_splits.json IN PIPELINE SCRIPTS
# =====================================================
#
# Standard loading pattern used by ALL scripts and
# tcn_HPT_binary.ipynb:
#
#   import json
#   from pathlib import Path
#
#   SPLITS_PATH = Path("/scratch/22206468/INPUT_DATA/"
#                      "data_splits_outputs/data_splits.json")
#
#   with open(SPLITS_PATH, "r", encoding="utf-8") as f:
#       splits = json.load(f)
#
#   # Convert to (filepath, label) tuples for
#   # make_loader() in tcn_utils.py
#   train_pairs = [(d["filepath"], d["label"])
#                  for d in splits["train"]]
#   val_pairs   = [(d["filepath"], d["label"])
#                  for d in splits["val"]]
#
# Test pairs -- load ONLY in final_evaluation.py:
#
#   if splits["metadata"]["test_status"] != "complete":
#       raise RuntimeError("Test split not available.")
#
#   test_pairs = [(d["filepath"], d["label"])
#                 for d in splits["test"]]
#
# -- WHEN TO RE-RUN THIS SCRIPT -----------------------
# 1. When TEST_DATA is ready:
#    python generate_data_splits.py --include-test
#
# 2. When any data file is added, removed, or renamed
#    (re-run with the same flag as before)
#
# 3. When the project is moved to a new directory
#    (absolute paths are stored -- re-run to update)
#
# -- PIPELINE ORDER ------------------------------------
# 1. python generate_data_splits.py --no-test
# 2. tcn_HPT_binary.ipynb
# 3. train_model1_TCN.ipynb
# 4. train_model2_tcn_attention.ipynb
# 5. train_model3_ssl_pretrain.ipynb
# 6. train_model3a_linear_probe.ipynb
# 7. train_model3b_finetune.ipynb
# -- when test data ready ------------------------------
# 8. python generate_data_splits.py --include-test
# 9. final_evaluation.ipynb
# =====================================================
