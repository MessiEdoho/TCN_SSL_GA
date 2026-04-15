"""
verify_data_integrity.py
========================
Post-incident data integrity check for all .npy segment files across
TRAIN_DATA (1-5), VAL_DATA, and TEST_DATA partitions.

Run this script after any /scratch filesystem incident to confirm that
segment files are intact and usable for training. The script performs
five checks on every .npy file:

    1. Readable    -- np.load() succeeds without OSError or ValueError
    2. Shape       -- array is 1-D with exactly 2,500 samples (5 s at 500 Hz)
    3. Dtype       -- array dtype is float32 (as saved by preprocessing)
    4. Finite      -- no NaN or Inf values present
    5. Amplitude   -- max|x| does not exceed the threshold (default 1,000)

Files that fail any check are logged with the specific failure reason.
A summary report is saved as JSON for programmatic consumption.

The script does NOT modify, delete, or move any files. It is read-only.

Usage
-----
python verify_data_integrity.py

Outputs
-------
{OUTPUT_DIR}/data_integrity_report.json
{OUTPUT_DIR}/data_integrity.log
"""

import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Expected segment properties (must match preprocessing_binary.py)
EXPECTED_LEN = 2500                     # 5 s * 500 Hz
EXPECTED_DTYPE = np.float32             # saved as float32 by preprocessing
AMPLITUDE_THRESHOLD = 1000.0            # z-scored EEG should not exceed this

# All data directories to scan
DATA_DIRS = [
    ("TRAIN_DATA",   Path("/scratch/22206468/TRAIN_DATA")),
    ("TRAIN_DATA_2", Path("/scratch/22206468/TRAIN_DATA_2")),
    ("TRAIN_DATA_3", Path("/scratch/22206468/TRAIN_DATA_3")),
    ("TRAIN_DATA_4", Path("/scratch/22206468/TRAIN_DATA_4")),
    ("TRAIN_DATA_5", Path("/scratch/22206468/TRAIN_DATA_5")),
    ("VAL_DATA",     Path("/scratch/22206468/VAL_DATA")),
    ("TEST_DATA",    Path("/scratch/22206468/TEST_DATA")),
]

# Output
OUTPUT_DIR = Path("/scratch/22206468/OUTPUT/DATA_INTEGRITY_CHECK")
REPORT_PATH = OUTPUT_DIR / "data_integrity_report.json"
LOG_PATH = OUTPUT_DIR / "data_integrity.log"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("data_integrity")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(fmt)
logger.addHandler(sh)

fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
fh.setFormatter(fmt)
logger.addHandler(fh)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_file(filepath):
    """Run all 5 integrity checks on a single .npy file.

    Returns
    -------
    tuple of (passed: bool, failure_reason: str or None)
    """
    # Check 1: Readable
    try:
        x = np.load(filepath)
    except Exception as e:
        return False, f"unreadable: {type(e).__name__}: {e}"

    # Check 2: Shape -- must be 1-D with exactly EXPECTED_LEN samples
    if x.ndim != 1:
        return False, f"wrong ndim: expected 1, got {x.ndim} (shape {x.shape})"
    if x.shape[0] != EXPECTED_LEN:
        return False, f"wrong length: expected {EXPECTED_LEN}, got {x.shape[0]}"

    # Check 3: Dtype -- must be float32
    if x.dtype != EXPECTED_DTYPE:
        return False, f"wrong dtype: expected {EXPECTED_DTYPE}, got {x.dtype}"

    # Check 4: Finite -- no NaN or Inf
    if not np.isfinite(x).all():
        n_nan = int(np.isnan(x).sum())
        n_inf = int(np.isinf(x).sum())
        return False, f"non-finite values: {n_nan} NaN, {n_inf} Inf"

    # Check 5: Amplitude -- max|x| must not exceed threshold
    max_abs = float(np.abs(x).max())
    if max_abs > AMPLITUDE_THRESHOLD:
        return False, f"extreme amplitude: max|x| = {max_abs:.4e} > {AMPLITUDE_THRESHOLD}"

    return True, None


def scan_directory(name, root_dir):
    """Scan all .npy files in root_dir/seizure/ and root_dir/non_seizure/.

    Returns
    -------
    dict with per-directory results
    """
    result = {
        "name": name,
        "path": str(root_dir),
        "exists": root_dir.exists(),
        "seizure_total": 0,
        "seizure_passed": 0,
        "non_seizure_total": 0,
        "non_seizure_passed": 0,
        "failures": [],
    }

    if not root_dir.exists():
        logger.warning("  %s: directory does not exist -- SKIPPED", name)
        return result

    for subfolder, label_name in [("seizure", "seizure"), ("non_seizure", "non_seizure")]:
        sub_path = root_dir / subfolder
        if not sub_path.exists():
            logger.warning("  %s/%s: subfolder does not exist", name, subfolder)
            continue

        npy_files = sorted(sub_path.glob("*.npy"))
        n_total = len(npy_files)
        n_passed = 0

        for i, fp in enumerate(npy_files):
            passed, reason = verify_file(fp)
            if passed:
                n_passed += 1
            else:
                result["failures"].append({
                    "filepath": str(fp),
                    "class": label_name,
                    "reason": reason,
                })
                # Log first 100 failures per directory to avoid log explosion
                if len(result["failures"]) <= 100:
                    logger.warning("    FAIL: %s -- %s", fp.name, reason)

            # Progress every 500,000 files
            if (i + 1) % 500000 == 0:
                logger.info("    %s/%s: scanned %d/%d (%.1f%%)",
                            name, subfolder, i + 1, n_total,
                            100 * (i + 1) / n_total)

        if label_name == "seizure":
            result["seizure_total"] = n_total
            result["seizure_passed"] = n_passed
        else:
            result["non_seizure_total"] = n_total
            result["non_seizure_passed"] = n_passed

        n_failed = n_total - n_passed
        status = "PASS" if n_failed == 0 else f"FAIL ({n_failed} corrupted)"
        logger.info("  %s/%s: %d files | %d passed | %d failed | %s",
                    name, subfolder, n_total, n_passed, n_failed, status)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("verify_data_integrity.py")
    logger.info("Timestamp       : %s", datetime.now().isoformat())
    logger.info("Purpose         : Post-incident data integrity check")
    logger.info("Expected shape  : (2500,)")
    logger.info("Expected dtype  : float32")
    logger.info("Amplitude thresh: %.0f", AMPLITUDE_THRESHOLD)
    logger.info("Directories     : %d", len(DATA_DIRS))
    logger.info("=" * 60)

    t0 = time.time()
    all_results = []
    grand_total = 0
    grand_passed = 0
    grand_failed = 0

    for name, root_dir in DATA_DIRS:
        logger.info("Scanning %s (%s)...", name, root_dir)
        result = scan_directory(name, root_dir)
        all_results.append(result)

        dir_total = result["seizure_total"] + result["non_seizure_total"]
        dir_passed = result["seizure_passed"] + result["non_seizure_passed"]
        dir_failed = dir_total - dir_passed
        grand_total += dir_total
        grand_passed += dir_passed
        grand_failed += dir_failed

    elapsed = time.time() - t0

    # -- Summary ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("INTEGRITY CHECK COMPLETE")
    logger.info("  Total files scanned : %d", grand_total)
    logger.info("  Passed              : %d", grand_passed)
    logger.info("  Failed              : %d", grand_failed)
    logger.info("  Scan time           : %.1f s (%.1f min)", elapsed, elapsed / 60)

    if grand_failed == 0:
        logger.info("  STATUS: ALL FILES INTACT")
    else:
        logger.warning("  STATUS: %d CORRUPTED FILES DETECTED", grand_failed)
        # Log all unique failure reasons
        all_failures = []
        for r in all_results:
            all_failures.extend(r["failures"])
        reason_counts = {}
        for f in all_failures:
            reason_type = f["reason"].split(":")[0]
            reason_counts[reason_type] = reason_counts.get(reason_type, 0) + 1
        logger.info("  Failure breakdown:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            logger.info("    %-25s: %d files", reason, count)

    logger.info("=" * 60)

    # -- Per-directory summary table --------------------------------------------
    logger.info("")
    logger.info("  %-15s %10s %10s %10s %8s",
                "Directory", "Seizure", "Non-seizure", "Total", "Failed")
    logger.info("  " + "-" * 58)
    for r in all_results:
        t = r["seizure_total"] + r["non_seizure_total"]
        f = t - r["seizure_passed"] - r["non_seizure_passed"]
        logger.info("  %-15s %10d %10d %10d %8d",
                    r["name"],
                    r["seizure_total"],
                    r["non_seizure_total"],
                    t, f)
    logger.info("  " + "-" * 58)
    logger.info("  %-15s %10s %10s %10d %8d",
                "TOTAL", "", "", grand_total, grand_failed)

    # -- Save JSON report ------------------------------------------------------
    report = {
        "timestamp": datetime.now().isoformat(),
        "script": "verify_data_integrity.py",
        "expected_shape": [EXPECTED_LEN],
        "expected_dtype": str(EXPECTED_DTYPE),
        "amplitude_threshold": AMPLITUDE_THRESHOLD,
        "scan_time_seconds": round(elapsed, 1),
        "grand_total": grand_total,
        "grand_passed": grand_passed,
        "grand_failed": grand_failed,
        "status": "ALL_INTACT" if grand_failed == 0 else "CORRUPTION_DETECTED",
        "directories": [],
    }

    for r in all_results:
        dir_entry = {
            "name": r["name"],
            "path": r["path"],
            "exists": r["exists"],
            "seizure_total": r["seizure_total"],
            "seizure_passed": r["seizure_passed"],
            "non_seizure_total": r["non_seizure_total"],
            "non_seizure_passed": r["non_seizure_passed"],
            "n_failures": len(r["failures"]),
            "failures": r["failures"],  # full list of {filepath, class, reason}
        }
        report["directories"].append(dir_entry)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("")
    logger.info("Report saved: %s", REPORT_PATH)


if __name__ == "__main__":
    main()
