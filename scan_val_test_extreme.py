"""
scan_val_test_extreme.py
========================
Diagnostic scan: count NaN/Inf and |x| > 1000 segments in the val and
test partitions of the splits manifest.

Purpose
-------
The current pipeline filters extreme/non-finite segments from the TRAIN
partition (in create_balanced_splits.py) but leaves VAL and TEST
unfiltered. This scan quantifies how many bad segments live in val and
test, so we can decide whether the NaN crash in MultiScaleTCN.py at the
post-processing stage is fully explained by input contamination.

Outputs
-------
- Console log: per-partition counts (clean / NaN-Inf / extreme / errors),
  worst amplitude, top subjects by bad-segment count, top 10 worst files.
- scan_results.json: machine-readable summary of all of the above for
  downstream filter integration.

Usage
-----
python scan_val_test_extreme.py
    [--manifest /path/to/data_splits_nonictal_sampled.json]
    [--alt-manifest /path/to/data_splits.json]
    [--workers 16]
    [--threshold 1000.0]

The val partition is read from the primary manifest. The test partition
is read from the primary manifest if present; otherwise the script falls
back to the alt manifest.

Runtime
-------
~5-15 minutes for ~4.3M val segments at 16 workers on Lustre/GPFS,
dominated by .npy file I/O (no GPU needed).
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Defaults (override via CLI args)
# ---------------------------------------------------------------------------
DEFAULT_MANIFEST     = "/scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits_nonictal_sampled.json"
DEFAULT_ALT_MANIFEST = "/scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits.json"
DEFAULT_WORKERS      = 16
DEFAULT_THRESHOLD    = 1000.0
DEFAULT_OUTPUT       = "scan_results.json"
MAX_BAD_RECORDS_KEPT = 500          # cap memory / output JSON size

# Subject-id pattern in filepaths (e.g. "/scratch/.../m338/...")
_SUBJECT_RE = re.compile(r"\bm\d{3,4}\b")


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
def setup_logging():
    log = logging.getLogger("scan")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"))
    log.addHandler(handler)
    return log


# ---------------------------------------------------------------------------
# Worker function (runs in subprocess)
# ---------------------------------------------------------------------------
def _scan_one(args):
    """Examine one .npy file. Returns a small dict describing the result."""
    fp, label, threshold = args
    try:
        x = np.load(fp)
        if not np.isfinite(x).all():
            n_nan = int(np.isnan(x).sum())
            n_inf = int(np.isinf(x).sum())
            mx = float(np.nanmax(np.abs(x))) if np.any(np.isfinite(x)) else float("inf")
            return {"path": str(fp), "label": int(label), "status": "nonfinite",
                    "max_abs": mx, "n_nan": n_nan, "n_inf": n_inf}
        mx = float(np.abs(x).max())
        if mx > threshold:
            return {"path": str(fp), "label": int(label), "status": "extreme",
                    "max_abs": mx}
        return {"status": "clean", "max_abs": mx}
    except Exception as exc:
        return {"path": str(fp), "label": int(label), "status": "error",
                "error": str(exc)}


# ---------------------------------------------------------------------------
# Per-partition scan
# ---------------------------------------------------------------------------
def scan_partition(name, pairs, threshold, n_workers, log):
    log.info("=" * 70)
    log.info("Scanning %s partition: %d segments | threshold=%.1f | workers=%d",
             name, len(pairs), threshold, n_workers)
    log.info("=" * 70)

    t0 = time.time()
    n_clean = 0
    n_nonfinite = 0
    n_extreme = 0
    n_error = 0
    worst_amp = 0.0
    worst_file = ""
    by_subject = defaultdict(lambda: {"clean": 0, "bad": 0, "total": 0})
    bad_records = []

    work = [(fp, label, threshold) for (fp, label) in pairs]
    chunksize = max(50, len(work) // (n_workers * 32))

    with Pool(n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_scan_one, work,
                                                       chunksize=chunksize)):
            status = result["status"]
            mx = result.get("max_abs", float("nan"))

            if status == "clean":
                n_clean += 1
            elif status == "nonfinite":
                n_nonfinite += 1
                if len(bad_records) < MAX_BAD_RECORDS_KEPT:
                    bad_records.append(result)
            elif status == "extreme":
                n_extreme += 1
                if len(bad_records) < MAX_BAD_RECORDS_KEPT:
                    bad_records.append(result)
            else:                                         # error
                n_error += 1
                if len(bad_records) < MAX_BAD_RECORDS_KEPT:
                    bad_records.append(result)

            if isinstance(mx, float) and np.isfinite(mx) and mx > worst_amp:
                worst_amp = mx
                worst_file = result.get("path", worst_file)

            # Per-subject tally (only when we have a path)
            if "path" in result:
                m = _SUBJECT_RE.search(result["path"])
                if m:
                    subj = m.group(0)
                    by_subject[subj]["total"] += 1
                    if status == "clean":
                        by_subject[subj]["clean"] += 1
                    else:
                        by_subject[subj]["bad"] += 1

            if (i + 1) % 50000 == 0:
                rate = (i + 1) / max(time.time() - t0, 1e-3)
                eta = (len(pairs) - i - 1) / max(rate, 1e-3)
                log.info("  scanned %d/%d (%.0f%%) | %.0f files/s | eta %.0f s",
                         i + 1, len(pairs), 100 * (i + 1) / len(pairs), rate, eta)

    elapsed = time.time() - t0
    n_bad = n_nonfinite + n_extreme + n_error

    # -- Summary -------------------------------------------------------------
    log.info("-" * 70)
    log.info("%s SUMMARY", name)
    log.info("  Total scanned        : %d", len(pairs))
    log.info("  Clean                : %d", n_clean)
    log.info("  NaN / Inf            : %d", n_nonfinite)
    log.info("  |x| > %.0f          : %d", threshold, n_extreme)
    log.info("  Read errors          : %d", n_error)
    log.info("  TOTAL BAD            : %d  (%.6f%%)",
             n_bad, 100 * n_bad / max(len(pairs), 1))
    log.info("  Worst amplitude      : %.4e  in %s", worst_amp, worst_file)
    log.info("  Scan duration        : %.1f s  (%.0f files/s)",
             elapsed, len(pairs) / max(elapsed, 1))

    # Top subjects with bad segments
    bad_subjects = sorted(
        [(s, d["bad"], d["total"]) for s, d in by_subject.items() if d["bad"] > 0],
        key=lambda r: r[1], reverse=True)[:15]
    if bad_subjects:
        log.info("  Top subjects by bad-segment count:")
        for subj, bad, total in bad_subjects:
            log.info("    %s: %d bad / %d total (%.3f%%)",
                     subj, bad, total, 100 * bad / max(total, 1))

    # Top worst records
    bad_records.sort(key=lambda r: r.get("max_abs", 0.0)
                     if np.isfinite(r.get("max_abs", float("nan"))) else float("inf"),
                     reverse=True)
    if bad_records:
        log.info("  Top 10 worst records by amplitude:")
        for r in bad_records[:10]:
            mx = r.get("max_abs", float("nan"))
            log.info("    %-12s | label=%d | %s | max_abs=%.4e",
                     r.get("status", "?"),
                     r.get("label", -1),
                     Path(r.get("path", "?")).name,
                     mx)

    return {
        "name": name,
        "n_total": len(pairs),
        "n_clean": n_clean,
        "n_nonfinite": n_nonfinite,
        "n_extreme": n_extreme,
        "n_error": n_error,
        "n_bad": n_bad,
        "bad_pct": 100 * n_bad / max(len(pairs), 1),
        "worst_amp": worst_amp,
        "worst_file": worst_file,
        "scan_seconds": elapsed,
        "by_subject": dict(by_subject),
        "bad_records_top": bad_records[:MAX_BAD_RECORDS_KEPT],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST,
                        help="Primary splits manifest (val + maybe test)")
    parser.add_argument("--alt-manifest", default=DEFAULT_ALT_MANIFEST,
                        help="Fallback manifest used to source test if absent in primary")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="Number of parallel scan workers")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Amplitude threshold |x|>thr flagged as extreme")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Where to write JSON summary")
    args = parser.parse_args()

    log = setup_logging()
    log.info("Scan: extreme/NaN/Inf segments")
    log.info("  manifest     : %s", args.manifest)
    log.info("  alt manifest : %s", args.alt_manifest)
    log.info("  threshold    : %.1f", args.threshold)
    log.info("  workers      : %d", args.workers)
    log.info("  output       : %s", args.output)

    # -- Load primary manifest ----------------------------------------------
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        log.error("Primary manifest not found: %s", manifest_path)
        sys.exit(1)
    log.info("Loading primary: %s", manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    results = {}

    # -- Val partition ------------------------------------------------------
    if "val" in splits and splits["val"]:
        val_pairs = [(rec["filepath"], rec["label"]) for rec in splits["val"]]
        results["val"] = scan_partition("VAL", val_pairs,
                                        args.threshold, args.workers, log)
    else:
        log.warning("No 'val' partition in primary manifest -- skipping val scan")

    # -- Test partition (try primary, fall back to alt) ---------------------
    test_pairs = None
    test_source = None
    if "test" in splits and splits["test"]:
        test_pairs = [(rec["filepath"], rec["label"]) for rec in splits["test"]]
        test_source = str(manifest_path)
        log.info("Found test partition in primary manifest (%d records)", len(test_pairs))
    else:
        alt_path = Path(args.alt_manifest)
        if alt_path.exists():
            log.info("No test in primary -- trying alt: %s", alt_path)
            with open(alt_path, "r", encoding="utf-8") as f:
                alt_splits = json.load(f)
            if "test" in alt_splits and alt_splits["test"]:
                test_pairs = [(rec["filepath"], rec["label"])
                              for rec in alt_splits["test"]]
                test_source = str(alt_path)
                log.info("Loaded test from alt: %d records", len(test_pairs))
            else:
                log.warning("No 'test' in alt manifest either -- skipping test scan")
        else:
            log.warning("Alt manifest not found at %s -- skipping test scan", alt_path)

    if test_pairs:
        results["test"] = scan_partition("TEST", test_pairs,
                                         args.threshold, args.workers, log)
        results["test"]["source_manifest"] = test_source

    # -- Persist JSON -------------------------------------------------------
    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info("Saved: %s", out_path.resolve())

    # -- Final compact summary ---------------------------------------------
    log.info("=" * 70)
    log.info("DONE")
    for part, r in results.items():
        log.info("  %-4s : %7d / %d bad (%.6f%%) | worst |x|=%.4e",
                 part.upper(), r["n_bad"], r["n_total"],
                 r["bad_pct"], r["worst_amp"])


if __name__ == "__main__":
    main()
