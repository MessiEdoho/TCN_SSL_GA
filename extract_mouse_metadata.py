"""
extract_mouse_metadata.py
=========================
One-off cluster script that extracts per-mouse EDF header metadata for
every mouse in the val + test partitions of the splits manifest.

Output is a single small JSON file consumed by the local recovery
script (recover_event_metrics.py) which rebuilds the chronological
ordering of segments and computes corrected event-level metrics.

Why this script exists
----------------------
The Excel seizure annotations store seizure times as absolute wall-clock
datetimes (DD/MM/YYYY HH:MM:SS.fff). Converting them to
seconds-from-recording-start requires the recording's start datetime,
which lives only in the EDF header. The local recovery script cannot
proceed without it.

Inputs
------
  data_splits_nonictal_sampled_filtered.json   (default; val + test mice)
  EDF files under /home/people/22206468/scratch/Raw EDF/

CLI overrides (full-train metadata coverage)
--------------------------------------------
  --include-train         : add train mice to PARTITIONS_OF_INTEREST.
                            Required when downstream full-train pipeline
                            (build_chronology --include-train,
                            enrich_manifest --include-train,
                            full_train_eval_*.py) needs metadata for the
                            ~72 train-only mice.
  --input-splits <path>   : override SPLITS_PATH. For full-train use,
                            point at the un-downsampled data_splits.json.

Output
------
  /home/people/22206468/scratch/INPUT_DATA/Data_diagnostic/mouse_recording_metadata.json
  /home/people/22206468/scratch/INPUT_DATA/Data_diagnostic/extract_mouse_metadata.log
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import mne


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SPLITS_PATH = Path("/scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits_nonictal_sampled_filtered.json")
EDF_ROOT    = Path("/home/people/22206468/scratch/Raw EDF")
OUT_DIR     = Path("/home/people/22206468/scratch/INPUT_DATA/Data_diagnostic")
OUT_PATH    = OUT_DIR / "mouse_recording_metadata.json"
LOG_PATH    = OUT_DIR / "extract_mouse_metadata.log"

# Partitions to extract metadata for. Train is excluded by default -- it is
# downsampled and shuffled at training time, so chronological order is not
# used there. Pass --include-train to add it (required for the full-train
# eval + sweep pipeline).
PARTITIONS_OF_INTEREST = ("val", "test")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--include-train", action="store_true",
                   help="Add train mice to PARTITIONS_OF_INTEREST (default off).")
    p.add_argument("--input-splits", type=Path, default=None,
                   help="Override SPLITS_PATH (e.g. un-downsampled data_splits.json).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------
def setup_logging():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("extract_mouse_metadata")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# collect_mouse_ids
# ---------------------------------------------------------------------------
def collect_mouse_ids(splits_path, partitions, logger):
    """Read the manifest and return the set of unique mouse_ids appearing
    in the requested partitions. Mouse_id is the prefix of each segment
    filename's stem before the first underscore (e.g. 'm1_ictal_00042.npy'
    -> 'm1'), matching the convention used by preprocessing_binary.py.
    """
    if not splits_path.exists():
        logger.error("Splits manifest not found: %s", splits_path)
        sys.exit(1)

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    mouse_to_partitions = {}
    for part in partitions:
        records = splits.get(part, [])
        if not records:
            logger.warning("Partition '%s' is empty in manifest.", part)
            continue
        for rec in records:
            mid = Path(rec["filepath"]).stem.split("_")[0]
            mouse_to_partitions.setdefault(mid, set()).add(part)

    logger.info("Found %d unique mice across partitions %s:",
                len(mouse_to_partitions), partitions)
    for mid in sorted(mouse_to_partitions):
        parts = sorted(mouse_to_partitions[mid])
        logger.info("  %-8s : %s", mid, ", ".join(parts))

    return mouse_to_partitions


# ---------------------------------------------------------------------------
# find_edf_for_mouse
# ---------------------------------------------------------------------------
def find_edf_for_mouse(mouse_id, edf_root, logger):
    """Locate the EDF file for a given mouse_id under edf_root. Files
    are named {mouse_id}.edf (e.g. m1.edf, m223.edf).
    """
    edf_path = edf_root / f"{mouse_id}.edf"
    if not edf_path.exists():
        logger.error("No EDF found for mouse %s at %s", mouse_id, edf_path)
        return None
    return edf_path


# ---------------------------------------------------------------------------
# extract_one_mouse
# ---------------------------------------------------------------------------
def extract_one_mouse(mouse_id, edf_path, partitions, logger):
    """Read the EDF header and return a metadata dict. Does NOT load
    sample data (preload=False) so this is fast even on multi-GB EDFs.
    """
    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")

    n_samples = int(raw.n_times)
    fs_hz     = float(raw.info["sfreq"])
    meas_date = raw.info["meas_date"]
    if meas_date is None:
        logger.error("EDF for mouse %s has no meas_date in header: %s",
                     mouse_id, edf_path)
        raise ValueError(f"meas_date missing for {mouse_id}")
    rec_start = meas_date.replace(tzinfo=None)
    duration  = n_samples / fs_hz

    record = {
        "mouse_id":           mouse_id,
        "edf_path":           str(edf_path),
        "n_samples":          n_samples,
        "fs_hz":              fs_hz,
        "recording_start_dt": rec_start.isoformat(),
        "duration_sec":       round(duration, 3),
        "duration_hr":        round(duration / 3600.0, 3),
        "channels":           list(raw.ch_names),
        "partitions":         sorted(partitions),
    }
    logger.info("  %-8s : %d samples @ %.0f Hz (%.1f h) | start=%s | edf=%s",
                mouse_id, n_samples, fs_hz, record["duration_hr"],
                record["recording_start_dt"], edf_path.name)
    return record


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    splits_path = args.input_splits or SPLITS_PATH
    partitions  = (("train",) + PARTITIONS_OF_INTEREST) if args.include_train else PARTITIONS_OF_INTEREST

    logger = setup_logging()
    logger.info("=" * 65)
    logger.info("extract_mouse_metadata.py")
    logger.info("Splits manifest : %s", splits_path)
    logger.info("EDF root        : %s", EDF_ROOT)
    logger.info("Output JSON     : %s", OUT_PATH)
    logger.info("Output log      : %s", LOG_PATH)
    logger.info("Partitions      : %s%s", partitions,
                "" if args.include_train else " (train excluded)")
    logger.info("=" * 65)

    if not EDF_ROOT.exists():
        logger.error("EDF root does not exist: %s", EDF_ROOT)
        sys.exit(1)

    mouse_to_partitions = collect_mouse_ids(splits_path, partitions, logger)
    if not mouse_to_partitions:
        logger.error("No mice collected from the manifest. Aborting.")
        sys.exit(1)

    logger.info("-" * 65)
    logger.info("Locating EDFs and reading headers...")

    metadata = {}
    failures = []
    for mouse_id in sorted(mouse_to_partitions):
        edf_path = find_edf_for_mouse(mouse_id, EDF_ROOT, logger)
        if edf_path is None:
            failures.append(mouse_id)
            continue
        try:
            record = extract_one_mouse(
                mouse_id, edf_path, mouse_to_partitions[mouse_id], logger)
            metadata[mouse_id] = record
        except Exception as exc:
            logger.error("Failed to read %s: %s", edf_path, exc)
            failures.append(mouse_id)

    OUT_PATH.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    logger.info("-" * 65)
    logger.info("Saved metadata for %d/%d mice to %s",
                len(metadata), len(mouse_to_partitions), OUT_PATH)
    if failures:
        logger.error("Missing/failed mice (%d): %s", len(failures), sorted(failures))
        sys.exit(2)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
