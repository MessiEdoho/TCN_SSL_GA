"""
build_chronology.py
===================
One-off cluster script that builds per-mouse chronology files for every
mouse in the val + test partitions of the splits manifest. A chronology
file is the deterministic output of replaying the preprocessing grid
walk against the same Excel annotations preprocessing consumed -- it
maps each generated .npy filename to its true chronological position
in the recording, enabling correct event-level metrics in downstream
training/eval scripts.

Inputs
------
  data_splits_nonictal_sampled_filtered.json
  mouse_recording_metadata.json                (per-mouse n_samples, fs_hz,
                                                recording_start_dt from EDF
                                                headers; produced by
                                                extract_mouse_metadata.py)
  /home/people/22206468/scratch/seizure_times_updated/{mouse}_xlsx.xlsx

Output (under OUT_DIR)
----------------------
  {mouse_id}_chronology.npz   -- one per val/test mouse, with arrays:
      fnames        : numpy string array, sorted by chronological position
      chrono_idx    : int64,  values 0..n_grid-1 (== the array index)
      t_start_sec   : float64, segment start time in seconds from
                      recording start
      label         : int8,    1 = ictal, 0 = non-ictal
    Plus 0-d metadata arrays:
      mouse_id, n_samples, fs_hz, recording_start_dt, n_grid,
      n_ictal, n_nonictal, partitions
  build_chronology.log        -- persistent log
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

from eval_utils import build_mouse_chronology, load_annotations


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SPLITS_PATH    = Path("/scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits_nonictal_sampled_filtered.json")
METADATA_PATH  = Path("/home/people/22206468/scratch/INPUT_DATA/Data_diagnostic/mouse_recording_metadata.json")
ANNOT_DIR      = Path("/home/people/22206468/scratch/seizure_times_updated")
OUT_DIR        = Path("/home/people/22206468/scratch/INPUT_DATA/Data_diagnostic/chronologies")
LOG_PATH       = Path("/home/people/22206468/scratch/INPUT_DATA/Data_diagnostic/build_chronology.log")

PARTITIONS_OF_INTEREST = ("val", "test")


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------
def setup_logging():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("build_chronology")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# collect_mouse_ids -- mirrors extract_mouse_metadata helper
# ---------------------------------------------------------------------------
def collect_mouse_ids(splits_path, partitions, logger):
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    mouse_to_partitions = {}
    for part in partitions:
        for rec in splits.get(part, []):
            mid = Path(rec["filepath"]).stem.split("_")[0]
            mouse_to_partitions.setdefault(mid, set()).add(part)
    logger.info("Found %d unique mice across partitions %s",
                len(mouse_to_partitions), partitions)
    return mouse_to_partitions


# ---------------------------------------------------------------------------
# save_chronology_npz
# ---------------------------------------------------------------------------
def save_chronology_npz(out_path, fname_to_chrono, mouse_id, mouse_meta,
                        n_grid, n_ictal, n_nonictal, partitions):
    """Convert the dict to parallel arrays sorted by chrono_idx and dump
    as a single compressed NPZ. Sorted arrays mean chrono_idx[i] == i, so
    a downstream lookup by position is just an array index.
    """
    items = sorted(fname_to_chrono.items(), key=lambda kv: kv[1]["chrono_idx"])
    fnames     = np.asarray([k for k, _ in items], dtype=object)
    chrono_idx = np.asarray([v["chrono_idx"]  for _, v in items], dtype=np.int64)
    t_start    = np.asarray([v["t_start_sec"] for _, v in items], dtype=np.float64)
    label      = np.asarray([v["label"]       for _, v in items], dtype=np.int8)

    np.savez_compressed(
        out_path,
        fnames=fnames,
        chrono_idx=chrono_idx,
        t_start_sec=t_start,
        label=label,
        mouse_id=np.array(mouse_id),
        n_samples=np.int64(mouse_meta["n_samples"]),
        fs_hz=np.float64(mouse_meta["fs_hz"]),
        recording_start_dt=np.array(mouse_meta["recording_start_dt"]),
        n_grid=np.int64(n_grid),
        n_ictal=np.int64(n_ictal),
        n_nonictal=np.int64(n_nonictal),
        partitions=np.array(sorted(partitions)),
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    logger = setup_logging()
    logger.info("=" * 65)
    logger.info("build_chronology.py")
    logger.info("Splits manifest : %s", SPLITS_PATH)
    logger.info("Mouse metadata  : %s", METADATA_PATH)
    logger.info("Annotations dir : %s", ANNOT_DIR)
    logger.info("Output dir      : %s", OUT_DIR)
    logger.info("Log             : %s", LOG_PATH)
    logger.info("Partitions      : %s (train excluded)", PARTITIONS_OF_INTEREST)
    logger.info("=" * 65)

    for p in [SPLITS_PATH, METADATA_PATH, ANNOT_DIR]:
        if not p.exists():
            logger.error("Required input missing: %s", p); sys.exit(1)

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    mouse_to_partitions = collect_mouse_ids(SPLITS_PATH, PARTITIONS_OF_INTEREST, logger)

    failures = []
    completed = 0
    import datetime
    for mouse_id in sorted(mouse_to_partitions):
        if mouse_id not in metadata:
            logger.error("  %s : no metadata record. Skipping.", mouse_id)
            failures.append(mouse_id); continue
        meta = metadata[mouse_id]
        xlsx_path = ANNOT_DIR / f"{mouse_id}_xlsx.xlsx"
        if not xlsx_path.exists():
            logger.error("  %s : annotation Excel not found at %s. Skipping.",
                         mouse_id, xlsx_path)
            failures.append(mouse_id); continue

        try:
            rec_start = datetime.datetime.fromisoformat(meta["recording_start_dt"])
            seizure_intervals = load_annotations(xlsx_path, rec_start)
            logger.info("  %s : %d ground-truth seizure(s) in %s",
                        mouse_id, len(seizure_intervals), xlsx_path.name)

            fname_to_chrono, n_ictal, n_nonictal = build_mouse_chronology(
                mouse_id, int(meta["n_samples"]), seizure_intervals, logger)

            out_path = OUT_DIR / f"{mouse_id}_chronology.npz"
            save_chronology_npz(out_path, fname_to_chrono, mouse_id, meta,
                                len(fname_to_chrono), n_ictal, n_nonictal,
                                mouse_to_partitions[mouse_id])
            size_mb = out_path.stat().st_size / 1e6
            logger.info("  %s : saved %s (%.2f MB)", mouse_id, out_path.name, size_mb)
            completed += 1
        except Exception as exc:
            logger.error("  %s : failed -- %s", mouse_id, exc)
            failures.append(mouse_id)

    logger.info("-" * 65)
    logger.info("Completed: %d/%d mice", completed, len(mouse_to_partitions))
    if failures:
        logger.error("Failures (%d): %s", len(failures), sorted(failures))
        sys.exit(2)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
