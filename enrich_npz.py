"""
enrich_npz.py
=============
One-off cluster script that takes the existing cached prediction NPZs
(M3 val + M3 test) produced before the chronology infrastructure
existed, and rewrites them with three additional per-segment arrays --
mouse_id, chrono_idx, t_start_sec -- so downstream consumers can
reorder them per-mouse-chronologically without rebuilding the
chronology each time.

The original y_true and y_prob (and any other arrays already present)
are preserved bit-identical -- only metadata arrays are added. No
model inference is re-run.

Inputs
------
  data_splits_nonictal_sampled_filtered.json
  /home/people/22206468/scratch/INPUT_DATA/Data_diagnostic/chronologies/{mouse_id}_chronology.npz
  /home/people/22206468/scratch/OUTPUT/MODEL3_OUTPUT/MultiScaleTCN/multiscale_tcn_predictions_raw.npz   (val)
  /home/people/22206468/scratch/OUTPUT/MODEL3_OUTPUT/MultiScaleTCN/evaluation/multiscale_tcn_test_predictions_raw.npz   (test)

Outputs
-------
  multiscale_tcn_predictions_raw_enriched.npz       (alongside val NPZ)
  multiscale_tcn_test_predictions_raw_enriched.npz  (alongside test NPZ)
  enrich_npz.log                                    (Data_diagnostic/)
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SPLITS_PATH    = Path("/scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits_nonictal_sampled_filtered.json")
CHRONOLOGY_DIR = Path("/home/people/22206468/scratch/INPUT_DATA/Data_diagnostic/chronologies")
LOG_PATH       = Path("/home/people/22206468/scratch/INPUT_DATA/Data_diagnostic/enrich_npz.log")

NPZ_INPUTS = {
    "val":  Path("/home/people/22206468/scratch/OUTPUT/MODEL3_OUTPUT/MultiScaleTCN/multiscale_tcn_predictions_raw.npz"),
    "test": Path("/home/people/22206468/scratch/OUTPUT/MODEL3_OUTPUT/MultiScaleTCN/evaluation/multiscale_tcn_test_predictions_raw.npz"),
}


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------
def setup_logging():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("enrich_npz")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# load_chronology -- one mouse, returns dict fname -> (chrono_idx, t_start, label)
# ---------------------------------------------------------------------------
def load_chronology(mouse_id, logger):
    path = CHRONOLOGY_DIR / f"{mouse_id}_chronology.npz"
    if not path.exists():
        logger.error("  chronology missing for %s: %s", mouse_id, path)
        return None
    npz = np.load(path, allow_pickle=True)
    fnames     = npz["fnames"]
    chrono_idx = npz["chrono_idx"]
    t_start    = npz["t_start_sec"]
    label      = npz["label"]
    return {str(fnames[i]): (int(chrono_idx[i]), float(t_start[i]), int(label[i]))
            for i in range(len(fnames))}


# ---------------------------------------------------------------------------
# enrich_one_partition
# ---------------------------------------------------------------------------
def enrich_one_partition(partition, manifest, logger):
    npz_in_path = NPZ_INPUTS[partition]
    if not npz_in_path.exists():
        logger.error("Input NPZ missing for %s: %s", partition, npz_in_path); return False

    logger.info("-" * 65)
    logger.info("Partition: %s", partition.upper())
    logger.info("Reading   : %s", npz_in_path)

    npz = np.load(npz_in_path, allow_pickle=False)
    existing = {k: npz[k] for k in npz.files}
    n_npz = len(existing["y_true"])

    records = manifest.get(partition, [])
    if len(records) != n_npz:
        logger.error("Length mismatch: manifest %d vs NPZ %d", len(records), n_npz)
        return False

    chronologies = {}
    mouse_id_arr   = np.empty(n_npz, dtype=object)
    chrono_idx_arr = np.full(n_npz, -1, dtype=np.int64)
    t_start_arr    = np.full(n_npz, np.nan, dtype=np.float64)

    n_unmatched = 0
    n_label_disagree = 0
    for npz_idx, rec in enumerate(records):
        fname = Path(rec["filepath"]).name
        mouse = fname.split("_")[0]

        if mouse not in chronologies:
            chronologies[mouse] = load_chronology(mouse, logger)
            if chronologies[mouse] is None:
                continue

        info = chronologies[mouse].get(fname)
        if info is None:
            n_unmatched += 1
            continue

        chrono_idx, t_start, label = info
        if int(rec["label"]) != label:
            n_label_disagree += 1
        if int(existing["y_true"][npz_idx]) != label:
            n_label_disagree += 1

        mouse_id_arr[npz_idx]   = mouse
        chrono_idx_arr[npz_idx] = chrono_idx
        t_start_arr[npz_idx]    = t_start

    if n_unmatched:
        logger.warning("%s : %d manifest segments not matched in chronology.",
                       partition, n_unmatched)
    if n_label_disagree:
        logger.warning("%s : %d label disagreements between manifest/NPZ and chronology.",
                       partition, n_label_disagree)

    mouse_id_arr_str = np.array([m if m is not None else "" for m in mouse_id_arr], dtype=object)

    out_path = npz_in_path.with_name(npz_in_path.stem + "_enriched.npz")
    np.savez_compressed(
        out_path,
        mouse_id=mouse_id_arr_str,
        chrono_idx=chrono_idx_arr,
        t_start_sec=t_start_arr,
        **existing,
    )
    size_mb = out_path.stat().st_size / 1e6
    logger.info("Wrote     : %s (%.2f MB)", out_path, size_mb)
    return True


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    logger = setup_logging()
    logger.info("=" * 65)
    logger.info("enrich_npz.py")
    logger.info("Splits manifest : %s", SPLITS_PATH)
    logger.info("Chronology dir  : %s", CHRONOLOGY_DIR)
    logger.info("Log             : %s", LOG_PATH)
    logger.info("=" * 65)

    if not SPLITS_PATH.exists():
        logger.error("Splits manifest missing: %s", SPLITS_PATH); sys.exit(1)
    if not CHRONOLOGY_DIR.exists():
        logger.error("Chronology dir missing: %s. Run build_chronology.py first.",
                     CHRONOLOGY_DIR); sys.exit(1)

    manifest = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))

    failures = []
    for partition in ("val", "test"):
        if not enrich_one_partition(partition, manifest, logger):
            failures.append(partition)

    if failures:
        logger.error("Failed partitions: %s", failures); sys.exit(2)
    logger.info("=" * 65); logger.info("DONE")


if __name__ == "__main__":
    main()
