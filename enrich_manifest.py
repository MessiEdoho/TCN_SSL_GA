"""
enrich_manifest.py
==================
One-off cluster script that produces a chronology-enriched copy of
data_splits_nonictal_sampled_filtered.json. Each val + test record
gains three additional fields from the per-mouse chronology files:

    "mouse_id"     -- e.g. "m223"
    "chrono_idx"   -- 0..n_grid-1, true chronological position
    "t_start_sec"  -- segment start time in seconds from recording start

Train records are passed through unchanged (chronology is not built for
train mice -- they are downsampled and shuffled at training time).

Existing fields ("filepath", "label") are preserved, so consumers
expecting the original schema continue to work; consumers expecting the
new fields (Phase B/C/D production code) read them directly.

Inputs
------
  data_splits_nonictal_sampled_filtered.json
  /home/people/22206468/scratch/INPUT_DATA/Data_diagnostic/chronologies/{mouse_id}_chronology.npz

Outputs
-------
  data_splits_nonictal_sampled_filtered_enriched.json   (alongside original)
  enrich_manifest.log
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SPLITS_DIR     = Path("/scratch/22206468/INPUT_DATA/data_splits_outputs")
SPLITS_PATH    = SPLITS_DIR / "data_splits_nonictal_sampled_filtered.json"
OUT_PATH       = SPLITS_DIR / "data_splits_nonictal_sampled_filtered_enriched.json"
CHRONOLOGY_DIR = Path("/home/people/22206468/scratch/INPUT_DATA/Data_diagnostic/chronologies")
LOG_PATH       = Path("/home/people/22206468/scratch/INPUT_DATA/Data_diagnostic/enrich_manifest.log")

PARTITIONS_TO_ENRICH = ("val", "test")


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------
def setup_logging():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("enrich_manifest")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# load_chronology -- one mouse, returns {fname: (chrono_idx, t_start, label)}
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
# enrich_partition
# ---------------------------------------------------------------------------
def enrich_partition(partition, records, logger):
    chronologies = {}
    n_unmatched = 0
    n_label_disagree = 0

    for rec in records:
        fname = Path(rec["filepath"]).name
        mouse = fname.split("_")[0]
        rec["mouse_id"] = mouse

        if mouse not in chronologies:
            chronologies[mouse] = load_chronology(mouse, logger)
        chmap = chronologies[mouse]
        if chmap is None:
            n_unmatched += 1
            rec["chrono_idx"]  = -1
            rec["t_start_sec"] = -1.0
            continue

        info = chmap.get(fname)
        if info is None:
            n_unmatched += 1
            rec["chrono_idx"]  = -1
            rec["t_start_sec"] = -1.0
            continue

        chrono_idx, t_start, label = info
        if int(rec["label"]) != label:
            n_label_disagree += 1
        rec["chrono_idx"]  = chrono_idx
        rec["t_start_sec"] = round(t_start, 4)

    logger.info("  %s : %d records | %d unmatched | %d label disagree",
                partition, len(records), n_unmatched, n_label_disagree)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    logger = setup_logging()
    logger.info("=" * 65)
    logger.info("enrich_manifest.py")
    logger.info("Splits manifest : %s", SPLITS_PATH)
    logger.info("Chronology dir  : %s", CHRONOLOGY_DIR)
    logger.info("Output          : %s", OUT_PATH)
    logger.info("Log             : %s", LOG_PATH)
    logger.info("Partitions      : %s (train passed through unchanged)",
                PARTITIONS_TO_ENRICH)
    logger.info("=" * 65)

    if not SPLITS_PATH.exists():
        logger.error("Splits manifest missing: %s", SPLITS_PATH); sys.exit(1)
    if not CHRONOLOGY_DIR.exists():
        logger.error("Chronology dir missing: %s. Run build_chronology.py first.",
                     CHRONOLOGY_DIR); sys.exit(1)

    manifest = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))

    for partition in PARTITIONS_TO_ENRICH:
        records = manifest.get(partition, [])
        if not records:
            logger.warning("Partition '%s' is empty.", partition); continue
        enrich_partition(partition, records, logger)

    if "metadata" not in manifest:
        manifest["metadata"] = {}
    manifest["metadata"]["enriched_with_chronology"] = True
    manifest["metadata"]["chronology_dir"]           = str(CHRONOLOGY_DIR)

    OUT_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    size_mb = OUT_PATH.stat().st_size / 1e6
    logger.info("-" * 65)
    logger.info("Wrote enriched manifest: %s (%.2f MB)", OUT_PATH, size_mb)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
