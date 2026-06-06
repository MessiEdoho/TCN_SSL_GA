"""
enrich_manifest.py
==================
One-off cluster script that produces a chronology-enriched copy of
data_splits_nonictal_sampled_filtered.json. Each val + test record
gains three additional fields from the per-mouse chronology files:

    "mouse_id"     -- e.g. "m223"
    "chrono_idx"   -- 0..n_grid-1, true chronological position
    "t_start_sec"  -- segment start time in seconds from recording start

By default train records are passed through unchanged (chronology is
not built for train mice in the standard pipeline -- they are
downsampled and shuffled at training time). With --include-train and
the appropriate --input / --output overrides, train records are also
enriched (full-train evaluation use-case; requires that
build_chronology.py was run with --include-train first).

Existing fields ("filepath", "label") are preserved, so consumers
expecting the original schema continue to work; consumers expecting the
new fields (Phase B/C/D production code) read them directly.

Inputs
------
  data_splits_nonictal_sampled_filtered.json    (default)
  /home/people/22206468/scratch/INPUT_DATA/Data_diagnostic/chronologies/{mouse_id}_chronology.npz

CLI overrides (full-train enrichment)
-------------------------------------
  --include-train     : also enrich the train partition
  --input <path>      : override SPLITS_PATH (e.g. data_splits.json un-downsampled)
  --output <path>     : override OUT_PATH

Outputs
-------
  data_splits_nonictal_sampled_filtered_enriched.json   (default)
  enrich_manifest.log
"""

import argparse
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--include-train", action="store_true",
                   help="Also enrich the train partition (default off).")
    p.add_argument("--input",  type=Path, default=None,
                   help="Override SPLITS_PATH (e.g. un-downsampled data_splits.json).")
    p.add_argument("--output", type=Path, default=None,
                   help="Override OUT_PATH.")
    return p.parse_args()


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
    args = parse_args()
    splits_path = args.input  or SPLITS_PATH
    out_path    = args.output or OUT_PATH
    partitions  = (("train",) + PARTITIONS_TO_ENRICH) if args.include_train else PARTITIONS_TO_ENRICH

    logger = setup_logging()
    logger.info("=" * 65)
    logger.info("enrich_manifest.py")
    logger.info("Splits manifest : %s", splits_path)
    logger.info("Chronology dir  : %s", CHRONOLOGY_DIR)
    logger.info("Output          : %s", out_path)
    logger.info("Log             : %s", LOG_PATH)
    logger.info("Partitions      : %s%s", partitions,
                "" if args.include_train else " (train passed through unchanged)")
    logger.info("=" * 65)

    if not splits_path.exists():
        logger.error("Splits manifest missing: %s", splits_path); sys.exit(1)
    if not CHRONOLOGY_DIR.exists():
        logger.error("Chronology dir missing: %s. Run build_chronology.py first.",
                     CHRONOLOGY_DIR); sys.exit(1)

    manifest = json.loads(splits_path.read_text(encoding="utf-8"))

    for partition in partitions:
        records = manifest.get(partition, [])
        if not records:
            logger.warning("Partition '%s' is empty.", partition); continue
        enrich_partition(partition, records, logger)

    if "metadata" not in manifest:
        manifest["metadata"] = {}
    manifest["metadata"]["enriched_with_chronology"] = True
    manifest["metadata"]["chronology_dir"]           = str(CHRONOLOGY_DIR)
    manifest["metadata"]["enriched_partitions"]      = list(partitions)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    size_mb = out_path.stat().st_size / 1e6
    logger.info("-" * 65)
    logger.info("Wrote enriched manifest: %s (%.2f MB)", out_path, size_mb)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
