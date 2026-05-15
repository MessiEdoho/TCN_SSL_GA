"""
count_train_per_mouse.py
========================
One-off helper. Reads the splits manifest, takes the train partition,
and emits a per-mouse segment-count CSV intended for stratifying a
k-fold cross-validation by subject AND ictal prevalence.

CPU-only, no GPU, no model loading -- pure JSON read + filename parse.
Runs in seconds.

Inputs
------
  data_splits_nonictal_sampled_filtered.json   (or _enriched, both work)

Output
------
  /home/people/22206468/scratch/INPUT_DATA/Data_diagnostic/train_per_mouse_counts.csv
  /home/people/22206468/scratch/INPUT_DATA/Data_diagnostic/count_train_per_mouse.log

CSV columns
-----------
  mouse_id
  n_ictal_segments
  n_nonictal_segments
  n_total_segments
  ictal_prevalence_pct
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path


SCRATCH = Path("/home/people/22206468/scratch")
DEFAULT_SPLITS = SCRATCH / "INPUT_DATA" / "data_splits_outputs" / "data_splits_nonictal_sampled_filtered.json"
DEFAULT_OUT_CSV = SCRATCH / "INPUT_DATA" / "Data_diagnostic" / "train_per_mouse_counts.csv"
DEFAULT_LOG_PATH = SCRATCH / "INPUT_DATA" / "Data_diagnostic" / "count_train_per_mouse.log"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=DEFAULT_SPLITS)
    p.add_argument("--output",   type=Path, default=DEFAULT_OUT_CSV)
    p.add_argument("--log",      type=Path, default=DEFAULT_LOG_PATH)
    return p.parse_args()


def setup_logging(log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("count_train_per_mouse")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8"); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger


def natural_key(mid):
    try:
        return (0, int(mid.lstrip("m")))
    except ValueError:
        return (1, mid)


def main():
    args = parse_args()
    logger = setup_logging(args.log)
    logger.info("=" * 65)
    logger.info("count_train_per_mouse.py")
    logger.info("Manifest : %s", args.manifest)
    logger.info("Output   : %s", args.output)
    logger.info("Log      : %s", args.log)
    logger.info("=" * 65)

    if not args.manifest.exists():
        logger.error("Manifest not found: %s", args.manifest); sys.exit(1)

    splits = json.loads(args.manifest.read_text(encoding="utf-8"))
    train_records = splits.get("train", [])
    if not train_records:
        logger.error("Train partition is empty in manifest %s", args.manifest); sys.exit(1)

    per_mouse = {}
    for rec in train_records:
        mid = Path(rec["filepath"]).stem.split("_")[0]
        d = per_mouse.setdefault(mid, {"n_ictal": 0, "n_nonictal": 0})
        if int(rec["label"]) == 1:
            d["n_ictal"] += 1
        else:
            d["n_nonictal"] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mouse_id", "n_ictal_segments", "n_nonictal_segments",
                    "n_total_segments", "ictal_prevalence_pct"])
        tot_i = tot_n = 0
        for mid in sorted(per_mouse.keys(), key=natural_key):
            d = per_mouse[mid]
            total = d["n_ictal"] + d["n_nonictal"]
            prev_pct = round(100.0 * d["n_ictal"] / total, 4) if total > 0 else 0.0
            w.writerow([mid, d["n_ictal"], d["n_nonictal"], total, prev_pct])
            tot_i += d["n_ictal"]; tot_n += d["n_nonictal"]
        tot_total = tot_i + tot_n
        tot_prev = round(100.0 * tot_i / tot_total, 4) if tot_total > 0 else 0.0
        w.writerow(["TOTAL", tot_i, tot_n, tot_total, tot_prev])

    logger.info("Mice in train  : %d", len(per_mouse))
    logger.info("Total segments : %d (%d ictal, %d non-ictal, %.2f%% ictal)",
                tot_total, tot_i, tot_n, tot_prev)
    logger.info("Saved CSV      : %s", args.output)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
