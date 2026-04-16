"""
create_balanced_splits.py
=========================
Standalone script that produces data_splits_nonictal_sampled.json --
a proximity-aware downsampled training manifest for the ablation study.

Run ONCE on the cluster before any tuning or training job. Reads the
authoritative sources (data_splits.json, EDF headers, seizure annotation
.xlsx files) and produces a balanced manifest without creating, moving,
or deleting any .npy files on disk. The original non_seizure/ folders
are preserved intact for future self-supervised learning (SSL) work.

Motivation
----------
The raw training corpus has a 1:438 ictal-to-non-ictal ratio (65,510
ictal vs ~28.7M non-ictal segments). Uniform random downsampling to
1:4 (the previous approach via downsample_non_ictal) discards non-ictal
segments indiscriminately, losing "hard negatives" near seizure
boundaries that are most informative for learning the decision boundary.

Proximity-aware sampling addresses this by retaining ALL non-ictal
segments within a temporal margin of seizure events (peri-ictal context)
and subsampling only the distant background. This approach is motivated
by curriculum learning and hard-example mining literature:

  - Shrivastava, A., Gupta, A., & Girshick, R. (2016). Training Region-
    Based Object Detectors with Online Hard Example Mining. CVPR 2016.
    Hard negatives (near-boundary examples) contribute disproportionately
    to gradient signal; easy negatives saturate quickly.

  - Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009).
    Curriculum Learning. ICML 2009. Training on progressively harder
    examples improves generalisation.

  - Roy, S., Kiral-Kornek, I., & Bhattachayya, S. (2019). ChronoNet:
    a deep recurrent neural network for abnormal EEG identification.
    Artificial Intelligence in Medicine, 103, 101789. Used stratified
    sampling to balance seizure/non-seizure classes in EEG datasets.

In seizure detection, the peri-ictal period (seconds before and after
a seizure) contains EEG patterns that are morphologically similar to
ictal activity -- pre-ictal discharges, post-ictal slowing, and
transitional rhythms. These segments are the hardest for a classifier
to distinguish from true seizures and therefore the most valuable
training examples (Litt & Echauz, 2002).

Downsampling strategy
---------------------
1. Remove m254 (ictal-free subject -- cannot participate in stratified
   sampling; identified during previous pipeline runs).
2. Remove extreme segments (max|x| > 1000, indicating failed z-score
   normalisation from interrupted preprocessing; 6 segments from 5
   subjects identified in debug runs).
3. Keep ALL ictal segments (~65,510).
4. Keep ALL non-ictal within ±MARGIN_SEC (default 60 s) of any seizure.
   These are the "hard negatives" -- peri-ictal segments near seizure
   boundaries. The 60 s margin captures pre-ictal build-up and post-
   ictal recovery periods typical in rodent EEG (Luttjohann et al., 2009).
5. Randomly sample FAR_SAMPLE_FRAC (default 5%) of the near-seizure
   count from distant non-ictal segments. This preserves diversity
   in the background class (quiet sleep, active waking, grooming
   artefacts) without overwhelming the corpus with easy negatives.
6. Val partition passed through unchanged -- no downsampling applied
   to validation data. Tuning scripts apply their own 10% stratified
   subset via downsample_val_stratified().

Time-position reconstruction
-----------------------------
The filename convention {mouse_id}_{ictal|nonictal}_{index:05d}.npy
uses a sequential per-class counter, NOT a grid position. To recover
each segment's temporal position (and thus distance to nearest seizure),
we re-read each mouse's EDF header (preload=False -- reads only
metadata, ~10 ms per file) and annotation file, rebuild the full
segment grid, and map the k-th non-ictal grid position to
nonictal_{k:05d}.npy.

Output
------
/scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits_nonictal_sampled.json

Pipeline position
-----------------
1. generate_data_splits.py        -> data_splits.json
2. create_balanced_splits.py      -> data_splits_nonictal_sampled.json  (this script)
3. tcn_HPT_binary.py / tune_*.py  -> tuning (loads balanced manifest)
4. TCN.py / MultiScaleTCN.py etc  -> training (loads balanced manifest)

References
----------
Shrivastava, A., Gupta, A., & Girshick, R. (2016). Training Region-
    Based Object Detectors with Online Hard Example Mining. CVPR 2016.
Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009).
    Curriculum Learning. ICML 2009.
Roy, S., Kiral-Kornek, I., & Bhattachayya, S. (2019). ChronoNet.
    Artificial Intelligence in Medicine, 103, 101789.
Litt, B. & Echauz, J. (2002). Prediction of epileptic seizures.
    The Lancet Neurology, 1(1), 22-30.
Luttjohann, A., Fabene, P. F., & van Luijtelaar, G. (2009). A revised
    Racine's scale for PTZ-induced seizures in rats. Physiology &
    Behavior, 98(5), 579-586.

Usage
-----
python create_balanced_splits.py

RESEARCH REPORTING NOTE
-----------------------
Report the following in the Methods section:

  "Non-ictal training segments were selected using proximity-aware
   stratified sampling. All non-ictal segments within ±60 s of any
   annotated seizure boundary were retained as hard negatives. An
   additional 5% of the near-seizure count was randomly sampled from
   distant non-ictal segments to preserve background diversity. This
   yielded a training corpus of N_ictal ictal and N_nonictal non-ictal
   segments (ratio X:1). The manifest was generated once offline and
   reused across all tuning and training runs. Seed=42 was fixed for
   reproducibility. Subject m254 (no ictal segments) was excluded.
   Six segments from five subjects with max|x| > 1000 (indicating
   incomplete z-score normalisation) were removed."

Parameters to report:
  MARGIN_SEC         : 60 s
  FAR_SAMPLE_FRAC    : 0.05
  AMPLITUDE_THRESHOLD: 1000.0
  SEED               : 42
  Excluded subjects  : m254
  N extreme removed  : 6
"""

import json
import logging
import sys
import time
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import mne
import pandas as pd
import matplotlib
matplotlib.use("Agg")                               # non-interactive backend for cluster
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42                        # reproducibility seed for random sampling of distant non-ictal
FS = 500                         # EEG sampling rate in Hz (must match preprocessing)
WIN_LEN = 2500                   # segment length in samples: 5 s * 500 Hz (must match preprocessing)
STEP = 1250                      # step size in samples: 2.5 s, 50% overlap (must match preprocessing)
MARGIN_SEC = 60.0                # ±60 s margin: non-ictal within this distance are "near-seizure"
FAR_SAMPLE_FRAC = 0.05           # fraction of near-seizure count to sample from distant non-ictal
AMPLITUDE_THRESHOLD = 1000.0     # segments with max|x| > this are removed (failed z-score normalisation)

# -- Paths -----------------------------------------------------------------
# Input: original data_splits.json produced by generate_data_splits.py
SPLITS_PATH = Path("/scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits.json")
# Output: balanced manifest consumed by all tuning and training scripts
OUTPUT_PATH = Path("/scratch/22206468/INPUT_DATA/data_splits_outputs/data_splits_nonictal_sampled.json")
# Seizure annotation Excel files (one per mouse: {mouse_id}_xlsx.xlsx)
ANNOTATION_DIR = Path("/home/people/22206468/scratch/seizure_times_updated")

# EDF directories -- one per preprocessing batch. Each contains the raw
# .edf recordings for a subset of mice. We only read headers (preload=False)
# to get n_samples for grid reconstruction, NOT the signal data.
EDF_DIRS = [
    Path("/home/people/22206468/scratch/EEG_TRAINING"),       # preprocessing_binary.py output
    Path("/home/people/22206468/scratch/EEG_TRAINING_2"),     # preprocessing_binary_train_2.py output
    Path("/home/people/22206468/scratch/EEG_TRAINING_3"),     # preprocessing_binary_train_3.py output
    Path("/home/people/22206468/scratch/EEG_TRAINING_4"),     # preprocessing_binary_train_4.py output
    Path("/home/people/22206468/scratch/EEG_TRAINING_5"),     # preprocessing_binary_train_5.py output
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("balanced_splits")
logger.setLevel(logging.INFO)
logger.handlers.clear()

fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(fmt)
logger.addHandler(sh)

fh = logging.FileHandler(OUTPUT_PATH.parent / "create_balanced_splits.log",
                          mode="a", encoding="utf-8")
fh.setFormatter(fmt)
logger.addHandler(fh)


# ---------------------------------------------------------------------------
# Helpers (reused from preprocessing_binary.py)
# ---------------------------------------------------------------------------
def load_annotations(xlsx_path, recording_start_dt):
    """Parse Excel seizure annotation file into seconds relative to recording start.

    Reused from preprocessing_binary.py -- same logic, same column names.
    The annotation .xlsx has columns 'start_time' and 'end_time' in
    DD/MM/YYYY HH:MM:SS.fff format. We convert to seconds from the EDF
    recording start for direct comparison with segment grid positions.

    Parameters
    ----------
    xlsx_path : Path
        Path to {mouse_id}_xlsx.xlsx annotation file.
    recording_start_dt : datetime
        Timezone-naive recording start from the EDF header.

    Returns
    -------
    list of (float, float) -- (seizure_start_sec, seizure_end_sec) pairs
    """
    df = pd.read_excel(xlsx_path)
    df['start_time'] = pd.to_datetime(df['start_time'], dayfirst=True)
    df['end_time'] = pd.to_datetime(df['end_time'], dayfirst=True)
    intervals = []
    for _, row in df.iterrows():
        start_dt = row['start_time'].to_pydatetime().replace(tzinfo=None)
        end_dt = row['end_time'].to_pydatetime().replace(tzinfo=None)
        start_sec = (start_dt - recording_start_dt).total_seconds()
        end_sec = (end_dt - recording_start_dt).total_seconds()
        intervals.append((start_sec, end_sec))
    return intervals


def find_edf_for_mouse(mouse_id):
    """Search all EDF directories for {mouse_id}.edf.

    Training data was preprocessed in 5 parallel batches, each reading
    from a different EEG_TRAINING_N folder. A given mouse's .edf file
    exists in exactly one of these directories.

    Returns
    -------
    Path or None -- path to the EDF file, or None if not found
    """
    for edf_dir in EDF_DIRS:
        edf_path = edf_dir / f"{mouse_id}.edf"
        if edf_path.exists():
            return edf_path
    return None


def min_distance_to_seizure(seg_start_sec, seg_end_sec, seizure_intervals):
    """Compute minimum temporal distance (seconds) from a segment to any seizure.

    If the segment overlaps a seizure, distance is 0.0.
    Otherwise, distance is the gap between the nearest edges of the
    segment and the closest seizure interval.

    Parameters
    ----------
    seg_start_sec, seg_end_sec : float
        Segment time boundaries in seconds from recording start.
    seizure_intervals : list of (float, float)
        Each tuple is (seizure_start_sec, seizure_end_sec).

    Returns
    -------
    float -- minimum distance in seconds (0.0 if overlapping)
    """
    min_dist = float("inf")
    for sz_start, sz_end in seizure_intervals:
        # Overlap check: segment starts before seizure ends AND ends after seizure starts
        if seg_start_sec < sz_end and seg_end_sec > sz_start:
            return 0.0  # overlaps -- this segment IS ictal, distance is zero
        # Gap = minimum of (time from segment end to seizure start) and
        #       (time from seizure end to segment start)
        dist = min(abs(seg_start_sec - sz_end), abs(seg_end_sec - sz_start))
        min_dist = min(min_dist, dist)
    return min_dist


# ---------------------------------------------------------------------------
# Seizure segment verification plots
# ---------------------------------------------------------------------------
def plot_seizure_verification(splits, output_dir, n_samples=3, seed=42):
    """Randomly select and plot seizure segments from train, val, and test.

    For each partition that contains ictal segments, n_samples segments
    are randomly drawn, loaded from disk, and plotted as time-domain
    waveforms. This provides a visual sanity check that seizure segments
    contain plausible ictal EEG morphology (rhythmic discharges, amplitude
    changes) rather than artefacts, flat lines, or non-EEG data.

    Plots are saved to output_dir and NOT displayed (Agg backend).

    Parameters
    ----------
    splits : dict
        The loaded data_splits.json with keys "train", "val", "test".
    output_dir : Path
        Directory to save the verification figures.
    n_samples : int
        Number of seizure segments to plot per partition (default 3).
    seed : int
        Random seed for reproducible segment selection.
    """
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    partitions = [
        ("train", splits.get("train", [])),
        ("val",   splits.get("val", [])),
        ("test",  splits.get("test", [])),
    ]

    for part_name, records in partitions:
        # Extract only ictal segments from this partition
        ictal_recs = [r for r in records if r["label"] == 1]

        if not ictal_recs:
            logger.info("Seizure verification: %s -- no ictal segments, skipping", part_name)
            continue

        # Randomly sample n_samples (or fewer if partition is small)
        k = min(n_samples, len(ictal_recs))
        sampled = rng.sample(ictal_recs, k)

        fig, axes = plt.subplots(k, 1, figsize=(14, 3 * k))
        if k == 1:
            axes = [axes]                              # ensure iterable for single subplot

        for idx, (ax, rec) in enumerate(zip(axes, sampled)):
            filepath = rec["filepath"]
            fname = Path(filepath).stem                # e.g. m1_ictal_00001
            mouse_id = fname.split("_", 1)[0]         # e.g. m1

            try:
                segment = np.load(filepath)
                time_axis = np.arange(len(segment)) / FS  # seconds

                ax.plot(time_axis, segment, color="#5A7DC8", linewidth=0.6)
                ax.set_title(
                    "%s | %s | shape=%s | min=%.2f | max=%.2f | std=%.2f" % (
                        part_name.upper(), fname, segment.shape,
                        segment.min(), segment.max(), segment.std()),
                    fontsize=9)
                ax.set_xlabel("Time (s)", fontsize=8)
                ax.set_ylabel("Amplitude (z-scored)", fontsize=8)
                ax.tick_params(labelsize=7)

                # Shade the full segment as ictal (light red background)
                ax.axvspan(0, time_axis[-1], alpha=0.08, color="#C85A5A",
                           label="Ictal segment")
                ax.legend(fontsize=7, loc="upper right")

            except Exception as e:
                ax.set_title("%s | %s | LOAD FAILED: %s" % (
                    part_name.upper(), fname, e), fontsize=9, color="red")
                logger.warning("Could not load %s: %s", filepath, e)

        plt.tight_layout()
        fig_path = output_dir / ("seizure_verification_%s.png" % part_name)
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Seizure verification: %s -- %d segments plotted -> %s",
                    part_name, k, fig_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("create_balanced_splits.py")
    logger.info("Timestamp       : %s", datetime.now().isoformat())
    logger.info("Purpose         : Proximity-aware downsampling of non-ictal segments")
    logger.info("Margin          : +/- %.0f s", MARGIN_SEC)
    logger.info("Far sample frac : %.2f", FAR_SAMPLE_FRAC)
    logger.info("Amplitude thresh: %.0f", AMPLITUDE_THRESHOLD)
    logger.info("=" * 60)

    random.seed(SEED)                              # fix seed for reproducible distant non-ictal sampling
    np.random.seed(SEED)                           # numpy seed for any array operations

    # -- Step 1: Load original data_splits.json --------------------------------
    # This is the unmodified manifest from generate_data_splits.py containing
    # ALL ~28.7M training segments and ~4.3M validation segments.
    if not SPLITS_PATH.exists():
        logger.error("data_splits.json not found at %s", SPLITS_PATH)
        raise FileNotFoundError(str(SPLITS_PATH))

    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_records = splits["train"]
    val_records = splits["val"]
    logger.info("Loaded data_splits.json: %d train, %d val", len(train_records), len(val_records))

    # -- Step 1b: Visual verification of seizure segments ----------------------
    # Randomly plot 3 ictal segments from each partition (train, val, test)
    # before any downsampling. This confirms that ictal .npy files contain
    # plausible seizure EEG morphology and are not corrupted or mislabelled.
    plot_seizure_verification(splits, OUTPUT_PATH.parent / "seizure_verification_plots",
                              n_samples=3, seed=SEED)

    # -- Step 2: Group train records by mouse ----------------------------------
    # Mouse ID is extracted from the filename stem: m1_ictal_00001 -> m1
    # This groups all segments (ictal + nonictal) for each mouse so we can
    # process them together with that mouse's EDF header and annotations.
    mouse_records = {}  # {mouse_id: [{"filepath": ..., "label": ...}, ...]}
    for rec in train_records:
        mouse_id = Path(rec["filepath"]).stem.split("_", 1)[0]
        mouse_records.setdefault(mouse_id, []).append(rec)

    logger.info("Unique mice in train: %d", len(mouse_records))

    # -- Step 3: Remove m254 ---------------------------------------------------
    if "m254" in mouse_records:
        n_removed = len(mouse_records.pop("m254"))
        logger.info("Removed m254: %d segments (ictal-free subject)", n_removed)

    # -- Step 4: For each mouse, classify non-ictal as near/far ----------------
    # "Near" = within ±MARGIN_SEC of any seizure boundary (hard negatives).
    # "Far"  = more than MARGIN_SEC from any seizure (easy negatives).
    # All ictal segments are kept unconditionally.
    kept_ictal = []                                # all ictal segments across all mice
    kept_near = []                                 # non-ictal within ±60s of any seizure
    kept_far_pool = []                             # non-ictal >60s from any seizure (pool for sampling)
    n_extreme_removed = 0

    t0 = time.time()
    for mouse_id in sorted(mouse_records.keys()):
        records = mouse_records[mouse_id]

        # Separate ictal and nonictal records for this mouse
        ictal_recs = [r for r in records if r["label"] == 1]
        nonictal_recs = [r for r in records if r["label"] == 0]

        # Guard: skip mice with no ictal segments (shouldn't happen after m254 removal)
        if not ictal_recs:
            logger.warning("Skipping %s: no ictal segments", mouse_id)
            continue

        # Locate the EDF file (across 5 possible directories) and annotation file
        edf_path = find_edf_for_mouse(mouse_id)
        xlsx_path = ANNOTATION_DIR / f"{mouse_id}_xlsx.xlsx"

        if edf_path is None or not xlsx_path.exists():
            # Can't compute distances without EDF header or annotations.
            # Conservative fallback: treat all non-ictal as near-seizure (keep all).
            logger.warning("%s: EDF or annotation not found. Keeping all non-ictal.", mouse_id)
            kept_ictal.extend(ictal_recs)
            kept_near.extend(nonictal_recs)
            continue

        # Read EDF header only (preload=False) to get total sample count.
        # This is cheap (~10 ms) -- no signal data is loaded into RAM.
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
        n_samples = int(raw.n_times)               # total samples in the recording
        rec_start = raw.info['meas_date'].replace(tzinfo=None)  # recording start (timezone-naive)
        del raw                                     # release file handle immediately

        # Load seizure annotations as (start_sec, end_sec) pairs
        seizure_intervals = load_annotations(xlsx_path, rec_start)

        # Rebuild the full segment grid -- same formula as preprocessing_binary.py
        # This gives every valid segment start index in the recording.
        segment_starts = np.arange(0, n_samples - WIN_LEN + 1, STEP, dtype=np.int64)

        # Walk the grid and assign each non-ictal position a 1-based index.
        # This mirrors the sequential counter used during preprocessing:
        # the k-th non-ictal grid position was saved as nonictal_{k:05d}.npy.
        nonictal_grid = []  # (1-based nonictal index, start_sec, end_sec, distance)
        nonictal_idx = 0
        for seg_start_idx in segment_starts:
            seg_start_sec = seg_start_idx / FS      # convert sample index to seconds
            seg_end_sec = (seg_start_idx + WIN_LEN) / FS
            # Check if this grid position overlaps any seizure
            is_ict = False
            for sz_start, sz_end in seizure_intervals:
                if seg_start_sec < sz_end and seg_end_sec > sz_start:
                    is_ict = True
                    break
            if not is_ict:
                nonictal_idx += 1                   # increment the 1-based non-ictal counter
                dist = min_distance_to_seizure(seg_start_sec, seg_end_sec, seizure_intervals)
                nonictal_grid.append((nonictal_idx, seg_start_sec, seg_end_sec, dist))

        # Build lookup: nonictal filename index -> distance to nearest seizure
        index_to_dist = {entry[0]: entry[3] for entry in nonictal_grid}

        # Classify each non-ictal .npy file as near-seizure or far-from-seizure
        for rec in nonictal_recs:
            # Extract the 1-based index from filename: m1_nonictal_00003.npy -> 3
            fname = Path(rec["filepath"]).stem
            parts = fname.rsplit("_", 1)
            try:
                seg_index = int(parts[-1])
            except ValueError:
                # Can't parse index -- keep as near (conservative fallback)
                kept_near.append(rec)
                continue

            dist = index_to_dist.get(seg_index, None)
            if dist is None:
                # Index not found in reconstructed grid -- may indicate
                # interrupted preprocessing for this subject. Keep as near.
                kept_near.append(rec)
                continue

            if dist <= MARGIN_SEC:
                kept_near.append(rec)               # hard negative: near seizure boundary
            else:
                kept_far_pool.append(rec)            # easy negative: distant background

        # All ictal segments for this mouse are kept unconditionally
        kept_ictal.extend(ictal_recs)

        logger.info("  %s: %d ictal | %d near | %d far | %d seizures | grid=%d",
                    mouse_id, len(ictal_recs),
                    sum(1 for r in nonictal_recs
                        if index_to_dist.get(
                            int(Path(r["filepath"]).stem.rsplit("_", 1)[-1]), 0) <= MARGIN_SEC),
                    sum(1 for r in nonictal_recs
                        if index_to_dist.get(
                            int(Path(r["filepath"]).stem.rsplit("_", 1)[-1]), 999999) > MARGIN_SEC),
                    len(seizure_intervals), len(segment_starts))

    elapsed_classify = time.time() - t0
    logger.info("Classification time: %.1f s", elapsed_classify)

    # -- Step 5: Sample distant non-ictal --------------------------------------
    # Keep 5% of the near-seizure count from the distant pool. This preserves
    # diversity in the background class (quiet sleep, active waking, grooming
    # artefacts) without overwhelming the corpus with easy negatives that
    # contribute diminishing gradient signal after the first few epochs.
    n_far_to_keep = max(1, round(len(kept_near) * FAR_SAMPLE_FRAC))
    if len(kept_far_pool) <= n_far_to_keep:
        kept_far = kept_far_pool                   # pool is smaller than target -- keep all
    else:
        kept_far = random.sample(kept_far_pool, n_far_to_keep)  # random sample, seed=42

    logger.info("Near-seizure non-ictal : %d", len(kept_near))
    logger.info("Far non-ictal pool     : %d", len(kept_far_pool))
    logger.info("Far non-ictal sampled  : %d (%.1f%% of near count)",
                len(kept_far), 100 * len(kept_far) / max(len(kept_near), 1))

    # -- Step 6: Filter extreme segments from all kept segments -----------------
    # Same threshold (1000.0) used in filter_extreme_segments() in tcn_utils.py.
    # Segments with max|x| > 1000 indicate failed z-score normalisation from
    # interrupted preprocessing jobs. Also catches NaN/Inf values.
    all_kept = kept_ictal + kept_near + kept_far
    logger.info("Pre-filter total: %d", len(all_kept))

    t0_filter = time.time()
    clean = []
    n_extreme = 0
    for i, rec in enumerate(all_kept):
        try:
            x = np.load(rec["filepath"])
            # Remove if contains NaN/Inf or amplitude exceeds threshold
            if not np.isfinite(x).all() or float(np.abs(x).max()) > AMPLITUDE_THRESHOLD:
                n_extreme += 1
                continue
            clean.append(rec)
        except OSError:
            # File unreadable (corrupted or missing) -- exclude from manifest
            n_extreme += 1
            logger.warning("Could not read %s -- excluded", rec["filepath"])
            continue
        if (i + 1) % 50000 == 0:
            logger.info("  filter_extreme: scanned %d/%d | removed: %d",
                        i + 1, len(all_kept), n_extreme)

    logger.info("Extreme segments removed: %d (%.2f%%)",
                n_extreme, 100 * n_extreme / max(len(all_kept), 1))
    logger.info("Filter time: %.1f s", time.time() - t0_filter)

    # -- Step 7: Build final train list ----------------------------------------
    final_train = clean
    n_ictal_final = sum(1 for r in final_train if r["label"] == 1)
    n_nonictal_final = sum(1 for r in final_train if r["label"] == 0)

    logger.info("=" * 60)
    logger.info("FINAL TRAINING CORPUS")
    logger.info("  Ictal                    : %d", n_ictal_final)
    logger.info("  Non-ictal (total)        : %d", n_nonictal_final)
    logger.info("    Near-seizure (<=%.0fs) : %d", MARGIN_SEC, len(kept_near))
    logger.info("    Far sampled (>%.0fs)   : %d", MARGIN_SEC, len(kept_far))
    logger.info("    Extreme removed        : %d", n_extreme)
    logger.info("  Total                    : %d", len(final_train))
    logger.info("  Ratio (ictal:non-ictal)  : 1:%.2f", n_nonictal_final / max(n_ictal_final, 1))
    logger.info("  Ratio (non-ictal:ictal)  : %.2f:1", n_nonictal_final / max(n_ictal_final, 1))
    logger.info("=" * 60)

    # -- Step 8: Save output JSON ----------------------------------------------
    # The output JSON has the same schema as data_splits.json so all pipeline
    # scripts can load it without modification (same "train" and "val" keys).
    # Metadata records all parameters for reproducibility and audit.
    output = {
        "metadata": {
            "created_by": "create_balanced_splits.py",
            "timestamp": datetime.now().isoformat(),
            "source": str(SPLITS_PATH),              # original manifest this was derived from
            "margin_sec": MARGIN_SEC,                 # ±60 s near-seizure margin
            "far_sample_frac": FAR_SAMPLE_FRAC,       # 5% of near count from distant pool
            "amplitude_threshold": AMPLITUDE_THRESHOLD, # max|x| filter threshold
            "seed": SEED,                             # random seed for distant sampling
            "n_train_ictal": n_ictal_final,           # final ictal count
            "n_train_nonictal": n_nonictal_final,     # final non-ictal count
            "n_val": len(val_records),                # val count (unchanged)
            "excluded_subjects": ["m254"],            # subjects removed
            "n_extreme_removed": n_extreme,           # segments removed by amplitude filter
        },
        "train": final_train,                         # list of {"filepath": ..., "label": 0/1}
        "val": val_records,                           # val partition unchanged from original
    }

    # Preserve test partition if it exists in original splits (for future use)
    if "test" in splits:
        output["test"] = splits["test"]

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved: %s", OUTPUT_PATH)

    # -- Step 9: Summary -------------------------------------------------------
    logger.info("=" * 60)
    logger.info("DONE")
    logger.info("  Train segments : %d", len(final_train))
    logger.info("  Val segments   : %d (unchanged)", len(val_records))
    logger.info("  Output         : %s", OUTPUT_PATH)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
