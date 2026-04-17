"""
tcn_utils.py -- Shared utilities for TCN-based EEG seizure detection.

This module centralises all architecture definitions, dataset classes,
training loops, and evaluation functions used across the pipeline:
  - tcn_HPT_binary.ipynb (hyperparameter tuning)
  - TCN.py, TCNTemporalAttention.py (single-branch models)
  - MultiScaleTCN.py, MultiScaleTCNAttention.py (multi-branch models)
  - tune_temporal_attention.py, tune_multiscale_tcn.py,
    tune_multiscale_attention.py (Optuna tuning scripts)

All components are parameterised -- no global variable references.
Every class and function includes a RESEARCH REPORTING NOTE block
documenting which parameters must be reported in the methods section.
"""

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score

try:
    import optuna
    from optuna.exceptions import TrialPruned
except ImportError:
    optuna = None
    TrialPruned = Exception


# ---------------------------------------------------------------------------
# 2. set_seed
# ---------------------------------------------------------------------------
def set_seed(seed=42):
    """Set random seeds for full reproducibility across all libraries.

    Parameters
    ----------
    seed : int, default 42
        The seed value to use for all random number generators.

    Returns
    -------
    None

    Example
    -------
    >>> set_seed(42)
    """
    random.seed(seed)                                # Python built-in RNG
    np.random.seed(seed)                             # NumPy RNG
    torch.manual_seed(seed)                          # PyTorch CPU RNG
    torch.cuda.manual_seed_all(seed)                 # PyTorch GPU RNG (all devices)
    torch.backends.cudnn.deterministic = True        # force deterministic CUDA operations
    torch.backends.cudnn.benchmark = False           # disable cuDNN auto-tuner


# -- RESEARCH REPORTING NOTE: set_seed -----------------------------------------
# Methods description:
#   All random number generators (Python, NumPy, PyTorch CPU and CUDA) were
#   seeded with a fixed value to ensure full reproducibility of weight
#   initialisation, data shuffling, and dropout masks across runs.
#
# Parameters to report in paper:
#   seed : the fixed seed value (standard for reproducibility claims)
#
# Design choices to justify:
#   cudnn.deterministic=True : trades marginal speed for exact reproducibility
#   cudnn.benchmark=False : prevents non-deterministic algorithm selection
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 3. EEGSegmentDataset
# ---------------------------------------------------------------------------
class EEGSegmentDataset(Dataset):
    """Memory-efficient EEG segment dataset that loads .npy files on demand.

    Each segment was already robust z-scored during preprocessing.
    No additional normalisation is applied at load time.

    Parameters
    ----------
    file_label_pairs : list of (str or Path, int)
        Each entry is (path_to_npy_file, label) where label is 0 or 1.

    Example
    -------
    >>> ds = EEGSegmentDataset([("seg_001.npy", 1), ("seg_002.npy", 0)])
    >>> x, y = ds[0]  # x: (1, 2500), y: scalar tensor
    """

    def __init__(self, file_label_pairs):
        self.pairs = file_label_pairs              # store (path, label) pairs

    def __len__(self):
        return len(self.pairs)                     # total number of segments

    def __getitem__(self, idx):
        path, label = self.pairs[idx]              # retrieve path and label
        # Retry loop for transient Lustre/GPFS I/O errors (Errno 121).
        # Shared cluster filesystems can drop reads under heavy load.
        # 3 retries with 5s sleep typically outlasts the transient.
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                x = np.load(path).astype(np.float32)
                break
            except OSError as e:
                if attempt < max_retries:
                    import logging
                    logging.getLogger("EEGSegmentDataset").warning(
                        "Retry %d/%d for %s: %s", attempt, max_retries, path, e)
                    time.sleep(5)
                else:
                    raise OSError(
                        f"Failed to load {path} after {max_retries} retries: {e}"
                    ) from e
        # No normalisation -- data was already robust z-scored during preprocessing
        x = torch.from_numpy(x).unsqueeze(0)       # shape: (1, segment_len) -- 1 EEG channel
        y = torch.tensor(label, dtype=torch.float32)  # scalar label: 0.0 or 1.0
        return x, y


# -- RESEARCH REPORTING NOTE: EEGSegmentDataset --------------------------------
# Methods description:
#   EEG segments were loaded on demand from individual .npy files to minimise
#   memory usage. Each segment had been previously normalised using robust
#   z-score (median and MAD) during the preprocessing pipeline; no additional
#   normalisation was applied at training time.
#
#   I/O resilience: transient network filesystem errors (Lustre Errno 121)
#   were handled by retrying failed reads up to 3 times with 5-second delays.
#   If all retries failed, the error was propagated with the failing filepath
#   included in the message for diagnostic purposes. At the Optuna study level,
#   OSError exceptions were caught per-trial (catch=(OSError,)) so that a
#   single I/O failure marks one trial as failed without terminating the
#   entire hyperparameter search.
#
# Parameters to report in paper:
#   segment_shape : (1, 2500) -- single-channel, 5 s at 500 Hz
#   normalisation : robust z-score applied during preprocessing (not at load time)
#   I/O retry     : 3 attempts, 5 s delay, OSError caught at trial level
#
# Design choices to justify:
#   No load-time normalisation : avoids double-normalising data that was already
#     robust z-scored, which would undo the MAD-based artefact resistance.
#   On-demand loading : keeps memory proportional to batch size, not dataset size.
#   Retry on OSError : shared HPC filesystems (Lustre/GPFS) experience transient
#     network errors under heavy multi-job I/O load. Retrying with a short delay
#     is standard practice for long-running training jobs on shared clusters.
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 4. compute_pos_weight
# ---------------------------------------------------------------------------
def compute_pos_weight(train_pairs, device):
    """Compute pos_weight for BCEWithLogitsLoss from training label distribution.

    Parameters
    ----------
    train_pairs : list of (str or Path, int)
        Training file-label pairs. Label 1 = ictal, 0 = non-ictal.
    device : torch.device
        Target device for the returned tensor.

    Returns
    -------
    pos_weight : torch.Tensor, shape (1,)
        Ratio n_non_ictal / n_ictal, on the specified device.

    Example
    -------
    >>> pw = compute_pos_weight(train_pairs, torch.device("cuda"))
    """
    n_ictal = sum(1 for _, l in train_pairs if l == 1)      # count positive (ictal) samples
    n_non_ictal = sum(1 for _, l in train_pairs if l == 0)   # count negative (non-ictal) samples
    ratio = n_non_ictal / max(n_ictal, 1)                    # avoid division by zero
    return torch.tensor([ratio], dtype=torch.float32).to(device)  # move to device for loss computation


# -- RESEARCH REPORTING NOTE: compute_pos_weight -------------------------------
# Methods description:
#   The positive class weight for binary cross-entropy was computed as the ratio
#   of non-ictal to ictal segments in the training partition, compensating for
#   class imbalance by upweighting the loss contribution of the minority class.
#
# Parameters to report in paper:
#   pos_weight value : the actual computed ratio (e.g. 15.3)
#   n_ictal, n_non_ictal : segment counts in the training partition
#
# Design choices to justify:
#   pos_weight = 1.0 after offline stratified downsampling to 1:4 ratio.
#     The downsampling is the sole imbalance correction mechanism.
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 5. filter_unpaired_subjects
# ---------------------------------------------------------------------------
def filter_unpaired_subjects(pairs, logger=None):
    """Remove segments from subjects with no ictal representation.

    Subjects with zero ictal segments cannot participate in stratified
    downsampling because the non-ictal quota is computed as
    len(ictal) * ratio. A subject with no ictal segments contributes
    a quota of zero, meaning their non-ictal segments cannot be
    allocated proportionally. Retaining such segments introduces an
    asymmetry: the subject contributes non-ictal EEG without any
    ictal counterpart, which may cause the model to learn
    subject-specific background morphology as a discriminative feature
    rather than a generalisable interictal characteristic.

    Subject m254 was identified as having no ictal segments in the
    training partition of this dataset. All non-ictal segments from
    m254 are removed before downsampling is applied. This exclusion
    is applied identically across all four models to ensure that
    every model trains on the same eligible corpus and that
    cross-model performance differences reflect architectural
    differences rather than corpus composition differences. The
    exclusion is logged explicitly for the experiment record and
    the Methods section.

    Parameters
    ----------
    pairs : list of (str or Path, int) tuples
        File-label pairs for the training partition as loaded from
        data_splits.json. Each tuple is (filepath, label) where
        label in {0, 1}.
    logger : logging.Logger or None, optional
        If provided, logs subjects removed and segment counts
        before and after filtering.

    Returns
    -------
    list of (str or Path, int) tuples
        Pairs with all segments from ictal-free subjects removed.

    Raises
    ------
    ValueError
        If filtering removes all pairs, leaving an empty corpus.

    Notes
    -----
    Call this function as Step 1 of the mandatory three-step corpus
    preparation sequence:

        train_pairs = filter_unpaired_subjects(
            train_pairs, logger=logger)
        train_pairs = downsample_non_ictal(
            train_pairs, ratio=4, seed=42)
        pos_weight = torch.tensor(
            [1.0], dtype=torch.float32)

    The subject identifier is extracted from the filename stem as
    stem.split('_', 1)[0], matching the mouse_id convention used
    by generate_data_splits.py.
    """
    from collections import defaultdict

    subject_has_ictal = defaultdict(bool)
    for f, l in pairs:
        subject_id = Path(f).stem.split("_", 1)[0]
        if l == 1:
            subject_has_ictal[subject_id] = True

    all_subjects = {Path(f).stem.split("_", 1)[0] for f, _ in pairs}
    ictal_free = sorted(s for s in all_subjects if not subject_has_ictal[s])

    if not ictal_free:
        if logger:
            logger.info(
                "filter_unpaired_subjects: no ictal-free "
                "subjects found. All subjects retained.")
        return pairs

    if logger:
        for s in ictal_free:
            n_segs = sum(
                1 for f, _ in pairs
                if Path(f).stem.split("_", 1)[0] == s)
            logger.warning(
                "EXCLUDING subject %s: 0 ictal segments, "
                "%d non-ictal segments removed. Reason: "
                "ictal-free subjects cannot participate in "
                "stratified downsampling.", s, n_segs)

    ictal_free_set = set(ictal_free)
    filtered = [
        (f, l) for f, l in pairs
        if Path(f).stem.split("_", 1)[0] not in ictal_free_set]

    if len(filtered) == 0:
        raise ValueError(
            "filter_unpaired_subjects removed all pairs. "
            "Training corpus is empty.")

    if logger:
        logger.info(
            "filter_unpaired_subjects: removed %d segments "
            "from %d ictal-free subject(s): %s. "
            "%d segments remain.",
            len(pairs) - len(filtered),
            len(ictal_free), ictal_free, len(filtered))

    return filtered


# -- RESEARCH REPORTING NOTE: filter_unpaired_subjects -------------------------
# Methods description:
#   Subjects with no ictal segments in the training partition were excluded
#   prior to downsampling. Subject m254 contributed 0 ictal and N non-ictal
#   segments and was removed to prevent asymmetric corpus composition.
#   This exclusion was applied identically across all four models.
#
# Parameters to report:
#   Excluded subjects and their segment counts (logged at runtime).
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 6. downsample_non_ictal
# ---------------------------------------------------------------------------
def downsample_non_ictal(pairs, ratio=4, seed=42):
    """Stratified offline downsampling of the non-ictal majority class.

    Why offline downsampling instead of WeightedRandomSampler
    ---------------------------------------------------------
    The training corpus contains approximately 28.7 million segments
    (65,510 ictal and 28,686,710 non-ictal), a 437:1 class imbalance.
    WeightedRandomSampler cannot be used because torch.multinomial,
    which it calls internally, imposes a hard ceiling of 2^24
    (16,777,216) on len(weights). The training corpus exceeds this
    limit. Offline downsampling eliminates the PyTorch constraint,
    produces a deterministic and reproducible corpus, and reduces
    per-epoch I/O by approximately 88x.

    Why the 1:4 ratio
    ---------------------------------------------------------
    A 1:1 ratio discards 99.77% of non-ictal data, risking
    underfitting of the interictal class. Ratios beyond 1:10
    approach the original imbalance. A ratio of 1:4 is a principled
    compromise consistent with prior work (Acharya et al., 2018;
    Roy et al., 2019), yielding 65,510 ictal + 262,040 non-ictal
    = 327,550 total segments.

    Why pos_weight is set to 1.0
    ---------------------------------------------------------
    After downsampling to 1:4, the per-sample gradient contribution
    of ictal segments is already implicitly upweighted by 4x. Adding
    pos_weight = n_neg/n_pos would compound to ~16x effective
    upweighting, degrading precision. pos_weight = 1.0 ensures the
    1:4 ratio is the sole imbalance correction.

    Why stratification by recording
    ---------------------------------------------------------
    Naive random downsampling risks drawing disproportionately from
    recordings with large non-ictal counts. Stratification by
    recording (using the filename stem prefix as the recording ID)
    allocates the non-ictal quota proportionally, preserving
    interictal EEG diversity across subjects.

    Parameters
    ----------
    pairs : list of (str or Path, int) tuples
        File-label pairs after filter_unpaired_subjects().
    ratio : int, optional (default 4)
        Target non-ictal to ictal ratio. Non-ictal pool is
        downsampled to len(ictal) * ratio segments.
    seed : int, optional (default 42)
        Random seed for reproducibility. Must be identical
        across all four models.

    Returns
    -------
    list of (str or Path, int) tuples
        Balanced and shuffled pairs containing all ictal segments
        and a stratified subsample of non-ictal segments.

    Raises
    ------
    ValueError
        If pairs contains no ictal segments or ratio < 1.

    Notes
    -----
    Call as Step 2 of the mandatory three-step sequence:

        train_pairs = filter_unpaired_subjects(
            train_pairs, logger=logger)
        train_pairs = downsample_non_ictal(
            train_pairs, ratio=4, seed=42)
        pos_weight = torch.tensor(
            [1.0], dtype=torch.float32)

    References
    ----------
    Acharya et al. (2018). Computers in Biology and Medicine, 100.
    Roy et al. (2019). Artificial Intelligence in Medicine, 99.
    """
    import random as _random
    from collections import defaultdict

    if ratio < 1:
        raise ValueError("ratio must be >= 1, got %d" % ratio)

    ictal = [(f, l) for f, l in pairs if l == 1]
    non_ictal = [(f, l) for f, l in pairs if l == 0]

    if len(ictal) == 0:
        raise ValueError(
            "No ictal segments found in pairs. "
            "Ensure filter_unpaired_subjects() has been "
            "called before downsample_non_ictal().")

    target = len(ictal) * ratio

    # Stratify by recording: group non-ictal by subject ID from filename stem
    by_recording = defaultdict(list)
    for f, l in non_ictal:
        recording_id = Path(f).stem.split("_", 1)[0]
        by_recording[recording_id].append((f, l))

    total_non_ictal = len(non_ictal)
    sampled = []
    rng = _random.Random(seed)
    for recording_id, segments in by_recording.items():
        proportion = len(segments) / total_non_ictal
        n = max(1, round(target * proportion))
        sampled.extend(rng.sample(segments, min(n, len(segments))))

    rng.shuffle(sampled)
    sampled = sampled[:target]

    balanced = ictal + sampled
    rng.shuffle(balanced)
    return balanced


# -- RESEARCH REPORTING NOTE: downsample_non_ictal -----------------------------
# Methods description:
#   "The non-ictal majority class was downsampled offline to a 1:4
#   ictal:non-ictal ratio, stratified by recording to preserve
#   interictal EEG diversity across subjects. The post-downsampling
#   corpus contained 65,510 ictal and 262,040 non-ictal segments
#   (327,550 total). pos_weight was set to 1.0 in BCEWithLogitsLoss
#   because the 1:4 downsampling is the sole imbalance correction.
#   Seed=42 was fixed across all four models for identical corpus
#   composition."
#
# Parameters to report:
#   ratio=4, seed=42, post-downsampling counts
#   Cite: Acharya et al. (2018), Roy et al. (2019)
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 7. filter_extreme_segments
# ---------------------------------------------------------------------------
def filter_extreme_segments(pairs, threshold=1000.0, logger=None):
    """Remove segments whose max absolute amplitude exceeds *threshold*.

    This is a hard filter, not a clipping operation.  Segments with
    catastrophic values (raw unscaled EEG from preprocessing failures)
    are removed entirely because their waveform morphology is not
    recoverable by clipping.

    The function also removes segments that contain NaN or Inf values,
    regardless of the amplitude threshold.

    Parameters
    ----------
    pairs : list of (str or Path, int)
        File-label pairs (filepath, label).
    threshold : float, default 1000.0
        Maximum allowed ``max(|x|)`` per segment.  Segments exceeding
        this value are removed.  1000.0 is appropriate for z-scored EEG
        (median / MAD normalisation).
    logger : logging.Logger, optional
        If provided, logs progress every 50 000 segments, a summary
        of removed segments, and per-subject breakdown.

    Returns
    -------
    list of (str or Path, int)
        Pairs with extreme segments removed.

    RESEARCH REPORTING NOTE
    -----------------------
    Report the threshold value, number of segments removed, and
    affected subjects.  Example:

      "Six segments from five subjects were removed prior to training
       because max(|x|) exceeded 1000 (worst: 4.36e+17 in m340),
       indicating incomplete z-score normalisation from interrupted
       preprocessing jobs."
    """
    t0 = time.time()
    clean = []
    removed = []
    worst_val = 0.0
    worst_file = ""

    for i, (fp, label) in enumerate(pairs):
        x = np.load(fp)

        # Check for NaN / Inf first
        if not np.isfinite(x).all():
            mx = float(np.nanmax(np.abs(x))) if np.any(np.isfinite(x)) else float("inf")
            removed.append((fp, label, mx))
            if mx > worst_val:
                worst_val = mx
                worst_file = fp
        else:
            mx = float(np.abs(x).max())
            if mx > threshold:
                removed.append((fp, label, mx))
                if mx > worst_val:
                    worst_val = mx
                    worst_file = fp
            else:
                clean.append((fp, label))

        if logger and (i + 1) % 50000 == 0:
            logger.info("  filter_extreme_segments: scanned %d/%d | removed so far: %d",
                        i + 1, len(pairs), len(removed))

    elapsed = time.time() - t0

    if logger:
        logger.info("=" * 60)
        logger.info("filter_extreme_segments summary:")
        logger.info("  Threshold          : %.1f", threshold)
        logger.info("  Total scanned      : %d", len(pairs))
        logger.info("  Removed            : %d (%.2f%%)",
                    len(removed), 100 * len(removed) / max(len(pairs), 1))
        logger.info("  Retained           : %d", len(clean))
        logger.info("  Scan time          : %.1f s", elapsed)

        if removed:
            logger.info("  Worst value        : %.4e in %s", worst_val, worst_file)

            subject_counts = {}
            for fp, label, mx in removed:
                subj = Path(fp).stem.split("_", 1)[0]
                if subj not in subject_counts:
                    subject_counts[subj] = {"count": 0, "worst": 0.0}
                subject_counts[subj]["count"] += 1
                subject_counts[subj]["worst"] = max(subject_counts[subj]["worst"], mx)

            logger.info("  Affected subjects  : %d", len(subject_counts))
            for subj in sorted(subject_counts):
                info = subject_counts[subj]
                logger.info("    %s: %d segments removed (worst=%.2e)",
                            subj, info["count"], info["worst"])

            catastrophic = sum(1 for _, _, mx in removed if mx > 1e6)
            mild = len(removed) - catastrophic
            logger.info("  Catastrophic (>1e6): %d", catastrophic)
            logger.info("  Mild (%.0f-1e6)     : %d", threshold, mild)

        logger.info("=" * 60)

    return clean


# ---------------------------------------------------------------------------
# 8. downsample_val_stratified
# ---------------------------------------------------------------------------
def downsample_val_stratified(val_pairs, fraction=0.01, seed=42):
    """Draw a stratified subset of the validation partition for use
    during hyperparameter tuning.

    With 4.3 million validation segments, full validation per epoch
    during Optuna trials is computationally prohibitive (67K batches
    per evaluation pass). A stratified 1% subset (~43K segments)
    preserves the original ictal/non-ictal class ratio and provides
    a sufficiently accurate F1 proxy for ranking trials, consistent
    with established practice in neural architecture search and
    hyperparameter optimisation (Li et al., 2017; Falkner et al.,
    2018; Jamieson & Talwalkar, 2016). See STUDY_REPORT.txt
    "Validation subset during tuning" for the timing computation.

    The subset is drawn once before tuning begins and held fixed
    across all trials and all tuning scripts, ensuring that
    trial-to-trial F1 comparisons reflect hyperparameter differences
    rather than validation subset composition differences.

    IMPORTANT: This function is used ONLY during tuning. Final
    training scripts (TCN.py, etc.) must use the full validation
    set for early stopping and model selection.

    Parameters
    ----------
    val_pairs : list of (str or Path, int) tuples
        Full validation file-label pairs from data_splits.json.
    fraction : float, default 0.10
        Fraction of each class to retain. 0.10 = 10%.
    seed : int, default 42
        Random seed for reproducibility. Must be identical across
        all tuning scripts.

    Returns
    -------
    list of (str or Path, int) tuples
        Stratified subset of val_pairs preserving the class ratio.

    References
    ----------
    Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar,
    A. (2017). Hyperband: A novel bandit-based approach to hyperparameter
    optimization. JMLR, 18(185), 1-52.

    Falkner, S., Klein, A., & Hutter, F. (2018). BOHB: Robust and
    efficient hyperparameter optimization at scale. ICML 2018.

    Jamieson, K. & Talwalkar, A. (2016). Non-stochastic best arm
    identification and hyperparameter optimization. AISTATS 2016.
    """
    import random as _random

    rng = _random.Random(seed)

    ictal = [(f, l) for f, l in val_pairs if l == 1]
    nonictal = [(f, l) for f, l in val_pairs if l == 0]

    n_ic = max(1, round(len(ictal) * fraction))
    n_nic = max(1, round(len(nonictal) * fraction))

    ic_sample = rng.sample(ictal, min(n_ic, len(ictal)))
    nic_sample = rng.sample(nonictal, min(n_nic, len(nonictal)))

    subset = ic_sample + nic_sample
    rng.shuffle(subset)
    return subset


# -- RESEARCH REPORTING NOTE: downsample_val_stratified ------------------------
# Methods description:
#   During hyperparameter tuning, a stratified 1% subset of the
#   validation partition was used to evaluate trial performance,
#   preserving the original class ratio. The subset was drawn once
#   with a fixed seed (42) and held constant across all trials and
#   tuning scripts. Full validation was reserved for final model
#   training and evaluation.
#
# Cite: Li et al. (2017) Hyperband; Falkner et al. (2018) BOHB;
#       Jamieson & Talwalkar (2016) AISTATS.
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 8. make_loader
# ---------------------------------------------------------------------------
def make_loader(file_label_pairs, batch_size, train, device, num_workers=4):
    """Build a DataLoader for training or evaluation.

    Class imbalance is addressed by filter_unpaired_subjects() and
    downsample_non_ictal() prior to DataLoader construction, with
    pos_weight = 1.0 in BCEWithLogitsLoss. The DataLoader itself
    uses simple shuffle=True for training (no weighted sampler).

    Data loading optimisations
    --------------------------
    num_workers=4 : overlaps CPU data loading with GPU computation,
        reducing GPU idle time between batches.
    persistent_workers=True : keeps worker processes alive between
        epochs, eliminating per-epoch process spawn/kill overhead.
    prefetch_factor=4 : each worker pre-loads 4 batches ahead,
        buffering against I/O latency spikes on parallel filesystems.
    These settings follow best practices from PyTorch documentation
    and Goyal et al. (2017) for large-scale training throughput.

    Parameters
    ----------
    file_label_pairs : list of (str or Path, int)
        File-label pairs for the dataset. Label 1 = ictal, 0 = non-ictal.
        For training, these should already be downsampled via the
        three-step corpus preparation sequence.
    batch_size : int
        Number of segments per batch.
    train : bool
        If True, shuffle the dataset each epoch.
        If False, load sequentially without shuffling (for val/test).
    device : torch.device
        Used to set pin_memory (True when device is CUDA).
    num_workers : int, default 4
        Number of parallel data loading workers.

    Returns
    -------
    loader : DataLoader

    Example
    -------
    >>> loader = make_loader(pairs, 32, train=True, device=torch.device("cuda"))
    """
    dataset = EEGSegmentDataset(file_label_pairs)
    pin = (device.type == "cuda")
    use_persistent = (num_workers > 0)

    if train:
        return DataLoader(dataset, batch_size=batch_size,
                          shuffle=True, num_workers=num_workers,
                          pin_memory=pin, drop_last=False,
                          persistent_workers=use_persistent,
                          prefetch_factor=4 if num_workers > 0 else None)
    else:
        return DataLoader(dataset, batch_size=batch_size,
                          shuffle=False, num_workers=num_workers,
                          pin_memory=pin, drop_last=False,
                          persistent_workers=use_persistent,
                          prefetch_factor=4 if num_workers > 0 else None)


# -- RESEARCH REPORTING NOTE: make_loader --------------------------------------
# Methods description:
#   Class imbalance is addressed by offline stratified downsampling of the
#   non-ictal class to a 1:4 ictal:non-ictal ratio, yielding a training
#   corpus of 65,510 ictal and 262,040 non-ictal segments (327,550 total).
#   Subjects with no ictal representation (m254) are excluded prior to
#   downsampling. pos_weight is set to 1.0 in BCEWithLogitsLoss because
#   the 1:4 downsampling is the sole imbalance correction mechanism. Both
#   operations are implemented in tcn_utils.py. See downsample_non_ictal()
#   for full methodological justification.
#
# Parameters to report in paper:
#   batch_size : affects gradient noise and GPU memory usage
#   num_workers=0 : cross-platform compatibility; can increase on Linux.
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 6. CausalConvBlock
# ---------------------------------------------------------------------------
class CausalConvBlock(nn.Module):
    """Two-convolution causal residual block following Bai et al. (2018).

    Architecture per block (both convolutions share the same dilation):
        Conv1d(in_ch -> out_ch) -> LayerNorm -> GELU -> Dropout1d ->
        Conv1d(out_ch -> out_ch) -> LayerNorm -> GELU -> Dropout1d ->
                                                          + Residual

    The two-convolution design provides two non-linear transformations
    per dilation level before advancing to the next scale. Causal
    padding ensures no information leaks from future time steps.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.
    kernel_size : int
        Convolution kernel size (must be odd).
    dilation : int
        Dilation factor for this block (same for both convolutions).
    dropout : float
        Spatial dropout rate (drops entire channels).

    Example
    -------
    >>> block = CausalConvBlock(1, 64, kernel_size=7, dilation=4, dropout=0.2)
    >>> out = block(torch.randn(8, 1, 2500))  # (8, 64, 2500)
    """

    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation    # total causal padding (left side only)

        # -- Sub-layer 1: expands channels from in_ch to out_ch ----------------
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               dilation=dilation, padding=self.pad)
        self.norm1 = nn.LayerNorm(out_ch)          # normalise across channel dim per time step
        self.act1 = nn.GELU()                      # smooth activation for bio-signal features
        self.drop1 = nn.Dropout1d(dropout)         # spatial dropout: drops entire channels

        # -- Sub-layer 2: operates at full width (out_ch -> out_ch) ------------
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               dilation=dilation, padding=self.pad)
        self.norm2 = nn.LayerNorm(out_ch)          # second normalisation layer
        self.act2 = nn.GELU()                      # second activation
        self.drop2 = nn.Dropout1d(dropout)         # second spatial dropout

        # 1x1 conv for residual projection when channel counts differ
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        """Forward pass. x shape: (batch, channels, time)."""
        seq_len = x.size(2)
        res = self.residual(x)                     # project input for skip connection

        # -- Sub-layer 1 -------------------------------------------------------
        out = self.conv1(x)                        # dilated conv with causal padding
        out = out[:, :, :seq_len]                  # trim right side to enforce causality
        out = out.transpose(1, 2)                  # (B, T, C) -- LayerNorm expects channels last
        out = self.norm1(out)                      # normalise across channel dimension
        out = out.transpose(1, 2)                  # (B, C, T) -- back to conv format
        out = self.act1(out)                       # GELU activation
        out = self.drop1(out)                      # spatial dropout

        # -- Sub-layer 2 -------------------------------------------------------
        out = self.conv2(out)                      # second dilated conv at same dilation
        out = out[:, :, :seq_len]                  # trim right side to enforce causality
        out = out.transpose(1, 2)                  # (B, T, C)
        out = self.norm2(out)                      # normalise across channel dimension
        out = out.transpose(1, 2)                  # (B, C, T)
        out = self.act2(out)                       # GELU activation
        out = self.drop2(out)                      # spatial dropout

        return out + res                           # residual connection


# -- RESEARCH REPORTING NOTE: CausalConvBlock ----------------------------------
# Methods description:
#   Each residual block contained two dilated causal 1-D convolutions at
#   the same dilation factor (Bai et al., 2018), each followed by layer
#   normalisation across the channel dimension, GELU activation, and
#   spatial dropout (Dropout1d). A 1x1 pointwise convolution projected
#   the block input to match the output channel count when necessary,
#   forming the residual skip connection.
#
# Parameters to report in paper:
#   kernel_size : determines temporal resolution per sub-layer
#   dilation : determines receptive field contribution per block
#   dropout : regularisation strength (affects generalisation)
#
# Design choices to justify:
#   Two convolutions per block (Bai et al., 2018) : provides two non-linear
#     transformations per dilation level before advancing to the next scale.
#   LayerNorm instead of BatchNorm : EEG amplitude varies across subjects;
#     BatchNorm statistics are unreliable at small batch sizes (16--64).
#   Dropout1d instead of Dropout : structured EEG feature maps benefit from
#     channel-level rather than element-level regularisation.
#   GELU instead of ReLU : smoother gradient flow suits bio-signal features.
#   Causal padding with right-trim : prevents temporal information leakage.
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 9. TCN
# ---------------------------------------------------------------------------
class TCN(nn.Module):
    """Temporal Convolutional Network for binary EEG classification.

    Stacks L CausalConvBlocks (each containing two dilated causal
    convolutions, following Bai et al. 2018) with exponential dilation
    d_l = 2^l, followed by global average pooling and a linear
    classification head.

    Receptive field: RF = 2 * (2^L - 1) * (k - 1) + 1 samples.
    The factor of 2 accounts for the two convolutions per block.

    Parameters
    ----------
    num_layers : int
        Number of stacked residual blocks (L).
    num_filters : int
        Output channels per convolutional layer.
    kernel_size : int
        Kernel size (must be odd).
    dropout : float
        Spatial dropout rate per block.
    fs : int, default 500
        Sampling rate in Hz (used only for RF logging).

    Example
    -------
    >>> model = TCN(7, 64, 5, 0.2)
    >>> logits = model(torch.randn(8, 1, 2500))  # (8,)
    """

    def __init__(self, num_layers, num_filters, kernel_size, dropout, fs=500):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_ch = 1 if i == 0 else num_filters   # first block takes single-channel EEG
            dilation = 2 ** i                       # exponential dilation schedule
            layers.append(CausalConvBlock(in_ch, num_filters, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)       # sequential stack of all blocks
        self.head = nn.Linear(num_filters, 1)       # classification head: 1 logit for binary
        # RF = 2 * (2^L - 1) * (k - 1) + 1: two convolutions per block
        # each contribute (k-1)*d to the receptive field at dilation d
        self.rf = 2 * (2 ** num_layers - 1) * (kernel_size - 1) + 1
        self.num_filters = num_filters              # store for external access

    def forward(self, x):
        """Forward pass. x: (batch, 1, segment_len). Returns logits: (batch,)."""
        out = self.network(x)                       # (batch, num_filters, time)
        out = out.mean(dim=2)                       # global average pooling over time
        return self.head(out).squeeze(-1)           # (batch,) raw logits


# -- RESEARCH REPORTING NOTE: TCN ----------------------------------------------
# Methods description:
#   The TCN comprised L stacked residual blocks with exponential dilation
#   schedule d_l = 2^l. Each block contained two dilated causal 1-D
#   convolutions at the same dilation factor (Bai et al., 2018), yielding
#   a receptive field of 2*(2^L - 1)*(k - 1) + 1 samples. Global average
#   pooling collapsed the temporal dimension before a single linear head
#   produced binary classification logits.
#
# Parameters to report in paper:
#   num_layers (L) : determines depth and RF -- compute and report RF in seconds
#   kernel_size (k) : determines local temporal resolution
#   num_filters : model capacity (width)
#   total trainable parameters : standard for reproducibility
#   receptive field : 2*(2^L - 1)*(k - 1) + 1 samples and RF / fs seconds
#
# Design choices to justify:
#   Two convolutions per block (Bai et al., 2018) : doubles non-linear
#     transformations per dilation level for richer feature extraction.
#   Global average pooling : makes model length-agnostic after causal trimming;
#     acts as a spatial regulariser reducing overfitting risk.
#   Single linear head : sufficient for binary classification; avoids
#     unnecessary complexity in the decision boundary.
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 10. TemporalAttention
# ---------------------------------------------------------------------------
class TemporalAttention(nn.Module):
    """Additive temporal attention: learns a scalar importance weight per time step.

    A single linear projection maps each feature vector to a scalar score.
    Scores are normalised with softmax over the temporal axis, producing a
    probability distribution (saliency map) across time steps. The output is
    the convex combination (weighted sum) of all input feature vectors.

    Complexity: O(T * D) -- linear in sequence length, versus O(T^2 * D * H)
    for multi-head self-attention. Equivalent to Bahdanau-style additive
    attention with a single global query vector.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input feature vectors (= num_filters from TCN).

    Example
    -------
    >>> attn = TemporalAttention(64)
    >>> context, weights = attn(torch.randn(8, 2500, 64))
    >>> context.shape  # (8, 64)
    >>> weights.shape  # (8, 2500)
    """

    def __init__(self, embed_dim):
        super().__init__()
        # Single linear layer maps each feature vector to a scalar logit
        self.score = nn.Linear(embed_dim, 1, bias=True)

    def forward(self, x):
        """Compute attended context vector and attention weights.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, time, embed_dim)
            TCN output in channels-last layout.

        Returns
        -------
        context : torch.Tensor, shape (batch, embed_dim)
            Attention-weighted sum over the temporal axis.
        weights : torch.Tensor, shape (batch, time)
            Softmax attention weights -- interpretable as temporal saliency.
        """
        logits = self.score(x).squeeze(-1)          # (batch, time) scalar score per step
        weights = torch.softmax(logits, dim=-1)     # (batch, time) normalised over T
        # Weighted sum: (batch, 1, time) x (batch, time, embed_dim) -> (batch, embed_dim)
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return context, weights


# -- RESEARCH REPORTING NOTE: TemporalAttention --------------------------------
# Methods description:
#   A lightweight additive temporal attention module was inserted between the
#   TCN stack and the classification head. A single linear layer projected each
#   time-step feature vector to a scalar logit; softmax over the time axis
#   produced a probability distribution over time steps. The attended context
#   vector was the convex combination of feature vectors weighted by this
#   distribution, providing an interpretable temporal saliency map.
#
# Parameters to report in paper:
#   embed_dim : equals num_filters (the TCN channel width)
#   Learnable parameters in attention: embed_dim + 1 (score layer weight + bias)
#
# Design choices to justify:
#   O(T * D) complexity : linear in sequence length vs O(T^2 * D * H) for MHA;
#     critical for 2,500-sample EEG segments at 500 Hz.
#   Single query vector : sufficient for seizure detection -- discriminative
#     information is a localised ictal discharge, not a pairwise positional
#     relationship between feature dimensions.
#   Softmax over T : produces a proper probability distribution, enabling
#     weights to be visualised directly as a temporal saliency map.
#   No residual connection needed : temporal attention replaces global average
#     pooling entirely; the weighted sum already preserves all TCN features.
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 11. TCNWithAttention
# ---------------------------------------------------------------------------
class TCNWithAttention(nn.Module):
    """TCN backbone followed by two-layer additive temporal attention.

    The TCN extracts temporal features using two-convolution residual
    blocks (Bai et al., 2018). A two-layer additive attention mechanism
    (tanh activation + linear scorer) learns a scalar importance weight
    per time step and collapses the sequence to a fixed-length context
    vector via a weighted sum. The attention weights form an interpretable
    saliency map over the segment window.

    This attention formulation matches MultiScaleTCNWithAttention exactly,
    ensuring the M1-vs-M2 and M3-vs-M4 ablations test the same mechanism.

    Architecture:
      TCN stack -> transpose -> tanh(W_a h_t + b_a) -> v^T e_t ->
      softmax over T -> weighted sum -> dropout -> classifier

    Receptive field: RF = 2 * (2^L - 1) * (k - 1) + 1 samples.

    Parameters
    ----------
    num_layers : int
        Number of TCN residual blocks (each with two convolutions).
    num_filters : int
        Channels per TCN layer (= input dimension to attention).
    kernel_size : int
        TCN kernel size (odd).
    dropout : float
        Spatial dropout rate for TCN blocks.
    attention_dim : int, default 64
        Projection dimension of the attention scoring function.
        Tuned by Optuna in tune_temporal_attention.py.
    attention_dropout : float, default 0.0
        Dropout rate on the attention context vector.
        Tuned by Optuna in tune_temporal_attention.py.
    return_embedding : bool, default False
        If True, return the context vector instead of logits.
    fs : int, default 500
        Sampling rate for RF logging.

    Example
    -------
    >>> model = TCNWithAttention(7, 64, 5, 0.2, attention_dim=64,
    ...                          attention_dropout=0.1)
    >>> logits = model(torch.randn(8, 1, 2500))  # (8,)
    """

    def __init__(self, num_layers, num_filters, kernel_size, dropout,
                 attention_dim=64, attention_dropout=0.0,
                 return_embedding=False, fs=500):
        super().__init__()
        # -- TCN backbone (frozen during attention tuning, unfrozen for final training)
        layers = []
        for i in range(num_layers):
            in_ch = 1 if i == 0 else num_filters
            dilation = 2 ** i
            layers.append(CausalConvBlock(in_ch, num_filters, kernel_size, dilation, dropout))
        self.tcn = nn.Sequential(*layers)

        # -- Two-layer additive temporal attention (matches MultiScaleTCNWithAttention)
        # e_t = tanh(W_a h_t + b_a), score = v^T e_t, alpha = softmax({score})
        # This nonlinear scoring can learn feature interactions that a single
        # linear layer (the old TemporalAttention) cannot.
        self.attention_fc   = nn.Linear(num_filters, attention_dim)   # project to attention space
        self.attention_v    = nn.Linear(attention_dim, 1, bias=False) # scalar score per time step
        self.attention_drop = nn.Dropout(p=attention_dropout)         # regularise context vector

        self.return_embedding = return_embedding
        self.head = nn.Linear(num_filters, 1)       # binary classification head
        # RF = 2 * (2^L - 1) * (k - 1) + 1: two convolutions per block
        self.rf = 2 * (2 ** num_layers - 1) * (kernel_size - 1) + 1
        self.num_filters = num_filters

    def forward(self, x):
        """Forward pass. x: (batch, 1, segment_len)."""
        out = self.tcn(x)                           # (batch, num_filters, time)
        feat_t = out.transpose(1, 2)                # (batch, time, num_filters)

        # Two-layer additive attention scoring
        e = torch.tanh(self.attention_fc(feat_t))   # (batch, time, attention_dim)
        e = self.attention_v(e)                     # (batch, time, 1)
        alpha = torch.softmax(e, dim=1)             # (batch, time, 1) sums to 1 over T

        # Weighted sum: context vector is the attended representation
        context = (feat_t * alpha).sum(dim=1)       # (batch, num_filters)
        context = self.attention_drop(context)       # dropout on context

        if self.return_embedding:
            return context                          # (batch, num_filters) embedding
        return self.head(context).squeeze(-1)       # (batch,) logits

    def get_attention_weights(self, x):
        """Return temporal attention weights for input x.

        Mirrors get_attention_weights() in MultiScaleTCNWithAttention
        so both attention-bearing models expose the same interface for
        saliency visualisation and interpretability analysis.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, 1, segment_len)
            Must be on the same device as the model.

        Returns
        -------
        np.ndarray, shape (batch, T)
            alpha_t per time step. Non-negative, sums to 1 along T.
            Use for temporal saliency maps over raw EEG.

        Usage
        -----
        model.eval()
        with torch.no_grad():
            weights = model.get_attention_weights(x)
        # weights[0]: temporal saliency for sample 0
        """
        self.eval()
        with torch.no_grad():
            out = self.tcn(x)                       # (batch, num_filters, time)
            feat_t = out.transpose(1, 2)            # (batch, time, num_filters)
            e = torch.tanh(self.attention_fc(feat_t))
            e = self.attention_v(e)                 # (batch, time, 1)
            alpha = torch.softmax(e, dim=1)         # (batch, time, 1)
            return alpha.squeeze(-1).detach().cpu().numpy()


# -- RESEARCH REPORTING NOTE: TCNWithAttention ---------------------------------
# Methods description:
#   A two-layer additive temporal attention mechanism was appended after
#   the TCN stack, replacing global average pooling. Per-timestep energy
#   e_t = tanh(W_a h_t + b_a) was computed via a learned projection
#   W_a in R^(attention_dim x num_filters); scalar scores v^T e_t were
#   normalised with softmax to produce attention weights alpha_t. The
#   context vector c = sum_t alpha_t h_t was regularised with dropout
#   before the classification head. This formulation matches
#   MultiScaleTCNWithAttention exactly, ensuring consistent ablation
#   comparison between M1/M2 (single-branch) and M3/M4 (multi-branch).
#
# Parameters to report in paper:
#   Backbone: num_layers, kernel_size, num_filters, dropout
#   Attention: attention_dim, attention_dropout
#   Total trainable parameters (backbone + attention + head)
#   Attention parameter count (attention_fc + attention_v + attention_drop + head)
#   RF = 2 * (2^L - 1) * (k - 1) + 1 samples
#
# Design choices to justify:
#   Two-layer scoring (tanh + linear) : can learn nonlinear feature
#     interactions when deciding which time steps matter. A single linear
#     layer can only learn a fixed linear combination of features.
#   attention_dim as tunable HP : controls the capacity of the scoring
#     function independently of num_filters.
#   attention_dropout : regularises the context vector, preventing the
#     model from over-relying on a few dominant time steps.
#   Matches MultiScaleTCNWithAttention : ensures M1-vs-M2 and M3-vs-M4
#     ablations test identical attention mechanisms.
#   return_embedding mode : enables reuse as encoder in downstream pipelines.
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 11. count_parameters
# ---------------------------------------------------------------------------
def count_parameters(model):
    """Count total trainable parameters in a model.

    Parameters
    ----------
    model : nn.Module
        The model to count parameters for.

    Returns
    -------
    n_params : int
        Total number of trainable parameters.

    Example
    -------
    >>> n = count_parameters(model)
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -- RESEARCH REPORTING NOTE: count_parameters ---------------------------------
# Methods description:
#   The total number of trainable parameters was computed as the sum of
#   elements across all parameter tensors with requires_grad=True.
#
# Parameters to report in paper:
#   total_params : standard metric for model complexity comparison
#
# Design choices to justify:
#   None -- standard implementation.
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 15. train_one_epoch
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimiser, criterion, device,
                    max_grad_norm=1.0, scaler=None):
    """Train the model for one epoch.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    loader : DataLoader
        Training data loader.
    optimiser : torch.optim.Optimizer
        The optimiser.
    criterion : nn.Module
        Loss function.
    device : torch.device
        Target device for tensor transfer.
    max_grad_norm : float, default 1.0
        Maximum gradient norm for clipping.
    scaler : torch.amp.GradScaler or None, default None
        If provided, enables mixed-precision training (FP16 forward
        and backward on CUDA, FP32 weight updates).  On GPUs without
        FP16 Tensor Cores the autocast context falls back to FP32
        transparently.

    Returns
    -------
    mean_loss : float
        Average training loss over all batches.

    Example
    -------
    >>> loss = train_one_epoch(model, loader, opt, criterion, device)
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    use_amp = scaler is not None                   # True when GradScaler was created (CUDA only)

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimiser.zero_grad()

        # autocast: forward pass and loss in FP16 on CUDA (FP32 fallback on CPU
        # or GPUs without Tensor Cores). Loss is still computed in FP32 internally
        # by BCEWithLogitsLoss to maintain numerical stability.
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        if use_amp:
            # AMP backward: scale loss to prevent FP16 underflow in gradients,
            # then unscale before clipping so clip threshold stays in FP32 units.
            scaler.scale(loss).backward()          # backward in scaled FP16
            scaler.unscale_(optimiser)              # restore FP32 gradients for clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimiser)                  # skips step if NaN/Inf detected
            scaler.update()                         # adjust scale factor for next iteration
        else:
            # Standard FP32 path (CPU or when scaler is not provided)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimiser.step()

        total_loss += loss.item()                  # .item() returns FP32 scalar regardless of AMP
        n_batches += 1

    return total_loss / max(n_batches, 1)


# -- RESEARCH REPORTING NOTE: train_one_epoch ----------------------------------
# Methods description:
#   Each training epoch iterated over all batches, computing the forward pass,
#   loss, and backward pass with gradient clipping before each weight update.
#   Mixed-precision training (PyTorch AMP) was enabled on CUDA devices:
#   forward and backward passes ran in FP16 via torch.amp.autocast, with
#   dynamic loss scaling (GradScaler) to prevent gradient underflow.
#   Gradient clipping was applied after unscaling to FP32.
#
# Parameters to report in paper:
#   max_grad_norm : gradient clipping threshold (affects training stability)
#   mixed_precision : enabled on CUDA (FP16 autocast + GradScaler)
#
# Design choices to justify:
#   Gradient clipping : prevents exploding gradients in deep TCN stacks;
#     max_norm=1.0 is a standard conservative choice.
#   Batch-by-batch device transfer : avoids exhausting GPU memory.
#   AMP : ~30-40% throughput gain on V100/A100/L40S Tensor Cores with
#     negligible effect on convergence (Micikevicius et al., 2018).
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 16. evaluate
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, use_amp=False):
    """Evaluate model and return macro F1, true labels, and predictions.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.
    loader : DataLoader
        Evaluation data loader.
    device : torch.device
        Target device for tensor transfer.
    use_amp : bool, default False
        If True, runs the forward pass under ``torch.amp.autocast``
        for FP16 inference speedup on supported GPUs.

    Returns
    -------
    macro_f1 : float
    y_true : np.ndarray
    y_pred : np.ndarray

    Example
    -------
    >>> f1, y_true, y_pred = evaluate(model, val_loader, device)
    """
    model.eval()
    all_true = []
    all_pred = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        # autocast during inference: FP16 forward for speed, no GradScaler
        # needed because no backward pass occurs under @torch.no_grad().
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
        # sigmoid + threshold are computed outside autocast in FP32
        # to avoid any FP16 rounding near the 0.5 decision boundary.
        preds = (torch.sigmoid(logits) >= 0.5).long()
        all_true.append(y.cpu().numpy())           # accumulate on CPU to prevent VRAM growth
        all_pred.append(preds.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return macro_f1, y_true, y_pred


# -- RESEARCH REPORTING NOTE: evaluate -----------------------------------------
# Methods description:
#   Model evaluation computed macro-averaged F1-score using a sigmoid threshold
#   of 0.5 for binary classification. Predictions were accumulated on CPU to
#   avoid GPU memory exhaustion on large validation sets.
#
# Parameters to report in paper:
#   threshold : 0.5 (standard binary classification threshold)
#   metric : macro F1-score (treats both classes equally)
#
# Design choices to justify:
#   Macro F1 : gives equal weight to ictal and non-ictal classes regardless
#     of prevalence, making it appropriate for imbalanced seizure detection.
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 17. run_training
# ---------------------------------------------------------------------------
def run_training(model, train_loader, val_loader, lr, weight_decay,
                 max_epochs, patience, device,
                 max_grad_norm=1.0, trial=None, logger=None):
    """Full training loop with early stopping, cosine annealing, and optional Optuna pruning.

    pos_weight is fixed at 1.0 because class imbalance is handled by
    offline proximity-aware downsampling in create_balanced_splits.py,
    which produces a pre-balanced manifest (data_splits_nonictal_sampled.json)
    loaded before DataLoader construction.

    Parameters
    ----------
    model : nn.Module
        Model to train (must already be on device).
    train_loader : DataLoader
        Training data loader (downsampled corpus).
    val_loader : DataLoader
        Validation data loader.
    lr : float
        Initial learning rate for AdamW.
    weight_decay : float
        L2 regularisation coefficient.
    max_epochs : int
        Maximum number of training epochs.
    patience : int
        Early stopping patience (epochs without val F1 improvement).
    device : torch.device
        Target device.
    max_grad_norm : float, default 1.0
        Gradient clipping threshold.
    trial : optuna.trial.Trial or None, default None
        Optuna trial for pruning. If None, Optuna calls are skipped.
    logger : logging.Logger or None, default None
        If provided, logs progress every 10 epochs, plus the first
        and last epoch, to keep log files small (~12 lines per trial).

    Returns
    -------
    best_val_f1 : float
        Best validation macro F1 achieved during training.

    Example
    -------
    >>> f1 = run_training(model, train_ld, val_ld, 1e-3, 1e-4, 100, 10, device,
    ...                   trial=trial, logger=logger)
    """
    # pos_weight = 1.0: class imbalance is handled by create_balanced_splits.py
    # (proximity-aware downsampling to 1:2.37 ratio). No loss reweighting.
    pw = torch.tensor([1.0], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    optimiser = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimiser, T_max=max_epochs, eta_min=lr * 0.01)

    # Mixed precision (AMP): use FP16 forward/backward on CUDA to leverage
    # Tensor Cores (V100, A100, L40S, T4). GradScaler dynamically adjusts
    # the loss scale to prevent FP16 gradient underflow. On CPU, use_amp is
    # False and all operations remain FP32 — no behavioural change.
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    best_val_f1 = 0.0
    epochs_no_improve = 0
    # Store best weights on CPU to avoid a second GPU copy occupying VRAM
    best_state = None

    for epoch in range(max_epochs):
        t0_train = time.time()
        train_loss = train_one_epoch(model, train_loader, optimiser, criterion,
                                     device, max_grad_norm, scaler=scaler)
        train_sec = time.time() - t0_train

        t0_val = time.time()
        val_f1, _, _ = evaluate(model, val_loader, device, use_amp=use_amp)
        val_sec = time.time() - t0_val

        scheduler.step()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            # Clone weights to CPU to save VRAM (only one model copy on GPU)
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        # Log every epoch — with 20-epoch tuning budget, full visibility is needed.
        # For 100-epoch training scripts that use their own loop, this is not called.
        if logger is not None:
            ep = epoch + 1  # 1-indexed for display
            logger.info(
                "  ep %3d/%d | loss=%.4f | f1=%.4f | best=%.4f | pat=%d/%d"
                " | train %.0fs | val %.0fs",
                ep, max_epochs, train_loss, val_f1,
                best_val_f1, epochs_no_improve, patience, train_sec, val_sec)

        # Optuna integration (guarded: only runs if trial is provided)
        if trial is not None:
            trial.report(val_f1, epoch)
            if trial.should_prune():
                if best_state is not None:
                    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
                raise TrialPruned()

        if epochs_no_improve >= patience:
            break

    # Restore best weights at end of training
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return best_val_f1


# -- RESEARCH REPORTING NOTE: run_training -------------------------------------
# Methods description:
#   Models were trained using AdamW with cosine annealing learning rate
#   scheduling. Training was terminated early if validation macro F1 did not
#   improve for a specified number of consecutive epochs, and the best
#   model weights (by validation F1) were restored.
#
# Parameters to report in paper:
#   lr : initial learning rate
#   weight_decay : L2 regularisation coefficient
#   max_epochs : maximum training duration
#   patience : early stopping patience
#   max_grad_norm : gradient clipping threshold
#   optimiser : AdamW (Loshchilov and Hutter, 2019)
#   scheduler : CosineAnnealingLR with eta_min = lr * 0.01
#   pos_weight : class imbalance correction (report the value)
#
# Design choices to justify:
#   AdamW : decouples weight decay from gradient update
#   Cosine annealing : smooth LR decay without abrupt drops
#   Best-state on CPU : saves VRAM by storing only one model copy on GPU
#   Imbalance handling : offline stratified downsampling (1:4) + pos_weight=1.0
# -----------------------------------------------------------------------------


# ═══════════════════════════════════════════════════
# POST-PROCESSING AND THRESHOLD OPTIMISATION
# Called by all four model training scripts and by
# final_evaluation.py
# ═══════════════════════════════════════════════════

# ── HOW TO REPORT THESE METHODS IN YOUR PAPER ──────
#
# Overview
# ────────
# The post-processing pipeline converts raw segment-level
# binary predictions from the TCN (or TCN-Attention / SSL)
# classifier into clinical event-level alarms before
# computing the false alarm rate per hour (FAR/hr).
# Three sequential stages are applied:
#   Stage 1 — Probability smoothing
#   Stage 2 — Refractory period merging
#   Stage 3 — Minimum duration filtering
# Threshold optimisation selects the classification
# threshold on the validation set before any test set
# evaluation.
#
# Methods section template (adapt and cite as needed)
# ────────────────────────────────────────────────────
# "Raw segment-level sigmoid probabilities were
# post-processed prior to computing false alarm rate.
# A [smoothing_window]-segment moving average was applied
# to the predicted probabilities to suppress isolated
# single-segment spikes caused by transient artefacts.
# Consecutive positive predictions separated by fewer
# than [refractory_period_sec] seconds were merged into
# a single event to prevent a single seizure from being
# counted as multiple alarms. Events shorter than
# [min_event_duration_sec] seconds were discarded, as
# genuine rodent seizures typically last at least 10
# seconds [CITE]. The classification threshold was
# selected by maximising the Youden J statistic
# (J = sensitivity + specificity - 1) on the validation
# set independently for each model. Event-level false
# alarm rate per hour (FAR/hr) was computed as the
# number of false alarm events divided by the total
# non-ictal recording duration in hours."
#
# Parameters to report in paper
# ────────────────────────────────────────────────────
# | Parameter                  | Recommended value     |
# |----------------------------|-----------------------|
# | smoothing_window           | 3 segments            |
# | refractory_period_sec      | 30 seconds            |
# | min_event_duration_sec     | 10 seconds            |
# | step_sec                   | 2.5 s (= seg - overlap)|
# | threshold objective        | Youden J statistic    |
# | threshold search range     | 0.1 to 0.9, step 0.01 |
# | FAR/hr denominator         | non-ictal hours only  |
#
# Report all six parameters in Table 1 or the methods
# section. Do not report only the final FAR/hr value
# without specifying which post-processing parameters
# produced it, as results are not reproducible otherwise.
#
# Results table structure (three rows per model)
# ────────────────────────────────────────────────────
# Row 1: threshold=0.5,       post-processing=No
#        Baseline raw classifier output at standard
#        threshold. Allows comparison with prior work.
#        FAR/hr here is segment-level only.
#
# Row 2: threshold=0.5,       post-processing=Yes
#        Isolates the contribution of post-processing
#        alone. Difference in FAR/hr between Row 1 and
#        Row 2 quantifies how much of the segment-level
#        alarm burden was classifier noise.
#
# Row 3: threshold=optimal,   post-processing=Yes
#        Best clinical operating point. Threshold selected
#        on validation set via Youden J. Apply to test set
#        without further adjustment.
#
# AUROC is threshold-invariant — report it once per model
# with a footnote: "AUROC is identical across all rows
# for a given model as it is computed from the full
# probability distribution."
#
# Important warnings
# ────────────────────────────────────────────────────
# WARNING 1 — Threshold must be selected on validation
#   set only. Never select threshold by evaluating on
#   the test set. Store the optimal threshold in
#   outputs/optimal_threshold_<model>.json and load it
#   in final_evaluation.ipynb without re-optimising.
#
# WARNING 2 — Post-processing parameters must be
#   identical across all four models. Do not tune
#   post-processing parameters per model. Fix them once
#   (smoothing_window=3, refractory_period_sec=30,
#   min_event_duration_sec=10) and apply to all models.
#   Only the threshold may differ per model.
#
# WARNING 3 — Smoothing introduces boundary uncertainty.
#   A window of W segments shifts event boundaries by
#   up to floor(W/2) x step_sec seconds. For W=3 and
#   step_sec=2.5, this is 2.5 seconds maximum. Report
#   this limitation in the discussion section.
#
# WARNING 4 — Segment-level FAR/hr inflates alarm count.
#   With 50% overlap, a single artefact can generate
#   2-3 consecutive FP segments from one noise event.
#   Always report event-level FAR/hr as the primary
#   clinical metric and segment-level FAR/hr only as
#   a secondary reference for comparison with prior work.
#
# ── END OF REPORTING GUIDANCE ────────────────────────


def segment_predictions_to_events(
        y_pred,
        y_prob,
        segment_len_sec,
        step_sec,
        min_event_duration_sec=10.0,
        refractory_period_sec=30.0,
        smoothing_window=3,
        threshold=0.5
):
    """Convert raw segment-level predictions into post-processed clinical event-level alarms.

    Clinical motivation
    -------------------
    Raw segment-level binary predictions are not clinically meaningful because
    overlapping segments (50 % overlap) cause a single noise transient to trigger
    multiple consecutive false-positive segments. Post-processing collapses these
    into discrete events, merges fragmented detections from a single seizure, and
    discards implausibly short events, producing an alarm stream that is
    interpretable by clinicians and suitable for computing event-level FAR/hr.

    Parameters
    ----------
    y_pred : array-like, shape (n_segments,)
        Binary segment-level predictions (0 or 1). Used only as a reference;
        smoothing is applied to y_prob and re-thresholded.
    y_prob : array-like, shape (n_segments,)
        Sigmoid probabilities for each segment (float in [0, 1]).
    segment_len_sec : float
        Duration of each segment in seconds (e.g. 5.0).
    step_sec : float
        Step between consecutive segment starts in seconds (e.g. 2.5).
    min_event_duration_sec : float, default 10.0
        Minimum event duration in seconds; shorter events are discarded.
    refractory_period_sec : float, default 30.0
        Maximum gap in seconds between consecutive positive runs that
        should be merged into a single event.
    smoothing_window : int, default 3
        Length of the uniform moving-average kernel applied to y_prob.
    threshold : float, default 0.5
        Classification threshold applied after probability smoothing.

    Returns
    -------
    result : dict
        Keys: 'smoothed_probs', 'smoothed_preds', 'events', 'n_events',
        'total_duration_sec', 'n_segments', 'parameters'.
        See source for full schema documentation.

    Example
    -------
    >>> from tcn_utils import segment_predictions_to_events
    >>> post = segment_predictions_to_events(
    ...     y_pred=val_preds, y_prob=val_probs,
    ...     segment_len_sec=5.0, step_sec=2.5,
    ...     threshold=0.5
    ... )
    >>> print(f"Detected {post['n_events']} events")
    """
    import numpy as np

    y_prob = np.asarray(y_prob, dtype=np.float64)
    n_segments = len(y_prob)

    # --- Edge case: empty input produces sensible zero-valued defaults ---
    if n_segments == 0:
        return {
            "smoothed_probs": np.array([], dtype=np.float64),
            "smoothed_preds": np.array([], dtype=np.int64),
            "events": [],
            "n_events": 0,
            "total_duration_sec": 0.0,
            "n_segments": 0,
            "parameters": {
                "smoothing_window": int(smoothing_window),
                "refractory_period_sec": round(float(refractory_period_sec), 4),
                "min_event_duration_sec": round(float(min_event_duration_sec), 4),
                "threshold": round(float(threshold), 6),
                "step_sec": round(float(step_sec), 4),
                "segment_len_sec": round(float(segment_len_sec), 4),
            }
        }

    # ── Stage 1: Probability smoothing ────────────────────────────────────
    # A uniform moving average suppresses isolated single-segment spikes
    # caused by transient artefacts before binarisation.
    kernel = np.ones(smoothing_window) / smoothing_window
    # mode="same" preserves the original array length so that the 1:1
    # mapping between segments and probabilities is maintained.
    smoothed_probs = np.convolve(y_prob, kernel, mode="same")
    # Threshold applied after smoothing so isolated artefact spikes are
    # suppressed before binarisation, reducing spurious positive segments.
    smoothed_preds = (smoothed_probs >= threshold).astype(np.int64)

    # ── Stage 2: Map segments to time coordinates ─────────────────────────
    # segment_starts uses step_sec (not segment_len_sec) because segments
    # overlap — each new segment begins step_sec after the previous one,
    # not segment_len_sec after it.
    segment_starts = np.arange(n_segments) * step_sec
    segment_ends = segment_starts + segment_len_sec

    # ── Stage 3: Identify consecutive positive runs ───────────────────────
    raw_events = []
    in_event = False
    event_start_sec = 0.0
    event_seg_indices = []

    for i in range(n_segments):
        if smoothed_preds[i] == 1 and not in_event:
            # Transition 0→1: a new positive run begins
            in_event = True
            event_start_sec = segment_starts[i]
            event_seg_indices = [i]
        elif smoothed_preds[i] == 1 and in_event:
            event_seg_indices.append(i)
        elif smoothed_preds[i] == 0 and in_event:
            # Transition 1→0: close the current positive run
            in_event = False
            raw_events.append({
                "start_sec": event_start_sec,
                "end_sec": segment_ends[event_seg_indices[-1]],
                "seg_indices": list(event_seg_indices),
            })

    # If the recording ends while still inside a positive run, close it
    # so the final event is not silently dropped.
    if in_event:
        raw_events.append({
            "start_sec": event_start_sec,
            "end_sec": segment_ends[event_seg_indices[-1]],
            "seg_indices": list(event_seg_indices),
        })

    # ── Stage 4: Refractory period merging ────────────────────────────────
    # Merging prevents a single seizure from being counted as multiple
    # alarms when the probability briefly dips below threshold mid-seizure.
    merged_events = []
    for evt in raw_events:
        if (merged_events and
                (evt["start_sec"] - merged_events[-1]["end_sec"]) < refractory_period_sec):
            # Gap is shorter than the refractory period — extend the
            # previous event rather than starting a new one.
            merged_events[-1]["end_sec"] = evt["end_sec"]
            merged_events[-1]["seg_indices"].extend(evt["seg_indices"])
        else:
            merged_events.append({
                "start_sec": evt["start_sec"],
                "end_sec": evt["end_sec"],
                "seg_indices": list(evt["seg_indices"]),
            })

    # ── Stage 5: Minimum duration filter ──────────────────────────────────
    final_events = []
    for evt in merged_events:
        duration = evt["end_sec"] - evt["start_sec"]
        if duration < min_event_duration_sec:
            continue

        # valid_idx clipping guards against index-out-of-bounds from
        # boundary segments whose indices may exceed array length after
        # merging across chunk edges.
        valid_idx = [idx for idx in evt["seg_indices"] if 0 <= idx < n_segments]

        if len(valid_idx) > 0:
            mean_prob = round(float(np.mean(smoothed_probs[valid_idx])), 6)
            max_prob = round(float(np.max(smoothed_probs[valid_idx])), 6)
        else:
            mean_prob = 0.0
            max_prob = 0.0

        final_events.append({
            "start_sec": round(float(evt["start_sec"]), 4),
            "end_sec": round(float(evt["end_sec"]), 4),
            "duration_sec": round(float(duration), 4),
            "mean_prob": mean_prob,
            "max_prob": max_prob,
        })

    total_duration_sec = round(float(segment_ends[-1]), 4) if n_segments > 0 else 0.0

    return {
        "smoothed_probs": smoothed_probs,
        "smoothed_preds": smoothed_preds,
        "events": final_events,
        "n_events": int(len(final_events)),
        "total_duration_sec": total_duration_sec,
        "n_segments": int(n_segments),
        "parameters": {
            "smoothing_window": int(smoothing_window),
            "refractory_period_sec": round(float(refractory_period_sec), 4),
            "min_event_duration_sec": round(float(min_event_duration_sec), 4),
            "threshold": round(float(threshold), 6),
            "step_sec": round(float(step_sec), 4),
            "segment_len_sec": round(float(segment_len_sec), 4),
        }
    }


# ── REPORTING NOTE: segment_predictions_to_events ───
# Stage 1 smoothing_window: report as "a W-segment
#   moving average was applied to predicted probabilities"
# Stage 4 refractory_period_sec: report as "events
#   separated by fewer than R seconds were merged"
# Stage 5 min_event_duration_sec: report as "events
#   shorter than D seconds were discarded"
# Boundary uncertainty from smoothing: ±floor(W/2)×step_sec
#   seconds. For W=3, step=2.5s: ±2.5 seconds maximum.
#   State this limitation in the discussion section.
# ────────────────────────────────────────────────────


def compute_event_level_far(
        y_true_segments,
        post_processed,
        step_sec,
        segment_len_sec
):
    """Compute event-level false alarm rate per hour (FAR/hr) from post-processed events.

    A false alarm event is an event that has no overlap with any truly ictal
    segment. An event overlaps a segment when the segment starts before the
    event ends AND the segment ends after the event starts — any temporal
    intersection counts as overlap.

    Parameters
    ----------
    y_true_segments : array-like, shape (n_segments,)
        Ground-truth binary labels per segment (0 = non-ictal, 1 = ictal).
    post_processed : dict
        Output of segment_predictions_to_events(). Must contain keys
        'events' and 'n_segments'.
    step_sec : float
        Step between consecutive segment starts in seconds.
    segment_len_sec : float
        Duration of each segment in seconds.

    Returns
    -------
    result : dict
        Keys: 'n_true_alarms', 'n_false_alarms', 'n_total_events',
        'total_non_ictal_hr', 'far_per_hour', 'event_details'.

    Example
    -------
    >>> from tcn_utils import segment_predictions_to_events, compute_event_level_far
    >>> post = segment_predictions_to_events(preds, probs, 5.0, 2.5)
    >>> far = compute_event_level_far(y_true, post, step_sec=2.5, segment_len_sec=5.0)
    >>> print(f"FAR/hr = {far['far_per_hour']}")
    """
    import numpy as np

    events = post_processed["events"]
    n_segments = post_processed["n_segments"]
    y_true_segments = np.asarray(y_true_segments, dtype=np.int64)

    # --- Edge case: no events means zero alarms of either kind ---
    if len(events) == 0 or n_segments == 0:
        # Only non-ictal segments contribute to the denominator because
        # FAR/hr measures alarms during normal brain activity, not during
        # seizures where detections are expected.
        n_non_ictal_segs = int(np.sum(y_true_segments == 0)) if len(y_true_segments) > 0 else 0
        total_non_ictal_hr = round(float(n_non_ictal_segs * segment_len_sec) / 3600.0, 4) if n_non_ictal_segs > 0 else 0.0
        return {
            "n_true_alarms": 0,
            "n_false_alarms": 0,
            "n_total_events": 0,
            "total_non_ictal_hr": total_non_ictal_hr,
            "far_per_hour": 0.0,
            "event_details": [],
        }

    # Step 1 — Segment time coordinates
    segment_starts = np.arange(n_segments) * step_sec

    # Step 2 — Classify each event as true alarm or false alarm
    n_true_alarms = 0
    n_false_alarms = 0
    event_details = []

    for evt in events:
        # A segment overlaps an event if the segment starts before the event
        # ends AND the segment ends after the event starts — this is the
        # standard interval overlap test.
        overlapping = np.where(
            (segment_starts < evt["end_sec"]) &
            (segment_starts + segment_len_sec > evt["start_sec"])
        )[0]

        is_true_alarm = False
        if len(overlapping) > 0:
            # True alarm if ANY overlapping segment is labelled ictal
            is_true_alarm = bool(np.any(y_true_segments[overlapping] == 1))

        if is_true_alarm:
            n_true_alarms += 1
        else:
            n_false_alarms += 1

        event_details.append({
            "start_sec": round(float(evt["start_sec"]), 4),
            "end_sec": round(float(evt["end_sec"]), 4),
            "duration_sec": round(float(evt.get("duration_sec", evt["end_sec"] - evt["start_sec"])), 4),
            "is_true_alarm": is_true_alarm,
        })

    # Step 3 — Compute non-ictal recording hours
    # Only non-ictal segments contribute to the denominator because FAR/hr
    # measures alarms during normal brain activity, not total recording time.
    n_non_ictal_segs = int(np.sum(y_true_segments == 0))
    total_non_ictal_hr = round(
        float(n_non_ictal_segs * segment_len_sec) / 3600.0, 4
    )

    # Step 4 — Compute FAR/hr with zero-division guard
    far_per_hour = round(
        float(n_false_alarms / total_non_ictal_hr) if total_non_ictal_hr > 0 else 0.0,
        4
    )

    return {
        "n_true_alarms": int(n_true_alarms),
        "n_false_alarms": int(n_false_alarms),
        "n_total_events": int(n_true_alarms + n_false_alarms),
        "total_non_ictal_hr": total_non_ictal_hr,
        "far_per_hour": far_per_hour,
        "event_details": event_details,
    }


# ── REPORTING NOTE: compute_event_level_far ──────────
# FAR/hr denominator: non-ictal segments only (not total
#   recording). State explicitly in methods: "FAR/hr was
#   computed as the number of false alarm events divided
#   by the total non-ictal recording duration in hours."
# Event overlap rule: an event is a true alarm if ANY
#   overlapping segment is labelled ictal. State this:
#   "An event was classified as a true alarm if it
#   overlapped with at least one truly ictal segment."
# Distinguish from segment-level FAR/hr in results table:
#   FAR/hr (seg) = raw FP segments / non-ictal hours
#   FAR/hr (event) = false alarm events / non-ictal hours
#   Both should be reported; event-level is the primary
#   clinical metric.
# ────────────────────────────────────────────────────


def find_optimal_threshold(
        y_true,
        y_prob,
        objective="youden",
        thresholds=None
):
    """Select the optimal classification threshold on the validation set.

    The Youden J statistic (J = sensitivity + specificity - 1) is the
    recommended objective because it treats sensitivity and specificity
    symmetrically, penalising missed seizures and false alarms equally.
    F1, by contrast, weights false negatives more heavily than false
    positives via the precision term, which may not reflect clinical
    priorities where both under- and over-detection carry significant cost.

    IMPORTANT: This function must only be called on the validation set —
    never on the test set. The returned threshold should be saved to
    outputs/optimal_threshold_<model>.json and loaded without modification
    for test-set evaluation.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground-truth binary labels (0 or 1).
    y_prob : array-like, shape (n_samples,)
        Predicted sigmoid probabilities (float in [0, 1]).
    objective : str, default 'youden'
        Objective function to maximise: 'youden' or 'f1'.
    thresholds : array-like or None, default None
        Threshold values to evaluate. If None, uses
        np.linspace(0.1, 0.9, 81) (step = 0.01).

    Returns
    -------
    result : dict
        Keys: 'optimal_threshold', 'optimal_score', 'objective',
        'sensitivity_at_opt', 'specificity_at_opt', 'youden_j_at_opt',
        'threshold_curve'.

    Example
    -------
    >>> from tcn_utils import find_optimal_threshold
    >>> opt = find_optimal_threshold(val_true, val_probs, objective='youden')
    >>> print(f"Optimal threshold: {opt['optimal_threshold']}")
    >>> print(f"Youden J: {opt['youden_j_at_opt']}")
    """
    import numpy as np
    from sklearn.metrics import confusion_matrix

    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    # Step 1 — Default threshold range: 0.1–0.9 excludes extremes because
    # thresholds < 0.1 or > 0.9 produce degenerate all-positive or
    # all-negative classifiers that are not clinically useful.
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 81)

    # Step 2 — Evaluate each threshold
    scores = {}
    for t in thresholds:
        preds = (y_prob >= t).astype(np.int64)
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        sensitivity = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        if objective == "youden":
            score = sensitivity + specificity - 1.0
        elif objective == "f1":
            from sklearn.metrics import f1_score as _f1_score
            score = _f1_score(y_true, preds, average="macro", zero_division=0)
        else:
            raise ValueError(
                f"Unknown objective: {objective}. Use 'youden' or 'f1'."
            )

        scores[round(float(t), 4)] = round(float(score), 6)

    # Step 3 — Select optimal threshold
    optimal_threshold = max(scores, key=scores.get)
    optimal_score = scores[optimal_threshold]

    # Step 4 — Recompute metrics at optimal threshold for logging
    opt_preds = (y_prob >= optimal_threshold).astype(np.int64)
    cm_opt = confusion_matrix(y_true, opt_preds, labels=[0, 1])
    tn_opt, fp_opt, fn_opt, tp_opt = cm_opt.ravel()

    sensitivity_at_opt = round(
        float(tp_opt / (tp_opt + fn_opt)) if (tp_opt + fn_opt) > 0 else 0.0, 6
    )
    specificity_at_opt = round(
        float(tn_opt / (tn_opt + fp_opt)) if (tn_opt + fp_opt) > 0 else 0.0, 6
    )
    youden_j_at_opt = round(float(sensitivity_at_opt + specificity_at_opt - 1.0), 6)

    return {
        "optimal_threshold": round(float(optimal_threshold), 6),
        "optimal_score": round(float(optimal_score), 6),
        "objective": str(objective),
        "sensitivity_at_opt": sensitivity_at_opt,
        "specificity_at_opt": specificity_at_opt,
        "youden_j_at_opt": youden_j_at_opt,
        "threshold_curve": scores,
    }


# ── REPORTING NOTE: find_optimal_threshold ───────────
# Objective function: Youden J statistic is recommended.
#   Report as: "The classification threshold was selected
#   by maximising the Youden J statistic (J = sensitivity
#   + specificity - 1) on the validation set."
# Threshold search range: report as "Thresholds from 0.1
#   to 0.9 in steps of 0.01 were evaluated."
# Validation-only rule: report as "Threshold selection
#   was performed exclusively on the validation set and
#   applied without modification to the test set."
# Per-model thresholds: report as "An independent
#   threshold was selected for each model architecture."
# Save optimal threshold to JSON after calling:
#   outputs/optimal_threshold_<model_name>.json
#   This file is loaded by final_evaluation.ipynb.
#   Include threshold value in Table 1 of the paper.
# ────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════
# MULTI-SCALE TCN ARCHITECTURES
# MultiScaleTCN        -- parallel multi-branch TCN
# MultiScaleTCNWithAttention -- backbone + temporal
#                              attention pooling
# CausalConvBlock is reused from this module.
# Tuning scripts:
#   tune_multiscale_tcn.py
#   tune_multiscale_attention.py
# ═══════════════════════════════════════════════════


# ---------------------------------------------------------------------------
# 20. MultiScaleTCN
# ---------------------------------------------------------------------------
class MultiScaleTCN(nn.Module):
    """Multi-Scale Temporal Convolutional Network for binary EEG seizure detection.

    Three parallel branches of CausalConvBlocks, each using a distinct
    dilation schedule, process the input simultaneously. Branch outputs
    are fused and globally average-pooled before a linear classification head.

    Architecture
    ------------
    Input: (batch, 1, SEGMENT_LEN)
        |
        +-- Branch 1 dilations [1,  2,   4]    -- fine scale         (spike morphology)
        +-- Branch 2 dilations [8,  16,  32]   -- intermediate scale (rhythmic bursts)
        +-- Branch 3 dilations [32, 64, 128]   -- coarse scale       (seizure evolution)
        |  Each branch: CausalConvBlock x len(dilations)
        |  Each block: same two-conv structure as in TCN
        |  Output per branch: (batch, num_filters, T)
        |
        v
    Fusion -> (batch, num_filters, T)
      "concat": cat on channel dim -> 1x1 Conv1d projection
      "average": element-wise mean of branch outputs
        |
        v
    Global average pool -> (batch, num_filters)
        |
        v
    Linear head -> scalar logit (batch,)

    Parameters
    ----------
    num_filters : int
        Output channels per branch and in the fused representation.
    kernel_size : int
        Kernel width for all CausalConvBlocks. Must be odd.
    dropout : float
        Spatial dropout rate inside each CausalConvBlock.
    branch1_dilations : list of int
        Dilation schedule for Branch 1. Default [1, 2, 4].
    branch2_dilations : list of int
        Dilation schedule for Branch 2. Default [8, 16, 32].
    branch3_dilations : list of int
        Dilation schedule for Branch 3. Default [32, 64, 128].
    fusion : str
        Branch fusion strategy: "concat" (default) or "average".
    return_embedding : bool
        If True, return the globally pooled feature vector
        (batch, num_filters) instead of the scalar logit. Default: False.

    Receptive field per branch
    --------------------------
    RF_branch = 1 + 2 * sum((kernel_size - 1) * d for d in dilations)
    Factor of 2 accounts for the two convolutions per CausalConvBlock.
    """

    def __init__(self,
                 num_filters,
                 kernel_size,
                 dropout,
                 branch1_dilations=None,
                 branch2_dilations=None,
                 branch3_dilations=None,
                 fusion="concat",
                 return_embedding=False):
        super().__init__()

        # Mutable default arguments handled here to avoid shared-list pitfall
        if branch1_dilations is None:
            branch1_dilations = [1, 2, 4]
        if branch2_dilations is None:
            branch2_dilations = [8, 16, 32]
        if branch3_dilations is None:
            branch3_dilations = [32, 64, 128]

        if fusion not in ("concat", "average"):
            raise ValueError(
                "fusion must be 'concat' or 'average', got '%s'" % fusion)

        self.fusion           = fusion
        self.return_embedding = return_embedding
        self.num_filters      = num_filters

        def build_branch(dilations):
            """Build one sequential branch of CausalConvBlocks.

            Uses the EXACT parameter names of CausalConvBlock.__init__:
                in_ch, out_ch, kernel_size, dilation, dropout
            The two-conv-per-block structure is inherited -- not reimplemented.
            """
            blocks = []
            in_ch = 1                                  # single-channel EEG input
            for d in dilations:
                blocks.append(
                    CausalConvBlock(
                        in_ch=in_ch,
                        out_ch=num_filters,
                        kernel_size=kernel_size,
                        dilation=d,
                        dropout=dropout))
                in_ch = num_filters                    # subsequent blocks use num_filters input
            return nn.Sequential(*blocks)

        self.branch1 = build_branch(branch1_dilations)
        self.branch2 = build_branch(branch2_dilations)
        self.branch3 = build_branch(branch3_dilations)

        # Fusion projection for concat mode
        # Projects 3*num_filters -> num_filters so pooling and head always
        # see a fixed channel dimension
        if self.fusion == "concat":
            self.fusion_conv = nn.Conv1d(
                3 * num_filters, num_filters, kernel_size=1, bias=False)
        else:
            self.fusion_conv = None                    # register None for consistent state_dict

        self.classifier = nn.Linear(num_filters, 1)    # binary classification head

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, 1, SEGMENT_LEN)

        Returns
        -------
        torch.Tensor
            return_embedding=False: shape (batch,) logit
            return_embedding=True : shape (batch, num_filters)
        """
        out1 = self.branch1(x)                         # (batch, num_filters, T)
        out2 = self.branch2(x)                         # (batch, num_filters, T)
        out3 = self.branch3(x)                         # (batch, num_filters, T)

        if self.fusion == "concat":
            # Concat then project to preserve branch-specific features
            fused = torch.cat([out1, out2, out3], dim=1)  # (batch, 3*num_filters, T)
            fused = self.fusion_conv(fused)            # (batch, num_filters, T)
        else:
            # Equal-weight average of branch outputs
            fused = (out1 + out2 + out3) / 3.0         # (batch, num_filters, T)

        # Global average pooling collapses T to a scalar per filter
        pooled = fused.mean(dim=-1)                    # (batch, num_filters)

        if self.return_embedding:
            return pooled

        return self.classifier(pooled).squeeze(-1)     # (batch,) logit


# -- RESEARCH REPORTING NOTE: MultiScaleTCN ------------------------------------
# Methods description:
#   "A Multi-Scale TCN was constructed with three parallel branches of
#   CausalConvBlocks using dilation schedules [1,2,4], [8,16,32],
#   [32,64,128] to capture ictal activity at fine, intermediate, and
#   coarse temporal scales simultaneously. Branch outputs were fused
#   by [concat+1x1 projection / averaging] before global average
#   pooling and linear classification. All hyperparameters were tuned
#   independently from scratch using Optuna TPE."
#
# Parameters to report:
#   num_filters, kernel_size, dropout -- tuned
#   fusion strategy -- tuned
#   branch dilation schedules -- fixed by design
#   RF per branch in samples and seconds at 500 Hz
#   Total trainable parameters
#
# Justification for independent tuning:
#   The optimal num_filters for three parallel branches differs from that
#   for a single branch because capacity is distributed across branches.
#   Independent tuning finds the true optimum for this architecture.
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 21. MultiScaleTCNWithAttention
# ---------------------------------------------------------------------------
class MultiScaleTCNWithAttention(nn.Module):
    """Multi-Scale TCN with Temporal Attention for binary EEG seizure detection.

    Mirrors TCNWithAttention structurally. The MultiScaleTCN backbone is
    stored as self.backbone and frozen during attention tuning. Temporal
    attention replaces global average pooling, learning which time steps
    in the fused multi-scale feature map are most relevant to seizure
    classification.

    Architecture
    ------------
    Input: (batch, 1, SEGMENT_LEN)
        |
        v
    MultiScaleTCN backbone (frozen during tuning)
    Three parallel branches fused to (batch, num_filters, T)
        |
        v
    Temporal Attention
      e_t  = tanh(W_a h_t + b_a)  h_t in R^num_filters
      alpha_t = softmax({e_t})     scalar weight per step
      c    = sum_t alpha_t * h_t   (batch, num_filters)
        |
        v
    Attention dropout -> (batch, num_filters)
        |
        v
    Linear head -> scalar logit (batch,)

    Parameters
    ----------
    num_filters : int
        Branch channel width. Transferred from best_multiscale_params.json.
    kernel_size : int
        CausalConvBlock kernel width. Transferred from best_multiscale_params.json.
    dropout : float
        Spatial dropout in CausalConvBlocks. Transferred from best_multiscale_params.json.
    fusion : str
        "concat" or "average". Transferred from best_multiscale_params.json.
    attention_dim : int
        Projection dimension W_a. Tuned by Optuna.
    attention_dropout : float
        Dropout on context vector c. Tuned by Optuna.
    branch1_dilations : list. Default [1, 2, 4].
    branch2_dilations : list. Default [8, 16, 32].
    branch3_dilations : list. Default [32, 64, 128].

    Notes
    -----
    The backbone is stored as self.backbone. Its classification head is
    never called. forward() accesses self.backbone.branch1/2/3 and
    self.backbone.fusion_conv directly to obtain the pre-pooled feature map.
    Call get_attention_weights(x) to retrieve alpha_t weights as a numpy
    array for saliency visualisation.
    """

    def __init__(self,
                 num_filters,
                 kernel_size,
                 dropout,
                 fusion,
                 attention_dim,
                 attention_dropout,
                 branch1_dilations=None,
                 branch2_dilations=None,
                 branch3_dilations=None):
        super().__init__()

        # Build MultiScaleTCN backbone
        # return_embedding=False because forward() bypasses backbone.forward()
        # entirely -- internal branch attributes are accessed directly
        self.backbone = MultiScaleTCN(
            num_filters=num_filters,
            kernel_size=kernel_size,
            dropout=dropout,
            branch1_dilations=branch1_dilations,
            branch2_dilations=branch2_dilations,
            branch3_dilations=branch3_dilations,
            fusion=fusion,
            return_embedding=False)

        self.num_filters = num_filters

        # Temporal attention layers
        # Attribute names are distinct from TCNWithAttention (which uses
        # TemporalAttention module) -- this uses a two-layer energy function
        # with attention_dim for richer multi-scale feature scoring
        self.attention_fc   = nn.Linear(num_filters, attention_dim)
        self.attention_v    = nn.Linear(attention_dim, 1, bias=False)
        self.attention_drop = nn.Dropout(p=attention_dropout)
        self.classifier     = nn.Linear(num_filters, 1)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, 1, SEGMENT_LEN)

        Returns
        -------
        torch.Tensor, shape (batch,) -- scalar logit.
        """
        # Access backbone internals directly to obtain the pre-pooled
        # fused map (batch, num_filters, T)
        out1 = self.backbone.branch1(x)
        out2 = self.backbone.branch2(x)
        out3 = self.backbone.branch3(x)

        if self.backbone.fusion == "concat":
            fused = torch.cat([out1, out2, out3], dim=1)
            fused = self.backbone.fusion_conv(fused)
        else:
            fused = (out1 + out2 + out3) / 3.0
        # fused: (batch, num_filters, T)

        # Transpose so attention operates over time steps
        feat_t = fused.transpose(1, 2)                 # (batch, T, num_filters)

        # Compute per-timestep attention energy
        # tanh bounds values to (-1,1) for stable softmax over long sequences
        e = torch.tanh(self.attention_fc(feat_t))      # (batch, T, attention_dim)
        e = self.attention_v(e)                        # (batch, T, 1)

        # Normalise across T: weights sum to 1 per sample
        alpha = torch.softmax(e, dim=1)                # (batch, T, 1)

        # Weighted sum: each h_t scaled by its attention weight
        context = (feat_t * alpha).sum(dim=1)          # (batch, num_filters)

        context = self.attention_drop(context)
        return self.classifier(context).squeeze(-1)    # (batch,) logit

    def get_attention_weights(self, x):
        """Return temporal attention weights for input x.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, 1, SEGMENT_LEN)
            Must be on the same device as the model.

        Returns
        -------
        np.ndarray, shape (batch, T)
            alpha_t per time step. Non-negative, sums to 1 along T.
            Use for saliency maps.

        Usage
        -----
        model.eval()
        with torch.no_grad():
            weights = model.get_attention_weights(x)
        # weights[0]: temporal saliency for sample 0
        """
        self.eval()
        with torch.no_grad():
            out1 = self.backbone.branch1(x)
            out2 = self.backbone.branch2(x)
            out3 = self.backbone.branch3(x)
            if self.backbone.fusion == "concat":
                fused = torch.cat([out1, out2, out3], dim=1)
                fused = self.backbone.fusion_conv(fused)
            else:
                fused = (out1 + out2 + out3) / 3.0
            feat_t = fused.transpose(1, 2)
            e = torch.tanh(self.attention_fc(feat_t))
            e = self.attention_v(e)
            alpha = torch.softmax(e, dim=1)
            return alpha.squeeze(-1).detach().cpu().numpy()


# -- RESEARCH REPORTING NOTE: MultiScaleTCNWithAttention -----------------------
# Methods description:
#   "MultiScaleTCNWithAttention was constructed by adding temporal attention
#   to the frozen MultiScaleTCN backbone, mirroring the design of
#   TCNWithAttention. Backbone weights were transferred from the MultiScaleTCN
#   tuning study and frozen. Only temporal attention parameters and the
#   classification head received gradient updates. Energy e_t = tanh(W_a h_t
#   + b_a) was computed per time step; weights alpha_t = softmax({e_t}) formed
#   context c = sum_t alpha_t h_t passed to the head."
#
# Parameters to report:
#   Backbone (fixed -- from best_multiscale_params.json):
#     num_filters, kernel_size, dropout, fusion
#     branch dilation schedules [1,2,4], [8,16,32], [32,64,128]
#   Attention (tuned -- from best_multiscale_attn_params.json):
#     attention_dim, attention_dropout,
#     learning_rate, weight_decay, batch_size
#   Frozen: all self.backbone parameters
#   Trainable: attention_fc, attention_v, attention_drop, classifier
#   Trainable parameter count (log from trial 0)
#
# Ablation framing:
#   M3 MultiScaleTCN vs M4 MultiScaleTCNWithAttention isolates temporal
#   attention at the multi-scale level, mirroring M1 TCN vs M2
#   TCNWithAttention at the single-branch level.
#
# Attention interpretability:
#   Call get_attention_weights() to extract alpha_t and average over ictal
#   vs non-ictal segments to verify preferential attention to seizure-
#   relevant time steps.
# -----------------------------------------------------------------------------


# ═══════════════════════════════════════════════════

# -- USAGE IN TRAINING NOTEBOOKS ---------------------------------------------------
#
# Load tuned params:
#   with open("outputs/best_multiscale_params.json") as f:
#       ms = json.load(f)
#   hp = ms["hyperparameters"]
#
#   with open("outputs/best_multiscale_attn_params.json") as f:
#       ma = json.load(f)
#   ah = ma["hyperparameters"]
#
# MultiScaleTCN:
#   model = MultiScaleTCN(
#       num_filters = hp["num_filters"],
#       kernel_size = hp["kernel_size"],
#       dropout     = hp["dropout"],
#       fusion      = hp["fusion"]
#   ).to(DEVICE)
#
# MultiScaleTCNWithAttention:
#   model = MultiScaleTCNWithAttention(
#       num_filters       = hp["num_filters"],
#       kernel_size       = hp["kernel_size"],
#       dropout           = hp["dropout"],
#       fusion            = hp["fusion"],
#       attention_dim     = ah["attention_dim"],
#       attention_dropout = ah["attention_dropout"]
#   ).to(DEVICE)
#
# Both return scalar logits for BCEWithLogitsLoss.
# count_parameters() works on both.
# -----------------------------------------------------------------------------
