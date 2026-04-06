# Class Imbalance Strategy: Offline Stratified Downsampling

**File:** `tcn_utils.py` -- `filter_unpaired_subjects()` + `downsample_non_ictal()`
**Date:** 2026-04-06 (supersedes WeightedRandomSampler approach)

---

## Current Strategy

Class imbalance is addressed by offline stratified downsampling of the
non-ictal class to a 1:4 ictal:non-ictal ratio, yielding a training corpus
of 65,510 ictal and 262,040 non-ictal segments (327,550 total). Subjects
with no ictal representation (m254) are excluded prior to downsampling.
pos_weight is set to 1.0 in BCEWithLogitsLoss because the 1:4 downsampling
is the sole imbalance correction mechanism.

## Why WeightedRandomSampler Was Replaced

The training corpus contains approximately 28.7 million segments. PyTorch's
`torch.multinomial`, which `WeightedRandomSampler` calls internally, imposes
a hard ceiling of 2^24 (16,777,216) on `len(weights)`. The training corpus
exceeds this limit by a factor of approximately 1.7, raising a RuntimeError
at the first training iteration. Capping `num_samples` does not resolve the
constraint because the error is triggered by `len(weights)`, not by
`num_samples`.

Offline downsampling eliminates the PyTorch constraint entirely, produces a
deterministic and reproducible corpus, and reduces per-epoch I/O by
approximately 88x.

## Three-Step Corpus Preparation Sequence

Applied in every training and tuning script, immediately after loading
train_pairs from data_splits.json:

```python
# Step 1: remove subjects with no ictal segments
train_pairs = filter_unpaired_subjects(train_pairs, logger=logger)

# Step 2: downsample non-ictal to 1:4 ratio
train_pairs = downsample_non_ictal(train_pairs, ratio=4, seed=42)

# Step 3: pos_weight = 1.0
pos_weight = torch.tensor([1.0], dtype=torch.float32)
```

## Methods Section Text (for paper)

> Class imbalance was addressed by offline stratified downsampling of the
> non-ictal majority class to a 1:4 ictal:non-ictal ratio, applied once to
> the training partition prior to DataLoader construction. Subjects with no
> ictal representation were excluded before downsampling. The non-ictal quota
> was allocated proportionally across recordings to preserve interictal EEG
> diversity across subjects (Acharya et al., 2018; Roy et al., 2019).
> pos_weight was set to 1.0 in BCEWithLogitsLoss because the 1:4
> downsampling is the sole imbalance correction mechanism. The random seed
> (42) was fixed across all four models to ensure identical corpus
> composition for controlled ablation comparison.
