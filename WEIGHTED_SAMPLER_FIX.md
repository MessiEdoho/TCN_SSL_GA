# WeightedRandomSampler Fix: torch.multinomial Category Limit

**File:** `tcn_utils.py` -- `make_loader()`
**Date:** 2026-04-05

---

## Fix

### Bug

`RuntimeError: number of categories cannot exceed 2^24` raised during
hyperparameter tuning (`tcn_HPT_binary.py`) when the training dataset
exceeded 16,777,216 segments.

### Root cause

`WeightedRandomSampler` internally calls `torch.multinomial()` to draw
sample indices. `torch.multinomial` uses 32-bit integer indexing for its
internal alias table, which limits the number of categories (i.e., the
`num_samples` argument) to 2^24 = 16,777,216. The original code passed
`num_samples=len(labels)` directly, which exceeded this limit when all
five training roots (`TRAIN_DATA` through `TRAIN_DATA_5`) were combined
into a single partition via `generate_data_splits.py`.

### Resolution

Cap `num_samples` at `min(len(labels), 2^24 - 1)`:

```python
MAX_SAMPLES = 2**24 - 1  # 16,777,215 -- PyTorch multinomial ceiling
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=min(len(labels), MAX_SAMPLES),
    replacement=True
)
```

For datasets below 16,777,215 samples, `min()` returns `len(labels)` and
behaviour is identical to the original code. For larger datasets, the
sampler draws 16,777,215 samples per epoch instead of `len(labels)`. With
replacement=True, every sample still has a nonzero probability of being
drawn; only the expected number of draws per epoch is reduced.

### Impact

- **Correctness:** The weighted sampling distribution is unchanged. Each
  sample's probability of being drawn remains proportional to its
  inverse-frequency weight. The only difference is the total number of
  draws per epoch, which is capped at 16,777,215 instead of N.

- **Training behaviour:** For a dataset of ~20 million segments, each
  epoch draws ~16.8M samples instead of ~20M. This means ~84% of the
  dataset is seen per epoch in expectation. Over 100 epochs with
  replacement, every sample is drawn multiple times. The effect on
  convergence is negligible.

- **Affected files:** `tcn_utils.py` (`make_loader`). All scripts and
  notebooks that call `make_loader(train=True)` inherit the fix
  automatically: `tcn_HPT_binary.py`, `TCN.py`, `TCNTemporalAttention.py`,
  `MultiScaleTCN.py`, `tune_temporal_attention.py`, `tune_multiscale_tcn.py`,
  `tune_multiscale_attention.py`.

---

## Methods section text (for paper)

> Training batches were constructed using PyTorch's `WeightedRandomSampler`
> with inverse-frequency class weights w_c = 1/N_c, where N_c is the number
> of segments belonging to class c. At each epoch, min(N, 2^24 - 1) samples
> were drawn with replacement from this weighted distribution, producing
> approximately class-balanced batches without discarding any majority-class
> samples. The 2^24 - 1 cap accommodates a PyTorch internal indexing
> constraint in `torch.multinomial`; for datasets below this threshold, it
> has no effect. This sampling strategy was combined with pos_weight =
> N_non-ictal / N_ictal in `BCEWithLogitsLoss` for dual class imbalance
> correction: the sampler balances the composition of each batch, while
> pos_weight adjusts the gradient contribution of minority-class samples.

---

## References

- PyTorch issue #2576: `torch.multinomial` 32-bit alias table limitation.
- PyTorch `WeightedRandomSampler` documentation: `num_samples` controls
  the number of draws per epoch, not the dataset size.
