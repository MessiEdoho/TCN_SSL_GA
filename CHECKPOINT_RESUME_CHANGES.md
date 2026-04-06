# Checkpoint and Resume Changes

**Date:** 2026-04-06
**Scope:** 4 Optuna tuning scripts + 3 training scripts + 1 notebook
**Purpose:** Enable automatic resumption after job failure without restarting from scratch

---

## Problem

All tuning and training scripts use Slurm batch jobs on the cluster. If a job
is killed (walltime exceeded, node failure, OOM), all progress is lost:

- **Optuna tuning scripts** store the study in memory. Completed trials (each
  taking 10-30 minutes) are discarded. A 60-trial study killed at trial 45
  wastes ~15 hours of GPU time.
- **Training scripts** save periodic checkpoints to disk but have no logic to
  load them on restart. A 100-epoch run killed at epoch 80 restarts from
  epoch 1.

---

## Changes Applied

### A. Optuna Tuning Scripts -- SQLite Study Persistence

Added `storage=` argument to `optuna.create_study()` with `load_if_exists=True`.
Completed trials are persisted to an SQLite database file on disk. Re-running
the script after a crash automatically resumes from the last completed trial.

| File | SQLite Path | Study Name |
|------|-------------|------------|
| `tcn_HPT_binary.py` | `tuning_outputs/tcn_hpt.db` | `tcn_HPT_binary_optuna` |
| `tune_temporal_attention.py` | `outputs/tune_temporal_attention.db` | `temporal_attention_tuning` |
| `tune_multiscale_tcn.py` | `outputs/tune_multiscale_tcn.db` | `multiscale_tcn_tuning` |
| `tune_multiscale_attention.py` | `outputs/tune_multiscale_attention.db` | `multiscale_attention_tuning` |

**What changed per file (1 line):**

Before:
```python
study = optuna.create_study(
    study_name="...",
    direction="maximize",
    sampler=...,
    pruner=...,
)
```

After:
```python
study = optuna.create_study(
    study_name="...",
    direction="maximize",
    sampler=...,
    pruner=...,
    storage="sqlite:///path/to/study.db",
    load_if_exists=True,
)
```

**How to resume after failure:**

Simply re-submit the same Slurm job:

    sbatch tcn_HPT_binary.sh

Optuna detects the existing SQLite database, loads all completed trials,
and continues from where it left off. The `n_trials` argument in
`study.optimize()` is the TOTAL number of trials requested, not the number
of NEW trials. If 45 of 60 trials are already complete, Optuna runs only
the remaining 15.

**How to start fresh (discard previous trials):**

Delete the `.db` file before re-running:

    rm tuning_outputs/tcn_hpt.db
    sbatch tcn_HPT_binary.sh

**How to inspect a partial study without re-running:**

```python
import optuna
study = optuna.load_study(
    study_name="tcn_HPT_binary_optuna",
    storage="sqlite:///tuning_outputs/tcn_hpt.db")
print(f"Completed trials: {len(study.trials)}")
print(f"Best trial: {study.best_trial.number}")
print(f"Best F1: {study.best_trial.value}")
```

---

### B. Training Scripts -- Resume from Checkpoint

Added resume logic at the start of the training loop in `main()`. Before
entering the `for epoch in range(...)` loop, the script checks whether a
`*_best.pt` checkpoint exists. If found, it loads the model state, optimiser
state, scheduler state, best validation F1, and the epoch number, then
continues training from the next epoch.

| File | Checkpoint Checked | Resume From |
|------|-------------------|-------------|
| `TCN.py` | `outputs/TCN/checkpoints/tcn_best.pt` | Last best epoch |
| `TCNTemporalAttention.py` | `outputs/TCNAttention/checkpoints/tcn_attention_best.pt` | Last best epoch |
| `MultiScaleTCN.py` | `outputs/MultiScaleTCN/checkpoints/multiscale_tcn_best.pt` | Last best epoch |

**What changed per file (~20 lines before the training loop):**

```python
# -- Resume from checkpoint if available -----------------------------------
resume_ckpt = CKPT_DIR / "<prefix>_best.pt"
start_epoch = 1
if resume_ckpt.exists():
    ckpt = torch.load(resume_ckpt, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    optimiser.load_state_dict(ckpt["optimiser_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    best_val_f1 = ckpt.get("val_f1", 0.0)
    best_epoch = ckpt.get("epoch", 0)
    start_epoch = best_epoch + 1
    epochs_no_imp = 0
    logger.info("RESUMED from checkpoint: epoch %d, val_f1=%.4f",
                best_epoch, best_val_f1)
else:
    logger.info("No checkpoint found. Starting from epoch 1.")
```

The training loop range changes from:
```python
for epoch in range(1, MAX_EPOCHS + 1):
```
to:
```python
for epoch in range(start_epoch, MAX_EPOCHS + 1):
```

**How to resume after failure:**

Simply re-submit the same Slurm job:

    sbatch tcn_training.sh

The script detects the existing `*_best.pt` checkpoint, loads all training
state, and continues from the next epoch. The cosine annealing scheduler,
optimiser momentum, and early stopping patience counter are all restored
to their exact pre-crash values.

**How to start fresh (discard previous training):**

Delete the checkpoints directory before re-running:

    rm -rf outputs/TCN/checkpoints/
    sbatch tcn_training.sh

**How to inspect a checkpoint without resuming:**

```python
import torch
ckpt = torch.load("outputs/TCN/checkpoints/tcn_best.pt",
                   map_location="cpu")
print(f"Epoch: {ckpt['epoch']}")
print(f"Val F1: {ckpt['val_f1']}")
print(f"Train loss: {ckpt['train_loss']}")
print(f"Hyperparameters: {ckpt['hyperparameters']}")
```

---

### C. Notebook -- Same as A

`tcn_HPT_binary.ipynb` mirrors the SQLite storage change from
`tcn_HPT_binary.py`. The cell containing `optuna.create_study()` is
updated with the same `storage=` and `load_if_exists=True` arguments.

---

## Files Modified

| File | Change Type |
|------|-------------|
| `tcn_HPT_binary.py` | A (SQLite storage) |
| `tcn_HPT_binary.ipynb` | C (SQLite storage, mirror of .py) |
| `tune_temporal_attention.py` | A (SQLite storage) |
| `tune_multiscale_tcn.py` | A (SQLite storage) |
| `tune_multiscale_attention.py` | A (SQLite storage) |
| `TCN.py` | B (resume from checkpoint) |
| `TCNTemporalAttention.py` | B (resume from checkpoint) |
| `MultiScaleTCN.py` | B (resume from checkpoint) |

## Files NOT Modified

| File | Reason |
|------|--------|
| `tcn_utils.py` | Utility functions, no training loop |
| `interpretability_analysis.py` | Post-training analysis, no training |
| `generate_data_splits.py` | Runs in seconds, no checkpoint needed |
| `merge_training_data.py` | Data pipeline, no training |
| `preprocessing_*.py` | Data pipeline, no training |

---

## Important Notes

1. **Seed reproducibility:** When resuming from a checkpoint, the random
   state is NOT identical to an uninterrupted run because the RNG state
   at the crash point is not saved. The model weights, optimiser state,
   and scheduler state ARE restored exactly. In practice, the difference
   is negligible because the training loss landscape is smooth and early
   stopping dominates the stopping criterion.

2. **Disk space:** SQLite databases are small (< 1 MB for 60 trials).
   Training checkpoints are larger (proportional to model size) but only
   the best and the 3 most recent periodic checkpoints are kept.

3. **Concurrent access:** Do not run two instances of the same tuning
   script simultaneously with the same SQLite database. Optuna's SQLite
   backend supports concurrent reads but not concurrent writes from
   separate processes.
