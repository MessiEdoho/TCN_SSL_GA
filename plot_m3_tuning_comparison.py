"""
plot_m3_tuning_comparison.py
============================
Visualises two M3 (MultiScaleTCN) Optuna tuning runs that share the
same trial sequence (identical Optuna seed) but differ in the training
class-balance ratio.

Run A: 134,957 train segments, 48.5% ictal -> ~1:1.06 ictal:non-ictal
Run B: 220,653 train segments, 29.7% ictal -> ~1:2.37 ictal:non-ictal

The validation partition is identical between runs (4,298,154 segments,
0.27% ictal), so the comparison is a clean A/B test of the train
downsampling ratio.

Best F1 per trial was extracted from the tuning .log files:
  Run A: tune_multiscale_tcn.py log started 2026-04-26 04:31:29
         (TRAIN_T_120 corpus, V100-PCIE-16GB)
  Run B: tune_multiscale_tcn.py log started 2026-04-18 17:50:55
         (current pipeline corpus, L40S)

Output: m3_tuning_comparison.png (two-panel figure).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data: best F1 per Optuna trial, as recorded in each tuning log
# ---------------------------------------------------------------------------
# Trial-level hyperparameters were identical between runs (same Optuna seed),
# so trial index maps to the same (filters, kernel, dropout, fusion, lr, bs)
# in both. Only the training data differs.
TRIALS = list(range(10))                              # T0..T9 (T10 incomplete in both logs)

RUN_A_BEST_F1 = [0.5917, 0.5357, 0.5748, 0.5355, 0.5543,
                 0.5884, 0.5747, 0.5401, 0.5472, 0.5771]

RUN_B_BEST_F1 = [0.7820, 0.7128, 0.7565, 0.7437, 0.6933,
                 0.7382, 0.7580, 0.6367, 0.6891, 0.7149]

RUN_A_LABEL = "1:1.06 ictal:non-ictal (134,957 segments)"
RUN_B_LABEL = "1:2.37 ictal:non-ictal (220,653 segments)"

RUN_A_COLOR = "#d95f02"                               # orange
RUN_B_COLOR = "#1b6ca8"                               # blue

OUTPUT_PATH = Path(__file__).resolve().parent / "m3_tuning_comparison.png"


# ---------------------------------------------------------------------------
# Helper: cumulative running maximum (Optuna's "best so far" curve)
# ---------------------------------------------------------------------------
def running_max(values):
    """Return the element-wise running maximum of `values`."""
    return np.maximum.accumulate(np.asarray(values, dtype=float))


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def main():
    run_a = np.asarray(RUN_A_BEST_F1)
    run_b = np.asarray(RUN_B_BEST_F1)
    run_a_running = running_max(run_a)
    run_b_running = running_max(run_b)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    # -- Panel A: per-trial best F1 ------------------------------------------
    ax_left.plot(TRIALS, run_a, marker="o", linewidth=1.8, markersize=8,
                 color=RUN_A_COLOR, label=RUN_A_LABEL)
    ax_left.plot(TRIALS, run_b, marker="s", linewidth=1.8, markersize=8,
                 color=RUN_B_COLOR, label=RUN_B_LABEL)
    ax_left.set_title("Per-trial best validation F1", fontsize=12, fontweight="bold")
    ax_left.set_xlabel("Optuna trial index")
    ax_left.set_ylabel("Best validation macro F1 in trial")
    ax_left.set_xticks(TRIALS)
    ax_left.set_ylim(0.50, 0.80)
    ax_left.grid(True, linestyle="--", alpha=0.4)
    ax_left.legend(loc="lower right", fontsize=9, framealpha=0.95)

    # -- Panel B: cumulative running best (Optuna's tuning curve) ------------
    ax_right.plot(TRIALS, run_a_running, marker="o", linewidth=2.2, markersize=8,
                  color=RUN_A_COLOR, label=RUN_A_LABEL)
    ax_right.plot(TRIALS, run_b_running, marker="s", linewidth=2.2, markersize=8,
                  color=RUN_B_COLOR, label=RUN_B_LABEL)
    ax_right.set_title("Running best (cumulative max across trials)",
                       fontsize=12, fontweight="bold")
    ax_right.set_xlabel("Optuna trial index")
    ax_right.set_xticks(TRIALS)
    ax_right.grid(True, linestyle="--", alpha=0.4)
    ax_right.legend(loc="lower right", fontsize=9, framealpha=0.95)

    # -- Figure-level title --------------------------------------------------
    fig.suptitle(
        "M3 (MultiScaleTCN) tuning -- effect of training class-balance ratio\n"
        "Identical Optuna seed and trial sequence; identical full-validation "
        "partition (4,298,154 segments, 0.27% ictal)",
        fontsize=12, y=1.02)

    plt.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved {OUTPUT_PATH}")

    # -- Print compact summary stats -----------------------------------------
    print()
    print(f"  Run A (1:1.06): mean F1 = {run_a.mean():.4f} | "
          f"max = {run_a.max():.4f} (T{int(run_a.argmax())})")
    print(f"  Run B (1:2.37): mean F1 = {run_b.mean():.4f} | "
          f"max = {run_b.max():.4f} (T{int(run_b.argmax())})")
    print(f"  Mean gap (B - A): {(run_b - run_a).mean():+.4f}")
    print(f"  Run B beats Run A on {int((run_b > run_a).sum())}/{len(TRIALS)} trials")


if __name__ == "__main__":
    main()
