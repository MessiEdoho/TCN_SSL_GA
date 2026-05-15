"""
plot_training_curves_overfitting.py
===================================
Local diagnostic. Reads the per-epoch training metrics CSVs for M3
(MultiScaleTCN) and M4 (MultiScaleTCNAttention) and produces a
side-by-side figure showing train loss and val F1 over epochs.

Visual overfitting cue: if train loss keeps falling after the best-epoch
marker while val F1 plateaus or declines, the model is overfitting from
that epoch onward. If both curves stabilise together, no overfitting.

Inputs  : per-epoch CSVs from each model's training run
Output  : MultiScaleTCN/event_metrics_recovery/figures/training_curves_M3_M4_overfitting.png
          + a short overfitting summary printed to stdout

Run     : python plot_training_curves_overfitting.py
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LOCAL_ROOT = Path(r"C:\Users\messi\OneDrive\Desktop\Desktop\TCN_UNIQURE_PROJECT")
OUT_PATH   = (LOCAL_ROOT / "MultiScaleTCN" / "event_metrics_recovery" / "figures"
              / "training_curves_M3_M4_overfitting.png")

CSVS = {
    "M3 (MultiScaleTCN)":          LOCAL_ROOT / "MultiScaleTCN" / "multiscale_tcn_epoch_metrics.csv",
    "M4 (MultiScaleTCNAttention)": LOCAL_ROOT / "MultiScaleTCNAttention" / "event_metrics" / "ms_attn_epoch_metrics.csv",
}


def load_csv(path):
    epochs, train_loss, val_f1, lr = [], [], [], []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_f1.append(float(row["val_f1"]))
            lr.append(float(row["lr"]))
    return (np.asarray(epochs), np.asarray(train_loss),
            np.asarray(val_f1), np.asarray(lr))


def ema_smooth(x, alpha=0.6):
    out = np.zeros_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * out[i-1] + (1 - alpha) * x[i]
    return out


def diagnose(label, epochs, loss, f1):
    best_idx     = int(np.argmax(f1))
    best_epoch   = int(epochs[best_idx])
    best_f1      = float(f1[best_idx])
    loss_at_best = float(loss[best_idx])
    loss_final   = float(loss[-1])
    f1_final     = float(f1[-1])
    f1_drop      = best_f1 - f1_final
    loss_drop    = loss_at_best - loss_final

    print(f"\n--- {label} ---")
    print(f"  Best epoch              : {best_epoch}/{int(epochs[-1])}")
    print(f"  Val F1 at best epoch    : {best_f1:.4f}")
    print(f"  Val F1 at final epoch   : {f1_final:.4f}  (delta = {-f1_drop:+.4f})")
    print(f"  Train loss at best epoch: {loss_at_best:.4f}")
    print(f"  Train loss at final     : {loss_final:.4f}  (delta = {-loss_drop:+.4f})")
    if best_epoch < int(epochs[-1]):
        if loss_drop > 0 and f1_drop > 0.01:
            print(f"  -> Overfitting signal: train loss kept dropping ({-loss_drop:+.4f}) "
                  f"while val F1 declined ({-f1_drop:+.4f}) after best epoch.")
        elif loss_drop > 0 and f1_drop <= 0.01:
            print(f"  -> Mild signal: train loss kept dropping ({-loss_drop:+.4f}) "
                  f"but val F1 was stable post-best ({-f1_drop:+.4f}). Borderline.")
        else:
            print(f"  -> No overfitting signal: train loss did not drop further "
                  f"after best epoch.")
    return best_epoch, best_f1


def plot_one(ax_loss, label, csv_path):
    epochs, loss, f1, _ = load_csv(csv_path)
    best_epoch, best_f1 = diagnose(label, epochs, loss, f1)
    loss_smooth = ema_smooth(loss)
    f1_smooth   = ema_smooth(f1)

    ax_f1 = ax_loss.twinx()

    ax_loss.plot(epochs, loss, color="#5A7DC8", alpha=0.25,
                 linewidth=1.0, label="Train loss (raw)")
    ax_loss.plot(epochs, loss_smooth, color="#5A7DC8", linewidth=1.8,
                 label="Train loss (EMA alpha=0.6)")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Train loss (BCEWithLogitsLoss)", color="#5A7DC8")
    ax_loss.tick_params(axis="y", labelcolor="#5A7DC8")

    ax_f1.plot(epochs, f1, color="#C85A5A", alpha=0.25,
               linewidth=1.0, label="Val F1 (raw)")
    ax_f1.plot(epochs, f1_smooth, color="#C85A5A", linewidth=1.8,
               label="Val F1 (EMA alpha=0.6)")
    ax_f1.set_ylabel("Val macro F1", color="#C85A5A")
    ax_f1.tick_params(axis="y", labelcolor="#C85A5A")
    ax_f1.set_ylim(0, 1.0)

    ax_loss.axvline(best_epoch, linestyle="--", color="gray", alpha=0.7,
                    label=f"Best epoch {best_epoch}")
    ax_f1.annotate(f"Best F1 = {best_f1:.4f}",
                   xy=(best_epoch, best_f1),
                   xytext=(8, -16), textcoords="offset points",
                   fontsize=9, color="gray")

    ax_loss.set_title(label, fontsize=11)
    h1, l1 = ax_loss.get_legend_handles_labels()
    h2, l2 = ax_f1.get_legend_handles_labels()
    ax_loss.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8)


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_loss, (label, csv_path) in zip(axes, CSVS.items()):
        if not csv_path.exists():
            print(f"WARNING: missing CSV {csv_path}")
            continue
        plot_one(ax_loss, label, csv_path)
    fig.suptitle("Training curves vs validation F1 -- overfitting diagnosis",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
