"""
recolor_m1_signed_contribution_big.py
=====================================
Re-render the "M1 -- Signed branch contribution by prediction category
(test)" branch-Shapley diverging bar chart (Figure C) with larger, clearer
fonts sized for an Elsevier single manuscript column.

The M1 Shapley CSV is not present on this machine (only M3/M4 are), so this
is reconstructed from the values printed in the existing PNG rather than
regenerated through branch_shapley_plots.py. Palette, bar layout, zero line
and labels mirror branch_shapley_plots.plot_c_mean_signed_phi; only the
figure size and font sizes change (bumped well above the source's 8/9 pt).

Outputs the PNG named for the LaTeX \\includegraphics line plus a vector PDF.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Values read directly from the existing Figure C ──────────────────────
CATEGORIES = ["TP", "FP", "TN", "FN"]
COUNTS     = [10955, 45460, 9693098, 4283]
B1 = [ 3.85,  0.88, -2.60, -1.70]   # fine
B2 = [ 2.24,  1.37, -0.84, -0.43]   # medium
B3 = [-0.26, -0.42, -1.12, -0.20]   # coarse
SERIES = [B1, B2, B3]

BRANCH_LABELS = ["B1 (fine, [1,2,4])",
                 "B2 (medium, [8,16,32])",
                 "B3 (coarse, [32,64,128])"]
BRANCH_COLORS = ["#08519C", "#4292C6", "#9ECAE1"]  # dark / mid / light blue
BRANCH_EDGE   = "black"

TITLE  = "M1 -- Signed branch contribution by prediction category (test)"
YLABEL = "Mean signed phi (logit units; positive = pushes toward seizure)"

OUT_DIR = Path(
    r"C:\Users\messi\OneDrive\Desktop\Desktop\TCN_UNIQURE_PROJECT"
    r"\MultiScaleTCN\interpret_branch_ablation\test\figures")
OUT_STEM = "m1_shapley_test_signed_branch_contribution"

# ── Legibility: fonts well above the source (8/9 pt). ─────────────────────
plt.rcParams.update({
    "font.size":       14,
    "axes.titlesize":  17,
    "xtick.labelsize": 16,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
})


def main():
    x = np.arange(len(CATEGORIES))
    bw = 0.27
    fig, ax = plt.subplots(figsize=(7.8, 6.0))

    for i, (vals, label, color) in enumerate(zip(SERIES, BRANCH_LABELS, BRANCH_COLORS)):
        bars = ax.bar(x + (i - 1) * bw, vals, bw, label=label, color=color,
                      edgecolor=BRANCH_EDGE, linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2,
                    v + (0.09 if v >= 0 else -0.09),
                    "%+0.2f" % v,
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=12, fontweight="bold")

    ax.axhline(0, color="black", linewidth=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(["%s\n(n=%s)" % (c, "{:,}".format(int(n)))
                        for c, n in zip(CATEGORIES, COUNTS)])
    ax.set_ylabel(YLABEL, fontsize=12.5)
    ax.set_title(TITLE)
    ax.set_ylim(-3.3, 4.4)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / (OUT_STEM + ".png")
    pdf = OUT_DIR / (OUT_STEM + ".pdf")
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", pdf)
    print("Saved:", png)


if __name__ == "__main__":
    main()
