"""
recolor_m1_winner_branch_big.py
===============================
Re-render the "M1 -- Dominant branch by prediction category (test)" branch-
Shapley histogram (Figure B) with larger, clearer fonts sized for a SINGLE
manuscript column.

The M1 Shapley CSV is not present on this machine (only M3/M4 are), so this
is reconstructed from the percentages printed in the existing PNG -- every
bar label and category count is legible -- rather than regenerated through
branch_shapley_plots.py. Palette, bar layout and labels mirror
branch_shapley_plots.plot_b_winner_histogram; only the figure size and font
sizes change (bumped well above the source's fontsize 8/9 so the figure
stays legible when scaled to column width).

Outputs a vector PDF (for \\includegraphics[width=\\linewidth]{...}) plus a
dpi-300 PNG preview.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Values read directly from the existing Figure B ──────────────────────
CATEGORIES = ["TP", "FP", "TN", "FN"]
COUNTS     = [10955, 45460, 9693098, 4283]
B1 = [65.9, 42.4, 93.4, 72.5]   # fine
B2 = [28.8, 42.2,  1.9, 11.5]   # medium
B3 = [ 5.3, 15.4,  4.8, 16.0]   # coarse
SERIES = [B1, B2, B3]

BRANCH_LABELS = ["B1 (fine, [1,2,4])",
                 "B2 (medium, [8,16,32])",
                 "B3 (coarse, [32,64,128])"]
BRANCH_COLORS = ["#08519C", "#4292C6", "#9ECAE1"]  # dark / mid / light blue
BRANCH_EDGE   = "black"

TITLE  = "M1 -- Dominant branch by prediction category (test)"
YLABEL = "% of segments where branch is argmax|phi|"

OUT_DIR = Path(
    r"C:\Users\messi\OneDrive\Desktop\Desktop\TCN_UNIQURE_PROJECT"
    r"\MultiScaleTCN\interpret_branch_ablation\test\figures")
OUT_STEM = "m1_shapley_test_fig_b_winner_branch_histogram_big"

# ── Legibility: fonts well above the source (8/9 pt) so they survive being
#    scaled down to one manuscript column. ────────────────────────────────
plt.rcParams.update({
    "font.size":       15,
    "axes.titlesize":  17,
    "axes.labelsize":  16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
})


def main():
    x = np.arange(len(CATEGORIES))
    bw = 0.27
    fig, ax = plt.subplots(figsize=(7.6, 5.8))

    # Per-series vertical offset for the value labels: lift the middle
    # series so the near-equal FP bars (42.4% vs 42.2%) and the tiny TN
    # bars don't collide. FN's B2 label is exempted (override) because its
    # bars differ enough in height that the lift would instead crowd 16.0%.
    yoff = [1.0, 6.5, 1.0]
    overrides = {(1, 3): 1.0}   # (series B2, category FN) -> no lift
    for i, (vals, label, color) in enumerate(zip(SERIES, BRANCH_LABELS, BRANCH_COLORS)):
        bars = ax.bar(x + (i - 1) * bw, vals, bw, label=label, color=color,
                      edgecolor=BRANCH_EDGE, linewidth=0.6)
        for k, (b, v) in enumerate(zip(bars, vals)):
            off = overrides.get((i, k), yoff[i])
            ax.text(b.get_x() + b.get_width() / 2, v + off, "%.1f%%" % v,
                    ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["%s\n(n=%s)" % (c, "{:,}".format(int(n)))
                        for c, n in zip(CATEGORIES, COUNTS)])
    ax.set_ylabel(YLABEL)
    ax.set_title(TITLE)
    ax.set_ylim(0, 116)
    # Upper-left is the only clear corner: tallest left bars are ~66% (TP),
    # while the TN bar (93.4%) sits centre-right under the old upper-right.
    ax.legend(loc="upper left", framealpha=0.95)
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
