"""
recolor_m1_phi_heatmap_blue.py
==============================
One-off recolour of the "M1 -- Mean |phi| per mouse over TP segments
(test)" branch-Shapley heatmap into shades of blue.

The M1 Shapley CSV is not present on this machine (only M3/M4 are), so
this cannot be regenerated through branch_shapley_plots.py. Instead the
figure is reconstructed from the values printed in the existing PNG --
every cell and TP count is legible -- and re-rendered with the "Blues"
colormap in place of the original "viridis". Layout (heatmap + TP-count
bar panel), row order and annotations mirror
branch_shapley_plots.plot_d_per_mouse_heatmap.

If the underlying M1 CSV becomes available, prefer regenerating via
branch_shapley_plots.py with cmap="Blues" instead of this reconstruction.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Values read directly from the existing M1 heatmap (top -> bottom) ─────
MICE   = ["m6", "m381", "m236", "m424", "m266", "m427", "m296", "m263",
          "m379", "m428", "m384", "m258", "m382", "m431", "m421", "m425",
          "m289", "m426", "m376"]
B1     = [2.94, 4.19, 3.57, 4.17, 2.79, 6.78, 4.21, 5.39, 1.52, 4.39,
          3.89, 3.99, 2.44, 4.17, 4.85, 3.49, 5.23, 1.68, 5.48]
B2     = [1.42, 1.69, 2.09, 2.03, 1.94, 2.09, 2.08, 1.88, 2.04, 1.56,
          2.31, 2.32, 2.22, 1.71, 2.01, 2.61, 1.80, 3.19, 2.18]
B3     = [0.32, 0.77, 0.65, 0.77, 0.71, 0.73, 1.03, 0.55, 0.78, 0.61,
          0.80, 0.56, 0.69, 0.71, 0.53, 0.67, 0.54, 0.64, 0.43]
COUNTS = [2587, 1880, 1300, 972, 689, 471, 466, 400, 318, 304,
          304, 245, 236, 188, 156, 138, 113, 97, 91]

BRANCH_LABELS = ["B1 (fine, [1,2,4])",
                 "B2 (medium, [8,16,32])",
                 "B3 (coarse, [32,64,128])"]
TITLE = "M1 -- Mean |phi| per mouse over TP segments (test)"

# Blue theme: sequential "Blues" for the heatmap; a mid-blue for the
# TP-count bars (replacing the original TP green) so the whole figure is
# one blue palette.
CMAP     = "Blues"
BAR_BLUE = "#3182BD"
DPI      = 300

OUT_DIR = Path(
    r"C:\Users\messi\OneDrive\Desktop\Desktop\TCN_UNIQURE_PROJECT"
    r"\MultiScaleTCN\interpret_branch_ablation\test\figures")
OUT_STEM = "m1_shapley_test_fig_d_per_mouse_heatmap_TP_blue"


def main():
    heat = np.column_stack([B1, B2, B3]).astype(float)
    n = len(MICE)

    fig, (ax_heat, ax_count) = plt.subplots(
        1, 2, figsize=(8, max(4, 0.35 * n)),
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05})

    im = ax_heat.imshow(heat, aspect="auto", cmap=CMAP)
    ax_heat.set_xticks(range(3))
    ax_heat.set_xticklabels(BRANCH_LABELS, rotation=20, ha="right")
    ax_heat.set_yticks(range(n))
    ax_heat.set_yticklabels(MICE)
    ax_heat.set_title(TITLE)

    # Annotation contrast is INVERTED relative to the viridis original:
    # in Blues the high values are dark (need white text) and low values
    # are light (need dark text) -- the opposite of viridis.
    thr = heat.max() * 0.5
    for i in range(n):
        for j in range(3):
            v = heat[i, j]
            ax_heat.text(j, i, "%.2f" % v, ha="center", va="center",
                         color="white" if v >= thr else "black", fontsize=8)

    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.85, pad=0.02)
    cbar.set_label("Mean |phi|")

    y = np.arange(n)
    ax_count.barh(y, COUNTS, color=BAR_BLUE, edgecolor="black", linewidth=0.3)
    ax_count.set_yticks(y)
    ax_count.set_yticklabels([])
    for i, v in enumerate(COUNTS):
        ax_count.text(v, i, " %s" % "{:,}".format(int(v)),
                      va="center", ha="left", fontsize=8)
    ax_count.set_xlabel("TP count")
    ax_count.grid(axis="x", alpha=0.3)
    ax_count.set_xlim(0, max(COUNTS) * 1.25)

    # Both panels: highest-count mouse (m6) at the top, matching the source.
    ax_heat.set_ylim(n - 0.5, -0.5)
    ax_count.set_ylim(n - 0.5, -0.5)

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / (OUT_STEM + ".png")
    pdf = OUT_DIR / (OUT_STEM + ".pdf")
    plt.savefig(png, dpi=DPI, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()
    print("Saved:", png)
    print("Saved:", pdf)


if __name__ == "__main__":
    main()
