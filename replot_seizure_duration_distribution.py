import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANN_DIR = Path(r"C:\Users\messi\OneDrive\Desktop\Desktop\TCN_UNIQURE_PROJECT\seizure_times_updated")
OUT_DIR = Path(r"C:\Users\messi\OneDrive\Desktop\Desktop\TCN_UNIQURE_PROJECT\figures")
# One-column Elsevier publication version (with MIN_EVENT_SEC operating point).
# Kept under a distinct name so the earlier "_no_min" figure is preserved.
OUT_PNG = OUT_DIR / "seizure_duration_distribution_min25_1col.png"
OUT_PDF = OUT_DIR / "seizure_duration_distribution_min25_1col.pdf"
LOG_PATH = OUT_DIR / "replot_seizure_duration_distribution.log"

BIN_WIDTH = 5
XMAX = 130
DPI = 300
MIN_EVENT_SEC = 25   # selected operating point drawn as the red dashed line

# Fonts sized well above matplotlib defaults so the figure stays legible
# once scaled to a single Elsevier column (~90 mm) via width=\linewidth.
plt.rcParams.update({
    "font.size":       14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 11,
})


def setup_logging():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("replot_seizure_duration")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger


def load_durations(logger):
    files = sorted(ANN_DIR.glob("m*_xlsx.xlsx"))
    if not files:
        logger.error("No annotation files found in %s", ANN_DIR)
        sys.exit(1)
    durations = []
    n_files_ok = 0
    for f in files:
        try:
            df = pd.read_excel(f)
        except Exception as e:
            logger.error("Failed to read %s: %s", f.name, e)
            continue
        if not {"start_time", "end_time"}.issubset(df.columns):
            logger.warning("%s missing start_time / end_time columns -- skipped", f.name)
            continue
        starts = pd.to_datetime(df["start_time"], dayfirst=True, errors="coerce")
        ends = pd.to_datetime(df["end_time"], dayfirst=True, errors="coerce")
        d = (ends - starts).dt.total_seconds().dropna()
        durations.extend(d.tolist())
        n_files_ok += 1
    return np.asarray(durations, dtype=float), len(files), n_files_ok


def main():
    logger = setup_logging()
    logger.info("Annotation dir: %s", ANN_DIR)
    logger.info("Output PNG    : %s", OUT_PNG)
    logger.info("Log file      : %s", LOG_PATH)

    d, n_files, n_files_ok = load_durations(logger)
    logger.info("Loaded %d seizures across %d annotation files (%d files OK)",
                len(d), n_files, n_files_ok)

    mean = float(np.mean(d))
    std = float(np.std(d, ddof=1))
    median = float(np.median(d))
    q1 = float(np.percentile(d, 25))
    q3 = float(np.percentile(d, 75))
    bins = np.arange(0, XMAX + BIN_WIDTH, BIN_WIDTH)
    counts, edges = np.histogram(d, bins=bins)
    mode_idx = int(np.argmax(counts))
    mode_bin = (int(edges[mode_idx]), int(edges[mode_idx + 1]))
    n_below = int(np.sum(d < MIN_EVENT_SEC))
    pct_below = 100.0 * n_below / len(d)
    logger.info("n=%d  mean=%.2f  std=%.2f  median=%.2f  Q1=%.2f  Q3=%.2f  mode bin=%s",
                len(d), mean, std, median, q1, q3, mode_bin)
    logger.info("< %d s: %d (%.1f%%)", MIN_EVENT_SEC, n_below, pct_below)

    # Column-friendly aspect ratio; large fonts (set via rcParams above).
    fig, ax = plt.subplots(figsize=(7.4, 5.0))

    # Shaded < MIN_EVENT_SEC region behind the bars.
    ax.axvspan(0, MIN_EVENT_SEC, color="#D9534F", alpha=0.07, zorder=0)
    ax.hist(d, bins=bins, color="#6E84B6", edgecolor="white", linewidth=0.5, zorder=2)

    # Operating point first so it heads the legend, then the quartiles.
    ax.axvline(MIN_EVENT_SEC, linestyle="--", color="#C0392B", linewidth=2.6,
               label="MIN_EVENT_SEC = %d s (selected operating point)" % MIN_EVENT_SEC)
    ax.axvline(q1, linestyle=":", color="#333333", linewidth=1.5,
               label="Q1 = %.1f s" % q1)
    ax.axvline(median, linestyle="-.", color="#333333", linewidth=1.5,
               label="Median = %.1f s" % median)
    ax.axvline(q3, linestyle=":", color="#333333", linewidth=1.5,
               label="Q3 = %.1f s" % q3)

    # < MIN_EVENT_SEC count/percentage annotation, inside the shaded band.
    # Stacked on three short lines so no line reaches the dashed line at
    # x = MIN_EVENT_SEC.
    ax.text(0.03, 0.90, "< %d s:\n%d\n(%.1f %%)" % (MIN_EVENT_SEC, n_below, pct_below),
            transform=ax.transAxes, color="#C0392B", fontsize=13, fontweight="bold",
            va="top", ha="left", linespacing=1.3)

    ax.set_xlim(0, XMAX)
    # Headroom above the tallest bar so the upper-right legend sits in clear
    # space instead of overlapping the central bars.
    ax.set_ylim(0, counts.max() * 1.32)
    ax.set_xlabel("Seizure duration (s)", fontsize=14)
    ax.set_ylabel("Number of annotated seizures", fontsize=14)
    # Single title block sitting directly above the plot (small pad) so
    # there is no gap between the title and the axes. Bold first line via
    # mathtext; the long stats string is split so it stays legible at
    # column width (a single line would not fit a ~90 mm column).
    ax.set_title(
        r"$\mathbf{Empirical\ distribution\ of\ annotated\ seizure\ durations}$"
        "\n"
        "n = %d seizures across %d mice; mean = %.1f $\\pm$ %.1f s\n"
        "median = %.1f s; mode bin = %d-%d s"
        % (len(d), n_files, mean, std, median, mode_bin[0], mode_bin[1]),
        fontsize=12.5, pad=6)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PDF, bbox_inches="tight")
    plt.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", OUT_PNG)
    logger.info("Saved: %s", OUT_PDF)


if __name__ == "__main__":
    main()
