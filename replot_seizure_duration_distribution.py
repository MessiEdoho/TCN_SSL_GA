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
OUT_PNG = OUT_DIR / "seizure_duration_distribution_no_min.png"
LOG_PATH = OUT_DIR / "replot_seizure_duration_distribution.log"

BIN_WIDTH = 5
XMAX = 130
DPI = 300


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
    logger.info("n=%d  mean=%.2f  std=%.2f  median=%.2f  Q1=%.2f  Q3=%.2f  mode bin=%s",
                len(d), mean, std, median, q1, q3, mode_bin)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.hist(d, bins=bins, color="#6E84B6", edgecolor="white", linewidth=0.5)
    ax.axvline(q1, linestyle=":", color="black", linewidth=1.0,
               label="Q1 = %.1f s" % q1)
    ax.axvline(median, linestyle="-.", color="black", linewidth=1.0,
               label="Median = %.1f s" % median)
    ax.axvline(q3, linestyle=":", color="black", linewidth=1.0,
               label="Q3 = %.1f s" % q3)
    ax.set_xlim(0, XMAX)
    ax.set_xlabel("Seizure duration (s)")
    ax.set_ylabel("Number of annotated seizures")
    ax.set_title("Empirical distribution of annotated seizure durations\n"
                 "(n = %d seizures across %d mice; mean = %.1f $\\pm$ %.1f s; "
                 "median = %.1f s; mode bin = %d-%d s)"
                 % (len(d), n_files, mean, std, median, mode_bin[0], mode_bin[1]))
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", OUT_PNG)


if __name__ == "__main__":
    main()
