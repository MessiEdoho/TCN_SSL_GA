"""
make_fn_diagnosis_chart.py
=========================
Build the FN-audit .xlsx from the audit .csv and embed a PROFESSIONAL bar
chart of the diagnosis_hint distribution as a high-resolution image inside
the workbook (renders identically in every Excel version, unlike a native
openpyxl chart whose long category labels and axes render inconsistently).

- Sheet 'FN_audit'         : the full false-negative table (one row per miss)
- Sheet 'diagnosis_chart'  : diagnosis_hint counts + the embedded chart image
"""

import csv
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage

FN_DIR = Path(r"C:\Users\messi\OneDrive\Desktop\Desktop\TCN_UNIQURE_PROJECT"
              r"\MultiScaleTCN\FN_audit")
STEM = "false_negative_audit_test_min25s_max120s_min_then_refractory"
CSV_PATH = FN_DIR / (STEM + ".csv")
XLSX_PATH = FN_DIR / (STEM + ".xlsx")
PNG_PATH = FN_DIR / "diagnosis_hint_chart.png"

BAR_BLUE = "#3182BD"
BAR_EDGE = "#1F4E79"


def coerce(v):
    if v is None or v == "":
        return None
    if v in ("True", "TRUE", "False", "FALSE"):
        return v in ("True", "TRUE")
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def render_chart(ordered, total, out_png):
    """Horizontal bar chart, largest at top, value + percentage labels."""
    names = [k for k, _ in ordered][::-1]     # reversed so largest ends on top
    vals = [v for _, v in ordered][::-1]

    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 16,
        "axes.labelsize": 14, "ytick.labelsize": 13, "xtick.labelsize": 12,
    })
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bars = ax.barh(names, vals, color=BAR_BLUE, edgecolor=BAR_EDGE, linewidth=0.8)

    xmax = max(vals)
    for b, v in zip(bars, vals):
        ax.text(v + xmax * 0.012, b.get_y() + b.get_height() / 2,
                f"{v}  ({100.0 * v / total:.1f}%)",
                va="center", ha="left", fontsize=12.5, fontweight="bold",
                color="#20303F")

    ax.set_xlabel("Number of missed seizures (false negatives)")
    ax.set_title("MultiScale TCN - Test false negatives by diagnosis "
                 f"(n = {total})", fontweight="bold", pad=12)
    ax.set_xlim(0, xmax * 1.18)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1:]
    dh_idx = header.index("diagnosis_hint")

    counts = Counter(r[dh_idx] for r in data)
    ordered = counts.most_common()
    total = sum(counts.values())
    render_chart(ordered, total, PNG_PATH)

    wb = Workbook()

    # ---- Sheet 1: full FN table ----
    ws = wb.active
    ws.title = "FN_audit"
    ws.append(header)
    for r in data:
        ws.append([coerce(v) for v in r])
    ws.freeze_panes = "A2"

    # ---- Sheet 2: counts table + embedded chart image ----
    cs = wb.create_sheet("diagnosis_chart")
    cs.append(["diagnosis_hint", "count", "percent"])
    for name, c in ordered:
        cs.append([name, c, round(100.0 * c / total, 1)])
    cs.append(["TOTAL", total, 100.0])

    img = XLImage(str(PNG_PATH))
    img.width = int(img.width * 0.62)     # display size (keeps hi-res crispness)
    img.height = int(img.height * 0.62)
    cs.add_image(img, "E2")

    wb.save(XLSX_PATH)
    print("Saved:", XLSX_PATH)
    print("Counts:", dict(ordered), "| total:", total)


if __name__ == "__main__":
    main()
