"""
missed_seizure_audit.py
=======================
Triage MultiScaleTCN false positives (FP) into "likely a real seizure missed by
the expert annotator" vs "likely a true false alarm".

Motivation
----------
Manual EEG annotation misses some seizures. When the model fires on a genuine
but unannotated seizure, the event is scored as a false positive even though it
is correct. This script re-examines every FP using electrographic features
computed DIRECTLY FROM THE RAW EEG (independent of the TCN), compares them with
the model's confirmed seizures (true positives, TP) and with interictal
background, and produces a ranked, calibrated "seizure-likeness" score per FP so
an expert can re-review the strongest candidates.

Method
------
1. For every detected event (TP and FP) and a set of random interictal
   background windows per mouse, extract the single EEG channel from the EDF and
   compute hand-crafted features (amplitude, line-length, band powers, spectral
   entropy, dominant frequency, rhythmicity).
2. Pool all TP (label 1, confirmed seizure) vs all BG (label 0, background)
   across mice and fit a standardised logistic-regression classifier. Report a
   grouped (leave-mice-out) cross-validated ROC-AUC so the discriminator's
   validity is quantified, NOT assumed.
3. Refit on all TP/BG and score every FP -> P(seizure-like) in [0, 1].
4. Outputs: a ranked CSV of all FPs with features and score, trace+spectrogram
   plots for the top-K candidates, and a JSON+text summary.

The output is a PRIORITISED CANDIDATE LIST for expert adjudication, not an
autonomous re-diagnosis.

Usage
-----
python missed_seizure_audit.py                 # defaults below (train, 10s)
python missed_seizure_audit.py --top-k 60
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib
from scipy import signal
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Defaults (the dataset selected for this audit)
# ---------------------------------------------------------------------------
DEFAULT_CSV = Path(
    "C:/Users/messi/OneDrive/Desktop/Desktop/TCN_UNIQURE_PROJECT/MultiScaleTCN/"
    "full_train_evaluation/post_process_varing_sec_train/train/min_then_refractory/"
    "MIN_EVENT_SEC_10s/train_event_details.csv")
DEFAULT_EDF_DIR = Path("D:/uniQure Data/Raw EDF")
DEFAULT_OUT_DIR = DEFAULT_CSV.parent / "missed_seizure_audit"

FS = 500.0                 # sampling rate (Hz) -- verified from EDF headers
CHANNEL = 0                # single EEG channel
BG_PER_MOUSE = 60          # random background windows per mouse
BG_WIN_SEC = 20.0          # background window length
BG_GUARD_SEC = 30.0        # keep background this far from any detected event
PLOT_MARGIN_SEC = 5.0      # context padding around an event in plots
RNG_SEED = 0

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12),
         "beta": (12, 30), "gamma": (30, 70)}
FEATURE_COLS = (["line_length", "rms", "ptp", "spec_entropy", "dom_freq",
                 "band_conc", "rhythmicity"] + ["bp_%s" % b for b in BANDS])

# Artifact QC thresholds. Amplitudes in this dataset saturate at exact integer
# digital rails (1, 2, 3, 4, 5), two-plus orders of magnitude above the
# physiological range (~1e-4 to 1e-2); such clipped windows are non-physiological.
# Flat-line / recording-dropout windows are near-constant, so their
# autocorrelation is ~1 at every lag (rhythmicity ~1). Both are flagged and
# excluded from the candidate ranking (kept, labelled, in the full CSV).
#
# NOTE: an earlier "dominant frequency < 1 Hz" rule was REMOVED -- it discarded
# ~11% of confirmed seizures, which are genuine high-amplitude slow spike-wave
# discharges whose dominant spectral peak sits in the sub-delta band. Low
# dominant frequency is NOT an artifact signature in this dataset.
SATURATION_PTP = 0.5      # >= this is a clipped/saturated rail, never physiological
FLAT_RHYTHMICITY = 0.98   # near-constant (dropout) signal: autocorrelation ~1
CLIP_FRAC = 0.02          # >2% of samples pinned at the window rail = clipping
HIGH_DOM_FREQ = 70.0      # dominant power above this = high-frequency noise


def sec_to_hms(sec):
    """Format seconds-from-EDF-start as 'HhMMmSS.s', matching the
    start_recording_time / end_recording_time columns in train_event_details.csv
    (e.g. 733777.5 -> '203h49m37.5')."""
    sec = float(sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return "%02dh%02dm%04.1f" % (h, m, s)


def add_recording_time(df):
    """Insert human-readable start/end recording-time columns next to the raw
    seconds columns (returns the same DataFrame, modified in place)."""
    df["start_recording_time"] = df["start_sec"].map(sec_to_hms)
    df["end_recording_time"] = df["end_sec"].map(sec_to_hms)
    return df


def artifact_mask(df):
    """Boolean mask: True where a window is a saturation, clipping, flat-line or
    high-frequency-noise artifact.

    - Clipping (flat-topping at a digital rail) is caught by clip_frac regardless
      of the rail amplitude, so it also handles the low (0.01) rail that sits
      inside the physiological amplitude range.
    - High-frequency noise (dominant spectral peak above 70 Hz) is not
      physiological seizure activity; the few confirmed-seizure windows that hit
      this are broadband/artifact-contaminated (low band concentration).
    Low dominant frequency is deliberately NOT flagged (genuine slow spike-wave
    seizures live there)."""
    saturating = df["ptp"] >= SATURATION_PTP
    flat = df["rhythmicity"] > FLAT_RHYTHMICITY
    clipped = df["clip_frac"] > CLIP_FRAC
    hi_freq = df["dom_freq"] > HIGH_DOM_FREQ
    return (saturating | flat | clipped | hi_freq).values


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def compute_features(x, fs=FS):
    """Hand-crafted electrographic features for a 1-D EEG segment.

    Returns None if the segment is too short (< 1 s) to be meaningful.
    Features are intentionally model-independent so the audit does not just
    re-derive the TCN's own decision.
    """
    if x is None or len(x) < int(fs):
        return None
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)

    line_length = np.sum(np.abs(np.diff(x))) / len(x)
    rms = np.sqrt(np.mean(x ** 2))
    ptp = float(np.ptp(x))

    # Clipping / flat-topping fraction: digital saturation pins many samples at
    # exactly the window max or min. Physiological signals touch their extremes
    # only momentarily, so a high clip fraction is a robust artifact signature
    # independent of the (dataset-specific, multi-rail) saturation amplitude.
    vmax, vmin = x.max(), x.min()
    rng = (vmax - vmin) + 1e-12
    tol = 1e-6 * rng
    clip_frac = float((np.isclose(x, vmax, atol=tol) |
                       np.isclose(x, vmin, atol=tol)).mean())

    nperseg = int(min(len(x), 2 * fs))
    fr, P = signal.welch(x, fs, nperseg=nperseg)
    total = np.trapz(P, fr) + 1e-12
    bp = {}
    for name, (lo, hi) in BANDS.items():
        m = (fr >= lo) & (fr < hi)
        bp["bp_%s" % name] = float(np.trapz(P[m], fr[m]) / total)
    # Band concentration: fraction of power in the single strongest band. High
    # for organised seizures (narrow evolving band), low for broadband EMG.
    band_conc = float(max(bp.values()))
    Pn = P / total
    spec_entropy = float(-np.sum(Pn * np.log(Pn + 1e-12)) / np.log(len(Pn)))
    dom_freq = float(fr[np.argmax(P)])

    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac = ac / (ac[0] + 1e-12)
    lo_lag, hi_lag = int(0.1 * fs), int(1.0 * fs)
    rhythmicity = float(np.max(ac[lo_lag:hi_lag])) if hi_lag < len(ac) else 0.0

    out = dict(line_length=line_length, rms=rms, ptp=ptp, clip_frac=clip_frac,
               spec_entropy=spec_entropy, dom_freq=dom_freq,
               band_conc=band_conc, rhythmicity=rhythmicity)
    out.update(bp)
    return out


def sample_background(events, n_total_sec, n_windows, win_sec, guard_sec, rng):
    """Pick random interictal windows that avoid all detected events."""
    ev = events[["start_sec", "end_sec"]].values
    out, tries = [], 0
    while len(out) < n_windows and tries < n_windows * 50:
        s = rng.uniform(100.0, n_total_sec - 100.0)
        e = s + win_sec
        tries += 1
        if not np.any((ev[:, 0] < e + guard_sec) & (ev[:, 1] > s - guard_sec)):
            out.append((s, e))
    return out


# ---------------------------------------------------------------------------
# Per-mouse feature build
# ---------------------------------------------------------------------------
def build_features_for_mouse(mouse, mdf, edf_dir, rng):
    """Extract features for every event (TP+FP) and background windows.

    Returns a DataFrame with one row per window, carrying the source event
    metadata, the group label (TP / FP / BG) and all feature columns.
    """
    edf_path = edf_dir / ("%s.edf" % mouse)
    if not edf_path.exists():
        return None
    f = pyedflib.EdfReader(str(edf_path))
    try:
        n_samp = f.getNSamples()[CHANNEL]
        fs = f.getSampleFrequency(CHANNEL)
        total_sec = n_samp / fs

        def read(s0, s1):
            a = int(max(0, s0 * fs))
            b = int(min(n_samp, s1 * fs))
            return f.readSignal(CHANNEL, a, b - a) if b > a else None

        rows = []
        # detected events
        for _, r in mdf.iterrows():
            ft = compute_features(read(r.start_sec, r.end_sec), fs)
            if ft is None:
                continue
            ft.update(dict(mouse_id=mouse, event_idx=int(r.event_idx),
                           group=("FP" if r.is_fp else "TP"),
                           start_sec=float(r.start_sec), end_sec=float(r.end_sec),
                           duration_sec=float(r.duration_sec),
                           mean_prob=float(r.mean_prob), max_prob=float(r.max_prob)))
            rows.append(ft)
        # background
        bg = sample_background(mdf, total_sec, BG_PER_MOUSE, BG_WIN_SEC,
                               BG_GUARD_SEC, rng)
        for s, e in bg:
            ft = compute_features(read(s, e), fs)
            if ft is None:
                continue
            ft.update(dict(mouse_id=mouse, event_idx=-1, group="BG",
                           start_sec=float(s), end_sec=float(e),
                           duration_sec=float(e - s), mean_prob=0.0, max_prob=0.0))
            rows.append(ft)
        return pd.DataFrame(rows)
    finally:
        f.close()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_candidate(row, edf_dir, out_dir, rank):
    """EEG trace + spectrogram for one candidate FP, with context margin."""
    mouse = row["mouse_id"]
    edf_path = edf_dir / ("%s.edf" % mouse)
    f = pyedflib.EdfReader(str(edf_path))
    try:
        n_samp = f.getNSamples()[CHANNEL]
        fs = f.getSampleFrequency(CHANNEL)
        s0 = max(0.0, row["start_sec"] - PLOT_MARGIN_SEC)
        s1 = min(n_samp / fs, row["end_sec"] + PLOT_MARGIN_SEC)
        a, b = int(s0 * fs), int(s1 * fs)
        x = f.readSignal(CHANNEL, a, b - a)
    finally:
        f.close()
    t = np.arange(len(x)) / fs + s0

    fig, ax = plt.subplots(2, 1, figsize=(11, 6), gridspec_kw={"height_ratios": [1, 1]})
    ax[0].plot(t, x, lw=0.4, color="k")
    ax[0].axvspan(row["start_sec"], row["end_sec"], color="red", alpha=0.15,
                  label="model event")
    ax[0].set_ylabel("EEG (uV)")
    ax[0].set_title("rank %d | %s event %d | t=%s | score=%.3f | max_prob=%.3f | dur=%.1fs"
                    % (rank, mouse, int(row["event_idx"]),
                       sec_to_hms(row["start_sec"]), row["seizure_likeness"],
                       row["max_prob"], row["duration_sec"]))
    ax[0].legend(loc="upper right", fontsize=8)
    ax[1].specgram(x - np.mean(x), NFFT=int(fs), Fs=fs, noverlap=int(fs * 0.75),
                   cmap="viridis")
    ax[1].set_ylim(0, 70)
    ax[1].set_ylabel("Freq (Hz)")
    ax[1].set_xlabel("Time (s, EDF-relative)")
    # specgram x-axis is 0-based; relabel to absolute seconds
    ticks = ax[1].get_xticks()
    ax[1].set_xticks(ticks)
    ax[1].set_xticklabels(["%.0f" % (s0 + tick) for tick in ticks])
    plt.tight_layout()
    path = out_dir / ("cand_%03d_%s_ev%d.png" % (rank, mouse, int(row["event_idx"])))
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--edf-dir", type=Path, default=DEFAULT_EDF_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--top-k", type=int, default=40,
                    help="Number of top-ranked FP candidates to plot.")
    ap.add_argument("--cache-features", action="store_true",
                    help="Reuse all_window_features.csv from a previous run "
                         "instead of re-reading the EDFs (feature extraction is "
                         "the expensive step; QC/scoring/plots are cheap).")
    args = ap.parse_args()

    rng = np.random.RandomState(RNG_SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.out_dir / "candidate_plots"
    plot_dir.mkdir(exist_ok=True)
    for stale in plot_dir.glob("cand_*.png"):   # clear stale plots from prior runs
        stale.unlink()

    print("Loading detections: %s" % args.csv)
    d = pd.read_csv(args.csv)
    d["is_fp"] = d["is_true_alarm"].astype(str).str.upper() == "FALSE"
    mice = sorted(d["mouse_id"].unique())
    print("Mice: %d | events: %d | TP: %d | FP: %d"
          % (len(mice), len(d), (~d.is_fp).sum(), d.is_fp.sum()))

    # 1. Feature extraction (or reuse cache)
    cache_path = args.out_dir / "all_window_features.csv"
    if args.cache_features and cache_path.exists():
        print("Reusing cached features: %s" % cache_path)
        feat = pd.read_csv(cache_path)
    else:
        feats = []
        for i, m in enumerate(mice, 1):
            fm = build_features_for_mouse(m, d[d.mouse_id == m], args.edf_dir, rng)
            if fm is not None and len(fm):
                feats.append(fm)
            print("  [%2d/%2d] %-6s rows=%s" % (i, len(mice), m,
                  0 if fm is None else len(fm)))
        feat = pd.concat(feats, ignore_index=True)
        feat.to_csv(cache_path, index=False)

    # Human-readable recording-time columns (HhMMmSS.s), matching the source
    # detection CSV. Rewrite the cache so it carries them too.
    add_recording_time(feat)
    feat.to_csv(cache_path, index=False)

    # Artifact QC: flag saturation/flat-line windows so they neither train the
    # classifier nor pollute the candidate ranking.
    feat["artifact"] = artifact_mask(feat)
    n_art = int(feat["artifact"].sum())
    print("Artifact-flagged windows (saturation/flat-line): %d / %d"
          % (n_art, len(feat)))

    clean = feat[~feat["artifact"]]
    tp = clean[clean.group == "TP"]
    bg = clean[clean.group == "BG"]
    fp = clean[clean.group == "FP"].copy()
    fp_artifact = feat[(feat.group == "FP") & feat["artifact"]].copy()
    print("\nClean feature rows -> TP=%d BG=%d FP=%d  (FP artifacts set aside: %d)"
          % (len(tp), len(bg), len(fp), len(fp_artifact)))

    # 2. Grouped CV ROC-AUC (validity of the discriminator)
    train = pd.concat([tp, bg], ignore_index=True)
    X = train[FEATURE_COLS].values
    y = (train.group == "TP").astype(int).values
    groups = train.mouse_id.values
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=1000, class_weight="balanced"))
    n_splits = min(5, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    proba_cv = cross_val_predict(clf, X, y, cv=gkf, groups=groups,
                                 method="predict_proba")[:, 1]
    auc = roc_auc_score(y, proba_cv)
    print("Grouped %d-fold CV ROC-AUC (TP vs background): %.3f" % (n_splits, auc))

    # 3. Refit on all TP/BG, score every FP
    clf.fit(X, y)
    fp["seizure_likeness"] = clf.predict_proba(fp[FEATURE_COLS].values)[:, 1]
    # reference: distribution of the scorer on confirmed seizures
    tp_scores = clf.predict_proba(tp[FEATURE_COLS].values)[:, 1]
    thr10 = float(np.quantile(tp_scores, 0.10))
    thr50 = float(np.quantile(tp_scores, 0.50))

    fp_ranked = fp.sort_values("seizure_likeness", ascending=False).reset_index(drop=True)
    rank_cols = (["mouse_id", "event_idx", "start_sec", "end_sec",
                  "start_recording_time", "end_recording_time", "duration_sec",
                  "mean_prob", "max_prob", "seizure_likeness"] + FEATURE_COLS)
    fp_ranked[rank_cols].to_csv(args.out_dir / "fp_ranked_seizure_likeness.csv",
                                index=False)
    # Artifact FPs kept separately (excluded from ranking, retained for audit).
    if len(fp_artifact):
        fp_artifact["seizure_likeness"] = clf.predict_proba(
            fp_artifact[FEATURE_COLS].values)[:, 1]
        fp_artifact[rank_cols].to_csv(
            args.out_dir / "fp_artifacts_excluded.csv", index=False)

    n_ge10 = int((fp_ranked.seizure_likeness >= thr10).sum())
    n_ge50 = int((fp_ranked.seizure_likeness >= thr50).sum())

    # 4. Plots for top-K
    k = min(args.top_k, len(fp_ranked))
    print("\nPlotting top %d candidates ..." % k)
    for i in range(k):
        plot_candidate(fp_ranked.iloc[i], args.edf_dir, plot_dir, i + 1)

    # 5. Summary
    per_mouse = (fp_ranked[fp_ranked.seizure_likeness >= thr10]
                 .groupby("mouse_id").size().sort_values(ascending=False))
    summary = {
        "csv": str(args.csv),
        "n_mice": len(mice),
        "n_events": int(len(d)),
        "n_TP": int((~d.is_fp).sum()),
        "n_FP": int(d.is_fp.sum()),
        "n_FP_artifact_excluded": int(len(fp_artifact)),
        "n_FP_scored": int(len(fp_ranked)),
        "cv_roc_auc_tp_vs_background": round(auc, 4),
        "tp_score_threshold_p10": round(thr10, 4),
        "tp_score_threshold_p50": round(thr50, 4),
        "n_FP_above_p10_of_seizures": n_ge10,
        "n_FP_above_p50_of_seizures": n_ge50,
        "pct_FP_above_p10": round(100.0 * n_ge10 / max(len(fp_ranked), 1), 2),
        "feature_columns": FEATURE_COLS,
    }
    with open(args.out_dir / "summary.json", "w", encoding="utf-8") as fjson:
        json.dump(summary, fjson, indent=2)

    lines = [
        "MISSED-SEIZURE AUDIT OF MULTISCALE-TCN FALSE POSITIVES",
        "=" * 60,
        "Source detections : %s" % args.csv,
        "Mice              : %d" % len(mice),
        "Detected events   : %d  (TP=%d, FP=%d)" % (len(d), (~d.is_fp).sum(), d.is_fp.sum()),
        "",
        "ARTIFACT QC",
        "  %d FP windows flagged as saturation/flat-line artifacts and excluded"
        % len(fp_artifact),
        "  from ranking (kept in fp_artifacts_excluded.csv). %d FPs scored."
        % len(fp_ranked),
        "",
        "DISCRIMINATOR VALIDITY",
        "  Features are computed from raw EEG, independent of the TCN.",
        "  Grouped %d-fold (leave-mice-out) CV ROC-AUC, confirmed seizures" % n_splits,
        "  vs interictal background: %.3f" % auc,
        "  (>=0.5 chance; higher means seizures are electrographically",
        "   separable from background on independent features.)",
        "",
        "FALSE-POSITIVE TRIAGE",
        "  Each FP scored P(seizure-like) in [0,1] by the same classifier.",
        "  Reference thresholds from confirmed seizures' own scores:",
        "    p10 of seizures = %.3f   p50 of seizures = %.3f" % (thr10, thr50),
        "  FPs scoring as seizure-like as real seizures (>= p10): %d / %d (%.1f%%)"
        % (n_ge10, len(fp_ranked), 100.0 * n_ge10 / max(len(fp_ranked), 1)),
        "  FPs scoring above the seizure median  (>= p50): %d / %d (%.1f%%)"
        % (n_ge50, len(fp_ranked), 100.0 * n_ge50 / max(len(fp_ranked), 1)),
        "",
        "  These are CANDIDATE missed seizures for expert re-review, NOT",
        "  confirmed diagnoses.",
        "",
        "TOP MICE BY CANDIDATE COUNT (FP >= p10 of seizures)",
    ]
    for m, c in per_mouse.head(15).items():
        lines.append("  %-6s %d" % (m, c))
    lines += [
        "",
        "OUTPUTS",
        "  fp_ranked_seizure_likeness.csv  - all FPs, ranked, with features",
        "  all_window_features.csv         - every window (TP/FP/BG) + features",
        "  candidate_plots/                - trace+spectrogram for top %d FPs" % k,
        "  summary.json                    - machine-readable summary",
    ]
    report = "\n".join(lines)
    with open(args.out_dir / "summary_report.txt", "w", encoding="utf-8") as ftxt:
        ftxt.write(report + "\n")
    print("\n" + report)
    print("\nAll outputs -> %s" % args.out_dir)


if __name__ == "__main__":
    main()
