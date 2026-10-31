"""
postproc_ablation.py
====================
Post-processing ablation for the MultiScale TCN (repo: M3; paper: M1) on the
independent TEST partition. Answers the reviewer request to analyse the
individual contribution of each post-processing stage. Reads cached
predictions; CPU-only, no model forward, no GPU.

Ablated stages (4 on/off factors, 2^4 = 16 configurations):
    S  probability smoothing    on: W = 3         off: W = 1
    M  minimum-duration filter  on: 25 s          off: 0 s
    R  refractory-period merge  on: 30 s          off: 0 s
    X  maximum-duration filter  on: 120 s         off: infinite

The threshold (tau = 0.5) and the stage order (M->R; max-duration last) are
fixed. Nothing is tuned: this explains the selected pipeline, it does not
re-select it. Metrics are event-level only.

Reuse of the production pipeline
--------------------------------
Event detection and ground-truth matching are the unchanged eval_utils
functions. SMOOTHING_WIN and REFRACTORY_SEC have no call-level parameter, so
they are rebound on the eval_utils module per configuration (the same
technique as m3_post_eval._redirect_outputs_to_post_eval) and restored
afterwards. eval_utils.py itself is not edited.

evaluate_event_level is deliberately not used: it computes segment-level
metrics (not reported here) and needs one Python dict per segment. The test
manifest section is instead streamed into compact arrays, which keeps memory
low on a workstation (the manifest is 3.5 GB of JSON).

Reproducibility guard
---------------------
The all-stages-on configuration runs first and must reproduce the paper's
Table 4 test row (TP 621, FP 535, FN 63). On a mismatch the script exits with
an error before writing any result file.

Outputs (under <MultiScaleTCN>/evaluation/postproc_ablation/)
    ablation_all_configs.csv   16 rows: every on/off combination
    ablation_paper_table.csv   10 rows: none, each stage alone, full minus
                               each stage, full; deltas are versus full
    postproc_ablation.log

Usage
-----
    python postproc_ablation.py            # local Windows-mirror paths
    python postproc_ablation.py --cluster  # cluster paths
"""

import argparse
import csv
import datetime
import itertools
import json
import logging
import sys
from pathlib import Path

import numpy as np

import eval_utils
from eval_utils import (
    detect_events_in_chunk,
    load_annotations,
    match_events_to_ground_truth,
    safe_div,
    split_into_chunks,
)
from postproc_sweep import CLUSTER_OUTPUT, LOCAL_DEFAULT, variant_config


VARIANT = "MultiScaleTCN"
ORDER   = "min_then_refractory"     # M->R

STAGES       = ["S", "M", "R", "X"]
STAGE_LABEL  = {
    "S": "smoothing",
    "M": "min_duration",
    "R": "refractory",
    "X": "max_duration",
}
SMOOTHING_ON,  SMOOTHING_OFF  = eval_utils.SMOOTHING_WIN,  1
MIN_SEC_ON,    MIN_SEC_OFF    = eval_utils.MIN_EVENT_SEC,  0.0
REFRACTORY_ON, REFRACTORY_OFF = eval_utils.REFRACTORY_SEC, 0.0
MAX_SEC_ON,    MAX_SEC_OFF    = eval_utils.MAX_EVENT_SEC,  float("inf")

# Paper Table 4, MultiScale TCN, independent test set.
EXPECTED_FULL = {"tp": 621, "fp": 535, "fn": 63}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cluster", action="store_true",
                   help="Use cluster paths. Default: local Windows-mirror paths.")
    p.add_argument("--local-root",   type=Path, default=LOCAL_DEFAULT)
    p.add_argument("--cluster-root", type=Path, default=CLUSTER_OUTPUT)
    p.add_argument("--output-dir",   type=Path, default=None,
                   help="Override output directory.")
    return p.parse_args()


def setup_logging(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "postproc_ablation.log"
    logger = logging.getLogger("postproc_ablation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8"); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger, log_path


def stream_test_chronology(manifest_path, n_expected, logger):
    """Stream the "test" section of the pretty-printed enriched manifest
    into arrays aligned with the raw NPZ (manifest order). Returns
    (mouse_id_list, label, chrono_idx, t_start_sec); chrono_idx is -1 for
    records without chronology.
    """
    labels      = np.empty(n_expected, dtype=np.int64)
    mouse_codes = np.empty(n_expected, dtype=np.int32)
    chrono_idx  = np.full(n_expected, -1, dtype=np.int64)
    t_start     = np.full(n_expected, np.nan, dtype=np.float64)
    mouse_to_code = {}

    i = -1
    in_test = False
    with open(manifest_path, "rb") as f:
        for line in f:
            if not in_test:
                in_test = line.startswith(b'  "test": [')
                continue
            s = line.strip()
            if s == b"{":
                i += 1
                if i >= n_expected:
                    raise ValueError("Manifest test section has more than "
                                     "%d records." % n_expected)
            elif s.startswith(b'"label":'):
                labels[i] = int(s[8:].rstrip(b","))
            elif s.startswith(b'"mouse_id":'):
                mouse = s[11:].strip(b' ",').decode("ascii")
                mouse_codes[i] = mouse_to_code.setdefault(mouse, len(mouse_to_code))
            elif s.startswith(b'"chrono_idx":'):
                value = s[13:].rstrip(b",").strip()
                if value != b"null":
                    chrono_idx[i] = int(value)
            elif s.startswith(b'"t_start_sec":'):
                value = s[14:].rstrip(b",").strip()
                if value != b"null":
                    t_start[i] = float(value)
            elif line.startswith(b"  ]"):
                break

    n_records = i + 1
    if n_records != n_expected:
        raise ValueError("Manifest test section has %d records; NPZ has %d."
                         % (n_records, n_expected))
    code_to_mouse = {code: mouse for mouse, code in mouse_to_code.items()}
    logger.info("Streamed %d test records (%d mice) from manifest.",
                n_records, len(code_to_mouse))
    return [code_to_mouse[c] for c in range(len(code_to_mouse))], \
        mouse_codes, labels, chrono_idx, t_start


def load_test_mice(npz_path, manifest_path, annot_dir, metadata_path, logger):
    """Return one dict per test mouse with chronologically sorted arrays,
    contiguous chunks, and ground-truth seizure intervals."""
    npz    = np.load(npz_path, allow_pickle=True)
    y_true = np.asarray(npz["y_true"]).astype(np.int64)
    y_prob = np.asarray(npz["y_prob"]).astype(np.float64)
    logger.info("NPZ loaded: %d segments | %d positive", len(y_true), int(y_true.sum()))

    mouse_names, mouse_codes, labels, chrono_idx, t_start = stream_test_chronology(
        manifest_path, len(y_true), logger)

    n_mismatch = int(np.sum(labels != y_true))
    if n_mismatch:
        raise ValueError("%d labels disagree between NPZ and manifest." % n_mismatch)

    keep = chrono_idx >= 0
    logger.info("Records with chronology: %d (%d dropped)",
                int(keep.sum()), int((~keep).sum()))

    mouse_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mice = []
    for code, mouse_id in sorted(enumerate(mouse_names), key=lambda kv: kv[1]):
        idx = np.where(keep & (mouse_codes == code))[0]
        idx = idx[np.argsort(chrono_idx[idx])]
        rec_start = datetime.datetime.fromisoformat(
            mouse_metadata[mouse_id]["recording_start_dt"])
        seizures = load_annotations(annot_dir / f"{mouse_id}_xlsx.xlsx", rec_start)
        mice.append({
            "mouse_id":    mouse_id,
            "t_start":     t_start[idx],
            "y_prob":      y_prob[idx],
            "chunks":      split_into_chunks(chrono_idx[idx]),
            "seizures":    seizures,
            "n_non_ictal": int(np.sum(y_true[idx] == 0)),
        })
    logger.info("Prepared %d mice | %d ground-truth seizures",
                len(mice), sum(len(m["seizures"]) for m in mice))
    return mice


def config_name(enabled):
    """'S+M+R+X' for the full pipeline; 'none' with every stage off."""
    on = [s for s in STAGES if enabled[s]]
    return "+".join(on) if on else "none"


def evaluate_config(mice, enabled):
    """Event-level evaluation of one on/off configuration."""
    smoothing_win  = SMOOTHING_ON  if enabled["S"] else SMOOTHING_OFF
    min_sec        = MIN_SEC_ON    if enabled["M"] else MIN_SEC_OFF
    refractory_sec = REFRACTORY_ON if enabled["R"] else REFRACTORY_OFF
    max_sec        = MAX_SEC_ON    if enabled["X"] else MAX_SEC_OFF

    tp = fp = fn = n_predicted = n_non_ictal = 0
    latencies = []
    eval_utils.SMOOTHING_WIN  = smoothing_win
    eval_utils.REFRACTORY_SEC = refractory_sec
    try:
        for mouse in mice:
            predicted = []
            for chunk_id, (s, e) in enumerate(mouse["chunks"]):
                events, _, _ = detect_events_in_chunk(
                    mouse["t_start"][s:e], mouse["y_prob"][s:e],
                    mouse["mouse_id"], chunk_id, order=ORDER,
                    min_event_duration_sec=min_sec,
                    max_event_duration_sec=max_sec)
                predicted.extend(events)
            m_tp, m_fp, m_fn, m_latencies = match_events_to_ground_truth(
                predicted, mouse["seizures"])
            tp += len(m_tp); fp += len(m_fp); fn += len(m_fn)
            n_predicted += len(predicted)
            n_non_ictal += mouse["n_non_ictal"]
            latencies.extend(m_latencies)
    finally:
        eval_utils.SMOOTHING_WIN  = SMOOTHING_ON
        eval_utils.REFRACTORY_SEC = REFRACTORY_ON

    non_ictal_hours = n_non_ictal * eval_utils.STEP_SEC / 3600.0
    far_per_hour    = safe_div(fp, non_ictal_hours)
    return {
        "config":             config_name(enabled),
        "smoothing":          int(enabled["S"]),
        "min_duration":       int(enabled["M"]),
        "refractory":         int(enabled["R"]),
        "max_duration":       int(enabled["X"]),
        "smoothing_win":      smoothing_win,
        "min_event_sec":      min_sec,
        "refractory_sec":     refractory_sec,
        "max_event_sec":      max_sec,
        "n_predicted_events": n_predicted,
        "tp":                 tp,
        "fp":                 fp,
        "fn":                 fn,
        "event_precision":    round(safe_div(tp, tp + fp), 6),
        "event_recall":       round(safe_div(tp, tp + fn), 6),
        "event_f1":           round(safe_div(2 * tp, 2 * tp + fp + fn), 6),
        "event_far_per_hour": round(far_per_hour, 6),
        "event_far_per_24h":  round(far_per_hour * 24.0, 6),
        "event_mean_latency_sec": (round(float(np.mean(latencies)), 4)
                                   if latencies else None),
    }


def build_paper_rows(rows):
    """Select the 10 reported configurations and add deltas versus full."""
    by_name = {r["config"]: r for r in rows}
    full    = by_name["+".join(STAGES)]

    selected = [("no_post_processing", "none")]
    selected += [(f"only_{STAGE_LABEL[s]}", s) for s in STAGES]
    selected += [(f"full_minus_{STAGE_LABEL[s]}",
                  "+".join(x for x in STAGES if x != s)) for s in STAGES]
    selected += [("full_pipeline", full["config"])]

    paper_rows = []
    for role, name in selected:
        r = dict(by_name[name])
        out = {"role": role}
        out.update(r)
        for metric in ("event_precision", "event_recall", "event_f1",
                       "event_far_per_hour"):
            out[f"delta_{metric}_vs_full"] = round(r[metric] - full[metric], 6)
        out["delta_fp_vs_full"] = r["fp"] - full["fp"]
        out["delta_fn_vs_full"] = r["fn"] - full["fn"]
        paper_rows.append(out)
    return paper_rows


def write_csv(out_path, rows, logger):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved %s (%d rows)", out_path, len(rows))


def main():
    args = parse_args()
    variants, _, test_annot, _, metadata, manifest, _ = variant_config(
        args.local_root, args.cluster_root, args.cluster)
    vcfg       = variants[VARIANT]
    output_dir = args.output_dir or vcfg["output"].parent / "postproc_ablation"
    logger, log_path = setup_logging(output_dir)

    logger.info("=" * 65)
    logger.info("postproc_ablation.py")
    logger.info("Model      : %s", vcfg["model_label"])
    logger.info("Partition  : test")
    logger.info("Order      : %s", ORDER)
    logger.info("Threshold  : %.2f", eval_utils.THRESHOLD)
    logger.info("Predictions: %s", vcfg["test_npz"])
    logger.info("Manifest   : %s", manifest)
    logger.info("Annotations: %s", test_annot)
    logger.info("Output     : %s", output_dir)
    logger.info("Log        : %s", log_path)
    logger.info("=" * 65)

    mice = load_test_mice(vcfg["test_npz"], manifest, test_annot, metadata, logger)

    # itertools.product yields all-stages-on first, so the guard below fails
    # fast and before any result file exists.
    rows = []
    for flags in itertools.product((True, False), repeat=len(STAGES)):
        enabled = dict(zip(STAGES, flags))
        row = evaluate_config(mice, enabled)
        rows.append(row)
        logger.info("%-8s | Pred=%6d TP=%3d FP=%6d FN=%3d | Prec=%.4f Rec=%.4f "
                    "F1=%.4f | FAR/hr=%.4f | Latency=%s",
                    row["config"], row["n_predicted_events"],
                    row["tp"], row["fp"], row["fn"],
                    row["event_precision"], row["event_recall"],
                    row["event_f1"], row["event_far_per_hour"],
                    row["event_mean_latency_sec"])
        if all(flags):
            observed = {k: row[k] for k in EXPECTED_FULL}
            if observed != EXPECTED_FULL:
                logger.error("Full pipeline %s does not reproduce Table 4 %s. "
                             "Aborting; no results written.",
                             observed, EXPECTED_FULL)
                sys.exit(1)
            logger.info("Reproducibility guard PASSED: %s", observed)

    write_csv(output_dir / "ablation_all_configs.csv", rows, logger)
    write_csv(output_dir / "ablation_paper_table.csv", build_paper_rows(rows), logger)

    logger.info("=" * 65)
    logger.info("DONE")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
