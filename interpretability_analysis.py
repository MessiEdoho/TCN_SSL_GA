"""
interpretability_analysis.py
=============================
Post-training interpretability analysis for all four
EEG seizure detection models.

Two execution modes:

--partition val   (development phase)
  Runs on the validation partition. Used before the
  test set is touched to verify the pipeline and
  diagnose any issues with interpretability results.
  Results carry indirect optimisation bias from early
  stopping, threshold selection, and hyperparameter
  tuning on the validation partition. Do NOT report
  these results as the paper's final interpretability
  findings.
  Output: outputs/interpretability/val/

--partition test  (paper reporting phase)
  Runs on the independent test partition. Produces
  the figures and statistics that appear in the paper.
  Requires data_splits.json test_status == "complete".
  Exits with error code 1 if test data is not ready.
  Output: outputs/interpretability/test/

Three interpretability layers:

Layer 1 -- Temporal attention saliency (M2, M4)
  Uses get_attention_weights() -- no post-hoc methods.
  Figures A B C D E

Layer 2 -- Branch contribution analysis (M3, M4)
  Branch ablation + temporal activation profiles.
  Figures F G H L

Layer 3 -- Feature map analysis (all models)
  1-D GradCAM and occlusion sensitivity.
  Figures I J K

Usage
-----
python interpretability_analysis.py --partition val
python interpretability_analysis.py --partition test

Prerequisites
-------------
All training scripts must have completed:
  TCN.py
  TCNTemporalAttention.py
  MultiScaleTCN.py
  MultiScaleTCNAttention.py  (M4 -- optional)
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import argparse                      # command-line --partition flag parsing
import json                          # read/write JSON config and results files
import logging                       # structured logging to file and stdout
import sys                           # stdout handle for StreamHandler
import csv                           # write CSV results (reserved for future use)
import datetime                      # ISO timestamps in saved JSON records
from pathlib import Path             # cross-platform path operations throughout

import numpy as np                   # array operations for attention weights, saliency maps
import torch                         # deep learning framework (model inference, GPU)
import torch.nn as nn                # neural network modules (not used directly here but needed for type hints)
import torch.nn.functional as F      # ReLU for GradCAM saliency computation

import matplotlib                    # plotting backend configuration
matplotlib.use("Agg")               # non-interactive backend -- must be set before pyplot import
import matplotlib.pyplot as plt      # figure creation for all 12 interpretability plots
import matplotlib.gridspec as gridspec  # advanced subplot layout (reserved for complex panels)
import seaborn as sns                # violin plots (Figure D) and heatmaps (Figure H)
from scipy import stats              # Mann-Whitney U test for entropy comparison (Figure D)
from sklearn.metrics import f1_score # macro F1 for branch ablation evaluation (Layer 2)

from tcn_utils import (
    set_seed,                        # fix Python/NumPy/PyTorch seeds for reproducibility
    TCN,                             # M1: single-branch TCN baseline
    TCNWithAttention,                # M2: TCN + two-layer additive temporal attention
    MultiScaleTCN,                   # M3: three parallel branches with distinct dilation schedules
    make_loader,                     # build DataLoader: WeightedRandomSampler (train) or sequential (val/test)
    evaluate,                        # inference pass returning (macro_f1, y_true, y_pred)
    count_parameters,                # sum of requires_grad=True parameter elements
)

# M4 (MultiScaleTCNWithAttention) may not be available if the class has not
# yet been appended to tcn_utils.py. Graceful fallback skips M4 analyses.
try:
    from tcn_utils import MultiScaleTCNWithAttention  # M4: multi-scale + temporal attention
    M4_AVAILABLE = True                               # flag: M4 class exists
except ImportError:
    M4_AVAILABLE = False                              # flag: skip M4 analyses


# ---------------------------------------------------------------------------
# Constants (partition-independent only)
# ---------------------------------------------------------------------------
SEED           = 42                   # global reproducibility seed (Python, NumPy, PyTorch CPU+CUDA)
FS             = 500                  # native EDF sampling rate (Hz)
SEGMENT_LEN    = 2500                 # samples per segment: 5 s * 500 Hz
SEGMENT_SEC    = 5.0                  # segment duration in seconds (= SEGMENT_LEN / FS)
STEP_SEC       = 2.5                  # step between segment starts: 50% overlap
GRADCAM_BATCH  = 16                   # batch size for GradCAM and occlusion forward passes
OCCLUSION_WIN  = 50                   # occlusion mask width in samples (= 0.1 s at 500 Hz)
OCCLUSION_STEP = 25                   # occlusion mask stride in samples (= 0.05 s)
N_EXAMPLE_SEGS = 8                    # number of TP and FN examples in Figure C heatmap grid
# Maximum possible entropy for a uniform attention distribution over T time steps.
# H_max = ln(T) = ln(2500) = 7.824 nats. Used as reference line in Figure D.
MAX_ENTROPY    = np.log(SEGMENT_LEN)

INTERP_BASE    = Path("outputs") / "interpretability"  # partition suffix appended in main()

# Model weights -- produced by the four training scripts.
# Exact filenames match the training script output paths.
M1_WEIGHTS = Path("outputs/TCN/tcn_final_weights.pt")                                         # M1: TCN baseline
M2_WEIGHTS = Path("outputs/TCNAttention/tcn_attention_final_weights.pt")                       # M2: TCN + attention
M3_WEIGHTS = Path("outputs/MultiScaleTCN/multiscale_tcn_final_weights.pt")                     # M3: multi-scale TCN
M4_WEIGHTS = Path("outputs/MultiScaleTCNAttention/multiscale_tcn_attention_final_weights.pt")   # M4: multi-scale + attention

# Config JSON paths -- produced by tuning scripts and notebooks
SPLITS_PATH_PRIMARY = Path("data_splits_outputs") / "data_splits.json"  # primary location from generate_data_splits.py
SPLITS_PATH_ALT     = Path("outputs") / "data_splits.json"             # fallback if user moved file
BEST_TCN_PATH       = Path("outputs") / "best_params.json"             # TCN backbone HPs from tcn_HPT_binary.ipynb
BEST_MS_PATH        = Path("outputs") / "best_multiscale_params.json"  # MultiScaleTCN HPs from tune_multiscale_tcn.py
BEST_ATTN_PATH      = Path("outputs") / "best_attention_params.json"   # attention HPs from tune_temporal_attention.py
BEST_MS_ATTN_PATH   = Path("outputs") / "best_multiscale_attn_params.json"  # MS attention HPs from tune_multiscale_attention.py

# Fallback dilation schedules if branch_dilations not found in JSON
# (must match tune_multiscale_tcn.py)
DEFAULT_BRANCH1 = [1, 2, 4]           # fine:         spike morphology
DEFAULT_BRANCH2 = [8, 16, 32]         # intermediate: rhythmic bursts
DEFAULT_BRANCH3 = [32, 64, 128]       # coarse:       seizure evolution


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------
def parse_args():
    """Parse command-line arguments. Returns Namespace with partition='val' or 'test'."""
    parser = argparse.ArgumentParser(
        description=("Interpretability analysis for EEG seizure detection models. "
                     "Use --partition val during development. "
                     "Use --partition test for paper reporting."))
    parser.add_argument(
        "--partition", choices=["val", "test"], required=True,
        help=("val: development phase (indirect optimisation bias). "
              "test: paper reporting phase (requires test_status complete)."))
    return parser.parse_args()


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------
def setup_logging(interp_root, partition):
    """Create partition-specific output directories and configure logger."""
    for subdir in ["figures", "attention_weights", "entropy", "branch_activations",
                   "branch_ablation", "gradcam", "occlusion", "logs"]:
        (interp_root / subdir).mkdir(parents=True, exist_ok=True)

    log_file = interp_root / "logs" / ("interpretability_%s.log" % partition)
    logger = logging.getLogger("interpretability_%s" % partition)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)

    logger.info("=" * 65)
    logger.info("interpretability_analysis.py -- partition: %s", partition.upper())
    logger.info("Timestamp: %s", datetime.datetime.now().isoformat())
    if partition == "val":
        logger.warning("PARTITION: VALIDATION -- development phase.")
        logger.warning("Validation partition carries indirect optimisation bias: "
                       "early stopping monitored val F1, threshold selected by Youden J on val, "
                       "hyperparameters tuned on val F1.")
        logger.warning("DO NOT report these results as the paper's final interpretability findings. "
                       "Run with --partition test for paper reporting.")
    else:
        logger.info("PARTITION: TEST -- paper reporting phase.")
        logger.info("Results on this partition are unbiased and suitable for publication.")
    logger.info("=" * 65)
    return logger


# ---------------------------------------------------------------------------
# load_model_weights
# ---------------------------------------------------------------------------
def load_model_weights(model_name, weights_path, hp, device, logger,
                       attn_hp=None, branch_dilations=None):
    """Instantiate model class and load trained weights.

    Parameters
    ----------
    model_name       : 'M1', 'M2', 'M3', or 'M4'
    weights_path     : Path to .pt state dict
    hp               : backbone hyperparameters dict
    device           : torch.device
    logger           : logging.Logger
    attn_hp          : attention hyperparameters (M2, M4)
    branch_dilations : branch dilation schedules (M3, M4)
    """
    if not weights_path.exists():
        logger.error("%s weights not found: %s", model_name, weights_path)
        raise FileNotFoundError(str(weights_path))

    if model_name == "M1":
        model = TCN(
            num_layers=int(hp["num_layers"]), num_filters=int(hp["num_filters"]),
            kernel_size=int(hp["kernel_size"]), dropout=float(hp["dropout"]), fs=FS)
    elif model_name == "M2":
        model = TCNWithAttention(
            num_layers=int(hp["num_layers"]), num_filters=int(hp["num_filters"]),
            kernel_size=int(hp["kernel_size"]), dropout=float(hp["dropout"]),
            attention_dim=int(attn_hp["attention_dim"]),
            attention_dropout=float(attn_hp["attention_dropout"]), fs=FS)
    elif model_name == "M3":
        bd = branch_dilations or {"branch1": DEFAULT_BRANCH1, "branch2": DEFAULT_BRANCH2, "branch3": DEFAULT_BRANCH3}
        model = MultiScaleTCN(
            num_filters=int(hp["num_filters"]), kernel_size=int(hp["kernel_size"]),
            dropout=float(hp["dropout"]),
            branch1_dilations=bd["branch1"], branch2_dilations=bd["branch2"],
            branch3_dilations=bd["branch3"], fusion=str(hp["fusion"]))
    elif model_name == "M4":
        if not M4_AVAILABLE:
            logger.warning("MultiScaleTCNWithAttention not available. Skipping M4.")
            return None
        bd = branch_dilations or {"branch1": DEFAULT_BRANCH1, "branch2": DEFAULT_BRANCH2, "branch3": DEFAULT_BRANCH3}
        model = MultiScaleTCNWithAttention(
            num_filters=int(hp["num_filters"]), kernel_size=int(hp["kernel_size"]),
            dropout=float(hp["dropout"]), fusion=str(hp["fusion"]),
            attention_dim=int(attn_hp["attention_dim"]),
            attention_dropout=float(attn_hp["attention_dropout"]),
            branch1_dilations=bd["branch1"], branch2_dilations=bd["branch2"],
            branch3_dilations=bd["branch3"])
    else:
        raise ValueError("Unknown model_name: %s" % model_name)

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    logger.info("Loaded %s: %s (%s params)", model_name, weights_path, "{:,}".format(count_parameters(model)))
    return model


# ---------------------------------------------------------------------------
# load_partition_data
# ---------------------------------------------------------------------------
def load_partition_data(partition, batch_size, device, logger):
    """Load file-label pairs and build DataLoader for specified partition.

    When partition=='test', validates test_status=='complete'. Raises RuntimeError if not ready.

    Returns (pairs, loader, y_true, x_all, n_ictal, n_nonictal).
    """
    if SPLITS_PATH_PRIMARY.exists():
        splits_path = SPLITS_PATH_PRIMARY
    elif SPLITS_PATH_ALT.exists():
        splits_path = SPLITS_PATH_ALT
    else:
        logger.error("data_splits.json not found at %s or %s.", SPLITS_PATH_PRIMARY, SPLITS_PATH_ALT)
        raise FileNotFoundError("data_splits.json")

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    # Validate test readiness
    if partition == "test":
        test_status = splits.get("metadata", {}).get("test_status", "pending")
        if test_status != "complete":
            logger.error("Test partition requested but test_status='%s'. "
                         "Run generate_data_splits.py --include-test first.", test_status)
            logger.error("This script will NOT fall back to the validation partition. Exiting.")
            raise RuntimeError("test_status is '%s', expected 'complete'" % test_status)

    records = splits[partition]
    pairs = [(r["filepath"], r["label"]) for r in records]
    if not pairs:
        raise RuntimeError("Partition '%s' is empty." % partition)

    loader = make_loader(pairs, batch_size, False, device)
    y_true = np.array([p[1] for p in pairs], dtype=int)

    # Load all segments into memory for occlusion
    x_all_list = []
    with torch.no_grad():
        for x_batch, _ in loader:
            x_all_list.append(x_batch.cpu().numpy())
    x_all = np.concatenate(x_all_list, axis=0)

    n_ictal = int((y_true == 1).sum())
    n_nonictal = int((y_true == 0).sum())
    logger.info("Partition: %s | Total: %d | Ictal: %d (%.1f%%) | Non-ictal: %d",
                partition.upper(), len(pairs), n_ictal, 100 * n_ictal / len(pairs), n_nonictal)
    return pairs, loader, y_true, x_all, n_ictal, n_nonictal


# ---------------------------------------------------------------------------
# collect_attention_weights
# ---------------------------------------------------------------------------
def collect_attention_weights(model, loader, y_true, device, model_name,
                              partition, interp_root, logger):
    """Extract alpha_t attention weights for all segments using get_attention_weights().

    Returns (weights_ictal, weights_nonictal) each shape (N_class, T).
    Saves arrays to interp_root/attention_weights/.
    """
    model.eval()                                       # disable dropout for deterministic attention weights
    all_weights = []                                   # accumulate per-batch attention arrays
    with torch.no_grad():                              # no gradients needed -- pure inference
        for x, _ in loader:                            # iterate over all partition batches
            x = x.to(device)                           # transfer input to GPU
            w = model.get_attention_weights(x)         # np.ndarray (batch, T) -- softmax-normalised
            all_weights.append(w)                      # collect batch results
    weights = np.concatenate(all_weights, axis=0)      # (N, T) -- all segments concatenated
    labels = y_true                                    # ground truth labels for class separation

    w_ictal = weights[labels == 1]                     # attention maps for seizure segments
    w_nonictal = weights[labels == 0]                  # attention maps for normal segments

    # Save arrays with partition suffix to prevent val/test overwrite
    out_dir = interp_root / "attention_weights"
    pfx = model_name.lower()                           # e.g. "m2" or "m4"
    np.save(out_dir / ("%s_attn_weights_ictal_%s.npy" % (pfx, partition)), w_ictal)
    np.save(out_dir / ("%s_attn_weights_nonictal_%s.npy" % (pfx, partition)), w_nonictal)
    logger.info("%s attention weights: %d ictal, %d nonictal segments",
                model_name, w_ictal.shape[0], w_nonictal.shape[0])
    return w_ictal, w_nonictal


# ---------------------------------------------------------------------------
# compute_entropy
# ---------------------------------------------------------------------------
def compute_entropy(weights, logger):
    """Compute Shannon attention entropy H(alpha) = -sum(alpha * ln(alpha)) per segment.

    Returns np.ndarray (N,) in nats. H_max = ln(T) for uniform attention.
    """
    # Clip to prevent log(0) from numerical underflow when softmax assigns near-zero
    # probability to many time steps in long sequences (T=2500)
    w_clip = np.clip(weights, 1e-10, 1.0)
    # Shannon entropy: H = -sum(alpha * ln(alpha)). Low H = focused, high H = diffuse.
    # Because sum(alpha)=1 (softmax), mean alpha = 1/T is constant across all segments.
    # Entropy measures the shape (concentration) of the distribution, not its sum.
    entropy = -np.sum(w_clip * np.log(w_clip), axis=1)  # (N,) in nats
    logger.info("  Entropy range: [%.4f, %.4f] nats (H_max=%.4f)", entropy.min(), entropy.max(), MAX_ENTROPY)
    return entropy


# ---------------------------------------------------------------------------
# run_branch_ablation
# ---------------------------------------------------------------------------
def run_branch_ablation(model, loader, y_true, device, model_name,
                        partition, interp_root, logger):
    """Measure each branch's F1 contribution by zeroing its output before fusion.

    Returns dict with baseline_f1, branch{1,2,3}_ablated_f1, contributions, pairwise F1s.
    """
    if partition == "val":
        logger.warning("%s branch ablation on VALIDATION partition -- F1 values carry "
                       "threshold-selection bias. Use --partition test for paper results.", model_name)

    def _eval_ablated(zeroed_branches):
        """Evaluate model with specified branches zeroed before fusion. Returns macro F1.

        This bypasses model.forward() and manually computes branch outputs, fusion,
        and pooling so that individual branches can be selectively zeroed.
        """
        model.eval()                                   # disable dropout for deterministic evaluation
        all_true, all_pred = [], []                    # accumulate labels and predictions
        with torch.no_grad():                          # no gradients needed
            for x, y in loader:                        # iterate over partition batches
                x = x.to(device)                       # transfer input to GPU
                # For M4, branches are inside self.backbone; for M3, directly on model
                bb = model.backbone if hasattr(model, "backbone") else model
                o1 = bb.branch1(x)                     # (batch, num_filters, T) -- fine scale
                o2 = bb.branch2(x)                     # (batch, num_filters, T) -- medium scale
                o3 = bb.branch3(x)                     # (batch, num_filters, T) -- coarse scale
                # Zero the specified branches to measure their contribution
                if 1 in zeroed_branches:
                    o1 = torch.zeros_like(o1)          # ablate fine-scale branch
                if 2 in zeroed_branches:
                    o2 = torch.zeros_like(o2)          # ablate medium-scale branch
                if 3 in zeroed_branches:
                    o3 = torch.zeros_like(o3)          # ablate coarse-scale branch
                # Fuse branch outputs using the model's learned fusion strategy
                if bb.fusion == "concat":              # concat + 1x1 Conv1d projection
                    fused = torch.cat([o1, o2, o3], dim=1)  # (batch, 3*F, T)
                    fused = bb.fusion_conv(fused)      # (batch, F, T)
                else:                                  # element-wise average
                    fused = (o1 + o2 + o3) / 3.0       # (batch, F, T)
                # Pooling: temporal attention (M4) or global average pooling (M3)
                if hasattr(model, "attention_fc"):     # M4 has attention layers on the model itself
                    feat_t = fused.transpose(1, 2)     # (batch, T, F) for attention
                    e = torch.tanh(model.attention_fc(feat_t))  # tanh energy scoring
                    e = model.attention_v(e)           # scalar score per time step
                    alpha = torch.softmax(e, dim=1)    # softmax over T
                    context = (feat_t * alpha).sum(dim=1)  # attention-weighted sum
                    context = model.attention_drop(context)  # dropout regularisation
                    logits = model.classifier(context).squeeze(-1)  # binary logit
                else:                                  # M3 uses global average pooling
                    pooled = fused.mean(dim=-1)        # GAP over time
                    logits = bb.classifier(pooled).squeeze(-1)  # binary logit
                preds = (torch.sigmoid(logits) >= 0.5).long()  # binarise at threshold 0.5
                all_true.extend(y.numpy())             # accumulate ground truth
                all_pred.extend(preds.cpu().numpy())   # accumulate predictions
        return f1_score(np.array(all_true), np.array(all_pred), average="macro", zero_division=0)

    # Single-branch ablation: zero one branch, keep the other two
    baseline = _eval_ablated([])                       # no ablation -- full model baseline F1
    f1_b1 = _eval_ablated([1])                         # zero branch 1 (fine scale)
    f1_b2 = _eval_ablated([2])                         # zero branch 2 (medium scale)
    f1_b3 = _eval_ablated([3])                         # zero branch 3 (coarse scale)
    # Pairwise ablation: keep two branches, zero the third
    f1_p12 = _eval_ablated([3])                        # keep B1+B2, zero B3
    f1_p13 = _eval_ablated([2])                        # keep B1+B3, zero B2
    f1_p23 = _eval_ablated([1])                        # keep B2+B3, zero B1

    result = {
        "model_name": model_name, "partition": partition,
        "baseline_f1": round(baseline, 6),
        "branch1_ablated_f1": round(f1_b1, 6), "branch1_contribution": round(baseline - f1_b1, 6),
        "branch2_ablated_f1": round(f1_b2, 6), "branch2_contribution": round(baseline - f1_b2, 6),
        "branch3_ablated_f1": round(f1_b3, 6), "branch3_contribution": round(baseline - f1_b3, 6),
        "pair12_f1": round(f1_p12, 6), "pair13_f1": round(f1_p13, 6), "pair23_f1": round(f1_p23, 6),
    }
    out_dir = interp_root / "branch_ablation"
    with open(out_dir / ("%s_ablation_%s.json" % (model_name.lower(), partition)), "w") as f:
        json.dump(result, f, indent=2)
    logger.info("%s ablation: baseline=%.4f | B1=%.4f B2=%.4f B3=%.4f",
                model_name, baseline, f1_b1, f1_b2, f1_b3)
    return result


# ---------------------------------------------------------------------------
# collect_branch_activations
# ---------------------------------------------------------------------------
def collect_branch_activations(model, loader, y_true, device, model_name,
                               partition, interp_root, logger):
    """Compute mean absolute activation per branch per time step.

    Returns (act_ictal, act_nonictal) each shape (3, N_class, T).
    """
    model.eval()
    all_acts = {1: [], 2: [], 3: []}
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            bb = model.backbone if hasattr(model, "backbone") else model
            for bi, branch in enumerate([bb.branch1, bb.branch2, bb.branch3], 1):
                out = branch(x)                    # (batch, num_filters, T)
                act = out.abs().mean(dim=1)        # mean over filters -> (batch, T)
                all_acts[bi].append(act.cpu().numpy())
    acts = np.stack([np.concatenate(all_acts[b], axis=0) for b in [1, 2, 3]])  # (3, N, T)

    mask_ic = (y_true == 1)
    act_ictal = acts[:, mask_ic, :]
    act_nonictal = acts[:, ~mask_ic, :]

    out_dir = interp_root / "branch_activations"
    pfx = model_name.lower()
    np.save(out_dir / ("%s_branch_act_ictal_%s.npy" % (pfx, partition)), act_ictal)
    np.save(out_dir / ("%s_branch_act_nonictal_%s.npy" % (pfx, partition)), act_nonictal)
    logger.info("%s branch activations: ictal=%d, nonictal=%d", model_name, act_ictal.shape[1], act_nonictal.shape[1])
    return act_ictal, act_nonictal


# ---------------------------------------------------------------------------
# compute_gradcam_1d
# ---------------------------------------------------------------------------
def compute_gradcam_1d(model, loader, y_true, device, model_name,
                       partition, interp_root, logger):
    """Compute 1-D GradCAM saliency for all segments.

    Target layer: last conv block of the backbone conv stack.
    Returns (gradcam_ictal, gradcam_nonictal) each shape (N_class, T).
    """
    # Identify the target convolutional layer for GradCAM.
    # We use the LAST conv layer (deepest features, most integrated representation).
    # For multi-branch models, branch3 has the largest RF (coarse scale).
    if model_name in ("M1",):
        target_layer = model.network[-1].conv2        # M1: last CausalConvBlock, 2nd conv
    elif model_name in ("M2",):
        target_layer = model.tcn[-1].conv2            # M2: backbone TCN, last block, 2nd conv
    elif model_name in ("M3",):
        target_layer = model.branch3[-1].conv2        # M3: coarse branch, last block
    elif model_name in ("M4",) and model is not None:
        target_layer = model.backbone.branch3[-1].conv2  # M4: backbone coarse branch
    else:
        logger.warning("GradCAM: unknown model %s, skipping.", model_name)
        return np.array([]), np.array([])

    all_cam = []                                       # accumulate per-batch GradCAM arrays
    model.eval()                                       # disable dropout but keep grad tracking
    for x_batch, _ in loader:
        x_batch = x_batch.to(device).requires_grad_(True)  # enable gradients on input
        activation, gradient = {}, {}                  # dicts to capture hook outputs

        # Forward hook: captures the activation map A at the target layer
        def fwd_hook(m, inp, out):
            activation["val"] = out

        # Backward hook: captures the gradient dL/dA flowing back through the target layer
        def bwd_hook(m, gin, gout):
            gradient["val"] = gout[0]

        # Register hooks -- MUST be removed after each batch to prevent accumulation
        h_fwd = target_layer.register_forward_hook(fwd_hook)
        h_bwd = target_layer.register_full_backward_hook(bwd_hook)

        output = model(x_batch)                        # forward pass through full model
        loss = output.mean()                           # scalar for backward (mean over batch)
        model.zero_grad()                              # clear any accumulated gradients
        loss.backward()                                # compute dL/dA at target layer

        h_fwd.remove()                                 # remove forward hook immediately
        h_bwd.remove()                                 # remove backward hook immediately

        # GradCAM formula: w^k = mean_t(dL/dA^k(t)), then L_cam(t) = ReLU(sum_k w^k * A^k(t))
        A = activation["val"]                          # (batch, C, T) -- feature maps
        G = gradient["val"]                            # (batch, C, T) -- gradients
        w = G.mean(dim=-1, keepdim=True)               # (batch, C, 1) -- channel importance weights
        cam = F.relu((w * A).sum(dim=1))               # (batch, T) -- ReLU to keep only positive influence
        # Normalise each segment's CAM to [0, 1] for comparable cross-segment visualisation
        cam_max = cam.max(dim=-1, keepdim=True)[0].clamp(min=1e-8)  # avoid div-by-zero
        cam = cam / cam_max                            # (batch, T) in [0, 1]
        all_cam.append(cam.detach().cpu().numpy())     # detach from graph, move to CPU

    cam_all = np.concatenate(all_cam, axis=0)      # (N, T)
    mask_ic = (y_true == 1)
    gc_ictal = cam_all[mask_ic]
    gc_nonictal = cam_all[~mask_ic]

    out_dir = interp_root / "gradcam"
    pfx = model_name.lower()
    np.save(out_dir / ("%s_gradcam_ictal_%s.npy" % (pfx, partition)), gc_ictal)
    np.save(out_dir / ("%s_gradcam_nonictal_%s.npy" % (pfx, partition)), gc_nonictal)
    logger.info("%s GradCAM: %d ictal, %d nonictal segments", model_name, gc_ictal.shape[0], gc_nonictal.shape[0])
    return gc_ictal, gc_nonictal


# ---------------------------------------------------------------------------
# compute_occlusion_sensitivity
# ---------------------------------------------------------------------------
def compute_occlusion_sensitivity(model, x_all, y_true, device, model_name,
                                  partition, interp_root, logger):
    """Compute occlusion sensitivity S(p) = P(y=1|x) - P(y=1|x_masked).

    Returns (occ_ictal, occ_nonictal) each shape (N_class, T).
    """
    model.eval()                                       # disable dropout for deterministic probabilities
    N, C, T = x_all.shape                              # N segments, C=1 channel, T=2500 time steps
    # Compute all mask positions: slide a zero-window across the segment
    positions = list(range(0, T - OCCLUSION_WIN + 1, OCCLUSION_STEP))  # start positions
    all_sens = np.zeros((N, len(positions)), dtype=np.float32)  # (N, n_positions)

    with torch.no_grad():                              # no gradients -- pure inference
        # Step 1: compute baseline P(y=1|x) for all segments (unmasked)
        baseline_probs = []
        for i in range(0, N, GRADCAM_BATCH):           # process in mini-batches for memory
            xb = torch.from_numpy(x_all[i:i + GRADCAM_BATCH]).to(device)
            baseline_probs.append(torch.sigmoid(model(xb)).cpu().numpy())  # sigmoid -> [0,1]
        baseline_probs = np.concatenate(baseline_probs)  # (N,) -- one probability per segment

        # Step 2: for each mask position, zero the window and measure prob drop
        for pi, pos in enumerate(positions):
            x_masked = x_all.copy()                    # copy to avoid modifying original
            x_masked[:, :, pos:pos + OCCLUSION_WIN] = 0.0  # zero out the window
            masked_probs = []
            for i in range(0, N, GRADCAM_BATCH):       # batch inference on masked input
                xb = torch.from_numpy(x_masked[i:i + GRADCAM_BATCH]).to(device)
                masked_probs.append(torch.sigmoid(model(xb)).cpu().numpy())
            masked_probs = np.concatenate(masked_probs)  # (N,)
            # S(p) = P(y=1|x) - P(y=1|x_masked). Positive = region was important.
            all_sens[:, pi] = baseline_probs - masked_probs
            if (pi + 1) % 20 == 0:                     # log progress periodically
                logger.debug("%s occlusion: %d/%d positions", model_name, pi + 1, len(positions))

    # Step 3: interpolate position-level sensitivity back to full T-length time axis
    from scipy.interpolate import interp1d             # lazy import -- only needed here
    pos_centers = np.array(positions) + OCCLUSION_WIN / 2.0  # center of each mask window
    t_axis = np.arange(T)                              # full time axis [0, 1, ..., T-1]
    occ_full = np.zeros((N, T), dtype=np.float32)      # interpolated output
    for i in range(N):                                 # per-segment interpolation
        fn = interp1d(pos_centers, all_sens[i], kind="linear", fill_value="extrapolate")
        occ_full[i] = fn(t_axis)                       # interpolate to every time step

    mask_ic = (y_true == 1)
    occ_ictal = occ_full[mask_ic]
    occ_nonictal = occ_full[~mask_ic]

    out_dir = interp_root / "occlusion"
    pfx = model_name.lower()
    np.save(out_dir / ("%s_occlusion_ictal_%s.npy" % (pfx, partition)), occ_ictal)
    np.save(out_dir / ("%s_occlusion_nonictal_%s.npy" % (pfx, partition)), occ_nonictal)
    logger.info("%s occlusion: %d ictal, %d nonictal", model_name, occ_ictal.shape[0], occ_nonictal.shape[0])
    return occ_ictal, occ_nonictal


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _time_axis():
    """Return time axis in seconds for T=SEGMENT_LEN at FS Hz."""
    return np.arange(SEGMENT_LEN) / FS

def _save_fig(fig, path, logger):
    """tight_layout -> savefig -> close -> log."""
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


# ---------------------------------------------------------------------------
# plot_figure_A -- Mean attention profile by class
# ---------------------------------------------------------------------------
def plot_figure_A(attn_data, partition, interp_root, logger):
    """Mean alpha_t +/- std for ictal vs non-ictal, one subplot per model (M2, M4)."""
    fig_dir = interp_root / "figures"
    time_s = _time_axis()
    n_models = len(attn_data)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 4), squeeze=False)
    for idx, (mname, w_ic, w_nic) in enumerate(attn_data):
        ax = axes[0, idx]
        m_ic, s_ic = w_ic.mean(0), w_ic.std(0)
        m_nic, s_nic = w_nic.mean(0), w_nic.std(0)
        ax.plot(time_s, m_ic, color="#C85A5A", lw=1.2, label="Ictal")
        ax.fill_between(time_s, m_ic - s_ic, m_ic + s_ic, color="#C85A5A", alpha=0.25)
        ax.plot(time_s, m_nic, color="#41B3A3", lw=1.2, label="Non-ictal")
        ax.fill_between(time_s, m_nic - s_nic, m_nic + s_nic, color="#41B3A3", alpha=0.25)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Mean alpha_t")
        ax.set_title("%s Mean Attention [%s]" % (mname, partition.upper())); ax.legend(fontsize=8)
    _save_fig(fig, fig_dir / ("fig_A_mean_attention_profile_%s.png" % partition), logger)


# ---------------------------------------------------------------------------
# plot_figure_B -- Differential attention
# ---------------------------------------------------------------------------
def plot_figure_B(attn_data, partition, interp_root, logger):
    """Delta alpha_t = mean(ictal) - mean(non-ictal) with signed fill."""
    fig_dir = interp_root / "figures"
    time_s = _time_axis()
    n = len(attn_data)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    for idx, (mname, w_ic, w_nic) in enumerate(attn_data):
        ax = axes[0, idx]
        diff = w_ic.mean(0) - w_nic.mean(0)
        ax.fill_between(time_s, 0, diff, where=(diff >= 0), color="#C85A5A", alpha=0.4, label="Ictal > Non-ictal")
        ax.fill_between(time_s, 0, diff, where=(diff < 0), color="#41B3A3", alpha=0.4, label="Non-ictal > Ictal")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Delta alpha_t")
        ax.set_title("%s Differential Attention [%s]" % (mname, partition.upper())); ax.legend(fontsize=8)
    _save_fig(fig, fig_dir / ("fig_B_differential_attention_%s.png" % partition), logger)


# ---------------------------------------------------------------------------
# plot_figure_C -- Attention heatmap grid (M2 only)
# ---------------------------------------------------------------------------
def plot_figure_C(model_m2, loader, y_true, x_all, device, partition, interp_root, logger):
    """4x4 grid: 8 TP + 8 FN panels with EEG + alpha_t overlay."""
    fig_dir = interp_root / "figures"
    model_m2.eval()
    # Collect predictions and attention weights
    all_probs, all_weights = [], []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            all_probs.extend(torch.sigmoid(model_m2(x)).cpu().numpy())
            all_weights.append(model_m2.get_attention_weights(x))
    probs = np.array(all_probs)
    weights = np.concatenate(all_weights, axis=0)
    preds = (probs >= 0.5).astype(int)

    ic_mask = (y_true == 1)
    tp_mask = ic_mask & (preds == 1)
    fn_mask = ic_mask & (preds == 0)
    tp_idx = np.where(tp_mask)[0]
    fn_idx = np.where(fn_mask)[0]

    # Select top-N by mean attention (TP) and bottom-N (FN)
    n_tp = min(N_EXAMPLE_SEGS, len(tp_idx))
    n_fn = min(N_EXAMPLE_SEGS, len(fn_idx))
    if n_tp > 0:
        tp_order = np.argsort(-weights[tp_idx].mean(1))[:n_tp]
        tp_sel = tp_idx[tp_order]
    else:
        tp_sel = np.array([], dtype=int)
    if n_fn > 0:
        fn_order = np.argsort(weights[fn_idx].mean(1))[:n_fn]
        fn_sel = fn_idx[fn_order]
    else:
        fn_sel = np.array([], dtype=int)

    total_panels = n_tp + n_fn
    if total_panels == 0:
        logger.warning("No TP or FN segments found for Figure C."); return

    ncols = 4
    nrows = max(1, (total_panels + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes)
    time_s = _time_axis()
    all_sel = list(zip(tp_sel, ["TP"] * n_tp)) + list(zip(fn_sel, ["FN"] * n_fn))
    for pi, (si, label) in enumerate(all_sel):
        r, c = divmod(pi, ncols)
        if r >= nrows:
            break
        ax = axes[r, c]
        eeg = x_all[si, 0, :]
        eeg_norm = eeg / (np.abs(eeg).max() + 1e-8)
        ax.plot(time_s, eeg_norm, color="gray", lw=0.4, alpha=0.7)
        ax.imshow(weights[si:si + 1], aspect="auto", cmap="hot", alpha=0.5,
                  extent=[0, SEGMENT_SEC, eeg_norm.min() - 0.3, eeg_norm.min() - 0.1])
        ax.set_title("%s (mean=%.4f)" % (label, weights[si].mean()), fontsize=7)
        ax.set_xlim(0, SEGMENT_SEC)
    # Hide unused axes
    for pi in range(len(all_sel), nrows * ncols):
        r, c = divmod(pi, ncols)
        if r < nrows:
            axes[r, c].set_visible(False)
    fig.suptitle("M2 Attention Heatmap: TP vs FN [%s]" % partition.upper(), fontsize=10)
    _save_fig(fig, fig_dir / ("fig_C_attention_heatmap_grid_%s.png" % partition), logger)


# ---------------------------------------------------------------------------
# plot_figure_D -- Attention entropy violin plots
# ---------------------------------------------------------------------------
def plot_figure_D(entropy_data, partition, interp_root, logger):
    """Violin + strip of H(alpha) for ictal vs non-ictal, per model."""
    fig_dir = interp_root / "figures"
    n = len(entropy_data)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    stats_results = {}
    for idx, (mname, h_ic, h_nic) in enumerate(entropy_data):
        ax = axes[0, idx]
        data = [h_ic, h_nic]
        labels_list = ["Ictal"] * len(h_ic) + ["Non-ictal"] * len(h_nic)
        values = np.concatenate([h_ic, h_nic])
        import pandas as pd
        df = pd.DataFrame({"Class": labels_list, "H(alpha)": values})
        sns.violinplot(x="Class", y="H(alpha)", data=df, ax=ax, palette=["#C85A5A", "#41B3A3"],
                       inner="quartile", cut=0)
        ax.axhline(MAX_ENTROPY, ls="--", color="gray", lw=0.8, label="H_max=%.3f" % MAX_ENTROPY)
        # Mann-Whitney U test
        stat, pval = stats.mannwhitneyu(h_ic, h_nic, alternative="two-sided")
        effect_r = stat / (len(h_ic) * len(h_nic))
        ax.set_title("%s Entropy [%s]\np=%.2e, r=%.3f" % (mname, partition.upper(), pval, effect_r), fontsize=9)
        ax.legend(fontsize=7)
        stats_results[mname] = {"U": float(stat), "p": float(pval), "r": round(effect_r, 6),
                                "median_ictal": round(float(np.median(h_ic)), 6),
                                "median_nonictal": round(float(np.median(h_nic)), 6)}
    _save_fig(fig, fig_dir / ("fig_D_attention_entropy_%s.png" % partition), logger)

    # Save stats JSON
    with open(interp_root / "entropy" / ("entropy_stats_%s.json" % partition), "w") as f:
        json.dump(stats_results, f, indent=2)
    return stats_results


# ---------------------------------------------------------------------------
# plot_figure_E -- Example segment with attention overlay
# ---------------------------------------------------------------------------
def plot_figure_E(model_m2, x_all, y_true, device, partition, interp_root, logger):
    """Single ictal segment (highest mean alpha_t) with EEG + attention overlay."""
    fig_dir = interp_root / "figures"
    model_m2.eval()
    ic_idx = np.where(y_true == 1)[0]
    if len(ic_idx) == 0:
        logger.warning("No ictal segments for Figure E."); return
    # Get attention for all ictal
    ic_weights = []
    with torch.no_grad():
        for i in range(0, len(ic_idx), 32):
            batch_idx = ic_idx[i:i + 32]
            xb = torch.from_numpy(x_all[batch_idx]).to(device)
            ic_weights.append(model_m2.get_attention_weights(xb))
    ic_weights = np.concatenate(ic_weights, axis=0)
    best_local = np.argmax(ic_weights.mean(1))
    best_global = ic_idx[best_local]

    eeg = x_all[best_global, 0, :]
    eeg_norm = eeg / (np.abs(eeg).max() + 1e-8)
    w = ic_weights[best_local]
    time_s = _time_axis()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), gridspec_kw={"height_ratios": [65, 35]}, sharex=True)
    ax1.plot(time_s, eeg_norm, color="gray", lw=0.5)
    ax1.set_ylabel("Normalised amplitude")
    ax1.set_title("Example Ictal Segment with Attention [%s %s]" % (partition.upper(), "example"))
    ax2.imshow(w[np.newaxis, :], aspect="auto", cmap="hot", extent=[0, SEGMENT_SEC, 0, 1])
    ax2.set_xlabel("Time (s)"); ax2.set_yticks([])
    ax2.set_ylabel("alpha_t")
    _save_fig(fig, fig_dir / ("fig_E_example_segment_overlay_%s.png" % partition), logger)


# ---------------------------------------------------------------------------
# plot_figure_F -- Branch ablation bar chart
# ---------------------------------------------------------------------------
def plot_figure_F(ablation_results, branch_dilations, partition, interp_root, logger):
    """Horizontal bars: contribution per branch for M3 and M4."""
    fig_dir = interp_root / "figures"
    models = [(k, v) for k, v in ablation_results.items() if v is not None]
    if not models:
        logger.warning("No ablation results for Figure F."); return
    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = []
    labels = []
    vals = []
    colors = []
    palette = {"M3": "#7B68AE", "M4": "#41B3A3"}
    for mi, (mname, res) in enumerate(models):
        for bi in [1, 2, 3]:
            y_pos.append(mi * 4 + bi)
            bd = branch_dilations.get("branch%d" % bi, [])
            labels.append("%s B%d %s" % (mname, bi, bd))
            vals.append(res["branch%d_contribution" % bi])
            colors.append(palette.get(mname, "#5A7DC8"))
    ax.barh(y_pos, vals, color=colors, edgecolor="white", height=0.7)
    for yp, v in zip(y_pos, vals):
        ax.annotate("%.4f" % v, xy=(v, yp), xytext=(3, 0), textcoords="offset points", fontsize=7, va="center")
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("F1 Contribution (baseline - ablated)")
    ax.set_title("Branch Ablation [%s]" % partition.upper())
    if partition == "val":
        fig.text(0.5, -0.02, "Note: computed on val partition. Ablation F1 carries threshold-selection bias.",
                 ha="center", fontsize=7, style="italic")
    _save_fig(fig, fig_dir / ("fig_F_branch_ablation_bar_%s.png" % partition), logger)


# ---------------------------------------------------------------------------
# plot_figure_G -- Branch RF diagram (architectural, no data needed)
# ---------------------------------------------------------------------------
def plot_figure_G(hp_ms, branch_dilations, partition, interp_root, logger):
    """RF per branch bar chart + dilation schedule scatter."""
    fig_dir = interp_root / "figures"
    ks = int(hp_ms["kernel_size"])
    branch_names = ["branch1", "branch2", "branch3"]
    rfs = []
    blabels = []
    for bname in branch_names:
        dils = branch_dilations.get(bname, [])
        rf = 1 + 2 * sum((ks - 1) * d for d in dils)
        rfs.append(rf)
        blabels.append("B%s\n%s" % (bname[-1], dils))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 5))
    bar_colors = ["#E8A87C", "#5A7DC8", "#41B3A3"]
    yp = np.arange(3)
    ax_l.barh(yp, rfs, color=bar_colors, height=0.5)
    for i, rf in enumerate(rfs):
        ax_l.annotate("%d (%.3fs)" % (rf, rf / FS), xy=(rf, i), xytext=(5, 0),
                      textcoords="offset points", fontsize=8, va="center")
    ax_l.axvline(SEGMENT_LEN, ls="--", color="#C85A5A", lw=1.2, label="Segment=%d" % SEGMENT_LEN)
    ax_l.set_yticks(yp); ax_l.set_yticklabels(blabels, fontsize=9)
    ax_l.set_xlabel("RF (samples)"); ax_l.set_title("Receptive Field per Branch"); ax_l.legend(fontsize=8)

    for i, bname in enumerate(branch_names):
        dils = branch_dilations.get(bname, [])
        bx = list(range(1, len(dils) + 1))
        ax_r.plot(bx, dils, "-o", color=bar_colors[i], lw=1.5, label=blabels[i].replace("\n", " "))
        for j, d in enumerate(dils):
            ax_r.annotate("d=%d" % d, xy=(bx[j], d), xytext=(0, 8), textcoords="offset points",
                          fontsize=7, ha="center", color=bar_colors[i])
    ax_r.set_xlabel("Block index"); ax_r.set_ylabel("Dilation")
    ax_r.set_title("Dilation Schedule per Branch"); ax_r.legend(fontsize=7)
    _save_fig(fig, fig_dir / ("fig_G_branch_rf_diagram_%s.png" % partition), logger)


# ---------------------------------------------------------------------------
# plot_figure_H -- Pairwise ablation heatmap
# ---------------------------------------------------------------------------
def plot_figure_H(ablation_results, partition, interp_root, logger):
    """Heatmap of pairwise ablation F1 for M3 and M4."""
    fig_dir = interp_root / "figures"
    models = [(k, v) for k, v in ablation_results.items() if v is not None]
    if not models:
        logger.warning("No ablation results for Figure H."); return
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    pair_labels = ["Keep B1+B2\n(zero B3)", "Keep B1+B3\n(zero B2)", "Keep B2+B3\n(zero B1)"]
    for idx, (mname, res) in enumerate(models):
        ax = axes[0, idx]
        vals = [[res["pair12_f1"]], [res["pair13_f1"]], [res["pair23_f1"]]]
        sns.heatmap(vals, annot=True, fmt=".4f", cmap="YlGnBu", ax=ax,
                    yticklabels=pair_labels, xticklabels=[mname], cbar=False)
        ax.set_title("%s Pairwise [%s]\nBaseline=%.4f" % (mname, partition.upper(), res["baseline_f1"]), fontsize=9)
    if partition == "val":
        fig.text(0.5, -0.02, "Note: val partition -- threshold-selection bias present.", ha="center", fontsize=7, style="italic")
    _save_fig(fig, fig_dir / ("fig_H_pairwise_ablation_%s.png" % partition), logger)


# ---------------------------------------------------------------------------
# plot_figure_I -- GradCAM saliency
# ---------------------------------------------------------------------------
def plot_figure_I(gradcam_data, partition, interp_root, logger):
    """2x2 grid of mean GradCAM +/- std for ictal and non-ictal, per model."""
    fig_dir = interp_root / "figures"
    time_s = _time_axis()
    n = len(gradcam_data)
    ncols = min(n, 2); nrows = max(1, (n + 1) // 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    for idx, (mname, gc_ic, gc_nic) in enumerate(gradcam_data):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        if gc_ic.size > 0:
            m_ic = gc_ic.mean(0); s_ic = gc_ic.std(0)
            ax.plot(time_s[:len(m_ic)], m_ic, color="#C85A5A", lw=1.2, label="Ictal")
            ax.fill_between(time_s[:len(m_ic)], m_ic - s_ic, m_ic + s_ic, color="#C85A5A", alpha=0.2)
        if gc_nic.size > 0:
            m_nic = gc_nic.mean(0); s_nic = gc_nic.std(0)
            ax.plot(time_s[:len(m_nic)], m_nic, color="#41B3A3", lw=1.2, label="Non-ictal")
            ax.fill_between(time_s[:len(m_nic)], m_nic - s_nic, m_nic + s_nic, color="#41B3A3", alpha=0.2)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("GradCAM")
        ax.set_title("%s GradCAM [%s]" % (mname, partition.upper())); ax.legend(fontsize=8)
    for idx in range(len(gradcam_data), nrows * ncols):
        r, c = divmod(idx, ncols); axes[r, c].set_visible(False)
    _save_fig(fig, fig_dir / ("fig_I_gradcam_saliency_%s.png" % partition), logger)


# ---------------------------------------------------------------------------
# plot_figure_J -- Occlusion sensitivity
# ---------------------------------------------------------------------------
def plot_figure_J(occ_data, partition, interp_root, logger):
    """2x2 grid of mean occlusion sensitivity per model."""
    fig_dir = interp_root / "figures"
    time_s = _time_axis()
    n = len(occ_data)
    ncols = min(n, 2); nrows = max(1, (n + 1) // 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    for idx, (mname, o_ic, o_nic) in enumerate(occ_data):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        if o_ic.size > 0:
            ax.plot(time_s, o_ic.mean(0), color="#C85A5A", lw=1.2, label="Ictal")
            ax.fill_between(time_s, o_ic.mean(0) - o_ic.std(0), o_ic.mean(0) + o_ic.std(0), color="#C85A5A", alpha=0.2)
        if o_nic.size > 0:
            ax.plot(time_s, o_nic.mean(0), color="#41B3A3", lw=1.2, label="Non-ictal")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Sensitivity S(t)")
        ax.set_title("%s Occlusion [%s]" % (mname, partition.upper())); ax.legend(fontsize=8)
    for idx in range(len(occ_data), nrows * ncols):
        r, c = divmod(idx, ncols); axes[r, c].set_visible(False)
    _save_fig(fig, fig_dir / ("fig_J_occlusion_sensitivity_%s.png" % partition), logger)


# ---------------------------------------------------------------------------
# plot_figure_K -- GradCAM vs attention overlay
# ---------------------------------------------------------------------------
def plot_figure_K(attn_data, gradcam_data, partition, interp_root, logger):
    """Mean alpha_t vs mean GradCAM for M2/M4 ictal segments. Annotate Pearson r."""
    fig_dir = interp_root / "figures"
    time_s = _time_axis()
    # Match models present in both dicts
    attn_dict = {m: (ic, nic) for m, ic, nic in attn_data}
    gc_dict = {m: (ic, nic) for m, ic, nic in gradcam_data}
    common = [m for m in attn_dict if m in gc_dict]
    if not common:
        logger.warning("No models with both attention and GradCAM for Figure K."); return

    pearson_results = {}
    fig, axes = plt.subplots(1, len(common), figsize=(6 * len(common), 4), squeeze=False)
    for idx, mname in enumerate(common):
        ax = axes[0, idx]
        m_attn = attn_dict[mname][0].mean(0)       # mean ictal attention
        m_gc = gc_dict[mname][0].mean(0)            # mean ictal GradCAM
        T_min = min(len(m_attn), len(m_gc))
        r_val, _ = stats.pearsonr(m_attn[:T_min], m_gc[:T_min])
        pearson_results[mname] = round(float(r_val), 6)
        ax.plot(time_s[:T_min], m_attn[:T_min], color="#7B68AE", lw=1.2, label="Attention")
        ax.plot(time_s[:T_min], m_gc[:T_min], color="#E8A87C", lw=1.2, label="GradCAM")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Normalised value")
        ax.set_title("%s Attn vs GradCAM [%s]\nr=%.4f" % (mname, partition.upper(), r_val)); ax.legend(fontsize=8)
    _save_fig(fig, fig_dir / ("fig_K_gradcam_vs_attention_%s.png" % partition), logger)
    return pearson_results


# ---------------------------------------------------------------------------
# plot_figure_L -- Branch activation profiles
# ---------------------------------------------------------------------------
def plot_figure_L(branch_act_data, partition, interp_root, logger):
    """2x2 grid: rows=(ictal, nonictal), cols=(M3, M4). Three branch lines each."""
    fig_dir = interp_root / "figures"
    time_s = _time_axis()
    models = [(m, ic, nic) for m, ic, nic in branch_act_data if ic is not None]
    if not models:
        logger.warning("No branch activation data for Figure L."); return
    n = len(models)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 8), squeeze=False)
    bcolors = ["#7B68AE", "#41B3A3", "#C85A5A"]
    blabels = ["B1 fine", "B2 medium", "B3 coarse"]
    for mi, (mname, act_ic, act_nic) in enumerate(models):
        for ri, (class_data, class_name) in enumerate([(act_ic, "Ictal"), (act_nic, "Non-ictal")]):
            ax = axes[ri, mi]
            for bi in range(3):
                m_act = class_data[bi].mean(0)
                s_act = class_data[bi].std(0)
                ax.plot(time_s, m_act, color=bcolors[bi], lw=1.2, label=blabels[bi])
                ax.fill_between(time_s, m_act - s_act, m_act + s_act, color=bcolors[bi], alpha=0.15)
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Mean |activation|")
            ax.set_title("%s %s [%s]" % (mname, class_name, partition.upper()))
            if ri == 0 and mi == n - 1:
                ax.legend(fontsize=7, loc="upper right")
    _save_fig(fig, fig_dir / ("fig_L_branch_activation_%s.png" % partition), logger)


# ---------------------------------------------------------------------------
# save_interpretability_report
# ---------------------------------------------------------------------------
def save_interpretability_report(interp_root, partition, entropy_stats, ablation_m3,
                                 ablation_m4, pearson_m2, pearson_m4, branch_dilations,
                                 hp, logger):
    """Write interpretability_report_{partition}.md with computed values."""
    report_lines = [
        "# Interpretability Report -- %s partition" % partition.upper(),
        "",
        "Generated: %s" % datetime.datetime.now().isoformat(),
        "",
    ]
    if partition == "val":
        report_lines += [
            "**WARNING: VALIDATION PARTITION**",
            "These results carry indirect optimisation bias.",
            "DO NOT report as final paper results.",
            "Run --partition test for publication-quality figures.",
            "",
        ]
    else:
        report_lines += [
            "**TEST PARTITION -- suitable for publication.**",
            "",
        ]

    report_lines += ["## Layer 1: Temporal Attention", ""]
    if entropy_stats:
        for mname, st in entropy_stats.items():
            report_lines.append("### %s Entropy Statistics" % mname)
            report_lines.append("- Median ictal: %.6f" % st["median_ictal"])
            report_lines.append("- Median non-ictal: %.6f" % st["median_nonictal"])
            report_lines.append("- Mann-Whitney U: %.2f, p: %.2e, r: %.6f" % (st["U"], st["p"], st["r"]))
            report_lines.append("")

    report_lines += ["## Layer 2: Branch Ablation", ""]
    for mname, res in [("M3", ablation_m3), ("M4", ablation_m4)]:
        if res:
            report_lines.append("### %s" % mname)
            report_lines.append("- Baseline F1: %.6f" % res["baseline_f1"])
            for bi in [1, 2, 3]:
                report_lines.append("- B%d contribution: %.6f" % (bi, res["branch%d_contribution" % bi]))
            report_lines.append("")

    report_lines += ["## Layer 3: GradCAM vs Attention", ""]
    if pearson_m2 is not None:
        report_lines.append("- M2 Pearson r (attn vs GradCAM): %.6f" % pearson_m2)
    if pearson_m4 is not None:
        report_lines.append("- M4 Pearson r (attn vs GradCAM): %.6f" % pearson_m4)
    report_lines.append("")

    report_path = interp_root / ("interpretability_report_%s.md" % partition)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info("Saved: %s", report_path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    """Main entry point. Parses args, validates partition, runs full pipeline."""
    args = parse_args()                                # parse --partition val or --partition test
    partition = args.partition                          # "val" or "test"
    interp_root = INTERP_BASE / partition              # outputs/interpretability/val/ or test/

    logger = setup_logging(interp_root, partition)     # create dirs + configure logger + log warnings
    set_seed(SEED)                                     # fix all RNG seeds for reproducibility
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # auto-detect GPU
    if torch.cuda.is_available():                      # log GPU details for reproducibility audit
        logger.info("GPU: %s | CUDA: %s", torch.cuda.get_device_name(0), torch.version.cuda)
    else:
        logger.info("Device: CPU")                     # warn: occlusion on CPU will be very slow

    # -- Load JSON configs (read-only -- these files are never modified) --------
    with open(BEST_TCN_PATH, "r", encoding="utf-8") as f:  # TCN backbone HPs (M1, M2)
        hp = json.load(f)["hyperparameters"]
    logger.info("Loaded backbone HP from %s", BEST_TCN_PATH)

    # MultiScaleTCN HPs -- may not exist if tune_multiscale_tcn.py has not run yet
    ms_hp, branch_dilations = None, {"branch1": DEFAULT_BRANCH1, "branch2": DEFAULT_BRANCH2, "branch3": DEFAULT_BRANCH3}
    if BEST_MS_PATH.exists():
        with open(BEST_MS_PATH, "r", encoding="utf-8") as f:  # M3 backbone HPs
            ms_cfg = json.load(f)
        ms_hp = ms_cfg["hyperparameters"]              # num_filters, kernel_size, dropout, fusion
        branch_dilations = ms_cfg.get("branch_dilations", branch_dilations)  # {"branch1": [...], ...}
        logger.info("Loaded multiscale HP from %s", BEST_MS_PATH)

    # Attention HPs -- may not exist if tune_temporal_attention.py has not run yet
    attn_hp = None
    if BEST_ATTN_PATH.exists():
        with open(BEST_ATTN_PATH, "r", encoding="utf-8") as f:  # M2 attention HPs
            attn_hp = json.load(f)["hyperparameters"]  # attention_dim, attention_dropout, lr, wd, bs
        logger.info("Loaded attention HP from %s", BEST_ATTN_PATH)

    # MultiScale attention HPs -- may not exist if tune_multiscale_attention.py has not run yet
    ms_attn_hp = None
    if BEST_MS_ATTN_PATH.exists():
        with open(BEST_MS_ATTN_PATH, "r", encoding="utf-8") as f:  # M4 attention HPs
            ms_attn_hp = json.load(f)["hyperparameters"]  # attention_dim, attention_dropout, lr, wd, bs
        logger.info("Loaded multiscale attention HP from %s", BEST_MS_ATTN_PATH)

    # -- Load partition data ---------------------------------------------------
    pairs, loader, y_true, x_all, n_ictal, n_nonictal = load_partition_data(
        partition, batch_size=32, device=DEVICE, logger=logger)

    # -- Load models -----------------------------------------------------------
    models = {}
    logger.info("=" * 65)
    logger.info("Loading models...")

    if M1_WEIGHTS.exists():
        models["M1"] = load_model_weights("M1", M1_WEIGHTS, hp, DEVICE, logger)
    else:
        logger.warning("M1 weights not found, skipping.")

    if M2_WEIGHTS.exists() and attn_hp is not None:
        models["M2"] = load_model_weights("M2", M2_WEIGHTS, hp, DEVICE, logger, attn_hp=attn_hp)
    else:
        logger.warning("M2 weights or attn_hp not found, skipping.")

    if M3_WEIGHTS.exists() and ms_hp is not None:
        models["M3"] = load_model_weights("M3", M3_WEIGHTS, ms_hp, DEVICE, logger,
                                          branch_dilations=branch_dilations)
    else:
        logger.warning("M3 weights or ms_hp not found, skipping.")

    m4_model = None
    if M4_WEIGHTS.exists() and M4_AVAILABLE and ms_hp is not None and ms_attn_hp is not None:
        m4_model = load_model_weights("M4", M4_WEIGHTS, ms_hp, DEVICE, logger,
                                      attn_hp=ms_attn_hp, branch_dilations=branch_dilations)
        if m4_model is not None:
            models["M4"] = m4_model
    else:
        logger.warning("M4 not available (weights/class/hp missing), skipping.")

    # ========================================================================
    # LAYER 1: Temporal attention saliency (M2, M4)
    # Clinical relevance: identifies WHICH part of the 5-second EEG window
    # triggered the seizure alarm, enabling clinician verification.
    # Uses get_attention_weights() -- no post-hoc attribution needed.
    # ========================================================================
    logger.info("=" * 65)
    logger.info("LAYER 1: Temporal Attention Saliency")
    attn_data = []
    entropy_data = []

    for mname in ["M2", "M4"]:
        if mname not in models:
            continue
        w_ic, w_nic = collect_attention_weights(
            models[mname], loader, y_true, DEVICE, mname, partition, interp_root, logger)
        attn_data.append((mname, w_ic, w_nic))
        h_ic = compute_entropy(w_ic, logger)
        h_nic = compute_entropy(w_nic, logger)
        entropy_data.append((mname, h_ic, h_nic))
        # Save entropy arrays
        pfx = mname.lower()
        np.save(interp_root / "entropy" / ("%s_entropy_ictal_%s.npy" % (pfx, partition)), h_ic)
        np.save(interp_root / "entropy" / ("%s_entropy_nonictal_%s.npy" % (pfx, partition)), h_nic)

    if attn_data:
        plot_figure_A(attn_data, partition, interp_root, logger)
        plot_figure_B(attn_data, partition, interp_root, logger)
    if "M2" in models:
        plot_figure_C(models["M2"], loader, y_true, x_all, DEVICE, partition, interp_root, logger)
        plot_figure_E(models["M2"], x_all, y_true, DEVICE, partition, interp_root, logger)
    entropy_stats = {}
    if entropy_data:
        entropy_stats = plot_figure_D(entropy_data, partition, interp_root, logger)

    # ========================================================================
    # LAYER 2: Branch contribution analysis (M3, M4)
    # Clinical relevance: determines whether each temporal scale (spikes,
    # bursts, envelopes) contributes independently to seizure detection.
    # In drug development, identifies which seizure features are most
    # affected by a pharmacological intervention.
    # ========================================================================
    logger.info("=" * 65)
    logger.info("LAYER 2: Branch Contribution Analysis")
    ablation_results = {}
    branch_act_data = []

    for mname in ["M3", "M4"]:
        if mname not in models:
            continue
        ablation_results[mname] = run_branch_ablation(
            models[mname], loader, y_true, DEVICE, mname, partition, interp_root, logger)
        act_ic, act_nic = collect_branch_activations(
            models[mname], loader, y_true, DEVICE, mname, partition, interp_root, logger)
        branch_act_data.append((mname, act_ic, act_nic))

    if ablation_results:
        plot_figure_F(ablation_results, branch_dilations, partition, interp_root, logger)
        plot_figure_H(ablation_results, partition, interp_root, logger)
    if ms_hp is not None:
        plot_figure_G(ms_hp, branch_dilations, partition, interp_root, logger)
    if branch_act_data:
        plot_figure_L(branch_act_data, partition, interp_root, logger)

    # ========================================================================
    # LAYER 3: Feature map analysis (all models, including M1 with no attention)
    # Clinical relevance: identifies which input EEG features drive the model's
    # prediction regardless of architecture. GradCAM and occlusion are model-
    # agnostic methods that work even on M1 (no built-in interpretability).
    # Agreement between methods validates faithfulness of explanations.
    # ========================================================================
    logger.info("=" * 65)
    logger.info("LAYER 3: Feature Map Analysis")
    gradcam_data = []
    occ_data = []

    for mname in ["M1", "M2", "M3", "M4"]:
        if mname not in models:
            continue
        logger.info("Computing GradCAM for %s...", mname)
        gc_ic, gc_nic = compute_gradcam_1d(
            models[mname], loader, y_true, DEVICE, mname, partition, interp_root, logger)
        gradcam_data.append((mname, gc_ic, gc_nic))

        logger.info("Computing occlusion sensitivity for %s...", mname)
        o_ic, o_nic = compute_occlusion_sensitivity(
            models[mname], x_all, y_true, DEVICE, mname, partition, interp_root, logger)
        occ_data.append((mname, o_ic, o_nic))

    if gradcam_data:
        plot_figure_I(gradcam_data, partition, interp_root, logger)
    if occ_data:
        plot_figure_J(occ_data, partition, interp_root, logger)

    pearson_results = {}
    if attn_data and gradcam_data:
        pearson_results = plot_figure_K(attn_data, gradcam_data, partition, interp_root, logger) or {}

    # ========================================================================
    # Save report
    # ========================================================================
    logger.info("=" * 65)
    logger.info("Saving interpretability report...")
    save_interpretability_report(
        interp_root, partition, entropy_stats,
        ablation_results.get("M3"), ablation_results.get("M4"),
        pearson_results.get("M2"), pearson_results.get("M4"),
        branch_dilations, hp, logger)

    # ========================================================================
    # Final inventory
    # ========================================================================
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("=" * 65)
    logger.info("ALL OUTPUTS SAVED to %s", interp_root)
    fig_dir = interp_root / "figures"
    for fig_name in sorted(fig_dir.glob("*.png")):
        logger.info("  [OK] %s", fig_name.name)
    logger.info("=" * 65)
    logger.info("Partition: %s | Figures: %d | Models analysed: %s",
                partition.upper(), len(list(fig_dir.glob("*.png"))), list(models.keys()))


if __name__ == "__main__":
    main()
