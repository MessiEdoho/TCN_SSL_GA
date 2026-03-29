"""
tcn_utils.py -- Shared utilities for TCN-based EEG seizure detection.

This module centralises all architecture definitions, dataset classes,
training loops, and evaluation functions used across the pipeline notebooks:
  - tcn_HPT_binary.ipynb (hyperparameter tuning)
  - TCN training and evaluation notebooks
  - TCN-Attention and SSL pipeline notebooks

All components are parameterised -- no global variable references.
Every class and function includes a RESEARCH REPORTING NOTE block
documenting which parameters must be reported in the methods section.
"""

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score

try:
    import optuna
    from optuna.exceptions import TrialPruned
except ImportError:
    optuna = None
    TrialPruned = Exception


# ---------------------------------------------------------------------------
# 2. set_seed
# ---------------------------------------------------------------------------
def set_seed(seed=42):
    """Set random seeds for full reproducibility across all libraries.

    Parameters
    ----------
    seed : int, default 42
        The seed value to use for all random number generators.

    Returns
    -------
    None

    Example
    -------
    >>> set_seed(42)
    """
    random.seed(seed)                                # Python built-in RNG
    np.random.seed(seed)                             # NumPy RNG
    torch.manual_seed(seed)                          # PyTorch CPU RNG
    torch.cuda.manual_seed_all(seed)                 # PyTorch GPU RNG (all devices)
    torch.backends.cudnn.deterministic = True        # force deterministic CUDA operations
    torch.backends.cudnn.benchmark = False           # disable cuDNN auto-tuner


# -- RESEARCH REPORTING NOTE: set_seed -----------------------------------------
# Methods description:
#   All random number generators (Python, NumPy, PyTorch CPU and CUDA) were
#   seeded with a fixed value to ensure full reproducibility of weight
#   initialisation, data shuffling, and dropout masks across runs.
#
# Parameters to report in paper:
#   seed : the fixed seed value (standard for reproducibility claims)
#
# Design choices to justify:
#   cudnn.deterministic=True : trades marginal speed for exact reproducibility
#   cudnn.benchmark=False : prevents non-deterministic algorithm selection
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 3. EEGSegmentDataset
# ---------------------------------------------------------------------------
class EEGSegmentDataset(Dataset):
    """Memory-efficient EEG segment dataset that loads .npy files on demand.

    Each segment was already robust z-scored during preprocessing.
    No additional normalisation is applied at load time.

    Parameters
    ----------
    file_label_pairs : list of (str or Path, int)
        Each entry is (path_to_npy_file, label) where label is 0 or 1.

    Example
    -------
    >>> ds = EEGSegmentDataset([("seg_001.npy", 1), ("seg_002.npy", 0)])
    >>> x, y = ds[0]  # x: (1, 2500), y: scalar tensor
    """

    def __init__(self, file_label_pairs):
        self.pairs = file_label_pairs              # store (path, label) pairs

    def __len__(self):
        return len(self.pairs)                     # total number of segments

    def __getitem__(self, idx):
        path, label = self.pairs[idx]              # retrieve path and label
        x = np.load(path).astype(np.float32)       # load segment as float32
        # No normalisation -- data was already robust z-scored during preprocessing
        x = torch.from_numpy(x).unsqueeze(0)       # shape: (1, segment_len) -- 1 EEG channel
        y = torch.tensor(label, dtype=torch.float32)  # scalar label: 0.0 or 1.0
        return x, y


# -- RESEARCH REPORTING NOTE: EEGSegmentDataset --------------------------------
# Methods description:
#   EEG segments were loaded on demand from individual .npy files to minimise
#   memory usage. Each segment had been previously normalised using robust
#   z-score (median and MAD) during the preprocessing pipeline; no additional
#   normalisation was applied at training time.
#
# Parameters to report in paper:
#   segment_shape : (1, 2500) -- single-channel, 5 s at 500 Hz
#   normalisation : robust z-score applied during preprocessing (not at load time)
#
# Design choices to justify:
#   No load-time normalisation : avoids double-normalising data that was already
#     robust z-scored, which would undo the MAD-based artefact resistance.
#   On-demand loading : keeps memory proportional to batch size, not dataset size.
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 4. compute_pos_weight
# ---------------------------------------------------------------------------
def compute_pos_weight(train_pairs, device):
    """Compute pos_weight for BCEWithLogitsLoss from training label distribution.

    Parameters
    ----------
    train_pairs : list of (str or Path, int)
        Training file-label pairs. Label 1 = ictal, 0 = non-ictal.
    device : torch.device
        Target device for the returned tensor.

    Returns
    -------
    pos_weight : torch.Tensor, shape (1,)
        Ratio n_non_ictal / n_ictal, on the specified device.

    Example
    -------
    >>> pw = compute_pos_weight(train_pairs, torch.device("cuda"))
    """
    n_ictal = sum(1 for _, l in train_pairs if l == 1)      # count positive (ictal) samples
    n_non_ictal = sum(1 for _, l in train_pairs if l == 0)   # count negative (non-ictal) samples
    ratio = n_non_ictal / max(n_ictal, 1)                    # avoid division by zero
    return torch.tensor([ratio], dtype=torch.float32).to(device)  # move to device for loss computation


# -- RESEARCH REPORTING NOTE: compute_pos_weight -------------------------------
# Methods description:
#   The positive class weight for binary cross-entropy was computed as the ratio
#   of non-ictal to ictal segments in the training partition, compensating for
#   class imbalance by upweighting the loss contribution of the minority class.
#
# Parameters to report in paper:
#   pos_weight value : the actual computed ratio (e.g. 15.3)
#   n_ictal, n_non_ictal : segment counts in the training partition
#
# Design choices to justify:
#   pos_weight = n_neg / n_pos : standard inverse-frequency weighting;
#     combined with WeightedRandomSampler for dual imbalance correction.
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 5. make_loader
# ---------------------------------------------------------------------------
def make_loader(file_label_pairs, batch_size, train, device):
    """Build a DataLoader with optional WeightedRandomSampler for training.

    Parameters
    ----------
    file_label_pairs : list of (str or Path, int)
        File-label pairs for the dataset.
    batch_size : int
        Number of segments per batch.
    train : bool
        If True, use WeightedRandomSampler to oversample the minority class.
    device : torch.device
        Used to set pin_memory (True when device is CUDA).

    Returns
    -------
    loader : DataLoader

    Example
    -------
    >>> loader = make_loader(pairs, 32, train=True, device=torch.device("cuda"))
    """
    dataset = EEGSegmentDataset(file_label_pairs)          # create dataset instance
    pin = (device.type == "cuda")                          # pin memory for faster GPU transfer

    if train:
        labels = [lbl for _, lbl in file_label_pairs]      # extract all labels as a list
        n_pos = sum(labels)                                # count ictal (positive) segments
        n_neg = len(labels) - n_pos                        # count non-ictal (negative) segments
        # Inverse-frequency weights: rarer class gets higher sampling probability
        w_per_class = {0: 1.0 / max(n_neg, 1),
                       1: 1.0 / max(n_pos, 1)}
        sample_weights = [w_per_class[l] for l in labels]  # per-sample weight list
        sampler = WeightedRandomSampler(
            weights=sample_weights,                        # sampling probability per segment
            num_samples=len(labels),                       # draw this many samples per epoch
            replacement=True                               # allow repeated draws for minority class
        )
        return DataLoader(dataset, batch_size=batch_size,
                          sampler=sampler, num_workers=0,
                          pin_memory=pin, drop_last=False)
    else:
        return DataLoader(dataset, batch_size=batch_size,
                          shuffle=False, num_workers=0,
                          pin_memory=pin, drop_last=False)


# -- RESEARCH REPORTING NOTE: make_loader --------------------------------------
# Methods description:
#   Training batches were constructed using a WeightedRandomSampler with
#   inverse-frequency class weights, producing approximately balanced batches
#   without discarding any samples. Validation batches were loaded sequentially
#   without shuffling.
#
# Parameters to report in paper:
#   batch_size : affects gradient noise and GPU memory usage
#   WeightedRandomSampler : cite as minority oversampling strategy
#   pin_memory : set True for CUDA devices (implementation detail, not reported)
#
# Design choices to justify:
#   WeightedRandomSampler + pos_weight : dual imbalance correction -- sampler
#     balances what the model sees, pos_weight adjusts gradient contribution.
#   num_workers=0 : cross-platform compatibility; can increase on Linux.
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 6. EEGSegmentSSLDataset
# ---------------------------------------------------------------------------
class EEGSegmentSSLDataset(Dataset):
    """Self-supervised EEG dataset returning two augmented views per segment.

    No labels are returned. Each call to __getitem__ produces two independently
    augmented views of the same segment for contrastive learning.

    Parameters
    ----------
    file_paths : list of str or Path
        Paths to .npy segment files (no labels).
    segment_len : int, default 2500
        Expected segment length in samples.

    Example
    -------
    >>> ds = EEGSegmentSSLDataset(["seg_001.npy", "seg_002.npy"])
    >>> v1, v2 = ds[0]  # both shape (1, 2500)
    """

    def __init__(self, file_paths, segment_len=2500):
        self.file_paths = file_paths               # list of .npy file paths
        self.segment_len = segment_len             # target segment length

    def __len__(self):
        return len(self.file_paths)                # total number of segments

    def __getitem__(self, idx):
        x = np.load(self.file_paths[idx]).astype(np.float32)  # load raw segment
        view1 = self._augment(x.copy())            # first independently augmented view
        view2 = self._augment(x.copy())            # second independently augmented view
        return view1, view2

    def _augment(self, x):
        """Apply random augmentations to create one view.

        Augmentations applied in order:
        1. Temporal jitter (random crop)
        2. Amplitude scaling
        3. Gaussian noise injection
        4. Z-score normalisation
        """
        # 1. Temporal jitter: random offset up to 125 samples
        max_offset = min(125, max(0, len(x) - self.segment_len))  # clamp to available range
        if max_offset > 0:
            offset = np.random.randint(0, max_offset + 1)         # random start offset
            x = x[offset : offset + self.segment_len]             # crop from offset
        x = x[:self.segment_len]                                  # fallback: take first segment_len samples
        if len(x) < self.segment_len:                             # pad if too short
            x = np.pad(x, (0, self.segment_len - len(x)))

        # 2. Amplitude scaling: uniform random in [0.5, 2.0]
        scale = np.random.uniform(0.5, 2.0)                      # random scale factor
        x = x * scale

        # 3. Gaussian noise: sigma = uniform(0.01, 0.1) * std(x)
        sigma = np.random.uniform(0.01, 0.1) * (x.std() + 1e-8)  # noise level proportional to signal
        x = x + np.random.normal(0, sigma, size=x.shape).astype(np.float32)

        # 4. Z-score normalisation after all augmentations
        mu = x.mean()
        std = x.std() + 1e-8                                     # epsilon to prevent div-by-zero
        x = (x - mu) / std

        return torch.from_numpy(x).unsqueeze(0)                  # shape: (1, segment_len)


# -- RESEARCH REPORTING NOTE: EEGSegmentSSLDataset -----------------------------
# Methods description:
#   For self-supervised pre-training, two augmented views of each EEG segment
#   were generated independently. Augmentations comprised temporal jitter
#   (random crop up to 125 samples), random amplitude scaling (0.5--2.0x),
#   additive Gaussian noise (sigma proportional to signal std), and per-view
#   z-score normalisation.
#
# Parameters to report in paper:
#   temporal_jitter_max : 125 samples (0.25 s at 500 Hz)
#   amplitude_scale_range : [0.5, 2.0]
#   noise_sigma_range : [0.01, 0.1] * std(segment)
#   segment_len : 2500 samples (5 s at 500 Hz)
#
# Design choices to justify:
#   Temporal jitter : simulates slight misalignment in seizure onset labelling
#   Amplitude scaling : accounts for inter-subject amplitude variability
#   Gaussian noise : improves robustness to recording noise
#   Post-augmentation z-score : ensures views have comparable scale for
#     contrastive loss computation
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 7. make_ssl_loader
# ---------------------------------------------------------------------------
def make_ssl_loader(file_paths, batch_size, device, segment_len=2500):
    """Build a DataLoader for self-supervised pre-training.

    Parameters
    ----------
    file_paths : list of str or Path
        Paths to .npy segment files (no labels).
    batch_size : int
        Number of segments per batch.
    device : torch.device
        Used to set pin_memory.
    segment_len : int, default 2500
        Expected segment length in samples.

    Returns
    -------
    loader : DataLoader

    Example
    -------
    >>> loader = make_ssl_loader(paths, 64, torch.device("cuda"))
    """
    dataset = EEGSegmentSSLDataset(file_paths, segment_len=segment_len)
    pin = (device.type == "cuda")                  # pin memory for faster GPU transfer
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,                              # random order for contrastive learning
        drop_last=True,                            # drop last incomplete batch to keep batch size constant for NT-Xent
        num_workers=0,                             # cross-platform compatibility
        pin_memory=pin
    )


# -- RESEARCH REPORTING NOTE: make_ssl_loader ----------------------------------
# Methods description:
#   Self-supervised training batches were shuffled randomly with the last
#   incomplete batch dropped to ensure a constant batch size, which is required
#   for correct NT-Xent contrastive loss computation.
#
# Parameters to report in paper:
#   batch_size : affects the number of negative pairs in contrastive loss
#   drop_last=True : required for NT-Xent (report as implementation detail)
#
# Design choices to justify:
#   drop_last=True : NT-Xent loss constructs a 2N x 2N similarity matrix;
#     variable batch sizes would produce inconsistent loss magnitudes.
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 8. CausalConvBlock
# ---------------------------------------------------------------------------
class CausalConvBlock(nn.Module):
    """Single causal convolutional block with residual connection.

    Architecture: Conv1d -> LayerNorm -> GELU -> Dropout1d -> Residual add.
    Causal padding ensures no information leaks from future time steps.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.
    kernel_size : int
        Convolution kernel size (must be odd).
    dilation : int
        Dilation factor for this layer.
    dropout : float
        Spatial dropout rate (drops entire channels).

    Example
    -------
    >>> block = CausalConvBlock(1, 64, kernel_size=7, dilation=4, dropout=0.2)
    >>> out = block(torch.randn(8, 1, 2500))  # (8, 64, 2500)
    """

    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation    # total causal padding (left side only)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              dilation=dilation, padding=self.pad)
        self.norm = nn.LayerNorm(out_ch)           # normalise across channel dim per time step
        self.act = nn.GELU()                       # smooth activation for bio-signal features
        self.drop = nn.Dropout1d(dropout)          # spatial dropout: drops entire channels
        # 1x1 conv for residual projection when channel counts differ
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        """Forward pass. x shape: (batch, channels, time)."""
        res = self.residual(x)                     # project input for skip connection
        out = self.conv(x)                         # dilated conv with causal padding
        out = out[:, :, :x.size(2)]                # trim right side to enforce causality (no future leak)
        out = out.transpose(1, 2)                  # (B, T, C) -- LayerNorm expects channels last
        out = self.norm(out)                       # normalise across channel dimension
        out = out.transpose(1, 2)                  # (B, C, T) -- back to conv format
        out = self.act(out)                        # GELU activation
        out = self.drop(out)                       # spatial dropout
        return out + res                           # residual connection


# -- RESEARCH REPORTING NOTE: CausalConvBlock ----------------------------------
# Methods description:
#   Each dilated causal convolutional block comprised a 1-D causal convolution
#   with exponential dilation, layer normalisation, GELU activation, spatial
#   dropout, and a residual skip connection with 1x1 projection when input
#   and output channel counts differed.
#
# Parameters to report in paper:
#   kernel_size : determines temporal resolution per layer
#   dilation : determines receptive field contribution per layer
#   dropout : regularisation strength (affects generalisation)
#
# Design choices to justify:
#   LayerNorm instead of BatchNorm : EEG amplitude varies across subjects;
#     BatchNorm statistics are unreliable at small batch sizes (16--64).
#   Dropout1d instead of Dropout : structured EEG feature maps benefit from
#     channel-level rather than element-level regularisation.
#   GELU instead of ReLU : smoother gradient flow suits bio-signal features.
#   Causal padding with right-trim : prevents temporal information leakage.
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 9. TCN
# ---------------------------------------------------------------------------
class TCN(nn.Module):
    """Temporal Convolutional Network for binary EEG classification.

    Stacks L CausalConvBlocks with exponential dilation d_l = 2^l,
    followed by global average pooling and a linear classification head.

    Parameters
    ----------
    num_layers : int
        Number of stacked causal conv blocks (L).
    num_filters : int
        Output channels per convolutional layer.
    kernel_size : int
        Kernel size (must be odd).
    dropout : float
        Spatial dropout rate per block.
    fs : int, default 500
        Sampling rate in Hz (used only for RF logging).

    Example
    -------
    >>> model = TCN(7, 64, 5, 0.2)
    >>> logits = model(torch.randn(8, 1, 2500))  # (8,)
    """

    def __init__(self, num_layers, num_filters, kernel_size, dropout, fs=500):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_ch = 1 if i == 0 else num_filters   # first block takes single-channel EEG
            dilation = 2 ** i                       # exponential dilation schedule
            layers.append(CausalConvBlock(in_ch, num_filters, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)       # sequential stack of all blocks
        self.head = nn.Linear(num_filters, 1)       # classification head: 1 logit for binary
        self.rf = (2 ** num_layers) * (kernel_size - 1)  # receptive field in samples
        self.num_filters = num_filters              # store for external access

    def forward(self, x):
        """Forward pass. x: (batch, 1, segment_len). Returns logits: (batch,)."""
        out = self.network(x)                       # (batch, num_filters, time)
        out = out.mean(dim=2)                       # global average pooling over time
        return self.head(out).squeeze(-1)           # (batch,) raw logits


# -- RESEARCH REPORTING NOTE: TCN ----------------------------------------------
# Methods description:
#   The TCN comprised L stacked dilated causal convolutional blocks with
#   exponential dilation schedule d_l = 2^l, yielding a receptive field of
#   2^L * (k-1) samples. Global average pooling collapsed the temporal
#   dimension before a single linear head produced binary classification logits.
#
# Parameters to report in paper:
#   num_layers (L) : determines depth and RF -- compute and report RF in seconds
#   kernel_size (k) : determines local temporal resolution
#   num_filters : model capacity (width)
#   total trainable parameters : standard for reproducibility
#   receptive field : 2^L * (k-1) samples and RF / fs seconds
#
# Design choices to justify:
#   Global average pooling : makes model length-agnostic after causal trimming;
#     acts as a spatial regulariser reducing overfitting risk.
#   Single linear head : sufficient for binary classification; avoids
#     unnecessary complexity in the decision boundary.
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 10. TCNWithAttention
# ---------------------------------------------------------------------------
class TCNWithAttention(nn.Module):
    """TCN backbone followed by multi-head self-attention.

    The TCN extracts temporal features; self-attention captures global
    dependencies across the full sequence. Optionally returns the embedding
    (before the classification head) for use as an encoder in SSL pipelines.

    Parameters
    ----------
    num_layers : int
        Number of TCN blocks.
    num_filters : int
        Channels per TCN layer and attention embedding dimension.
    kernel_size : int
        TCN kernel size (odd).
    dropout : float
        Spatial dropout rate for TCN blocks.
    n_heads : int, default 4
        Number of attention heads.
    return_embedding : bool, default False
        If True, return the pooled embedding instead of classification logits.
    fs : int, default 500
        Sampling rate for RF logging.

    Example
    -------
    >>> model = TCNWithAttention(7, 64, 5, 0.2, return_embedding=True)
    >>> emb = model(torch.randn(8, 1, 2500))  # (8, 64)
    """

    def __init__(self, num_layers, num_filters, kernel_size, dropout,
                 n_heads=4, return_embedding=False, fs=500):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_ch = 1 if i == 0 else num_filters
            dilation = 2 ** i
            layers.append(CausalConvBlock(in_ch, num_filters, kernel_size, dilation, dropout))
        self.tcn = nn.Sequential(*layers)

        # Multi-head self-attention: Q = K = V = TCN output
        self.attn = nn.MultiheadAttention(
            embed_dim=num_filters, num_heads=n_heads, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(num_filters)  # post-attention normalisation
        self.return_embedding = return_embedding
        self.head = nn.Linear(num_filters, 1)       # classification head
        self.rf = (2 ** num_layers) * (kernel_size - 1)
        self.num_filters = num_filters

    def forward(self, x):
        """Forward pass. x: (batch, 1, segment_len)."""
        out = self.tcn(x)                           # (batch, num_filters, time)
        out = out.transpose(1, 2)                   # (batch, time, num_filters) for attention

        # Self-attention with residual connection and layer norm
        attn_out, _ = self.attn(out, out, out)      # Q = K = V = TCN output
        out = self.attn_norm(out + attn_out)         # residual + LayerNorm

        out = out.mean(dim=1)                       # global average pooling over time

        if self.return_embedding:
            return out                              # (batch, num_filters) embedding
        return self.head(out).squeeze(-1)           # (batch,) logits


# -- RESEARCH REPORTING NOTE: TCNWithAttention ---------------------------------
# Methods description:
#   A multi-head self-attention layer was appended after the TCN stack. The
#   TCN output served as query, key, and value (self-attention). A residual
#   connection and layer normalisation followed the attention output. Global
#   average pooling produced a fixed-length embedding vector.
#
# Parameters to report in paper:
#   n_heads : number of attention heads
#   num_filters : embedding dimension (= TCN channel width)
#   All TCN parameters (num_layers, kernel_size, dropout)
#   total trainable parameters
#
# Design choices to justify:
#   Self-attention after TCN : captures long-range temporal dependencies
#     beyond the TCN receptive field without increasing dilation depth.
#   Residual + LayerNorm after attention : standard transformer practice;
#     stabilises training and preserves TCN features.
#   return_embedding mode : enables reuse as encoder in SSL pipelines.
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 11. SSLModel
# ---------------------------------------------------------------------------
class SSLModel(nn.Module):
    """Self-supervised learning model with encoder and projection head.

    The encoder is a TCNWithAttention in embedding mode. The projector
    maps embeddings to a lower-dimensional space for contrastive loss.
    Outputs are L2-normalised to the unit hypersphere.

    Parameters
    ----------
    num_layers : int
        Number of TCN blocks in the encoder.
    num_filters : int
        TCN channel width and encoder embedding dimension.
    kernel_size : int
        TCN kernel size.
    dropout : float
        Spatial dropout rate.
    projection_dim : int, default 128
        Output dimension of the projection head.
    n_heads : int, default 4
        Number of attention heads in the encoder.

    Example
    -------
    >>> ssl = SSLModel(7, 64, 5, 0.2, projection_dim=128)
    >>> z = ssl(torch.randn(8, 1, 2500))  # (8, 128), L2-normalised
    """

    def __init__(self, num_layers, num_filters, kernel_size, dropout,
                 projection_dim=128, n_heads=4):
        super().__init__()
        self.encoder = TCNWithAttention(
            num_layers, num_filters, kernel_size, dropout,
            n_heads=n_heads, return_embedding=True
        )
        self.projector = nn.Sequential(
            nn.Linear(num_filters, num_filters),    # first linear layer
            nn.GELU(),                              # non-linear activation
            nn.Linear(num_filters, projection_dim)  # project to contrastive space
        )

    def forward(self, x):
        """Forward pass. Returns L2-normalised projection."""
        emb = self.encoder(x)                       # (batch, num_filters)
        proj = self.projector(emb)                  # (batch, projection_dim)
        return F.normalize(proj, dim=-1)            # L2 normalise to unit hypersphere


# -- RESEARCH REPORTING NOTE: SSLModel -----------------------------------------
# Methods description:
#   The self-supervised model comprised a TCN-Attention encoder followed by
#   a two-layer projection head (Linear-GELU-Linear) mapping to a
#   projection_dim-dimensional space. Outputs were L2-normalised to the
#   unit hypersphere for NT-Xent contrastive loss computation.
#
# Parameters to report in paper:
#   projection_dim : output dimension of the projection head
#   All encoder parameters (num_layers, num_filters, kernel_size, dropout, n_heads)
#
# Design choices to justify:
#   Two-layer projector with GELU : standard SimCLR-style projection head;
#     non-linear projector empirically outperforms linear for contrastive learning.
#   L2 normalisation : required for cosine-similarity-based NT-Xent loss.
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 12. nt_xent_loss
# ---------------------------------------------------------------------------
def nt_xent_loss(z1, z2, temperature=0.5):
    """Normalised Temperature-scaled Cross-Entropy (NT-Xent) contrastive loss.

    Parameters
    ----------
    z1 : torch.Tensor, shape (batch, dim)
        L2-normalised projections from view 1.
    z2 : torch.Tensor, shape (batch, dim)
        L2-normalised projections from view 2.
    temperature : float, default 0.5
        Temperature scaling factor for the similarity matrix.

    Returns
    -------
    loss : torch.Tensor, scalar
        Mean NT-Xent loss over all positive pairs.

    Example
    -------
    >>> loss = nt_xent_loss(z1, z2, temperature=0.5)
    """
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)                 # (2*batch, dim) -- concatenate both views
    sim = torch.mm(z, z.t()) / temperature         # (2N, 2N) cosine similarity / temperature

    # Mask diagonal (self-similarity) with -inf to exclude from softmax
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim.masked_fill_(mask, float('-inf'))

    # Positive pair labels: view1[i] <-> view2[i] at index i+batch; view2[i] <-> view1[i] at index i
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z.device),  # view1[i] -> view2[i]
        torch.arange(0, batch_size, device=z.device)                # view2[i] -> view1[i]
    ])

    return F.cross_entropy(sim, labels)            # NT-Xent loss


# -- RESEARCH REPORTING NOTE: nt_xent_loss -------------------------------------
# Methods description:
#   Contrastive pre-training used the NT-Xent loss (Chen et al., 2020).
#   Cosine similarity between all pairs in a batch of 2N projections was
#   computed and scaled by a temperature parameter. Each sample's positive
#   pair was its corresponding augmented view; all other samples served as
#   negatives.
#
# Parameters to report in paper:
#   temperature : controls the sharpness of the similarity distribution
#   batch_size : determines the number of negative pairs (2N-2 per sample)
#
# Design choices to justify:
#   temperature=0.5 : standard default; lower values sharpen the distribution
#     but may cause training instability.
#   Cosine similarity (via L2-normalised inputs) : standard for NT-Xent.
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 13. FineTunedModel
# ---------------------------------------------------------------------------
class FineTunedModel(nn.Module):
    """Fine-tuned model: pre-trained TCN-Attention encoder + linear classifier.

    The encoder can be frozen during initial fine-tuning epochs, then unfrozen
    for end-to-end training with a lower learning rate.

    Parameters
    ----------
    encoder : TCNWithAttention
        Pre-trained encoder in return_embedding=True mode.
    num_filters : int
        Encoder embedding dimension (must match encoder output).
    freeze_encoder : bool, default True
        If True, freeze encoder parameters (requires_grad=False).

    Example
    -------
    >>> encoder = ssl_model.encoder
    >>> ft = FineTunedModel(encoder, 64, freeze_encoder=True)
    >>> logits = ft(torch.randn(8, 1, 2500))  # (8,)
    """

    def __init__(self, encoder, num_filters, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder                     # pre-trained TCN-Attention encoder
        self.classifier = nn.Linear(num_filters, 1)  # binary classification head

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False            # freeze encoder weights

    def forward(self, x):
        """Forward pass. Returns logits: (batch,)."""
        emb = self.encoder(x)                      # (batch, num_filters) -- encoder embedding
        return self.classifier(emb).squeeze(-1)    # (batch,) raw logits


# -- RESEARCH REPORTING NOTE: FineTunedModel -----------------------------------
# Methods description:
#   For fine-tuning, a linear classification head was appended to the
#   pre-trained TCN-Attention encoder. During the initial phase, the encoder
#   was frozen and only the classification head was trained. In the second
#   phase, the encoder was unfrozen and trained end-to-end with a reduced
#   learning rate.
#
# Parameters to report in paper:
#   freeze_epochs : number of epochs with frozen encoder
#   encoder_lr : learning rate for encoder during unfrozen phase
#   classifier_lr : learning rate for classification head
#
# Design choices to justify:
#   Two-phase training : prevents catastrophic forgetting of pre-trained
#     representations by allowing the classifier to converge before updating
#     encoder weights.
#   Lower encoder LR : standard transfer learning practice to preserve
#     pre-trained features while allowing task-specific adaptation.
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 14. count_parameters
# ---------------------------------------------------------------------------
def count_parameters(model):
    """Count total trainable parameters in a model.

    Parameters
    ----------
    model : nn.Module
        The model to count parameters for.

    Returns
    -------
    n_params : int
        Total number of trainable parameters.

    Example
    -------
    >>> n = count_parameters(model)
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -- RESEARCH REPORTING NOTE: count_parameters ---------------------------------
# Methods description:
#   The total number of trainable parameters was computed as the sum of
#   elements across all parameter tensors with requires_grad=True.
#
# Parameters to report in paper:
#   total_params : standard metric for model complexity comparison
#
# Design choices to justify:
#   None -- standard implementation.
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 15. train_one_epoch
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimiser, criterion, device,
                    max_grad_norm=1.0):
    """Train the model for one epoch.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    loader : DataLoader
        Training data loader.
    optimiser : torch.optim.Optimizer
        The optimiser.
    criterion : nn.Module
        Loss function.
    device : torch.device
        Target device for tensor transfer.
    max_grad_norm : float, default 1.0
        Maximum gradient norm for clipping.

    Returns
    -------
    mean_loss : float
        Average training loss over all batches.

    Example
    -------
    >>> loss = train_one_epoch(model, loader, opt, criterion, device)
    """
    model.train()                                  # set model to training mode
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)         # transfer batch to device (batch-by-batch)
        optimiser.zero_grad()                      # clear accumulated gradients
        logits = model(x)                          # forward pass
        loss = criterion(logits, y)                # compute loss
        loss.backward()                            # backward pass
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # clip gradients
        optimiser.step()                           # update weights
        total_loss += loss.item()                  # accumulate scalar loss
        n_batches += 1

    return total_loss / max(n_batches, 1)


# -- RESEARCH REPORTING NOTE: train_one_epoch ----------------------------------
# Methods description:
#   Each training epoch iterated over all batches, computing the forward pass,
#   loss, and backward pass with gradient clipping before each weight update.
#
# Parameters to report in paper:
#   max_grad_norm : gradient clipping threshold (affects training stability)
#
# Design choices to justify:
#   Gradient clipping : prevents exploding gradients in deep TCN stacks;
#     max_norm=1.0 is a standard conservative choice.
#   Batch-by-batch device transfer : avoids exhausting GPU memory.
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 16. evaluate
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model and return macro F1, true labels, and predictions.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.
    loader : DataLoader
        Evaluation data loader.
    device : torch.device
        Target device for tensor transfer.

    Returns
    -------
    macro_f1 : float
    y_true : np.ndarray
    y_pred : np.ndarray

    Example
    -------
    >>> f1, y_true, y_pred = evaluate(model, val_loader, device)
    """
    model.eval()
    all_true = []
    all_pred = []

    for x, y in loader:
        x = x.to(device)                          # transfer input to device
        logits = model(x)                          # forward pass
        preds = (torch.sigmoid(logits) >= 0.5).long()
        all_true.append(y.cpu().numpy())           # move to CPU to prevent VRAM accumulation
        all_pred.append(preds.cpu().numpy())       # move to CPU to prevent VRAM accumulation

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return macro_f1, y_true, y_pred


# -- RESEARCH REPORTING NOTE: evaluate -----------------------------------------
# Methods description:
#   Model evaluation computed macro-averaged F1-score using a sigmoid threshold
#   of 0.5 for binary classification. Predictions were accumulated on CPU to
#   avoid GPU memory exhaustion on large validation sets.
#
# Parameters to report in paper:
#   threshold : 0.5 (standard binary classification threshold)
#   metric : macro F1-score (treats both classes equally)
#
# Design choices to justify:
#   Macro F1 : gives equal weight to ictal and non-ictal classes regardless
#     of prevalence, making it appropriate for imbalanced seizure detection.
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 17. run_training
# ---------------------------------------------------------------------------
def run_training(model, train_loader, val_loader, lr, weight_decay,
                 max_epochs, patience, device, train_pairs=None,
                 max_grad_norm=1.0, trial=None):
    """Full training loop with early stopping, cosine annealing, and optional Optuna pruning.

    Parameters
    ----------
    model : nn.Module
        Model to train (must already be on device).
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    lr : float
        Initial learning rate for AdamW.
    weight_decay : float
        L2 regularisation coefficient.
    max_epochs : int
        Maximum number of training epochs.
    patience : int
        Early stopping patience (epochs without val F1 improvement).
    device : torch.device
        Target device.
    train_pairs : list of (str or Path, int), optional
        Training file-label pairs for computing pos_weight. If None,
        pos_weight defaults to 1.0 (no class weighting).
    max_grad_norm : float, default 1.0
        Gradient clipping threshold.
    trial : optuna.trial.Trial or None, default None
        Optuna trial for pruning. If None, Optuna calls are skipped.

    Returns
    -------
    best_val_f1 : float
        Best validation macro F1 achieved during training.

    Example
    -------
    >>> f1 = run_training(model, train_ld, val_ld, 1e-3, 1e-4, 100, 10, device,
    ...                   train_pairs=pairs, trial=trial)
    """
    # Compute pos_weight from training data if provided
    if train_pairs is not None:
        pw = compute_pos_weight(train_pairs, device)  # move to device for loss
    else:
        pw = torch.tensor([1.0], dtype=torch.float32).to(device)  # default: no weighting

    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    optimiser = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimiser, T_max=max_epochs, eta_min=lr * 0.01)

    best_val_f1 = 0.0
    epochs_no_improve = 0
    # Store best weights on CPU to avoid a second GPU copy occupying VRAM
    best_state = None

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, train_loader, optimiser, criterion,
                                     device, max_grad_norm)
        val_f1, _, _ = evaluate(model, val_loader, device)
        scheduler.step()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            # Clone weights to CPU to save VRAM (only one model copy on GPU)
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        # Optuna integration (guarded: only runs if trial is provided)
        if trial is not None:
            trial.report(val_f1, epoch)
            if trial.should_prune():
                if best_state is not None:
                    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
                raise TrialPruned()

        if epochs_no_improve >= patience:
            break

    # Restore best weights at end of training
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return best_val_f1


# -- RESEARCH REPORTING NOTE: run_training -------------------------------------
# Methods description:
#   Models were trained using AdamW with cosine annealing learning rate
#   scheduling. Training was terminated early if validation macro F1 did not
#   improve for a specified number of consecutive epochs, and the best
#   model weights (by validation F1) were restored.
#
# Parameters to report in paper:
#   lr : initial learning rate
#   weight_decay : L2 regularisation coefficient
#   max_epochs : maximum training duration
#   patience : early stopping patience
#   max_grad_norm : gradient clipping threshold
#   optimiser : AdamW (Loshchilov and Hutter, 2019)
#   scheduler : CosineAnnealingLR with eta_min = lr * 0.01
#   pos_weight : class imbalance correction (report the value)
#
# Design choices to justify:
#   AdamW : decouples weight decay from gradient update
#   Cosine annealing : smooth LR decay without abrupt drops
#   Best-state on CPU : saves VRAM by storing only one model copy on GPU
#   Dual imbalance handling : WeightedRandomSampler (in loader) + pos_weight
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 18. run_ssl_pretraining
# ---------------------------------------------------------------------------
def run_ssl_pretraining(model, loader, lr, temperature, max_epochs, device,
                        max_grad_norm=1.0):
    """Self-supervised pre-training loop using NT-Xent contrastive loss.

    Parameters
    ----------
    model : SSLModel
        Self-supervised model with encoder and projector.
    loader : DataLoader
        SSL data loader returning (view1, view2) pairs.
    lr : float
        Learning rate for AdamW.
    temperature : float
        NT-Xent temperature parameter.
    max_epochs : int
        Number of pre-training epochs (no early stopping).
    device : torch.device
        Target device.
    max_grad_norm : float, default 1.0
        Gradient clipping threshold.

    Returns
    -------
    losses : list of float
        Average contrastive loss per epoch.

    Example
    -------
    >>> losses = run_ssl_pretraining(ssl_model, loader, 1e-3, 0.5, 100, device)
    """
    optimiser = AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimiser, T_max=max_epochs, eta_min=lr * 0.01)
    losses = []

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for v1, v2 in loader:
            v1, v2 = v1.to(device), v2.to(device)  # transfer views to device
            optimiser.zero_grad()
            z1 = model(v1)                          # project view 1
            z2 = model(v2)                          # project view 2
            loss = nt_xent_loss(z1, z2, temperature)  # contrastive loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimiser.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:                   # log every 10 epochs
            print(f"  SSL epoch {epoch+1}/{max_epochs}: avg loss = {avg_loss:.4f}")

    return losses


# -- RESEARCH REPORTING NOTE: run_ssl_pretraining ------------------------------
# Methods description:
#   Self-supervised pre-training optimised the NT-Xent contrastive loss over
#   pairs of augmented EEG segment views for a fixed number of epochs using
#   AdamW with cosine annealing.
#
# Parameters to report in paper:
#   lr : pre-training learning rate
#   temperature : NT-Xent temperature
#   max_epochs : total pre-training epochs
#   batch_size : determines number of negative pairs in contrastive loss
#
# Design choices to justify:
#   No early stopping : standard for SSL pre-training; validation signal
#     is not available without labels.
#   Fixed epochs : pre-training budget is set a priori based on dataset size.
# -----------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 19. run_finetuning
# ---------------------------------------------------------------------------
def run_finetuning(model, train_loader, val_loader, lr, weight_decay,
                   warmup_epochs, freeze_epochs, max_epochs, patience,
                   device, train_pairs=None, max_grad_norm=1.0):
    """Two-phase fine-tuning: frozen encoder then end-to-end.

    Phase 1 (epochs 0 to freeze_epochs-1): encoder frozen, only classifier trained.
    Phase 2 (epochs freeze_epochs onward): encoder unfrozen with lower LR.

    Parameters
    ----------
    model : FineTunedModel
        Fine-tuning model with encoder and classifier.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    lr : float
        Learning rate for the classifier head.
    weight_decay : float
        L2 regularisation coefficient.
    warmup_epochs : int
        (Reserved for future use; currently unused.)
    freeze_epochs : int
        Number of epochs to keep the encoder frozen.
    max_epochs : int
        Total maximum epochs (both phases combined).
    patience : int
        Early stopping patience on val macro F1.
    device : torch.device
        Target device.
    train_pairs : list, optional
        For pos_weight computation.
    max_grad_norm : float, default 1.0
        Gradient clipping threshold.

    Returns
    -------
    best_val_f1 : float
        Best validation macro F1 achieved.

    Example
    -------
    >>> f1 = run_finetuning(ft_model, train_ld, val_ld, 1e-3, 1e-4, 0, 10, 100, 10, device)
    """
    # Compute pos_weight for class imbalance
    if train_pairs is not None:
        pw = compute_pos_weight(train_pairs, device)
    else:
        pw = torch.tensor([1.0], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    # Phase 1 optimiser: only classifier parameters (encoder is frozen)
    optimiser = AdamW(model.classifier.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimiser, T_max=max_epochs, eta_min=lr * 0.01)

    best_val_f1 = 0.0
    epochs_no_improve = 0
    best_state = None

    for epoch in range(max_epochs):

        # Phase 2 switch: unfreeze encoder and create two-param-group optimiser
        if epoch == freeze_epochs:
            for p in model.encoder.parameters():
                p.requires_grad = True             # unfreeze encoder weights
            # Two param groups: encoder at lr*0.1, classifier at lr
            optimiser = AdamW([
                {"params": model.encoder.parameters(), "lr": lr * 0.1},
                {"params": model.classifier.parameters(), "lr": lr}
            ], weight_decay=weight_decay)
            scheduler = CosineAnnealingLR(optimiser, T_max=max_epochs - freeze_epochs,
                                          eta_min=lr * 0.01)

        train_loss = train_one_epoch(model, train_loader, optimiser, criterion,
                                     device, max_grad_norm)
        val_f1, _, _ = evaluate(model, val_loader, device)
        scheduler.step()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return best_val_f1


# -- RESEARCH REPORTING NOTE: run_finetuning -----------------------------------
# Methods description:
#   Fine-tuning followed a two-phase strategy. In Phase 1, the pre-trained
#   encoder was frozen and only the linear classification head was trained.
#   In Phase 2, the encoder was unfrozen and trained end-to-end with a
#   10x lower learning rate to preserve pre-trained features. Early stopping
#   on validation macro F1 was applied throughout.
#
# Parameters to report in paper:
#   freeze_epochs : duration of frozen-encoder phase
#   lr : classifier learning rate
#   lr * 0.1 : encoder learning rate during unfrozen phase
#   weight_decay, max_epochs, patience : same as run_training
#   pos_weight : class imbalance correction
#
# Design choices to justify:
#   Two-phase training : prevents catastrophic forgetting by stabilising the
#     classifier before updating encoder weights.
#   Encoder LR = classifier LR * 0.1 : standard discriminative fine-tuning
#     practice; preserves pre-trained representations.
#   Separate optimiser after unfreeze : ensures correct momentum statistics
#     for the newly trainable encoder parameters.
# -----------------------------------------------------------------------------
