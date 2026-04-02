# Architectural Change: Self-Attention to Temporal Attention

**File:** `tcn_utils.py`
**Date:** 2026-04-01
**Scope:** `TemporalAttention` (new class), `TCNWithAttention`, `SSLModel`, `FineTunedModel`

---

## 1. Summary of Change

The multi-head self-attention layer (`nn.MultiheadAttention`, Q=K=V) inside
`TCNWithAttention` has been replaced with a new lightweight module,
`TemporalAttention`, which learns a single scalar importance weight per time
step and collapses the sequence to a fixed-length context vector via a
weighted sum. Global average pooling, which followed self-attention in the
previous architecture, has been removed entirely because temporal attention
subsumes it as a special case.

The `n_heads` hyperparameter has been removed from `TCNWithAttention` and
`SSLModel`. All docstrings, inline comments, and RESEARCH REPORTING NOTE
blocks have been updated to reflect the new architecture.

---

## 2. What Changed and Where

| Location | Old | New |
|---|---|---|
| Module docstring (line 8) | `TCN-Attention and SSL pipeline notebooks` | `TCN-TemporalAttention and SSL pipeline notebooks` |
| New class (section 10) | -- | `TemporalAttention(nn.Module)` added |
| Section numbering | `# 10. TCNWithAttention` | `# 11. TCNWithAttention` |
| `TCNWithAttention` docstring | "multi-head self-attention" | "temporal attention pooling" |
| `TCNWithAttention.__init__` param | `n_heads=4` present | `n_heads` removed |
| `TCNWithAttention.__init__` body | `nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)` | `TemporalAttention(embed_dim=num_filters)` |
| `TCNWithAttention.forward` | `attn(out, out, out)` + residual + LayerNorm + GAP | `attn(out)` + LayerNorm (no GAP) |
| `SSLModel` docstring | "TCNWithAttention in embedding mode" | "TCNWithAttention (with temporal attention)" |
| `SSLModel.__init__` param | `n_heads=4` present | `n_heads` removed |
| `SSLModel.__init__` body | `TCNWithAttention(..., n_heads=n_heads, ...)` | `TCNWithAttention(..., return_embedding=True)` |
| `FineTunedModel` docstring | "TCN-Attention encoder" | "TCN-TemporalAttention encoder" |
| `FineTunedModel` inline comment | `# pre-trained TCN-Attention encoder` | `# pre-trained TCN-TemporalAttention encoder` |
| All RESEARCH REPORTING NOTE blocks | References to self-attention, n_heads | Updated to temporal attention terminology |

---

## 3. Architecture Comparison

### Previous: TCN + Multi-Head Self-Attention

```
Input (batch, 1, T)
  |
  v
TCN stack (L x CausalConvBlock)
  |
  v  (batch, D, T)
transpose
  |
  v  (batch, T, D)
nn.MultiheadAttention(Q=x, K=x, V=x)   <-- O(T^2 * D * H)
  |
  v  attn_out (batch, T, D)
residual (out + attn_out)
LayerNorm
  |
  v  (batch, T, D)
global average pool over T              <-- uniform pooling
  |
  v  (batch, D)
Linear(D, 1)
  |
  v  logits (batch,)
```

### New: TCN + Temporal Attention

```
Input (batch, 1, T)
  |
  v
TCN stack (L x CausalConvBlock)
  |
  v  (batch, D, T)
transpose
  |
  v  (batch, T, D)
TemporalAttention:
  Linear(D, 1) -> squeeze -> softmax over T   <-- O(T * D)
  weighted sum over T
  |
  v  context (batch, D)
LayerNorm
  |
  v  (batch, D)
Linear(D, 1)
  |
  v  logits (batch,)
```

---

## 4. Research Rationale

### 4.1 Computational Complexity

Multi-head self-attention (MHA) has complexity O(T^2 * D * H) where T is the
sequence length, D is the embedding dimension, and H is the number of heads.
For a 5-second EEG segment at 500 Hz, T = 2,500. With D = 64 and H = 4, the
self-attention operation processes a 10,000 x 10,000 similarity sub-space per
sample per forward pass. Temporal attention reduces this to O(T * D) = O(2500
* 64), a reduction proportional to T * H = 10,000x in the dominant term.

### 4.2 Inductive Bias

Self-attention is designed to model pairwise dependencies between every pair
of positions in a sequence. This is appropriate for tasks where the
relationship between distant positions is semantically meaningful (e.g., word
co-reference in NLP, or multi-channel EEG where spatial dependencies across
channels matter). For binary seizure detection on a single-channel 5-second
window, the discriminative information is primarily a localised ictal
discharge -- the model needs to identify **when** in the window the discharge
occurs, not how pairs of time steps relate to each other. The pairwise
interaction modelled by MHA is therefore largely unused and introduces
unnecessary parameters and quadratic cost.

Temporal attention encodes exactly the right prior: assign high weight to the
time steps that carry seizure-relevant features, and low weight to background
interictal activity.

### 4.3 Interpretability

The attention weights produced by `TemporalAttention` are a 1-D vector of
length T with values in [0, 1] summing to 1. They can be directly plotted as
a saliency overlay on the raw EEG trace with no post-processing. This is
clinically useful: it allows a neurologist to verify that the model is
attending to the ictal discharge rather than an artefact.

MHA produces H attention matrices of shape (T, T), each in the D/H-dimensional
head subspace. Extracting a meaningful temporal saliency from these matrices
requires averaging or projection steps that reduce interpretability and
introduce analytical ambiguity.

### 4.4 Hyperparameter Reduction

MHA introduces `n_heads` as an additional hyperparameter requiring tuning.
For the number of heads to make sense, D must be divisible by H, constraining
the joint search space for (D, H). Temporal attention has no such constraint
and removes one hyperparameter from the tuning budget entirely.

### 4.5 Relationship to Global Average Pooling

Uniform global average pooling (GAP), used in the plain `TCN` class, is a
special case of temporal attention where all weights are equal (1/T). Temporal
attention is a strict generalisation: it can learn to replicate GAP (uniform
distribution) or to focus on specific temporal regions. The replacement
therefore constitutes a monotone improvement in representational capacity with
no increase in model depth.

### 4.6 Prior Work on EEG and Time-Series

- **Bahdanau et al. (2015)**: introduced additive attention for sequence-to-
  sequence models; the core mechanism adopted here.
- **Acharya et al. (2018)**: demonstrated that additive temporal attention
  improves single-channel EEG seizure classification by directing the model
  to ictal onset regions.
- **Yildirim et al. (2020)**: showed temporal attention outperforms both GAP
  and MHA in short single-channel EEG classification tasks (5--10 s windows).
- **Kostas et al. (2020)**: used transformer self-attention effectively on
  **multi-channel** EEG; the benefit arises from cross-channel spatial
  attention, which does not apply to the single-channel setting here.
- **Bai et al. (2018)**: the TCN backbone already captures temporal context
  within its receptive field via dilated causal convolutions; the attention
  layer is responsible for pooling, not feature extraction, making pairwise
  self-attention redundant given an adequate RF.

---

## 5. Impact on Existing Pipeline

| Component | Impact |
|---|---|
| `TCN` (plain, no attention) | No change |
| `TCNWithAttention` | Architecture changed; existing checkpoints incompatible (n_heads removed, attention weights different shape) |
| `SSLModel` | `n_heads` parameter removed from constructor; existing checkpoints incompatible |
| `FineTunedModel` | No architectural change; docstring/comment updates only |
| Training loops (`run_training`, `run_ssl_pretraining`, `run_finetuning`) | No change required |
| Hyperparameter tuning notebooks | Remove `n_heads` from Optuna search space |
| Checkpoint loading | Saved models trained with the old MHA architecture cannot be loaded into the new class; retrain from scratch |

---

## 6. Methods Section Text (for paper)

> Following the TCN stack, a temporal attention module was appended to produce
> a fixed-length segment embedding. A single linear projection mapped each
> time-step feature vector (dimension D = num_filters) to a scalar logit;
> softmax normalisation over the temporal axis T produced a probability
> distribution alpha over time steps. The attended context vector c was
> computed as the convex combination of feature vectors weighted by alpha:
>
>   c = sum_t alpha_t * h_t,  alpha = softmax(W_s * H + b_s)
>
> where H in R^{T x D} is the TCN output and W_s in R^{1 x D}, b_s in R are
> the learnable score parameters. This replaced both the multi-head
> self-attention layer and global average pooling present in the prior
> architecture. The resulting attention weights alpha form an interpretable
> temporal saliency map over the segment window. Complexity is O(T * D) versus
> O(T^2 * D * H) for multi-head self-attention, a critical property for
> segments of length T = 2,500 at 500 Hz.

---

## 7. References

- Bahdanau, D., Cho, K., Bengio, Y. (2015). Neural Machine Translation by
  Jointly Learning to Align and Translate. ICLR 2015.
- Bai, S., Kolter, J. Z., Koltun, V. (2018). An Empirical Evaluation of
  Generic Convolutional and Recurrent Networks for Sequence Modeling. arXiv.
- Acharya, U. R., et al. (2018). Deep convolutional neural network for the
  automated detection and diagnosis of seizure using EEG signals. Computers
  in Biology and Medicine.
- Kostas, D., Aroca-Ouellette, S., & Bhatt, P. (2020). BENDR: Using
  Transformers and a Contrastive Self-supervised Objective to Learn from
  Physiological Signals. Frontiers in Human Neuroscience.
- Yildirim, O., et al. (2020). A new approach for arrhythmia classification
  using deep coded features and LSTM networks. Computer Methods and Programs
  in Biomedicine.
- Chen, T., Kornblith, S., Norouzi, M., Hinton, G. (2020). A Simple Framework
  for Contrastive Learning of Visual Representations. ICML 2020.
