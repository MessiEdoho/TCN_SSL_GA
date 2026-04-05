# Architectural Change Log: Temporal Attention in TCNWithAttention

**File:** `tcn_utils.py`
**Initial change:** 2026-04-01 (self-attention to temporal attention)
**Latest revision:** 2026-04-04 (upgrade to two-layer additive attention + SSL removal)

---

## 1. Summary of Changes

### Phase 1 (2026-04-01): Self-Attention to Temporal Attention
The multi-head self-attention layer (`nn.MultiheadAttention`, Q=K=V) inside
`TCNWithAttention` was replaced with a lightweight `TemporalAttention` module
(single `nn.Linear(num_filters, 1)` scorer). The `n_heads` hyperparameter was
removed. Global average pooling was replaced by attention-weighted pooling.

### Phase 2 (2026-04-04): Two-Layer Additive Attention + SSL Removal
The single-layer `TemporalAttention` scorer inside `TCNWithAttention` was
upgraded to a two-layer additive attention mechanism (tanh + linear) matching
`MultiScaleTCNWithAttention` exactly. This adds two tunable hyperparameters:
`attention_dim` and `attention_dropout`. SSL-related classes and functions
(`SSLModel`, `FineTunedModel`, `EEGSegmentSSLDataset`, `make_ssl_loader`,
`nt_xent_loss`, `run_ssl_pretraining`, `run_finetuning`) were removed from
`tcn_utils.py` -- SSL will be developed in a separate study.

---

## 2. Current Architecture: TCN + Two-Layer Additive Attention

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
Two-layer additive attention:
  e_t = tanh(W_a h_t + b_a)       W_a in R^(attention_dim x D)
  score_t = v^T e_t               v in R^(attention_dim)
  alpha_t = softmax({score_t})     sums to 1 over T
  context = sum_t alpha_t * h_t    (batch, D)
  |
  v  context (batch, D)
Dropout(attention_dropout)
  |
  v  (batch, D)
Linear(D, 1)
  |
  v  logits (batch,)
```

Complexity: O(T * D * attention_dim), still linear in T.

---

## 3. What Changed and Where (Phase 2)

| Location | Before (Phase 1) | After (Phase 2) |
|---|---|---|
| `TCNWithAttention.__init__` params | `(num_layers, num_filters, kernel_size, dropout, return_embedding, fs)` | Added `attention_dim=64, attention_dropout=0.0` |
| `TCNWithAttention.__init__` body | `self.attn = TemporalAttention(embed_dim=num_filters)` + `self.attn_norm = LayerNorm` | `self.attention_fc = Linear(num_filters, attention_dim)` + `self.attention_v = Linear(attention_dim, 1)` + `self.attention_drop = Dropout(attention_dropout)` |
| `TCNWithAttention.forward()` | `context, _ = self.attn(out)` + LayerNorm | `tanh(attention_fc(feat_t))` -> `attention_v(e)` -> softmax -> weighted sum -> dropout |
| `TCNWithAttention.get_attention_weights()` | Called `self.attn(out)` | Calls `attention_fc` + `attention_v` + softmax directly |
| `tune_temporal_attention.py` search space | Only training HPs (lr, wd, bs, max_grad_norm) | Added `attention_dim` [32,64,128] and `attention_dropout` [0.0-0.4] |
| `TCNTemporalAttention.py` `build_model()` | Only `backbone_hp` | Now passes `attn_hp["attention_dim"]` and `attn_hp["attention_dropout"]` |
| `SSLModel` | Present | **Removed** (SSL in separate study) |
| `FineTunedModel` | Present | **Removed** (SSL in separate study) |
| `EEGSegmentSSLDataset` | Present | **Removed** |
| `make_ssl_loader` | Present | **Removed** |
| `nt_xent_loss` | Present | **Removed** |
| `run_ssl_pretraining` | Present | **Removed** |
| `run_finetuning` | Present | **Removed** |
| `import torch.nn.functional as F` | Present (used by SSLModel) | **Removed** (no longer needed) |

---

## 4. Research Rationale

### 4.1 Why Two-Layer Attention (not single linear)

The single linear scorer (`W^T h_t + b`) can only learn a fixed linear weighting
of feature channels. It cannot learn that "high activation in channel 3 combined
with low activation in channel 7 means this time step is important."

The two-layer scorer (`tanh(W_a h_t + b_a)` followed by `v^T e_t`) introduces a
nonlinearity between the projection and the scoring, allowing the attention to
learn nonlinear feature interactions when deciding which time steps matter.

### 4.2 Why Match MultiScaleTCNWithAttention

The M1-vs-M2 (single-branch TCN vs TCN+Attention) and M3-vs-M4 (multi-scale vs
multi-scale+attention) ablations must test the same attention mechanism. If M2
used a weaker attention than M4, any performance difference could be attributed
to the attention formulation rather than the multi-scale structure. Matching them
eliminates this confound.

### 4.3 Tunable Parameters

`attention_dim` controls the capacity of the scoring function independently of
`num_filters`. Larger values allow more complex feature interactions but risk
overfitting on small datasets. `attention_dropout` regularises the context vector,
preventing the model from over-relying on a few dominant time steps.

### 4.4 Computational Complexity

O(T * D * attention_dim) is still linear in sequence length T. For T=2500,
D=64, attention_dim=64: 10.2M multiply-adds per sample, versus 400M+ for
multi-head self-attention with H=4 heads.

### 4.5 Interpretability

The attention weights alpha_t are still a 1-D vector of length T with values in
[0, 1] summing to 1. They can be extracted via `get_attention_weights()` (present
on both `TCNWithAttention` and `MultiScaleTCNWithAttention`) and plotted as a
temporal saliency overlay on raw EEG.

---

## 5. Impact on Existing Pipeline

| Component | Impact |
|---|---|
| `TCN` (plain, no attention) | No change |
| `TCNWithAttention` | `__init__` signature changed (new `attention_dim`, `attention_dropout` params); old checkpoints incompatible |
| `TemporalAttention` class | Still present in tcn_utils.py but no longer used by TCNWithAttention; kept for reference |
| `tune_temporal_attention.py` | Search space expanded to include `attention_dim` and `attention_dropout`; `max_grad_norm` removed from tuning |
| `TCNTemporalAttention.py` | `build_model()` now passes attention params from `best_attention_params.json` |
| `MultiScaleTCN` | No change |
| `MultiScaleTCNWithAttention` | No change (already had this attention formulation) |
| `TCN.py` | No change |
| `MultiScaleTCN.py` | No change |
| SSL classes/functions | **Removed from tcn_utils.py** (SSLModel, FineTunedModel, EEGSegmentSSLDataset, make_ssl_loader, nt_xent_loss, run_ssl_pretraining, run_finetuning) |
| `best_attention_params.json` | Will now contain `attention_dim` and `attention_dropout` under `"hyperparameters"` |
| Checkpoint loading | All M2 checkpoints must be retrained from scratch |

---

## 6. Methods Section Text (for paper)

> Following the TCN stack, a two-layer additive temporal attention mechanism
> was appended to produce a fixed-length segment representation. For each
> time-step feature vector h_t in R^F (where F = num_filters), energy scores
> were computed as e_t = tanh(W_a h_t + b_a), where W_a in R^(D_a x F) is a
> learned projection matrix and D_a is the attention dimension. Scalar scores
> v^T e_t were normalised with softmax over the temporal axis to produce
> attention weights alpha_t = softmax({v^T e_t}). The context vector
> c = sum_t alpha_t h_t was regularised with dropout (rate p_a) before the
> linear classification head. This formulation was used consistently across
> both TCNWithAttention (M2) and MultiScaleTCNWithAttention (M4) to ensure
> the M1-vs-M2 and M3-vs-M4 ablations test identical attention mechanisms.
> Attention hyperparameters (D_a, p_a) were tuned via Optuna TPE with the
> TCN backbone frozen, then all parameters were trained jointly for 100
> epochs in the final training run.

---

## 7. References

- Bahdanau, D., Cho, K., Bengio, Y. (2015). Neural Machine Translation by
  Jointly Learning to Align and Translate. ICLR 2015.
- Bai, S., Kolter, J. Z., Koltun, V. (2018). An Empirical Evaluation of
  Generic Convolutional and Recurrent Networks for Sequence Modeling. arXiv.
- Acharya, U. R., et al. (2018). Deep convolutional neural network for the
  automated detection and diagnosis of seizure using EEG signals. Computers
  in Biology and Medicine.
- Yildirim, O., et al. (2020). A new approach for arrhythmia classification
  using deep coded features and LSTM networks. Computer Methods and Programs
  in Biomedicine.
