# Interpretability Analysis Documentation

**Script:** `interpretability_analysis.py`
**Date:** 2026-04-04
**Scope:** Post-training interpretability for all four seizure detection models (M1-M4)

---

## 1. Overview

This script produces 12 interpretability figures and structured data files that explain
**how** and **why** each model makes seizure detection decisions. It operates on either the
validation partition (development) or the independent test partition (paper reporting).

### Execution Modes

| Mode | Command | Purpose | Output Directory |
|------|---------|---------|-----------------|
| Development | `python interpretability_analysis.py --partition val` | Verify pipeline correctness before touching test data | `outputs/interpretability/val/` |
| Paper | `python interpretability_analysis.py --partition test` | Produce publication figures on unbiased data | `outputs/interpretability/test/` |

### Why Two Modes

The validation partition carries indirect optimisation bias: early stopping monitored
validation F1, the Youden J threshold was selected on validation data, and
hyperparameters were tuned using validation F1 as the objective. Reporting
interpretability results from this partition is analogous to reporting training
accuracy -- the model was effectively optimised partly on it. The test partition
was never accessed during training, tuning, or threshold selection, making it the
only source of genuinely unbiased interpretability evidence.

---

## 2. Three Interpretability Layers

### Layer 1: Temporal Attention Saliency (M2, M4)

**What it finds out:** Whether the attention mechanism has learned to focus on
seizure-relevant time steps. This is the most clinically important layer because
it provides a built-in explanation that does not require any post-hoc method.

**Method:** The trained TCNWithAttention and MultiScaleTCNWithAttention models
contain a two-layer additive attention mechanism that produces per-timestep
weights alpha_t via:

    e_t = tanh(W_a h_t + b_a)
    alpha_t = softmax(v^T e_t)

These weights sum to 1 and form a probability distribution over T=2500 time steps
(5 seconds at 500 Hz). They are extracted via `get_attention_weights()` with no
post-hoc computation.

**Clinical Relevance:** A clinician interpreting a seizure detection alarm needs
to know *which part of the 5-second EEG window triggered the alarm*. If the
attention weights peak at the ictal discharge region, the clinician can visually
verify the detection against the raw EEG trace. If the weights are diffuse
(spread uniformly), the model is making a prediction without localised evidence,
which reduces clinical trust. In preclinical drug trials, knowing the temporal
location of the detected seizure within the segment informs the neurologist
about seizure onset timing -- a key pharmacological endpoint.

**Procedures:**

| Function | What It Computes | Clinical Question Answered |
|----------|-----------------|--------------------------|
| `collect_attention_weights()` | alpha_t for all segments, separated by class | Where does the model look within each segment? |
| `compute_entropy()` | H(alpha) = -sum(alpha * ln(alpha)) per segment | Is the model focused (low H) or uncertain (high H)? |
| `plot_figure_A()` | Mean alpha_t +/- std per class | Does the model attend to different time regions for seizures vs normal? |
| `plot_figure_B()` | Delta alpha_t = mean(ictal) - mean(non-ictal) | At which specific time steps is attention class-dependent? |
| `plot_figure_C()` | TP vs FN heatmap grid with EEG + attention overlay | Do missed seizures correspond to segments where attention failed? |
| `plot_figure_D()` | Attention entropy violin plots with Mann-Whitney U test | Is the model significantly more focused on seizures than on normal activity? |
| `plot_figure_E()` | Single best ictal segment with attention overlay | Case study: what does focused attention look like on a real seizure? |

**Key Statistical Test (Figure D):**

Shannon entropy H(alpha) measures attention concentration:

    H(alpha) = -sum_j alpha_j * ln(alpha_j)   (nats)
    H_max = ln(T) = ln(2500) = 7.824 nats    (uniform attention)

Low entropy = focused attention (narrow peak over seizure).
High entropy = diffuse attention (spread across the segment).

**Why entropy and not mean:** Because sum(alpha_j) = 1 (softmax constraint), the
mean alpha_j = 1/T is constant for all segments. It carries zero information
about the shape of the distribution. Entropy measures shape.

The Mann-Whitney U test (two-sided) compares H(alpha) between ictal and non-ictal
segments. Effect size r = U / (n1 * n2). We hypothesise that ictal segments have
lower entropy (more focused attention) than non-ictal segments.

---

### Layer 2: Branch Contribution Analysis (M3, M4)

**What it finds out:** Whether each of the three parallel branches in the
Multi-Scale TCN contributes independently to seizure detection, and whether the
branches capture genuinely different temporal phenomena.

**Method:** Single-branch ablation zeros one branch's output before fusion and
measures the F1 drop. Pairwise ablation keeps two branches and zeros the third.
Branch activation profiles measure mean absolute activation per time step per branch.

**Clinical Relevance:** Rodent seizures have temporal structure at multiple scales
simultaneously -- sharp spikes (milliseconds), rhythmic bursts (hundreds of
milliseconds), and sustained envelopes (seconds). If all three branches
contribute independently, the multi-scale architecture is justified: each branch
captures a different clinical feature. If one branch contributes nothing, the
architecture has redundancy that could be eliminated for deployment on resource-
constrained monitoring hardware. In drug development, knowing which temporal scale
is most affected by a compound (e.g., a drug that reduces spike frequency but not
seizure duration) informs the pharmacological mechanism of action.

**Procedures:**

| Function | What It Computes | Clinical Question Answered |
|----------|-----------------|--------------------------|
| `run_branch_ablation()` | Baseline F1 minus ablated F1 per branch | Does each branch contribute independently to detection accuracy? |
| `collect_branch_activations()` | Mean abs activation per branch per time step | Do branches respond to different temporal features (spikes vs envelopes)? |
| `plot_figure_F()` | Branch contribution bar chart | Which branch matters most? Is one dispensable? |
| `plot_figure_G()` | Receptive field diagram (architectural, no data) | Do branches cover distinct temporal scales? Is there excessive overlap? |
| `plot_figure_H()` | Pairwise ablation heatmap | Are any two branches jointly sufficient, or are all three needed? |
| `plot_figure_L()` | Branch activation temporal profiles by class | Do branches detect genuinely different temporal phenomena? |

**Threshold-Selection Bias Warning:**

Branch ablation measures F1, which depends on the classification threshold. Because
the threshold was selected by Youden J on the validation partition, ablation F1
values computed on validation data carry threshold-selection bias -- they are
optimistic for the threshold that was tuned for that exact partition. This is
why `--partition test` matters most for Layer 2. The script logs a WARNING when
ablation is run on validation data.

---

### Layer 3: Feature Map Analysis (All Models)

**What it finds out:** Which input time steps most strongly influence the model's
prediction, regardless of whether the model has an attention mechanism or not.
This is the only layer that applies to M1 (plain TCN), which has no built-in
interpretability mechanism.

**Method:** Two complementary approaches:

1. **1-D GradCAM:** Gradient-weighted Class Activation Mapping adapted for 1-D
   temporal signals. Registers forward and backward hooks on the last
   convolutional layer, computes channel-wise gradient weights, and produces
   a per-timestep saliency map.

   Formula:
       w^k = (1/T) sum_t (d_y^c / d_A^k(t))    (channel weight)
       L_GradCAM(t) = ReLU(sum_k w^k * A^k(t))  (saliency)

   Normalised to [0, 1] per segment.

2. **Occlusion Sensitivity:** Model-agnostic perturbation method. A sliding
   zero-mask (50 samples = 0.1 s, stride 25 samples) is applied across the
   input, and the drop in predicted seizure probability is measured.

   Formula:
       S(p) = P(y=1 | x) - P(y=1 | x with x[p:p+W] = 0)

   Positive S(p) = occluding this region reduced seizure probability = the
   region was important for the prediction.

**Clinical Relevance:** GradCAM and occlusion sensitivity answer the question:
"if I removed this part of the EEG, would the model still detect the seizure?"
For a clinician reviewing a flagged segment, this identifies the specific EEG
morphology that triggered the detection. Agreement between GradCAM and occlusion
validates that the gradient-based explanation is faithful to the model's actual
input dependencies. Disagreement indicates GradCAM may not be a reliable
explanation -- an important limitation to report.

For M2 and M4, Figure K overlays GradCAM against the built-in attention weights
and computes Pearson r. If r > 0.7, attention faithfully reflects gradient
importance -- "attention IS explanation." If r < 0.5, attention diverges from
what actually matters to the model -- "attention is not explanation" (Jain &
Wallace, 2019). This distinction is critical for papers claiming that attention
provides interpretability.

**Procedures:**

| Function | What It Computes | Clinical Question Answered |
|----------|-----------------|--------------------------|
| `compute_gradcam_1d()` | Per-timestep gradient saliency for all segments | Which EEG time steps most influence the model's gradient? |
| `compute_occlusion_sensitivity()` | Per-timestep sensitivity to input zeroing | Which regions, when removed, cause the largest drop in seizure probability? |
| `plot_figure_I()` | Mean GradCAM saliency per model per class | Is gradient importance concentrated on seizure regions across all 4 models? |
| `plot_figure_J()` | Mean occlusion sensitivity per model per class | Does model-agnostic analysis agree with gradient-based analysis? |
| `plot_figure_K()` | GradCAM vs attention overlay with Pearson r | Does the attention mechanism faithfully explain model predictions? |

---

## 3. Output File Structure

```
outputs/interpretability/{val|test}/
    figures/
        fig_A_mean_attention_profile_{partition}.png
        fig_B_differential_attention_{partition}.png
        fig_C_attention_heatmap_grid_{partition}.png
        fig_D_attention_entropy_{partition}.png
        fig_E_example_segment_overlay_{partition}.png
        fig_F_branch_ablation_bar_{partition}.png
        fig_G_branch_rf_diagram_{partition}.png
        fig_H_pairwise_ablation_{partition}.png
        fig_I_gradcam_saliency_{partition}.png
        fig_J_occlusion_sensitivity_{partition}.png
        fig_K_gradcam_vs_attention_{partition}.png
        fig_L_branch_activation_{partition}.png
    attention_weights/         -- saved .npy arrays per model per class
    entropy/                   -- H(alpha) arrays + Mann-Whitney stats JSON
    branch_activations/        -- branch activation arrays per model per class
    branch_ablation/           -- ablation results JSON per model
    gradcam/                   -- GradCAM arrays per model per class
    occlusion/                 -- occlusion sensitivity arrays per model per class
    logs/                      -- full DEBUG-level log
    interpretability_report_{partition}.md  -- summary with computed values
```

---

## 4. Methods Section Template (for paper)

### 4.1 Temporal Attention Analysis

> "Temporal attention weights alpha_t were extracted from M2 (TCNWithAttention)
> and M4 (MultiScaleTCNWithAttention) using the built-in `get_attention_weights()`
> method, which returns the softmax-normalised scores of the two-layer additive
> attention mechanism (Bahdanau et al., 2015) without requiring any post-hoc
> attribution method. For each segment, the T=2500 attention weights (5 seconds at
> 500 Hz) form a probability distribution over time steps, with alpha_t indicating
> the learned importance of time step t for the classification decision.
>
> To quantify attention focus, Shannon entropy H(alpha) = -sum_j alpha_j ln(alpha_j)
> was computed per segment. Because the softmax constraint ensures sum_j alpha_j = 1
> for all segments, the mean weight (1/T) is constant and uninformative; entropy
> captures the shape (concentration vs diffuseness) of the attention distribution.
> Lower entropy indicates focused attention (narrow peak), expected for ictal
> segments; higher entropy indicates diffuse attention (uniform spread), expected for
> non-ictal segments. The difference in entropy between classes was tested using the
> Mann-Whitney U test (two-sided) with effect size r = U/(n_ictal * n_nonictal).
>
> All attention analyses were conducted on the independent test partition, which
> was never accessed during training, hyperparameter tuning, or threshold selection."

### 4.2 Branch Contribution Analysis

> "To assess the independent contribution of each branch in the Multi-Scale TCN
> architectures (M3, M4), single-branch ablation was performed: for each branch
> b in {1, 2, 3} (corresponding to dilation schedules [1,2,4], [2,4,8], [4,8,16]),
> the branch output was zeroed before fusion, and the resulting macro F1-score was
> evaluated on the test partition. The contribution of branch b was defined as
> Contribution_b = F1_baseline - F1_ablated_b. Pairwise ablation (retaining two
> branches and zeroing the third) assessed whether any two branches were jointly
> sufficient for near-baseline performance.
>
> Branch temporal activation profiles were computed as the mean absolute activation
> a_b(t) = (1/N)(1/F) sum_s sum_f |branch_b^f(x_s, t)| averaged over all segments
> and filters, providing a per-branch temporal response pattern that reveals whether
> branches capture distinct ictal phenomena (e.g., sharp spikes vs sustained envelopes)."

### 4.3 Feature Map Analysis

> "Gradient-weighted Class Activation Mapping (GradCAM; Selvaraju et al., 2017) was
> adapted for 1-D temporal signals. Channel-wise gradient weights w^k = (1/T) sum_t
> (d_score / d_A^k(t)) were computed at the last convolutional layer of each model,
> and the resulting saliency map L_GradCAM(t) = ReLU(sum_k w^k A^k(t)) was normalised
> to [0, 1] per segment. Occlusion sensitivity (Zeiler & Fergus, 2014) was computed
> by sliding a 50-sample (0.1 s) zero-mask across the input at 25-sample stride and
> measuring the drop in predicted seizure probability S(p) = P(y=1|x) - P(y=1|x_masked).
>
> For M2 and M4, the temporal agreement between attention weights and GradCAM saliency
> was quantified using Pearson correlation r between the mean ictal profiles. A high
> correlation (r > 0.7) indicates that attention faithfully reflects gradient-based
> input importance; a low correlation (r < 0.5) suggests that attention may not
> constitute a faithful explanation of the model's input dependencies (Jain & Wallace,
> 2019; Wiegreffe & Pinter, 2019)."

---

## 5. Results Section Template (for paper)

> "**Temporal attention analysis.** Attention entropy H(alpha) was significantly lower
> for ictal segments than for non-ictal segments in both M2 (median ictal: [X] nats,
> median non-ictal: [Y] nats; Mann-Whitney U = [Z], p < [P], r = [R]) and M4.
> This confirms that the temporal attention mechanism learned to focus on seizure-
> relevant time steps rather than distributing attention uniformly. The differential
> attention profile (Figure B) revealed that the attention difference was concentrated
> in [describe temporal region], consistent with the known morphology of rodent
> seizures. Visual inspection of individual segments (Figure C) showed that true
> positive detections exhibited sharply focused attention peaks coinciding with the
> ictal discharge, while false negative segments showed diffuse attention without
> localised focus, suggesting that attention failure is a mechanism of missed
> detections.
>
> **Branch contribution analysis.** Single-branch ablation (Figure F) revealed that
> all three branches contributed independently in M3 (B1: [X], B2: [Y], B3: [Z] F1
> drop) and M4. Branch 3 (coarse scale, dilations [4,8,16]) contributed the largest
> F1 drop in both models, consistent with its role in capturing sustained seizure
> envelopes. Pairwise ablation (Figure H) showed that no two branches were sufficient
> to recover baseline performance, confirming the complementary nature of the multi-
> scale design. Branch activation profiles (Figure L) demonstrated that Branch 1
> exhibited sharp activation peaks at ictal onset (consistent with fine-scale spike
> detection), while Branch 3 showed broad sustained elevation throughout the ictal
> period (consistent with coarse-scale envelope detection).
>
> **GradCAM and occlusion validation.** GradCAM saliency and occlusion sensitivity
> showed [consistent / inconsistent] temporal profiles across all four models
> (Figure I, J), confirming that gradient-based explanations [are / are not] faithful
> representations of the models' input dependencies. For M2, the Pearson correlation
> between mean attention and mean GradCAM saliency over ictal segments was r = [X],
> indicating that temporal attention [faithfully reflects / does not reliably reflect]
> gradient importance. [If r > 0.7: This validates the use of attention weights as
> a built-in interpretability mechanism.] [If r < 0.5: This result suggests that
> attention weights should be supplemented with post-hoc methods when explaining
> individual predictions.]"

---

## 6. Clinical Relevance Summary

| Analysis | Clinical Question | Why It Matters for Drug Development |
|----------|------------------|-------------------------------------|
| Attention saliency (Figs A, B, E) | Where within the 5-second EEG window did the model detect the seizure? | Enables neurologist verification of automated detections; informs seizure onset timing as a pharmacological endpoint |
| Attention entropy (Fig D) | Is the model more certain about seizures than normal activity? | Low entropy = high-confidence detection; useful for triaging alarms in 24/7 monitoring |
| TP vs FN attention (Fig C) | Why did the model miss some seizures? | Identifies failure modes: diffuse attention on missed seizures suggests the model could not localise the ictal feature |
| Branch ablation (Figs F, H) | Does each temporal scale contribute independently? | Confirms multi-scale design is justified; identifies which scales matter most for seizure detection |
| Branch activation (Fig L) | Do branches detect different temporal features? | Links model behaviour to clinical features (spikes vs bursts vs envelopes); informs which seizure features a drug affects |
| GradCAM (Fig I) | Which input features most influence the model's gradient? | Provides model-agnostic evidence of what drives the prediction, applicable even to M1 (no attention) |
| Occlusion (Fig J) | Which input regions are necessary for the prediction? | Validates GradCAM; identifies whether the model relies on the ictal discharge or on artefacts |
| GradCAM vs Attention (Fig K) | Is attention a faithful explanation of the model's predictions? | Determines whether attention can be trusted as an explanation or requires supplementation with post-hoc methods |
| RF diagram (Fig G) | Do branches cover distinct temporal scales? | Verifies architectural design hypothesis; confirms no redundant branches for hardware-constrained deployment |

---

## 7. References

- Bahdanau, D., Cho, K., Bengio, Y. (2015). Neural Machine Translation by
  Jointly Learning to Align and Translate. ICLR 2015.
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep
  Networks via Gradient-based Localization. ICCV 2017.
- Zeiler, M. D. & Fergus, R. (2014). Visualizing and Understanding Convolutional
  Networks. ECCV 2014.
- Jain, S. & Wallace, B. C. (2019). Attention is not Explanation. NAACL 2019.
- Wiegreffe, S. & Pinter, Y. (2019). Attention is not not Explanation. EMNLP 2019.
- Luttjohann, A., et al. (2009). Dynamics of absence seizure discharges in
  WAG/Rij rats. European Journal of Neuroscience.
