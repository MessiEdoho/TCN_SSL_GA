# Reporting the post-processing ablation in the paper

This note sets out how to report the post-processing ablation in the IEEE Access
manuscript. All values come from
`MultiScaleTCN/evaluation/postproc_ablation/ablation_paper_table.csv` and
`ablation_all_configs.csv`, produced by `postproc_ablation.py`.

Model naming: the paper's **M1** (MultiScale TCN) is **M3** in the repository.
All text below uses the paper's naming.

## 1. Key findings

- **Post-processing as a whole is essential.** Compared with no post-processing,
  the full pipeline removed 95.6% of false alarms (12,219 to 535), and FAR/hr
  fell from 1.807 to 0.079. F1 rose from 0.097 to 0.675. The cost was 39 missed
  seizures, with recall falling from 0.965 to 0.908.
- **Minimum-duration filtering is the dominant stage.** Removing it from the full
  pipeline multiplied false alarms by 5.3 (535 to 2,840) and cut F1 from 0.675 to
  0.313. It is also the main cause of lost sensitivity: removing it recovers 32
  of the 39 seizures lost.
- **Refractory merging is a low-cost gain.** Removing it adds 224 false alarms and
  lowers F1 by 0.073, with no change in recall.
- **Maximum-duration filtering is also a low-cost gain.** Removing it adds 86
  false alarms and lowers F1 by 0.030, with no change in recall.
- **Smoothing does not improve F1 in the full pipeline.** Removing it raises F1
  slightly (0.675 to 0.686) and lowers FAR/hr (0.079 to 0.072), but loses 9
  seizures (recall 0.908 to 0.895) and delays detection by 0.93 s (2.66 s to
  3.59 s). The manuscript must report this as a sensitivity-versus-false-alarm
  trade-off.

**Do not remove smoothing on the basis of this result.** The ablation used the
test set, so changing the pipeline because of it would be test-set selection.

## 2. Methods text

Add at the end of Section II-G, after the maximum-duration bullet.

> **Post-processing ablation.** To quantify the individual contribution of each
> stage, an ablation analysis was performed on the MultiScale TCN predictions for
> the independent test set. Each of the four stages was treated as an on/off
> factor: probability smoothing (window of 3 segments versus no smoothing),
> minimum-duration filtering (T_min = 25 s versus 0 s), refractory-period merging
> (T_ref = 30 s versus 0 s), and maximum-duration filtering (120 s versus no upper
> limit). All 2^4 = 16 configurations were evaluated at τ = 0.5 using the selected
> M→R order, with the maximum-duration filter applied last. Parameter values were
> those of the final pipeline and were not re-selected, so the analysis explains
> the selected pipeline rather than tuning it. Performance was assessed using
> event-level precision, recall, F1, FAR/hr, and mean detection latency.

## 3. Results text

Add as a new subsection of Section III-D, after "Final Event-Level Performance",
so that the full pipeline has already been introduced.

> **5) Post-Processing Ablation.** Table 5 summarises the contribution of each
> post-processing stage on the independent test set. Without post-processing, the
> model detected 660 of 684 seizures (recall 0.965) but generated 12,219 false
> alarms (FAR/hr 1.807; F1 0.097). The full pipeline removed 95.6% of these false
> alarms, reaching FAR/hr 0.079 and F1 0.675, at a cost of 39 missed seizures
> (recall 0.908).
>
> Applied alone, minimum-duration filtering had by far the largest effect, reducing
> false alarms by 94.1% to 720, followed by smoothing (57.5%) and refractory merging
> (48.0%), whereas maximum-duration filtering removed only 0.3%. Minimum-duration
> filtering also remained the most influential stage when removed from the full
> pipeline, with false alarms increasing from 535 to 2,840 and F1 decreasing from
> 0.675 to 0.313, although this removal recovered 32 seizures (recall 0.955).
>
> Removing refractory merging increased false alarms by 224 and decreased F1 from
> 0.675 to 0.602, and removing maximum-duration filtering increased false alarms by
> 86 and decreased F1 to 0.645; neither removal changed recall. Removing smoothing
> reduced false alarms by 46 (FAR/hr 0.072) and increased F1 to 0.686, but 9 fewer
> seizures were detected (recall 0.895) and mean detection latency increased from
> 2.66 s to 3.59 s. Smoothing therefore traded a small increase in false alarms for
> higher sensitivity and earlier detection once minimum-duration filtering was
> active.

## 4. Table 5

Report three decimals here, because several differences would disappear at the
two decimals used in Table 4. Renumber later tables: current Tables 5, 6 and 7
become 6, 7 and 8.

| Configuration | Stages active | TP | FP | FN | Precision | Recall | F1 | FAR/hr | Latency (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| No post-processing | none | 660 | 12,219 | 24 | 0.051 | 0.965 | 0.097 | 1.807 | 3.46 |
| Smoothing only | S | 657 | 5,196 | 27 | 0.112 | 0.961 | 0.201 | 0.768 | 3.42 |
| Min-duration only | M | 612 | 720 | 72 | 0.459 | 0.895 | 0.607 | 0.106 | 4.25 |
| Refractory only | R | 660 | 6,353 | 24 | 0.094 | 0.965 | 0.171 | 0.939 | 1.30 |
| Max-duration only | X | 660 | 12,179 | 24 | 0.051 | 0.965 | 0.098 | 1.801 | 3.46 |
| Full minus smoothing | M+R+X | 612 | 489 | 72 | 0.556 | 0.895 | 0.686 | 0.072 | 3.59 |
| Full minus min-duration | S+R+X | 653 | 2,840 | 31 | 0.187 | 0.955 | 0.313 | 0.420 | 2.20 |
| Full minus refractory | S+M+X | 621 | 759 | 63 | 0.450 | 0.908 | 0.602 | 0.112 | 3.30 |
| Full minus max-duration | S+M+R | 621 | 621 | 63 | 0.500 | 0.908 | 0.645 | 0.092 | 2.66 |
| **Full pipeline** | **S+M+R+X** | **621** | **535** | **63** | **0.537** | **0.908** | **0.675** | **0.079** | **2.66** |

Suggested caption:

> TABLE 5. Event-level post-processing ablation of the MultiScale TCN on the
> independent test set (684 seizures; τ = 0.5; M→R order). S: probability smoothing
> (window = 3 segments); M: minimum-duration filtering (25 s); R: refractory-period
> merging (30 s); X: maximum-duration filtering (120 s). Inactive stages used no
> smoothing, T_min = 0 s, T_ref = 0 s, or no upper duration limit.

LaTeX version:

```latex
\begin{table*}[!t]
\centering
\caption{Event-level post-processing ablation of the MultiScale TCN on the
independent test set (684 seizures; $\tau = 0.5$; M$\rightarrow$R order).
S: probability smoothing (window = 3 segments); M: minimum-duration filtering
(25~s); R: refractory-period merging (30~s); X: maximum-duration filtering
(120~s). Inactive stages used no smoothing, $T_{\min} = 0$~s,
$T_{\mathrm{ref}} = 0$~s, or no upper duration limit.}
\label{tab:postproc_ablation}
\begin{tabular}{llrrrcccrc}
\toprule
Configuration & Stages & TP & FP & FN & Precision & Recall & F1 & FAR/hr & Latency (s) \\
\midrule
No post-processing      & none    & 660 & 12,219 & 24 & 0.051 & 0.965 & 0.097 & 1.807 & 3.46 \\
\midrule
Smoothing only          & S       & 657 & 5,196  & 27 & 0.112 & 0.961 & 0.201 & 0.768 & 3.42 \\
Min-duration only       & M       & 612 & 720    & 72 & 0.459 & 0.895 & 0.607 & 0.106 & 4.25 \\
Refractory only         & R       & 660 & 6,353  & 24 & 0.094 & 0.965 & 0.171 & 0.939 & 1.30 \\
Max-duration only       & X       & 660 & 12,179 & 24 & 0.051 & 0.965 & 0.098 & 1.801 & 3.46 \\
\midrule
Full minus smoothing    & M+R+X   & 612 & 489    & 72 & 0.556 & 0.895 & 0.686 & 0.072 & 3.59 \\
Full minus min-duration & S+R+X   & 653 & 2,840  & 31 & 0.187 & 0.955 & 0.313 & 0.420 & 2.20 \\
Full minus refractory   & S+M+X   & 621 & 759    & 63 & 0.450 & 0.908 & 0.602 & 0.112 & 3.30 \\
Full minus max-duration & S+M+R   & 621 & 621    & 63 & 0.500 & 0.908 & 0.645 & 0.092 & 2.66 \\
\midrule
\textbf{Full pipeline}  & \textbf{S+M+R+X} & \textbf{621} & \textbf{535} & \textbf{63} & \textbf{0.537} & \textbf{0.908} & \textbf{0.675} & \textbf{0.079} & \textbf{2.66} \\
\bottomrule
\end{tabular}
\end{table*}
```

The table needs the `booktabs` package. If the template does not allow it, replace
the rules with `\hline`.

**Precision denominator.** TP + FP in this table counts as in Table 4 (621/1156 for
the full pipeline). The full pipeline produced 1,148 predicted events because one
predicted event can overlap more than one annotated seizure, and each seizure then
counts as a separate TP.

## 5. Discussion text

Add after the paragraph on the M→R order in Section IV.

> The ablation analysis showed that the stages of the pipeline are not equally
> important. Minimum-duration filtering provided most of the false-alarm reduction
> and also accounted for most of the sensitivity lost to post-processing, which is
> consistent with the short-seizure limitation noted below. Refractory merging and
> maximum-duration filtering both removed false alarms without reducing recall on
> the test set, making them low-cost refinements of the final event stream.
>
> The contribution of smoothing depended on the other stages. When minimum-duration
> filtering was inactive, smoothing substantially reduced false alarms, in agreement
> with its intended role of suppressing brief spurious activity. When
> minimum-duration filtering was active, short detections were already removed and
> smoothing instead increased both true detections and false alarms in every
> configuration evaluated. A likely explanation is that smoothing bridges brief
> probability dips, lengthening candidate events so that more of them pass the
> duration filter. The stage therefore acts mainly to preserve sensitivity in the
> final pipeline. Because seizure sensitivity was prioritised, and because the
> pipeline was fixed before this analysis, smoothing was retained.

Evidence for the interaction, from `ablation_all_configs.csv`:

| Stages other than S | FP without S | FP with S | TP without S | TP with S |
|---|---:|---:|---:|---:|
| none | 12,219 | 5,196 | 660 | 657 |
| R | 6,353 | 3,000 | 660 | 657 |
| X | 12,179 | 5,143 | 660 | 657 |
| R+X | 6,119 | 2,840 | 653 | 653 |
| M | 720 | 812 | 612 | 621 |
| M+X | 680 | 759 | 612 | 621 |
| M+R | 550 | 621 | 612 | 621 |
| M+R+X | 489 | 535 | 612 | 621 |

The explanation of bridging is a hypothesis. It fits the pattern above but was not
tested directly, so keep the wording "a likely explanation".

Treat latency as secondary evidence. Refractory merging can join a false alarm to a
later seizure detection, which moves the event start earlier. Part of the latency
improvement from merging may therefore come from this rather than from truly
earlier detection.

## 6. Wording for other sections

- **Abstract:** keep the current results. Optionally add: "Ablation showed that
  minimum-duration filtering was the dominant contributor, and that the full
  pipeline removed 95.6% of false alarms."
- **Contributions list:** add "stage-level ablation" to the event-level bullet.
- **Limitations:** the existing short-seizure limitation can now be quantified: 32 of
  the 39 seizures lost to post-processing are recovered when minimum-duration
  filtering is removed.
- **Conclusion:** the sentence on order and duration constraints can add that
  minimum-duration filtering contributed most to false-alarm reduction.

## 7. Corrections to make before resubmission

| Location | Issue | Correction |
|---|---|---|
| II-G | "fixed 3-step" pipeline, but four stages are listed | Say "four-stage" |
| II-G and III-D3 | 120 s is said to match the longest annotated seizure, but durations reach about 125 s | Check the true maximum, then correct the rationale, for example "approximately the longest" |
| Section IV | Says the "ordering and parameters" of smoothing, refractory merging and minimum duration were optimised on validation | Only order and T_min were swept; W and T_ref were fixed in advance |
| III-D2 | Test-set recall is cited when justifying T_min = 25 s, although selection is described as validation-based | Base the justification on validation results and report test as confirmation |
| Table 4 and Results | Test latency shown as 2.65 s; computed value is 2.6586 s | Use 2.66 s |
| Tables 4 and 7 | FAR/24 h was computed from rounded FAR/hr values | Use the exact values from the sweep CSVs: M1 validation ~0.85 (not ~0.72), M1 test ~1.90 (not ~1.92), M2 validation ~1.61 (not ~1.44), M2 test ~2.76 (not ~2.88) |
| Section IV | Discussion lists smoothing, refractory merging and minimum-duration filtering, omitting maximum duration | Add maximum-duration filtering |

## 8. Response to the reviewer

> We agree that the original analysis did not separate the contribution of each
> stage. We have therefore added a full-factorial ablation of probability smoothing,
> minimum-duration filtering, refractory-period merging, and maximum-duration
> filtering (16 configurations) on the independent test set, using the selected M→R
> order and fixed parameters (Section II-G; Section III-D5; Table 5). Minimum and
> maximum duration were ablated separately. Minimum-duration filtering was the
> dominant contributor; refractory merging and maximum-duration filtering reduced
> false alarms without affecting recall; and smoothing mainly preserved sensitivity
> and earlier detection, at the cost of a small increase in false alarms, once
> minimum-duration filtering was active.

## 9. Scope and limits

- The ablation covers the MultiScale TCN on the test set only. The attention
  variant and the validation set were not ablated, so do not generalise the results
  to them.
- Results are descriptive point estimates. No confidence intervals or significance
  tests are included, so avoid words such as "significant". Small differences, such
  as the F1 change of 0.011 from removing smoothing, should be read with care.
  Bootstrapping of overall post-processed performance is planned as a separate step.
- Only event-level metrics are reported. Segment-level metrics were excluded.
- All stages were switched off rather than varied, so the ablation does not evaluate
  other values of W, T_ref or the maximum duration.
