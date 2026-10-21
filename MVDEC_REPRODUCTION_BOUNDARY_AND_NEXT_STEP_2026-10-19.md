# MvDEC reproduction boundary and next step (2026-10-19)

## Decision

The project objective remains reproduction of MvDEC 2025. Work must therefore stop at the published method boundary: no learned fusion weights, latent-scale calibration, Procrustes alignment, label-guided scalar sweeps, whitening, or other invented fusion rules may be promoted to the reproduction protocol.

The completed true-bottleneck run remains a `controlled_diagnostic_not_exact_paper_truth`. It established that the earlier direct split head gave the 10D View2 latent zero reconstruction gradient, and that making reconstruction depend on a 10D latent improves View2. It does not identify the authors' architecture.

The corresponding author has already been contacted and has not replied. Work therefore continues as a constrained forensic reproduction: infer missing implementation details from the paper, DEKM source, and falsifiable controlled experiments, without introducing objectives or fusion rules outside MvDEC. Numerical proximity to Table 2 is evidence for ranking hypotheses, not proof of authorship.

## Evidence rechecked

Authoritative paper: `docs/2025_Multi-view Deep Embedded Clustering- Exploring a new dimension of air pollution.pdf`.

Relevant passages were checked in the rendered pages and extracted text:

| Topic | Paper evidence | Classification | Consequence |
| --- | --- | --- | --- |
| Two views | Section 3.2 says two different autoencoders create distinct views and each is tailored to a specific set of attributes. | `paper_explicit` | A reproduction needs the exact public-dataset view construction or feature-index rule. |
| Inputs to encoders | Eq. (3) writes both embeddings as functions of the same `x_i`; Figure 2 shows the same 13D input feeding both branches. | `paper_explicit_conflicting_statement` | The published material does not uniquely specify whether public datasets use distinct feature subsets or the same matrix. |
| View1 | Section 3.2 says the first model is the same as DEKM; Section 4.2 says Proposed method_View1 is the same as DEKM_F. | `paper_explicit` | DEKM_F is the strongest available View1 anchor, but the paper does not disclose whether it is frozen or jointly changed for the fused run. |
| View2 backbone | Section 3.2 and Figure 2 specify a dense U-Net-inspired encoder/decoder with skip connections. | `paper_explicit` | The broad backbone is supported. |
| View2 latent boundary | Figure 2 ends View2 in one 23D linear output for a 13D example and does not label a separate 10D latent node. Eq. (3)-(5) nevertheless requires an encoder representation and decoder. | `missing_or_ambiguous` | Direct 10+13 split and explicit 10D bottleneck are both reconstructions, not author-verified architecture. |
| Fusion | Eq. (4) defines `(h1 + h2) / 2`. Figure 1 depicts AVG, while Figure 2 depicts concatenation to 46D. | `paper_explicit_conflicting_statement` | Eq. (4) raw average is the clearest mathematical rule; alternative calibration or weighted fusion is outside the paper. |
| Objective | Eq. (11) is `L1 + L2 + L3 + L4`; no coefficients are supplied. | `paper_explicit` plus `implementation_assumption` for unit coefficients | Unit weights are transparent but not author-confirmed hyperparameters. |
| Greedy direction | Section 3.3.4 says smallest eigenvalue, then Eq. (10) prose says the last dimension is modified; the DEKM discussion says eigenvectors are ascending and the last contributes least. | `paper_explicit_conflicting_statement` | No unique eigen-direction implementation can be recovered from the paper alone. |
| Refinement | Section 3.3.5 says joint alternating optimization until convergence. | `paper_explicit_but_incomplete` | Batch size, refresh interval, optimizer parameters, and exact stop implementation are not disclosed. |
| Evaluation selection | Section 4.1 says competing results are reproduced from their papers; Section 4.2 describes best performance metrics. | `paper_explicit_but_incomplete` | Seeds, run count, variance, checkpoint selection, and label-use policy for MvDEC are unknown. |
| REUTERS size | Table 1 prints 210 samples although the dataset is named REUTERS-10K and Table 2 reuses DEKM_F. | `paper_internal_inconsistency` | There is no disclosed 210-row sampling rule; using 10,000 rows remains the only source-backed choice. |

## Current implementation status

The current true-bottleneck controlled diagnostic holds these choices fixed:

- imported DEKM_F View1;
- joint Eq. (11) optimization;
- unit loss weights `1/1/1/1`;
- full-dataset refinement;
- smallest eigen direction;
- selected-dimension-only greedy target;
- raw Eq. (4) average;
- same full REUTERS feature matrix for both views;
- release row-L2 normalization;
- seed 42.

Evidence classification:

| Choice | Status |
| --- | --- |
| View1 is DEKM-derived | `paper_explicit` |
| Raw equal latent average | `paper_explicit`, but contradicted by Figure 2 |
| Joint optimization and reconstruction in Eq. (11) | `paper_explicit` |
| Dense U-Net-inspired View2 backbone | `paper_explicit` |
| Explicit post-skip 10D View2 bottleneck | `controlled_architecture_diagnostic` |
| Imported seed-42 DEKM_F checkpoint | `cross_paper_inference` |
| Same full feature matrix for both views | `paper_resolved_interpretation` forced by missing feature split |
| Unit loss weights | `implementation_assumption` |
| Full-batch refinement | `paper_table_inference` from DEKM_F |
| Smallest eigen direction | `paper_explicit_conflicting_statement` |
| Selected-dimension-only target | `paper_resolved_interpretation` |
| Release row-L2 normalization | `ablation`, not paper normalization |
| Seed 42 and one run | `diagnostic_execution_choice` |

## Result of the bounded architecture diagnostic

On REUTERS seed 42:

| Representation | Direct split head ACC / NMI | True bottleneck ACC / NMI | MvDEC Table 2 ACC / NMI |
| --- | ---: | ---: | ---: |
| View1 | 0.7555 / 0.592553 | 0.7561 / 0.594244 | 0.7642 / 0.5999 |
| View2 | 0.4759 / 0.213110 | 0.5139 / 0.326052 | 0.6501 / 0.5129 |
| Fused | 0.7555 / 0.592523 | 0.7563 / 0.594447 | 0.7813 / 0.6141 |

The true bottleneck fixes a concrete modeling defect and improves View2, but it leaves a substantial gap to Table 2 and barely changes fused performance. This is evidence that the direct split interpretation is defective; it is not evidence that the explicit bottleneck is the authors' exact implementation.

## Reproduction boundary

The following must not be used as the next reproduction step without new author evidence:

- centered-RMS or other View2 scale calibration;
- learned or swept fusion weights;
- Procrustes, whitening, z-scoring, or per-view normalization absent from the paper;
- label-guided checkpoint, scalar, or protocol selection;
- new feature splits invented from labels or metric proximity;
- seeds 43/44 before the protocol ambiguity is resolved;
- calling any controlled diagnostic an official or exact-paper implementation.

These operations may remain archived as forensic diagnostics, but they cannot define the MvDEC reproduction.

## Author-contact status and unresolved evidence

A focused request has already been sent to corresponding author Hassan Kassem (`hassan.kassem@liser.lu`) and no response has been received. The unanswered items remain the evidence gaps that the forensic reconstruction must resolve or bound:

1. Source code or an immutable commit/notebook for MvDEC Table 2.
2. Exact View1/View2 feature construction for REUTERS-10K, 20NEWS, and RCV1-10K, including feature indices if the views use different attributes.
3. Complete View2 layer graph, especially the 10D embedding and reconstruction boundary corresponding to Figure 2's 23D output.
4. The actual fusion used for Table 2: Eq. (4) averaging versus Figure 2 concatenation, and any normalization or scaling before fusion.
5. Whether View1 is imported/frozen DEKM_F or jointly updated during the fused run.
6. Exact definitions and coefficients/reductions for `L1`-`L4`, plus the eigenvector index/order and `Y'` update rule.
7. Optimizer, learning rate, initialization, pretraining/refinement epochs or steps, batch size, K-Means refresh interval, stopping rule, and checkpoint-selection rule.
8. Seeds, number of trials, and how the reported "best performance" was selected.
9. Clarification of the REUTERS sample count `210` in Table 1 versus REUTERS-10K.
10. If available, final checkpoints, cluster assignments, logs, and the processed public-dataset arrays used for Table 2.

## Forensic reconstruction plan

Use a staged, falsifiable search over implementation hypotheses that remain inside the published method family:

1. Reproduce the exact three-run DEKM_F View1 selection envelope and retain its checkpoints as the View1 anchor.
2. Screen View2 architectures implied by Figure 2 and Eqs. (3)-(5): the literal terminal `10 + d` head, a true post-skip 10D bottleneck, and an encoder-side 10D bottleneck. Reject architectures whose reconstruction path cannot train the reported embedding.
3. For surviving View2 graphs, inherit only disclosed DEKM optimizer, initialization, pretraining, full-batch, K-Means, eigen, and stopping choices. Test the paper's conflicting smallest-versus-last eigen interpretations as named hypotheses.
4. Evaluate View1, View2, and Eq. (4) raw-average fusion for seeds 42-44. Rank hypotheses by cross-seed agreement across all three Table 2 rows, not by one fused seed.
5. Use Figure 1 full-output averaging and Figure 2 concatenation only as literal paper alternatives. Do not introduce calibrated, weighted, whitened, Procrustes, or label-tuned fusion into the candidate reproduction.
6. Lock the best paper-constrained hypothesis before extending it to 20NEWS and RCV1. A hypothesis that only matches REUTERS is not sufficient evidence for the reconstructed source.

Every candidate remains `controlled_diagnostic_not_exact_paper_truth` until it reproduces the public-dataset pattern robustly and its inferred choices are fully recorded.

## Execution decision

- Preserve all audited seed-42 artifacts unchanged.
- The generic post-run manifest hash bug is fixed and covered by regression tests; it must not alter scientific protocols.
- Continue with the forensic reconstruction plan above. Start with local structural/tests and reuse existing checkpoints; launch only preregistered GPU candidates that distinguish paper-constrained hypotheses.
- Do not tune against labels, add non-paper fusion/calibration, or call numerical agreement proof of official source equivalence.
- The target deliverable is the most probable paper-faithful reconstructed implementation, accompanied by an explicit inference ledger and cross-dataset/cross-seed validation.
