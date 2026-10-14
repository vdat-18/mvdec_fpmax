# MvDEC Seed-42 Fusion Failure Audit (2026-10-14)

## Scope and verdict

This audit covers the Kaggle Version 3 seed-42 run for REUTERS, RCV1, and the
canonical 18,846-row 20NEWS source under
`mvdec_2025_paper_informed_full_batch_v2`, training commit
`9b0c89afe522c009354bdd783b70fd182ec95d22`.

Training completed for all three datasets without OOM. All three manifests are
complete, every declared output hash validates, and all three archive SHA-256
files match. ACC/NMI recomputed independently from the saved fused assignments
also match the manifests exactly.

The reproduction nevertheless fails scientifically: the raw encoder average
specified by MvDEC Eq. 4 destructively cancels the two learned latent spaces.
The current seed-42 candidate must not be frozen as canonical MvDEC.

## Paper comparison

All values are percentages. MvDEC View1 in this run is jointly trained and is
not the standalone DEKM_F control printed as Proposed method View1 in Table 2.

| Dataset | Paper View1 ACC/NMI | Run View1 ACC/NMI | Paper View2 ACC/NMI | Run View2 ACC/NMI | Paper fused ACC/NMI | Run fused ACC/NMI | Fused delta ACC/NMI |
|---|---:|---:|---:|---:|---:|---:|---:|
| REUTERS | 76.42 / 59.99 | 59.27 / 30.76 | 65.01 / 51.29 | 63.02 / 39.62 | 78.13 / 61.41 | 35.10 / 4.33 | -43.03 / -57.08 |
| 20NEWS | 41.20 / 43.76 | 39.28 / 34.19 | 40.86 / 40.32 | 34.93 / 31.29 | 41.63 / 45.43 | 17.00 / 9.66 | -24.63 / -35.77 |
| RCV1 | 62.12 / 35.98 | 46.87 / 22.49 | 62.35 / 36.59 | 70.54 / 40.71 | 71.93 / 48.85 | 31.80 / 5.28 | -40.13 / -43.57 |

The decisive pattern is that useful single-view structure exists, especially in
RCV1 View2, but raw fusion destroys it.

## Root cause evidence

### 1. The two latent spaces are anti-aligned

The saved artifact satisfies the declared implementation exactly:
`h_fused == (h_view1 + h_view2) / 2` within float tolerance. The failure is not
an accidental concatenation, wrong slice, or stale tensor.

Label-free paired-latent diagnostics show that corresponding rows point in
opposite directions:

| Dataset | Mean paired cosine | Mean same-coordinate correlation | Mean norm retained after averaging |
|---|---:|---:|---:|
| REUTERS | -0.7600 | -0.8939 | 0.3773 |
| RCV1 | -0.6682 | -0.8838 | 0.4403 |
| 20NEWS | -0.5298 | -0.8089 | 0.5300 |

For REUTERS, raw averaging retains only 37.7% of the typical component norm.
This is destructive cancellation, not merely a modest difference in scale.

### 2. Label-free alignment controls recover much of the lost structure

These are diagnostics only and are not official MvDEC results. Procrustes uses
paired embeddings but no class labels.

| Dataset | Raw average ACC/NMI | Procrustes-aligned average ACC/NMI | Concatenation control ACC/NMI |
|---|---:|---:|---:|
| REUTERS | 35.10 / 4.33 | 59.68 / 39.91 | 59.47 / 39.95 |
| RCV1 | 31.80 / 5.28 | 66.13 / 40.28 | 65.78 / 39.53 |
| 20NEWS | 17.00 / 9.66 | 44.01 / 37.95 | 43.94 / 37.88 |

The large recovery after an unsupervised orthogonal alignment isolates latent
coordinate mismatch as the immediate cause of fusion collapse. Row-wise L2
normalization and per-coordinate z-scoring do not solve the problem, so this is
not a simple magnitude imbalance.

### 3. The architecture/objective does not identify a shared coordinate system

View1 and View2 are independent nonlinear autoencoders. Latent coordinates of
an autoencoder are free to rotate, reflect, permute, and rescale if its decoder
adapts. The paper averages coordinates directly but specifies no shared weights,
contrastive alignment, covariance alignment, or `||h1 - h2||^2` term.

The implementation's clustering terms consume only
`h = (h1 + h2) / 2`. Equal-and-opposite changes to the two views can therefore
leave the fused clustering input unchanged, while each decoder independently
supports reconstruction. The observed near coordinate-wise sign reversal is a
concrete manifestation of this non-identifiability.

### 4. Unweighted Eq. 11 is numerically dominated by reconstruction

At the final completed update window, the recorded losses are:

| Dataset | L1 reconstruction | L2 K-means | L3 trace | L4 greedy | L1 share |
|---|---:|---:|---:|---:|---:|
| REUTERS | 2513.1973 | 1.8105 | 1.8105 | 0.1255 | 99.8511% |
| RCV1 | 2563.6789 | 1.8213 | 1.8213 | 0.1278 | 99.8532% |
| 20NEWS | 2455.8121 | 2.0614 | 2.0614 | 0.1332 | 99.8270% |

The 2,000-dimensional summed reconstruction loss and 10-dimensional clustering
losses are added with unit weights. Consequently, the clustering portion is
only about 0.15-0.17% of the reported objective. The paper gives no coefficients
or loss normalization rule that uniquely resolves this scale mismatch.

L3 is also mathematically equal to L2 for orthonormal V, so literal Eq. 11
counts the same K-means residual twice without addressing cross-view alignment.

### 5. Assignment convergence is not semantic convergence

The runs stopped with fewer than 0.1% assignment changes, but converged to weak
fused partitions. This stopping rule only establishes stability of the current
pseudo-labels. It does not establish agreement with ground truth or recovery of
the paper's partition.

## Audit and reporting defects found

1. `public_mvdec_matrix_runs.csv` omitted View1, View2, and explicit fused metric
   columns even though the manifest contained them. The notebook then failed in
   its display cell after all training and packaging had completed.
2. Only fused assignments were persisted by training. View1 and View2 metrics
   were recorded without their row-level assignments, preventing exact
   assignment-level verification of those recorded values.
3. K-means was evaluated before canonical row-order restoration. Re-clustering
   the saved canonical-order latent arrays with the same seed can differ slightly
   because seeded K-means++ is row-order sensitive. The new audit therefore
   exports fresh canonical-order assignments and labels them as independent
   reclustering diagnostics rather than silently claiming byte-identical replay.

The matrix summary schema has been corrected to include all representation
metrics. A new `mvdec-audit-fusion` command now verifies the fusion contract,
exports independent assignments for every representation/control, computes
ACC/NMI, and records label-free alignment diagnostics.

## Pretrain-versus-final evaluation-only audit

Kaggle Version 7 completed an inference-only replay of the saved pretrain and
final weights for all three datasets. It used the same source identities, seed
42, architecture, and K-means policy as the training artifacts. No model was
retrained. All 30 exported assignment files were independently rescored, and
every ACC/NMI value matched the audit CSV to floating precision.

The final-weight replay reproduced the saved canonical latent arrays with the
following maximum absolute errors, all below the fail-closed `1e-5` threshold:

| Dataset | View1 max error | View2 max error | Fused max error |
|---|---:|---:|---:|
| 20NEWS | 5.72e-6 | 8.11e-6 | 4.29e-6 |
| RCV1 | 8.58e-6 | 7.96e-6 | 4.77e-6 |
| REUTERS | 4.77e-6 | 8.58e-6 | 4.53e-6 |

### Alignment changes during joint refinement

| Dataset | Stage | Paired cosine | Same-coordinate correlation | Retained norm |
|---|---|---:|---:|---:|
| REUTERS | Pretrain | -0.1755 | 0.0510 | 0.9472 |
| REUTERS | Final | -0.7600 | -0.8939 | 0.3773 |
| RCV1 | Pretrain | -0.1771 | 0.0509 | 0.9453 |
| RCV1 | Final | -0.6682 | -0.8838 | 0.4403 |
| 20NEWS | Pretrain | -0.2082 | -0.0466 | 0.9476 |
| 20NEWS | Final | -0.5298 | -0.8089 | 0.5300 |

The independently pretrained views are mildly negatively oriented, as expected
for non-identifiable latent coordinate systems, but their average still retains
about 95% of component norm. The severe coordinate-wise sign reversal and
destructive cancellation are created during joint refinement, consistently on
all three datasets.

### Representation quality before and after refinement

| Dataset | Stage | View1 ACC/NMI | View2 ACC/NMI | Raw fused ACC/NMI | Aligned control ACC/NMI |
|---|---|---:|---:|---:|---:|
| REUTERS | Pretrain | 49.98 / 17.36 | 48.02 / 16.70 | 49.12 / 16.32 | 50.06 / 17.62 |
| REUTERS | Final | 59.31 / 30.80 | 62.86 / 39.54 | 35.10 / 4.32 | 59.68 / 39.91 |
| RCV1 | Pretrain | 44.90 / 17.24 | 47.28 / 16.06 | 44.77 / 17.30 | 45.09 / 17.38 |
| RCV1 | Final | 46.84 / 22.36 | 70.54 / 40.68 | 31.80 / 5.28 | 66.13 / 40.28 |
| 20NEWS | Pretrain | 33.99 / 29.88 | 18.74 / 14.95 | 34.59 / 30.16 | 34.51 / 30.19 |
| 20NEWS | Final | 39.19 / 33.88 | 34.62 / 31.00 | 17.00 / 9.66 | 44.20 / 37.99 |

Joint refinement improves useful structure in one or both individual views, but
simultaneously drives their corresponding coordinates into opposite directions.
Consequently, the raw fused representation becomes substantially worse than the
pretrained fusion and both final individual views. This isolates the primary
failure to the joint objective's unconstrained two-view geometry, not to data
loading, weight serialization, pretraining alone, or K-means evaluation.

The audit output is retained at
`artifacts/kaggle/mvdec-2025-seed42-pretrain-final-alignment-audit`.

## Decision

- Keep DEKM frozen and unchanged.
- Preserve Version 3 as an immutable failed diagnostic.
- Do not run seeds 43-44 for this protocol.
- Do not rename Procrustes or concatenation controls as official MvDEC.
- The evaluation-only replay proves that severe anti-alignment is created by
  joint refinement, not by pretraining alone.
- The next training gate must use a new named protocol with an explicit
  cross-view alignment constraint and loss-scale correction, tested on seed 42
  only before any multi-seed run.
- Any corrected training protocol must be a new named protocol. The minimum
  scientifically justified candidates are an explicit latent-alignment loss or
  an explicitly aligned fusion step plus loss-scale normalization. Neither is
  uniquely specified by the paper, so it cannot be presented as exact official
  MvDEC without author code or clarification.
