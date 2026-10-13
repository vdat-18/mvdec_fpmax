# DEKM 2021 / MvDEC 2025 Audit (2026-10-13)

## Scope

The next cloud runs will exclude 20NEWS. The audit scope is REUTERS-10K and
RCV1-10K only, with the DEKM 2021 released protocol used as the control and the
MvDEC 2025 protocol evaluated only after the control is verified.

This report is a static code/paper audit. It does not claim that the current
Kaggle numbers are a faithful MvDEC reproduction.

## Executive verdict

The low Kaggle result is not evidence that the datasets are intrinsically hard
or that more seeds will fix the problem. The run used the repository's
`mvdec_2025_paper_resolved_full_batch_v1` interpretation. That protocol changes
three unresolved choices at once: smallest eigen-direction, selected-dimension
only target, and full-dataset refinement. The MvDEC paper does not provide
pseudocode or source code that fixes those choices, and the DEKM 2021 release
uses a different greedy target and mini-batch schedule. Therefore the run was
not an apples-to-apples test of either paper.

## Findings

### F0 - The historical REUTERS control was mislabeled at the batch-policy level

The official DEKM repository currently sets `batch_size = 256`; the nearby
`# batch_size=10000` line is commented out. The historical local
`release-2021` wrapper instead defaulted REUTERS to a 10,000-row refinement
batch. Its metrics are valid and close to the DEKM row, but its schedule is
full-batch-like and therefore cannot simultaneously serve as a clean official
mini-batch DEKM control.

The audit now preserves that legacy protocol identity and introduces two
separate contracts: `release-minibatch-2021` for the official 256-row source
behavior and `full-batch-2021` for DEKM_F. The official repository revision
checked on 2026-10-13 was `e58f506922137abe2905aa0e4eb04a6bf952e80d`; the
SHA-256 of its `DEKM_dense.py` was
`8536630e35b05e5ad8cd0f9235c2012537f31e5b3b2a5b3313ee1c246cb3dec9`.

### F1 - The current MvDEC run is a compound interpretation, not the paper

Evidence:

- `src/representation_learning/MVDEC_dense.py` defines
  `PAPER_RESOLVED_FULL_BATCH_PROTOCOL` with `smallest`,
  `selected_dimension_only`, and `full_dataset`.
- MvDEC 2025 says the last dimension is changed, while another paragraph says
  the smallest-eigenvalue direction is least informative. Those statements
  conflict.
- DEKM 2021 Algorithm 1 and the release replace only the last transformed
  coordinate and use a frozen target.
- MvDEC Table 2 says View 1 equals `DEKM_F`, but that does not prove the
  complete multi-view refinement schedule or loss implementation.

Consequence: the 71% REUTERS result cannot be used to conclude that the
architecture itself fails. It mainly tells us that this particular compound
interpretation is not close to Table 2.

### F2 - DEKM and MvDEC are not currently separated by a clean control ladder

The repository contains both implementations, but the correct order must be:

1. Reproduce DEKM 2021 REUTERS and RCV1 with the release contract.
2. Reproduce DEKM_F (full-batch) with only the batch policy changed.
3. Add the MvDEC View 2 and latent average, keeping the DEKM control schedule.
4. Only then test Eq. 11 loss terms and the MvDEC-specific ambiguities one at a
   time.

Without this ladder, a failed MvDEC run mixes data, architecture, objective,
refinement schedule, and model-selection differences.

### F3 - MvDEC Eq. 11 scaling is a high-risk implementation mismatch

The paper writes all four terms as sums over samples. The current joint-loss
path in `MVDEC_dense.py` computes per-batch means while summing dimensions for
some terms. In particular, reconstruction, K-Means, and orthonormal terms use
`squared_euclidean_per_sample` followed by `reduce_mean`, while the selected
Greedy term uses a mean over one coordinate.

This makes the relative weights of L1/L2/L3/L4 depend on input width and batch
size. For REUTERS/RCV1, the 2,000-dimensional reconstruction term can dominate
the one-dimensional greedy term even though Eq. 11 gives each term unit weight.
This must be resolved with an explicit reduction contract before treating Eq. 11
as paper-faithful. The release-faithful DEKM MSE path is a separate valid
control, not a substitute for Eq. 11.

### F4 - The View 2 definition is under-specified and is the largest MvDEC risk

The paper says the two views use different attribute sets, but its equations
feed the same `x_i` to both encoders and the figure only gives a 13-feature air
pollution example. For REUTERS/RCV1 there is no published feature partition,
View 2 latent head, or source code.

The current implementation uses the same 2,000-dimensional X for both views and
chooses a latent/reconstruction split for the U-Net branch. That is a documented
assumption, not an established paper fact. Existing diagnostics show View 1 is
near the paper while View 2 is far below it, so this ambiguity is a stronger
candidate than random seed variation.

### F5 - The primary MvDEC metric is not the paper's undocumented ?best? rule

The paper reports ?best performance metrics? but gives no seed list,
checkpoint-selection rule, or run count. The repository reports all explicit
seeds and uses final fused metrics as the fair primary result; best-of-run values
are diagnostics. This is methodologically correct for a reproducible study, but
it means exact Table 2 matching cannot be claimed without the authors' selection
rule.

### F6 - Data preprocessing is consistent with the released DEKM text loader

`src/pipeline/public_data.py` and `external_repos/DEKM/utils.py` both L2-normalize
rows and use the released REUTERS arrays / RCV1 split and top-2,000 feature
contract. This is not currently the leading suspected bug. The MvDEC paper's
REUTERS sample-count text is inconsistent with the DEKM benchmark; the canonical
10,000-row DEKM release is the defensible choice.

### F7 - DEKM protocol variants must not be mixed

`src/representation_learning/DEKM_dense.py` now exposes separate modern,
official-source mini-batch, DEKM_F full-batch, and legacy contracts. For paper
comparison, use `--protocol release-minibatch-2021` for DEKM and
`--protocol full-batch-2021` for DEKM_F. The legacy `release-2021` identity is
retained only so existing artifacts are not silently reinterpreted; the modern
contract remains a deterministic engineering regression test.

## Required implementation order

1. Freeze the dataset scope to REUTERS and RCV1 in the cloud notebook only;
   leave the loader capable of handling 20NEWS but do not launch it.
2. Run no MvDEC training yet.
3. Verify the DEKM 2021 release control on both datasets, including exact source
   hashes, row counts, class counts, normalization, and three-run mean/std.
4. Verify DEKM_F separately by changing only the refinement batch policy.
5. Add a two-view model with the DEKM release objective and schedule. Compare
   View 1, View 2, and fused results separately.
6. Add MvDEC Eq. 11 terms with an explicit reduction/weight contract and test
   each term in isolation. Do not silently reuse the release MSE scaling.
7. Evaluate the eigen-direction and target choices as a 2x2 ablation, not as a
   single ?paper-resolved? run.
8. Only after these controls pass, launch the two-dataset multi-seed Kaggle run.

## Acceptance criteria for ?match?

A result should be called a match only if:

- DEKM REUTERS/RCV1 controls reproduce the 2021 reference range under the
  release contract.
- The exact dataset fingerprints and preprocessing are recorded.
- View 1 reproduces the MvDEC paper's DEKM_F reference within a declared
  tolerance, before judging View 2 or fusion.
- Every MvDEC ambiguity is isolated in a named protocol.
- Metrics are reported per seed and as mean plus sample standard deviation.
- ?Best? values are labeled sensitivity diagnostics, not silently substituted
  for the primary aggregate.

## Files reviewed

- `docs/2021_Deep Embedded K-Means Clustering.pdf`
- `docs/2025_Multi-view Deep Embedded Clustering- Exploring a new dimension of air pollution.pdf`
- `external_repos/DEKM/DEKM_dense.py`
- `src/representation_learning/DEKM_dense.py`
- `src/representation_learning/MVDEC_dense.py`
- `src/pipeline/public_data.py`
- `src/representation_learning/utils.py`
- `MVDEC_PAPER_CODE_AUDIT.md`
- `MVDEC_REUTERS_ARCHITECTURE_DIAGNOSIS.md`


## Post-run RCV1 provenance audit

The completed Kaggle DEKM_F run was independently checked against a freshly
built local scikit-learn RCV1 cache. The split is structurally valid: 9,000 test
rows plus 1,000 validation rows, 10,000 unique rows, no overlap, and exactly
2,500 samples in each of CCAT/ECAT/GCAT/MCAT. All three saved assignment files
contain the canonical true labels, and their ACC/NMI values recompute exactly.
There is no evidence of label leakage, incorrect metric calculation, or a wrong
RCV1 subset.

The byte-level source hash is runtime-sensitive. Under NumPy 2.1.3 and
scikit-learn 1.6.1, the local loader reproduces the Kaggle hash exactly:
`083201cff76ee4952531472fcd630bd6366ca9b2a34ec08a3092c6f4f1aef90c`.
Under NumPy 2.4.6 and scikit-learn 1.9.0, it produces
`e03069e5dd5841ede5915267228c7d8222b63620e8946d7b3eba2fd667692743`.
The selected top-2,000 original feature IDs are identical in both environments,
but `numpy.argpartition` returns them in a different column order; only 74 of
2,000 positions coincide. Row normalization also differs at floating-point
roundoff scale (maximum absolute difference `1.1920929e-07`). Thus the hash
mismatch does not mean different documents, labels, or selected vocabulary, but
it proves that the released preprocessing is not byte-reproducible across
NumPy versions. The neural network can still be seed-sensitive to this column
permutation because fixed initialized weights are attached to different input
coordinates.

A raw-data K-Means control with `n_init=10` gives the same result under both
runtime families for seed 42: ACC `0.6100`, NMI `0.409672`. Seeds 43 and 44 give
ACC `0.6103`/`0.6106` and NMI `0.410413`/`0.416054`. These are already well above
DEKM Table III's K-Means reference of ACC `0.5196`, NMI `0.3135`. Therefore the
paper table cannot be treated as an exact executable oracle under the released
loader without the authors' missing run-selection/randomness details and the
historical dependency stack.

The next experiment is intentionally bounded to RCV1 seed 42. It uses the
paper-text settings (`glorot_uniform`, symmetric `eigh`, full-dataset
refinement, assignment tolerance `0.001`) while retaining deterministic seeding
for auditability. Seeds 43-44 will not be launched unless this one diagnostic is
materially closer to the DEKM_F paper row than the released-source seed-42 run.
