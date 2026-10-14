# MvDEC legacy-scaled alignment-1.0 seed-42 audit (2026-10-15)

## Verdict

`mvdec_2025_legacy_scaled_alignment_100_full_batch_v1` **fails the scientific gate**. It is a corrected diagnostic, not an exact-paper reproduction, not official MvDEC, and seeds 43-44 must not be run.

Increasing raw row-wise alignment weight from 0.05 to 1.0 improves the uncentered paired cosine and raises fused ACC/NMI over legacy-scaled A0.05 and old v2. It does not preserve the individual views, does not beat the balanced A0.05 fused result, and remains far below the paper.

The deeper audit identifies why the apparent alignment gain is insufficient: L5 mainly aligns the two latent **mean vectors**, whose cosine is 0.925-0.960, while the centered sample-dependent structure remains strongly anti-aligned. Centered paired cosine is -0.824 to -0.765, same-coordinate correlation is -0.833 to -0.752, and centered raw averaging retains only 0.331-0.387 of the component norm. The uncentered L5 can therefore improve its own objective by learning shared offsets without establishing a common coordinate system for cluster-discriminative variation.

This run also stops solely when fused KMeans assignments stabilize, even though L5 remains 1.50-1.92 and centered anti-alignment remains severe. Assignment-only stopping is not sufficient for a protocol that adds an alignment objective.

## Run identity

- Kaggle kernel: `vandatzz/mvdec-2025-legacy-scaled-a100-seed-42`, Version 1, GPU T4, completed successfully.
- Protocol: `mvdec_2025_legacy_scaled_alignment_100_full_batch_v1`.
- Training commit: `7b30c8405acd6615cfec86cac931e9cf2bc9aa9a`.
- Notebook commit: `0f323c7d5063a27571b3d1b8f072f9c50d26493c`.
- Notebook SHA-256: `0fdda611ed13ab0dd01f0ee62c18c1722ddddc18533f6055894b18e034124124`; the local notebook matches the committed blob exactly.
- Seed: 42 only, executed in order REUTERS -> RCV1 -> canonical 20NEWS.
- Refinement: full dataset per optimizer update; no mini-batch substitution.
- Kaggle runtime: NumPy 2.1.3, scikit-learn 1.6.1, TensorFlow 2.20.0.
- Single causal change from legacy-scaled A0.05: `alignment_weight` increased from 0.05 to 1.0.

## Provenance and integrity

| Dataset | Source ID | Source SHA-256 | Rows / features / classes | Config hash |
|---|---|---|---:|---|
| REUTERS | `dekm_google_drive_reuters_10k` | `7eb3f41aca38ffb93c104eef261adc3273d1da57c3750e04c92a67e335d423a1` | 10000 / 2000 / 4 | `69d28d8893ae48d16a3b61892b570fcd7651fbe11458e175915372cd4cd26fac` |
| RCV1 | `dekm_google_drive_indices_sklearn_rcv1` | `305877a57be6f4984b4f890d1b4809bf631582116e7a1c23045ae93a540574d6` | 10000 / 2000 / 4 | `31299930bb1cc98cb0d5f7bbefa41de9d8a4290896bd30fd5508055427e043ce` |
| 20NEWS | `sklearn_fetch_20newsgroups_all_tfidf2000_18846` | `663e02a432bfd1e0a06b4f9863fb34b2c04f0888d1f15387bc5bbd051080ddbc` | 18846 / 2000 / 20 | `505e898ff4971e27b8dfe868ac4512c5c8670aa49651f6bce90ce7abb68525b3` |

All three manifests are complete and `failures.json` is empty. Independent validation passed for protocol, commits, seed, dataset order, runtime versions, source identity/hash, dimensions, assignments, weights, exact raw-average fusion, and every manifest-declared output hash under the repository's normalized-text hash contract.

The corrected progress metadata also passes exactly:

| Dataset | Optimizer updates | Recorded progress | Unit | Expected `updates // 10` |
|---|---:|---:|---|---:|
| REUTERS | 800 | 80 | `kmeans_refresh_windows` | 80 |
| RCV1 | 670 | 67 | `kmeans_refresh_windows` | 67 |
| 20NEWS | 1510 | 151 | `kmeans_refresh_windows` | 151 |

| Package | SHA-256 | Result |
|---|---|---|
| REUTERS archive | `3d9ef2d01e865ca9df717a66cfbd789c06073b8569dd02c45f7947dd53e6be10` | PASS |
| RCV1 archive | `a97b86a3e7e1fb27efd4997f35d06bbd2fa27494c20a62b159982edb8b068fc7` | PASS |
| 20NEWS archive | `7bc6a8d30c07ac42479cd294306acf76a502bd2ee3ca73902152692800936652` | PASS |
| Combined ZIP | `1f83c3323bddd6c9674599c6cc6d5e55064c7d0c38119fd3980bfa697c81c09f` | PASS |

Machine-readable audit: `artifacts/kaggle/mvdec-2025-legacy-scaled-a100-seed42/legacy_scaled_a100_seed42_independent_audit.json`.

## Independently verified metrics

ACC/NMI are percentages. Fused ACC/NMI recomputed from the saved assignments match each manifest exactly.

| Dataset | View1 diagnostic | View2 | Fused | Paper fused | Fused delta vs paper |
|---|---:|---:|---:|---:|---:|
| REUTERS | 51.34 / 18.35 | 59.80 / 28.64 | 42.58 / 8.10 | 78.13 / 61.41 | -35.55 / -53.31 |
| RCV1 | 47.49 / 21.39 | 57.44 / 27.33 | 35.42 / 7.98 | 71.93 / 48.85 | -36.51 / -40.87 |
| 20NEWS | 37.80 / 32.68 | 31.41 / 26.94 | 19.38 / 12.31 | 41.63 / 45.43 | -22.25 / -33.12 |

MvDEC View1 above is a jointly trained diagnostic. It is **not** standalone `DEKM_F`.

### Standalone DEKM_F paper control

The paper's Proposed View1 values are the separately reported standalone DEKM_F control, not the jointly trained MvDEC View1 values above:

| Dataset | Standalone DEKM_F control in paper |
|---|---:|
| REUTERS | 76.42 / 59.99 |
| RCV1 | 62.12 / 35.98 |
| 20NEWS | 41.20 / 43.76 |

## Comparison with legacy-scaled A0.05

| Dataset | Representation | Legacy A0.05 | Legacy A1.0 | Delta |
|---|---|---:|---:|---:|
| REUTERS | View1 | 54.15 / 21.10 | 51.34 / 18.35 | -2.81 / -2.75 |
| REUTERS | View2 | 62.22 / 38.15 | 59.80 / 28.64 | -2.42 / -9.51 |
| REUTERS | Fused | 33.78 / 3.25 | 42.58 / 8.10 | +8.80 / +4.85 |
| RCV1 | View1 | 46.49 / 22.30 | 47.49 / 21.39 | +1.00 / -0.91 |
| RCV1 | View2 | 64.92 / 37.75 | 57.44 / 27.33 | -7.48 / -10.42 |
| RCV1 | Fused | 30.70 / 2.65 | 35.42 / 7.98 | +4.72 / +5.33 |
| 20NEWS | View1 | 38.26 / 33.44 | 37.80 / 32.68 | -0.46 / -0.76 |
| 20NEWS | View2 | 35.84 / 31.44 | 31.41 / 26.94 | -4.43 / -4.50 |
| 20NEWS | Fused | 16.22 / 9.35 | 19.38 / 12.31 | +3.16 / +2.96 |

A1.0 improves raw fusion consistently, but the gain is purchased partly by degrading View2. The individual-view preservation gate therefore fails.

## Comparison with old v2 and balanced A0.05

| Dataset | Old-v2 fused | Balanced A0.05 fused | Legacy A1.0 fused | Delta vs old v2 | Delta vs balanced A0.05 |
|---|---:|---:|---:|---:|---:|
| REUTERS | 35.10 / 4.33 | 45.55 / 15.47 | 42.58 / 8.10 | +7.48 / +3.77 | -2.97 / -7.37 |
| RCV1 | 31.80 / 5.28 | 45.33 / 18.42 | 35.42 / 7.98 | +3.62 / +2.70 | -9.91 / -10.44 |
| 20NEWS | 17.00 / 9.66 | 28.71 / 29.63 | 19.38 / 12.31 | +2.38 / +2.65 | -9.33 / -17.32 |

Legacy A1.0 beats old-v2 fused on all three datasets, but does not beat the strongest corrected diagnostic already available, balanced A0.05.

## Alignment audit: raw alignment is misleading

| Dataset | Raw paired cosine | Mean-vector cosine | Centered paired cosine | Same-coordinate correlation | Raw retained norm | Centered retained norm |
|---|---:|---:|---:|---:|---:|---:|
| REUTERS | 0.0416 | 0.9249 | -0.8236 | -0.8328 | 0.7423 | 0.3305 |
| RCV1 | 0.0402 | 0.9321 | -0.7824 | -0.7992 | 0.7373 | 0.3583 |
| 20NEWS | 0.2485 | 0.9604 | -0.7646 | -0.7523 | 0.8080 | 0.3868 |

Compared with legacy A0.05, the uncentered paired cosine improves from strongly negative values to approximately zero or positive, and raw retained norm rises to 0.737-0.808. Those headline diagnostics alone would overstate success.

After subtracting each view's latent mean, the cluster-bearing deviations remain strongly anti-aligned and retain only 33.1%-38.7% of their component norm when averaged. L5's row normalization is applied before any centering, so a shared offset can reduce L5 even while centered geometry remains sign-reversed. This is confirmed by the near-parallel latent mean vectors.

The fusion diagnostic has been extended to report mean-vector cosine, centered paired cosine, and centered retained norm so this failure mode cannot be hidden by offsets in future audits.

## Label-free controls

These are diagnostics only and are not official MvDEC results.

| Dataset | Raw fused | Procrustes-aligned average | Concatenation control |
|---|---:|---:|---:|
| REUTERS | 42.58 / 8.10 | 59.91 / 35.48 | 59.20 / 34.97 |
| RCV1 | 35.42 / 7.98 | 65.49 / 37.56 | 61.63 / 32.60 |
| 20NEWS | 19.38 / 12.31 | 42.47 / 35.78 | 43.65 / 36.77 |

Useful structure remains in the individual views. The large control-vs-raw gap confirms that incompatible centered coordinate systems, not absence of all cluster information, remain the primary fusion failure.

## Optimization and stopping evidence

| Dataset | Update windows | Final L1 / L2 / L4 / L5 | L5 share | Reconstruction share | Stop |
|---|---:|---:|---:|---:|---|---|
| REUTERS | 80 | 2527.1330 / 4.2421 / 0.2343 / 1.9200 | 0.0758% | 99.7475% | converged assignment |
| RCV1 | 67 | 2570.7950 / 4.0019 / 0.2115 / 1.9107 | 0.0741% | 99.7624% | converged assignment |
| 20NEWS | 151 | 2454.3754 / 3.6287 / 0.2065 / 1.5043 | 0.0612% | 99.7829% | converged assignment |

Raising L5 weight by 20x makes alignment materially affect training, but reconstruction still contributes 99.75%-99.78% of the reported objective. More importantly, training stops on fused assignment stability alone. The stopping checkpoint accepts 5 changed assignments on REUTERS, 10 on RCV1, and 18 on 20NEWS, while the alignment objective is still far from zero and centered alignment is strongly negative.

This establishes two distinct defects in the corrected diagnostic:

1. The uncentered row-wise L5 can align mean offsets rather than sample-dependent geometry.
2. The stopping rule does not require the added alignment objective to converge.

## Gate decision

| Gate | Result | Evidence |
|---|---|---|
| Complete, provenance-valid seed-42 runs | PASS | Three datasets complete; manifests, normalized hashes, archives and ZIP validate |
| Correct progress metadata | PASS | Unit is `kmeans_refresh_windows`; values equal optimizer updates divided by 10 |
| Preserve individual views near legacy A0.05 | FAIL | View2 loses 2.42-7.48 ACC and 4.50-10.42 NMI points |
| Avoid destructive cancellation | FAIL | Centered cosine remains -0.824 to -0.765; centered retained norm only 0.331-0.387 |
| Beat old-v2 fused | PASS | Fused ACC/NMI improves on all three datasets |
| Beat balanced A0.05 fused | FAIL | Fused remains 2.97-9.91 ACC and 7.37-17.32 NMI points lower |
| Recover paper-level performance | FAIL | Fused remains 22.25-36.51 ACC and 33.12-53.31 NMI points below paper |

Final decision: **GATE FAIL**. Do not run seeds 43-44 and do not promote this protocol as official MvDEC.

## Required next correction

Do not continue increasing the same uncentered L5 weight. The next seed-42 diagnostic should isolate the newly demonstrated defect:

- Keep legacy `sum_squared_dimensions`, disabled duplicate L3, L4, architecture, data, raw equal-average fusion, full-batch refinement and seed 42 unchanged.
- Replace uncentered row normalization in L5 with full-batch mean centering followed by row L2 normalization, so L5 acts on sample-dependent variation rather than shared offsets.
- Give the new alignment definition a new immutable protocol ID; do not silently change the A1.0 protocol.
- Add an alignment-aware stopping condition or a minimum refinement window only in a separate ablation, unless the centered-only run proves assignment stopping terminates before centered alignment converges.
- Continue to block seeds 43-44 until a seed-42 protocol preserves individual views and removes centered cancellation.

This is still a corrected diagnostic. The paper does not specify centered alignment, alignment weighting, or an alignment-aware stopping rule, so none of these corrections may be labeled exact-paper reproduction.
