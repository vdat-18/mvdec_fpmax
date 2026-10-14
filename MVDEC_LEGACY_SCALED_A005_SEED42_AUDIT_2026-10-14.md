# MvDEC legacy-scaled alignment-0.05 seed-42 audit (2026-10-14)

## Verdict

`mvdec_2025_legacy_scaled_alignment_005_full_batch_v1` **fails the scientific gate**. It is a corrected diagnostic, not an exact-paper reproduction, not official MvDEC, and seeds 43-44 must not be run.

Restoring the old-v2 sum-over-dimensions L1/L2 scaling largely restores useful individual-view structure, especially View2, but it also restores the old failure mode: the two latent spaces become strongly anti-aligned and raw equal-average fusion destructively cancels them. Weak alignment weight 0.05 is numerically too small under summed 2,000-dimensional reconstruction loss. Fused ACC/NMI is worse than old v2 and far worse than the balanced alignment-0.05 diagnostic.

This ablation resolves the causal question: mean-per-dimension scaling caused much of the individual-view degradation, while sum-over-dimensions scaling makes reconstruction dominate and prevents weak L5 from establishing a shared coordinate system. A successful corrected protocol needs both view preservation and materially stronger cross-view alignment under the legacy scale.

## Run identity

- Kaggle kernel: `vandatzz/mvdec-2025-legacy-scaled-a005-seed-42`, Version 1, GPU T4.
- Protocol: `mvdec_2025_legacy_scaled_alignment_005_full_batch_v1`.
- Training commit: `4aaabef91f15f0879728877a4e21524a3a2c0f0d`.
- Notebook commit: `66dbe47e3d2f76e025f554acc26b90c86a0db6c8`.
- Notebook SHA-256: `c36648b17c1aec29d5c24556b063f4e1d88a0ae9c349b08f0638d83b4aebad8e`; the pushed notebook copy matches exactly.
- Seed: 42 only, in order REUTERS -> RCV1 -> canonical sklearn 20NEWS.
- Refinement: full batch; no mini-batch substitution.
- Kaggle runtime: NumPy 2.1.3, scikit-learn 1.6.1, TensorFlow 2.20.0.
- Single causal change from balanced alignment 0.05: `loss_scale_policy` changed from `mean_squared_dimensions` to `sum_squared_dimensions`.

## Provenance and integrity

| Dataset | Source ID | Source SHA-256 | Rows / features / classes | Config hash |
|---|---|---|---:|---|
| REUTERS | `dekm_google_drive_reuters_10k` | `7eb3f41aca38ffb93c104eef261adc3273d1da57c3750e04c92a67e335d423a1` | 10000 / 2000 / 4 | `ff76ee73ba1300f37d47339db4ce83aca43181cbafbda98aee30541ee387d754` |
| RCV1 | `dekm_google_drive_indices_sklearn_rcv1` | `305877a57be6f4984b4f890d1b4809bf631582116e7a1c23045ae93a540574d6` | 10000 / 2000 / 4 | `445d8ab7a5b790d9805f5928dc631d8bd19e16ebba9983e81b0be2c1be9570b1` |
| 20NEWS | `sklearn_fetch_20newsgroups_all_tfidf2000_18846` | `663e02a432bfd1e0a06b4f9863fb34b2c04f0888d1f15387bc5bbd051080ddbc` | 18846 / 2000 / 20 | `6cc6e21800c085a7a39704125d54019c229f35bf8d73768cba8990c8a3d50267` |

All three runs are `complete`; `failures.json` is empty. Independent validation passed for protocol, commit, seed, runtime versions, source identity/hash, dimensions, canonical sample indices, assignments, weights, artifacts, training logs, exact raw-average fusion, and every manifest-declared output hash using the repository's cross-platform normalized-text hash contract.

| Artifact | SHA-256 | Result |
|---|---|---|
| REUTERS archive | `35c2ccf11a0e08820b0035794e133cec447aa533985d560d1081836765242ab1` | PASS |
| RCV1 archive | `5602d58622f7cfcbd1fcc171e80290962ad7e60f7665843d30d23a89e0cc33b6` | PASS |
| 20NEWS archive | `c1bd3dbb6392aa1592a94597ec3552c08f5f097b948da8123f4437d1956a6819` | PASS |
| Combined ZIP | `70981810db7ed0992d1d925851552f6d33100d86a62be36b7f8058df59239789` | PASS |

Machine-readable audit: `artifacts/kaggle/mvdec-2025-legacy-scaled-a005-seed42/legacy_scaled_a005_seed42_independent_audit.json`.

## Independently verified metrics

ACC/NMI are percentages. Fused metrics independently recomputed from saved assignments match each manifest exactly.

| Dataset | View1 diagnostic | View2 | Fused | Paper fused | Fused delta vs paper |
|---|---:|---:|---:|---:|---:|
| REUTERS | 54.15 / 21.10 | 62.22 / 38.15 | 33.78 / 3.25 | 78.13 / 61.41 | -44.35 / -58.16 |
| RCV1 | 46.49 / 22.30 | 64.92 / 37.75 | 30.70 / 2.65 | 71.93 / 48.85 | -41.23 / -46.20 |
| 20NEWS | 38.26 / 33.44 | 35.84 / 31.44 | 16.22 / 9.35 | 41.63 / 45.43 | -25.41 / -36.08 |

MvDEC View1 is a jointly trained View1 diagnostic. It is not standalone `DEKM_F`; the frozen standalone DEKM_F control must remain separately reported.

## Comparison with old v2

The restored dimensional scaling substantially recovers individual views, but not every view fully reaches old v2 and fused performance regresses on all three datasets.

| Dataset | Representation | Old v2 | Legacy-scaled A005 | Delta |
|---|---|---:|---:|---:|
| REUTERS | View1 | 59.31 / 30.80 | 54.15 / 21.10 | -5.16 / -9.70 |
| REUTERS | View2 | 62.86 / 39.54 | 62.22 / 38.15 | -0.64 / -1.39 |
| REUTERS | Fused | 35.10 / 4.32 | 33.78 / 3.25 | -1.32 / -1.07 |
| RCV1 | View1 | 46.84 / 22.36 | 46.49 / 22.30 | -0.35 / -0.06 |
| RCV1 | View2 | 70.54 / 40.68 | 64.92 / 37.75 | -5.62 / -2.93 |
| RCV1 | Fused | 31.80 / 5.28 | 30.70 / 2.65 | -1.10 / -2.63 |
| 20NEWS | View1 | 39.19 / 33.88 | 38.26 / 33.44 | -0.93 / -0.44 |
| 20NEWS | View2 | 34.62 / 31.00 | 35.84 / 31.44 | +1.22 / +0.44 |
| 20NEWS | Fused | 17.00 / 9.66 | 16.22 / 9.35 | -0.78 / -0.31 |

The experiment therefore confirms that the balanced mean-per-dimension objective was responsible for much of the View2 collapse. It does not pass the preservation gate because REUTERS View1 and RCV1 View2 remain materially below old v2.

## Comparison with balanced alignment 0.05

| Dataset | Balanced A005 fused | Legacy-scaled A005 fused | Delta |
|---|---:|---:|---:|
| REUTERS | 45.55 / 15.47 | 33.78 / 3.25 | -11.77 / -12.22 |
| RCV1 | 45.33 / 18.42 | 30.70 / 2.65 | -14.63 / -15.77 |
| 20NEWS | 28.71 / 29.63 | 16.22 / 9.35 | -12.49 / -20.28 |

The legacy scale recovers views but loses the alignment/fusion benefit of the balanced scale. This is a clear tradeoff, not random seed noise: the direction is consistent on all three datasets.

## Alignment and fusion diagnosis

| Dataset | Paired cosine | Same-coordinate correlation | Retained norm | Old-v2 retained norm | Balanced A005 retained norm |
|---|---:|---:|---:|---:|---:|
| REUTERS | -0.7396 | -0.8878 | 0.3891 | 0.3773 | 0.9512 |
| RCV1 | -0.6359 | -0.8907 | 0.4550 | 0.4403 | 0.9659 |
| 20NEWS | -0.4634 | -0.7930 | 0.5589 | 0.5300 | 0.9825 |

Weak L5 slightly improves alignment over old v2, but the views remain strongly sign-reversed coordinate by coordinate. Raw averaging retains only 38.9%-55.9% of the component norm. Catastrophic cancellation has therefore returned.

Unsupervised diagnostic controls confirm that useful structure still exists before raw averaging:

| Dataset | Raw fused | Procrustes-aligned average | Concatenation control |
|---|---:|---:|---:|
| REUTERS | 33.78 / 3.25 | 59.47 / 38.99 | 58.98 / 38.78 |
| RCV1 | 30.70 / 2.65 | 60.56 / 32.03 | 60.24 / 31.49 |
| 20NEWS | 16.22 / 9.35 | 43.55 / 37.46 | 43.74 / 37.31 |

These controls are diagnostics only and must not be labeled official MvDEC results.

## Optimization evidence

| Dataset | Update windows | Final L1 / L2 / L4 / raw L5 | Weighted L5 | Reconstruction share | Stop |
|---|---:|---:|---:|---:|---:|---|
| REUTERS | 154 | 2509.9175 / 3.1880 / 0.1959 / 3.4770 | 0.1739 | 99.8585% | converged assignment |
| RCV1 | 201 | 2544.9690 / 2.7390 / 0.1721 / 3.2735 | 0.1637 | 99.8793% | converged assignment |
| 20NEWS | 150 | 2454.1562 / 3.5477 / 0.2110 / 2.9494 | 0.1475 | 99.8411% | converged assignment |

Under summed dimensional scaling, 2,000-dimensional reconstruction again contributes about 99.84%-99.88% of the total objective. The weighted alignment contribution is only 0.15-0.17, smaller than L2 and negligible relative to L1. Alignment weight 0.05 is therefore not scale-compatible with the legacy objective and cannot prevent latent sign reversal.

## Reproducibility note

- Saved-assignment fused ACC/NMI is authoritative and exactly matches every manifest.
- Local replay used scikit-learn 1.6.1 but NumPy 2.4.6 versus Kaggle NumPy 2.1.3.
- Replay differences are small and expected from KMeans initialization sensitivity; they do not affect assignment-level verification or the scientific conclusion.
- The largest result-level conclusions are tens of percentage points, far exceeding replay variation.

## Gate decision

| Gate | Result | Evidence |
|---|---|---|
| Complete, provenance-valid seed-42 runs | PASS | Three datasets complete; all normalized manifest hashes and package checksums pass |
| Substantially preserve individual views | FAIL | Most structure returns, but REUTERS View1 and RCV1 View2 remain materially below old v2 |
| Avoid destructive cancellation | FAIL | Paired cosine is -0.74 to -0.46; retained norm only 0.389-0.559 |
| Beat old-v2 fused | FAIL | ACC/NMI regress on all three datasets |
| Beat balanced alignment 0.05 fused | FAIL | Large regressions on all three datasets |
| Recover paper-level performance | FAIL | Fused remains 25.41-44.35 ACC and 36.08-58.16 NMI points below paper |

Final decision: **GATE FAIL**. Do not run seeds 43-44. Do not promote this protocol as official MvDEC.

## Next experiment

The next seed-42 ablation should isolate alignment strength under the now-restored legacy scale:

- Keep `sum_squared_dimensions`, disabled duplicate L3, L4, architecture, data, stopping rules, full-batch refinement, and raw equal-average fusion unchanged.
- Change only `alignment_weight` from 0.05 to 1.0.
- Require individual-view quality to remain close to this legacy-scaled run while paired cosine/retained norm and fused ACC/NMI improve materially.
- Continue to block seeds 43-44 until the seed-42 gate passes.

This tests whether unit-weight alignment, comparable to the other non-reconstruction terms, can preserve the recovered views while preventing the old-v2 anti-alignment failure. It remains a corrected diagnostic, not exact-paper reproduction.
