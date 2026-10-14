# MvDEC aligned-balanced alignment-0.05 seed-42 audit (2026-10-14)

## Verdict

`mvdec_2025_aligned_balanced_alignment_005_full_batch_v1` **fails the scientific gate**. It is a corrected diagnostic, not an exact-paper reproduction, not the official MvDEC result, and seeds 43-44 must not be run.

Reducing `alignment_weight` from 1.0 to 0.05 removes destructive cancellation and substantially improves fused metrics over old v2. However, it still damages the individual views, fails to beat alignment-1.0 on RCV1, and remains far below the paper on all three datasets. The remaining failure is therefore not explained by an excessively large L5 weight alone; the balanced L1/L2 formulation and/or joint refinement is still degrading cluster-discriminative structure.

## Run identity

- Kaggle kernel: `vandatzz/mvdec-2025-aligned-balanced-a005-seed-42`, Version 2, GPU T4.
- Protocol: `mvdec_2025_aligned_balanced_alignment_005_full_batch_v1`.
- Training commit: `047f04ee3bce0bd255f6e46731f36201e96e15e9`.
- Notebook commit: `5743739`.
- Seed: 42 only, in order REUTERS -> RCV1 -> canonical sklearn 20NEWS.
- Refinement: full batch; no mini-batch substitution.
- Runtime contract: scikit-learn 1.6.1; Kaggle NumPy 2.1.3.
- Version 1 failed before training because it expected scikit-learn 1.9.0 after Kaggle had loaded 1.6.1. Version 2 corrected the contract to 1.6.1 and completed all datasets.

## Provenance and integrity

| Dataset | Source ID | Source SHA-256 | Rows / features / classes | Config hash |
|---|---|---|---:|---|
| REUTERS | `dekm_google_drive_reuters_10k` | `7eb3f41aca38ffb93c104eef261adc3273d1da57c3750e04c92a67e335d423a1` | 10000 / 2000 / 4 | `578134fdbb995aaff9f25edf986dac8871ba9e7fd978e3fb736ddf63867685bf` |
| RCV1 | `dekm_google_drive_indices_sklearn_rcv1` | `305877a57be6f4984b4f890d1b4809bf631582116e7a1c23045ae93a540574d6` | 10000 / 2000 / 4 | `f5f59ea59c25d6fc3d3f655af49b5ed175344e4049b12c0594d2e010f06c57c9` |
| 20NEWS | `sklearn_fetch_20newsgroups_all_tfidf2000_18846` | `663e02a432bfd1e0a06b4f9863fb34b2c04f0888d1f15387bc5bbd051080ddbc` | 18846 / 2000 / 20 | `4bf940aecc7f18d908148e4de1699d1a09b6ca3d843763d0264053decf6645f0` |

All three runs are `complete`; `failures.json` is empty. The independent audit verified dataset identity, source hash, row and class counts, contiguous indices, seed, protocol, commit, runtime scikit-learn version, manifests, assignments, weight files, training logs, and exact raw-average fusion. Every declared output hash passed.

| Artifact | SHA-256 | Result |
|---|---|---|
| REUTERS archive | `dcd1dd3890889844765bdac548b2c93f3acf91aa54b6489f4a5584b55cd87f8b` | PASS |
| RCV1 archive | `610636d67bd7f253cd3d0174607de7d10cf5fe06bd041b3a25e8cd2791fe2f2f` | PASS |
| 20NEWS archive | `2d8cf65b7d1860086ea8e78707160f536635d89889cfeeb6b7891c50366ebd84` | PASS |
| Combined ZIP | `4f9d580d06d69726e913bd0750f3beb5b263c917e02ac2db359e313ae2811008` | PASS |

Machine-readable audit: `artifacts/kaggle/mvdec-2025-aligned-balanced-a005-seed42/a005_seed42_independent_audit.json`.

## Independently verified metrics

ACC/NMI are percentages. Fused metrics recomputed directly from saved assignments match each manifest exactly.

| Dataset | View1 diagnostic | View2 | Fused | Paper fused | Fused delta vs paper |
|---|---:|---:|---:|---:|---:|
| REUTERS | 45.67 / 15.43 | 42.20 / 9.74 | 45.55 / 15.47 | 78.13 / 61.41 | -32.58 / -45.94 |
| RCV1 | 46.36 / 18.88 | 35.44 / 5.61 | 45.33 / 18.42 | 71.93 / 48.85 | -26.60 / -30.43 |
| 20NEWS | 26.50 / 27.34 | 16.06 / 10.87 | 28.71 / 29.63 | 41.63 / 45.43 | -12.92 / -15.80 |

MvDEC View1 is a jointly trained diagnostic. It is not standalone `DEKM_F`; the frozen standalone DEKM control must be reported separately.

## Comparison with previous protocols

### Fused versus old v2

| Dataset | Old v2 | Alignment 0.05 | Delta |
|---|---:|---:|---:|
| REUTERS | 35.10 / 4.32 | 45.55 / 15.47 | +10.45 / +11.15 |
| RCV1 | 31.80 / 5.28 | 45.33 / 18.42 | +13.53 / +13.14 |
| 20NEWS | 17.00 / 9.66 | 28.71 / 29.63 | +11.71 / +19.97 |

### Fused versus alignment 1.0

| Dataset | Alignment 1.0 | Alignment 0.05 | Delta |
|---|---:|---:|---:|
| REUTERS | 44.87 / 15.57 | 45.55 / 15.47 | +0.68 / -0.10 |
| RCV1 | 45.38 / 18.70 | 45.33 / 18.42 | -0.05 / -0.28 |
| 20NEWS | 21.18 / 21.46 | 28.71 / 29.63 | +7.53 / +8.18 |

The 0.05 run does not satisfy the gate requiring improvement over alignment 1.0 on every dataset: RCV1 regresses slightly and REUTERS NMI also regresses slightly.

### Individual-view preservation versus old v2

| Dataset | Old v2 View1 | New View1 | Old v2 View2 | New View2 |
|---|---:|---:|---:|---:|
| REUTERS | 59.31 / 30.80 | 45.67 / 15.43 | 62.86 / 39.54 | 42.20 / 9.74 |
| RCV1 | 46.84 / 22.36 | 46.36 / 18.88 | 70.54 / 40.68 | 35.44 / 5.61 |
| 20NEWS | 39.19 / 33.88 | 26.50 / 27.34 | 34.62 / 31.00 | 16.06 / 10.87 |

The individual-view preservation gate fails decisively. View2 collapses on all three datasets, especially RCV1, while View1 also degrades materially except for RCV1 ACC.

## Alignment and fusion diagnosis

| Dataset | Paired cosine | Same-coordinate correlation | Retained norm | Old v2 retained norm |
|---|---:|---:|---:|---:|
| REUTERS | 0.7680 | -0.1339 | 0.9512 | 0.3773 |
| RCV1 | 0.8069 | 0.0224 | 0.9659 | 0.4403 |
| 20NEWS | 0.9197 | -0.3571 | 0.9825 | 0.5300 |

Destructive cancellation is fixed: raw averaging now retains 95.1%-98.2% of the component norm instead of only 37.7%-53.0%. The remaining poor fused score is therefore not a fusion arithmetic failure. It reflects weak jointly refined representations, especially View2.

The positive row-wise cosine together with weak or negative same-coordinate correlation also shows that row-direction alignment does not guarantee coordinate-wise semantic alignment or clustering quality.

## Optimization evidence

| Dataset | Steps | Final raw L1 / L2 / L4 / L5 | Weighted L5 (`0.05 * L5`) | Stop |
|---|---:|---:|---:|---|
| REUTERS | 70 | 1.52274 / 0.05958 / 0.00288 / 0.47534 | 0.02377 | converged assignment |
| RCV1 | 70 | 1.52932 / 0.05198 / 0.00323 / 0.40601 | 0.02030 | converged assignment |
| 20NEWS | 230 | 1.32126 / 0.01171 / 0.00034 / 0.16472 | 0.00824 | converged assignment |

L5 is small in the final weighted objective, yet the representations still lose substantial cluster quality. Lowering L5 avoids the near-perfect over-alignment seen at weight 1.0, but it does not restore the views. This rules out the hypothesis that only an excessively large alignment coefficient caused the failure.

The evidence now points to the balanced L1/L2 objective and/or the joint refinement dynamics as the remaining cause. It does not yet distinguish between those two possibilities.

## Reproducibility note

- Assignment-derived fused ACC/NMI exactly matches the manifests and is authoritative.
- Local KMeans replay used scikit-learn 1.6.1 as required, but local NumPy was 2.4.6 while Kaggle used NumPy 2.1.3.
- RCV1 replays exactly. Small differences remain for some REUTERS and 20NEWS View1/View2 results and 20NEWS fused NMI; the largest observed differences are about 0.40 ACC points and 0.44 NMI points.
- Future bit-stable replay must pin and record NumPy as well as scikit-learn. This does not affect the independently verified saved-assignment metrics.

## Gate decision

| Gate | Result | Evidence |
|---|---|---|
| Complete, provenance-valid seed-42 runs | PASS | Three datasets complete; all identity and hash checks pass |
| No destructive cancellation | PASS | Retained norm 0.951-0.982 |
| Preserve individual-view quality | FAIL | Large View1/View2 losses versus old v2 |
| Beat old v2 fused | PASS | ACC and NMI improve on all datasets |
| Beat alignment 1.0 fused | FAIL | RCV1 ACC/NMI and REUTERS NMI regress |
| Recover paper-level performance | FAIL | Fused remains 12.92-32.58 ACC and 15.80-45.94 NMI points below paper |

Final decision: **GATE FAIL**. Do not run seeds 43-44. Do not promote this protocol as official MvDEC.

## Next experiment

Run one seed-42 ablation with exactly one objective change:

- Keep balanced L1/L2, disabled duplicate L3, L4, architecture, data, full-batch refinement, and raw equal-average fusion unchanged.
- Set `alignment_weight` from `0.05` to `0.0`.
- Reuse the same integrity, individual-view preservation, cancellation, and paper-comparison gates.

This isolates whether even the small L5 term is causing the representation damage. If the no-alignment balanced-loss run still collapses the views, stop tuning L5 and restore the old-v2 L1/L2 scaling while testing a small alignment term as the next single-change experiment.
