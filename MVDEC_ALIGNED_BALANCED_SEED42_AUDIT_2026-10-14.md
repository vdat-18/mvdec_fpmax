# MvDEC aligned-balanced seed-42 audit (2026-10-14)

## Verdict

`mvdec_2025_aligned_balanced_full_batch_v1` fixes the destructive raw-average fusion failure, but it does **not** pass the scientific performance gate and is **not** the official MvDEC result. Do not run seeds 43-44.

The two views are now almost parallel (`paired_cosine_mean` 0.992-0.998; retained fused norm 0.998-0.999), so averaging no longer cancels them. However, the individual representations lost substantial clustering quality and fused performance remains 20.45-33.26 ACC points and 23.97-45.84 NMI points below the paper.

This run is a corrected diagnostic, not an exact-paper reproduction.

## Run identity and integrity

- Kaggle kernel: `vandatzz/mvdec-2025-aligned-balanced-seed-42`, Version 1, GPU T4.
- Protocol: `mvdec_2025_aligned_balanced_full_batch_v1`.
- Training commit: `b1fae61fd46c423c06366b338fe6a407f859c27d`.
- Seed: 42 only, order REUTERS -> RCV1 -> canonical sklearn 20NEWS.
- All three manifests are `complete`; `failures.json` is empty.
- Dataset source IDs, source SHA-256 values, row counts, class counts, contiguous sample indices, seed, protocol, and commit all match the expected contracts.
- All three archive SHA-256 sidecars match.
- Every declared output (artifact, assignments, four weight files, and training log) passes `validate_manifest_outputs`.
- Text hashes use the repository's CRLF/LF-normalized SHA-256 contract; raw training-log bytes differ only by newline encoding.
- `assignments.csv` exactly matches the fused labels and true labels stored in `artifact.pkl`.
- `h_fused` is exactly the raw equal average `(h_view1 + h_view2) / 2` within `1e-7`.

Independent machine-readable audit: `artifacts/kaggle/mvdec-2025-aligned-balanced-seed42/aligned_balanced_seed42_independent_audit.json`.

## Independently verified metrics

ACC/NMI are percentages. The fused values were independently recomputed from `assignments.csv` and match the manifests exactly.

| Dataset | View1 diagnostic | View2 | Fused | Paper fused | Fused delta vs paper |
|---|---:|---:|---:|---:|---:|
| REUTERS | 45.24 / 15.36 | 40.30 / 10.52 | 44.87 / 15.57 | 78.13 / 61.41 | -33.26 / -45.84 |
| RCV1 | 44.96 / 17.22 | 42.91 / 15.21 | 45.38 / 18.70 | 71.93 / 48.85 | -26.55 / -30.15 |
| 20NEWS | 13.62 / 10.93 | 11.60 / 9.76 | 21.18 / 21.46 | 41.63 / 45.43 | -20.45 / -23.97 |

MvDEC View1 is a jointly trained diagnostic. It is not standalone `DEKM_F`.

## Change versus old v2

| Dataset | Old v2 fused | New fused | New minus old |
|---|---:|---:|---:|
| REUTERS | 35.10 / 4.32 | 44.87 / 15.57 | +9.77 / +11.25 |
| RCV1 | 31.80 / 5.28 | 45.38 / 18.70 | +13.58 / +13.42 |
| 20NEWS | 17.00 / 9.66 | 21.18 / 21.46 | +4.18 / +11.80 |

Fusion improves over old v2 on every dataset, but the improvement comes from eliminating cancellation, not from recovering paper-level representations.

## Alignment diagnosis

| Dataset | Paired cosine | Same-coordinate correlation | Retained norm | Old v2 paired cosine | Old v2 retained norm |
|---|---:|---:|---:|---:|---:|
| REUTERS | 0.9917 | 0.3228 | 0.9983 | -0.7600 | 0.3773 |
| RCV1 | 0.9929 | 0.3309 | 0.9984 | -0.6682 | 0.4403 |
| 20NEWS | 0.9975 | -0.1713 | 0.9994 | -0.5298 | 0.5300 |

The explicit L5 term over-corrects orientation: row directions become nearly identical, while coordinate-level correlation is only moderate and is still negative on average for 20NEWS after centering. Raw averaging is numerically safe now, but the shared direction is not sufficiently cluster-discriminative.

## Optimization evidence

| Dataset | Steps | First L1/L2/L4/L5 | Final L1/L2/L4/L5 | Stop |
|---|---:|---:|---:|---|
| REUTERS | 70 | 1.673 / 8.657 / 7.312 / 1.206 | 1.628 / 0.057 / 0.003 / 0.017 | converged assignment |
| RCV1 | 70 | 1.834 / 7.529 / 5.732 / 1.193 | 1.619 / 0.051 / 0.002 / 0.015 | converged assignment |
| 20NEWS | 510 | 1.628 / 2.876 / 1.988 / 1.046 | 1.350 / 0.008 / 0.001 / 0.005 | converged assignment |

L5 is not the largest scalar term at initialization, so the evidence does not support the simple claim that it numerically dominates the total loss. Nevertheless, weight 1 is strong enough to drive cosine alignment to almost 1.0. At convergence, L2/L4/L5 are almost exhausted and reconstruction dominates; the model has converged to highly aligned but weakly discriminative latents. Because dimensional loss normalization and L5 were introduced together, this run alone cannot assign all degradation to one of them.

## Reproducibility note

Independent local KMeans with `n_init=100` and seed 42 exactly reproduces fused ACC for all datasets and fused NMI for REUTERS/RCV1. Small discrepancies occur for some View1/View2 results and 20NEWS fused NMI (maximum ACC difference 0.58 percentage points; maximum NMI difference 0.24 points). The Kaggle environment did not record or pin `scikit-learn`, while the local audit used 1.9.0. The exact saved fused assignments remain fully reproducible from `assignments.csv`, but future runs should record and pin scikit-learn before treating cross-machine KMeans replay as bit-stable.

## Next experiment

Run one seed-42 ablation only:

- Keep the balanced L1/L2 formulation, disabled duplicate L3, full-batch refinement, raw equal-average fusion, architecture, data, and all other settings unchanged.
- Reduce only `alignment_weight` from `1.0` to `0.05`.
- Add gates requiring: no destructive cancellation, individual View1/View2 quality not materially below old v2, fused quality above both old v2 and this run, and exact artifact/provenance audit.
- Record and pin the scikit-learn version.

This isolates whether L5 over-alignment caused the representation collapse. Do not tune multiple weights simultaneously and do not start seeds 43-44 until this seed-42 ablation passes.
