# MvDEC legacy-scaled centered-alignment-1.0 seed-42 audit (2026-10-15)

## Verdict

`mvdec_2025_legacy_scaled_centered_alignment_100_full_batch_v1` **fails the scientific gate**. It is a corrected diagnostic, not an exact-paper reproduction, not official MvDEC, and seeds 43-44 must not be run.

The centered L5 implementation is mathematically correct and the run is provenance-valid, but the objective scaling is not. With legacy sum-over-2,000-dimensional reconstruction, unit-weight centered alignment contributes only 0.142%-0.143% of the final objective. During every dataset run, L5 moves in the wrong direction: it starts near 2.00-2.17 and rises monotonically to 3.50-3.67, close to the maximum value 4 for opposite normalized centered vectors. Assignment-only stopping then accepts the run while centered structure remains strongly anti-aligned.

Mean centering removes the previous shared-offset loophole, but it does not by itself make alignment strong enough to compete with reconstruction. Raw paired cosine becomes negative (-0.588 to -0.463), centered paired cosine remains -0.836 to -0.750, and centered raw averaging retains only 0.312-0.399 of component norm. Fused quality improves over uncentered legacy A1.0 because useful individual-view structure is partly recovered, not because the views now share a coordinate system.

## Run identity

- Kaggle kernel: `vandatzz/mvdec-2025-legacy-scaled-centered-a100-seed-42`, Version 2, GPU T4, completed successfully.
- Version 1 failed before training because the canonical 20NEWS Figshare redirect returned HTTP 403. Version 2 downloaded the same archive through a fail-closed two-source provisioner and required SHA-256 `8f1b2514ca22a5ade8fbb9cfa5727df95fa587f4c87b786e15c759fa66d95610`.
- Protocol: `mvdec_2025_legacy_scaled_centered_alignment_100_full_batch_v1`.
- Training commit: `499e4668264203ca24f70cdea6a3a2f132e86b90`.
- Notebook commit: `46a6d605f28d79c46ba41a5a8f227c4d8fd1883b`.
- Notebook SHA-256: `527a3f66242661cf10fd514f0a49d14a16bbb6ad07348536e82756bddaa80476`.
- Seed: 42 only, verified from Kaggle logs in order REUTERS -> RCV1 -> canonical 20NEWS.
- Refinement: full-dataset optimizer updates; no mini-batch substitution.
- Kaggle runtime: NumPy 2.1.3, scikit-learn 1.6.1, TensorFlow 2.20.0.
- Controlled change from legacy-scaled uncentered A1.0: L5 changed from uncentered row-L2 to full-batch mean-centered row-L2.

## Provenance and integrity

| Dataset | Source ID | Source SHA-256 | Rows / features / classes | Config hash |
|---|---|---|---:|---|
| REUTERS | `dekm_google_drive_reuters_10k` | `7eb3f41aca38ffb93c104eef261adc3273d1da57c3750e04c92a67e335d423a1` | 10000 / 2000 / 4 | `d93bd4f8ec3c217840519ee5d5ecb7499bdd955d75c2efbc2399cc16b9580b0d` |
| RCV1 | `dekm_google_drive_indices_sklearn_rcv1` | `305877a57be6f4984b4f890d1b4809bf631582116e7a1c23045ae93a540574d6` | 10000 / 2000 / 4 | `f3070f9b030af6c60a579d3c006093003abb56b80108bdbe8e8934a587d704cc` |
| 20NEWS | `sklearn_fetch_20newsgroups_all_tfidf2000_18846` | `663e02a432bfd1e0a06b4f9863fb34b2c04f0888d1f15387bc5bbd051080ddbc` | 18846 / 2000 / 20 | `187f97b3674d2330876fa45c0a6d33a336eba9d1d0b74c105389a9dd24b7922f` |

All three manifests are complete and `failures.json` is empty. Independent checks passed for protocol, training commit, seed, runtime versions, source identity/hash, dimensions, alignment mode, config hash, assignments, pretrain/final weights, training logs, exact raw-average fusion, every manifest-declared normalized output hash, archives, and the combined ZIP.

| Package | SHA-256 | Result |
|---|---|---|
| REUTERS archive | `3f5600d32be87512b8277a163ead849cf396388f6ea332165c4d14b7d23bc31b` | PASS |
| RCV1 archive | `8fc6edbb11c62a618412f754f963c14a7a4c7a620db5c87267c735e8c2ae5046` | PASS |
| 20NEWS archive | `88ac21e317a1f369137a3c33f09f48cf61228aad2e6fa561428ec13fccbae36f` | PASS |
| Combined ZIP | `79cf49999e34ac323d0cac95a17df39e8ab0b0f5ba3e72f6f17eb642bc0530ba` | PASS |

Machine-readable audit: `artifacts/kaggle/mvdec-2025-legacy-scaled-centered-a100-seed42/legacy_scaled_centered_a100_seed42_independent_audit.json`.

## Independently verified metrics

ACC/NMI are percentages. Fused ACC/NMI independently recomputed from saved assignments match every manifest exactly.

| Dataset | View1 diagnostic | View2 | Fused | Paper fused | Fused delta vs paper |
|---|---:|---:|---:|---:|---:|
| REUTERS | 52.18 / 19.64 | 64.43 / 31.95 | 44.65 / 10.69 | 78.13 / 61.41 | -33.48 / -50.72 |
| RCV1 | 47.43 / 21.89 | 69.73 / 39.75 | 36.74 / 9.54 | 71.93 / 48.85 | -35.19 / -39.31 |
| 20NEWS | 38.39 / 33.59 | 28.95 / 24.41 | 25.02 / 19.10 | 41.63 / 45.43 | -16.61 / -26.33 |

MvDEC View1 above is a jointly trained diagnostic. It is **not** the standalone `DEKM_F` control.

### Standalone DEKM_F paper control

| Dataset | Standalone DEKM_F control in paper |
|---|---:|
| REUTERS | 76.42 / 59.99 |
| RCV1 | 62.12 / 35.98 |
| 20NEWS | 41.20 / 43.76 |

## Comparison with uncentered legacy A1.0

| Dataset | Representation | Uncentered A1.0 | Centered A1.0 | Delta |
|---|---|---:|---:|---:|
| REUTERS | View1 | 51.34 / 18.35 | 52.18 / 19.64 | +0.84 / +1.29 |
| REUTERS | View2 | 59.80 / 28.64 | 64.43 / 31.95 | +4.63 / +3.31 |
| REUTERS | Fused | 42.58 / 8.10 | 44.65 / 10.69 | +2.07 / +2.58 |
| RCV1 | View1 | 47.49 / 21.39 | 47.43 / 21.89 | -0.06 / +0.51 |
| RCV1 | View2 | 57.44 / 27.33 | 69.73 / 39.75 | +12.29 / +12.42 |
| RCV1 | Fused | 35.42 / 7.98 | 36.74 / 9.54 | +1.32 / +1.56 |
| 20NEWS | View1 | 37.80 / 32.68 | 38.39 / 33.59 | +0.59 / +0.91 |
| 20NEWS | View2 | 31.41 / 26.94 | 28.95 / 24.41 | -2.46 / -2.53 |
| 20NEWS | Fused | 19.38 / 12.31 | 25.02 / 19.10 | +5.64 / +6.79 |

Centering improves fused metrics and usually restores individual-view quality, but it does not remove coordinate cancellation.

## Alignment audit

| Dataset | Raw cosine | Mean-vector cosine | Centered cosine | Same-coordinate corr | Raw retained | Centered retained |
|---|---:|---:|---:|---:|---:|---:|
| REUTERS | -0.5883 | 0.2035 | -0.8215 | -0.8339 | 0.4977 | 0.3461 |
| RCV1 | -0.5767 | 0.4707 | -0.8362 | -0.8502 | 0.4896 | 0.3125 |
| 20NEWS | -0.4626 | 0.4497 | -0.7503 | -0.7406 | 0.5630 | 0.3986 |

Compared with uncentered A1.0, the centered cosines change by only +0.002 (REUTERS), -0.054 (RCV1), and +0.014 (20NEWS). The original anti-aligned sample-dependent geometry therefore remains essentially unchanged. What disappears is the artificial parallel mean offset: mean-vector cosine falls from 0.925-0.960 to 0.203-0.471. This makes raw diagnostics look worse but exposes the real geometry more honestly.

## Label-free controls

These are diagnostic controls, not official MvDEC results.

| Dataset | Raw fused | Procrustes-aligned average | Concatenation control |
|---|---:|---:|---:|
| REUTERS | 44.65 / 10.69 | 67.26 / 40.43 | 65.45 / 39.23 |
| RCV1 | 36.74 / 9.54 | 66.70 / 40.36 | 65.60 / 38.05 |
| 20NEWS | 25.03 / 19.11 | 38.46 / 31.54 | 43.05 / 35.96 |

The large gap confirms that useful cluster structure remains in the views, while raw coordinate averaging is still the dominant failure.

## Optimization evidence

| Dataset | Windows / updates | Final L1 / L2 / L4 / L5 | L5 share | Reconstruction share | Stop |
|---|---:|---:|---:|---:|---|
| REUTERS | 64 / 640 | 2531.9957 / 4.7053 / 0.2524 / 3.6421 | 0.1434% | 99.6615% | converged assignment |
| RCV1 | 111 / 1110 | 2558.5752 / 3.2731 / 0.1935 / 3.6731 | 0.1432% | 99.7217% | converged assignment |
| 20NEWS | 148 / 1480 | 2454.6692 / 3.7417 / 0.2083 / 3.5010 | 0.1422% | 99.6974% | converged assignment |

For normalized vectors, centered L5 equals `2 - 2 * centered_cosine`. The saved L5 values exactly agree with the independently measured centered cosine, confirming that the implementation is evaluating the intended quantity.

However, L5 rises monotonically rather than converging:

| Dataset | First-window L5 | Final-window L5 | Direction |
|---|---:|---:|---|
| REUTERS | 2.0274 | 3.6421 | worse |
| RCV1 | 2.0025 | 3.6731 | worse |
| 20NEWS | 2.1748 | 3.5010 | worse |

This rules out “train longer with the same objective” as the next fix. Assignment-aware stopping is inadequate, but stopping is not the primary cause: reconstruction actively drives the model toward stronger anti-alignment because unit-weight L5 is too small under summed dimensional scaling.

## Gate decision

| Gate | Result | Evidence |
|---|---|---|
| Complete, provenance-valid seed-42 runs | PASS | Three datasets complete; manifests, hashes, archives and ZIP validate |
| Centered L5 implementation correct | PASS | L5 matches `2 - 2 * centered cosine`; full-batch constraint and metadata validate |
| Preserve individual views near legacy A0.05/A1.0 | PARTIAL | RCV1 View2 improves strongly, but 20NEWS View2 regresses and views remain below paper controls |
| Remove centered destructive cancellation | FAIL | Centered cosine -0.836 to -0.750; retained norm 0.312-0.399 |
| Improve over uncentered A1.0 fused | PASS | Fused ACC/NMI improves on all three datasets |
| Beat balanced A0.05 fused | FAIL | Centered fused remains lower on all three datasets |
| Recover paper-level performance | FAIL | Large fused gaps remain on every dataset |

Final decision: **GATE FAIL**. Do not run seeds 43-44 and do not promote this protocol as official MvDEC.

## Required next correction

The next controlled seed-42 diagnostic must correct loss scale, not merely run longer:

- Keep centered L5, legacy `sum_squared_dimensions`, disabled duplicate L3, L4, architecture, data, raw equal-average fusion and full-batch refinement unchanged.
- Increase centered alignment strength according to the dimensional-scale conversion. Balanced A0.05 used mean-over-2,000-dimensional reconstruction; its equivalent under summed reconstruction is approximately `0.05 * 2000 = 100`.
- Give this a new immutable protocol ID and make the dimensional-scale rationale explicit in its contract; do not mutate centered A1.0.
- Continue using assignment stopping for the first scale-only ablation so only alignment strength changes. If L5 finally decreases but assignment stopping terminates before alignment stabilizes, add alignment-aware stopping in a separate protocol.
- Require L5 to decrease from its initial value, centered cosine to become non-negative, centered retained norm to rise materially, individual-view quality to remain near the legacy runs, and fused quality to beat balanced A0.05 before seeds 43-44 are allowed.

This proposed scale correction is still a diagnostic. The paper does not specify cross-view latent alignment, its weight, coordinate matching, seeds, or the exact refinement schedule, so it cannot be called an exact official reproduction.
