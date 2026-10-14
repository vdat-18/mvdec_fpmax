# MvDEC aligned-balanced alignment-0.00 seed-42 audit (2026-10-14)

## Verdict

`mvdec_2025_aligned_balanced_alignment_000_full_batch_v1` **fails the scientific gate**. It is a corrected diagnostic, not an exact-paper reproduction, not the official MvDEC result, and seeds 43-44 must not be run.

Disabling L5 improves some individual-view metrics, especially on 20NEWS, but fused ACC/NMI becomes slightly worse than alignment 0.05 on all three datasets. The views remain substantially weaker than old v2 and fused performance remains far below the paper. This isolates the remaining failure: L5 is not the primary cause of representation collapse. The balanced L1/L2 formulation and/or the associated joint-refinement dynamics is the next component to remove.

## Run identity

- Kaggle kernel: `vandatzz/mvdec-2025-aligned-balanced-no-alignment-seed-42`, Version 1, GPU T4.
- Protocol: `mvdec_2025_aligned_balanced_alignment_000_full_batch_v1`.
- Training commit: `73dc3151b81cacc41e4e421be0892f6b2bcd3405`.
- Notebook commit: `3359049d7b8d9d475f9d53fd7728fcacf1302ded`.
- Seed: 42 only, in order REUTERS -> RCV1 -> canonical sklearn 20NEWS.
- Refinement: full batch; no mini-batch substitution.
- Kaggle runtime: NumPy 2.1.3, scikit-learn 1.6.1, TensorFlow 2.20.0.
- Only objective change from alignment 0.05: `alignment_weight` 0.05 -> 0.0.

## Provenance and integrity

| Dataset | Source ID | Source SHA-256 | Rows / features / classes | Config hash |
|---|---|---|---:|---|
| REUTERS | `dekm_google_drive_reuters_10k` | `7eb3f41aca38ffb93c104eef261adc3273d1da57c3750e04c92a67e335d423a1` | 10000 / 2000 / 4 | `894c9ee25f0f4f727dd9798ac641812c42e243d0adf14c430ada50ddcfc1c84e` |
| RCV1 | `dekm_google_drive_indices_sklearn_rcv1` | `305877a57be6f4984b4f890d1b4809bf631582116e7a1c23045ae93a540574d6` | 10000 / 2000 / 4 | `bf68e6d992511bc91c2ba47bf57e556f7411265e1dfd0545afc4849c6ea5be44` |
| 20NEWS | `sklearn_fetch_20newsgroups_all_tfidf2000_18846` | `663e02a432bfd1e0a06b4f9863fb34b2c04f0888d1f15387bc5bbd051080ddbc` | 18846 / 2000 / 20 | `94f86703ce7ad102cb0d068966bae201c286920dadf4bf94dfec81a1fbf098ec` |

All three runs are complete and `failures.json` is empty. Independent validation passed for protocol, commit, seed, runtime versions, source identity and hash, row/class counts, sample indices, manifests, every declared output hash, assignments, weights, training logs, and exact raw equal-average fusion.

| Artifact | SHA-256 | Result |
|---|---|---|
| REUTERS archive | `cb5196c34c17a732456698499c87f13d4bd3de50b59dd2cdc48c6dc90e0fc736` | PASS |
| RCV1 archive | `04a2be23148e0fceea940075858c2054296d4f9bb6173507d7bd8469247572dc` | PASS |
| 20NEWS archive | `88d9f8a8275762a18acdbe3aed9910706cf1401e6d4bc0bb1cd26d9fb4f145d2` | PASS |
| Combined ZIP | `5fb37a133da058f266c09496affc184c82cbc1342beb7461e0ca4f687c7bc914` | PASS |

Machine-readable audit: `artifacts/kaggle/mvdec-2025-aligned-balanced-a000-seed42/a000_seed42_independent_audit.json`.

## Independently verified metrics

ACC/NMI are percentages. Fused metrics recomputed directly from saved assignments match the manifests exactly.

| Dataset | View1 diagnostic | View2 | Fused | Paper fused | Fused delta vs paper |
|---|---:|---:|---:|---:|---:|
| REUTERS | 45.45 / 15.35 | 42.49 / 12.23 | 45.32 / 15.39 | 78.13 / 61.41 | -32.81 / -46.02 |
| RCV1 | 45.78 / 18.55 | 36.89 / 5.81 | 44.79 / 17.99 | 71.93 / 48.85 | -27.14 / -30.86 |
| 20NEWS | 29.94 / 31.79 | 20.24 / 16.88 | 28.46 / 29.45 | 41.63 / 45.43 | -13.17 / -15.98 |

MvDEC View1 is a jointly trained diagnostic. It is not standalone `DEKM_F`; the frozen standalone DEKM control must be reported separately.

## Comparison with alignment 0.05

| Dataset | Alignment 0.05 fused | No-alignment fused | Delta |
|---|---:|---:|---:|
| REUTERS | 45.55 / 15.47 | 45.32 / 15.39 | -0.23 / -0.08 |
| RCV1 | 45.33 / 18.42 | 44.79 / 17.99 | -0.54 / -0.43 |
| 20NEWS | 28.71 / 29.63 | 28.46 / 29.45 | -0.25 / -0.18 |

The no-alignment run loses fused ACC and NMI on every dataset. Therefore, a small L5 term is mildly beneficial to raw-average fusion, but it is not sufficient to recover representation quality.

| Dataset | View1 delta vs 0.05 | View2 delta vs 0.05 |
|---|---:|---:|
| REUTERS | -0.22 / -0.08 | +0.29 / +2.49 |
| RCV1 | -0.58 / -0.33 | +1.45 / +0.20 |
| 20NEWS | +3.44 / +4.45 | +4.18 / +6.01 |

Removing L5 helps View2 on every dataset and both views on 20NEWS, confirming that L5 imposes a representation-quality cost. However, the gains are too small to explain or repair the full collapse.

## Comparison with old v2

| Dataset | Old v2 fused | No-alignment fused | Delta |
|---|---:|---:|---:|
| REUTERS | 35.10 / 4.32 | 45.32 / 15.39 | +10.22 / +11.07 |
| RCV1 | 31.80 / 5.28 | 44.79 / 17.99 | +12.99 / +12.71 |
| 20NEWS | 17.00 / 9.66 | 28.46 / 29.45 | +11.46 / +19.79 |

The balanced-loss family still improves fused results over old v2 because it avoids severe anti-alignment, but it does not preserve the strong old-v2 individual views.

| Dataset | Old v2 View1 | New View1 | Old v2 View2 | New View2 |
|---|---:|---:|---:|---:|
| REUTERS | 59.31 / 30.80 | 45.45 / 15.35 | 62.86 / 39.54 | 42.49 / 12.23 |
| RCV1 | 46.84 / 22.36 | 45.78 / 18.55 | 70.54 / 40.68 | 36.89 / 5.81 |
| 20NEWS | 39.19 / 33.88 | 29.94 / 31.79 | 34.62 / 31.00 | 20.24 / 16.88 |

The individual-view preservation gate fails decisively, especially for View2.

## Alignment and fusion diagnosis

| Dataset | Paired cosine | Same-coordinate correlation | Retained norm | Alignment 0.05 retained norm | Old v2 retained norm |
|---|---:|---:|---:|---:|---:|
| REUTERS | 0.4108 | -0.2172 | 0.8680 | 0.9512 | 0.3773 |
| RCV1 | 0.5223 | -0.0323 | 0.9248 | 0.9659 | 0.4403 |
| 20NEWS | 0.5894 | -0.3849 | 0.9357 | 0.9825 | 0.5300 |

Disabling L5 reduces alignment and slightly increases cancellation, but the result is still far safer than old v2. Raw averaging retains 86.8%-93.6% of component norm, so catastrophic fusion cancellation is not the remaining blocker.

## Optimization evidence

| Dataset | Steps | First L1 / L2 / L4 | Final L1 / L2 / L4 | Stop |
|---|---:|---:|---:|---|
| REUTERS | 70 | 1.52375 / 9.06094 / 7.30547 | 1.49893 / 0.06472 / 0.00330 | converged assignment |
| RCV1 | 70 | 1.65770 / 7.86749 / 5.66810 | 1.51034 / 0.05479 / 0.00323 | converged assignment |
| 20NEWS | 250 | 1.47755 / 2.96103 / 1.98384 | 1.31107 / 0.01293 / 0.00047 | converged assignment |

There is no L5 contribution in this run. L2 and L4 are still driven almost to zero while the final views remain weak. This is direct evidence that representation degradation persists without alignment loss and points to balanced L1/L2 scaling and/or the joint-refinement target dynamics.

## Reproducibility note

- Assignment-derived fused ACC/NMI exactly matches every manifest and is authoritative.
- Local replay used scikit-learn 1.6.1 as required, but local NumPy was 2.4.6 versus Kaggle NumPy 2.1.3.
- Fused ACC replays exactly on all datasets; fused NMI differs only by 0.0012 percentage points on 20NEWS.
- Some View1/View2 KMeans solutions differ more, especially 20NEWS View2, confirming that NumPy must also match for bit-stable KMeans replay. This does not affect saved-assignment verification.

## Gate decision

| Gate | Result | Evidence |
|---|---|---|
| Complete, provenance-valid seed-42 runs | PASS | Three datasets complete; all identity and hash checks pass |
| Avoid catastrophic cancellation | PASS | Retained norm 0.868-0.936 versus old-v2 0.377-0.530 |
| Preserve individual-view quality | FAIL | Large losses versus old v2, especially View2 |
| Beat old v2 fused | PASS | ACC/NMI improve on all datasets |
| Beat alignment 0.05 fused | FAIL | ACC/NMI regress on all datasets |
| Recover paper-level performance | FAIL | Fused remains 13.17-32.81 ACC and 15.98-46.02 NMI points below paper |

Final decision: **GATE FAIL**. Do not run seeds 43-44. Do not promote this protocol as official MvDEC.

## Next experiment

Stop tuning L5. Run one seed-42 ablation that restores the old-v2 L1/L2 loss scaling while keeping the corrected objective structure:

- Restore old-v2 L1/L2 dimensional scaling.
- Keep duplicate L3 disabled, full-batch refinement, raw equal-average fusion, architecture, data, stopping rules, and all other settings fixed.
- Use `alignment_weight = 0.05`, because it improves fusion slightly without the weight-1.0 over-alignment.
- Apply the same individual-view preservation, cancellation, fused-performance, provenance, and paper-comparison gates.

This makes one causal change: loss scaling. If individual views recover, balanced dimensional scaling caused the collapse. If they do not, the remaining problem lies in joint-refinement targets or architecture rather than loss normalization.
