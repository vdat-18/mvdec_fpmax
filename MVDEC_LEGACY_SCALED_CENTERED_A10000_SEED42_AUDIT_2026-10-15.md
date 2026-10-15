# MvDEC legacy-scaled centered A100.0 seed-42 cancellation audit

## Decision

The scale-matched centered-alignment diagnostic is closed as a failed gate. It
is not an exact-paper reproduction and must not be rerun, extended to seeds
43-44, or silently converted to mini-batch training.

Kaggle accepted cancellation of Version 1, session `358487209`, and reported
`CANCEL_ACKNOWLEDGED` at `2026-10-15T13:55:55+07:00`. The final GPU quota
snapshot was 29.60 of 30.00 hours used, with 0.40 hours remaining and refresh at
`2026-10-17T00:00:00`.

## Immutable run identity

- Kernel: `vandatzz/mvdec-2025-legacy-scaled-centered-a10000-seed-42`
- Protocol: `mvdec_2025_legacy_scaled_centered_alignment_10000_full_batch_v1`
- Training commit: `956c4b304f750e3dda9730648832d4d9e320166f`
- Notebook commit: `ae9ae27bcda137431725fd0d90b65f92eac24c76`
- Seed: 42 only
- Dataset order: REUTERS, RCV1, canonical 20NEWS
- Numerical runtime recorded by the manifests: NumPy 2.1.3 and scikit-learn 1.6.1
- Refinement: full dataset, fixed 10-update K-means windows, maximum 14,000
  training steps
- Only controlled change from centered A1.0: alignment weight 1.0 -> 100.0

## Recovered outputs

Kaggle preserved complete, internally packaged outputs for REUTERS and RCV1,
plus the running manifest, pretrained weights, partial training log, and stdout
for 20NEWS. The durable download is under
`artifacts/kaggle/mvdec-2025-legacy-scaled-centered-a10000-seed42-cancelled`.

The completed archive checksums match their sidecars:

| Dataset | Archive SHA-256 | Check |
| --- | --- | --- |
| REUTERS | `15ed9df39e397f1bc316e415257c5c58e4e41b074cfa0fde1005ea50d1e5868a` | PASS |
| RCV1 | `c53454976f9eca982d23d67be21da0a43935b797d937dfea3aba90d0f6b4641c` | PASS |

The local audit command cannot perform its exact-source recomputation because
the current local environment has NumPy 2.4.6 while the manifests require
NumPy 2.1.3. Independent ACC/NMI recomputation from the downloaded assignment
files nevertheless exactly matches both completed manifests.

## Completed metrics

| Dataset | View1 ACC / NMI | View2 ACC / NMI | Fused ACC / NMI | Paper fused ACC / NMI | Fused delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| REUTERS | 56.66 / 27.29 | 59.99 / 28.50 | 56.63 / 27.21 | 78.13 / 61.41 | -21.50 / -34.20 pp |
| RCV1 | 48.30 / 26.61 | 46.61 / 27.14 | 48.28 / 26.62 | 71.93 / 48.85 | -23.65 / -22.23 pp |

Relative to centered A1.0, fused ACC rises by 11.98 pp on REUTERS and 11.54 pp
on RCV1. That improvement is not a reproduction success: the stronger alignment
also erases the previously strong RCV1 View2 result, dropping it from 69.73 ACC
to 46.61 ACC, and both fused results remain far below Table 2.

## Optimization behavior

The logged `L5_alignment` value is the unweighted alignment loss. Its actual
objective contribution is `100 * L5_alignment`.

| Dataset | Windows / steps | Raw L5 first -> last | Weighted L5 share first -> last | Stop |
| --- | ---: | ---: | ---: | --- |
| REUTERS | 16 / 160 | 1.12123 -> 0.07331 | 3.83% -> 0.28% | converged assignment |
| RCV1 | 19 / 190 | 1.08949 -> 0.05966 | 3.73% -> 0.23% | converged assignment |
| 20NEWS | 1,169 / 11,690 | 1.45887 -> 0.02264 | 5.22% -> 0.09% | cancelled while running |

The scale match therefore does not remain scale matched during optimization.
The model rapidly drives the alignment term down until reconstruction again
dominates the objective. On REUTERS and RCV1 this produces quick assignment
convergence but poor representations. On 20NEWS the assignments never meet the
0.1% stopping threshold: the smallest completed-window change count is 25,
while convergence requires at most 18 changed samples. The final recovered
window has 80 changes, and an earlier late checkpoint spikes to 807 changes.

At cancellation, 20NEWS had consumed about 24,918 seconds (6.92 hours) and
completed 11,690 of 14,000 allowed steps. No final assignments, final weights,
archive, or valid metrics exist for 20NEWS. The partial manifest correctly
remains `running`; it must not be relabeled as complete or failed after the fact.

## Reproduction conclusion

Increasing centered alignment strength is not the missing paper detail. A100.0
improves fused metrics over A1.0 on two datasets, but it does so by forcing the
views toward a common low-quality geometry, fails badly against the paper, and
is practically non-convergent on 20NEWS.

The next reproduction work therefore returned to a paper-constrained ambiguity
rather than tuning alignment weights again. Figures 1-2 show 23-wide per-view
outputs (latent plus reconstruction for the 13-feature example), whereas Eq. 4
defines an average of encoded representations. The bounded, explicitly named
seed-42 diagnostic below tests both full-output interpretations. Its results may
be compared with Table 2, but no protocol is selected solely because it is
numerically close to the published labels.

## Operational closure

- Do not rerun A100.0.
- Do not run A100.0 seeds 43-44.
- Do not reinterpret the cancelled 20NEWS partial state as a result.
- Preserve the downloaded outputs and logs as negative evidence.
- Resume GPU work only after quota refresh and only with a preregistered,
  bounded seed-42 diagnostic targeting a documented paper ambiguity.

## Completed paper-figure fusion diagnostic

The repository now defines two immutable post-training fusion contracts for the
paper's diagram ambiguity:

- `mvdec2025_figure1_full_output_average_diagnostic`: elementwise average of
  the two complete per-view outputs. For the paper's 13-feature example this is
  23 -> 23 dimensions; for the 2,000-feature public benchmarks it is
  2,010 -> 2,010 dimensions.
- `mvdec2025_figure2_full_output_concatenation_diagnostic`: concatenate the two
  complete per-view outputs. For the paper example this is 23 + 23 -> 46
  dimensions; for public benchmarks it is 2,010 + 2,010 -> 4,020 dimensions.

Both contracts record their paper source, operation, input width, output width,
and claim scope `post_training_diagnostic_not_exact_paper_truth`. They are
implemented only as post-training diagnostic transformations; no existing
training protocol or artifact identity was changed.

Kaggle kernel
`vandatzz/mvdec-2025-reuters-paper-figure-audit-seed-42`, Version 5, completed
successfully on CPU on 2026-10-15. It reused the complete centered A1.0 REUTERS
seed-42 weights; it did not pretrain or refine either network. CPU execution was
intentional: this bounded job only performs inference and scikit-learn K-Means,
and scikit-learn K-Means is CPU-based. The retained log reached the final metric
table after about 340 seconds.

Immutable source identity:

- Protocol: `mvdec_2025_legacy_scaled_centered_alignment_100_full_batch_v1`
- Training commit: `499e4668264203ca24f70cdea6a3a2f132e86b90`
- Config hash: `d93bd4f8ec3c217840519ee5d5ecb7499bdd955d75c2efbc2399cc16b9580b0d`
- Source: `dekm_google_drive_reuters_10k`
- Source SHA-256: `7eb3f41aca38ffb93c104eef261adc3273d1da57c3750e04c92a67e335d423a1`
- Numerical runtime: NumPy 2.1.3, scikit-learn 1.6.1
- Audit runner commit: `0f282c502c75b6d3679e2e8d4d14531077c37d56`

| Interpretation | Width | ACC / NMI | Delta from paper ACC / NMI |
| --- | ---: | ---: | ---: |
| Eq. 4 latent average | 10 | 44.65 / 10.69 | -33.48 / -50.72 pp |
| Figure 1 full-output average | 2,010 | 54.73 / 43.51 | -23.40 / -17.90 pp |
| Figure 2 full-output concatenation | 4,020 | 54.07 / 37.47 | -24.06 / -23.94 pp |

Figure 1 full-output averaging is the strongest of these three post-training
diagnostics on this one seed, especially for NMI. It still remains far below
the paper's REUTERS fused result of 78.13 ACC / 61.41 NMI. This does not identify
the paper's hidden implementation and is not a reason to promote Figure 1 as the
canonical protocol. The result only shows that including reconstruction outputs
changes the clustering geometry substantially compared with Eq. 4 latent-only
averaging.

Independent local recomputation from all three assignment files exactly matches
the reported ACC/NMI. Each file contains 10,000 row-aligned samples and four
clusters. Assignment checksums also match the diagnostic manifest:

| Output | SHA-256 |
| --- | --- |
| Eq. 4 assignments | `f473378d1a7dc99ef91829cba37f5f26ff91eb0f14b21c017bbb47b59f3e15a3` |
| Figure 1 assignments | `2ee2b19bd28ef622ecc75fd5e8b8a524ba2de09b309b59a8027fb8f074c53d04` |
| Figure 2 assignments | `400fcb95be751b84d0dccda596d0967c506e0cb99e16d3948c8178f8d514b82f` |
| Metrics CSV | `6abd58cdf46fd60bf3d0fa6b932930cb56b67e76f707f651683022cac347c520` |
| Retained ZIP | `a99cc87f798ca58bc40c8f8bfe2a63ca9fe4911c1d9dbf11f0b3e448521ba07f` |

The durable output is under
`artifacts/kaggle/mvdec-2025-reuters-paper-figure-audit-seed42`.

### Bounded runner status

The post-training runner is now implemented as
`pipeline.mvdec_paper_figure_audit` with CLI entry point
`mvdec-audit-paper-figures`. It accepts exactly one complete run manifest and
fails closed before inference unless the run identity, config hash, complete
output set, HDF5 weight containers, output SHA-256 values, recorded numerical
runtime, canonical source identity/hash, dimensions, architecture identity,
view-input contract, seed, and K-Means restart count are consistent.

Inference uses bounded batches and disk-backed temporary arrays rather than
persisting both complete public outputs in memory. It evaluates the declared
Eq. 4 latent fusion, Figure 1 full-output average, and Figure 2 full-output
concatenation with independent K-Means (`n_init=100`) and the manifest seed. It
exports per-representation ACC/NMI, row-aligned assignments, source/width/claim
metadata, and a diagnostic manifest containing checksums for every retained
output. Temporary full-output matrices are closed and removed after the audit.

Focused validation passes with 54 tests across the new runner, fusion contracts,
manifest handling, public data, and architecture suites. The runner completed
on Kaggle CPU, and no GPU run or seed 43-44 run was created. The REUTERS-only
notebook `notebooks/mvdec_2025_reuters_paper_figure_audit_seed42_kaggle.ipynb`
pins the runner commit `0f282c502c75b6d3679e2e8d4d14531077c37d56` and enforces
a 20-minute subprocess timeout.
