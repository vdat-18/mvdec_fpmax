# MvDEC REUTERS reproduction plan (2026-10-21)

Goal: reproduce MvDEC 2025 Table 2 on REUTERS-10K as closely as the published
evidence allows. All results remain `controlled_diagnostic_not_exact_paper_truth`
(see `MVDEC_REPRODUCTION_BOUNDARY_AND_NEXT_STEP_2026-10-19.md`).

## Verified state

Paper Table 2 (REUTERS) versus the best completed seed-42 runs:

| Representation | Paper ACC / NMI | Best run ACC / NMI | Run |
| --- | ---: | ---: | --- |
| View1 | 0.7642 / 0.5999 | 0.7557 / 0.5932 | imported DEKM_F (seed 42) |
| View2 | 0.6501 / 0.5129 | 0.7199 / 0.5154 | calibrated independent View2 |
| View2 (no calibration) | 0.6501 / 0.5129 | 0.5155 / 0.3223 | joint Eq. 11, true bottleneck (Kaggle snapshot, View1 trainable) |
| Fused | 0.7813 / 0.6141 | 0.7560 / 0.5937 | calibrated independent View2 |

DEKM_F View1 over seeds 42/43/44 (`dekm_f_2021_official_source_full_batch_v1`):
ACC 0.7557 / 0.7582 / 0.7484, NMI 0.5932 / 0.5903 / 0.5729. The spread is under
1 pp ACC, so seed noise does not explain the gap to the paper's 0.7642 / 0.5999.

## Findings that change the plan

1. **Joint refinement stalls after 10 updates.** With imported, already
   converged DEKM_F View1, fused K-Means assignments do not change, so the
   stop rule fires at the first check. View2 is then effectively its
   pretrain-only state (0.515 ACC), and the loss never shapes it. View2 only
   reaches 0.72 when refinement is driven through a representation in which
   it contributes comparably (the calibrated arm).
2. **Remaining fused gap = View1 offset + missing fusion gain.** Of the 2.5 pp
   ACC gap to the paper's fused row, about 0.9 pp is the View1 offset and about
   1.6 pp is the paper's fusion gain over View1, which no completed run
   reproduces under raw Eq. (4) averaging.
3. **Raw averaging cannot add information while View2 is 4x smaller.** Scale
   mismatch (centered RMS 0.055 vs 0.237) makes the fused assignments equal
   View1's. Any fix that rescales View2 is outside the published method and is
   therefore a diagnostic, not the reproduction.
4. **Operational limits.** Kaggle grants 30 GPU hours per week and one run
   was already refused for exhausted quota. A seed-42 REUTERS arm costs
   roughly 3-6 minutes of training; budget arms accordingly.
5. **Frozen-View1 arm not yet run.** Commit `bf80610` defines the frozen
   DEKM_F View1 + true-bottleneck protocol
   (`mvdec_2025_frozen_dekm_f_view1_eq11_true_bottleneck_release_l2_seed42_v1`).
   The 0.515 View2 numbers above come from older joint-View1 snapshots kept
   under `artifacts/kaggle/`, so this arm is the next unrun baseline (S0).
6. **Fixed: loader/protocol mismatch in `bf80610`.** The protocol froze View1
   but `data.py` classified its ID as imported-joint and would have rejected a
   real run's artifact. The ID is now registered as frozen, with a regression
   test that fails on the old code (commit after `4c78012`).
7. **Repository state.** The protocols behind these runs live on
   `codex/frozen-view1-true-bottleneck`; they are merged into `dev` as of
   commit `cb2c775` (519 tests pass, 2 skipped).

## Success criteria (fixed before running)

A REUTERS candidate counts as a reproduction only if, over seeds 42-44 and
without labels or the paper's numbers entering any choice:

- View1 within 1.5 pp ACC and 1.5 pp NMI of Table 2;
- View2 within 3 pp ACC and 3 pp NMI of Table 2, with no View2 rescaling;
- fused at least View1 on both metrics and within 2 pp of Table 2;
- every choice recorded in the inference ledger as `paper_explicit`,
  `dekm_inherited`, `implementation_assumption`, or `ablation`.

Meeting these still would not prove equivalence to the authors' code.

## Screening arms (seed 42, one GPU run each, stop on first NaN/OOM/audit failure)

Every arm keeps the imported DEKM_F View1, full-batch refinement, Eq. 11 with
1/1/1/1 weights, release row-L2 input, and raw Eq. (4) averaging. Only the
named factor changes.

| Arm | Change | Question answered |
| --- | --- | --- |
| S0 | Run the committed frozen-View1 true-bottleneck protocol unchanged | Baseline for the current code line |
| S1 | Disable the assignment-change stop for the first N windows (fixed refinement budget, N preregistered) | Does a real refinement budget lift View2 without rescaling? |
| S2 | Joint training from scratch: View1 not imported, both views pretrained, same Eq. 11 refinement | Does co-training raise the fusion gain over View1? |
| S3 | Wider View2 stem (first layer not 64 for 2000-d input) | Is the 2000 to 64 compression what weakens View2? |
| S4 | Label-free deterministic feature split (columns by parity) as `ablation` | Does the "different attribute sets" prose reproduce View2? |

Lock at most two arms. Only then run seeds 42-44, then RCV1 and 20NEWS.
Eigen direction is dropped as a factor: smallest and largest gave
0.5139 / 0.3260 and 0.5151 / 0.3206 View2, a negligible difference.

## Workflow

1. Code and tests are committed and pushed to `origin/dev`.
2. The Kaggle notebook clones `dev`, checks out the pinned commit SHA, and
   aborts if `HEAD` differs.
3. The run is started on the Kaggle website with GPU and Internet enabled.
4. Output ZIP and its SHA-256 are returned for local `mvdec-audit-runs` and
   comparison with Table 2.

## Status of S1

S1 is implemented as `mvdec_2025_frozen_dekm_f_view1_eq11_true_bottleneck_min500_release_l2_seed42_v1`
(code commit `ea219bf`): identical to the frozen true-bottleneck protocol except
the assignment-change stop may not fire before update 500. The field defaults
to 0, so every existing protocol's contract and config hash is unchanged. The
Kaggle notebook is
`notebooks/mvdec_2025_frozen_view1_true_bottleneck_min500_reuters_seed42_kaggle.ipynb`;
it clones `dev`, checks out the pinned SHA, validates the DEKM_F checkpoint by
SHA-256, runs the tests, trains, audits, and fails if fewer than 500 updates ran.
Attach the dataset `datnguyends/mvdec-v1-true-bottleneck-largest-eigen-s42-input`
for the checkpoint only. S0 (the plain frozen arm) has no separate notebook; run
S1 first because S0's expected outcome (stop at update 10) is already known.

## Results (REUTERS seed 42, run 2026-10-21)

All arms: imported DEKM_F View1, true-bottleneck View2, full-batch Eq. 11 with
1/1/1/1 weights, release row-L2 input, raw Eq. (4) average. ACC / NMI.

| Arm | View1 | View2 | Fused | Audit |
| --- | ---: | ---: | ---: | --- |
| Paper Table 2 | 0.7642 / 0.5999 | 0.6501 / 0.5129 | 0.7813 / 0.6141 | - |
| Joint, stop at update 10 (older snapshot) | 0.7561 / 0.5941 | 0.5151 / 0.3206 | 0.7562 / 0.5942 | passed |
| **S1 frozen View1, floor 500** | 0.7557 / 0.5932 | **0.7404 / 0.5558** | 0.7563 / 0.5942 | passed, bit-identical on rerun |
| S1j joint View1, floor 500 | 0.7563 / 0.5944 | 0.4588 / 0.1999 | 0.7563 / 0.5944 | training complete; loader hash bug on the joint path |

Findings:

1. The 10-update stall was the cause of the weak View2. A 500-update floor
   lifts View2 from 0.515 to 0.740 ACC without any rescaling, and it exceeds
   Table 2 (0.650). The run is deterministic: a second Kaggle run at a later
   commit returned identical metrics.
2. Fused never exceeds View1 in any arm. With the stronger View2, every
   alternative fusion control (row-L2, z-score, Procrustes, centered-RMS,
   concatenation) is equal to or worse than the raw average (0.634 to 0.753 ACC
   against 0.756), so rescaling is not the missing ingredient.
3. Training View1 jointly (S1j) hurts View2 (0.459) and does not improve fused.
4. Remaining gap to Table 2: View1 -0.85 pp ACC (runtime offset, DEKM_F seeds
   42-44 span under 1 pp), fused -2.5 pp ACC. The paper's +1.7 pp fusion gain
   over View1 is not reproduced by any tested published-method variant.

Open items: the joint path still fails `_load_validated_mvdec_artifact` with a
config-hash mismatch (the joint transition proof is added after hashing); fix
before any further joint arm. Multi-seed S1 needs DEKM_F checkpoints for seeds
43-44 packaged like the seed-42 one.

## CPU diagnostics on the S1 artifact (label-aware, diagnostic only)

- K-Means seed variance is negligible: View1 and View2 give identical ACC for
  30 seeds; fused spans 0.702 to 0.756 at n_init=1 and is 0.7563 at the
  recorded n_init=100. Best-of-N K-Means cannot explain the paper's fused row.
- The views are complementary: ARI(View1, View2) = 0.812; both correct 58.7%,
  only View1 17.0%, only View2 15.3%, neither 9.1% (oracle best-of-two ACC
  0.909). Raw coordinate averaging does not exploit this because the two
  latents are independently trained and not axis-aligned; Procrustes and
  rescaling controls do not recover it either.
- Seeds 43-44 would need DEKM_F checkpoints with weights (only the seed-42
  weights were kept; the validator is pinned to that single checkpoint) and a
  gate generalization. That would test robustness of the View2 result but not
  the fused gap, so it is deferred.

Conclusion for REUTERS: View1 and View2 are reproduced to within or above the
paper's numbers under a published-method-only protocol with one preregistered
refinement floor. The fused row is not, and no tested variant of Eq. (4) fusion
closes it. Closing it needs information the paper does not give (how or whether
the two latents are aligned during joint training, and the run-selection rule).
