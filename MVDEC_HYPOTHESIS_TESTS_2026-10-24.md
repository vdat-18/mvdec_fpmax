# MvDEC Table 2: hypothesis ledger and tests (2026-10-24)

Scope: the author has not answered (`AUTHOR_REQUEST_2026-10-21.md`,
`AUTHOR_FOLLOWUP_2026-10-22.md`), so the remaining Table 2 gaps (REUTERS Fused
0.7813/0.6141 and View2 0.6501/0.5129) are attacked as a list of falsifiable
hypotheses. Every result here is `controlled_diagnostic_not_exact_paper_truth`.
Label-guided selections are reported only to locate the paper numbers inside the
run-to-run distribution; they are never promoted to the reproduction protocol
(`MVDEC_REPRODUCTION_BOUNDARY_AND_NEXT_STEP_2026-10-19.md`). Earlier result files
are left unchanged.

## Evidence classes used below

- `paper_explicit`: stated in MvDEC 2025.
- `dekm_inherited`: from DEKM 2021 paper or its release code.
- `inference`: technical inference from the two papers and the code.
- `assumption`: chosen because the paper is silent.

## Hypotheses, ranked by prior plausibility and test cost

| ID | Hypothesis | Basis | Test | Status |
| --- | --- | --- | --- | --- |
| H1 | "Best performance" = best of N seeds chosen with labels | `paper_explicit` (Sec. 4.2 "best performance metrics") | 10 seeds (done), 40 seeds 52-91 | rejected (fused NMI max 0.5765 over 50 seeds) |
| H2 | "Best performance" = best intermediate checkpoint chosen with labels | `dekm_inherited`: release `DEKM.py` prints ACC/NMI at every K-Means refresh | per-checkpoint label trajectory, 40 seeds | rejected (0 of 2046 checkpoints) |
| H3 | Best ACC and best NMI are taken from different runs | `inference` from "best performance metrics" | max over seeds per metric | rejected on REUTERS (NMI max 0.5751) |
| H4 | NMI uses another normalisation (geometric/min/max) | `assumption` (older sklearn default was geometric) | recompute on saved latents | rejected |
| H5 | "ACC" is purity | `assumption` | recompute on saved latents | weak, inconsistent across datasets |
| H6 | Fusion is Fig. 2 concatenation, not Eq. 4 average | `paper_explicit_conflicting_statement` | K-Means on saved `[h1, h2]` | partial support (from-scratch runs only) |
| H7 | K-Means restart picked with labels | `inference` | 30 restarts, n_init=1 | rejected |
| H8 | View2 row is a standalone single-view model ("considering the individual views independently", "single-view models") | `paper_explicit` wording, Sec. 4.2 | independent View2 DEKM | earlier run 0.5277/0.343 under the 10-update stall; needs a re-run with the true bottleneck |
| H9 | Paper-text input normalisation `(1/d)||x||^2 = 1` | `dekm_inherited` (DEKM paper text) | MvDEC S2 under that normalisation | not run (DEKM_F evidence says it lowers ACC) |

Already closed before this ledger (see `MVDEC_REUTERS_REPRODUCTION_PLAN_2026-10-21.md`):
eigen direction, View2 stem width, even-column feature split, post-hoc linear
alignment (Procrustes, least squares, CCA), training-time alignment, row-L2 and
z-score fusion, joint vs frozen View1, the 10-update stall.

## Test 1: CPU checks on saved S2 latents (H3-H7)

Script: `scripts/mvdec_hypothesis_latent_checks.py`. Input: the `artifact.pkl` of
every finished S2 run (`mvdec_2025_scratch_both_views_true_bottleneck_min500_release_l2_v1`):
REUTERS seeds 42-51, RCV1 seeds 42-51, 20NEWS seeds 42-47. K-Means uses
`n_init=100`, `random_state=seed`, as in the runs. Full numbers per seed:
`docs/audit_evidence/mvdec_hypotheses_2026-10-24/latent_checks_{reuters,rcv1,20news}.json`.

Command (run with `OMP_NUM_THREADS=1`; without it sklearn threads contend with
TensorFlow and K-Means stalls):

```bash
OMP_NUM_THREADS=1 python scripts/mvdec_hypothesis_latent_checks.py <root with seed_*/artifact.pkl> \
  --output docs/audit_evidence/mvdec_hypotheses_2026-10-24/latent_checks_<ds>.json
```

### REUTERS (paper fused 0.7813 / 0.6141, View2 0.6501 / 0.5129)

Mean / max over 10 seeds:

| Representation | ACC | Purity | NMI arith. | NMI geom. | NMI min | NMI max |
| --- | --- | --- | --- | --- | --- | --- |
| View1 | 0.7198 / 0.7805 | 0.7927 / 0.8374 | 0.5102 / 0.5701 | 0.5105 / 0.5702 | 0.5273 / 0.5805 | 0.4943 / 0.5601 |
| View2 | 0.5155 / 0.5858 | 0.5647 / 0.6151 | 0.2897 / 0.3714 | 0.2914 / 0.3746 | 0.3230 / 0.4276 | 0.2632 / 0.3291 |
| Fused (Eq. 4) | 0.7086 / 0.7842 | 0.7915 / 0.8418 | 0.5099 / 0.5751 | 0.5102 / 0.5751 | 0.5273 / 0.5847 | 0.4937 / 0.5658 |
| Concat (Fig. 2) | 0.7303 / 0.8164 | 0.8018 / 0.8405 | 0.5268 / 0.5753 | 0.5271 / 0.5755 | 0.5425 / 0.5888 | 0.5121 / 0.5624 |

Best label-guided K-Means restart on the fused latent: seed 48, 0.7806 / 0.5914.

### RCV1 (paper View1 0.6212/0.3598, View2 0.6235/0.3659, fused 0.7193/0.4885)

| Representation | ACC mean / max | NMI mean / max |
| --- | --- | --- |
| View1 | 0.6793 / 0.7630 | 0.4459 / 0.5293 |
| View2 | 0.5218 / 0.6605 | 0.3117 / 0.3979 |
| Fused (Eq. 4) | 0.6721 / 0.7631 | 0.4375 / 0.5286 |
| Concat (Fig. 2) | 0.6531 / 0.7107 | 0.4315 / 0.5035 |

Purity equals ACC on RCV1 (one cluster per class in every run). Two single runs land
near paper rows: View2 seed 44 0.6605/0.3968 and seed 43 0.6010/0.3979 (paper
0.6235/0.3659); concat seed 48 0.7107/0.4827 (paper fused 0.7193/0.4885).

### 20NEWS, 6 seeds (paper View1 0.4120/0.4376, View2 0.4086/0.4032, fused 0.4163/0.4543)

| Representation | ACC mean / max | NMI mean / max |
| --- | --- | --- |
| View1 | 0.4386 / 0.4530 | 0.4147 / 0.4253 |
| View2 | 0.2755 / 0.2937 | 0.2733 / 0.2939 |
| Fused (Eq. 4) | 0.3548 / 0.3958 | 0.3011 / 0.3393 |
| Concat (Fig. 2) | 0.4227 / 0.4311 | 0.4066 / 0.4160 |

### Verdicts

- **H4 rejected.** Geometric NMI equals arithmetic to 3 decimals; the most generous
  (min) normalisation adds 1-2 pp and the REUTERS fused maximum is 0.5847 < 0.6141.
- **H7 rejected.** Label-picked K-Means restarts move fused at most to 0.7806 / 0.5914.
- **H3 rejected for REUTERS.** Best ACC (0.7842, seed 46) and best NMI (0.5751, seed 48)
  from different runs still miss NMI by 3.9 pp.
- **H5 weak.** Purity lifts REUTERS (fused purity mean 0.79 ~ paper 0.78) but equals ACC
  on RCV1, where the paper fused row is still 4.7 pp above our mean; it also leaves NMI
  untouched. Not adopted.
- **H6 partial.** With both views trained from scratch, concatenation beats Eq. 4 on
  REUTERS (+2.2 pp ACC, +1.7 pp NMI mean) and strongly on 20NEWS (+6.8 pp ACC,
  +10.6 pp NMI), and is worse on RCV1 (-1.9 pp ACC). With the frozen DEKM_F View1 (S1)
  concat was slightly worse than Eq. 4 (0.7534 vs 0.7563). Fig. 2 concat cannot be
  ruled out as the authors' fusion; it does not close the REUTERS NMI gap on its own.
- **REUTERS View2 remains unreached:** no from-scratch seed exceeds 0.586 / 0.371,
  while the frozen-View1 S1 arm gives 0.740 / 0.556; the paper's 0.6501 / 0.5129 sits
  between the two regimes. On RCV1 the paper View2 row lies inside our distribution.

## Test 2: label trajectory per checkpoint (H1 + H2)

Code: `src/pipeline/mvdec_trajectory.py`, hooked into the refinement loop of
`src/representation_learning/MVDEC_dense.py`. Enabled only when
`MVDEC_LABEL_TRAJECTORY_LOG=<path>` is set; it writes one JSON line per K-Means
refresh with fused ACC/NMI (refinement assignment) and View1/View2 ACC/NMI
(K-Means `n_init=10` on each latent). It is a pure observer: no feedback into
training, no change to the global NumPy/TensorFlow random state (unit test
`tests/test_mvdec_trajectory.py`), and no change to protocol contracts or config
hashes.

CPU run (seed 42, 4-core container, 57 min, not bit-identical to the Kaggle GPU run
which gave 0.7025/0.5168): final fused 0.7146/0.5104, View1 0.7140/0.5074, View2
0.5020/0.2881. Trajectory over 51 refreshes (steps 0-500):

| Step | Fused ACC / NMI | View2 ACC / NMI |
| ---: | --- | --- |
| 0 | 0.7109 / 0.4734 | 0.5284 / 0.3678 |
| 10 | **0.7165** / 0.5106 | **0.5402 / 0.3687** |
| 50 | 0.7150 / 0.5106 | 0.5168 / 0.2934 |
| 200 | 0.7148 / 0.5107 | 0.4949 / 0.2641 |
| 500 | 0.7146 / 0.5104 | 0.5022 / 0.2883 |

On this seed the fused partition is fixed after about 10 updates: the best
checkpoint gains only +0.2 pp ACC / +0.05 pp NMI over the final state, while View2
is best right after pretraining and drifts down during refinement. Early evidence
against H2 as the main explanation; the 40-seed GPU run decides.

Kaggle notebook: `notebooks/mvdec_2025_s2_trajectory_reuters_seeds52_91_kaggle.ipynb`
(pins training commit `231f72914c3ed5bd738443069437d8be698bc4a4`, branch
`claude/mvdec-hypotheses-2026-10-24`). It runs seeds 52-91 and reports, per seed, the
best fused checkpoint and the maximum View2 ACC/NMI along the trajectory. The
original 40-seed notebook's summary cell asserted 10 manifests and would have failed
after training; the new notebook prints the count instead.

Decision rule fixed before the run: H1/H2 are "supported" only if one run (H1) or one
checkpoint of one run (H2) reaches fused ACC >= 0.7713 **and** NMI >= 0.6041 (within
1 pp of Table 2) together. Reaching only ACC is the situation already seen at 10 seeds.

### Result (Kaggle `datnguyends/mvdec-s2-trajectory-40-reuters`, downloaded 2026-10-26)

All 40 manifests (seeds 52-91) are complete; `mvdec_summary.csv` reports 40 completed,
0 failed, and the results zip matches its SHA-256. Kaggle marked the kernel ERROR
after training, as in the earlier multi-seed runs; the log had not been retrieved
when this was written, so the cause is not recorded. 2046 trajectory checkpoints
(51 per seed, 57 for seed 59). Raw files stay on the local machine under
`artifacts/kaggle/mvdec-s2traj40-reuters-run/output/` (git-ignored); the per-seed
tables are committed as
`docs/audit_evidence/mvdec_hypotheses_2026-10-24/s2traj40_reuters_final_metrics.csv`
and `.../s2traj40_reuters_trajectory_summary.csv`.

Final state, 40 seeds (mean ± std, max):

| Row | Paper | ACC | NMI |
| --- | --- | --- | --- |
| View1 | 0.7642 / 0.5999 | 0.7233 ± 0.0369, max 0.8184 | 0.5176 ± 0.0339, max 0.5762 |
| View2 | 0.6501 / 0.5129 | 0.5186 ± 0.0808, max 0.7301 | 0.2847 ± 0.0800, max 0.4597 |
| Fused | 0.7813 / 0.6141 | 0.7223 ± 0.0366, max 0.8177 | 0.5155 ± 0.0368, max 0.5765 |

Along the trajectories: highest fused NMI at any checkpoint of any seed 0.5767 (seed
62); highest fused ACC 0.8181 (seed 79, NMI 0.5659); highest View2 NMI 0.4657.
Seeds whose best fused ACC passes 0.7713: 79 (0.8181/0.5659), 88 (0.7940/0.5389),
57 (0.7783/0.5331). The best checkpoint is on average within a few tenths of a point
of the final state, matching the CPU seed-42 trajectory. Fused ACC exceeds View1 ACC
in 20 of 40 seeds: Eq. 4 fusion adds nothing systematic in this regime.

**Rule: 0 of 40 seeds and 0 of 2046 checkpoints pass.**

Verdicts:

- **H1 rejected** for the full row. Over 50 from-scratch seeds (42-91) the paper's fused
  ACC is reachable by label-guided best-of-N (seeds 46, 57, 79, 88), the fused NMI never
  is (max 0.5765 final, 0.5914 with a label-picked K-Means restart on seed 48).
- **H2 rejected.** Label-guided checkpoint selection changes fused metrics by tenths of
  a point; the partition is essentially fixed after the first refinement windows.
- **REUTERS View2 row not reachable in the from-scratch regime:** View2 ACC 0.65 occurs
  (seeds 53, 60, 89 near 0.72-0.73) but View2 NMI never exceeds 0.466 (paper 0.5129).
- Combined with Test 1, no reporting or selection choice applied to the from-scratch
  joint model (S2) reproduces REUTERS Table 2. The paper's fused NMI 0.6141 is above the
  DEKM_F View1 NMI it reports (0.5999), so the published row needs a View1 at least as
  strong as DEKM_F *and* a real fusion gain on NMI. S1 (frozen DEKM_F View1) supplies
  the first but no tested fusion supplies the second (ceiling 0.5942 NMI).

## Overall assessment (2026-10-26)

| Item | Status |
| --- | --- |
| DEKM / DEKM_F baseline | reproduced (frozen, `DEKM_CANONICAL_BASELINE.md`) |
| MvDEC architecture, Eq. 3-11 | implemented; several details are `inference`/`assumption` (see boundary doc) |
| REUTERS View1 | within 1 pp via imported DEKM_F (S1) |
| REUTERS View2 | not matched: S1 overshoots (0.740/0.556), S2 undershoots (≤0.586 mean regime, NMI ≤0.466) |
| REUTERS Fused | not reproduced: no protocol, fusion, metric variant, seed or checkpoint selection reaches 0.7813/0.6141 together |
| RCV1 rows | inside the S2 seed distribution, so not discriminative |
| 20NEWS rows | Eq. 4 fusion lowers ACC; Fig. 2 concat comes close on ACC (0.42) but not NMI (0.41 vs 0.454) |

Remaining explanations that the available evidence cannot test: a View1/View2 feature
split that the paper does not disclose, an unpublished alignment or fusion step, a
different REUTERS preprocessing than the DEKM release files, or a reporting error in
Table 2. These need the authors' code or data.

## Open items

- H8: rerun independent single-view View2 with the true bottleneck and DEKM's own stop
  rule; it needs the DEKM_F View1 checkpoint dataset on Kaggle for the fused row.
- H6 follow-up: an arm whose refinement uses the Fig. 2 concatenated embedding.
- RCV1 trajectory run (10 seeds) once the REUTERS trajectory result is known.
