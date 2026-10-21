Subject: Question on reproducing MvDEC (Table 2, REUTERS) - fused row and run selection

Dear Dr. Kassem,

I am reproducing "Multi-view Deep Embedded Clustering: Exploring a new dimension of air pollution" (EAAI 139, 2025) on REUTERS-10K. After extensive controlled experiments I can reproduce the View1 and View2 rows of Table 2, but not the fused row, and I would be grateful for two clarifications.

What I reproduced (seed 42, DEKM release data, full-batch refinement, Eq. 11 with unit weights, Eq. 4 averaging):

| Row | Table 2 ACC / NMI | My run ACC / NMI |
| --- | ---: | ---: |
| View1 (DEKM_F) | 0.7642 / 0.5999 | 0.7557 / 0.5932 |
| View2 | 0.6501 / 0.5129 | 0.7404 / 0.5558 |
| Fused | 0.7813 / 0.6141 | 0.7563 / 0.5942 |

What I found:

1. With an imported, converged DEKM_F View1, the 0.1% assignment-change stop fires after 10 updates and View2 stays at its pretrained state (ACC 0.515). Forcing at least 500 refinement updates lifts View2 to 0.740 without any rescaling.
2. The fused result then equals View1. The two views are complementary (15% of rows are right only in View2, 17% only in View1), yet no fusion I tried exceeds 0.7565 ACC: Eq. 4, the Fig. 1 full-output average, the Fig. 2 concatenation, row-L2 or z-score normalization, centered-RMS scaling, and Procrustes, least-squares, or CCA alignment.
3. Training View1 jointly with View2 lowers View2 (ACC 0.459) and does not raise the fused score.

Questions:

1. How are the two latent spaces aligned or combined during joint training so that the fused result exceeds View1? In particular, is View1 frozen, jointly trained, or retrained, and is any scale, alignment, or loss term used that Eq. 4 and Eq. 11 do not show?
2. How were the "best performance" values in Table 2 selected (number of runs, seeds, checkpoint or stopping rule, and whether labels were used)?

If code, a configuration file, or the run logs for the REUTERS row are available, they would answer both questions at once. I can share my code, run manifests, and audit reports.

Thank you for your time.

Best regards,
[Your name]
