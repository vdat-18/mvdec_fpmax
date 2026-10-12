# MvDEC REUTERS protocol results

## Scope

These runs compare the named MvDEC Equation 11 ambiguity protocols defined in
`src/representation_learning/MVDEC_dense.py`. They are reproduction experiments,
not claims of official MvDEC results. The immutable literal baseline remains
`mvdec_2025_equation11_literal_v1`.

All GPU runs used the official DEKM REUTERS arrays (10,000 samples), TensorFlow
2.20.0, and Kaggle T4 x2. Labels were used only for final ACC/NMI reporting.

## Seed-42 ambiguity matrix

| ID | Protocol | Eigen direction | Greedy target | Refinement batch | Fused ACC | Fused NMI | Silhouette | Steps |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| B0 | `mvdec_2025_equation11_literal_v1` | largest | frozen full vector | 256 | 72.21% | 53.0693% | not recorded | not recorded |
| B1 | `mvdec_2025_equation11_smallest_frozen_v1` | smallest | frozen full vector | 256 | 71.48% | 52.1383% | 0.517505 | 60 |
| B2 | `mvdec_2025_equation11_largest_selected_only_v1` | largest | selected dimension only | 256 | 72.53% | 54.0297% | 0.513962 | 70 |
| B3 | `mvdec_2025_equation11_smallest_selected_only_v1` | smallest | selected dimension only | 256 | 72.39% | 53.7273% | 0.476200 | 90 |
| F0 | `mvdec_2025_equation11_largest_frozen_full_batch_v1` | largest | frozen full vector | 10,000 | 71.54% | 51.9219% | 0.588083 | 30 |
| P0 | `mvdec_2025_paper_resolved_full_batch_v1` | smallest | selected dimension only | 10,000 | 71.94% | 52.8645% | 0.654397 | 40 |

B2 has the best seed-42 supervised metrics. P0 has the strongest seed-42
silhouette and is the best-supported paper-resolved interpretation because it
combines MvDEC's smallest-eigen prose, the selected-coordinate reading of Eq.
10, and the full-batch `DEKM_F` evidence in Table 2.

## Selected paper-resolved protocol: seeds 42-44

Kaggle notebook version 5, script version `357790756`, completed successfully in
14m 15s. Every run used one real 10,000-row gradient batch per refinement epoch.

| Seed | Fused ACC | Fused NMI | View1 ACC | View1 NMI | View2 ACC | View2 NMI | Silhouette | Steps | Cluster sizes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 42 | 71.94% | 52.8645% | 71.93% | 52.8496% | 46.31% | 18.1907% | 0.654397 | 40 | 2270, 2892, 3638, 1200 |
| 43 | 72.22% | 51.2758% | 72.22% | 51.2758% | 39.93% | 10.2839% | 0.657328 | 40 | 3621, 2589, 1646, 2144 |
| 44 | 69.54% | 49.9599% | 69.56% | 49.9787% | 35.79% | 12.2915% | 0.688525 | 30 | 3277, 2385, 2956, 1382 |
| Mean | 71.2333% | 51.3667% | 71.2367% | 51.3681% | 40.6767% | 13.5887% | 0.666750 | - | - |
| Sample std | 1.4731 pp | 1.4544 pp | 1.4593 pp | 1.4377 pp | 5.2996 pp | 4.1099 pp | 0.018914 | - | - |

All three runs converged by assignment stability. Their shared config hash is:

```text
928d4c50364bea91b37e4b9c5f811be25c357056e49c24f09222789a265f315b
```

Kaggle outputs:

```text
mvdec-p0-paper-resolved-reuters-seeds42-44-full.zip      327.8 MB
mvdec-p0-paper-resolved-reuters-seeds42-44-summary.zip   26.96 kB
```

## Comparison with the release-faithful DEKM baseline

| Method | ACC mean +/- sample std | NMI mean +/- sample std |
| --- | ---: | ---: |
| DEKM release-faithful, seeds 42-44 | 76.0433% +/- 0.1457 pp | 59.1668% +/- 0.5602 pp |
| MvDEC P0 paper-resolved, seeds 42-44 | 71.2333% +/- 1.4731 pp | 51.3667% +/- 1.4544 pp |

P0 trails the verified DEKM baseline by 4.81 percentage points ACC and 7.80
percentage points NMI. The result does not support a claim that this recoverable
MvDEC interpretation improves on DEKM for REUTERS.

## Decision

Use P0 as the primary **paper-resolved reproduction protocol**, while retaining
B0 as the immutable **literal-equation baseline** and B2 as the strongest
mini-batch ambiguity ablation. Do not rename P0 to "official" or "literal".

The remaining performance gap is not evidence of a simple implementation bug.
The MvDEC paper still leaves the feature-view split, Eq. 4 versus Figure 2
fusion, View2 latent placement, seed policy, and model-selection procedure
unidentified. Exact Table 2 reproduction therefore remains unclaimable without
author code/data or a separately named architecture protocol.
