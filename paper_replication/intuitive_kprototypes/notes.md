# Reading Notes: Intuitive-K-prototypes

These notes capture the current understanding of the paper before any
algorithm code is written. Use `spec.md` for implementation details.

## Paper Goal

The paper proposes Intuitive-K-prototypes, a clustering algorithm for mixed
data with both numerical and categorical attributes.

It improves standard K-Prototypes in three places:

1. initial prototype selection;
2. categorical cluster-center representation;
3. attribute weighting.

## Why Standard K-Prototypes Is Not Enough

Standard K-Prototypes represents:

- numerical center by mean;
- categorical center by mode;
- categorical distance by equal/different matching.

The paper argues that categorical mode loses distribution information. For
example, if a value appears in two clusters with the same proportion, it is not
very discriminative even if it appears often. A mode-based center cannot express
that.

## Main Idea

For categorical attributes, the paper replaces the mode with an
intuitionistic distribution centroid.

For every categorical value in every cluster, this centroid stores:

- `mu`: how strongly the value belongs to the cluster;
- `nu`: how strongly the value does not belong to the cluster.

The useful intuition:

- high `mu`, low `nu`: the value characterizes the cluster;
- low `mu`, high `nu`: the value is evidence against the cluster;
- similar `mu` and `nu`: the value is ambiguous across clusters.

## Prototype Structure

Each cluster prototype has two parts:

- numerical part: mean of numerical attributes in the cluster;
- categorical part: intuitionistic distribution centroid for categorical
  attributes.

## Attribute Weights

The paper gives higher weight to attributes that are:

- less complex inside clusters;
- less similar between clusters.

This means an attribute is useful if it forms compact patterns within clusters
and separates clusters clearly.

For numerical attributes:

- intra-cluster complexity is based on coefficient of variation;
- inter-cluster similarity uses global/local cluster radii and distances.

For categorical attributes:

- intra-cluster complexity is based on intuitionistic fuzzy entropy;
- inter-cluster similarity compares intuitionistic distribution centroids.

## Categorical Distance

For an object value and a cluster's categorical centroid, the paper defines:

- membership distance `d1`;
- non-membership distance `d2`;
- composite distance controlled by `mu_param`.

This is different from simple `0/1` mismatch distance.

## Full Algorithm

At a high level:

1. choose dispersed initial prototypes;
2. assign objects using initial mixed distance;
3. compute numerical means and categorical IDCs;
4. compute numerical and categorical attribute weights;
5. compute weighted mixed distances;
6. reassign objects;
7. repeat until assignments stop changing or max iterations is reached.

## What To Verify First

Do not start with the full loop. Verify these blocks first:

1. Eq. (2)-(4): intuitionistic distribution centroid;
2. Eq. (24)-(26): categorical distance;
3. Eq. (17)-(22): categorical complexity, similarity, and weights;
4. Eq. (5)-(16): numerical complexity, similarity, and weights;
5. Eq. (27)-(29): total weighted mixed distance.

## Paper Example To Reproduce

The paper includes an artificial categorical dataset and reports:

- Table 2: intuitionistic distribution centroids;
- Table 6: categorical intra-cluster complexity;
- Table 7: categorical inter-cluster similarity;
- Table 8: categorical weights;
- Table 9: categorical distances for `x0 = (b, e, h)`.

These tables should be the first replication target because they validate the
categorical core of the algorithm.

The sandbox also checks the Iris numerical example from Table 3, Table 4, and
Table 5. Most values reproduce closely with sklearn Iris plus min-max
normalization; one Table 3 petal-width value differs slightly, so the test
documents that tolerance instead of changing Eq. (5).

## Known Unclear Or Risky Points

The following details need special care before implementation:

- Eq. (4) is ambiguous/inconsistent between the paper text/proof and the
  artificial example tables. The sandbox supports `non_membership="paper"` for
  the formal Eq. (4) reading and `non_membership="table"` for reproducing
  Table 2, Table 6, and Table 9. The estimator defaults to `"paper"` because
  the table-consistent rule can violate `mu + nu <= 1` on imbalanced real
  datasets.
- exact transcription of Eq. (11)-(12) for numerical inter-cluster similarity;
- exact transcription of Eq. (19) for categorical inter-cluster similarity;
- Table 8 appears to contain a typo for Attribute2 weights because the printed
  weights do not sum to the required totals;
- behavior when a cluster becomes empty;
- behavior when a weight composite measure is zero;
- fallback when Algorithm 1 cannot find enough prototypes under `d_lambda`;
- whether experiments should use one shared `gamma` or separate
  `gamma_num`/`gamma_cat`.

## Relation To mvdec_fpmax

For the main repository, the natural mapping is:

```text
X_num = h_fused_df
X_cat = FP-Max binary itemset features
```

The algorithm should be tested as an alternative final clustering step after
MvDEC representation learning and FP-Max feature extraction. It should not
replace the current K-Prototypes baseline until the sandbox implementation is
verified.
