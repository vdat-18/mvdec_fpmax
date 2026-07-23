# Intuitive-K-prototypes Implementation Spec

This document rewrites the paper into an implementation-oriented
specification. It intentionally contains no algorithm code.

Paper:

Wang, H., & Mi, J. (2025). "Intuitive-K-prototypes: A mixed data clustering
algorithm with intuitionistic distribution centroid." Pattern Recognition,
158, 111062. https://doi.org/10.1016/j.patcog.2024.111062

## 1. Objective

Replicate Intuitive-K-prototypes as described in the paper before integrating
it into the main `mvdec_fpmax` pipeline.

The algorithm clusters mixed-type data with:

- numerical attributes represented by cluster means;
- categorical attributes represented by intuitionistic distribution centroids;
- attribute weights updated from intra-cluster complexity and inter-cluster
  similarity;
- a composite categorical distance based on membership and non-membership.

## 2. Notation

Dataset:

- `X = {x_1, ..., x_n}`
- `n`: number of objects
- `m`: total number of attributes
- `m_n`: number of numerical attributes
- `m_c`: number of categorical attributes
- `k`: number of clusters
- `C_l`: cluster `l`, for `l = 1..k`
- `|C_l|`: number of objects in cluster `l`
- `A_j`: attribute `j`
- `A^n_j`: numerical attribute `j`
- `A^c_j`: categorical attribute `j`
- `Dom(A^c_j) = {a_1j, ..., a_tj}`: values of categorical attribute `j`

Implementation shapes:

- `X_num`: `(n, m_n)` numeric matrix
- `X_cat`: `(n, m_c)` categorical matrix
- `labels`: `(n,)` integer cluster labels in `0..k-1`
- `numeric_centroids`: `(k, m_n)`
- `idc`: nested structure indexed by cluster, categorical attribute, value
- `weights_num`: `(m_n,)`
- `weights_cat`: `(m_c,)`

Use zero-based indexes in code, but keep paper equations documented with
one-based symbols.

## 3. Algorithm 1: Initial Prototype Selection

Purpose:

Choose initial prototypes that are more evenly dispersed than random
initialization.

Inputs:

- mixed dataset `X`
- number of clusters `k`
- distance parameter `lambda`, where `0 < lambda <= 1`
- numerical and categorical attribute indexes

Initial mixed distance, Eq. (1):

```text
d(x_i, x_l)
  = sum over numerical attributes |x_ij - x_lj|
    + sum over categorical attributes delta(x_ij, x_lj)

delta(a, b) = 1 if a != b, else 0
```

Steps:

1. Randomly choose an object `x_0`.
2. Find `x_1`, the object farthest from `x_0` under Eq. (1).
3. Find `x_2`, the object farthest from `x_1`.
4. Define `d_approx = d(x_1, x_2)`.
5. Define `d_lambda = lambda * d_approx / k`.
6. Start prototype set `Q = {x_1}`.
7. Add objects whose distances to every existing prototype in `Q` are at
   least `d_lambda` until `|Q| = k`.

Output:

- initial prototype object indexes or prototype rows.

Checks:

- output has exactly `k` prototypes;
- no duplicate prototype indexes;
- selected prototypes should be pairwise at least approximately dispersed.

Open implementation decision:

- The paper does not specify a fallback when no candidate satisfies
  `d_lambda`. Sandbox implementation should record this as an error first.
  A production integration may later lower `lambda` or use the farthest
  remaining point as fallback.

## 4. Definition 1: Intuitionistic Distribution Centroid

Purpose:

Represent a categorical cluster centroid using both membership and
non-membership of each categorical value.

For categorical attribute `A^c_j` and cluster `C_l`, define:

```text
q^c_lj = {
  <a_1j, mu_lj^1, nu_lj^1>,
  <a_2j, mu_lj^2, nu_lj^2>,
  ...,
  <a_tj, mu_lj^t, nu_lj^t>
}
```

Where:

- `[a_rj]` is the set of all objects whose categorical attribute `j` equals
  value `a_rj`;
- `[a_rj] intersect C_l` is the set of objects in cluster `l` with value
  `a_rj`.

Membership, Eq. (3):

```text
mu_lj^r = |[a_rj] intersect C_l|^2 / (|[a_rj]| * |C_l|)
```

Non-membership, Eq. (4):

```text
nu_lj^r = sum over clusters s != l of
          (|[a_rj] intersect C_s| / |[a_rj]|)^2
```

Implementation note:

The printed proof and Eq. (4) are ambiguous for values whose frequency differs
strongly across clusters. To reproduce the paper's own Table 2, Table 6, and
Table 9, the sandbox implementation supports a table-consistent rule:

```text
if |[a_rj] intersect C_l| == 0:
    nu_lj^r = 1
else:
    nu_lj^r = sum over clusters s != l of
              (|[a_rj] intersect C_s| / |C_s|)^2
```

This remains documented and tested because it differs from a literal reading
of the PDF text extraction.

The estimator default is the formal Eq. (4) reading:

```text
nu_lj^r = sum over clusters s != l of
          (|[a_rj] intersect C_s| / |[a_rj]|)^2
```

Reason: the table-consistent rule can violate the intuitionistic fuzzy-set
constraint `mu + nu <= 1` on imbalanced real datasets. Therefore:

- use `non_membership="paper"` for real UCI and mvdec_fpmax experiments;
- use `non_membership="table"` only for reproducing the paper's artificial
  categorical tables.

Output:

- `idc[l][j][value] = {mu, nu}`.

Checks:

- `0 <= mu <= 1`;
- `0 <= nu <= 1`;
- `mu + nu <= 1`;
- all categorical values in the global domain of attribute `j` are present
  for every cluster centroid.

Edge cases:

- if `|C_l| == 0`, the cluster is empty; the paper does not define this.
  Sandbox should fail loudly or mark the iteration invalid.
- if `|[a_rj]| == 0`, that value should not be in the observed domain.

Interpretation:

- high `mu`, low `nu`: value strongly characterizes the cluster;
- low `mu`, high `nu`: value is evidence against the cluster;
- similar `mu` and `nu`: value is ambiguous across clusters.

## 5. Numerical Attribute Weights

The paper uses intra-cluster complexity and inter-cluster similarity to
determine numerical attribute priority. Higher complexity or similarity means
the attribute is less useful for distinguishing clusters, so its weight should
be lower.

### 5.1 Intra-Cluster Complexity For Numerical Attributes

For numerical attribute `A^n_j` in cluster `C_l`, compute the coefficient of
variation, Eq. (5):

```text
xi_j^n(C_l) = 0                    if mean_j(C_l) == 0
            = std_j(C_l) / mean_j(C_l) otherwise
```

Normalize across clusters:

```text
xi_j^*(C_l) = xi_j^n(C_l) / max_i xi_j^n(C_i)
```

Overall intra-cluster complexity, Eq. (6):

```text
Xi^n(A^n_j) = sum_i alpha_i * xi_j^*(C_i)
alpha_i = |C_i| / n
```

Checks:

- expected range is `[0, 1]` after normalization;
- attributes with more scattered values inside clusters should have larger
  complexity.

Open implementation decision:

- If all cluster coefficients are zero, the max denominator is zero. Treat the
  normalized coefficients as zero and document the case.

### 5.2 Inter-Cluster Similarity For Numerical Attributes

For each numerical attribute and cluster, the paper defines:

- global cluster attribute radius, Eq. (7);
- local cluster attribute radius using points within the global radius,
  Eq. (8);
- asymmetric global and local distances between clusters, Eq. (9)-(10);
- pairwise similarity through Eq. (11)-(12);
- overall similarity through weighted cluster-pair averaging, Eq. (14).

Implementation-level meaning:

1. Compute the cluster mean for attribute `A^n_j`.
2. Compute each object's absolute distance to that mean.
3. The global radius is the average of those distances.
4. The local radius is the average distance among "good" objects whose
   distance is no larger than the global radius.
5. Compute directed distances from objects in one cluster to the other
   cluster's center, both globally and locally.
6. Combine the global and local terms into pair similarity.
7. Average pair similarities using pair-size weights.

Checks:

- expected range is `[0, 1]`;
- attributes whose clusters overlap strongly should have larger similarity;
- attributes whose clusters are well separated should have lower similarity.

Open implementation decision:

- The PDF extraction around Eq. (11)-(12) is dense. This block should be
  implemented only after manually checking the rendered PDF equations.

### 5.3 Numerical Weight Formula

Composite numerical measure, Eq. (15):

```text
Phi^n(A^n_j) = gamma * Xi^n(A^n_j)
             + (1 - gamma) * S^n(A^n_j)
```

where:

- `gamma in [0, 1]`;
- `Xi^n` is intra-cluster complexity;
- `S^n` is inter-cluster similarity.

Weight, Eq. (16):

```text
weight_num_j is inversely related to Phi^n(A^n_j)
```

Implementation form:

```text
score_j = Phi_j^(-1 / (beta - 1))
weight_j = score_j / sum(score)
```

where `beta > 1`.

The paper rescales numerical weights by the number of numerical attributes:

```text
weight_num_j_scaled = weight_num_j * m_n
```

Checks:

- unscaled numerical weights sum to `1`;
- scaled numerical weights sum to `m_n`;
- lower `Phi` should produce higher weight.

The implementation follows the printed piecewise case exactly: `Phi == 0`
receives weight `0`, while positive `Phi` values are normalized over the
nonzero attributes.

## 6. Categorical Attribute Weights

Categorical weights use intuitionistic distribution centroids.

### 6.1 Intra-Cluster Complexity For Categorical Attributes

The paper uses intuitionistic fuzzy entropy, Eq. (17):

```text
Entropy(q^c_lj)
  = sum_r min(mu_lj^r, nu_lj^r)
    / sum_r max(mu_lj^r, nu_lj^r)
```

Overall categorical intra-cluster complexity, Eq. (18):

```text
E^c(A^c_j) = sum_i alpha_i * Entropy(q^c_ij)
alpha_i = |C_i| / n
```

Checks:

- expected range is `[0, 1]`;
- more ambiguous categorical distributions should have higher entropy.

Edge cases:

- if the denominator is zero, the paper does not define behavior. Sandbox
  should fail loudly until an explicit rule is chosen.

### 6.2 Inter-Cluster Similarity For Categorical Attributes

For two clusters and categorical attribute `A^c_j`, the paper computes
similarity between intuitionistic fuzzy sets, Eq. (19), then aggregates across
all cluster pairs, Eq. (20).

Implementation-level meaning:

1. For every cluster pair `(C_p, C_q)`, compare their IDC entries for the same
   categorical attribute.
2. Use both membership and non-membership differences.
3. Convert the difference into a similarity score in `[0, 1]`.
4. Average pair similarities using pair-size weights.

Checks:

- expected range is `[0, 1]`;
- if two clusters have nearly identical categorical value distributions, the
  similarity should be high;
- if the categorical value distributions are distinct, the similarity should
  be low.

Open implementation decision:

- Eq. (19) should be transcribed carefully from the rendered PDF before
  implementation because the PDF text extraction can mangle superscripts and
  subscripts.

### 6.3 Categorical Weight Formula

Composite categorical measure, Eq. (21):

```text
Phi^c(A^c_j) = gamma * E^c(A^c_j)
             + (1 - gamma) * S^c(A^c_j)
```

Weight, Eq. (22):

```text
score_j = Phi_j^(-1 / (beta - 1))
weight_j = score_j / sum(score)
```

The paper rescales categorical weights by the number of categorical
attributes:

```text
weight_cat_j_scaled = weight_cat_j * m_c
```

Checks:

- unscaled categorical weights sum to `1`;
- scaled categorical weights sum to `m_c`;
- lower `Phi` should produce higher weight.

## 7. Unified Attribute Weights

Eq. (23) combines the scaled numerical and categorical weights:

```text
W = {
  weight_num_1_scaled, ..., weight_num_mn_scaled,
  weight_cat_1_scaled, ..., weight_cat_mc_scaled
}
```

Checks:

- numerical block sums to `m_n`;
- categorical block sums to `m_c`;
- full vector sums to `m`.

## 8. Definition 8-9: Categorical Distance

Purpose:

Compute distance between an object's categorical value and a cluster's
intuitionistic distribution centroid.

For object value `x_ij^c` and cluster IDC `q^c_lj`, Eq. (24)-(25):

```text
d1(x_ij^c, q^c_lj) = sum over values a_rj != x_ij^c of mu_lj^r
d2(x_ij^c, q^c_lj) = nu_lj^r for the matching value a_rj == x_ij^c
```

Composite categorical distance, Eq. (26):

```text
d(x_ij^c, q^c_lj) = mu_param * d1(x_ij^c, q^c_lj)
                  + (1 - mu_param) * d2(x_ij^c, q^c_lj)
```

Use `mu_param` to avoid confusing this paper parameter with membership
`mu_lj^r`.

Checks:

- distance should be low when the object value strongly belongs to the
  cluster;
- distance should be high when the object value has high non-membership for
  the cluster;
- reproduce Table 9 from the paper's artificial categorical example.

## 9. Full Mixed Distance And Cost

Cost function, Eq. (27):

```text
Cost(U, Q, W) = sum_l sum_i u_il * Delta(x_i, Q_l)
```

where `u_il = 1` if object `i` belongs to cluster `l`, else `0`.

Mixed distance, Eq. (28):

```text
Delta(x_i, Q_l) = Delta_num(x_i, Q_l) + Delta_cat(x_i, Q_l)
```

Numerical and categorical parts, Eq. (29):

```text
Delta_num(x_i, Q_l)
  = sum_j weight_num_j_scaled * |x_ij^n - centroid_lj^n|

Delta_cat(x_i, Q_l)
  = sum_j weight_cat_j_scaled * d(x_ij^c, q^c_lj)
```

Checks:

- distance matrix shape is `(n, k)`;
- every object is assigned to the nearest cluster;
- the objective should not increase unexpectedly after a reassignment and
  update step, but the paper does not prove monotonicity.

## 10. Algorithm 2: Full Intuitive-K-prototypes Loop

Inputs:

- mixed dataset `X`
- number of clusters `k`
- distance parameter `mu_param` for Eq. (26)
- balancing parameter `gamma`
- weight index `beta`
- maximum number of iterations `max_iter`

Paper flow:

1. Initialize membership matrix `U` to zeros.
2. Select initial prototypes.
3. Assign every object to its nearest prototype using Eq. (1).
4. Update `U`.
5. Update attribute weights using Eq. (5)-(23).
6. Update categorical IDCs using Eq. (2)-(4).
7. Update numerical centroids using cluster means.
8. Compute categorical and numerical distances using Eq. (24)-(29).
9. Reassign objects to the nearest cluster.
10. Repeat until `U` is unchanged or `max_iter` is reached.

Outputs:

- final labels;
- final numerical centroids;
- final IDCs;
- final attribute weights;
- distance history;
- number of iterations;
- convergence flag.

Checks:

- no empty cluster unless an explicit policy is used;
- all labels are in `0..k-1`;
- all weight vectors are finite;
- all distances are finite;
- deterministic output when `random_state` is fixed.

Open implementation decisions:

- empty-cluster policy;
- fallback for Algorithm 1 when `lambda` is too strict;
- exact behavior for zero composite measures in weight formulas;
- whether to use one `gamma` for both numerical and categorical weights or
  separate `gamma_num` and `gamma_cat` for experiments.

## 11. Suggested Parameters For First Replication

Use conservative defaults for the first sandbox run:

```text
k = 5 for mvdec_fpmax seller segmentation
lambda = 0.8
mu_param = 0.5
gamma = 0.5
beta = 2.0
max_iter = 30
random_state = 42
```

For paper-style sensitivity later:

```text
mu_param in {0.1, 0.2, 0.3, 0.5}
gamma in {0.0, 0.5, 1.0}
beta in {1.5, 2.0, 3.0}
lambda in {0.5, 0.8, 1.0}
```

## 12. Validation Plan

### 12.1 Paper Artificial Categorical Example

Recreate the paper's artificial categorical dataset from Table 1.

Expected validation targets:

- Table 2: IDCs;
- Table 6: categorical intra-cluster complexity;
- Table 7: categorical inter-cluster similarity;
- Table 8: categorical weights;
- Table 9: distances for the example object `x0 = (b, e, h)`.

This is the first correctness gate for categorical logic.

Table 8 caution:

The printed weights for Attribute2 appear inconsistent with Eq. (22): the
unscaled weights in the table do not sum to `1`, and the scaled weights do not
sum to `m_c = 3`. Tests should prefer Eq. (22) applied to the printed `phi`
values while documenting the discrepancy.

### 12.2 Iris Numerical Example

Recreate the Iris example used in Table 3, Table 4, and Table 5 with min-max
normalization and true class labels.

Expected validation targets:

- Table 3: coefficient of variation for each numerical attribute and class;
- Table 4: normalized coefficient of variation;
- Table 5: pairwise numerical inter-cluster similarity.

The sandbox implements Eq. (5) with the population variance denominator printed
in the paper. Most Table 3 values match closely using sklearn Iris with min-max
normalization. The Setosa petal-width coefficient differs slightly, likely due
to an unspecified data/normalization detail in the paper.

### 12.3 Small Independent Toy Tests

Toy categorical pattern:

```text
Cluster 1: A, A, B, B
Cluster 2: A, C, C, C
```

Expected behavior:

- `B` should be characteristic of Cluster 1;
- `C` should be characteristic of Cluster 2;
- `A` should be less discriminative because it appears in both clusters;
- distance from value `B` to Cluster 1 should be lower than to Cluster 2.

Toy numerical pattern:

- one numeric feature with compact, separated clusters should get higher
  weight;
- one noisy overlapping feature should get lower weight.

### 12.4 Main Repo Smoke

After formula blocks pass:

```text
X_num = h_fused_df
X_cat = FP-Max binary features
```

Start with a small FP-Max configuration:

```text
strategy = "uniform" or "quantile"
n_bins = 3
min_support = 0.60 or 0.65
n_clusters = 5
```

Compare against existing K-Prototypes outputs without replacing them.

## 13. Known Paper Limitations To Track

The paper itself notes:

- the initial prototype selection is best suited to reasonably uniform data
  distributions;
- parameter selection is empirical and lacks a theoretical derivation.

Track these in experiments rather than hiding them.

## 14. Integration Boundary

Do not integrate into `src/pipeline` until:

- Eq. (2)-(4) passes IDC checks;
- Eq. (24)-(26) passes categorical distance checks;
- categorical tables from the paper are approximately reproduced;
- the full loop converges on at least one toy mixed dataset;
- the implementation has focused unit tests for all formula blocks.
