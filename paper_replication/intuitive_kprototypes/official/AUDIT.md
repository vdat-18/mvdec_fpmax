# Audit of the authors' Intuitive-K-prototypes code

Audited on 2026-11-25. Subject: `Code.py` and `processed.cleveland.txt` as received
from the authors (see [README.md](README.md)). The audit compares the code against:

- the paper's equations and algorithms;
- our implementation (`src/intuitive_kprototypes/model.py`);
- the UCI data;
- the paper's reported numbers.

The scripts are in `audit/`.

## Verdict

- **Algorithm:** `Code.py` implements Eqs. (1)-(29) and Algorithms 1-2 faithfully, with
  one documented difference: Eq. (26) is applied as `(1 - mu) d1 + mu d2`, the reverse
  of the printed formula. Our model gives **identical partitions** to it on 7 of the 8
  paper datasets: 47 of 47 comparable runs (ARI 1.000 between the two), plus one run in
  which the official code collapsed to an empty cluster. Chess was not
  compared because the official code is too slow on 4151 rows; it runs through the same
  mixed-data code path as Australian.
- **Surroundings:** the code around the algorithm has defects that change reported
  numbers:
  - the demo's missing-value imputation is a no-op under pandas 3;
  - F1 is not permutation-invariant;
  - empty clusters are carried on silently and can collapse a run.

  None of these is present in our pipeline.

## 1. Data

- `processed.cleveland.txt` equals our UCI copy (`data/uci/raw/heart_cleveland_*.csv`)
  value for value (303 x 13 features, NaN-aware), including both label encodings
  (5-class: 164/55/36/35/13; binary: 164/139).
- Six values are missing: `ca` in 4 rows and `thal` in 2.
- The mode imputation the authors describe gives the same result as ours (`ca` = 0,
  `thal` = 3).

## 2. Equation-by-equation check

| Paper | `Code.py` | Match | Our model |
| --- | --- | --- | --- |
| Eq. (1) mixed distance (Manhattan + Hamming) | `_mixed_distance` | yes | `mixed_initial_distance_matrix` |
| Alg. 1 initial prototypes | `_select_initial_prototypes` | yes, with additions (§3.3) | `select_initial_prototypes(strategy="first")` |
| Eqs. (2)-(4) membership / non-membership | `_update_intuitionistic_centroid` | yes | `compute_intuitionistic_centroids` |
| Eqs. (5)-(6) coefficient of variation, normalised, cluster-weighted | `_num_intra_complexity` (population std) | yes | `numeric_complexity` |
| Eqs. (7)-(14) global-local granular-ball similarity | `_num_inter_similarity` | yes, except the Eq. (12) `d* = 0` branch (§3.4) | `numeric_similarity` |
| Eqs. (15)-(16), (21)-(22) weights, `1/(beta-1)`, weight 0 when phi = 0, times m | `_calc_num_weight`, `_calc_cat_weight` | yes | `_inverse_phi_weights` |
| Eqs. (17)-(20) intuitionistic entropy and similarity | `_cat_intuitionistic_entropy`, `_cat_inter_similarity` | yes | `categorical_complexity`, `categorical_similarity` |
| Eqs. (24)-(25) d1, d2 | `_cat_composite_distance_sample` | yes | `categorical_attribute_distances` |
| Eq. (26) composite distance `mu d1 + (1 - mu) d2` | `(1 - lamda) d1 + lamda d2` | **reversed** | `mu_convention="swapped"` |
| Eqs. (27)-(29) cost and weighted distance | `_compute_cost`, assignment loop | yes | `_state_from_labels` |
| Alg. 2 loop: stop when labels unchanged or `max_iter` | `fit_predict` | yes | `fit` |

**Eq. (26).**
- The paper's parameter `mu` is the code's `lamda`. The code weights the membership
  distance by `1 - lamda`.
- At the paper's recommended values (0 to 0.4), the printed form collapses clusters,
  while the code's form does not. The code's form is also the one consistent with Fig. 5.
- Table 9 uses 0.5, where the two forms coincide. This is why the printed formula looked
  right on the worked example.

## 3. Defects and deviations, by impact

### 3.1 Demo imputation is silently skipped under pandas >= 3 (high, demo only)

The demo imputes with `df_raw[col].fillna(..., inplace=True)`.

- **What happens.** Under pandas 3 Copy-on-Write this chained call raises only a
  `ChainedAssignmentError` warning, and the demo itself suppresses all warnings. The
  frame is left unchanged: the 6 missing values become a category `"nan"` after label
  encoding.
- **Effect.** Run as shipped (pandas 3.0.3), the demo gives AC 0.815, ARI 0.395,
  NMI 0.307. With the imputation actually applied it gives AC 0.838, ARI 0.456,
  NMI 0.360.
- **Fix.** Assign the result: `df[col] = df[col].fillna(...)`.
- **Our pipeline** imputes by assignment and is unaffected.

### 3.2 F1 is not permutation-invariant (high, if the paper's F1/PR/RE used it)

- **What happens.** `evaluate_clustering` passes raw cluster ids to
  `f1_score(average="macro")`. Only AC goes through the Hungarian matching.
- **Effect.** The same partition with its two cluster ids swapped scores F1 0.899 vs
  0.091, while AC stays 0.900 (`audit/edge_cases.py`). The demo's F1 looks right only
  because its ids happen to align.
- **Consequence.** If the paper's PR, RE and F1 columns were computed this way, they
  depend on arbitrary cluster numbering.
- **Our handling.** Our pipeline does not report F1. The study uses ACC (Hungarian),
  ARI and NMI.

### 3.3 Algorithm 1 details the paper leaves open (informational)

The code settles these points:

- `x0` is random (`default_rng(random_state)`).
- `belt` (the paper's lambda) defaults to 1.0.
- Line 18 takes the **first admissible row in file order**.
- When no row is admissible, the code multiplies the threshold by 0.8 and retries. Below
  1e-6 it takes the farthest row. This is not in the paper.

Our `init_strategy="first"` with `lambda_init=1.0` matches this for every case tested.
Our `strict_init=False` fallback (farthest row) differs from the 0.8 retry only when no
row clears the threshold, which happened in no audited run.

### 3.4 Eq. (12) `d* = 0` branch missing (low)

- **Paper.** Eq. (12) sets the similarity to 1 when `d* = 0`.
- **Code.** It divides instead:
  - When the denominator is 0 and the numerator is not, `1 - exp(-inf)` still gives 1.
  - For an attribute that is constant across the clusters, it gives 0/0 = NaN.
  - A NaN in any phi turns the whole weight vector into uniform weights. The NaN makes
    `denom > EPS` false, so the code falls back to `1 / len`, and the weighting scheme
    is switched off without notice.
- **Our model** implements the branch.
- **Impact.** No impact on the paper datasets: constant attributes are removed in
  preprocessing.

### 3.5 Empty clusters are carried on (medium)

- **Behaviour.** When a cluster empties, the code warns and continues:
  - Its membership degrees become 0, so its categorical membership distance is 0.
  - With numerical attributes, its centre is NaN, and `np.argmin` picks NaN positions.
- **Example.** On BCW, random initialisation with `random_state=0` empties a cluster in
  the first iteration. The run then ends with all 699 rows in one cluster after 100
  iterations, with 99 suppressed warnings.
- **Our model** raises on empty clusters by default, or repairs them under
  `empty_cluster_policy="farthest"`.
- **Implication for the paper.** BCW's reported standard deviation (0.001) implies such
  runs were not part of the paper's 50.

### 3.6 Weight 0 for a "perfect" attribute (paper-level)

- **Rule.** Eqs. (16)/(22) give weight 0 when phi = 0. Phi is 0 when an attribute is
  constant within every cluster and fully separates them, which is the most informative
  attribute.
- **Both implementations** follow the paper here. This is a property of the method, not
  a code defect, and is worth one sentence in our paper.

### 3.7 Minor

- **NMI.** NMI uses scikit-learn's arithmetic normalisation; Eq. (37) is the geometric
  one. The difference is around 1e-4 in our probe; negligible.
- **Cost.** The verbose cost pairs the new labels with the previous centres. It is a
  diagnostic only and does not affect results.
- **Speed.** The implementation loops row by row in pandas: about 15 s per Heart run
  and 60-100 s per Australian run, against well under a second for ours. Chess is
  impractical with it.
- **Categorical-only input.** It crashes on categorical-only data unless categories are
  integer-encoded: string categories make the empty numerical block sum to a string.
  The authors' label encoding avoids this.

## 4. Equivalence with our model

Commands: `audit/equivalence.py <dataset> <select|random> <seeds>`. Parameters come
from `configs/fpmax_study.json`. "Random" initialisation uses the official
`DataFrame.sample(random_state)` indices in both implementations.

| Dataset | Init | Seeds | Identical partitions |
| --- | --- | --- | --- |
| Iris | select / random | 0-2 / 0-2 | 3 / 3 |
| Soybean (21 attrs) | select / random | 0-2 / 0-2 | 3 / 3 |
| BCW | select / random | 0-2 / 0-2 | 3 / 2 (seed 0: official collapses, ours refuses the empty cluster) |
| Heart-2 | select / random | 0-4 at beta 6 and 20, plus 0-2 / 0-2 | 13 / 3 |
| Heart-5 | select / random | 0-2 / 0-2 | 3 / 3 |
| Australian | select / random | 0-1 / 0-1 | 2 / 2 |
| Ionosphere | select / random | 0-1 / 0-1 | 2 / 2 |

## 5. What the official code reproduces from the paper

All values are from 50 runs; the comparison target is the paper's tables.

**Heart-2** (`mu = 0.3`, `gamma = 0.2`; paper Table 16):

| Init | beta | AC | ARI | NMI |
| --- | --- | --- | --- | --- |
| random | 6 (authors' demo) | 0.821 (0.009) | 0.410 (0.023) | 0.323 (0.019) |
| random | 20 (our fit) | 0.821 (0.040) | 0.417 (0.064) | 0.327 (0.051) |
| **paper, random** | | **0.826 (0.012)** | **0.423 (0.033)** | **0.334 (0.025)** |
| select | 6 | 0.820 (0.009) | 0.407 (0.024) | 0.320 (0.019) |
| select | 20 | 0.827 (0.011) | 0.426 (0.027) | 0.334 (0.024) |
| **paper, select** | | **0.825 (0.013)** | **0.420 (0.035)** | **0.333 (0.027)** |

`beta = 6`, the value in the authors' demo, matches the paper's random row within one
standard deviation, and with a similar spread. `beta = 2` does not (AC 0.775).

**Ionosphere** (paper Table 12: random 0.710, select 0.800):
- The authors' algorithm reaches at most AC 0.686 over a grid of `gamma` in
  {0, 0.25, 0.5, 0.75, 1} and `beta` in {1.5, 2, 4, 6, 10, 20}, with 20 runs each and
  Algorithm 1 initialisation.
- The paper's Ionosphere rows are therefore **not reproducible with the shipped code**.
- Our earlier Chebyshev-distance initialisation, the only way we found to reach 0.800,
  is not what the authors do. Treat it as a fitted workaround, not the method.

## 6. Answers to our questions to the authors

The questions are those in `EMAIL_TO_AUTHORS.md`.

1. **beta per dataset.** Only Heart-2 is answered: the demo uses 6.
2. **mu orientation.** Settled: the code uses `1 - mu` on `d1` (our `swapped`).
3. **mu for Soybean, Heart-5, Chess.** Not answered.
4. **Soybean attributes.** Not answered by the code. The paper removes single-valued
   attributes, which is consistent with our 21.
5. **Algorithm 1.**
   - (a) `x0` is random.
   - (b) The first admissible row is taken, with a 0.8 threshold retry.
   - (c) The 0.000 standard deviations are not explained by the code. A fixed
     `random_state` (default 42) across runs would explain them; this is a guess.
6. **Restarts.** The code has none: one fit per run.
7. **Heart-5 filtering.** Not answered. The code drops nothing; it carries empty
   clusters on.
8. **Preprocessing and row order.** Confirmed: mode/median imputation, label encoding,
   min-max scaling, file order.

## 7. Consequences for this project

- **Correctness.** Our model is a verified, faster equivalent of the authors' code.
  Cite "matches the authors' implementation" for the swapped Eq. (26).
- **Heart-2 config.** The FP-Max study's `beta = 20` for Heart-2 is a fitted value. The
  authors' 6 gives the same mean with a smaller spread. Changing it would alter a
  running study, so it stays as pre-registered and is listed as a deviation.
- **Initialisation.** The study's `farthest_first` initialisation remains a documented
  deviation from Algorithm 1. It applies to both `none` and FP-Max arms, so it does not
  bias the comparison.
- **Ionosphere.** Its paper row should be reported as not reproduced by the authors' own
  code.
