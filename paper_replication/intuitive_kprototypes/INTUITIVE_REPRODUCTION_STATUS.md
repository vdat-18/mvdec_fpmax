# Intuitive-K-prototypes: benchmark reproduction status

Targets are Tables 11-18 of Wang & Mi (2025). Each table has two Intuitive rows:
"Intuitive-K-prototypes" (random initial prototypes, Algorithm 2) and "with
select prototypes" (Algorithm 1). Scenarios are scored against the matching row.
All numbers below are means over 50 runs, `empty_cluster_policy="farthest"`,
Soybean without constant columns, produced by
`scripts/run_scenario_sweep.py --grid rows --runs 50`.

## Formula level (done)

Paper tables 2-9 and the Iris numeric example reproduce in `tests/` (13 tests
pass). Eq. (26) in the code matches the printed formula.

## Benchmark level: what holds

1. **`mu` convention decides collapse.** With Eq. (26) as printed, 202-222 of
   400 runs collapse into clusters of at most two objects. Passing `1 - mu`
   (`--mu-convention swapped`) cuts that to 0 on the datasets with categorical
   attributes except Soybean. See `INTUITIVE_MU_CONVENTION.md`.
2. **Other knobs do not matter much**: `lambda_init`, `max_iter`, `gamma`, and
   Soybean 35 versus 21 columns change the gap by less than 0.01. The
   `non_membership="table"` rule fails (`mu + nu <= 1` violated) on 5 of 8
   datasets and is rejected.
3. **Best match, Algorithm 1 + `swapped`, against the "with select" row** (accuracy):

| Dataset | Ours | Paper | Note |
| --- | ---: | ---: | --- |
| iris | 0.960 | 0.960 | exact |
| bcw | 0.951 | 0.950 | ARI 0.811 / 0.807, NMI 0.702 / 0.697 |
| heart_cleveland_2 | 0.790 | 0.825 | |
| chess_ten_fifteen | 0.815 | 0.756 | above paper; paper std 0.086 on this row |
| australian_credit_approval | 0.756 | 0.856 | |
| heart_cleveland_5 | 0.450 | 0.547 | |
| soybean_small | 0.761 | 1.000 | 46 of 50 runs collapse |
| ionosphere | 0.680 | 0.800 | unreachable, see below |

## Open problems

- **Ionosphere "with select" (0.800 / 0.334 / 0.275, std 0.000) cannot be
  produced.** Across all 33 prototype pairs that Algorithm 1 can pick, accuracy
  peaks at 0.704; varying `gamma`, `beta`, distance form and normalisation does
  not help. Our result (0.683) is consistent with the paper's plain row
  (0.710, std 0.119) and with its K-prototypes row (0.711). Plain k-means,
  Ward and spectral clustering reach 0.65-0.72; only a full-covariance GMM
  (0.815) gets near 0.80.
- **Random-initialisation stability.** The paper's plain row has accuracy std
  0.001 on BCW, 0.011 on Australian, 0.012 on Heart-2 and 0.000 on Chess. Our
  uniformly random rows give std 0.10-0.15 even with `swapped`. The paper's
  "random" initialisation is therefore probably not plain uniform sampling of
  rows; restarts with cost selection are an untested explanation.
- **Soybean** (paper: 1.000). Table 14 lists only `gamma = 0.5`, no `mu`, so the
  benchmark's default `mu = 0.5` was an assumption, and it sits in the collapse
  region (49 of 50 runs). Passing `mu = 0.7-0.8` to the code removes the
  collapse (0 of 50). The remaining shortfall comes from Algorithm 1 line 18,
  which the shipped code resolves by taking the first admissible row: the
  Soybean file is sorted by class, so the prototypes come from classes
  `D2, D3, D1, D1` (none from D4). Picking randomly among admissible rows
  (`--grid init`, `alg1rand`) lifts mean accuracy from 0.790 to 0.927; the
  paper's 1.000 with std 0.000 is still not reached (farthest-first: 0.991).

## Initialisation variants on all 8 datasets

`swapped` mu, Soybean `mu = 0.8`, 50 runs, accuracy (paper "with select" row in
the last column). For `k = 2` Algorithm 1 has no free choice at line 18, so
`alg1` and `alg1rand` coincide there; they differ only on Soybean (k=4) and
Heart-5 (k=5).

| Dataset | alg1 | alg1rand | farthest | random | Paper |
| --- | ---: | ---: | ---: | ---: | ---: |
| iris | 0.960 | 0.960 | 0.952 | 0.901 | 0.960 |
| ionosphere | 0.680 | 0.680 | 0.690 | 0.690 | 0.800 |
| bcw | 0.951 | 0.951 | 0.932 | 0.885 | 0.950 |
| soybean_small | 0.790 | 0.927 | 0.991 | 0.906 | 1.000 |
| australian_credit_approval | 0.756 | 0.756 | 0.797 | 0.769 | 0.856 |
| heart_cleveland_2 | 0.790 | 0.790 | 0.804 | 0.774 | 0.825 |
| chess_ten_fifteen | 0.815 | 0.815 | 0.776 | 0.730 | 0.756 |
| heart_cleveland_5 | 0.450 | 0.402 | 0.422 | 0.385 | 0.547 |
| mean gap (acc, ARI, NMI) | 0.103 | 0.084 | 0.060 | 0.103 | |

No initialisation matches every dataset: `alg1rand` fixes Soybean but worsens
Heart-5, and `farthest` has the lowest mean gap but produces 3 collapsed BCW
runs. Heart-5 (paper 0.547, ours 0.385-0.450) and Ionosphere remain unexplained.
Gaps between variants are small relative to run-to-run spread, and the variants
were scored against the paper's own numbers, so none should be read as the
authors' choice.
- **Table 17 (Chess)** prints accuracy std 0.000 on the plain row and 0.086 on
  the "with select" row, the reverse of the other tables.

## Round 2: restarts (H1) and exhaustive prototype pairs (H2)

The model now exposes `init_strategy` and `mu_convention` (defaults reproduce the
earlier numbers exactly; 32 tests pass). Restarts keep the lowest assigned-distance
fit per run (`--grid restarts`, `swapped` mu, Soybean mu 0.8, 50 runs).

**H1, random initialisation with min-cost restarts, accuracy mean (std) against
the paper's plain row:**

| Dataset | 1 restart | 5 restarts | 10 restarts | Paper |
| --- | ---: | ---: | ---: | ---: |
| iris | 0.884 (0.164) | 0.960 (0.000) | 0.960 (0.000) | 0.960 (0.000) |
| bcw | 0.838 (0.145) | 0.950 (0.000) | 0.950 (0.000) | 0.949 (0.001) |
| soybean_small | 0.905 (0.132) | 0.994 (0.032) | 1.000 (0.000) | 1.000 (0.000) |
| ionosphere | 0.683 (0.017) | 0.697 (0.003) | 0.697 (0.002) | 0.710 (0.119) |
| australian_credit_approval | 0.771 (0.089) | 0.786 (0.082) | 0.827 (0.048) | 0.848 (0.011) |
| chess_ten_fifteen | 0.703 (0.136) | 0.689 (0.160) | 0.733 (0.149) | 0.842 (0.000) |
| heart_cleveland_2 | 0.773 (0.062) | 0.695 (0.091) | 0.654 (0.088) | 0.826 (0.012) |
| heart_cleveland_5 | 0.396 (0.041) | 0.376 (0.027) | 0.369 (0.018) | 0.546 (0.018) |

Min-cost restarts explain the paper's tiny std on Iris, BCW and Soybean: with 10
restarts all three match the plain row (std 0.000). Ionosphere falls inside the
paper's wide band. Australian is close but outside one paper std. Chess, Heart-2
and Heart-5 do not match, and for Heart-2 more restarts make accuracy worse, so
assigned-distance cost is not what picks the paper's runs on those sets.

**H2, exhaustive 2-row prototype pairs on Ionosphere** (61,425 pairs, `gamma=0`):
accuracy ranges 0.65-0.82, but only 297 pairs (0.5%) reach 0.75 and Algorithm 1
selects the extreme pair that gives 0.67. The best pairs reach 0.821 / 0.398 /
0.304, above the paper's 0.800 / 0.334 / 0.275; the closest pair gives 0.809 /
0.369 / 0.272. No pair matches the paper within 0.01 accuracy and 0.02 ARI. So 0.80
is reachable in principle, but not through the prototype selection we implement and
not exactly.

Not yet run: H3 (Fig. 5 calibration), H4 (Eq. (4) reading), H5 (lambda, beta),
H6 (metric definitions).

## Caveats

Scenarios were chosen against the paper's numbers, so a matching scenario shows
which reading is plausible, not what the authors ran. Using `swapped` in any
experiment must be reported as a deviation from the printed Eq. (26).
