# Intuitive-K-prototypes: the `mu` convention in Eq. (26)

> **Confirmed by the authors' code (2026-11-25).** The authors (Wang & Mi) sent
> their Python implementation. Its categorical distance is
> `(1 - lamda) * d1 + lamda * d2`, which is `--mu-convention swapped` here, not
> Eq. (26) as printed. Run side by side on Cleveland Heart-2 (`lamda = 0.3`,
> `gamma = 0.2`, `beta` 6 and 20, seeds 0-4, Algorithm 1 initialisation), the
> official code and this model (`mu_convention="swapped"`, `init_strategy="first"`,
> `lambda_init=1.0`, the official `belt`) give identical partitions in all 10 runs
> (ARI 1.000 between them). The official code stays outside the repository.
> Other details the code settles: Algorithm 1 uses `belt = 1.0` and takes the first
> admissible row in index order; the authors' Heart-2 demo uses random initial
> prototypes with `beta = 6` (ACC 0.838, ARI 0.456, NMI 0.360). Their preprocessing:
> mode/median imputation, label-encoded categories, min-max numerics.

## What the paper prints

Definition 9, Eq. (26): `d = mu * d1 + (1 - mu) * d2`, with `d1` the membership
distance and `d2` the non-membership distance. The model code matches this
exactly, and the Table 9 unit test pins it at `mu = 0.5`, where the two
orientations are indistinguishable. The model was therefore left unchanged.

## What the paper's experiments imply

Section 5.2 (Fig. 5) says the clustering collapses into one cluster when the
parameter is larger than a threshold, and recommends `0 <= mu <= 0.4`. Tables
13-18 use `mu` between 0.2 and 0.5.

With Eq. (26) as printed, this implementation does the opposite: at `mu = 0.2`
BCW collapses in 40 of 50 runs (cluster sizes like `[1, 698]`, accuracy 0.716),
while `mu = 0.5-0.7` gives no collapse and accuracy 0.95. Passing `1 - mu`
removes the collapse on every dataset with categorical attributes and matches
Table 13 (BCW) almost exactly.

## Benchmark results (50 runs, accuracy)

| Dataset | Eq. (26) as printed | `1 - mu` (`--mu-convention swapped`) | Paper |
| --- | ---: | ---: | ---: |
| iris | 0.960 | 0.960 | 0.960 |
| ionosphere (numeric only) | 0.683 | 0.683 | 0.800 |
| bcw | 0.716 | 0.951 | 0.950 |
| soybean_small (35 cols) | 0.766 | 0.766 | 1.000 |
| australian_credit_approval | 0.578 | 0.743 | 0.856 |
| heart_cleveland_2 | 0.566 | 0.805 | 0.825 |
| chess_ten_fifteen | 0.660 | 0.812 | 0.756 |
| heart_cleveland_5 | 0.551 | 0.450 | 0.547 |

## Status

This is empirical evidence, not a proof about the authors' code, which is not
public. The benchmark exposes both conventions via `--mu-convention
{eq26,swapped}`; the default stays `eq26`. Any use of `swapped` must be reported
as a deviation from the printed Eq. (26).

Still unexplained after the swap: ionosphere (no categorical attributes, so
`mu` is irrelevant), soybean_small with 35 columns (still collapses), the
Australian gap, and the heart 5-cluster result.
