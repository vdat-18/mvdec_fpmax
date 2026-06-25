# Intuitive-K-prototypes Paper Replication

This folder is a sandbox for replicating the paper:

Wang, H., & Mi, J. (2025). "Intuitive-K-prototypes: A mixed data clustering
algorithm with intuitionistic distribution centroid." Pattern Recognition,
158, 111062. https://doi.org/10.1016/j.patcog.2024.111062

The goal here is correctness-first replication of the paper, not immediate
integration with the main `mvdec_fpmax` pipeline.

## Scope

This sandbox should:

- translate the paper equations into implementation-ready specifications;
- validate each formula block on small artificial examples before full use;
- compare against the paper's artificial categorical example where possible;
- keep the current `src/pipeline` implementation untouched until the
  replication is verified.

This sandbox should not yet:

- replace the existing K-Prototypes pipeline;
- run full MiMvDEC/FP-Max sensitivity experiments;
- treat legacy notebooks as the source of truth.

## Source Of Truth

Use the paper as the primary source of truth. The notebooks in
`../../../legacy/ikc.ipynb` and `../../../legacy/ikc_cleaned.ipynb` can be used
as secondary references only, because they are independent draft
implementations.

## Recommended Build Order

1. Specify all symbols, shapes, equations, and edge cases in `spec.md`.
2. Implement and test Eq. (2)-(4), the intuitionistic distribution centroid.
3. Implement and test categorical distance Eq. (24)-(26).
4. Implement numeric and categorical attribute weight blocks.
5. Assemble the full iterative algorithm only after the blocks pass tests.
6. Add a small adapter for `h_fused_df` plus FP-Max binary features.
7. Refactor into `src/pipeline` only after the sandbox results are stable.

## Current Sandbox Code

The current implementation lives in:

```text
src/intuitive_kprototypes/
```

It includes:

- `IntuitiveKPrototypes`, an estimator-style Algorithm 2 implementation;
- initial prototype selection from Algorithm 1;
- intuitionistic distribution centroids;
- numerical and categorical complexity/similarity/weights;
- categorical and mixed distance calculations;
- tests for the paper's artificial categorical example and smoke tests.
- tests for the paper's Iris numerical examples.

Run checks from the repository root:

```bash
uv run pytest paper_replication/intuitive_kprototypes/tests -q
uv run ruff check paper_replication/intuitive_kprototypes/src paper_replication/intuitive_kprototypes/tests
```

Important: Eq. (4) has two documented modes. The estimator defaults to
`non_membership="paper"`, the formal Eq. (4) reading used for real datasets.
The tests for the artificial categorical example use `non_membership="table"`
because the paper text/proof and the paper tables are not perfectly consistent.
Table 8 also appears to have a small printed weight typo; tests follow Eq.
(22). For the Iris numerical example, Eq. (5) is implemented with the
population variance printed in the paper; one Table 3 value differs slightly
under sklearn Iris min-max normalization, so that regression test allows a
documented tolerance.

## Verification Targets

The first meaningful milestone is to reproduce the paper's artificial
categorical example:

- Table 2: intuitionistic distribution centroids;
- Table 6: categorical intra-cluster complexity;
- Table 7: categorical inter-cluster similarity;
- Table 8: categorical weights;
- Table 9: categorical distances for the example object.

## Fetching Paper UCI Data

Table 10 of the paper uses UCI datasets. To fetch and preprocess those datasets
inside this sandbox, run from the repository root:

```bash
uv run --link-mode=copy --with ucimlrepo python paper_replication/intuitive_kprototypes/scripts/fetch_paper_uci_datasets.py
```

The script writes:

```text
paper_replication/intuitive_kprototypes/data/uci/raw/
paper_replication/intuitive_kprototypes/data/uci/processed/
paper_replication/intuitive_kprototypes/data/uci/manifest.json
```

The processed files follow the paper's Table 10 feature split as closely as
possible: missing values are filled with the mode, numerical columns are
min-max normalized, Chess is filtered to labels `ten` and `fifteen`, and Heart
Disease is saved as both two-cluster and five-cluster variants.

Soybean Small is saved in two variants: `soybean_small` keeps all 35
categorical attributes to match Table 10, while `soybean_small_no_constants`
removes constant columns to follow the paper's preprocessing sentence.

Important: these UCI data runs are benchmark checks, not the first proof of
equation correctness. The equation-level correctness gate remains the unit
tests against the paper's artificial categorical tables and Iris numerical
tables.

After fetching the data, run a small smoke benchmark with:

```bash
uv run python paper_replication/intuitive_kprototypes/scripts/run_uci_smoke.py
```

To run every fetched Table 10 dataset:

```bash
uv run python paper_replication/intuitive_kprototypes/scripts/run_uci_smoke.py --dataset all
```

The smoke script prints convergence status, cluster sizes, and external
metrics against the UCI labels. Exact equality with the paper's reported
50-run averages is not expected until the remaining undocumented settings are
reproduced and the same repeated-run protocol is implemented.

Current caveat: `soybean_small` with all 35 categorical columns can still
collapse into an empty cluster under the sandbox's strict empty-cluster policy.
`soybean_small_no_constants`, which removes constant columns according to the
paper's preprocessing sentence, does run successfully.

## Reproducing Paper Tables 11-18

After the smoke checks pass, run the repeated benchmark shaped like the paper's
summary tables:

```bash
uv run python paper_replication/intuitive_kprototypes/scripts/run_paper_benchmark.py
```

The benchmark uses 50 repeated runs by default, paper-facing parameters
transcribed from Tables 11-18 where available, and
`empty_cluster_policy="farthest"` because the paper does not define
empty-cluster recovery for real benchmark runs. The estimator itself still
defaults to `empty_cluster_policy="raise"` for formula-level strictness. The
benchmark writes:

```text
paper_replication/intuitive_kprototypes/output/paper_benchmark_summary.csv
paper_replication/intuitive_kprototypes/output/paper_benchmark_runs.jsonl
```

For a fast development check:

```bash
uv run python paper_replication/intuitive_kprototypes/scripts/run_paper_benchmark.py --runs 2 --dataset iris --dataset bcw
```

To run every configured paper dataset explicitly:

```bash
uv run python paper_replication/intuitive_kprototypes/scripts/run_paper_benchmark.py --dataset all
```

To audit the effect of rejecting singleton or non-converged runs:

```bash
uv run python paper_replication/intuitive_kprototypes/scripts/run_paper_benchmark.py --dataset all --valid-runs-only
```

This retries seeds until each recorded run converges and every cluster has at
least `--min-cluster-size` objects, up to `--max-attempts-per-run`. It is a
diagnostic protocol rather than a claim that the paper used the same rejection
rule.

For difficult datasets, run a small parameter grid:

```bash
uv run python paper_replication/intuitive_kprototypes/scripts/run_parameter_grid.py --valid-runs-only
```

The grid defaults to the `quick` preset on the difficult datasets and writes
`paper_replication/intuitive_kprototypes/output/parameter_grid.csv`.
Available presets:

```text
quick:
  mu_param      = 0.3, 0.5, 0.8
  gamma         = 0.0, 0.5, 1.0
  lambda_init   = 0.2, 0.3, 0.5
  beta          = 2.0, 3.0

balanced:
  mu_param      = 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0
  gamma         = 0.0, 0.2, 0.5, 0.7, 1.0
  lambda_init   = 0.1, 0.2, 0.3, 0.5, 0.8, 1.0
  beta          = 1.5, 2.0, 3.0, 5.0

paper:
  mu_param      = 0.1, 0.2, 0.3, 0.4
  gamma         = 0.0, 0.2, 0.4, 0.5, 0.7, 1.0
  lambda_init   = 0.3
  beta          = 2.0
```

Use a preset with:

```bash
uv run python paper_replication/intuitive_kprototypes/scripts/run_parameter_grid.py --grid balanced --valid-runs-only
```

Individual parameter lists still override the preset:

```bash
uv run python paper_replication/intuitive_kprototypes/scripts/run_parameter_grid.py --grid quick --mu-param 0.5,0.8,1.0 --valid-runs-only
```

Run grid configs concurrently with:

```bash
uv run python paper_replication/intuitive_kprototypes/scripts/run_parameter_grid.py --grid balanced --valid-runs-only --n-jobs 4
```

Parallel execution uses independent worker processes per dataset/parameter
combo and writes rows in deterministic grid order.

The broad audit artifacts currently used for replication triage are:

```text
paper_replication/intuitive_kprototypes/output/wide_mu_gamma_grid.csv
paper_replication/intuitive_kprototypes/output/best_config_report.csv
paper_replication/intuitive_kprototypes/output/final_best_config_report.csv
```

The summary CSV includes mean/std metrics plus deltas against the paper's
reported "with select prototypes" rows. Exact equality is not guaranteed
because the paper does not publish code, random seeds, or all preprocessing
details.

Debug notes from the first 50-run audit:

- PR/RE/F1 are computed as macro object-level scores after best
  cluster-to-label mapping. This matches the paper's metric description better
  than pair-count precision/recall.
- The benchmark summary includes `converged_runs` and `singleton_runs`.
  Singleton-heavy outputs such as `[1, n-1]` are the main remaining source of
  metric gaps on BCW, ACA, Heart, and Soybean under the transcribed paper
  parameters.
- Filtering to converged, non-singleton BCW runs gives metrics close to the
  paper, which suggests the categorical formula blocks are mostly sound while
  the remaining gap is driven by initialization/empty-cluster behavior and
  undocumented experiment protocol details.
- The estimator supports opt-in `min_cluster_size` repair with
  `empty_cluster_policy="farthest"`. A small grid probe shows Soybean and
  Heart Disease with five clusters improve when `mu_param` is raised from
  `0.2` to `0.5`; Soybean's all-columns and no-constant variants behave
  similarly in this probe, so constant columns are not the main gap source.
