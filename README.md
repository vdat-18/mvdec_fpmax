# mvdec-fpmax

Source code and processed experiment artifacts for:

**Enhancing Multi-View Deep Embedded Clustering using Maximal Frequent Itemset
Mining: Application to E-Commerce Store Segmentation**

This repository implements the MiMvDEC workflow used in the manuscript. The
pipeline starts from preprocessed Tiki storefront features, learns a fused
MvDEC representation, enriches the representation with FP-Max maximal frequent
itemsets, and evaluates the resulting mixed-type clustering with and without
Forward Feature Selection (FFS).

## Repository Structure

```text
data/
  preprocessed_data/
    tiki_preprocessed.csv        # 1,799 x 7 preprocessed Tiki features
    *_fused_representation.pkl   # GPU-produced MvDEC representations
  public/                        # reproducible public dataset workspace
  raw_data/
    seller_store_urls.csv        # seller storefront URLs used during collection
output/
  without_ffs_results.csv
  ffs_results.csv
  post_without_ffs_intuitive_results.csv
  post_ffs_intuitive_results.csv
src/
  data_preprocessing/            # raw-data preprocessing utilities
  representation_learning/       # MvDEC representation code
  pipeline/                      # FP-Max, K-Prototypes, FFS, experiments
docs/
  *.pdf                         # reference papers
```

## Environment

The local experiment pipeline is designed for Python 3.11.

```bash
uv sync
```

TensorFlow and PyTorch are optional because the local FP-Max/clustering pipeline
can run without GPU frameworks. Install those frameworks only in GPU runtimes
that need them:

```bash
uv pip install tensorflow  # TensorFlow MvDEC representation stage
uv pip install torch       # PyTorch public MvDEC-paper baseline
```

The MvDEC representation-learning stage is expected to run in a GPU runtime,
then the dataset-specific `*_fused_representation.pkl` artifacts are pushed
back into `data/preprocessed_data/` for local full-grid experiments.

## Workflow

### 0. Public Dataset Benchmarks

Public continuous-feature benchmarks are managed by a dataset registry and a
fetch script. Raw downloads and standardized processed files are created under
`data/public/` and are ignored by git to keep the repository clean.

```bash
uv run python scripts/fetch_public_datasets.py
```

The registry is stored at:

```text
configs/public_datasets.json
```

Each processed dataset is written as:

```text
data/public/processed/<dataset_name>/
  X.csv          # numeric continuous features only
  y.csv          # labels/sample metadata for external evaluation only
  views.json     # natural or constructed feature views
  metadata.json  # source, parser, cleaning notes, n_clusters, shape
```

Use `X.csv` as clustering input. Do not feed `y.csv` into clustering; it is only
for external evaluation or auditing. To fetch a subset first, run:

```bash
uv run python scripts/fetch_public_datasets.py --datasets uci_seeds uci_wine
```

### 1. Data Preprocessing

The preprocessing module converts raw Tiki seller data into the standardized
continuous feature matrix used by the representation-learning stage.
It removes rows with invalid join years while preserving their original row
indices, then applies the canonical DBSCAN parameters (`eps=1.5`,
`min_samples=14`).

```bash
uv run python -m data_preprocessing.cli \
  --input data/raw_data/tiki_raw_data.xlsx \
  --output data/preprocessed_data/tiki_preprocessed.csv
```

The repository already includes the processed dataset used in the experiments:

```text
data/preprocessed_data/tiki_preprocessed.csv
```

To reproduce the saved MvDEC assignments and export interpretation-ready shop
rows plus raw numeric cluster profiles, run:

```bash
uv run python scripts/recluster_mvdec.py \
  --representation-path data/preprocessed_data/tiki_mvdec_fused_representation.pkl \
  --data-path data/preprocessed_data/tiki_preprocessed.csv \
  --mapping-path data/preprocessed_data/tiki_row_mapping.csv \
  --output output/tiki_cluster_interpretation.csv \
  --profile-output output/tiki_cluster_profiles.csv \
  --force
```

The loader validates the ordered source CSV fingerprint recorded by the MvDEC
artifact. CSV line-ending differences between Linux and Windows do not change
the fingerprint, but row reordering or value changes are rejected.

### 2. Representation Learning

Run representation-learning stages on a GPU runtime. TensorFlow is intentionally
not part of the local clustering dependencies, so install it only in the GPU
environment that will train MvDEC.

```bash
uv pip install tensorflow
```

On Colab, clone the `dev` branch and sync the repo:

```bash
git clone -b dev https://github.com/vdat-18/mvdec_fpmax.git
cd mvdec_fpmax
pip install -q uv
uv sync
uv add tensorflow
```

For the air-pollution case study, the current in-repo producer is:

```bash
uv run python src/representation_learning/MVDEC_dense.py AIRPOLLUTION \
  --runs 3 \
  --max-refinement-epochs 1400 \
  --greedy-eigen-direction largest \
  --greedy-target-mode selected_dimension_only \
  --artifact-path data/preprocessed_data/airpollution_demvk_fused_representation.pkl
```

The greedy step exposes the eigen-direction and target semantics independently:

```bash
# Literal MvDEC 2025 configuration
uv run python src/representation_learning/MVDEC_dense.py AIRPOLLUTION \
  --runs 3 --seed 42 \
  --greedy-eigen-direction smallest \
  --greedy-target-mode selected_dimension_only

# Largest-eigen direction used by the original DEKM 2021 implementation
uv run python src/representation_learning/MVDEC_dense.py AIRPOLLUTION \
  --runs 3 --seed 42 \
  --greedy-eigen-direction largest \
  --greedy-target-mode selected_dimension_only
```

The default is `largest` plus `selected_dimension_only`, matching the
eigen-direction selected by the original DEKM 2021 implementation while
retaining the explicit MvDEC target semantics. It keeps the standard artifact
path. The literal MvDEC 2025 interpretation remains available with
`--greedy-eigen-direction smallest`; non-default combinations add both mode
names to the artifact filename. Final weights, cluster CSVs, and log files also
include both modes so experiments do not overwrite each other.

### Public text benchmark from the DEKM release

REUTERS, 20NEWS, and RCV1 use the existing files under
`external_repos/DEKM/datasets/` as the single source of truth. These commands
do not read or regenerate `data/public/processed/`. Every saved run records a
source fingerprint, seed, canonical row index, cluster assignment, ACC, and
NMI.

Run K-Means with the same explicit seeds used by the deep methods:

```bash
uv run python scripts/run_public_kmeans_baseline.py \
  --source dekm-release \
  --datasets REUTERS 20NEWS RCV1 \
  --seeds 42 43 44 \
  --output output/public_benchmark/kmeans/kmeans_baseline.xlsx \
  --force
```

Run DEKM and MvDEC once per dataset. `--seed 42 --runs 3` produces seeds
42, 43, and 44 for both methods:

```bash
uv run python src/representation_learning/DEKM_dense.py REUTERS \
  --seed 42 --runs 3 \
  --dataset-root external_repos/DEKM/datasets \
  --output-dir output/public_benchmark/dekm

uv run python src/representation_learning/MVDEC_dense.py REUTERS \
  --seed 42 --runs 3 \
  --dataset-root external_repos/DEKM/datasets \
  --public-output-dir output/public_benchmark/mvdec
```

Replace `REUTERS` with `20NEWS` or `RCV1`. Each MvDEC pickle is directly
usable by MiMvDEC. Pass its matching `*_assignments.csv` as `--data-path`; the
CSV is a row manifest, not a regenerated feature dataset:

```bash
uv run mvdec-fpmax ffs-kprototypes \
  --representation-path output/public_benchmark/mvdec/reuters/mvdec_reuters_seed_42.pkl \
  --data-path output/public_benchmark/mvdec/reuters/mvdec_reuters_seed_42_assignments.csv \
  --output-dir output/public_benchmark/mimvdec/reuters/seed_42
```

After selecting one fixed without-FFS or FFS configuration by Silhouette, run
`mvdec-evaluate-best` with the matching seed. For labeled public artifacts its
outputs additionally include ACC, NMI, and their deltas versus MvDEC. Ground
truth labels are never used by FP-Max, FFS, or model selection.

The released RCV1 directory contains the paper split indices; matching the
original DEKM loader also requires a readable local scikit-learn RCV1 cache.
The loader fails instead of downloading or substituting another dataset when
that cache is unavailable.

This reads `data/preprocessed_data/data_demvk.csv`, applies column-wise Min-Max
scaling to `[0, 1]`, and saves:

```text
data/preprocessed_data/airpollution_demvk_fused_representation.pkl
output/AIRPOLLUTION_clusters.csv
```

The loader rejects the previous legacy full-view-output artifact. Regenerate it
before running air-pollution FP-Max/clustering modes.

For MvDEC 2025 air-pollution artifacts, each model output uses the Eq.
(5)-compatible layout `10 latent + 13 reconstruction`, while clustering and
export use only the latent heads: `h_fused = (h_view1_latent +
h_view2_latent) / 2`. Therefore, `h_fused`, `h_view1`, and `h_view2` are all
10-dimensional. Joint training combines reconstruction, K-Means, and greedy
losses; the trace-equivalent L3 term is logged with weight zero to avoid
duplicating L2's gradient.

The run logs both `baseline_raw_input` and `baseline_scaled_input` so scale
dominance is visible. The pickle records the scaling method, fitted per-column
minimum/maximum values, feature order, and SHA-256 of the source CSV. Min-Max is
an explicit reproduction assumption based on Table 4 of the 2025 paper; the
paper does not publish its fitted scaler.

Step 9 logs split `L4_greedy` into `L4_selected_direction` and
`L4_nonselected_snapshot_anchor`, plus their fractions of total L4. Their sum
equals `L4_greedy`. In `selected_dimension_only`, the snapshot-anchor component
and its gradient are exactly zero. In `frozen_snapshot`, both components can be
non-zero.

Every K-Means refresh uses a fixed `n_init=100`. The previous inherited DEKM
behavior that replaced `n_init` with `2 * n_iter_` has been removed because the
number of random initializations and the winning run's convergence iterations
are different concepts.

K-Means refresh follows a `one_epoch` policy. A refinement cycle trains every
mini-batch exactly once before recomputing the full fused embedding, centroids,
scatter matrix, eigenvectors, and greedy target. With batch size 256 this means
15 balanced updates of 251 samples per Air Pollution cycle and 7 balanced
updates of 257 samples per TIKI cycle. Samples are deterministically reshuffled
each epoch, so every sample appears exactly once without an overweighted short
batch. The maximum remains 1400 refinement epochs, matching the former maximum
number of K-Means refreshes.

`--max-refinement-epochs` is a safety cap and can be overridden. Each run logs
`stop_reason=converged_assignment` when assignment changes satisfy the tolerance,
or `stop_reason=max_epochs_reached` when the safety cap is exhausted. The
artifact stores the stop reason and completed refinement epoch count.

Refinement stops when no more than 1% of samples change their K-Means assignment
for the unlabeled Air Pollution/Tiki case studies
(`assignment_change_tolerance=0.01`). Public REUTERS/20NEWS/RCV1 benchmarks use
the DEKM 2021 stopping protocol of 0.1% (`assignment_change_tolerance=0.001`).
Use `--assignment-change-tolerance` only for an explicitly documented ablation.

### 3. FP-Max and Clustering Experiments

Run a quick data and representation loading check:

```bash
uv run mvdec-fpmax smoke
```

Run smoke checks for the two experiment variants:

```bash
uv run mvdec-fpmax without-ffs-smoke
uv run mvdec-fpmax without-ffs-smoke --backend intuitive
uv run mvdec-fpmax ffs-smoke
```

Run the full sensitivity grids:

```bash
uv run mvdec-fpmax without-ffs-kprototypes
uv run mvdec-fpmax without-ffs-intuitive
uv run mvdec-fpmax without-ffs-intuitive-view-weighted
uv run mvdec-fpmax without-ffs-intuitive-native
uv run mvdec-fpmax without-ffs-intuitive-view-weighted-native
uv run mvdec-fpmax ffs-kprototypes
uv run mvdec-fpmax ffs-intuitive
uv run mvdec-fpmax ffs-intuitive-view-weighted
uv run mvdec-fpmax ffs-intuitive-native
uv run mvdec-fpmax ffs-intuitive-view-weighted-native
```

Run full grids with parallel worker processes:

```bash
uv run mvdec-fpmax without-ffs-kprototypes --workers 4
uv run mvdec-fpmax without-ffs-intuitive --param-workers 3
uv run mvdec-fpmax without-ffs-intuitive-best --param-workers 3
uv run mvdec-fpmax without-ffs-intuitive-exhaustive-best --param-workers 3
uv run mvdec-fpmax without-ffs-intuitive-view-weighted --param-workers 3
uv run mvdec-fpmax without-ffs-intuitive-view-weighted-best --param-workers 3
uv run mvdec-fpmax without-ffs-intuitive-view-weighted-exhaustive-best --param-workers 3
uv run mvdec-fpmax without-ffs-intuitive-native --workers 4 --param-workers 1
uv run mvdec-fpmax without-ffs-intuitive-paper-native --workers 4 --param-workers 1
uv run mvdec-fpmax without-ffs-intuitive-exhaustive-native --workers 4 --param-workers 1
uv run mvdec-fpmax without-ffs-intuitive-paper-exhaustive-native --workers 4 --param-workers 1
uv run mvdec-fpmax without-ffs-intuitive-view-weighted-native --workers 4 --param-workers 1
uv run mvdec-fpmax without-ffs-intuitive-view-weighted-paper-native --workers 4 --param-workers 1
uv run mvdec-fpmax without-ffs-intuitive-view-weighted-exhaustive-native --workers 4 --param-workers 1
uv run mvdec-fpmax without-ffs-intuitive-view-weighted-paper-exhaustive-native --workers 4 --param-workers 1
uv run mvdec-fpmax ffs-kprototypes --workers 4 --candidate-workers 5
uv run mvdec-fpmax ffs-intuitive --param-workers 3
uv run mvdec-fpmax ffs-intuitive-best --param-workers 3
uv run mvdec-fpmax ffs-intuitive-exhaustive-best --param-workers 3
uv run mvdec-fpmax ffs-intuitive-view-weighted --param-workers 3
uv run mvdec-fpmax ffs-intuitive-view-weighted-best --param-workers 3
uv run mvdec-fpmax ffs-intuitive-view-weighted-exhaustive-best --param-workers 3
uv run mvdec-fpmax ffs-intuitive-native --workers 4 --candidate-workers 5 --param-workers 1
uv run mvdec-fpmax ffs-intuitive-paper-native --workers 4 --candidate-workers 5 --param-workers 1
uv run mvdec-fpmax ffs-intuitive-exhaustive-native --workers 4 --candidate-workers 5 --param-workers 1
uv run mvdec-fpmax ffs-intuitive-paper-exhaustive-native --workers 4 --candidate-workers 5 --param-workers 1
uv run mvdec-fpmax ffs-intuitive-view-weighted-native --workers 4 --candidate-workers 5 --param-workers 1
uv run mvdec-fpmax ffs-intuitive-view-weighted-paper-native --workers 4 --candidate-workers 5 --param-workers 1
uv run mvdec-fpmax ffs-intuitive-view-weighted-exhaustive-native --workers 4 --candidate-workers 5 --param-workers 1
uv run mvdec-fpmax ffs-intuitive-view-weighted-paper-exhaustive-native --workers 4 --candidate-workers 5 --param-workers 1
```

Run only the first few jobs for a quick check:

```bash
uv run mvdec-fpmax without-ffs-kprototypes --limit 2
uv run mvdec-fpmax ffs-kprototypes --limit 2
```

Grid modes run `(strategy, n_bins)` groups in parallel by default. Existing
output is resumed by default; add `--no-resume` to rerun the selected jobs
from scratch. `--workers` parallelizes outer grid groups. `--candidate-workers`
parallelizes candidate feature sets inside each FFS step for `ffs-kprototypes`
and native FFS Intuitive modes. Post-FFS modes do not perform FFS candidate
selection, so use `--param-workers` for those modes instead.

`without-ffs-intuitive` reads completed without-FFS K-Prototypes rows,
rebuilds each exact FP-Max feature set, keeps that feature set fixed, and
grid-searches Intuitive-K-prototypes parameters on top of each row's
features. `ffs-intuitive` does the same for every usable K-Prototypes
FFS-selected feature set. The `*-view-weighted` post modes keep those fixed
feature sets but run Intuitive with an additional `alpha` grid that balances
MVDEC latent distance against FP-Max pattern distance. The `*-best` post modes
use two-stage Intuitive tuning and keep one best row per base result instead
of writing one row per parameter combination. The `*-exhaustive-best` and
`*-exhaustive-native` modes use the full Intuitive parameter grid and write to
separate output files. `without-ffs-intuitive-native`
runs Intuitive as the clustering backend directly inside the main grid.
`ffs-intuitive-native` runs forward feature selection with Intuitive as the
native evaluator, so candidate features are selected by their best Intuitive
silhouette rather than by K-Prototypes. The `*-view-weighted-native` modes
apply the same native logic with `alpha` inside Intuitive's clustering
distance. When no FP-Max or FFS feature survives, the native modes keep the
baseline score and mark rows with no FP-Max features as `baseline_no_features`.
FFS-native rows that have candidate features but no improving feature are marked
as `baseline_no_improvement`. FFS-native rows that have FP-Max candidates but no
valid Intuitive candidate are marked as `baseline_no_valid_candidate`.

Outputs keep the fields needed to compare and interpret configurations: grid
settings, final score columns, selected features, cluster sizes, row-ordered
cluster assignments, Intuitive parameters when applicable, and status/error
details. K-Prototypes FFS and native Intuitive FFS outputs use `final_score` as
the selected-feature score.
Post-FFS Intuitive follow-up uses `final_score` when that column is
available. On Windows, prefer one inner parallelism axis at a time. For native
FFS Intuitive modes, either use `--candidate-workers > 1 --param-workers 1` or
use `--candidate-workers 1 --param-workers > 1`; the CLI rejects using both
above 1 because that would create nested multiprocessing. A practical resume
command is:

```bash
uv run mvdec-fpmax ffs-intuitive-native --workers 4 --candidate-workers 5 --param-workers 1
```

Native Intuitive modes without `paper` in the name use deterministic
`farthest_first` initialization for robustness. The `*-paper-native` modes use
the paper's Algorithm-1 initialization (`init_strategy="paper"`,
`strict_init=True`) and write to separate output files so resume never mixes
initialization strategies. Runtime failures that are not algorithm-defined
statuses are not persisted; those missing `job_index` rows are retried on the
next resume run. CSV writes are atomic, so a crash during save keeps either the
previous complete CSV or the new complete CSV. FP-Max itemset feature names are
sorted deterministically before FFS so repeated runs keep stable candidate
order.

The Intuitive parameter grid follows the current two-stage tuning policy used
consistently across Intuitive modes:

```text
coarse mu_param = 0.2, 0.5, 0.8
coarse gamma    = 0.2, 0.5, 0.8
beta            = 2.0, 3.0, 4.0, 5.0
refine          = mu_param/gamma +/- 0.1 around the best coarse trial
```

`beta` is not refined. Exhaustive modes use the full candidate range for
`mu_param` and `gamma`:

```text
mu_param = 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
gamma    = 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
beta     = 2.0, 3.0, 4.0, 5.0
```

The view-weighted modes additionally tune `alpha` with the same coarse/refine
logic:

```text
coarse alpha = 0.2, 0.5, 0.8
full alpha   = 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
```

## Outputs

Default experiment outputs are isolated under `output/<dataset>/`, where the
dataset directory comes from the selected MvDEC artifact:

```text
output/<dataset>/without_ffs_results.csv
output/<dataset>/ffs_results.csv
output/<dataset>/without_ffs_intuitive_native_results.csv
output/<dataset>/without_ffs_intuitive_paper_native_results.csv
output/<dataset>/without_ffs_intuitive_exhaustive_native_results.csv
output/<dataset>/without_ffs_intuitive_paper_exhaustive_native_results.csv
output/<dataset>/without_ffs_intuitive_view_weighted_native_results.csv
output/<dataset>/without_ffs_intuitive_view_weighted_paper_native_results.csv
output/<dataset>/without_ffs_intuitive_view_weighted_exhaustive_native_results.csv
output/<dataset>/without_ffs_intuitive_view_weighted_paper_exhaustive_native_results.csv
output/<dataset>/ffs_intuitive_native_results.csv
output/<dataset>/ffs_intuitive_paper_native_results.csv
output/<dataset>/ffs_intuitive_exhaustive_native_results.csv
output/<dataset>/ffs_intuitive_paper_exhaustive_native_results.csv
output/<dataset>/ffs_intuitive_view_weighted_native_results.csv
output/<dataset>/ffs_intuitive_view_weighted_paper_native_results.csv
output/<dataset>/ffs_intuitive_view_weighted_exhaustive_native_results.csv
output/<dataset>/ffs_intuitive_view_weighted_paper_exhaustive_native_results.csv
output/<dataset>/post_without_ffs_intuitive_results.csv
output/<dataset>/post_without_ffs_intuitive_best_results.csv
output/<dataset>/post_without_ffs_intuitive_exhaustive_best_results.csv
output/<dataset>/post_without_ffs_intuitive_view_weighted_results.csv
output/<dataset>/post_without_ffs_intuitive_view_weighted_best_results.csv
output/<dataset>/post_without_ffs_intuitive_view_weighted_exhaustive_best_results.csv
output/<dataset>/post_ffs_intuitive_results.csv
output/<dataset>/post_ffs_intuitive_best_results.csv
output/<dataset>/post_ffs_intuitive_exhaustive_best_results.csv
output/<dataset>/post_ffs_intuitive_view_weighted_results.csv
output/<dataset>/post_ffs_intuitive_view_weighted_best_results.csv
output/<dataset>/post_ffs_intuitive_view_weighted_exhaustive_best_results.csv
output/<dataset>/mvdec_experiment_manifest.json
```

The manifest records the dataset, data hash, artifact hash, cluster count,
random seed, and evaluation-distance contract. Resume is rejected if any of
these differ. To run another artifact or distance contract, use its default
dataset directory or provide a new `--output-dir`.

Every summary result row produced by a clustering run also includes:

- `cluster_assignments`: compact JSON labels in the exact row order of the
  selected preprocessed CSV. Element `i` is the cluster for preprocessed row
  `i`, so Tiki labels can be joined to `tiki_row_mapping.csv` by row position.
- `silhouette_sample_std`: dispersion of per-sample Silhouette values. Smaller
  values mean cluster quality is more consistent across rows.
- `silhouette_negative_fraction`: fraction of rows with negative Silhouette.
  Values near zero indicate few rows are closer to another cluster.

These diagnostics deliberately remain Silhouette-only; the pipeline does not
use Davies-Bouldin or Calinski-Harabasz scores. Baseline-only rows that do not
run a new clustering keep empty assignments because their labels remain in the
selected MvDEC artifact.

The shared Silhouette contract is
`gower_numeric_asymmetric_binary_v1`: continuous `h_fused` columns use
range-normalized Gower contributions, while FP-Max binary columns use
asymmetric/Jaccard contributions. A shared `0-0` absence is ignored rather than
counted as evidence that two rows are similar. The MvDEC baseline is recomputed
with numeric Gower on `h_fused`; the artifact's Euclidean Silhouette remains an
audit value and is not used as the FFS improvement threshold.

### Native numeric-Gower baselines

Evaluate continuous K-Means, DEKM, single-view, and MvDEC baselines in their
own learned representations with numeric Gower:

```powershell
uv run mvdec-evaluate-native `
  --n-clusters 5 `
  --source "DEKM=output/dekm_seed42.pkl" `
  --source "Single-view=output/single_view_seed42.pkl" `
  --source "MvDEC=data/preprocessed_data/tiki_mvdec_fused_representation.pkl" `
  --runs-output output/tiki_v4/native_baseline_runs.csv `
  --summary-output output/tiki_v4/native_baseline_summary.csv
```

Pickle sources must be trusted local dictionaries containing `h_fused` and
`labels`; append `#KEY` to select another continuous representation key when
the matching labels are stored in the same artifact. Row-level CSV sources are
also accepted when they contain numeric representation columns plus a
`cluster`, `label`, or `kmeans_label` column. Repeat the same method name with
one artifact per seed to obtain `silhouette_mean` and
`silhouette_std_across_seeds`. MiMvDEC mixed representations remain evaluated
by `mvdec-evaluate-best` with asymmetric mixed Gower.

### Multi-seed validation and distance ablation

After the grid finishes, validate one fixed best configuration across explicit
seeds. The command selects the successful row with the largest requested score,
rebuilds its exact FP-Max feature set, and evaluates every seed with both the
primary asymmetric contract and the symmetric-Gower ablation:

```bash
uv run mvdec-evaluate-best \
  --results-path output/tiki_v4/ffs_results.csv \
  --backend kprototypes \
  --score-column final_score \
  --seeds 40 41 42 43 44 \
  --representation-path data/preprocessed_data/tiki_mvdec_fused_representation.pkl \
  --data-path data/preprocessed_data/tiki_preprocessed.csv \
  --runs-output output/tiki_v4/ffs_best_seed_runs.csv \
  --summary-output output/tiki_v4/ffs_best_seed_summary.csv \
  --mapping-path data/preprocessed_data/tiki_row_mapping.csv \
  --interpretation-seed 42 \
  --interpretation-output output/tiki_v4/ffs_best_interpretation.csv \
  --profile-output output/tiki_v4/ffs_best_cluster_profiles.csv
```

Use `--score-column silhouette_score` for without-FFS summary files and
`--score-column intuitive_score` for post-Intuitive files. Intuitive validation
also requires `--backend intuitive`; paper-initialized modes additionally use
`--init-strategy paper --strict-init`. Pass `--job-index` to validate a specific
successful row instead of selecting the maximum score.

For each distance contract, the command scores both the selected MiMvDEC labels
and the original MvDEC artifact labels on the exact same rebuilt FP-Max mixed
representation. The per-seed CSV stores
`mvdec_reference_silhouette_score`, the selected-model `silhouette_score`, and
`silhouette_delta_vs_mvdec`. The summary CSV aggregates the selected score and
delta across seeds. A positive delta means the selected clustering improves on
the fixed MvDEC assignments under the same representation and distance.
When the MvDEC artifact contains public ground-truth labels, the same files also
store `acc`, `nmi`, `mvdec_reference_acc`, `mvdec_reference_nmi`, and the two
external-metric deltas.
Feature selection remains fixed, so the variance measures final-model
stability rather than repeating model selection.

### Optional common-space robustness check

The native mixed-representation comparison above is the primary evaluation for
the FP-Max representation contribution. As an optional robustness check, the
common evaluator can also score row-ordered assignments on the same original
preprocessed numeric data:

```powershell
uv run mvdec-evaluate-common `
  --data-path data/preprocessed_data/tiki_preprocessed.csv `
  --n-clusters 5 `
  --source "MvDEC=output/TIKI_largest_eigen_selected_dimension_only_clusters.csv" `
  --source "MiMvDEC without FFS=output/tiki_v4/without_ffs_best_seed_runs.csv" `
  --source "MiMvDEC with FFS=output/tiki_v4/ffs_best_seed_runs.csv" `
  --runs-output output/tiki_v4/common_evaluation_runs.csv `
  --summary-output output/tiki_v4/common_evaluation_summary.csv
```

Repeat `--source METHOD=PATH.csv` for DEKM, single-view, and other baselines.
A source may be a row-level CSV with a `cluster` column or a multi-seed CSV
with JSON `cluster_assignments`. The command computes both
`common_numeric_euclidean_v1` and `common_numeric_gower_v1` from the same
ordered data for every method. It rejects ambiguous result grids containing
more than one assignment for the same method and seed; select and validate one
fixed configuration first with `mvdec-evaluate-best`.

Native Intuitive modes also write `*_trials.csv` sidecar files for auditing all
coarse/refine parameter trials. Summary CSV files keep only the best valid trial
per grid row or FFS candidate selection result.

The included outputs reproduce the sensitivity-analysis tables used in the
manuscript.

## Reproducibility Notes

- Public benchmark seeds are explicit; examples use consecutive seeds starting
  at `42`, and every method must receive the same list.
- Every K-Prototypes fit explicitly uses `n_init=10`; the value is recorded in
  the experiment manifest so resume cannot mix another initialization budget.
- The cached MvDEC representation contains `h_fused`, `labels`, `init`,
  `score`, and `iteration`. MvDEC 2025 artifacts additionally include
  `h_view1`, `h_view2`, `fusion_dim`, `fusion_contract`,
  `view_output_layout`, and `final_training_objective`.
- Representation learning is GPU-dependent. For the air-pollution case study,
  rerun `src/representation_learning/MVDEC_dense.py` if a fresh
  `airpollution_demvk_fused_representation.pkl` artifact is required.
- Raw Tiki storefront data collected from publicly accessible Tiki pages are
  included as `data/raw_data/tiki_raw_data.xlsx`. The preprocessing pipeline
  excludes the recovered workbook's invalid `Year Joined = 0` row and
  reproduces the canonical `tiki_preprocessed.csv`. A 1-to-1 shop mapping for
  cluster interpretation is provided as
  `data/preprocessed_data/tiki_row_mapping.csv`.

## Data and Code Availability

The code and data used in this study are available in this repository.

## Citation

If you use this repository, please cite the associated manuscript. Citation
metadata are provided in `CITATION.cff`, which GitHub can use to display a
"Cite this repository" link on the repository page.

## Archival DOI

For journal submission, create a GitHub release for the final paper version and
archive that release on Zenodo. The included `.zenodo.json` file provides
release metadata for DOI generation.

## License

This repository is released under the MIT License. See `LICENSE` for details.

