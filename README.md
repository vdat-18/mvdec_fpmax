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

```bash
uv run python -m data_preprocessing.cli \
  --input data/raw_data/tiki_raw_data.xlsx \
  --output data/preprocessed_data/tiki_preprocessed.csv
```

The repository already includes the processed dataset used in the experiments:

```text
data/preprocessed_data/tiki_preprocessed.csv
```

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
15 updates per Air Pollution cycle and 8 per TIKI cycle. The maximum remains
1400 refinement epochs, matching the former maximum number of K-Means refreshes.

`--max-refinement-epochs` is a safety cap and can be overridden. Each run logs
`stop_reason=converged_assignment` when assignment changes satisfy the tolerance,
or `stop_reason=max_epochs_reached` when the safety cap is exhausted. The
artifact stores the stop reason and completed refinement epoch count.

Refinement stops when no more than 1% of samples change their K-Means assignment
between consecutive epochs (`assignment_change_tolerance=0.01`).

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

Outputs keep only the fields needed to compare configurations: grid settings,
final score columns, selected features, cluster sizes, Intuitive parameters
when applicable, and status/error details. K-Prototypes FFS and native
Intuitive FFS outputs use `final_score` as the selected-feature score.
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

Default experiment outputs are written to:

```text
output/without_ffs_results.csv
output/ffs_results.csv
output/without_ffs_intuitive_native_results.csv
output/without_ffs_intuitive_paper_native_results.csv
output/without_ffs_intuitive_exhaustive_native_results.csv
output/without_ffs_intuitive_paper_exhaustive_native_results.csv
output/without_ffs_intuitive_view_weighted_native_results.csv
output/without_ffs_intuitive_view_weighted_paper_native_results.csv
output/without_ffs_intuitive_view_weighted_exhaustive_native_results.csv
output/without_ffs_intuitive_view_weighted_paper_exhaustive_native_results.csv
output/ffs_intuitive_native_results.csv
output/ffs_intuitive_paper_native_results.csv
output/ffs_intuitive_exhaustive_native_results.csv
output/ffs_intuitive_paper_exhaustive_native_results.csv
output/ffs_intuitive_view_weighted_native_results.csv
output/ffs_intuitive_view_weighted_paper_native_results.csv
output/ffs_intuitive_view_weighted_exhaustive_native_results.csv
output/ffs_intuitive_view_weighted_paper_exhaustive_native_results.csv
output/post_without_ffs_intuitive_results.csv
output/post_without_ffs_intuitive_best_results.csv
output/post_without_ffs_intuitive_exhaustive_best_results.csv
output/post_without_ffs_intuitive_view_weighted_results.csv
output/post_without_ffs_intuitive_view_weighted_best_results.csv
output/post_without_ffs_intuitive_view_weighted_exhaustive_best_results.csv
output/post_ffs_intuitive_results.csv
output/post_ffs_intuitive_best_results.csv
output/post_ffs_intuitive_exhaustive_best_results.csv
output/post_ffs_intuitive_view_weighted_results.csv
output/post_ffs_intuitive_view_weighted_best_results.csv
output/post_ffs_intuitive_view_weighted_exhaustive_best_results.csv
```

Native Intuitive modes also write `*_trials.csv` sidecar files for auditing all
coarse/refine parameter trials. Summary CSV files keep only the best valid trial
per grid row or FFS candidate selection result.

The included outputs reproduce the sensitivity-analysis tables used in the
manuscript.

## Reproducibility Notes

- Random seeds are fixed at `42` for local experiment code.
- The cached MvDEC representation contains `h_fused`, `labels`, `init`,
  `score`, and `iteration`. MvDEC 2025 artifacts additionally include
  `h_view1`, `h_view2`, `fusion_dim`, `fusion_contract`,
  `view_output_layout`, and `final_training_objective`.
- Representation learning is GPU-dependent. For the air-pollution case study,
  rerun `src/representation_learning/MVDEC_dense.py` if a fresh
  `airpollution_demvk_fused_representation.pkl` artifact is required.
- Raw Tiki storefront data are not fully released due to data governance
  considerations. Processed data required for reproducing the reported
  experiments are included where permitted.

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

