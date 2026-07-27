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
  public/                        # reproducible public dataset workspace
  raw_data/
    seller_store_urls.csv        # seller storefront URLs used during collection
output/
  mvdec_runs/                    # protocol/config/seed-isolated GPU outputs
  tiki/fpmax_without_ffs/seed_44/ # six method-specific downstream CSV files
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

The MvDEC representation-learning stage is expected to run in a GPU runtime.
Copy the complete `output/mvdec_runs/<dataset>/<protocol>/<config>/<seed>/`
directory back for local experiments; standalone legacy pickles are rejected.

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
  --representation-path output/mvdec_runs/tiki/mvdec_dekm_consistent_v1/<config_hash_12>/seed_42/artifact.pkl \
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
  --seed 42 \
  --max-refinement-epochs 1400 \
  --protocol mvdec_dekm_consistent_v1 \
  --greedy-eigen-direction largest \
  --greedy-target-mode frozen_snapshot \
  --output-dir output/mvdec_runs
```

The primary protocol combines the MvDEC 2025 multi-view architecture with the
DEKM 2021 greedy refinement selected by Algorithm 1 and Fig. 4:

```bash
# Primary: released-DEKM frozen target in the largest/last eigen direction
uv run python src/representation_learning/MVDEC_dense.py AIRPOLLUTION \
  --runs 3 --seed 42 \
  --protocol mvdec_dekm_consistent_v1

# Explicit target-semantics ablation
uv run python src/representation_learning/MVDEC_dense.py AIRPOLLUTION \
  --runs 3 --seed 42 \
  --protocol custom \
  --greedy-eigen-direction largest \
  --greedy-target-mode selected_dimension_only
```

`mvdec_dekm_consistent_v1` is immutable: `L1=1`, `L2=0`, `L3=0`, `L4=1`,
ascending eigenvalues, the largest/last eigen direction, and the released-DEKM
`frozen_snapshot` target. DEKM 2021 reports on MNIST that optimizing this last
direction outperforms pulling every transformed dimension. Because `V` is
orthonormal, `L3` is trace-equivalent to `L2`; both remain logged diagnostics
rather than optimized terms. This is a documented hybrid protocol, not a claim
to implement MvDEC 2025 Eq. (11) literally. Any loss, smallest-eigen, or
selected-dimension-only experiment must use `--protocol custom` and is labeled
as an ablation in its artifact contract.

The largest direction is intentional. DEKM sorts eigenvalues ascending and
identifies the last/largest direction as least informative. This also matches
the MvDEC 2025 reference to the "last dimension"; its separate statement that
the smallest eigenvalue is least informative is treated as an internal paper
inconsistency rather than the repository default.

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
  --protocol mvdec_dekm_consistent_v1 \
  --dataset-root external_repos/DEKM/datasets \
  --public-output-dir output/public_benchmark/mvdec
```

For the separate MvDEC 2025 public reproduction experiment, first validate
seed 42 only:

```bash
PYTHONHASHSEED=42 uv run --no-sync python -u \
  src/representation_learning/MVDEC_dense.py REUTERS \
  --runs 1 \
  --seed 42 \
  --protocol mvdec_2025_public_reproduction_v1 \
  --progress-interval 100 \
  --dataset-root external_repos/DEKM/datasets \
  --public-output-dir output/public_benchmark/mvdec
```

This immutable protocol keeps the MvDEC architecture and encoder-average
fusion, but uses the historical/released optimization behavior: pretraining
MSE averaged over dimensions, sequential batches, K-Means refresh every 10
updates, 14,000 maximum updates, dynamic `n_init = 2 * previous_n_iter`, frozen
greedy targets, and the last/largest eigen direction. Refinement optimizes MvDEC
L1 reconstruction plus release-scaled DEKM L4 greedy loss; L2 K-Means and L3
scatter trace remain disabled. It is a documented
reproduction contract, not a claim that the internally inconsistent 2025 paper
specifies every optimization detail. Ground-truth labels are deferred until the
final independent View1/View2/fused evaluation. Do not start seeds 43/44 until
the seed-42 artifact has passed `mvdec-audit-runs` and its metrics are reviewed.
View2 places its latent bottleneck after the complete Fig. 2 skip decoder and
reconstructs the input from that bottleneck, ensuring reconstruction pretraining
updates the latent representation. The architecture identifier is persisted in
the protocol contract so artifacts from earlier View2 heads cannot mix.

The isolated output directory is
`output/public_benchmark/mvdec/reuters/mvdec_2025_public_reproduction_v1/<config_hash_12>/seed_42/`.
Its artifact and manifest retain independent View1, View2, and fused ACC/NMI;
`acc` and `nmi` remain aliases of the fused metrics for compatibility.

To diagnose whether the under-specified View2 placement causes the REUTERS gap,
run the immutable encoder-bottleneck ablation with the same seed and schedule:

```bash
PYTHONHASHSEED=42 uv run --no-sync python -u \
  src/representation_learning/MVDEC_dense.py REUTERS \
  --runs 1 \
  --seed 42 \
  --protocol mvdec_2025_view2_encoder_bottleneck_v1 \
  --progress-interval 100 \
  --dataset-root external_repos/DEKM/datasets \
  --public-output-dir output/public_benchmark/mvdec
```

This ablation moves only the 10-dimensional View2 latent to the encoder
bottleneck and decodes through the same dense skip path. It does not change
View1, preprocessing, losses, K-Means, stopping, seed, or final evaluation, and
its protocol/output identity remains separate from the reproduction run.

The completed REUTERS seed-42 diagnosis and the remaining Fig. 2/Eq. (4)
ambiguities are documented in `MVDEC_REUTERS_ARCHITECTURE_DIAGNOSIS.md`.

The next preregistered diagnostic follows the single linear View2 terminal head
drawn in Fig. 2, then reads the first latent dimensions and final reconstruction
dimensions from that one output:

```bash
PYTHONHASHSEED=42 uv run --no-sync python -u \
  src/representation_learning/MVDEC_dense.py REUTERS \
  --runs 1 \
  --seed 42 \
  --protocol mvdec_2025_view2_direct_23_split_v1 \
  --progress-interval 100 \
  --dataset-root external_repos/DEKM/datasets \
  --public-output-dir output/public_benchmark/mvdec
```

The `23` in the protocol name records the Air Pollution Fig. 2 layout
(`10 + 13`). For public 2,000-dimensional inputs the same contract creates a
`10 + 2,000` joint head and persists the resolved dimensions in the manifest.

The controlled fusion-sensitivity arm keeps that exact View2 head and training
schedule, but concatenates the two 10-dimensional latents instead of averaging
them. It has an independent protocol, config hash, output directory, and
20-dimensional fused representation:

```bash
PYTHONHASHSEED=42 uv run --no-sync python -u \
  src/representation_learning/MVDEC_dense.py REUTERS \
  --runs 1 \
  --seed 42 \
  --protocol mvdec_2025_view2_direct_23_split_concat_v1 \
  --progress-interval 100 \
  --dataset-root external_repos/DEKM/datasets \
  --public-output-dir output/public_benchmark/mvdec
```

This is a diagnostic fusion ablation, not a literal-paper reproduction. Ground
truth remains deferred to final evaluation and must not be used to select the
fusion arm as a confirmatory result.

Replace `REUTERS` with `20NEWS` or `RCV1`. Each `artifact.pkl` is directly
usable by MiMvDEC. Pass the same run directory's `assignments.csv` as
`--data-path`; the CSV is a row manifest, not a regenerated feature dataset:

```bash
uv run mvdec-fpmax ffs-kprototypes \
  --representation-path output/public_benchmark/mvdec/reuters/mvdec_dekm_consistent_v1/<config_hash_12>/seed_42/artifact.pkl \
  --data-path output/public_benchmark/mvdec/reuters/mvdec_dekm_consistent_v1/<config_hash_12>/seed_42/assignments.csv \
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
scaling to `[0, 1]`, and saves each seed under an immutable config directory:

```text
output/mvdec_runs/airpollution/mvdec_dekm_consistent_v1/<config_hash_12>/seed_42/
  manifest.json
  artifact.pkl
  assignments.csv
  pretrain_view1.weights.h5
  pretrain_view2.weights.h5
  final_view1.weights.h5
  final_view2.weights.h5
  training_log.csv
```

`run_id` contains dataset, protocol, the first 12 characters of the full config
SHA-256, and seed. Existing non-empty run directories are rejected by default;
`--force` clears only the files owned by that exact run identity. Different
seeds and custom configurations therefore cannot overwrite one another.
`mvdec_runs.csv` and `mvdec_summary.csv` live in the config directory and are
rebuilt from `status=complete` manifests; failed manifests remain available for
audit but are excluded from metric aggregation.

After copying GPU outputs, audit every output hash and reload every complete
artifact through the production loader:

```bash
uv run mvdec-audit-runs --root output/mvdec_runs
```

The loader rejects the previous legacy full-view-output artifact. Regenerate it
before running air-pollution FP-Max/clustering modes.

For DEKM-consistent MvDEC air-pollution artifacts, each model output uses the Eq.
(5)-compatible layout `10 latent + 13 reconstruction`, while clustering and
export use only the latent heads: `h_fused = (h_view1_latent +
h_view2_latent) / 2`. Therefore, `h_fused`, `h_view1`, and `h_view2` are all
10-dimensional. Primary joint training optimizes reconstruction `L1` and greedy
`L4`. K-Means `L2` and trace-equivalent `L3` are measured but have zero weight,
so the artifact objective must not be described as MvDEC 2025 Eq. (11).

The run logs both `baseline_raw_input` and `baseline_scaled_input` so scale
dominance is visible. The pickle records the scaling method, fitted per-column
minimum/maximum values, feature order, and SHA-256 of the source CSV. Min-Max is
an explicit reproduction assumption based on Table 4 of the 2025 paper; the
paper does not publish its fitted scaler.

Use the following sensitivity run to preserve the 13 source columns exactly as
stored. This does not change the primary Min-Max default, and `preprocessing`
remains part of the config hash so the artifacts cannot collide:

```bash
uv run python src/representation_learning/MVDEC_dense.py AIRPOLLUTION \
  --runs 3 --seed 42 \
  --protocol mvdec_dekm_consistent_v1 \
  --preprocessing none \
  --output-dir output/mvdec_runs_no_scaling
```

The StandardScaler sensitivity mode uses the fitted population mean and
standard deviation for each column and stores both arrays in the artifact:

```bash
uv run python src/representation_learning/MVDEC_dense.py AIRPOLLUTION \
  --runs 3 --seed 42 \
  --protocol mvdec_dekm_consistent_v1 \
  --preprocessing standard \
  --output-dir output/mvdec_runs_standard_scaling
```

Step 9 logs split `L4_greedy` into `L4_selected_direction` and
`L4_nonselected_snapshot_anchor`, plus their fractions of total L4. Their sum
equals `L4_greedy`. The primary `frozen_snapshot` mode reproduces the target
construction in the released DEKM code, so both components can be non-zero.
In the `selected_dimension_only` custom ablation, the snapshot-anchor component
and its gradient are exactly zero. The current squared-distance reduction is
recorded in the protocol contract; changing it is a separate, not-yet-applied
fidelity decision.

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
The private 1% tolerance is an intentional repository design choice, not a
value claimed from either paper. The resolved tolerance and stop reason are
stored in every new protocol artifact. Use `--assignment-change-tolerance` only
with `--protocol custom`; the primary protocol rejects a different tolerance.

New protocol artifacts are validated when loaded: the loader recomputes the
protocol-contract SHA-256 and checks the objective, loss weights, eigen mode,
target mode, and dataset-specific tolerance. CSV assignments and summaries use
`MvDEC-DEKM-consistent` as the method label while retaining `MvDEC` separately
as the algorithm family.

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
uv run mvdec-fpmax ffs-kprototypes
```

For configured private artifacts, use the seed-aware native runner. It rebuilds
the FP-Max candidate pool directly from `H_fused`; it does not reuse features
selected by K-Prototypes:

```bash
uv run mvdec-intuitive TIKI --seed 44 --source without-ffs --protocol mimvdec_intuitive_v1 --param-workers 4
uv run mvdec-intuitive TIKI --seed 44 --source ffs --protocol mimvdec_intuitive_v1 --param-workers 4
uv run mvdec-intuitive TIKI --seed 44 --source without-ffs --protocol mimvdec_intuitive_view_weighted_v1 --param-workers 4
uv run mvdec-intuitive TIKI --seed 44 --source ffs --protocol mimvdec_intuitive_view_weighted_v1 --param-workers 4
```

The default source is `ffs` and the default protocol is
`mimvdec_intuitive_v1`. `without-ffs` clusters every FP-Max feature set directly
with Intuitive. `ffs` runs Intuitive for every candidate feature set at every
forward-selection step and selects by mixed asymmetric Gower Silhouette. Its
numeric input is the unnormalized fused MvDEC latent representation; FP-Max
features are categorical binary attributes. The protocol file
`configs/intuitive_protocols.json` defines the FP-Max grid, zero FFS tolerance,
and the two-stage `mu`, `gamma`, and `beta` search space. Both K-Prototypes
and Intuitive FFS retain a feature only when its Silhouette is strictly greater
than the current score.
All six downstream results share `<output_root>/seed_<seed>/` and use distinct
method-specific filenames. Existing method output is replaced unless `--resume`
is supplied. Resume validates the protocol config,
artifact, seed, FP-Max grid, initialization, stopping settings, distance and
selection metric contracts, and the complete parameter grid through the shared
`mvdec_experiment_manifest.json`. Use only one parallel axis above 1:
`--workers`, `--param-workers`, or `--candidate-workers`.

Run full grids with parallel worker processes:

```bash
uv run mvdec-fpmax without-ffs-kprototypes --workers 4
uv run mvdec-fpmax ffs-kprototypes --workers 4 --candidate-workers 5
```

Run only the first few jobs for a quick check:

```bash
uv run mvdec-fpmax without-ffs-kprototypes --limit 2
uv run mvdec-fpmax ffs-kprototypes --limit 2
```

Grid modes run `(strategy, n_bins)` groups in parallel by default. Existing
output is resumed by default; add `--no-resume` to rerun the selected jobs
from scratch. `--workers` parallelizes outer grid groups. `--candidate-workers`
parallelizes candidate feature sets inside each FFS step for `ffs-kprototypes`.
The configured `mvdec-intuitive` runner supports either candidate-level or
parameter-level parallelism.

The configured `mvdec-intuitive` runner uses Intuitive directly as the
clustering backend for both sources. With `--source ffs`, candidate features
are selected by their best Intuitive silhouette rather than by K-Prototypes.
The `mimvdec_intuitive_view_weighted_v1` protocol is a separate ablation that
adds `alpha` inside Intuitive's clustering distance. When no FP-Max or FFS
feature survives, the native workflows keep the
baseline score and mark rows with no FP-Max features as `baseline_no_features`.
FFS-native rows that have candidate features but no improving feature are marked
as `baseline_no_improvement`. FFS-native rows that have FP-Max candidates but no
valid Intuitive candidate are marked as `baseline_no_valid_candidate`.

Outputs keep the fields needed to compare and interpret configurations: grid
settings, final score columns, selected features, cluster sizes, row-ordered
cluster assignments, Intuitive parameters when applicable, and status/error
details. K-Prototypes FFS and native Intuitive FFS outputs use `final_score` as
the selected-feature score. On Windows, prefer one inner parallelism axis at a time. For native
FFS Intuitive modes, either use `--candidate-workers > 1 --param-workers 1` or
use `--candidate-workers 1 --param-workers > 1`; the CLI rejects using both
above 1 because that would create nested multiprocessing. A practical resume
command is:

```bash
uv run mvdec-intuitive TIKI --seed 44 --source ffs --param-workers 4 --resume
```

Native Intuitive modes use deterministic `farthest_first` initialization.
Runtime failures that are not algorithm-defined statuses are not persisted;
those missing `job_index` rows are retried on the next resume run. CSV writes
are atomic, so a crash during save keeps either the previous complete CSV or
the new complete CSV. FP-Max itemset feature names are sorted deterministically
before FFS so repeated runs keep stable candidate order.

The Intuitive parameter grid follows the current two-stage tuning policy used
consistently across Intuitive modes:

```text
coarse mu_param = 0.2, 0.5, 0.8
coarse gamma    = 0.2, 0.5, 0.8
beta            = 2.0, 3.0, 4.0, 5.0
refine          = mu_param/gamma +/- 0.1 around the best coarse trial
```

`beta` is not refined. The full candidate range supplies the neighboring values
used by the refinement stage; it is not searched exhaustively. If every coarse
trial fails, the configuration is marked failed instead of falling back to the
full grid:

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

Configured downstream outputs share one seed directory. Filenames identify the
clustering backend, view-weighting ablation, and FFS setting:

```text
<output_root>/seed_<seed>/
  mimvdec_without_ffs_results.csv
  mimvdec_with_ffs_results.csv
  mimvdec_intuitive_without_ffs_results.csv
  mimvdec_intuitive_with_ffs_results.csv
  mimvdec_intuitive_view_weighted_without_ffs_results.csv
  mimvdec_intuitive_view_weighted_with_ffs_results.csv
  mvdec_experiment_manifest.json
```

The manifest records the dataset, data hash, artifact hash, cluster count,
random seed, evaluation-distance contract, and both versioned Intuitive
contracts. Resume is rejected if any of these differ.

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
  --source "MvDEC-DEKM-consistent=output/mvdec_runs/tiki/mvdec_dekm_consistent_v1/<config_hash_12>/seed_42/artifact.pkl" `
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
  --results-path output/tiki_v4/mimvdec_with_ffs_results.csv \
  --backend kprototypes \
  --score-column final_score \
  --seeds 40 41 42 43 44 \
  --representation-path output/mvdec_runs/tiki/mvdec_dekm_consistent_v1/<config_hash_12>/seed_42/artifact.pkl \
  --data-path data/preprocessed_data/tiki_preprocessed.csv \
  --runs-output output/tiki_v4/ffs_best_seed_runs.csv \
  --summary-output output/tiki_v4/ffs_best_seed_summary.csv \
  --mapping-path data/preprocessed_data/tiki_row_mapping.csv \
  --interpretation-seed 42 \
  --interpretation-output output/tiki_v4/ffs_best_interpretation.csv \
  --profile-output output/tiki_v4/ffs_best_cluster_profiles.csv
```

Use `--score-column silhouette_score` for without-FFS Intuitive files and
`--score-column final_score` for FFS Intuitive files. Intuitive validation also
requires `--backend intuitive`. Pass `--job-index` to validate a specific
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
  --source "MvDEC-DEKM-consistent=output/TIKI_mvdec_dekm_consistent_v1_largest_eigen_frozen_snapshot_clusters.csv" `
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

Native Intuitive result CSV files keep the best valid two-stage trial per grid
row or FFS candidate selection result, including the selected parameters,
assignments, phi values, and final weights.

Paper-ready MvDEC outputs are not bundled with the repository. Regenerate the
required Air Pollution and Tiki runs under `output/mvdec_runs/` and audit them
before reproducing the manuscript tables.

## Reproducibility Notes

- Public benchmark seeds are explicit; examples use consecutive seeds starting
  at `42`, and every method must receive the same list.
- Every K-Prototypes fit explicitly uses `n_init=10`; the value is recorded in
  the experiment manifest so resume cannot mix another initialization budget.
- Each paper-ready MvDEC run stores `artifact.pkl`, assignments, weights, log,
  output hashes, and a complete `manifest.json`. The loader verifies the run
  manifest before exposing `h_fused` or labels.
- Representation learning is GPU-dependent. For the air-pollution case study,
  rerun `src/representation_learning/MVDEC_dense.py`; no pre-C1 Air/Tiki MvDEC
  pickle is retained as a fallback.
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

