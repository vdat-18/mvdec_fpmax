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
    fused_representation.pkl     # cached MvDEC fused representation
  raw_data/
    seller_store_urls.csv        # seller storefront URLs used during collection
output/
  without_ffs_results.csv
  without_ffs_results.jsonl
  ffs_results.csv
  ffs_results.jsonl
src/
  data_preprocessing/            # raw-data preprocessing utilities
  representation_learning/       # Colab GPU notebook for MvDEC representation
  pipeline/                      # FP-Max, K-Prototypes, FFS, experiments
docs/
  Clustering_English_ver02.docx  # manuscript draft
```

## Environment

The local experiment pipeline is designed for Python 3.11.

```bash
uv sync
```

TensorFlow is intentionally not included in `pyproject.toml`. The MvDEC
representation-learning stage is expected to run in Google Colab with a GPU
runtime enabled.

## Workflow

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

Run the Colab notebook below with a GPU runtime:

```text
src/representation_learning/mvdec_representation_colab.ipynb
```

The notebook reads:

```text
data/preprocessed_data/tiki_preprocessed.csv
```

and saves:

```text
data/preprocessed_data/fused_representation.pkl
```

The repository already includes the cached fused representation used by the
downstream experiments.

### 3. FP-Max and Clustering Experiments

Run a quick data and representation loading check:

```bash
uv run mvdec-fpmax smoke
```

Run smoke checks for the two experiment variants:

```bash
uv run mvdec-fpmax without-ffs-smoke
uv run mvdec-fpmax ffs-smoke
```

Run the full sensitivity grids:

```bash
uv run mvdec-fpmax without-ffs
uv run mvdec-fpmax ffs
```

Run full grids with parallel worker processes:

```bash
uv run mvdec-fpmax without-ffs --workers 4
uv run mvdec-fpmax ffs --workers 4
```

Run only the first few jobs for a quick check:

```bash
uv run mvdec-fpmax without-ffs --limit 2
uv run mvdec-fpmax ffs --limit 2
```

Grid modes run `(strategy, n_bins)` groups in parallel. Within each group,
`min_support` values are processed sequentially and saved as soon as each one
finishes. Existing output is resumed by default; add `--no-resume` to rerun the
selected jobs from scratch.

## Outputs

Default experiment outputs are written to:

```text
output/without_ffs_results.csv
output/without_ffs_results.jsonl
output/ffs_results.csv
output/ffs_results.jsonl
```

The included outputs reproduce the sensitivity-analysis tables used in the
manuscript.

## Reproducibility Notes

- Random seeds are fixed at `42` for local experiment code.
- The cached MvDEC representation contains `h_fused`, `labels`, `init`, `score`,
  and `iteration`.
- Representation learning is GPU-dependent and should be rerun through the
  Colab notebook if a fresh `fused_representation.pkl` artifact is required.
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
