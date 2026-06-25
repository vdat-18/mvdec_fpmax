# Public Experiment Datasets

This folder is the local workspace for the selected public continuous-feature datasets used to stress-test the MvDEC + FP-Max integration pipeline.

The registry intentionally keeps only datasets whose input variables are treated as continuous measurements for the current experiment design:

```text
uci_gas_sensor_drift
uci_sensorless_drive
uci_wine_quality
uci_seeds
uci_wine
uci_wdbc
uci_sonar
uci_parkinsons
uci_glass
uci_libras
```

The raw and processed datasets are intentionally ignored by git:

```text
data/public/raw/
data/public/processed/
```

Recreate them with:

```bash
uv run python scripts/fetch_public_datasets.py
```

Each processed dataset is written as:

```text
data/public/processed/<dataset_name>/
  X.csv          # numeric continuous input features only
  y.csv          # labels/sample metadata for external evaluation only, if available
  views.json     # natural or constructed feature views
  metadata.json  # source URL, parser, size, cleaning notes, n_clusters
```

`X.csv` is the only file that should be used as clustering input. `y.csv` must not be used by the clustering pipeline; it exists only for external metrics such as ARI/NMI/ACC or for dataset auditing.

The dataset registry lives in:

```text
configs/public_datasets.json
```

Useful commands:

```bash
# Fetch and process every selected dataset.
uv run python scripts/fetch_public_datasets.py

# Fetch a small subset first.
uv run python scripts/fetch_public_datasets.py --datasets uci_seeds uci_wine uci_sonar

# Redownload and rebuild processed files.
uv run python scripts/fetch_public_datasets.py --force

# Keep raw downloads only.
uv run python scripts/fetch_public_datasets.py --raw-only
```
