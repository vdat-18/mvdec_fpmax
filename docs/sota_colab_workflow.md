# SOTA Colab GPU Workflow

This document describes the reproducible GPU workflow for the current `dev`
branch. The main rule is simple: keep research logic in `.py` files, use Colab
only as the GPU runner, then download artifacts back to the local repository for
full FP-Max and clustering experiments.

## 1. Start Colab GPU

In Colab, select:

```text
Runtime -> Change runtime type -> GPU
```

Verify the GPU:

```python
!nvidia-smi
```

## 2. Clone The Dev Branch

```python
!git clone -b dev https://github.com/vdat-18/mvdec_fpmax.git
%cd mvdec_fpmax
```

If the repo already exists in the runtime:

```python
%cd /content/mvdec_fpmax
!git pull origin dev
```

## 3. Train The Tiki MvDEC Representation

Install the project dependencies and TensorFlow in the Colab environment:

```python
!pip install -q uv
!uv sync
!uv pip install tensorflow
```

Check TensorFlow GPU visibility:

```python
!uv run python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Run a smoke check first:

```python
!uv run python scripts/run_mvdec_representation.py \
  --smoke \
  --output data/preprocessed_data/fused_representation_smoke.pkl \
  --history-output data/preprocessed_data/mvdec_representation_history_smoke.csv \
  --force
```

Run the full representation-learning stage:

```python
!uv run python scripts/run_mvdec_representation.py --force
```

This writes:

```text
data/preprocessed_data/fused_representation.pkl
data/preprocessed_data/mvdec_representation_history.csv
```

The pickle preserves the downstream schema expected by the local pipeline:

```text
h_fused
labels
init
score
iteration
```

Additional metadata such as config, score history, input shape, and feature
columns may also be present, but the legacy keys above must not be removed.

## 4. Download Representation Artifacts

```python
from google.colab import files

files.download("data/preprocessed_data/fused_representation.pkl")
files.download("data/preprocessed_data/mvdec_representation_history.csv")
```

Place the downloaded pickle in the local repository at:

```text
data/preprocessed_data/fused_representation.pkl
```

Then verify locally:

```powershell
uv run mvdec-fpmax smoke
```

## 5. Run The Local MiMvDEC Pipeline

After the fresh representation is in place, run local full-grid experiments.
The six main modes are:

```powershell
uv run mvdec-fpmax without-ffs-kprototypes --workers 20 --no-resume
uv run mvdec-fpmax without-ffs-intuitive-native --workers 20 --param-workers 10 --no-resume
uv run mvdec-fpmax without-ffs-intuitive-view-weighted-native --workers 20 --param-workers 10 --no-resume
uv run mvdec-fpmax ffs-kprototypes --workers 20 --candidate-workers 10 --no-resume
uv run mvdec-fpmax ffs-intuitive-native --workers 5 --candidate-workers 10 --param-workers 5 --no-resume
uv run mvdec-fpmax ffs-intuitive-view-weighted-native --workers 5 --candidate-workers 10 --param-workers 5 --no-resume
```

## 6. Public Dataset Baselines On Colab

The public continuous-dataset K-Means baseline runs on CPU/local, but the
paper-style MvDEC baseline should run on Colab GPU.

Install PyTorch in the Colab environment if it is not already available in the
`uv` environment:

```python
!uv pip install torch
```

The DEKM/MvDEC text benchmark datasets are expected in the processed public
dataset layout:

```text
data/public/processed/reuters10k/
data/public/processed/20news/
data/public/processed/rcv1_10k/
```

If the UCI public data are not present in the Colab clone, fetch them. The
paper text datasets above are `processed_only` registry entries and should be
copied/synced as processed folders:

```python
!uv run python scripts/fetch_public_datasets.py
```

Smoke test:

```python
!uv run python scripts/run_public_mvdec_paper_baseline.py \
  --device cuda \
  --paper-strict \
  --smoke \
  --datasets reuters10k \
  --output output/public_baselines/mvdec_paper_smoke.xlsx \
  --force
```

Full DEKM/MvDEC text benchmark baseline:

```python
!uv run python scripts/run_public_mvdec_paper_baseline.py \
  --device cuda \
  --paper-strict \
  --force
```

Optional CPU K-Means baseline for the same datasets:

```python
!uv run python scripts/run_public_kmeans_baseline.py \
  --datasets reuters10k 20news rcv1_10k \
  --force
```

Download the workbook:

```python
from google.colab import files

files.download("output/public_baselines/mvdec_paper_baseline.xlsx")
```

## 7. Reproducibility Notes

- Representation learning uses TensorFlow and is GPU-dependent.
- Local MiMvDEC experiments do not require TensorFlow or PyTorch.
- `fused_representation.pkl` is the bridge between Colab and local runs.
- The fused representation must stay 11-dimensional to match
  `config.H_FUSED_COLUMNS`.
- Keep Colab notebooks as launchers only; implementation logic should remain in
  scripts/modules for reproducibility.
