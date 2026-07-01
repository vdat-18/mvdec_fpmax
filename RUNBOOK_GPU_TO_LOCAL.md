# GPU-To-Local Runbook

## 1. Run On Colab

```python
%cd /content
!rm -rf mvdec_fpmax

!git clone -b dev https://github.com/vdat-18/mvdec_fpmax.git
%cd /content/mvdec_fpmax

%env MPLBACKEND=Agg
%env UV_LINK_MODE=copy

!pip install -q uv
!uv sync
!uv add torch tensorflow

!uv run python scripts/run_colab_gpu_pipeline.py \
  --datasets airpollution_demvk \
  --tasks all \
  --force
```

`MPLBACKEND=Agg` avoids Kaggle/Colab notebook backend crashes when TensorFlow
imports Keras/Matplotlib. `UV_LINK_MODE=copy` avoids Kaggle hardlink warnings.

For Tiki, replace the dataset name:

```python
!uv run python scripts/run_colab_gpu_pipeline.py \
  --datasets tiki \
  --tasks all \
  --force
```

## 2. Push GPU Outputs

```bash
git add data/preprocessed_data/*_fused_representation.pkl \
  data/preprocessed_data/*_mvdec_history.csv \
  output/public_baselines/mvdec_gpu_pipeline.xlsx
git commit -m "Add GPU MvDEC artifacts"
git push origin dev
```

## 3. Pull Locally

```powershell
git pull origin dev
```

## 4. Verify Artifact

Air pollution:

```powershell
uv run mvdec-fpmax smoke `
  --data-path data/preprocessed_data/data_demvk.csv `
  --representation-path data/preprocessed_data/airpollution_demvk_fused_representation.pkl
```

Tiki:

```powershell
uv run mvdec-fpmax smoke `
  --data-path data/preprocessed_data/tiki_preprocessed.csv `
  --representation-path data/preprocessed_data/tiki_fused_representation.pkl
```

## 5. Run 6 Local Modes

Air pollution:

```powershell
$DATA = "data/preprocessed_data/data_demvk.csv"
$REP = "data/preprocessed_data/airpollution_demvk_fused_representation.pkl"

uv run mvdec-fpmax without-ffs-kprototypes --data-path $DATA --representation-path $REP --workers 20 --no-resume
uv run mvdec-fpmax without-ffs-intuitive-native --data-path $DATA --representation-path $REP --workers 20 --param-workers 10 --no-resume
uv run mvdec-fpmax without-ffs-intuitive-view-weighted-native --data-path $DATA --representation-path $REP --workers 20 --param-workers 10 --no-resume
uv run mvdec-fpmax ffs-kprototypes --data-path $DATA --representation-path $REP --workers 20 --candidate-workers 10 --no-resume
uv run mvdec-fpmax ffs-intuitive-native --data-path $DATA --representation-path $REP --workers 5 --candidate-workers 10 --param-workers 5 --no-resume
uv run mvdec-fpmax ffs-intuitive-view-weighted-native --data-path $DATA --representation-path $REP --workers 5 --candidate-workers 10 --param-workers 5 --no-resume
```

Tiki: use the same commands, but set:

```powershell
$DATA = "data/preprocessed_data/tiki_preprocessed.csv"
$REP = "data/preprocessed_data/tiki_fused_representation.pkl"
```
