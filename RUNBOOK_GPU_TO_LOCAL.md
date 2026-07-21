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
!uv add tensorflow

!uv run python src/representation_learning/MVDEC_dense.py AIRPOLLUTION \
  --runs 3 --seed 42 \
  --protocol mvdec_dekm_consistent_v1 \
  --output-dir output/mvdec_runs
```

`MPLBACKEND=Agg` avoids Kaggle/Colab notebook backend crashes when TensorFlow
imports Keras/Matplotlib. `UV_LINK_MODE=copy` avoids Kaggle hardlink warnings.

The command creates one isolated directory for each seed below
`output/mvdec_runs/airpollution/mvdec_dekm_consistent_v1/<config_hash_12>/`.
Copy the complete config directory, not a single pickle, so manifests, weights,
assignments, hashes, and logs remain auditable.

The GPU producer supports both AIRPOLLUTION and TIKI through
`src/representation_learning/MVDEC_dense.py`. Use regenerated run directories;
do not mix the checked-in legacy Tiki/Air artifacts with this protocol.

For MvDEC 2025 air-pollution artifacts, verify that:

```text
h_view1.shape == (n_samples, 10)
h_view2.shape == (n_samples, 10)
h_fused.shape == (n_samples, 10)
h_fused == (h_view1 + h_view2) / 2
view_output_layout == "eq5_compatible_10_plus_13"
final_training_objective == "mvdec_dekm_consistent_l1_reconstruction_plus_l4_greedy"
method_name == "MvDEC-DEKM-consistent"
protocol_id == "mvdec_dekm_consistent_v1"
```

The full model view output has width 23 (`10 latent + 13 reconstruction`), but
the saved `h_view1`, `h_view2`, and `h_fused` arrays contain only the 10 latent
dimensions used for clustering.

## 2. Push GPU Outputs

```bash
git add -f output/mvdec_runs
git commit -m "Add GPU MvDEC artifacts"
git push origin dev
```

## 3. Pull Locally

```powershell
git pull origin dev
uv run mvdec-audit-runs --root output/mvdec_runs
```

## 4. Verify Artifact

Air pollution:

```powershell
$REP = Get-ChildItem "output/mvdec_runs/airpollution/mvdec_dekm_consistent_v1" `
  -Filter artifact.pkl -Recurse | Where-Object { $_.Directory.Name -eq "seed_42" } |
  Select-Object -First 1 -ExpandProperty FullName

if (!$REP) { throw "Missing seed-42 air-pollution MvDEC artifact." }

uv run mvdec-fpmax smoke `
  --data-path data/preprocessed_data/data_demvk.csv `
  --representation-path $REP
```

Tiki:

```powershell
$REP = Get-ChildItem "output/mvdec_runs/tiki/mvdec_dekm_consistent_v1" `
  -Filter artifact.pkl -Recurse | Where-Object { $_.Directory.Name -eq "seed_42" } |
  Select-Object -First 1 -ExpandProperty FullName

if (!$REP) { throw "Missing seed-42 Tiki MvDEC artifact." }

uv run mvdec-fpmax smoke `
  --data-path data/preprocessed_data/tiki_preprocessed.csv `
  --representation-path $REP
```

## 5. Run 6 Local Modes

Air pollution:

```powershell
$DATA = "data/preprocessed_data/data_demvk.csv"
$REP = Get-ChildItem "output/mvdec_runs/airpollution/mvdec_dekm_consistent_v1" `
  -Filter artifact.pkl -Recurse | Where-Object { $_.Directory.Name -eq "seed_42" } |
  Select-Object -First 1 -ExpandProperty FullName
$OUTPUT = "output/airpollution"

if (!(Test-Path $REP)) {
  throw "Missing air-pollution MvDEC artifact. Generate it on GPU first."
}

uv run mvdec-fpmax without-ffs-kprototypes --data-path $DATA --representation-path $REP --output-dir $OUTPUT --workers 20 --no-resume
uv run mvdec-fpmax without-ffs-intuitive-native --data-path $DATA --representation-path $REP --output-dir $OUTPUT --workers 20 --param-workers 10 --no-resume
uv run mvdec-fpmax without-ffs-intuitive-view-weighted-native --data-path $DATA --representation-path $REP --output-dir $OUTPUT --workers 20 --param-workers 10 --no-resume
uv run mvdec-fpmax ffs-kprototypes --data-path $DATA --representation-path $REP --output-dir $OUTPUT --workers 20 --candidate-workers 10 --no-resume
uv run mvdec-fpmax ffs-intuitive-native --data-path $DATA --representation-path $REP --output-dir $OUTPUT --workers 5 --candidate-workers 10 --param-workers 5 --no-resume
uv run mvdec-fpmax ffs-intuitive-view-weighted-native --data-path $DATA --representation-path $REP --output-dir $OUTPUT --workers 5 --candidate-workers 10 --param-workers 5 --no-resume
```

Tiki: use the same commands, but set:

```powershell
$DATA = "data/preprocessed_data/tiki_preprocessed.csv"
$REP = Get-ChildItem "output/mvdec_runs/tiki/mvdec_dekm_consistent_v1" `
  -Filter artifact.pkl -Recurse | Where-Object { $_.Directory.Name -eq "seed_42" } |
  Select-Object -First 1 -ExpandProperty FullName
$OUTPUT = "output/tiki"
```
