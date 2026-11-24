# FP-Max Study Runner (Step 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One runner that evaluates algorithm × FP-Max arm × seed on any dataset and
writes long-format CSV, so Step 3 only has to point it at data and Step 4 only has
to read the CSV.

**Architecture:**
- A `StudyData` object (numeric block, original categorical block, labels, reference
  data) is the single input for all backbones. Loaders exist for raw UCI tables,
  DEKM embeddings and unlabeled applications.
- A pure arm-fitting layer (`kmeans`, `kprototypes`, `intuitive`) takes an optional
  FP-Max itemset block.
- A study FFS selects itemsets, and a runner pre-selects the FP-Max configuration on
  separate seeds, then reports on report seeds.
- The legacy MiMvDEC job runners in `src/pipeline/experiments.py` are left untouched;
  retiring them is deferred.

**Tech Stack:** Python 3.12 via `uv run --python 3.12 --link-mode=copy`, pandas, numpy,
scikit-learn, kmodes, mlxtend, gower, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-11-24-fpmax-study-design.md`. Builds on
`docs/superpowers/plans/2026-11-24-fpmax-foundation.md` (done).

## Global Constraints

- Shell commands: `uv run --python 3.12 --link-mode=copy python -m pytest ...`
  (`uv run pytest` is blocked). Ignore `tests/test_representation_utils.py` and
  `tests/test_airpollution_no2.py` (missing optional dependencies, pre-existing).
- True labels are read only inside the evaluation step of the runner. Every fit,
  FP-Max, selection and FFS function must work when `StudyData.labels is None`; tests
  pass `labels=None` to prove it.
- Reported silhouette, Davies-Bouldin and Calinski-Harabasz come only from
  `reference_evaluation`. FFS may use its own candidate-space silhouette internally
  and never reports it as quality.
- Intuitive options are identical for the `none`, `fpmax` and `fpmax_ffs` arms of one
  dataset: `mu_convention="swapped"`, `init_strategy="farthest_first"`, 5 restarts
  chosen by lowest assignment cost, `max_iter=100`, per-dataset `beta`, `mu`, `gamma`
  from the table in Task 1. No per-arm tuning.
- FP-Max grid: strategies `uniform`, `quantile` (not `kmeans`: its binning is
  random); `n_bins` 3 and 5; `min_support` 0.4, 0.3, 0.2; at most 60 itemsets per
  configuration. A configuration over the cap or with no itemsets is recorded with
  status `too_many_itemsets` or `no_itemsets`, never silently dropped.
- Seeds: selection seeds `[100, 101, 102]`, report seeds `[0..9]`. Selection seeds
  never appear in report rows.
- FFS: `min_improvement = 0.005` (`forward_selection.STUDY_FFS_MIN_IMPROVEMENT`),
  criterion computed on at most 1500 rows (fixed subsample).
- Failed fits become rows with a `status` and an `error` message. Only `RuntimeError`
  and `ValueError` raised by a model fit are caught; any other exception propagates.
- Do not commit `paper_replication/intuitive_kprototypes/data/`. Commit with
  `git -c user.name="ngvdat.w" -c user.email="ngvdat.w@gmail.com" commit` and end each
  message with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Categorical-only datasets** (BCW, Soybean: 0 numeric columns). `KPrototypes`
   asserts at least one numeric column, so `kprototypes` must fall back to
   `KModes`, and `kmeans` must be reported as `not_applicable` (Task 4, Task 6 tests).
2. **FP-Max returns nothing or too much.** No itemsets, or more than the cap: the row
   exists with the status, and the arm is skipped instead of crashing (Task 5, Task 8).
3. **Text backbones.** Reference space is cosine on the original tf-idf rows,
   subsampled to 3000, and labels still cover every row (Task 2, Task 3 tests).
4. **Selection leakage across seeds.** Changing report-seed results must not change
   which FP-Max configuration is chosen (Task 8 test).
5. **Zero features selected by FFS.** `fpmax_ffs` then equals `none` exactly and says
   `n_selected = 0`, rather than reporting a made-up arm (Task 7 test).

---

### Task 1: Study config

**Files:**
- Create: `configs/fpmax_study.json`, `src/pipeline/study_config.py`
- Test: `tests/test_study_config.py`

**Interfaces:**
- Produces:
  - `load_study_config(path: Path) -> StudyConfig`, a frozen dataclass with
    `selection_seeds: tuple[int, ...]`, `report_seeds: tuple[int, ...]`,
    `fpmax_strategies`, `fpmax_n_bins`, `fpmax_min_supports`, `max_itemsets: int`,
    `ffs_min_improvement: float`, `ffs_max_rows: int`, `kprototypes_init_methods`,
    `intuitive: IntuitiveStudyParams`, `uci_data_dir: str`, `uci_datasets:
    tuple[str, ...]`, `applications: dict[str, ApplicationSpec]`.
  - `IntuitiveStudyParams.for_dataset(name: str) -> DatasetIntuitiveParams`
    (`beta`, `mu`, `gamma`; unknown names get `defaults`), plus shared
    `mu_convention`, `init_strategy`, `restarts`, `max_iter`.
  - `ApplicationSpec(csv_path: str, n_clusters: int)`.
- The JSON uses these per-dataset values (`beta`, `mu`, `gamma`): iris 2.0, 0.5, 0.0;
  ionosphere 1.05, 0.5, 0.0; bcw 8.0, 0.2, 0.4; soybean_small_no_constants 8.0, 0.2,
  0.5; australian_credit_approval 50.0, 0.2, 0.7; heart_cleveland_2 20.0, 0.3, 0.2;
  chess_ten_fifteen 1.25, 0.3, 0.0; heart_cleveland_5 2.0, 0.44, 0.5. Defaults for
  datasets without an entry: 2.0, 0.5, 0.5. Applications: `tiki`
  (`data/preprocessed_data/tiki_preprocessed.csv`, k = 5), `airpollution`
  (`data/preprocessed_data/data_demvk.csv`, k = 4). `uci_data_dir` is
  `paper_replication/intuitive_kprototypes/data/uci`.

- [ ] **Step 1: Write failing tests.**
  - `test_default_config_loads_and_matches_constraints`: loads `configs/fpmax_study.json`;
    asserts `selection_seeds == (100, 101, 102)`, `report_seeds == tuple(range(10))`,
    `max_itemsets == 60`, `ffs_min_improvement == 0.005`, `ffs_max_rows == 1500`,
    `intuitive.mu_convention == "swapped"`, `intuitive.restarts == 5`,
    `intuitive.for_dataset("heart_cleveland_5").mu == 0.44`,
    `intuitive.for_dataset("unseen").beta == 2.0`.
  - `test_seed_sets_must_be_disjoint`: a config whose selection and report seeds
    overlap raises `ValueError`.
  - `test_unknown_or_missing_fields_rejected`: an extra key raises `ValueError`.
  - `test_strategy_kmeans_is_rejected`: `fpmax_strategies=["kmeans"]` raises
    `ValueError` (non-deterministic binning).
- [ ] **Step 2: Run to verify they fail** (`ImportError`).
- [ ] **Step 3: Implement** the loader with strict field checking in the style of
  `src/pipeline/intuitive_protocol.py:94-127`; write the JSON.
- [ ] **Step 4: Run tests; expect PASS.**
- [ ] **Step 5: Commit** "Add the FP-Max study config".

---

### Task 2: Reference space for text and a shared Gower helper

**Files:**
- Modify: `src/pipeline/reference_evaluation.py`
- Test: `tests/test_reference_evaluation.py`

**Interfaces:**
- Produces:
  - `gower_distance_matrix(df: pd.DataFrame) -> np.ndarray`: the string-dtype cast
    plus `gower.gower_matrix` currently inline in `build_reference_distance`,
    extracted.
  - `build_reference_distance(df, max_rows=3000, random_state=0, metric="gower")`
    where `metric in {"gower", "cosine"}`; `"cosine"` uses
    `sklearn.metrics.pairwise.cosine_distances` on a numeric `df`; any other value
    raises `ValueError`.

- [ ] **Step 1: Write failing tests.**
  - `test_cosine_reference_space_ignores_row_scale`: two rows `[1, 0]` and `[5, 0]`
    have distance 0; `[1, 0]` and `[0, 1]` have distance 1.
  - `test_cosine_space_subsamples_and_scores_full_labelling`: 400 rows, 2 well
    separated directions, `max_rows=100`; `score_internal(space, labels).silhouette > 0.5`.
  - `test_unknown_metric_is_rejected`.
  - `test_gower_distance_matrix_accepts_string_columns` (extension string dtype).
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement.** Keep `build_reference_distance(df)` behaviour unchanged
  for the default metric.
- [ ] **Step 4: Run** `tests/test_reference_evaluation.py`; expect PASS.
- [ ] **Step 5: Commit** "Add a cosine reference space for text backbones".

---

### Task 3: StudyData and loaders

**Files:**
- Create: `src/pipeline/study_data.py`
- Test: `tests/test_study_data.py`

**Interfaces:**
- Consumes: `clustering.minmax_scale_block`; `public_data.load_dekm_public_dataset`.
- Produces:
  - `StudyData` frozen dataclass: `name: str`, `backbone: str` (`"raw"` or `"dekm"`),
    `numeric: pd.DataFrame` (min-max scaled, may have 0 columns), `categorical:
    pd.DataFrame` (original attributes as `str`, may have 0 columns),
    `labels: np.ndarray | None`, `n_clusters: int`, `reference_df: pd.DataFrame`,
    `reference_metric: str`. All frames use the index `0..n-1`.
  - `load_uci_study_data(key: str, data_dir: Path) -> StudyData`: reads
    `data_dir/manifest.json` (`processed_csv`, `numeric_cols`, `categorical_cols`,
    `target_col`, `n_clusters`); `reference_df` is numeric (scaled) plus categorical;
    metric `"gower"`.
  - `load_application_study_data(name: str, csv_path: Path, n_clusters: int) -> StudyData`:
    numeric-only CSV, `labels=None`, reference metric `"gower"`.
  - `load_dekm_study_data(artifact_path: Path, dataset_name: str, dataset_root: Path,
    twenty_news_source: str = TWENTY_NEWS_SKLEARN_SOURCE) -> StudyData`: numeric = the artifact's `"embedding"`
    (min-max scaled), labels = the artifact's `"true_labels"` (the canonical 18,846-row
    20NEWS source is the default, see the audit); `reference_df` = the
    original tf-idf matrix from `load_dekm_public_dataset`; metric `"cosine"`. Raises
    `ValueError` if the artifact and dataset disagree on row count or `source_sha256`.

- [ ] **Step 1: Write failing tests** (tmp files; for DEKM, monkeypatch
  `study_data.load_dekm_public_dataset` and write a small pickle artifact):
  - `test_uci_mixed_dataset_splits_and_scales`: numeric columns in `[0, 1]`,
    categorical columns are `str`, `labels` has one entry per row, `reference_df` has
    numeric plus categorical columns.
  - `test_uci_categorical_only_has_empty_numeric_block`: `numeric.shape[1] == 0`.
  - `test_application_data_is_unlabeled`: `labels is None`, all values in `[0, 1]`.
  - `test_dekm_loader_uses_embedding_and_cosine_reference`.
  - `test_dekm_loader_rejects_source_mismatch` and `..._row_count_mismatch`.
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement** the dataclass and three loaders.
- [ ] **Step 4: Run tests; expect PASS.**
- [ ] **Step 5: Commit** "Add StudyData and loaders for UCI, DEKM and application data".

---

### Task 4: Shared K-Prototypes fit (refactor)

**Files:**
- Modify: `src/pipeline/clustering.py` (`run_kprototypes`)
- Test: `tests/test_clustering.py` (existing tests stay green)

**Interfaces:**
- Produces: `fit_kprototypes_by_cost(combined_df: pd.DataFrame, categorical_idx:
  list[int], numeric_cols: list[str], n_clusters: int, init_methods: tuple[str, ...],
  random_state: int, verbose: bool = False) -> KPrototypesFit`
  - `KPrototypesFit` fields: `labels`, `init`, `cost`, `fit_seconds`, `backend`.
  - Backends: `"kprototypes"` when both blocks exist, `"kmodes"` when there are no
    numeric columns, `"kmeans_equivalent"` when there are no categorical columns.
  - Restart choice by lowest cost among valid partitions; the `could not initialize`
    skip; `RuntimeError("No valid clustering result was produced.")` when none is valid.
- `run_kprototypes` calls it, then computes silhouette diagnostics as before. Its
  public signature and results do not change.

- [ ] **Step 1: Write failing tests.**
  - `test_categorical_only_uses_kmodes`: three-attribute string data in two clear
    groups; `backend == "kmodes"` and the partition is valid.
  - `test_fit_kprototypes_by_cost_matches_run_kprototypes_labels`: same labels and
    init as `run_kprototypes` on a blobs fixture.
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement** by moving the restart loop out of `run_kprototypes`
  unchanged, and adding the `KModes` branch (`kmodes.kmodes.KModes`, `init` taken
  from `init_methods`, `n_init=KPROTOTYPES_N_INIT`).
- [ ] **Step 4: Run** `tests/test_clustering.py tests/test_clustering_diagnostics.py
  tests/test_forward_selection.py`; expect PASS with no existing test edited.
- [ ] **Step 5: Commit** "Extract the cost-based K-Prototypes fit and add the K-Modes backend".

---

### Task 5: FP-Max itemset block

**Files:**
- Create: `src/pipeline/study_itemsets.py`
- Test: `tests/test_study_itemsets.py`

**Interfaces:**
- Consumes: `fpmax.prepare_fpmax_data`, `fpmax.extract_fpmax_features_from_prepared`.
- Produces:
  - `FpmaxSetting(strategy: str, n_bins: int, min_support: float)` (frozen).
  - `ItemsetBlock(columns: pd.DataFrame, status: str, n_itemsets: int)`; status is
    `"ok"`, `"no_itemsets"` or `"too_many_itemsets"`.
  - `build_itemset_block(data: StudyData, setting: FpmaxSetting, max_itemsets: int) ->
    ItemsetBlock`: runs FP-Max on `numeric` (discretized) together with `categorical`
    (one-hot). The block holds only itemset indicator columns (int 0/1), named with
    the prefix `fp|`, with constant columns removed. Over the cap or empty yields an
    empty frame (0 columns) and the matching status. Works with `labels=None`.

- [ ] **Step 1: Write failing tests** on a small fixture (two numeric columns, one
  categorical column, 40 rows):
  - `test_block_has_only_binary_itemset_columns_with_prefix`.
  - `test_constant_itemset_columns_are_dropped` (support 1.0 item).
  - `test_too_many_itemsets_is_reported_not_raised`: `max_itemsets=1`.
  - `test_no_itemsets_status_at_unreachable_support`.
  - `test_block_is_deterministic`: two calls give equal frames.
  - `test_categorical_only_data_is_supported` (0 numeric columns).
  - `test_block_ignores_labels`: `labels=None` and real labels give equal blocks.
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement.** Combine `numeric` and `categorical` into one frame, run
  FP-Max, keep columns that are not original attributes.
- [ ] **Step 4: Run tests; expect PASS.**
- [ ] **Step 5: Commit** "Add the FP-Max itemset block for study data".

---

### Task 6: Arm fitting

**Files:**
- Create: `src/pipeline/study_arms.py`
- Test: `tests/test_study_arms.py`

**Interfaces:**
- Consumes: `fit_kprototypes_by_cost` (Task 4), `IntuitiveKPrototypes`,
  `is_valid_partition`, `StudyConfig` (Task 1), `StudyData` (Task 3).
- Produces:
  - `ArmFit(labels: np.ndarray | None, status: str, cost: float, error: str | None,
    fit_seconds: float, init: str | None)`; status is `"ok"`, `"invalid_partition"`,
    `"error"` or `"not_applicable"`.
  - `fit_arm(data: StudyData, algorithm: str, itemsets: pd.DataFrame | None, seed:
    int, study: StudyConfig) -> ArmFit` with `algorithm in {"kmeans", "kprototypes",
    "intuitive"}`.
    - `kmeans`: numeric block only; `not_applicable` when there are 0 numeric columns;
      ignores `itemsets`.
    - `kprototypes`: numeric, categorical and itemset columns; categorical as `str`.
    - `intuitive`: `X_num` is the numeric block, `X_cat` is categorical plus itemset
      columns as `str`; parameters from `study.intuitive.for_dataset(data.name)`;
      restart `r` uses `random_state = seed * 1000 + r`; the restart with the lowest
      assignment cost (`result.distances[arange, labels].sum()`) among valid
      partitions is kept.
    - Only `RuntimeError` and `ValueError` from the fit become `status="error"` with
      the message in `error`.

- [ ] **Step 1: Write failing tests.**
  - `test_kmeans_not_applicable_without_numeric_columns`.
  - `test_kprototypes_categorical_only_runs_via_kmodes`.
  - `test_intuitive_uses_study_params_and_swapped_mu`: monkeypatch
    `study_arms.IntuitiveKPrototypes` to record kwargs; assert `mu_convention ==
    "swapped"`, `init_strategy == "farthest_first"`, and the dataset's `beta`.
  - `test_intuitive_keeps_lowest_cost_valid_restart`: scripted fake model.
  - `test_itemsets_change_the_intuitive_categorical_block`: the recorded `X_cat` has
    `categorical.shape[1] + itemsets.shape[1]` columns.
  - `test_fit_error_is_recorded_not_raised` (`ValueError` from the fit) and
    `test_unexpected_exception_propagates` (`TypeError`).
  - `test_fit_arm_works_without_labels` (`StudyData.labels is None`).
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests; expect PASS.**
- [ ] **Step 5: Commit** "Add arm fitting for kmeans, kprototypes and intuitive".

---

### Task 7: Study forward selection

**Files:**
- Create: `src/pipeline/study_ffs.py`
- Test: `tests/test_study_ffs.py`

**Interfaces:**
- Consumes: `fit_arm`, `gower_distance_matrix` (Task 2), `StudyConfig`.
- Produces:
  - `candidate_silhouette(data: StudyData, itemsets: pd.DataFrame, labels:
    np.ndarray, max_rows: int) -> float`: silhouette of `labels` under Gower on
    numeric, categorical and the given itemset columns, on a fixed subsample
    (`random_state=0`) of at most `max_rows` rows; `nan` for an invalid partition.
  - `FfsResult(selected: list[str], labels: np.ndarray | None, n_candidates_evaluated:
    int, final_score: float)`.
  - `forward_select_itemsets(data, algorithm, itemsets, seed, study, baseline_labels)
    -> FfsResult`:
    - The baseline score is `candidate_silhouette` of `baseline_labels` with no
      itemsets.
    - Each step fits every remaining itemset added to the selected ones through
      `fit_arm`.
    - The best candidate is kept only if its score exceeds the current score by more
      than `study.ffs_min_improvement`; invalid or failed candidates never win.
    - With nothing selected, `labels` equals `baseline_labels` and `selected == []`.

- [ ] **Step 1: Write failing tests** (monkeypatch `study_ffs.fit_arm` with scripted
  labels):
  - `test_ffs_stops_when_gain_is_below_threshold`.
  - `test_ffs_skips_invalid_and_error_candidates`.
  - `test_ffs_with_no_improvement_returns_baseline_labels` (Review Focus 5).
  - `test_ffs_counts_every_candidate_fit`.
  - `test_candidate_silhouette_nan_for_singleton_partition`.
  - `test_candidate_silhouette_subsample_is_deterministic`.
  - `test_ffs_runs_without_labels`.
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests; expect PASS.**
- [ ] **Step 5: Commit** "Add the study forward selection over FP-Max itemsets".

---

### Task 8: Study runner and CLI script

**Files:**
- Create: `src/pipeline/study_runner.py`, `scripts/run_fpmax_study.py`
- Test: `tests/test_study_runner.py`

**Interfaces:**
- Consumes: everything above, `reference_evaluation`, `external_metrics`.
- Produces:
  - `STUDY_COLUMNS: tuple[str, ...]` = `dataset, backbone, algorithm, arm, phase, seed,
    strategy, n_bins, min_support, n_itemsets, n_selected, status, error, acc, nmi, ari,
    ref_silhouette, ref_davies_bouldin, ref_calinski_harabasz, ffs_candidates,
    fit_seconds, labels_hash`.
  - `select_fpmax_setting(data, algorithm, study, reference) -> SettingChoice`
    (`setting: FpmaxSetting | None`, `rows: list[dict]`): for each grid setting, build
    the block, fit the `fpmax` arm on the selection seeds, take the mean reference
    silhouette of the valid fits, and pick the maximum (ties: grid order). Emits one
    row per setting and seed with `phase="select"`. Returns `setting=None` when no
    setting has a valid fit.
  - `run_group(data: StudyData, algorithm: str, study: StudyConfig, out_dir: Path) ->
    Path`: writes `<out_dir>/<dataset>__<algorithm>.csv` (`STUDY_COLUMNS`) and
    `<out_dir>/<dataset>__<algorithm>.labels.npz` (key `"<arm>|<seed>"` for every
    valid partition).
    - Arms are `none`, `fpmax`, `fpmax_ffs` on every report seed.
    - `kmeans` gets only `none`.
    - If no setting was chosen, the `fpmax` and `fpmax_ffs` rows carry status
      `no_valid_fpmax_setting`.
    - Metrics: ACC, NMI, ARI only when `data.labels` is not `None` and the fit is
      valid (`allow_missing_clusters=True`); the reference scores come from
      `score_internal` on `build_reference_distance(data.reference_df, metric=...)`
      built once per group.
  - `scripts/run_fpmax_study.py`: `--config`, `--dataset` (a UCI key or application
    name), `--algorithm`, `--out-dir`, `--report-seeds` (optional cap for smoke runs);
    builds `StudyData` from the config; prints the output path. One process per
    dataset and algorithm, so a group can be sharded across background processes.

- [ ] **Step 1: Write failing tests** (small synthetic `StudyData`, monkeypatched
  `fit_arm` where speed matters):
  - `test_group_writes_all_arms_for_every_report_seed_and_no_selection_seed_rows_in_report`.
  - `test_kmeans_group_has_only_the_none_arm`.
  - `test_setting_choice_ignores_report_seed_results` (Review Focus 4): fake fits
    return different labels on report seeds; the chosen setting is unchanged.
  - `test_failed_fit_is_a_row_with_status_and_error` (no missing rows).
  - `test_no_valid_setting_marks_fpmax_arms`.
  - `test_unlabeled_data_has_nan_external_metrics_but_reference_scores`.
  - `test_labels_npz_contains_valid_partitions_only`.
  - `test_setting_selection_runs_without_labels`.
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement** the runner and the script.
- [ ] **Step 4: Run the full suite** (`tests`, ignoring the two known files); expect
  PASS.
- [ ] **Step 5: Commit** "Add the FP-Max study runner and CLI script".

---

### Task 9: Smoke acceptance — DONE (2026-11-25)

Ran `iris` × {`intuitive`, `kprototypes`, `kmeans`} and `bcw` × {`intuitive`,
`kprototypes`} with `--report-seeds 2`, one background process per group (each group
finished in under 100 s). All rows have status `ok` or an explicit non-`ok` status
(`no_itemsets` for settings with nothing to mine); no row is missing.

| group | arm | ACC | ARI | NMI | reference silhouette | itemsets / selected |
|---|---|---|---|---|---|---|
| iris intuitive | none | 0.960 | 0.886 | 0.864 | 0.516 | |
| iris intuitive | fpmax | 0.880 | 0.702 | 0.719 | 0.517 | 1 itemset |
| iris intuitive | fpmax_ffs | 0.960 | 0.886 | 0.864 | 0.516 | 0 selected (= none) |
| iris kprototypes | none / fpmax / fpmax_ffs | 0.887 | 0.716 | 0.742 | 0.533 | 1 itemset, 0 selected |
| iris kmeans | none | 0.887 | 0.716 | 0.742 | 0.533 | |
| bcw intuitive | none | 0.950 | 0.807 | 0.697 | 0.408 | |
| bcw intuitive | fpmax / fpmax_ffs | 0.933 | 0.745 | 0.632 | 0.405 | 1 itemset, 1 selected |
| bcw kprototypes | none | 0.927 | 0.725 | 0.622 | 0.398 | KModes backend |
| bcw kprototypes | fpmax / fpmax_ffs | 0.934 | 0.751 | 0.633 | 0.403 | 1 itemset, 1 selected |

Acceptance checks: Iris and BCW Intuitive `none` equal the paper's select-row values
(0.960 / 0.886 / 0.864 and 0.950 / 0.807 / 0.697), well inside the 0.05 tolerance;
BCW `kprototypes` runs through `KModes`; every group has all arms on both seeds plus
`phase="select"` rows.

**Findings.**
- **Selection favours the least intrusive setting.** The setting is chosen by mean
  reference silhouette, and the reference space does not contain itemsets. A setting
  with 1 itemset barely changes the clustering and keeps the score; settings with
  more itemsets lower it (BCW: 0.405 with 1 itemset, 0.301 with 15; K-Prototypes down
  to -0.010). So the chosen `fpmax` arm drifts toward `none` by construction. This is
  not a bug but it limits what the study can show. Options for the Step 3 plan:
  report every grid setting as its own arm (the select rows already hold them), or
  pre-register one fixed default setting per dataset family, or select with a
  different label-free criterion. Not changed here.
- **FFS often selects nothing** (Iris: 0 of 1 itemset), so `fpmax_ffs` equals `none`
  and says `n_selected = 0`, as intended.
- Binning is irrelevant for categorical-only data, so the grid keeps one
  (strategy, bins) pair there (BCW and Soybean).

**Deviations from the draft above.**
- `min_supports` is `[0.8 ... 0.2]`, validated strictly descending. Ionosphere (33
  numeric attributes) already yields 1,828 itemsets at support 0.4, so the runner
  mines supports high to low and skips the rest of a (strategy, bins) pair once one
  exceeds `max_itemsets`, recording `skipped_after_too_many_itemsets`.
- The labels archive keys are `<arm>__<seed>` (not `<arm>|<seed>`).
- Selection rows for non-`ok` settings carry no seed (one row per setting).

---

## Out of scope (next plans)

- **Step 3 plan:** run all eight UCI datasets and both applications; train DEKM
  embeddings (Kaggle) and run `load_dekm_study_data`; the MvDEC loader and appendix
  (needs the `MVDEC_dense.py` global-state refactor); runtime budgeting and sharding.
- **Step 4 plan:** Wilcoxon with Holm, effect sizes, `pairwise_ari_stability` over the
  saved labels, and the paper tables.
- **Deferred:** retiring the legacy job runners, `best_evaluation` and
  `common_evaluation`.
