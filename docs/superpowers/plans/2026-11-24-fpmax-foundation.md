# FP-Max Study Foundation (Steps 0–1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the clustering building blocks trustworthy before the factorial FP-Max study:
- one tested Intuitive model;
- a label-free, non-circular way to pick clustering restarts;
- a fixed reference space for scoring;
- ARI;
- an FFS that rejects degenerate partitions;
- no label leakage in the public text data.

**Architecture:**
- The validated sandbox model becomes the single `intuitive_kprototypes` package under `src/`. The pipeline-only options (`farthest_first`, `view_weight_alpha`) move into it, and the sandbox copy is deleted.
- A new `src/pipeline/reference_evaluation.py` scores any labelling in one fixed space per dataset, independent of the features a method used.
- K-Prototypes and FFS gain a shared partition-validity check.
- The K-Prototypes init is chosen by the algorithm's own cost, not by silhouette.
- Later plans (Step 2: backbones and factorial runner; Step 3: runs; Step 4: statistics) build on these interfaces.

**Tech Stack:** Python 3.12 via `uv run --python 3.12 --link-mode=copy`, numpy, pandas, scikit-learn, kmodes, mlxtend, gower, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-11-24-fpmax-study-design.md`

## Global Constraints

- Every shell command runs as `uv run --python 3.12 --link-mode=copy python -m pytest ...`. `uv run pytest` is blocked by the application-control policy. `rtk` is not installed; run commands directly.
- Never use true labels in any selection step (init choice, FFS, hyperparameters, grid row). Labels are only for ARI/NMI/ACC after selection.
- Silhouette that is reported must be computed in the dataset's fixed reference space, never in a space that contains a method's own FP-Max columns.
- The Intuitive `mu_convention` used by the pipeline is `"swapped"`. Record it as a deviation from printed Eq. (26).
- The FFS minimum improvement is `0.005`.
- A valid partition has exactly `n_clusters` non-empty clusters, each with at least `2` rows.
- Do not commit `paper_replication/intuitive_kprototypes/data/`.
- Commit with `git -c user.name="ngvdat.w" -c user.email="ngvdat.w@gmail.com" commit`. End every message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Paper reproduction must not change.** Iris and BCW "with select" rows after the package merge must still give ACC 0.960 and 0.950 (Task 1 verification step).
2. **K-Prototypes with an empty categorical block.** kmodes refuses to run with no categorical columns, yet the `none` arm on a purely numeric backbone needs exactly that. Expected: K-Means with the same `n_init`, flagged `backend="kmeans_equivalent"` (Task 4 test).
3. **A single-label or singleton partition reaching silhouette.** Expected: the candidate is skipped, not crashed and not selected (Task 4 and Task 5 tests).
4. **A binary feature identical to the labels.** Expected: reference-space silhouette does not reward it the way mixed-space silhouette does (Task 3 regression test).
5. **Negative-valued numeric input to Intuitive.** For example, a raw latent before scaling. Expected: the coefficient of variation stays finite (`|mean|`) and the min-max helper maps the input to [0, 1] (Task 1 and Task 2 tests).

---

### Task 1: One Intuitive package (fixes C5, M1)

**Files:**
- Replace: `src/intuitive_kprototypes/model.py` with `paper_replication/intuitive_kprototypes/src/intuitive_kprototypes/model.py`, then re-add the pipeline features listed in Step 3.
- Update: `src/intuitive_kprototypes/__init__.py` so it exports the union of both `__init__` files.
- Delete:
  - `paper_replication/intuitive_kprototypes/src/`
  - `paper_replication/intuitive_kprototypes/tests/conftest.py`
- Modify: the `sys.path.insert(... SANDBOX_SRC ...)` lines in:
  - `paper_replication/intuitive_kprototypes/scripts/run_paper_benchmark.py:35`
  - `run_parameter_grid.py:21`
  - `run_scenario_sweep.py:29`
  - `run_uci_smoke.py:40`
  
  Remove them. Keep the `parent` insert in `run_scenario_sweep.py:30`.
- Modify: rename the init value `"paper"` to `"first"` at:
  - `src/pipeline/clustering.py:445`
  - `src/pipeline/best_evaluation.py:645`
- Test: `tests/test_intuitive_model.py` (new). Existing sandbox tests now run against `src/`.

**Interfaces:**
- Produces:
  - `IntuitiveKPrototypes(..., init_strategy: InitStrategy = "first", mu_convention: MuConvention = "eq26", view_weight_alpha: float | None = None)`
  - `InitStrategy = Literal["first", "random_candidate", "farthest", "farthest_first", "random", "chebyshev"]`

- [ ] **Step 1: Write the failing tests** in `tests/test_intuitive_model.py`.

```python
def test_single_package_is_loaded():
    import intuitive_kprototypes
    assert Path(intuitive_kprototypes.__file__).parts[-3:-1] == ("src", "intuitive_kprototypes")

def test_farthest_first_starts_from_point_farthest_from_random_row():
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    idx = select_initial_prototypes(X, None, n_clusters=2, random_state=0, strategy="farthest_first")
    assert set(idx.tolist()) == {0, 3}  # extremes, whichever row is drawn first

def test_view_weight_alpha_one_ignores_categorical_block():
    # numeric block separates {0,1} vs {2,3}; categorical block says the opposite
    model = IntuitiveKPrototypes(n_clusters=2, view_weight_alpha=1.0, random_state=0, init_strategy="farthest_first")
    labels = model.fit(X_num=[[0.0],[0.1],[1.0],[1.1]], X_cat=[["a"],["b"],["a"],["b"]]).labels_
    assert labels[0] == labels[1] != labels[2] == labels[3]

def test_negative_numeric_means_give_finite_weights():
    X = np.array([[-1.0, -2.0], [-1.1, -2.2], [-5.0, -9.0], [-5.2, -9.1]])
    model = IntuitiveKPrototypes(n_clusters=2, random_state=0).fit(X_num=X, X_cat=None)
    assert np.isfinite(model.result_.weights.numeric).all()
```

Use the attribute names the merged model actually exposes (`labels_`, `result_.weights`); adjust only those names, not the assertions.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --python 3.12 --link-mode=copy python -m pytest tests/test_intuitive_model.py -q`

Expected: `test_single_package_is_loaded` fails because the path resolves to `paper_replication`; `farthest_first` raises `ValueError`.

- [ ] **Step 3: Merge the models.** Start from the sandbox file. Then:
  - Add `"farthest_first"` to `INIT_STRATEGIES`. It is the pipeline's `select_farthest_first_prototypes` behaviour: random row, then the row farthest from it, then farthest-first. Implement it as a branch in `select_initial_prototypes` that reuses `_gap_to_selected`.
  - Restore `view_weight_alpha` validation and the `_assignment_distances` branch exactly as in the pipeline file (the diff hunk at old lines 1061–1092), placing `mu_convention` inside both branches.
  - Use `np.abs(means)` in the CV weights. This is identical on non-negative data, so reproduction is unchanged.
  - Set `lambda_init` to apply only when `init_strategy in {"first", "random_candidate", "chebyshev"}`, and update `clustering.py:445` accordingly.

- [ ] **Step 4: Run every test plus the reproduction check**

Run: `uv run --python 3.12 --link-mode=copy python -m pytest tests paper_replication/intuitive_kprototypes/tests -q`

Expected: all pass.

Run: `uv run --python 3.12 --link-mode=copy python paper_replication/intuitive_kprototypes/scripts/run_paper_benchmark.py --dataset iris --beta-mode tuned --mu-convention swapped`, then the same for `--dataset bcw`.

Expected: "with select" ACC 0.960 (Iris) and 0.950 (BCW).

- [ ] **Step 5: Commit** with the message "Merge the two Intuitive models into one tested package".

---

### Task 2: Frozen pipeline protocol for Intuitive (fixes M1, M2)

**Files:**
- Modify:
  - `src/pipeline/intuitive_protocol.py` (`IntuitiveProtocol` at line 38)
  - `configs/intuitive_protocols.json`
  - `src/pipeline/clustering.py` (`run_intuitive_kprototypes`, line 353)
- Test: `tests/test_intuitive_protocol.py` (extend the existing file, or create it if absent)

**Interfaces:**
- Produces:
  - `IntuitiveProtocol.mu_convention: str`, which must be `"swapped"` for the new protocol id `intuitive_fpmax_study_v1`.
  - `minmax_scale_block(df: pd.DataFrame) -> pd.DataFrame` in `clustering.py`. It maps each column to [0, 1] and turns constant columns into `0.0`.
  - `run_intuitive_kprototypes(..., mu_convention: str = "swapped")`, passed through to the model.

- [ ] **Step 1: Write failing tests**
  - `test_study_protocol_uses_swapped_mu`: load `intuitive_fpmax_study_v1`, then assert that `mu_convention == "swapped"` and `ffs_min_improvement == 0.005`.
  - `test_minmax_scale_block_bounds_and_constant_column`: input columns `[-3, 0, 3]` and `[5, 5, 5]` produce `[0, 0.5, 1]` and `[0, 0, 0]`.
  - `test_run_intuitive_passes_mu_convention`: monkeypatch `IntuitiveKPrototypes` to capture kwargs, then assert that `kwargs["mu_convention"] == "swapped"`.

- [ ] **Step 2: Run to verify they fail**

Run: `... -m pytest tests/test_intuitive_protocol.py -q`

Expected: KeyError or AttributeError.

- [ ] **Step 3: Implement.**
  - Add the field and the validation (allowed values: `MU_CONVENTIONS`).
  - Add the protocol `intuitive_fpmax_study_v1` to the JSON. Copy the current default protocol and change three fields:
    - `"mu_convention": "swapped"`
    - `"ffs_min_improvement": 0.005`
    - `"numeric_preprocessing": "minmax"`
  - Wire `mu_convention` into `run_intuitive_kprototypes`.
  - Existing protocols get `"mu_convention": "eq26"` so that old results remain reproducible.

- [ ] **Step 4: Run all tests**

Run: `... -m pytest tests -q`

Expected: all pass.

- [ ] **Step 5: Commit** with the message "Add the FP-Max study Intuitive protocol with swapped mu and min-max input".

---

### Task 3: Fixed reference-space evaluation and ARI (fixes C1, C2 reporting side)

**Files:**
- Create: `src/pipeline/reference_evaluation.py`
- Modify: `src/pipeline/external_metrics.py`. Add `ari: float` to `ExternalMetrics` using `adjusted_rand_score`. Drop the `n_clusters` exact-count requirement into a separate `strict` flag, so that an empty predicted cluster still scores (fixes M6's crash source).
- Test:
  - `tests/test_reference_evaluation.py`
  - extend `tests/test_external_metrics.py`

**Interfaces:**
- Produces:
  - `build_reference_distance(df: pd.DataFrame, max_rows: int = 3000, random_state: int = 0) -> ReferenceSpace`
    - `ReferenceSpace` is a frozen dataclass with fields `distance: np.ndarray` and `row_index: np.ndarray`.
    - The distance is a Gower matrix over the original attributes: numeric columns are range-normalised; object/category columns use simple matching.
    - Above `max_rows`, a fixed random subsample is taken with `random_state`, and its indices are kept in `row_index`.
  - `score_internal(space: ReferenceSpace, labels: np.ndarray) -> InternalScores`
    - Fields: `silhouette`, `davies_bouldin`, `calinski_harabasz`.
    - DB and CH are computed on the classical-MDS embedding of the reference distance (`sklearn.manifold.MDS(dissimilarity="precomputed", n_components=min(10, n-1), random_state=0)`). Compute it once per space and cache it in `ReferenceSpace.embedding`.
    - All three are `nan` when the labels are not a valid partition.
  - `pairwise_ari_stability(label_sets: list[np.ndarray]) -> float`: mean ARI over all pairs.

- [ ] **Step 1: Write failing tests**
  - `test_label_copy_feature_is_not_rewarded_in_reference_space`. Regression test for C2, using two Gaussian blobs that overlap heavily (seed 0), with `y` as the generating labels.
    - Compute the mixed-space silhouette with `compute_mixed_gower_distance(X, pd.DataFrame({"leak": y}))` and labels `y`.
    - Compute the reference silhouette with `score_internal(build_reference_distance(X), y)`.
    - Assert `mixed > reference + 0.2`. This documents the inflation that the reference space avoids.
  - `test_reference_space_is_independent_of_method_features`: the same `X` with two different labellings gives distances that are bitwise equal across calls.
  - `test_invalid_partition_scores_nan`: labels all `0` give `nan` for all three scores.
  - `test_subsample_is_deterministic`: with n = 5000 and `max_rows=3000`, `row_index` is identical across two calls.
  - `test_pairwise_ari_stability_identical_and_permuted`: `[a, a]` gives 1.0, and relabelled `a` also gives 1.0.
  - `test_external_metrics_reports_ari` and `test_external_metrics_tolerates_empty_predicted_cluster` (non-strict mode).

- [ ] **Step 2: Run to verify they fail** (ImportError).

- [ ] **Step 3: Implement** the functions above. Reuse `is_valid_partition` from Task 4. If Task 4 is not merged yet, define it here in `clustering.py` and let Task 4 consume it.

- [ ] **Step 4: Run tests.** Expected: pass.

- [ ] **Step 5: Commit** with the message "Add fixed reference-space scoring and ARI".

---

### Task 4: K-Prototypes without circular init choice, and with an empty-block arm (fixes C2 selection side, C4, M3)

**Files:**
- Modify: `src/pipeline/clustering.py` (`run_kprototypes`, line 289, and the result dataclass)
- Test: `tests/test_clustering.py` (new)

**Interfaces:**
- Produces:
  - `is_valid_partition(labels: np.ndarray, n_clusters: int, min_size: int = 2) -> bool`
  - `run_kprototypes(continuous_df, binary_df, n_clusters, init_methods=DEFAULT_INIT_METHODS, random_state=RANDOM_STATE, verbose=True) -> KPrototypesResult`
    - Restarts are chosen by **lowest `model.cost_`**, the algorithm's own objective, among valid partitions only.
    - With `binary_df.shape[1] == 0`, it runs `sklearn.cluster.KMeans(n_clusters, n_init=KPROTOTYPES_N_INIT, random_state=random_state)`.
  - `KPrototypesResult.backend: str`, either `"kprototypes"` or `"kmeans_equivalent"`.
  - `KPrototypesResult.cost: float`.
  - The silhouette field stays only as a diagnostic.

- [ ] **Step 1: Write failing tests**
  - `test_is_valid_partition`: three cases. `[0, 0, 1, 1]` with k = 2 is valid; `[0, 0, 0, 1]` with k = 2 is invalid (singleton); `[0, 0, 0, 0]` with k = 2 is invalid.
  - `test_empty_binary_block_uses_kmeans_equivalent`: run on two separated blobs with an empty `binary_df`. Assert `backend == "kmeans_equivalent"` and that the result is a valid partition.
  - `test_init_choice_uses_cost_not_silhouette`: monkeypatch `KPrototypes` so that init `"Huang"` returns cost 10 and init `"Cao"` returns cost 5. Assert that `result.init == "Cao"`.
  - `test_invalid_restart_is_skipped`: monkeypatch so that one init returns a singleton partition. Assert that the other init is chosen. If both are invalid, it raises `RuntimeError("No valid clustering result")`.

- [ ] **Step 2: Run to verify they fail.**

- [ ] **Step 3: Implement.**

- [ ] **Step 4: Run** `... -m pytest tests -q`. Expected: all pass. Update existing tests that asserted silhouette-based init selection. Change only their expectations, and name each changed test in the commit body.

- [ ] **Step 5: Commit** with the message "Pick K-Prototypes restarts by cost and add the empty-block arm".

---

### Task 5: FFS that rejects degenerate partitions and needs a real gain (fixes M3, M5, minor tolerance)

**Files:**
- Modify:
  - `src/pipeline/forward_selection.py` (`run_forward_selection_candidate` at line 92, `DEFAULT_FFS_MIN_IMPROVEMENT` at line 19)
  - The candidate-loop counterpart for Intuitive in `src/pipeline/experiments.py:1840-1961`, which applies the same validity rule.
- Test: `tests/test_forward_selection.py` (new)

**Interfaces:**
- Consumes: `is_valid_partition`, and `run_kprototypes` as chosen by cost (Task 4).
- Produces:
  - `DEFAULT_FFS_MIN_IMPROVEMENT = 0.005`.
  - Candidates whose best partition is invalid get `score = -inf` and are never selected.
  - Silhouette is computed only for valid partitions.
  - `ForwardSelectionResult.n_candidates_evaluated: int`, so the search size can be reported.
- The FFS criterion is unchanged: silhouette in the candidate's mixed space. This is label-free and internal to the method. The spec allows it because the reported metrics come from Task 3.

- [ ] **Step 1: Write failing tests**
  - `test_ffs_stops_below_min_improvement`: a candidate improves the score by 0.001. Assert no feature is selected.
  - `test_ffs_skips_candidate_with_singleton_partition`: monkeypatch the K-Prototypes fit so that feature `"a"` yields a singleton partition and feature `"b"` yields a valid one with a higher score than the baseline. Assert the selection is `["b"]`.
  - `test_ffs_single_label_candidate_does_not_crash`: a fit that returns all-zero labels gives an empty selection and no exception.
  - `test_ffs_counts_candidates`: with 3 features and no selection, `n_candidates_evaluated == 3`.

- [ ] **Step 2: Run to verify they fail.**

- [ ] **Step 3: Implement.** Also replace the K-Prototypes init-by-silhouette loop inside the candidate with a call to `run_kprototypes`, so that the init choice is DRY and cost-based.

- [ ] **Step 4: Run** `... -m pytest tests -q`. Expected: all pass.

- [ ] **Step 5: Commit** with the message "Make FFS skip degenerate partitions and require a 0.005 gain".

---

### Task 6: Remove label leakage from public text data

**Files:**
- Modify: `scripts/fetch_public_datasets.py`
  - `top_k_chi2_features`, line 665;
  - its call sites, lines 783 and 866;
  - the `chi2` import, line 24.
- Test: `tests/test_fetch_public_datasets.py`

**Interfaces:**
- Produces: `top_k_frequent_features(X: sparse.spmatrix, k: int) -> sparse.spmatrix`. It keeps the k columns with the highest document frequency, with ties broken by lower column index; the kept columns are sorted. This matches DEKM's "2,000 most frequent word stems" and takes no `y`.

- [ ] **Step 1: Write the failing test**
  - `test_text_feature_selection_ignores_labels`: the same `X` with two different `y` gives identical selected columns.
  - Also check that the function signature has no `y` parameter: `inspect.signature(top_k_frequent_features).parameters` keys are exactly `["X", "k"]`.

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement.** Replace both call sites and delete `top_k_chi2_features` and the `chi2` import.

- [ ] **Step 4: Run** `... -m pytest tests/test_fetch_public_datasets.py -q`. Expected: pass.

- [ ] **Step 5: Commit** with the message "Select public text features by document frequency, not labels".

---

### Task 7: DEKM_F checkpoint accepts current manifests; record the MvDEC protocol choice

**Files:**
- Modify: `src/pipeline/dekm_checkpoint.py` (`_EXPECTED_MANIFEST_KEYS`, lines 34–58)
- Create: `docs/superpowers/specs/2026-11-24-mvdec-protocol-choice.md`
- Test: `tests/test_dekm_checkpoint.py`

**Interfaces:**
- Produces: the validator accepts either the legacy key set (the current constant) or the current writer's key set from `DEKM_dense.py:629-661`, including `source_id` and the `official_repository_paper_era_*` and `official_repository_latest_audited_*` keys. Any other key set is rejected, as before. Hash pinning for the legacy REUTERS seed-42 checkpoint stays unchanged.

- [ ] **Step 1: Write failing tests**
  - `test_current_writer_manifest_keys_are_accepted`: build a manifest dict with exactly the writer's keys. The key-set check passes.
  - `test_unknown_manifest_key_is_rejected`: one extra key raises `ValueError`.

- [ ] **Step 2: Run to verify they fail.**

- [ ] **Step 3: Implement** with two named frozensets, `_LEGACY_MANIFEST_KEYS` and `_CURRENT_MANIFEST_KEYS`. Membership in either is enough.

- [ ] **Step 4: Write the MvDEC protocol note.** One page:
  - **Chosen protocol for the FP-Max appendix:** `mvdec_dekm_consistent_v1`. It is what the private pipeline and its configs already consume.
  - **Why the alternatives are not used:**
    - S1 and S2 are REUTERS research arms;
    - `paper_informed_full_batch_v2` is not reproduced either.
  - **Statement for the paper:** "not a reproduction of Kassem et al. Eq. 11".
  - **Deferred:** the global-state refactor of `MVDEC_dense.py`, to the Step 3 plan.

- [ ] **Step 5: Run** `... -m pytest tests -q`. Expected: all pass.

- [ ] **Step 6: Commit** with the message "Accept current DEKM_F manifests and record the MvDEC protocol choice".

---

## Out of scope for this plan (next plans)

- **Step 2 plan:** `load_backbone(kind, artifact_path)`; the factorial runner (backbone × algorithm × FP-Max level × seed) that writes long-format CSV; `configs/fpmax_study.json`; deciding whether to retire `experiments.py` job runners, `best_evaluation.py` and `common_evaluation.py`. The legacy M5/M6/M7 fixes are folded into that decision.
- **Step 3 plan:** runs on UCI (CPU), DEKM embeddings (Kaggle), MvDEC appendix, Tiki and AIRPOLLUTION.
- **Step 4 plan:** Wilcoxon with Holm correction, effect sizes, and the paper tables.
