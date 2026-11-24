# FP-Max Study Runs (Step 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the full factorial study on the eight UCI tables and the two applications
within a feasible compute budget, producing one CSV per (dataset, algorithm, seed).

**Architecture:** Cost caps live in the study config and apply identically to every arm of
a dataset. The runner can execute a subset of report seeds, so one job is one
(dataset, algorithm, seed). A launcher script runs the job list as independent
subprocesses with a fixed worker count (`multiprocessing` pools fail on this machine) and
skips jobs whose output already exists, so it is resumable.

**Tech Stack:** Python 3.12 via `uv run --python 3.12 --link-mode=copy`, subprocess, pandas.

**Spec:** `docs/superpowers/specs/2026-11-24-fpmax-study-design.md`. Builds on
`docs/superpowers/plans/2026-11-25-fpmax-study-runner.md` (done, including its
runtime-measurement section).

## Global Constraints

- Cost caps apply to every arm of a dataset and are recorded in `configs/fpmax_study.json`;
  nothing is tuned per arm or per outcome.
- FFS: at most 15 candidate itemsets (the 15 with highest support), at most 5 accepted
  steps, 1 Intuitive restart per candidate fit; the final labels of a non-empty selection
  are refit with the full restart count on the selected itemsets only.
- Intuitive restarts: 3 (as in the paper's select rows).
- Worker processes set `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` to
  1 to avoid oversubscription.
- Never commit `paper_replication/intuitive_kprototypes/data/` or `output/`.
- Commit with `git -c user.name="ngvdat.w" -c user.email="ngvdat.w@gmail.com" commit` and end
  each message with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Review Focus

1. A shard that reruns the same seed must overwrite, not append, its own files.
2. Shards of one (dataset, algorithm) must not overwrite each other's files.
3. A cap that removes all FFS candidates leaves `fpmax_ffs` equal to `none` with
   `n_selected = 0`, not an error.
4. The launcher must not rerun finished jobs and must report failed jobs.

---

### Task 1: Cost caps (config, FFS, arms)

**Files:**
- Modify: `configs/fpmax_study.json`, `src/pipeline/study_config.py`,
  `src/pipeline/study_arms.py`, `src/pipeline/study_ffs.py`
- Test: `tests/test_study_config.py`, `tests/test_study_arms.py`, `tests/test_study_ffs.py`

**Interfaces:**
- Produces:
  - `StudyConfig.ffs_max_candidates: int` (15), `ffs_max_steps: int` (5),
    `ffs_restarts: int` (1); `intuitive.restarts` becomes 3.
  - `fit_arm(data, algorithm, itemsets, seed, study, restarts: int | None = None)`:
    `restarts` overrides `study.intuitive.restarts` for `intuitive` only.
  - `forward_select_itemsets` considers only `itemsets.columns[:study.ffs_max_candidates]`,
    stops after `study.ffs_max_steps` accepted steps, evaluates candidates with
    `restarts=study.ffs_restarts`, and refits a non-empty selection once with the full
    restart count (falls back to the last accepted labels if that refit is invalid).

- [ ] **Step 1: Write failing tests.** `test_ffs_only_considers_the_first_candidates`
  (`ffs_max_candidates=2` of 3 columns evaluates 2), `test_ffs_stops_after_max_steps`,
  `test_ffs_candidates_use_reduced_restarts_and_final_refit_uses_full` (record the
  `restarts` argument of scripted `fit_arm`), `test_ffs_refit_failure_keeps_last_labels`,
  `test_restarts_override_changes_intuitive_restart_count`, and config tests for the new
  fields (positive integers, `ffs_restarts <= intuitive.restarts`).
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the full suite; expect PASS.**
- [ ] **Step 5: Commit** "Cap FFS cost and use fewer Intuitive restarts during selection".

### Task 2: Seed shards

**Files:**
- Modify: `src/pipeline/study_runner.py` (`run_group`), `scripts/run_fpmax_study.py`
- Test: `tests/test_study_runner.py`

**Interfaces:**
- Produces: `run_group(data, algorithm, study, out_dir, stem_suffix: str = "") -> Path`
  (files `<dataset>__<algorithm><stem_suffix>.csv` and `.labels.npz`); the script gets
  `--seeds 0,3` (a subset of the config's report seeds; unknown seeds are an error) and
  derives `stem_suffix = "__seeds-0-3"` when given.

- [ ] **Step 1: Write failing tests.** `test_stem_suffix_changes_file_names`,
  `test_two_shards_do_not_overwrite_each_other`, script-level
  `test_parse_seed_subset_rejects_unknown_seeds`.
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run tests; expect PASS.**
- [ ] **Step 5: Commit** "Let the study runner execute seed shards".

### Task 3: Launcher

**Files:**
- Create: `scripts/launch_fpmax_study.py`
- Test: `tests/test_launch_fpmax_study.py`

**Interfaces:**
- Produces:
  - `build_jobs(study: StudyConfig, datasets: list[str], algorithms: list[str]) ->
    list[Job]`; `Job(dataset, algorithm, seed)`; `kmeans` is only scheduled for datasets
    with numeric attributes (all UCI keys except `bcw` and `soybean_small_no_constants`;
    both applications).
  - `job_output_path(out_dir, job) -> Path` (the CSV the job will write).
  - CLI: `--out-dir`, `--workers` (default 12), `--datasets`, `--algorithms`,
    `--dry-run` (print the job list). Skips jobs whose CSV exists; writes
    `logs/<stem>.log`; prints a final summary with failed jobs and exits non-zero if any
    failed.

- [ ] **Step 1: Write failing tests.** `test_jobs_cover_every_dataset_algorithm_seed`,
  `test_kmeans_skipped_for_categorical_only_datasets`,
  `test_existing_output_is_skipped`, `test_job_command_uses_seed_shard_flag`.
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement** with `subprocess.Popen` and a fixed worker window.
- [ ] **Step 4: Run tests; expect PASS.**
- [ ] **Step 5: Commit** "Add the study launcher".

### Task 4: Run

- [ ] **Step 1:** `uv run --python 3.12 --link-mode=copy python scripts/launch_fpmax_study.py
  --out-dir output/fpmax_study --workers 12` in the background; resume by rerunning the
  same command.
- [ ] **Step 2:** Count finished and failed jobs; inspect logs of failures; rerun.
- [ ] **Step 3:** Record in this file: jobs total / finished / failed, wall time, rows by
  status.

## Out of scope

- **DEKM and MvDEC backbones** need embeddings trained on a GPU (Kaggle). The loader and
  runner support them; running them needs artifacts that do not exist locally.
- **Step 4:** statistics and tables (Wilcoxon with Holm over every setting,
  `pairwise_ari_stability`, deduplication by `itemset_signature`).
