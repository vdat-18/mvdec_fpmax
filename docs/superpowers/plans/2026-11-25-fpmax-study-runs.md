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

## Outcome: the 8 UCI datasets (2026-11-25)

- **Jobs.** 220 of 220 UCI jobs finished with 0 failures. The launcher ran from
  2026-11-24 evening to 2026-11-25 17:30, with one reboot in between; finished shards
  were skipped on resume. Tiki and AIRPOLLUTION (60 jobs) were still running at this
  point.
- **Rows by status.**

  | Arm | ok | Other |
  | --- | ---: | --- |
  | `none` | 220 | — |
  | `fpmax` | 2440 | 66 `no_itemsets`, 32 `too_many_itemsets`, 22 `skipped_after_too_many_itemsets` |
  | `fpmax_ffs` | 2440 | — |

  No `error` and no `invalid_partition` rows.
- **Summary.** `results/summary_uci/`, made by `scripts/summarize_fpmax_study.py` on the
  finished UCI shards only.

Each delta is `arm - none` for the same algorithm and seed. The value per dataset is the
median over distinct itemset blocks of the seed mean. The test is a Wilcoxon over 8
datasets, Holm-adjusted over all 8 external tests.

| Metric | Algorithm | Arm | Median delta | Wins / losses | p | p (Holm) |
| --- | --- | --- | ---: | --- | ---: | ---: |
| ARI | K-Prototypes | fpmax | -0.025 | 1 / 7 | 0.039 | 0.23 |
| ARI | K-Prototypes | fpmax_ffs | +0.005 | 4 / 1 | 0.31 | 0.63 |
| ARI | Intuitive | fpmax | -0.060 | 2 / 5 | 0.078 | 0.39 |
| ARI | Intuitive | fpmax_ffs | -0.055 | 1 / 6 | 0.078 | 0.39 |
| NMI | K-Prototypes | fpmax | -0.009 | 2 / 6 | 0.11 | 0.39 |
| NMI | K-Prototypes | fpmax_ffs | +0.001 | 4 / 1 | 0.31 | 0.63 |
| NMI | Intuitive | fpmax | -0.053 | 1 / 6 | 0.031 | 0.22 |
| NMI | Intuitive | fpmax_ffs | -0.039 | 0 / 7 | 0.016 | 0.13 |

Findings:

- **No arm differs significantly from `none` after Holm.**
- **Adding every itemset tends to hurt** both algorithms.
- **FFS turns the K-Prototypes effect slightly positive.** The ARI median is +0.005, with
  4 datasets better and 1 worse. The largest gains are on Chess (+0.082), Iris (+0.036)
  and Soybean (+0.025).
- **FFS does not rescue Intuitive.** NMI is lower on 7 of 8 datasets; Heart-5 is the
  only gain (+0.032 ARI).
- **Intuitive starts from a much stronger baseline.** Its `none` ARI is:
  Australian 0.512 (K-Prototypes 0.334), BCW 0.807, Soybean 1.000, Iris 0.886. The
  Intuitive baselines match the paper.
- **Stability.** The median pairwise ARI across seeds is above 0.95 for every arm.
  `fpmax_ffs` is slightly more stable than `none`: 0.980 vs 0.957 for Intuitive, and
  1.000 vs 0.991 for K-Prototypes.

## DEKM backbone (branch `dekm-backbone`)

**Why bundles.**
- At full size (10,000–18,846 rows), one study job would take tens of hours on this
  CPU. Chess, with 4,151 rows, already took about 3 h per job.
- DEKM is therefore trained on every row on a Kaggle GPU. The study then clusters a
  fixed, label-free sample of `data.dekm.n_rows = 2000` rows.
- A study bundle (`write_dekm_study_bundle`) holds those rows' embedding, their tf-idf
  vectors (the cosine scoring space) and their labels.
- The CPU study needs neither the text sources nor pickles.
- Report seed `s` clusters the embedding of seed `42 + s`, so backbone variance enters
  the seed spread. Every arm of a seed uses the same embedding.

**Steps.**

1. **Train and export on Kaggle.** Run the three Kaggle notebooks with a GPU and
   internet on. Each trains seeds 42–51 with the frozen canonical protocol and writes
   `dekm-fpmax-<dataset>.zip`. The notebooks are:
   - `notebooks/dekm_fpmax_bundles_reuters_seeds42_51_kaggle.ipynb`
   - `notebooks/dekm_fpmax_bundles_rcv1_seeds42_51_kaggle.ipynb`
   - `notebooks/dekm_fpmax_bundles_20news_seeds42_51_kaggle.ipynb`
2. **Bring the results back.** Unzip each archive at the repository root. This yields
   `output/dekm_bundles/dekm_<dataset>__seed-<42..51>.npz` and the run records in
   `output/dekm_runs/<dataset>/`.
3. **Launch the study** on the DEKM datasets only:
   `uv run --python 3.12 --link-mode=copy python scripts/launch_fpmax_study.py
   --out-dir output/fpmax_study --workers 12 --datasets dekm_reuters dekm_rcv1
   dekm_20news`.
   The launcher refuses to start if any bundle is missing.
4. **Summarize.** Summarize `output/fpmax_study` as for the UCI datasets. The DEKM rows
   carry `backbone = dekm`.

**Intuitive parameters.** Intuitive uses the config defaults on these datasets
(`beta = 2`, `mu = 0.5`, `gamma = 0.5`); the embedding is numeric-only.
