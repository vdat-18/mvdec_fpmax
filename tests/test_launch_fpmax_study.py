"""Tests for the study launcher."""

import dataclasses
import sys

from pipeline.study_config import DEFAULT_STUDY_CONFIG_PATH, load_study_config
from pipeline.study_runner import done_marker_path
from scripts import launch_fpmax_study as launcher
from scripts.launch_fpmax_study import (
    Job,
    build_jobs,
    job_command,
    job_is_done,
    job_output_path,
    missing_inputs,
    run_jobs,
)

STUDY = dataclasses.replace(
    load_study_config(DEFAULT_STUDY_CONFIG_PATH), report_seeds=(0, 1, 2)
)


def test_jobs_cover_every_dataset_algorithm_seed():
    jobs = build_jobs(STUDY, ["iris", "tiki"], ["intuitive", "kprototypes"])

    assert len(jobs) == 2 * 2 * 3
    assert Job("tiki", "kprototypes", 2) in jobs


def test_kmeans_skipped_for_categorical_only_datasets():
    jobs = build_jobs(STUDY, ["bcw", "iris"], ["kmeans"])

    assert {job.dataset for job in jobs} == {"iris"}


def test_job_output_path_matches_the_shard_file_name(tmp_path):
    job = Job("iris", "intuitive", 3)

    assert job_output_path(tmp_path, job).name == "iris__intuitive__seeds-3.csv"


def test_job_command_uses_seed_shard_flag(tmp_path):
    command = job_command(DEFAULT_STUDY_CONFIG_PATH, tmp_path, Job("iris", "kmeans", 4))

    assert command[0] == sys.executable
    assert command[command.index("--seeds") + 1] == "4"
    assert command[command.index("--dataset") + 1] == "iris"
    assert command[command.index("--algorithm") + 1] == "kmeans"


def test_existing_output_is_skipped_and_failures_are_reported(tmp_path, monkeypatch):
    done = Job("iris", "kmeans", 0)
    todo_ok = Job("iris", "kmeans", 1)
    todo_bad = Job("iris", "kmeans", 2)
    job_output_path(tmp_path, done).write_text("x", encoding="utf-8")
    done_marker_path(job_output_path(tmp_path, done)).write_text(
        "done", encoding="utf-8"
    )
    # a partial CSV without a marker must not count as finished
    job_output_path(tmp_path, todo_ok).write_text("partial", encoding="utf-8")
    started: list[Job] = []

    class FakeProcess:
        def __init__(self, code):
            self.code = code

        def poll(self):
            return self.code

    def fake_popen(command, **_kwargs):
        seed = int(command[command.index("--seeds") + 1])
        started.append(Job("iris", "kmeans", seed))
        return FakeProcess(1 if seed == 2 else 0)

    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(launcher.time, "sleep", lambda _seconds: None)

    failed = run_jobs(
        [done, todo_ok, todo_bad], DEFAULT_STUDY_CONFIG_PATH, tmp_path, workers=2
    )

    assert started == [todo_ok, todo_bad]
    assert failed == [todo_bad]


def test_partial_csv_is_not_done_until_the_marker_exists(tmp_path):
    job = Job("iris", "kmeans", 0)
    job_output_path(tmp_path, job).write_text("partial", encoding="utf-8")

    assert not job_is_done(tmp_path, job)

    done_marker_path(job_output_path(tmp_path, job)).write_text(
        "done", encoding="utf-8"
    )

    assert job_is_done(tmp_path, job)


def test_missing_dekm_bundles_are_reported_once_per_file(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "PROJECT_DIR", tmp_path)
    study = dataclasses.replace(
        STUDY,
        dekm=dataclasses.replace(STUDY.dekm, bundle_dir="bundles"),
    )
    jobs = build_jobs(study, ["dekm_reuters", "iris"], ["kmeans", "intuitive"])
    (tmp_path / "bundles").mkdir()
    (tmp_path / "bundles" / "dekm_reuters__seed-42.npz").write_bytes(b"x")

    missing = missing_inputs(study, jobs)

    assert [path.name for path in missing] == [
        "dekm_reuters__seed-43.npz",
        "dekm_reuters__seed-44.npz",
    ]
