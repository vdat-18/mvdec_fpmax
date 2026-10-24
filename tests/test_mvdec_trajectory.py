import json

import numpy as np

from pipeline import mvdec_trajectory


def test_trajectory_disabled_without_env(monkeypatch):
    monkeypatch.delenv(mvdec_trajectory.TRAJECTORY_ENV_VAR, raising=False)
    assert mvdec_trajectory.trajectory_path() is None


def test_trajectory_path_from_env(monkeypatch, tmp_path):
    target = tmp_path / "traj.jsonl"
    monkeypatch.setenv(mvdec_trajectory.TRAJECTORY_ENV_VAR, str(target))
    assert mvdec_trajectory.trajectory_path() == target


def test_record_checkpoint_appends_metrics_without_touching_global_rng(tmp_path):
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 50)
    h1 = np.vstack([rng.normal(0, 0.1, (50, 3)), rng.normal(5, 0.1, (50, 3))])
    h2 = rng.normal(size=(100, 3))
    path = tmp_path / "traj.jsonl"
    np.random.seed(123)
    expected_next = np.random.rand()
    np.random.seed(123)
    for step in (0, 10):
        mvdec_trajectory.record_checkpoint(
            path,
            seed=1,
            step=step,
            n_clusters=2,
            true_labels=y,
            fused_assignment=y,
            h_view1=h1,
            h_view2=h2,
            n_changed=0,
        )
    assert np.random.rand() == expected_next
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["step"] for row in rows] == [0, 10]
    assert rows[0]["fused_acc"] == 1.0
    assert rows[0]["view1_acc"] == 1.0
    assert 0.0 <= rows[0]["view2_nmi"] <= 1.0
