import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pipeline.io import save_results


@dataclass(frozen=True)
class DummyRecord:
    job_index: int
    value: int


@dataclass(frozen=True)
class AssignmentRecord:
    """Minimal experiment record carrying row-level assignments."""

    job_index: int
    cluster_assignments: str


def test_save_results_retries_transient_replace_permission_error(tmp_path, monkeypatch):
    original_replace = Path.replace
    calls = {"count": 0}

    def flaky_replace(self, target):
        if self.name.endswith(".tmp") and calls["count"] == 0:
            calls["count"] += 1
            raise PermissionError("temporary lock")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)

    output_path = tmp_path / "results.csv"
    result = save_results(
        existing_df=pd.DataFrame(columns=["job_index", "value"]),
        records={0: DummyRecord(job_index=0, value=42)},
        save_path=output_path,
        label="test",
        columns=["job_index", "value"],
    )

    assert calls["count"] == 1
    assert output_path.exists()
    assert result.loc[0, "value"] == 42


def test_save_results_persists_row_level_cluster_assignments(tmp_path) -> None:
    """Assignment JSON must remain usable after a CSV round trip."""

    output_path = tmp_path / "results.csv"
    save_results(
        existing_df=pd.DataFrame(columns=["job_index", "cluster_assignments"]),
        records={
            0: AssignmentRecord(
                job_index=0,
                cluster_assignments="[2,0,1]",
            )
        },
        save_path=output_path,
        label="test",
        columns=["job_index", "cluster_assignments"],
    )

    saved = pd.read_csv(output_path)
    assert json.loads(saved.loc[0, "cluster_assignments"]) == [2, 0, 1]
