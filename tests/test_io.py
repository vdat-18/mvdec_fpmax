from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pipeline.io import save_results


@dataclass(frozen=True)
class DummyRecord:
    job_index: int
    value: int


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
