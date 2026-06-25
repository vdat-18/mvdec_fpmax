import pickle

import numpy as np
import pandas as pd
import pytest

from config import H_FUSED_COLUMNS
from pipeline.data import load_mvdec_result


def _write_mvdec_inputs(tmp_path, n_rows=4):
    data_path = tmp_path / "tiki_preprocessed.csv"
    result_path = tmp_path / "fused_representation.pkl"

    pd.DataFrame({"feature": range(n_rows)}).to_csv(data_path, index=False)
    h_fused = np.arange(n_rows * len(H_FUSED_COLUMNS), dtype=np.float32).reshape(
        n_rows, len(H_FUSED_COLUMNS)
    )
    labels = np.array([0, 0, 1, 1][:n_rows])
    with result_path.open("wb") as file:
        pickle.dump(
            {
                "h_fused": h_fused,
                "labels": labels,
                "init": "k-means++",
                "score": 0.1,
                "iteration": 1,
            },
            file,
        )

    return result_path, data_path


def test_load_mvdec_result_uses_fused_embedding_columns(tmp_path):
    result_path, data_path = _write_mvdec_inputs(tmp_path)

    result = load_mvdec_result(result_path=result_path, data_path=data_path)

    assert list(result.h_fused_df.columns) == [
        f"fused_{index}" for index in range(1, 12)
    ]


def test_load_mvdec_result_rejects_preprocessed_row_mismatch(tmp_path):
    result_path, data_path = _write_mvdec_inputs(tmp_path)
    pd.DataFrame({"feature": [1, 2, 3]}).to_csv(data_path, index=False)

    with pytest.raises(ValueError, match="row count does not match"):
        load_mvdec_result(result_path=result_path, data_path=data_path)
