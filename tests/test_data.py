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


def test_load_mvdec_result_accepts_mvdec2025_figure_output_average_contract(tmp_path):
    data_path = tmp_path / "airpollution.csv"
    result_path = tmp_path / "mvdec2025.pkl"
    pd.DataFrame({"feature": range(4)}).to_csv(data_path, index=False)
    h_view1 = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.1, 5.0],
        ],
        dtype=np.float32,
    )
    h_view2 = h_view1 + 0.2
    h_fused = (h_view1 + h_view2) / 2
    labels = np.array([0, 0, 1, 1])
    with result_path.open("wb") as file:
        pickle.dump(
            {
                "fusion_contract": "mvdec2025_figure_output_average",
                "view_output_layout": "eq5_compatible_10_plus_13",
                "final_training_objective": (
                    "dekm2021_greedy_cluster_loss_after_reconstruction_pretrain"
                ),
                "h_view1": h_view1,
                "h_view2": h_view2,
                "h_fused": h_fused,
                "labels": labels,
                "fusion_dim": 2,
                "init": "k-means",
                "score": 0.5,
                "iteration": 3,
            },
            file,
        )

    result = load_mvdec_result(result_path=result_path, data_path=data_path)

    assert result.h_fused.shape == (4, 2)
    assert list(result.h_fused_df.columns) == ["fused_1", "fused_2"]
    np.testing.assert_allclose(result.h_fused, h_fused)


def test_load_mvdec_result_rejects_unexpected_mvdec2025_layout(tmp_path):
    data_path = tmp_path / "airpollution.csv"
    result_path = tmp_path / "mvdec2025.pkl"
    pd.DataFrame({"feature": range(4)}).to_csv(data_path, index=False)
    h_view1 = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.1, 5.0],
        ],
        dtype=np.float32,
    )
    h_view2 = h_view1 + 0.2
    labels = np.array([0, 0, 1, 1])
    with result_path.open("wb") as file:
        pickle.dump(
            {
                "fusion_contract": "mvdec2025_figure_output_average",
                "view_output_layout": "figure_dense_23",
                "h_view1": h_view1,
                "h_view2": h_view2,
                "h_fused": (h_view1 + h_view2) / 2,
                "labels": labels,
                "fusion_dim": 2,
                "init": "k-means",
                "score": 0.5,
                "iteration": 3,
            },
            file,
        )

    with pytest.raises(ValueError, match="view_output_layout"):
        load_mvdec_result(result_path=result_path, data_path=data_path)


def test_load_mvdec_result_rejects_legacy_concat_named_h_fused(tmp_path):
    data_path = tmp_path / "airpollution.csv"
    result_path = tmp_path / "legacy.pkl"
    pd.DataFrame({"feature": range(4)}).to_csv(data_path, index=False)
    h_fused = np.arange(4 * 17, dtype=np.float32).reshape(4, 17)
    labels = np.array([0, 0, 1, 1])
    with result_path.open("wb") as file:
        pickle.dump(
            {
                "h_fused": h_fused,
                "labels": labels,
                "input_dim": 13,
                "config": {"view1_latent_dim": 4},
                "init": "k-means++",
                "score": 0.1,
                "iteration": 1,
            },
            file,
        )

    with pytest.raises(ValueError, match="legacy concat representation"):
        load_mvdec_result(result_path=result_path, data_path=data_path)
