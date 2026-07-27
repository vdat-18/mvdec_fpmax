import importlib
from pathlib import Path

import numpy as np
import pytest


def _load_mvdec(monkeypatch):
    tensorflow = pytest.importorskip("tensorflow")
    representation_dir = (
        Path(__file__).resolve().parents[1] / "src" / "representation_learning"
    )
    monkeypatch.syspath_prepend(str(representation_dir))
    return tensorflow, importlib.import_module("MVDEC_dense")


def test_squared_euclidean_loss_sums_dimensions_before_batch_mean(monkeypatch):
    tensorflow, mvdec = _load_mvdec(monkeypatch)
    input_zeros = tensorflow.zeros((2, 13))
    input_ones = tensorflow.ones((2, 13))
    latent_zeros = tensorflow.zeros((2, 10))
    latent_ones = tensorflow.ones((2, 10))

    reconstruction_loss = tensorflow.reduce_mean(
        mvdec.squared_euclidean_per_sample(input_zeros, input_ones)
        + mvdec.squared_euclidean_per_sample(input_zeros, input_ones)
    )
    kmeans_loss = tensorflow.reduce_mean(
        mvdec.squared_euclidean_per_sample(latent_zeros, latent_ones)
    )

    assert float(reconstruction_loss) == pytest.approx(26.0)
    assert float(kmeans_loss) == pytest.approx(10.0)


def test_pretrain_loss_uses_only_reconstruction_tail(monkeypatch):
    tensorflow, mvdec = _load_mvdec(monkeypatch)
    monkeypatch.setattr(mvdec, "input_shape", 3)
    monkeypatch.setattr(mvdec, "hidden_units", 2)
    y_true = tensorflow.zeros((2, 3))
    y_pred = tensorflow.constant(
        [
            [100.0, 100.0, 1.0, 2.0, 2.0],
            [200.0, 200.0, 3.0, 4.0, 0.0],
        ]
    )

    loss = mvdec.loss_train_base(y_true, y_pred)

    np.testing.assert_allclose(loss.numpy(), [9.0, 25.0])


def test_release_losses_average_transformed_dimensions(monkeypatch):
    tensorflow, mvdec = _load_mvdec(monkeypatch)
    monkeypatch.setattr(mvdec, "input_shape", 3)
    monkeypatch.setattr(mvdec, "hidden_units", 2)
    y_true = tensorflow.zeros((2, 3))
    y_pred = tensorflow.constant(
        [
            [100.0, 100.0, 1.0, 2.0, 2.0],
            [200.0, 200.0, 3.0, 4.0, 0.0],
        ]
    )

    pretrain_loss = mvdec.release_loss_train_base(y_true, y_pred)
    selected, nonselected = mvdec.release_greedy_loss_components(
        tensorflow.zeros((2, 2)),
        tensorflow.constant([[1.0, 2.0], [3.0, 4.0]]),
        -1,
    )

    np.testing.assert_allclose(pretrain_loss.numpy(), [3.0, 25.0 / 3.0])
    assert float(selected + nonselected) == pytest.approx(7.5)


def test_release_refinement_optimizes_both_reconstructions_and_greedy(monkeypatch):
    tensorflow, mvdec = _load_mvdec(monkeypatch)
    monkeypatch.setattr(mvdec, "input_shape", 2)
    monkeypatch.setattr(mvdec, "hidden_units", 1)
    x_batch = tensorflow.zeros((2, 2))
    y_pred1 = tensorflow.Variable([[10.0, 1.0, 2.0], [20.0, 3.0, 4.0]])
    y_pred2 = tensorflow.Variable([[30.0, 2.0, 1.0], [40.0, 4.0, 3.0]])
    y_true_cluster = tensorflow.zeros((2, 1))
    y_pred_cluster = tensorflow.Variable([[1.0], [2.0]])

    with tensorflow.GradientTape() as tape:
        total_loss, reconstruction_loss, greedy_loss = (
            mvdec.release_reconstruction_greedy_losses(
                x_batch,
                y_pred1,
                y_pred2,
                y_true_cluster,
                y_pred_cluster,
                reconstruction_weight=1.0,
                greedy_weight=1.0,
            )
        )
    view1_gradient, view2_gradient, greedy_gradient = tape.gradient(
        total_loss,
        [y_pred1, y_pred2, y_pred_cluster],
    )

    np.testing.assert_allclose(reconstruction_loss.numpy(), [5.0, 25.0])
    np.testing.assert_allclose(greedy_loss.numpy(), [1.0, 4.0])
    np.testing.assert_allclose(total_loss.numpy(), [6.0, 29.0])
    np.testing.assert_allclose(view1_gradient[:, :1].numpy(), 0.0)
    np.testing.assert_allclose(view2_gradient[:, :1].numpy(), 0.0)
    assert np.all(view1_gradient[:, 1:].numpy() != 0.0)
    assert np.all(view2_gradient[:, 1:].numpy() != 0.0)
    assert np.all(greedy_gradient.numpy() != 0.0)


def test_orthonormal_loss_keeps_the_same_squared_distance(monkeypatch):
    tensorflow, mvdec = _load_mvdec(monkeypatch)
    residual = tensorflow.constant([[2.0, 1.0], [-3.0, 4.0]])
    rotation = tensorflow.constant([[0.0, -1.0], [1.0, 0.0]])
    transformed = tensorflow.matmul(residual, rotation)

    kmeans_loss = tensorflow.reduce_mean(
        mvdec.squared_euclidean_per_sample(
            tensorflow.zeros_like(residual),
            residual,
        )
    )
    orthonormal_loss = tensorflow.reduce_mean(
        mvdec.squared_euclidean_per_sample(
            tensorflow.zeros_like(transformed),
            transformed,
        )
    )

    assert float(orthonormal_loss) == pytest.approx(float(kmeans_loss))


@pytest.mark.parametrize(
    ("eigen_index", "expected_selected", "expected_nonselected"),
    [
        (0, 5.0, 10.0),
        (-1, 10.0, 5.0),
    ],
)
def test_greedy_loss_components_partition_total_loss(
    monkeypatch,
    eigen_index,
    expected_selected,
    expected_nonselected,
):
    tensorflow, mvdec = _load_mvdec(monkeypatch)
    y_true = tensorflow.zeros((2, 2))
    y_pred = tensorflow.constant([[1.0, 2.0], [3.0, 4.0]])

    selected, nonselected = mvdec.greedy_loss_components(
        y_true,
        y_pred,
        eigen_index,
    )
    total = tensorflow.reduce_mean(mvdec.squared_euclidean_per_sample(y_true, y_pred))

    assert float(selected) == pytest.approx(expected_selected)
    assert float(nonselected) == pytest.approx(expected_nonselected)
    assert float(selected + nonselected) == pytest.approx(float(total))


def test_greedy_loss_component_logging_preserves_gradient(monkeypatch):
    tensorflow, mvdec = _load_mvdec(monkeypatch)
    y_true = tensorflow.zeros((2, 3))
    y_pred = tensorflow.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    with tensorflow.GradientTape() as original_tape:
        original_loss = tensorflow.reduce_mean(
            mvdec.squared_euclidean_per_sample(y_true, y_pred)
        )
    original_gradient = original_tape.gradient(original_loss, y_pred)

    with tensorflow.GradientTape() as component_tape:
        selected, nonselected = mvdec.greedy_loss_components(
            y_true,
            y_pred,
            -1,
        )
        component_loss = selected + nonselected
    component_gradient = component_tape.gradient(component_loss, y_pred)

    np.testing.assert_allclose(component_loss.numpy(), original_loss.numpy())
    np.testing.assert_allclose(component_gradient.numpy(), original_gradient.numpy())


@pytest.mark.parametrize("eigen_index", [0, -1])
def test_selected_dimension_only_has_zero_gradient_elsewhere(
    monkeypatch,
    eigen_index,
):
    tensorflow, mvdec = _load_mvdec(monkeypatch)
    embeddings = tensorflow.Variable(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=tensorflow.float32,
    )
    centroids = tensorflow.zeros_like(embeddings)

    with tensorflow.GradientTape() as tape:
        loss = mvdec.selected_direction_greedy_loss(
            embeddings,
            centroids,
            eigen_index,
        )
    gradient = tape.gradient(loss, embeddings).numpy()

    selected_position = eigen_index % embeddings.shape[1]
    nonselected_positions = [
        index for index in range(embeddings.shape[1]) if index != selected_position
    ]
    np.testing.assert_allclose(gradient[:, nonselected_positions], 0.0)
    assert np.any(gradient[:, selected_position] != 0.0)
