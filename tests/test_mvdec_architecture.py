import importlib
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("input_dim", "latent_dim", "base_units", "expected_concats"),
    [
        (2000, 10, 64, [768, 384, 192, 96, 2010]),
        (13, 10, 64, [768, 384, 192, 96, 23]),
        (7, 5, 32, [384, 192, 96, 48, 12]),
    ],
)
def test_view2_matches_paper_skip_dimensions(
    monkeypatch,
    input_dim,
    latent_dim,
    base_units,
    expected_concats,
):
    tensorflow = pytest.importorskip("tensorflow")
    representation_dir = (
        Path(__file__).resolve().parents[1] / "src" / "representation_learning"
    )
    monkeypatch.syspath_prepend(str(representation_dir))
    mvdec = importlib.import_module("MVDEC_dense")
    monkeypatch.setattr(mvdec, "input_shape", input_dim)
    monkeypatch.setattr(mvdec, "hidden_units", latent_dim)
    monkeypatch.setattr(mvdec, "view2_base_units", base_units)

    model = mvdec.model_view2(load_weights=False)
    dense_units = [
        layer.units
        for layer in model.layers
        if isinstance(layer, tensorflow.keras.layers.Dense)
    ]
    concat_widths = [
        int(layer.output.shape[-1])
        for layer in model.layers
        if isinstance(layer, tensorflow.keras.layers.Concatenate)
    ]

    assert dense_units == [
        base_units,
        base_units,
        2 * base_units,
        2 * base_units,
        4 * base_units,
        4 * base_units,
        8 * base_units,
        8 * base_units,
        16 * base_units,
        8 * base_units,
        4 * base_units,
        8 * base_units,
        4 * base_units,
        2 * base_units,
        4 * base_units,
        2 * base_units,
        base_units,
        2 * base_units,
        base_units,
        base_units // 2,
        base_units,
        latent_dim,
        input_dim,
    ]
    assert concat_widths == expected_concats
    assert model.output_shape == (None, latent_dim + input_dim)
    assert model.get_layer("view2_latent").units == latent_dim
    assert model.get_layer("view2_reconstruction").units == input_dim


def test_view2_pretraining_updates_post_skip_latent(monkeypatch):
    """Reconstruction pretraining must propagate through the View2 latent head."""

    tensorflow = pytest.importorskip("tensorflow")
    representation_dir = (
        Path(__file__).resolve().parents[1] / "src" / "representation_learning"
    )
    monkeypatch.syspath_prepend(str(representation_dir))
    mvdec = importlib.import_module("MVDEC_dense")
    monkeypatch.setattr(mvdec, "input_shape", 13)
    monkeypatch.setattr(mvdec, "hidden_units", 10)
    monkeypatch.setattr(mvdec, "view2_base_units", 64)
    tensorflow.keras.utils.set_random_seed(42)
    model = mvdec.model_view2(load_weights=False)
    inputs = tensorflow.ones((4, 13), dtype=tensorflow.float32)

    with tensorflow.GradientTape() as tape:
        outputs = model(inputs)
        loss = tensorflow.reduce_mean(mvdec.release_loss_train_base(inputs, outputs))
    gradient = tape.gradient(loss, model.get_layer("view2_latent").kernel)

    assert gradient is not None
    assert float(tensorflow.linalg.norm(gradient)) > 0.0


def test_view2_encoder_bottleneck_precedes_skip_decoder(monkeypatch):
    """The diagnostic View2 must encode to latent space before decoding."""

    tensorflow = pytest.importorskip("tensorflow")
    representation_dir = (
        Path(__file__).resolve().parents[1] / "src" / "representation_learning"
    )
    monkeypatch.syspath_prepend(str(representation_dir))
    mvdec = importlib.import_module("MVDEC_dense")
    monkeypatch.setattr(mvdec, "input_shape", 13)
    monkeypatch.setattr(mvdec, "hidden_units", 10)
    monkeypatch.setattr(mvdec, "view2_base_units", 64)
    tensorflow.keras.utils.set_random_seed(42)
    model = mvdec.model_view2(
        load_weights=False,
        architecture_id=mvdec.VIEW2_ENCODER_BOTTLENECK_ARCHITECTURE_ID,
    )
    latent = model.get_layer("view2_latent")
    reconstruction = model.get_layer("view2_reconstruction")

    assert latent.input.shape[-1] == 1024
    assert reconstruction.input.shape[-1] == 64
    assert model.output_shape == (None, 23)

    inputs = tensorflow.ones((4, 13), dtype=tensorflow.float32)
    with tensorflow.GradientTape() as tape:
        outputs = model(inputs)
        loss = tensorflow.reduce_mean(mvdec.release_loss_train_base(inputs, outputs))
    gradient = tape.gradient(loss, latent.kernel)

    assert gradient is not None
    assert float(tensorflow.linalg.norm(gradient)) > 0.0


def test_view2_direct_joint_head_matches_figure_output(monkeypatch):
    """The Fig. 2 diagnostic must emit latent and reconstruction in one head."""

    tensorflow = pytest.importorskip("tensorflow")
    representation_dir = (
        Path(__file__).resolve().parents[1] / "src" / "representation_learning"
    )
    monkeypatch.syspath_prepend(str(representation_dir))
    mvdec = importlib.import_module("MVDEC_dense")
    monkeypatch.setattr(mvdec, "input_shape", 13)
    monkeypatch.setattr(mvdec, "hidden_units", 10)
    monkeypatch.setattr(mvdec, "view2_base_units", 64)
    tensorflow.keras.utils.set_random_seed(42)
    model = mvdec.model_view2(
        load_weights=False,
        architecture_id=mvdec.VIEW2_DIRECT_JOINT_HEAD_ARCHITECTURE_ID,
    )
    joint_head = model.get_layer("view2_joint_head")

    assert joint_head.input.shape[-1] == 64
    assert joint_head.units == 23
    assert model.output_shape == (None, 23)

    inputs = tensorflow.ones((4, 13), dtype=tensorflow.float32)
    with tensorflow.GradientTape() as tape:
        outputs = model(inputs)
        loss = tensorflow.reduce_mean(mvdec.release_loss_train_base(inputs, outputs))
    gradient = tape.gradient(loss, joint_head.kernel)

    assert mvdec.latent_embedding(outputs).shape == (4, 10)
    assert mvdec.reconstruction_output(outputs).shape == (4, 13)
    assert gradient is not None
    assert float(tensorflow.linalg.norm(gradient)) > 0.0
