import importlib
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("input_dim", "latent_dim", "base_units", "expected_concats"),
    [
        (2000, 10, 64, [768, 384, 192, 96]),
        (13, 10, 64, [768, 384, 192, 96]),
        (7, 5, 32, [384, 192, 96, 48]),
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
        latent_dim + input_dim,
    ]
    assert concat_widths == expected_concats
    assert model.output_shape == (None, latent_dim + input_dim)
    assert isinstance(model.layers[-1], tensorflow.keras.layers.Dense)
    assert model.layers[-1].units == latent_dim + input_dim
