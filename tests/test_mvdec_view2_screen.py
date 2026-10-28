import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "mvdec_view2_screen",
    Path(__file__).resolve().parents[1] / "scripts" / "mvdec_view2_screen.py",
)
screen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(screen)


def test_parse_config_fills_defaults_and_casts_types():
    assert screen.parse_config("") == screen.DEFAULTS
    assert screen.parse_config("epochs=1000,lr=1e-4,latent=20") == {
        "epochs": 1000,
        "lr": 1e-4,
        "latent": 20,
        "width": 64,
    }


@pytest.mark.parametrize("text", ["depth=3", "epochs=", "epochs"])
def test_parse_config_rejects_unknown_or_empty_settings(text):
    with pytest.raises(ValueError, match="Unsupported View2 screen setting"):
        screen.parse_config(text)
