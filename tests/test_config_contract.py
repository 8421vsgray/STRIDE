"""Dependency-light checks for the NEWSTRIDE YAML contract."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ["scene15.yaml", "handwritten.yaml", "ccv20.yaml", "food101.yaml"]
WARMUP_EPOCHS = {name: (30 if name == "food101.yaml" else 20) for name in CONFIGS}


def test_all_configs_are_yaml_and_have_explicit_views():
    for name in CONFIGS:
        config = yaml.safe_load((ROOT / "config" / name).read_text(encoding="utf-8"))
        assert config["n_views"] == 2
        assert len(config["view_indices"]) == 2
        assert len(config["encoder_dim"]) == 2
        assert config["diagonal_epochs"] == WARMUP_EPOCHS[name]
        assert config["init"] == "diagonal"


def test_view_contracts():
    expected = {
        "scene15.yaml": [1, 0],
        "handwritten.yaml": [0, 1],
        "ccv20.yaml": [1, 2],
        "food101.yaml": [0, 1],
    }
    for name, views in expected.items():
        config = yaml.safe_load((ROOT / "config" / name).read_text(encoding="utf-8"))
        assert config["view_indices"] == views
