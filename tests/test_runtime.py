from pathlib import Path

import pytest
import torch

from sam_audio_comfy.runtime import (
    SpanPrompt,
    discover_local_models,
    ode_options,
    register_model_folder,
    validate_spans,
)


class FakeFolderPaths:
    def __init__(self, root: Path):
        self.models_dir = str(root)
        self.paths = []

    def add_model_folder_path(self, name, path, is_default=False):
        assert name == "sam_audio"
        if path not in self.paths:
            self.paths.append(path)

    def get_folder_paths(self, name):
        assert name == "sam_audio"
        return self.paths


def _make_model(path: Path):
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    (path / "checkpoint.pt").write_bytes(b"checkpoint")


def test_register_and_discover_nested_local_models(tmp_path):
    folders = FakeFolderPaths(tmp_path)
    model_root = register_model_folder(folders)
    _make_model(model_root / "custom" / "my-model")

    assert discover_local_models(folders) == {
        "local/custom/my-model": model_root / "custom" / "my-model"
    }


def test_official_download_directories_are_not_duplicated_as_local(tmp_path):
    folders = FakeFolderPaths(tmp_path)
    model_root = register_model_folder(folders)
    _make_model(model_root / "sam-audio-large")

    assert discover_local_models(folders) == {}


def test_ode_options_match_upstream_default():
    assert ode_options(32) == {
        "method": "midpoint",
        "options": {"step_size": 2.0 / 32},
    }
    with pytest.raises(ValueError, match="even"):
        ode_options(31)


def test_span_validation_requires_positive_and_checks_duration():
    validate_spans((SpanPrompt("+", 0.25, 0.75),), duration=1.0)

    with pytest.raises(ValueError, match="positive"):
        validate_spans((SpanPrompt("-", 0.25, 0.75),), duration=1.0)
    with pytest.raises(ValueError, match="only 1.000s"):
        validate_spans((SpanPrompt("+", 0.25, 1.25),), duration=1.0)


def test_seed_context_restores_cpu_rng():
    from sam_audio_comfy.runtime import seeded_inference

    torch.manual_seed(123)
    expected_first = torch.rand(1)
    expected_second = torch.rand(1)

    torch.manual_seed(123)
    actual_first = torch.rand(1)
    with seeded_inference(999, torch.device("cpu")):
        inside_first = torch.rand(1)
    actual_second = torch.rand(1)

    torch.manual_seed(999)
    assert torch.equal(inside_first, torch.rand(1))
    assert torch.equal(actual_first, expected_first)
    assert torch.equal(actual_second, expected_second)
