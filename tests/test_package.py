import json
import sys
import types
from pathlib import Path

import pytest
import torch

from sam_audio_comfy import (
    audiotools_compat,
    torchcodec_compat,
    xformers_compat,
)


def test_node_does_not_require_xformers():
    root = Path(__file__).parents[1]
    dependency_files = [
        root / "requirements.txt",
        root / "pyproject.toml",
    ]
    for path in dependency_files:
        dependency_lines = [
            line
            for line in path.read_text().lower().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        assert not any("xformers" in line for line in dependency_lines)


def test_node_does_not_require_torchcodec():
    root = Path(__file__).parents[1]
    for filename in ("requirements.txt", "pyproject.toml"):
        dependency_lines = [
            line
            for line in (root / filename).read_text().lower().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        assert not any("torchcodec" in line for line in dependency_lines)


def test_example_sampler_widgets_include_seed_control():
    root = Path(__file__).parents[1]
    expected = {
        "text_separation_workflow.json": (
            "SAMAudioTextSeparator",
            ["man speaking", False, 0, "fixed", 32],
        ),
        "span_separation_workflow.json": (
            "SAMAudioSpanSeparator",
            ["car honking", 0, "fixed", 32],
        ),
        "visual_separation_workflow.json": (
            "SAMAudioVisualSeparator",
            ["", 0, "fixed", 32],
        ),
    }
    for filename, (node_type, widget_values) in expected.items():
        workflow = json.loads((root / "examples" / filename).read_text())
        sampler = next(node for node in workflow["nodes"] if node["type"] == node_type)
        assert sampler["widgets_values"] == widget_values


def test_upstream_installer_uses_no_deps():
    root = Path(__file__).parents[1]
    installer = (root / "install.py").read_text().lower()
    assert 'run_pip("--no-deps", package)' in installer
    assert "perception-models @ git+" in installer
    assert '"xformers @' not in installer


def test_xformers_compatibility_uses_torch_sdpa(monkeypatch):
    monkeypatch.delitem(sys.modules, "xformers", raising=False)
    monkeypatch.delitem(sys.modules, "xformers.ops", raising=False)
    monkeypatch.delitem(sys.modules, "xformers.ops.fmha", raising=False)
    with xformers_compat.xformers_import_compatibility():
        fmha = sys.modules["xformers.ops.fmha"]
        query = torch.randn(2, 5, 3, 4)
        key = torch.randn(2, 5, 3, 4)
        value = torch.randn(2, 5, 3, 4)
        actual = fmha.memory_efficient_attention(query, key, value)
        expected = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        ).transpose(1, 2)

    assert torch.allclose(actual, expected)
    assert "xformers" not in sys.modules


def test_audiotools_compatibility_is_import_only_and_scoped(monkeypatch):
    monkeypatch.delitem(sys.modules, "audiotools", raising=False)
    monkeypatch.delitem(sys.modules, "audiotools.ml", raising=False)

    with audiotools_compat.audiotools_import_compatibility():
        audiotools = sys.modules["audiotools"]
        model = audiotools.ml.BaseModel()
        assert isinstance(model, torch.nn.Module)
        assert audiotools.ml.BaseModel.INTERN == []

    assert "audiotools" not in sys.modules


def test_torchcodec_compatibility_is_import_only_and_scoped(monkeypatch):
    monkeypatch.delitem(sys.modules, "torchcodec", raising=False)
    monkeypatch.delitem(sys.modules, "torchcodec.decoders", raising=False)
    monkeypatch.delitem(sys.modules, "torchcodec.encoders", raising=False)

    with torchcodec_compat.torchcodec_import_compatibility():
        decoders = sys.modules["torchcodec.decoders"]
        assert decoders.AudioDecoder is torchcodec_compat.AudioDecoder
        assert decoders.VideoDecoder is torchcodec_compat.VideoDecoder
        with pytest.raises(RuntimeError, match="decoded ComfyUI"):
            decoders.AudioDecoder("input.wav")

    assert "torchcodec" not in sys.modules
    assert "torchcodec.decoders" not in sys.modules


def test_existing_torchcodec_modules_are_restored(monkeypatch):
    original = types.ModuleType("torchcodec")
    original_decoders = types.ModuleType("torchcodec.decoders")
    monkeypatch.setitem(sys.modules, "torchcodec", original)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", original_decoders)

    with torchcodec_compat.torchcodec_import_compatibility():
        assert sys.modules["torchcodec"] is not original
        assert sys.modules["torchcodec.decoders"] is not original_decoders

    assert sys.modules["torchcodec"] is original
    assert sys.modules["torchcodec.decoders"] is original_decoders
