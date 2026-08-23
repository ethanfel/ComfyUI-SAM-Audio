import importlib.util
import sys
import types
from pathlib import Path

import torch


def _load_node_package(monkeypatch, tmp_path):
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    management = types.ModuleType("comfy.model_management")
    management.load_models_gpu = lambda models, force_full_load=False: None
    patcher = types.ModuleType("comfy.model_patcher")
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = str(tmp_path)
    folder_paths._paths = []
    folder_paths.add_model_folder_path = (
        lambda name, path, is_default=False: folder_paths._paths.append(path)
        if path not in folder_paths._paths
        else None
    )
    folder_paths.get_folder_paths = lambda name: folder_paths._paths

    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)
    monkeypatch.setitem(sys.modules, "comfy.model_patcher", patcher)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)

    root = Path(__file__).parents[1]
    package_name = "comfyui_sam_audio_node_test"
    spec = importlib.util.spec_from_file_location(
        package_name,
        root / "__init__.py",
        submodule_search_locations=[str(root)],
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, package_name, module)
    spec.loader.exec_module(module)
    return module, management


class FakeBatch:
    def __init__(self, audios, descriptions, anchors, masked_videos):
        self.audios = audios
        self.descriptions = descriptions
        self.anchors = anchors
        self.masked_videos = masked_videos

    def to(self, device):
        return self


class FakeProcessor:
    audio_sampling_rate = 8

    def __call__(self, audios, descriptions, anchors=None, masked_videos=None):
        return FakeBatch(audios, descriptions, anchors, masked_videos)

    def mask_videos(self, videos, masks):
        return [
            video * mask.eq(0) for video, mask in zip(videos, masks, strict=False)
        ]


class FakeModel(torch.nn.Module):
    def separate(self, batch, **kwargs):
        target = [row.mean(0) for row in batch.audios]
        return types.SimpleNamespace(target=target, residual=[-row for row in target])


def test_text_node_runs_full_audio_batch_with_standard_shapes(monkeypatch, tmp_path):
    package, management = _load_node_package(monkeypatch, tmp_path)
    nodes = sys.modules[f"{package.__name__}.nodes"]
    runtime = sys.modules[f"{package.__name__}.sam_audio_comfy.runtime"]
    pipeline = runtime.SAMAudioPipeline(
        model=FakeModel(),
        processor=FakeProcessor(),
        patcher=object(),
        device=torch.device("cpu"),
        source="fake",
    )

    target, residual = nodes.SAMAudioTextSeparator().separate(
        pipeline=pipeline,
        audio={"waveform": torch.ones(2, 2, 8), "sample_rate": 8},
        description="test sound",
        predict_spans=False,
        seed=7,
        inference_steps=32,
    )

    assert target["waveform"].shape == (2, 1, 8)
    assert residual["waveform"].shape == (2, 1, 8)
    assert target["sample_rate"] == 8
    loader_inputs = package.NODE_CLASS_MAPPINGS[
        "SAMAudioPipelineLoader"
    ].INPUT_TYPES()["required"]
    assert loader_inputs["attention_backend"][0] == ["pytorch", "comfy_kitchen"]
    assert set(package.NODE_CLASS_MAPPINGS) == {
        "SAMAudioPipelineLoader",
        "SAMAudioSpanPrompt",
        "SAMAudioTextSeparator",
        "SAMAudioSpanSeparator",
        "SAMAudioVisualSeparator",
    }
    assert management.load_models_gpu is not None
