import hashlib
from pathlib import Path

import pytest
import torch

from sam_audio_comfy import runtime
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


def test_gated_download_uses_pinned_verified_mirror(tmp_path, monkeypatch):
    class GatedRepoError(Exception):
        pass

    checkpoint = b"verified mirror checkpoint"
    spec = runtime.MirrorSpec(
        repo_id="mirror/sam-audio-small",
        revision="pinned-revision",
        checkpoint_size=len(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint).hexdigest(),
    )
    monkeypatch.setitem(runtime.MIRROR_MODELS, "facebook/sam-audio-small", spec)
    calls = []

    class FakeHub:
        @staticmethod
        def snapshot_download(**kwargs):
            calls.append(kwargs)
            if kwargs["repo_id"] == "facebook/sam-audio-small":
                raise GatedRepoError("access denied")
            destination = Path(kwargs["local_dir"])
            (destination / "config.json").write_text("{}")
            (destination / "checkpoint.pt").write_bytes(checkpoint)
            (destination / "LICENSE").write_text("SAM License")
            return str(destination)

    monkeypatch.setattr(runtime.importlib, "import_module", lambda _: FakeHub)
    result = runtime._download_official_model(
        "facebook/sam-audio-small", tmp_path
    )

    assert result == tmp_path / "sam-audio-small"
    assert calls[1]["repo_id"] == spec.repo_id
    assert calls[1]["revision"] == spec.revision
    assert calls[1]["allow_patterns"] == [
        "config.json",
        "checkpoint.pt",
        "LICENSE",
    ]
    assert (result / runtime.MIRROR_MARKER).read_text() == (
        f"{len(checkpoint)} {spec.checkpoint_sha256}\n"
    )


def test_non_gate_download_error_does_not_use_mirror(tmp_path, monkeypatch):
    calls = []

    class FakeHub:
        @staticmethod
        def snapshot_download(**kwargs):
            calls.append(kwargs)
            raise ConnectionError("offline")

    monkeypatch.setattr(runtime.importlib, "import_module", lambda _: FakeHub)
    with pytest.raises(RuntimeError, match="Could not download"):
        runtime._download_official_model("facebook/sam-audio-small", tmp_path)

    assert [call["repo_id"] for call in calls] == ["facebook/sam-audio-small"]


def test_mirror_checksum_mismatch_is_rejected(tmp_path, monkeypatch):
    class GatedRepoError(Exception):
        pass

    spec = runtime.MirrorSpec(
        repo_id="mirror/sam-audio-small",
        revision="pinned-revision",
        checkpoint_size=3,
        checkpoint_sha256=hashlib.sha256(b"good").hexdigest(),
    )
    monkeypatch.setitem(runtime.MIRROR_MODELS, "facebook/sam-audio-small", spec)

    class FakeHub:
        @staticmethod
        def snapshot_download(**kwargs):
            if kwargs["repo_id"] == "facebook/sam-audio-small":
                raise GatedRepoError("access denied")
            destination = Path(kwargs["local_dir"])
            (destination / "config.json").write_text("{}")
            (destination / "checkpoint.pt").write_bytes(b"bad")
            (destination / "LICENSE").write_text("SAM License")
            return str(destination)

    monkeypatch.setattr(runtime.importlib, "import_module", lambda _: FakeHub)
    with pytest.raises(RuntimeError, match="verified public mirror"):
        runtime._download_official_model("facebook/sam-audio-small", tmp_path)


def test_local_checkpoint_falls_back_for_legacy_hub_signature(tmp_path):
    calls = []

    class FakeSAMAudio:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("public", model_id, kwargs))
            raise TypeError(
                "BaseModel._from_pretrained() missing 2 required keyword-only "
                "arguments: 'proxies' and 'resume_download'"
            )

        @classmethod
        def _from_pretrained(cls, **kwargs):
            calls.append(("legacy", kwargs))
            return "model"

    assert runtime._load_sam_audio_checkpoint(FakeSAMAudio, tmp_path) == "model"
    assert calls[1][0] == "legacy"
    assert calls[1][1]["model_id"] == str(tmp_path)
    assert calls[1][1]["local_files_only"] is True
    assert calls[1][1]["proxies"] is None
    assert calls[1][1]["resume_download"] is False
    assert calls[1][1]["strict"] is True
    assert calls[1][1]["text_ranker"] is None
    assert calls[1][1]["visual_ranker"] is None


def test_checkpoint_loader_uses_upstream_weight_loading_path(tmp_path):
    class UpstreamStyleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def load_state_dict(self, state_dict, strict=True):
            # SAM-Audio's pinned implementation has no false branch.
            if strict:
                return super().load_state_dict(state_dict, strict=False)
            return None

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            model = cls()
            model.load_state_dict({"weight": torch.ones(1)}, strict=kwargs["strict"])
            return model

    model = runtime._load_sam_audio_checkpoint(UpstreamStyleModel, tmp_path)

    assert torch.equal(model.weight, torch.ones(1))


def test_support_snapshot_uses_local_cache_without_network(monkeypatch):
    calls = []

    class FakeHub:
        @staticmethod
        def snapshot_download(**kwargs):
            calls.append(kwargs)
            return "/cached/support-model"

    monkeypatch.setattr(runtime.importlib, "import_module", lambda _: FakeHub)
    spec = runtime.SupportModelSpec("owner/model", "pinned", ("config.json",))

    assert runtime._cached_support_snapshot(spec) == "/cached/support-model"
    assert calls == [
        {
            "repo_id": "owner/model",
            "revision": "pinned",
            "allow_patterns": ["config.json"],
            "local_files_only": True,
        }
    ]


def test_support_snapshot_downloads_only_when_cache_is_absent(monkeypatch):
    calls = []

    class FakeHub:
        @staticmethod
        def snapshot_download(**kwargs):
            calls.append(kwargs)
            if kwargs.get("local_files_only"):
                raise FileNotFoundError("not cached")
            return "/downloaded/support-model"

    monkeypatch.setattr(runtime.importlib, "import_module", lambda _: FakeHub)
    spec = runtime.SupportModelSpec("owner/model", "pinned")

    assert runtime._cached_support_snapshot(spec) == "/downloaded/support-model"
    assert [call.get("local_files_only", False) for call in calls] == [True, False]


def test_model_kwargs_replace_support_ids_with_local_snapshots(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(
        '{"text_encoder":{"name":"t5-base","dim":768},'
        '"span_predictor":"pe-a-frame-large"}'
    )
    monkeypatch.setattr(
        runtime,
        "_cached_support_snapshot",
        lambda spec: f"/cache/{spec.repo_id.replace('/', '--')}",
    )

    kwargs = runtime._sam_audio_model_kwargs(tmp_path)

    assert kwargs["text_encoder"] == {
        "name": "/cache/t5-base",
        "dim": 768,
    }
    assert kwargs["span_predictor"] == "/cache/facebook--pe-a-frame-large"
    assert kwargs["text_ranker"] is None
    assert kwargs["visual_ranker"] is None
    assert kwargs["strict"] is True


def test_local_checkpoint_does_not_hide_unrelated_type_errors(tmp_path):
    class FakeSAMAudio:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            raise TypeError("bad model config")

    with pytest.raises(TypeError, match="bad model config"):
        runtime._load_sam_audio_checkpoint(FakeSAMAudio, tmp_path)


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
