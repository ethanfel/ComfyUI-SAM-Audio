import sys
import types

import pytest
import torch
import torch.nn.functional as torch_functional

from sam_audio_comfy import attention


class FakeComfyKitchen(types.ModuleType):
    def __init__(self, available=True):
        super().__init__("comfy_kitchen")
        self.available = available
        self.calls = []

    def int8_attention_is_available(self, device):
        self.checked_device = device
        return self.available

    def int8_attention(self, query, key, value, *, scale=None, attn_mask=None):
        self.calls.append(
            {
                "query": query,
                "key": key,
                "value": value,
                "scale": scale,
                "attn_mask": attn_mask,
            }
        )
        return query + 1


def test_kitchen_backend_is_scoped_to_upstream_modules(monkeypatch):
    kitchen = FakeComfyKitchen()
    monkeypatch.setattr(attention.importlib, "import_module", lambda _: kitchen)

    target = types.ModuleType("sam_audio.fake_transformer")
    target.F = torch_functional
    target.direct_sdpa = torch_functional.scaled_dot_product_attention
    unrelated = types.ModuleType("unrelated.fake_transformer")
    unrelated.F = torch_functional
    monkeypatch.setitem(sys.modules, target.__name__, target)
    monkeypatch.setitem(sys.modules, unrelated.__name__, unrelated)

    query = torch.zeros(1, 2, 3, 4)
    with attention.attention_backend_context("comfy_kitchen"):
        assert target.F is not torch_functional
        assert unrelated.F is torch_functional
        assert target.direct_sdpa is not torch_functional.scaled_dot_product_attention
        actual = target.F.scaled_dot_product_attention(query, query, query, scale=0.5)

    assert torch.equal(actual, query + 1)
    assert kitchen.calls[0]["scale"] == 0.5
    assert target.F is torch_functional
    assert target.direct_sdpa is torch_functional.scaled_dot_product_attention


def test_kitchen_backend_preserves_pytorch_for_causal_attention(monkeypatch):
    kitchen = FakeComfyKitchen()
    monkeypatch.setattr(attention.importlib, "import_module", lambda _: kitchen)
    target = types.ModuleType("core.fake_transformer")
    target.F = torch_functional
    monkeypatch.setitem(sys.modules, target.__name__, target)

    query = torch.randn(1, 1, 3, 4)
    with attention.attention_backend_context("comfy_kitchen"):
        actual = target.F.scaled_dot_product_attention(
            query, query, query, is_causal=True
        )
    expected = torch_functional.scaled_dot_product_attention(
        query, query, query, is_causal=True
    )

    assert torch.allclose(actual, expected)
    assert kitchen.calls == []


def test_kitchen_backend_contiguates_the_head_dimension(monkeypatch):
    kitchen = FakeComfyKitchen()
    monkeypatch.setattr(attention.importlib, "import_module", lambda _: kitchen)
    target = types.ModuleType("sam_audio.fake_transformer")
    target.F = torch_functional
    monkeypatch.setitem(sys.modules, target.__name__, target)
    strided = torch.zeros(1, 2, 3, 8)[..., ::2]
    assert strided.stride(-1) == 2

    with attention.attention_backend_context("comfy_kitchen"):
        actual = target.F.scaled_dot_product_attention(strided, strided, strided)

    assert actual.shape == strided.shape
    assert all(
        kitchen.calls[0][name].stride(-1) == 1
        for name in ("query", "key", "value")
    )


def test_kitchen_backend_validation(monkeypatch):
    kitchen = FakeComfyKitchen(available=False)
    monkeypatch.setattr(attention.importlib, "import_module", lambda _: kitchen)

    with pytest.raises(RuntimeError, match="not available"):
        attention.validate_attention_backend(
            "comfy_kitchen", torch.device("cuda", 0)
        )

    attention.validate_attention_backend("pytorch", torch.device("cpu"))
