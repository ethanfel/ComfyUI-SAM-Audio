from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
import torchaudio


def _float_waveform(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.dtype.is_floating_point:
        return waveform.float()
    if waveform.dtype == torch.int16:
        return waveform.float().div_(2**15)
    if waveform.dtype == torch.int32:
        return waveform.float().div_(2**31)
    if waveform.dtype == torch.uint8:
        return waveform.float().sub_(128).div_(128)
    raise TypeError(f"Unsupported audio dtype: {waveform.dtype}")


def prepare_audio(
    audio: Mapping[str, object], target_sample_rate: int
) -> tuple[list[torch.Tensor], int]:
    """Convert a ComfyUI AUDIO value to SAM-Audio's list-of-waveforms form."""
    if not isinstance(audio, Mapping):
        raise TypeError("audio must be a ComfyUI AUDIO value")

    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if not isinstance(waveform, torch.Tensor):
        raise TypeError("audio['waveform'] must be a torch.Tensor")
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("audio['sample_rate'] must be a positive integer")

    if waveform.ndim == 1:
        waveform = waveform[None, None, :]
    elif waveform.ndim == 2:
        waveform = waveform[None, :, :]
    elif waveform.ndim != 3:
        raise ValueError(
            "audio waveform must have shape [batch, channels, samples], "
            "[channels, samples], or [samples]"
        )
    if waveform.shape[0] < 1 or waveform.shape[1] < 1 or waveform.shape[2] < 1:
        raise ValueError("audio waveform cannot have an empty dimension")

    # Audio values in ComfyUI conventionally live on the CPU. Keeping preprocessing
    # there also prevents the input tensor from occupying VRAM alongside SAM-Audio.
    waveform = _float_waveform(waveform.detach()).cpu().contiguous()
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, target_sample_rate
        )

    target_length = waveform.shape[-1]
    return [row.contiguous() for row in waveform], target_length


def result_to_audio(
    waveforms: Sequence[torch.Tensor], sample_rate: int, target_length: int
) -> dict[str, object]:
    """Pack SAM-Audio's variable-length result list into a ComfyUI AUDIO value."""
    if not waveforms:
        raise RuntimeError("SAM-Audio returned no waveforms")

    rows = []
    for waveform in waveforms:
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.as_tensor(waveform)
        waveform = waveform.detach().float().cpu()
        if waveform.ndim == 2:
            if waveform.shape[0] != 1:
                waveform = waveform.mean(dim=0, keepdim=True)
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            raise RuntimeError(
                f"SAM-Audio returned an unsupported waveform shape: {tuple(waveform.shape)}"
            )

        waveform = waveform[..., :target_length]
        if waveform.shape[-1] < target_length:
            waveform = F.pad(waveform, (0, target_length - waveform.shape[-1]))
        rows.append(waveform)

    return {
        "waveform": torch.stack(rows, dim=0).contiguous(),
        "sample_rate": sample_rate,
    }


def prepare_visual_prompt(
    images: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert Comfy IMAGE/MASK values to SAM-Audio NCHW uint8/bool tensors.

    A white ComfyUI mask marks the sound-producing object. SAM-Audio represents
    that prompt by blacking out the same pixels before visual encoding.
    """
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("images must have ComfyUI IMAGE shape [frames, height, width, channels]")
    if images.shape[-1] not in (1, 3, 4):
        raise ValueError("images must have 1, 3, or 4 channels")
    if images.shape[0] < 1:
        raise ValueError("images must contain at least one frame")

    images = images.detach().float().cpu()
    if images.shape[-1] == 1:
        images = images.expand(-1, -1, -1, 3)
    elif images.shape[-1] == 4:
        images = images[..., :3]
    frames = images.clamp(0, 1).mul(255).round().to(torch.uint8)
    frames = frames.permute(0, 3, 1, 2).contiguous()

    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a ComfyUI MASK tensor")
    mask = mask.detach().float().cpu()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    elif mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    elif mask.ndim != 3:
        raise ValueError("mask must have shape [frames, height, width] or [height, width]")

    if mask.shape[0] == 1 and frames.shape[0] > 1:
        mask = mask.expand(frames.shape[0], -1, -1)
    elif mask.shape[0] != frames.shape[0]:
        raise ValueError(
            f"mask has {mask.shape[0]} frames but images has {frames.shape[0]}; "
            "use one mask for every frame or provide matching frame counts"
        )

    if tuple(mask.shape[-2:]) != tuple(frames.shape[-2:]):
        mask = F.interpolate(
            mask.unsqueeze(1), size=frames.shape[-2:], mode="nearest"
        ).squeeze(1)
    mask = mask.gt(0.5).unsqueeze(1).contiguous()
    return frames, mask
