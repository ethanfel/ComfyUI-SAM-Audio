from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torchaudio


@dataclass(frozen=True)
class AudioChunk:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def plan_audio_chunks(
    total_samples: int,
    sample_rate: int,
    chunk_duration: float,
    chunk_overlap: float,
) -> list[AudioChunk]:
    """Plan overlapping sample ranges; zero duration keeps the original full pass."""
    if total_samples < 1:
        raise ValueError("total_samples must be positive")
    if sample_rate < 1:
        raise ValueError("sample_rate must be positive")
    if chunk_duration < 0:
        raise ValueError("chunk_duration cannot be negative")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_duration == 0:
        return [AudioChunk(0, total_samples)]

    chunk_samples = round(chunk_duration * sample_rate)
    overlap_samples = round(chunk_overlap * sample_rate)
    if chunk_samples < 1:
        raise ValueError("chunk_duration is shorter than one audio sample")
    if overlap_samples >= chunk_samples:
        raise ValueError("chunk_overlap must be shorter than chunk_duration")
    if total_samples <= chunk_samples:
        return [AudioChunk(0, total_samples)]

    hop_samples = chunk_samples - overlap_samples
    chunks = []
    start = 0
    while True:
        end = min(start + chunk_samples, total_samples)
        chunks.append(AudioChunk(start, end))
        if end == total_samples:
            return chunks
        start += hop_samples


def crossfade_audio_chunks(
    chunks: Sequence[tuple[AudioChunk, torch.Tensor]], total_samples: int
) -> torch.Tensor:
    """Reassemble [batch, channels, samples] chunks with normalized linear fades."""
    if not chunks:
        raise ValueError("chunks cannot be empty")

    first = chunks[0][1]
    if first.ndim != 3:
        raise ValueError("chunk waveforms must have [batch, channels, samples] shape")
    output = first.new_zeros((*first.shape[:-1], total_samples))
    weights = first.new_zeros(total_samples)

    for index, (chunk, waveform) in enumerate(chunks):
        if waveform.ndim != 3 or waveform.shape[:-1] != first.shape[:-1]:
            raise ValueError("all chunk waveforms must have matching batch and channels")
        if waveform.shape[-1] != chunk.length:
            raise ValueError("chunk waveform length does not match its sample range")

        window = waveform.new_ones(chunk.length)
        if index > 0:
            fade_in = max(0, chunks[index - 1][0].end - chunk.start)
            if fade_in:
                window[:fade_in] *= torch.linspace(
                    0, 1, fade_in + 2, dtype=window.dtype, device=window.device
                )[1:-1]
        if index + 1 < len(chunks):
            fade_out = max(0, chunk.end - chunks[index + 1][0].start)
            if fade_out:
                window[-fade_out:] *= torch.linspace(
                    1, 0, fade_out + 2, dtype=window.dtype, device=window.device
                )[1:-1]

        output[..., chunk.start : chunk.end] += waveform * window
        weights[chunk.start : chunk.end] += window

    if torch.any(weights == 0):
        raise RuntimeError("chunk plan left uncovered audio samples")
    return output / weights


def slice_visual_frames(
    frames: torch.Tensor, chunk: AudioChunk, total_samples: int
) -> torch.Tensor:
    """Select frames whose uniformly spaced time range intersects an audio chunk."""
    if frames.ndim < 1 or frames.shape[0] < 1:
        raise ValueError("frames must contain at least one frame")
    if total_samples < 1:
        raise ValueError("total_samples must be positive")
    if frames.shape[0] == 1 or chunk == AudioChunk(0, total_samples):
        return frames

    frame_count = frames.shape[0]
    start = math.floor(chunk.start * frame_count / total_samples)
    end = math.ceil(chunk.end * frame_count / total_samples)
    start = min(start, frame_count - 1)
    end = min(max(start + 1, end), frame_count)
    return frames[start:end]


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
