from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import comfy.model_management as model_management
import comfy.model_patcher as model_patcher
import comfy.utils as comfy_utils
import folder_paths
import torch

from .sam_audio_comfy.attention import ATTENTION_BACKENDS, attention_backend_context
from .sam_audio_comfy.audio import (
    AudioChunk,
    crossfade_audio_chunks,
    plan_audio_chunks,
    prepare_audio,
    prepare_visual_prompt,
    result_to_audio,
    slice_visual_frames,
)
from .sam_audio_comfy.runtime import (
    SAMAudioPipeline,
    SpanPrompt,
    load_pipeline,
    model_choices,
    ode_options,
    register_model_folder,
    seeded_inference,
    validate_spans,
)

CATEGORY = "audio/SAM-Audio"
PIPELINE_TYPE = "SAM_AUDIO_PIPELINE"
SPANS_TYPE = "SAM_AUDIO_SPANS"
MAX_SEED = 2**64 - 1

register_model_folder(folder_paths)


def _sampling_inputs() -> dict[str, tuple[Any, ...]]:
    return {
        "seed": (
            "INT",
            {
                "default": 0,
                "min": 0,
                "max": MAX_SEED,
                "control_after_generate": True,
                "tooltip": "Controls SAM-Audio's initial noise for reproducible separation.",
            },
        ),
        "inference_steps": (
            "INT",
            {
                "default": 32,
                "min": 2,
                "max": 128,
                "step": 2,
                "tooltip": "Number of midpoint function evaluations. Higher values are slower and may improve quality.",
            },
        ),
    }


def _chunking_inputs() -> dict[str, tuple[Any, ...]]:
    return {
        "chunk_duration": (
            "FLOAT",
            {
                "default": 10.0,
                "min": 0.0,
                "max": 3600.0,
                "step": 0.5,
                "tooltip": "Seconds processed per pass. Use 0 to process the entire clip at once.",
            },
        ),
        "chunk_overlap": (
            "FLOAT",
            {
                "default": 1.0,
                "min": 0.0,
                "max": 60.0,
                "step": 0.1,
                "tooltip": "Seconds shared by adjacent chunks for a smooth crossfade.",
            },
        ),
    }


def _as_result_list(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value] if value.ndim == 1 else list(value)
    if isinstance(value, Sequence):
        return list(value)
    raise RuntimeError(f"SAM-Audio returned an unexpected result type: {type(value).__name__}")


def _spans_for_chunk(
    spans: tuple[SpanPrompt, ...], chunk: AudioChunk, sample_rate: int
) -> tuple[SpanPrompt, ...]:
    chunk_start = chunk.start / sample_rate
    chunk_end = chunk.end / sample_rate
    localized = []
    for span in spans:
        start = max(span.start, chunk_start)
        end = min(span.end, chunk_end)
        if end > start:
            localized.append(
                SpanPrompt(span.token, start - chunk_start, end - chunk_start)
            )
    return tuple(localized)


def _run_separation(
    pipeline: SAMAudioPipeline,
    audio: dict[str, Any],
    description: str,
    seed: int,
    inference_steps: int,
    chunk_duration: float,
    chunk_overlap: float,
    *,
    predict_spans: bool = False,
    spans: tuple[SpanPrompt, ...] | None = None,
    visual_prompt: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(pipeline, SAMAudioPipeline):
        raise TypeError("pipeline must come from a SAM-Audio Model Loader node")
    description = description.strip()
    if not description and visual_prompt is None:
        raise ValueError("description cannot be empty for text or span separation")
    sampling_options = ode_options(inference_steps)

    waveforms, target_length = prepare_audio(
        audio, pipeline.processor.audio_sampling_rate
    )
    sample_rate = pipeline.processor.audio_sampling_rate
    batch_size = len(waveforms)
    duration = target_length / sample_rate
    chunks = plan_audio_chunks(
        target_length,
        sample_rate,
        chunk_duration,
        chunk_overlap,
    )

    if spans is not None:
        validate_spans(spans, duration)

    if predict_spans:
        pipeline.ensure_span_predictor(model_management)

    masked_video = None
    if visual_prompt is not None:
        frames, mask = visual_prompt
        masked_video = pipeline.processor.mask_videos([frames], [mask])[0]

    pipeline.load(model_management)
    target_chunks = []
    residual_chunks = []
    progress = comfy_utils.ProgressBar(len(chunks))
    for chunk_index, chunk in enumerate(chunks):
        chunk_waveforms = [
            waveform[..., chunk.start : chunk.end] for waveform in waveforms
        ]

        anchors = None
        if spans is not None:
            localized_spans = _spans_for_chunk(spans, chunk, sample_rate)
            anchors = [list(localized_spans) for _ in range(batch_size)]

        masked_videos = None
        if masked_video is not None:
            chunk_video = slice_visual_frames(masked_video, chunk, target_length)
            masked_videos = [chunk_video for _ in range(batch_size)]

        batch = pipeline.processor(
            audios=chunk_waveforms,
            descriptions=[description for _ in range(batch_size)],
            anchors=anchors,
            masked_videos=masked_videos,
        )
        batch = batch.to(pipeline.device)

        chunk_seed = (seed + chunk_index) % (2**64)
        with (
            seeded_inference(chunk_seed, pipeline.device),
            attention_backend_context(pipeline.attention_backend),
            torch.inference_mode(),
        ):
            result = pipeline.model.separate(
                batch,
                ode_opt=sampling_options,
                reranking_candidates=1,
                predict_spans=predict_spans,
            )

        target = result_to_audio(
            _as_result_list(result.target), sample_rate, chunk.length
        )
        residual = result_to_audio(
            _as_result_list(result.residual), sample_rate, chunk.length
        )
        target_chunks.append((chunk, target["waveform"]))
        residual_chunks.append((chunk, residual["waveform"]))
        progress.update(1)

    target = {
        "waveform": crossfade_audio_chunks(target_chunks, target_length).contiguous(),
        "sample_rate": sample_rate,
    }
    residual = {
        "waveform": crossfade_audio_chunks(
            residual_chunks, target_length
        ).contiguous(),
        "sample_rate": sample_rate,
    }
    return target, residual


class SAMAudioPipelineLoader:
    DESCRIPTION = (
        "Loads an official SAM-Audio checkpoint or a local model from "
        "ComfyUI/models/sam_audio. If Hugging Face denies gated access, the loader "
        "uses a pinned, checksum-verified public mirror of the same checkpoint."
    )

    @classmethod
    def INPUT_TYPES(cls):
        choices = model_choices(folder_paths)
        return {
            "required": {
                "model_name": (
                    choices,
                    {
                        "default": "facebook/sam-audio-large",
                        "tooltip": "Models download into models/sam_audio; valid local model folders appear automatically.",
                    },
                ),
                "attention_backend": (
                    list(ATTENTION_BACKENDS),
                    {
                        "default": "pytorch",
                        "tooltip": "PyTorch SDPA is the accurate default. Comfy Kitchen uses its quantized INT8 attention kernel when supported.",
                    },
                ),
            }
        }

    RETURN_TYPES = (PIPELINE_TYPE,)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load_model"
    CATEGORY = CATEGORY

    def load_model(self, model_name: str, attention_backend: str = "pytorch"):
        return (
            load_pipeline(
                model_name,
                folder_paths,
                model_management,
                model_patcher,
                attention_backend,
            ),
        )


class SAMAudioSpanPrompt:
    DESCRIPTION = (
        "Adds a temporal prompt. Positive spans identify where the target sound is "
        "present; negative spans identify where it is absent. Chain nodes to add spans."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "polarity": (
                    ["positive", "negative"],
                    {"tooltip": "Whether the target is present or absent in this span."},
                ),
                "start_time": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 86400.0, "step": 0.01},
                ),
                "end_time": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 86400.0, "step": 0.01},
                ),
            },
            "optional": {
                "spans": (
                    SPANS_TYPE,
                    {"tooltip": "Optional prior span chain to append to."},
                )
            },
        }

    RETURN_TYPES = (SPANS_TYPE,)
    RETURN_NAMES = ("spans",)
    FUNCTION = "add_span"
    CATEGORY = CATEGORY

    def add_span(
        self,
        polarity: str,
        start_time: float,
        end_time: float,
        spans: tuple[SpanPrompt, ...] | None = None,
    ):
        if end_time <= start_time:
            raise ValueError("end_time must be greater than start_time")
        if polarity not in ("positive", "negative"):
            raise ValueError("polarity must be 'positive' or 'negative'")
        token = "+" if polarity == "positive" else "-"
        current = tuple(spans or ())
        return (current + (SpanPrompt(token, float(start_time), float(end_time)),),)


class SAMAudioTextSeparator:
    DESCRIPTION = (
        "Separates a described sound from audio. Short lowercase noun or verb phrases "
        "such as 'man speaking' or 'dog barking' best match SAM-Audio training."
    )

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "pipeline": (PIPELINE_TYPE,),
            "audio": ("AUDIO",),
            "description": (
                "STRING",
                {
                    "default": "man speaking",
                    "multiline": False,
                    "tooltip": "A concise lowercase description of the sound to isolate.",
                },
            ),
            "predict_spans": (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": "Ask supported models to locate non-ambient sound events before separation. Uses additional memory and time.",
                },
            ),
        }
        required.update(_sampling_inputs())
        required.update(_chunking_inputs())
        return {"required": required}

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("target", "residual")
    FUNCTION = "separate"
    CATEGORY = CATEGORY

    def separate(
        self,
        pipeline: SAMAudioPipeline,
        audio: dict[str, Any],
        description: str,
        predict_spans: bool,
        seed: int,
        inference_steps: int,
        chunk_duration: float = 10.0,
        chunk_overlap: float = 1.0,
    ):
        return _run_separation(
            pipeline,
            audio,
            description,
            seed,
            inference_steps,
            chunk_duration,
            chunk_overlap,
            predict_spans=predict_spans,
        )


class SAMAudioSpanSeparator:
    DESCRIPTION = (
        "Separates a described sound with explicit positive and optional negative "
        "time-span guidance."
    )

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "pipeline": (PIPELINE_TYPE,),
            "audio": ("AUDIO",),
            "description": (
                "STRING",
                {"default": "car honking", "multiline": False},
            ),
            "spans": (SPANS_TYPE,),
        }
        required.update(_sampling_inputs())
        required.update(_chunking_inputs())
        return {"required": required}

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("target", "residual")
    FUNCTION = "separate"
    CATEGORY = CATEGORY

    def separate(
        self,
        pipeline: SAMAudioPipeline,
        audio: dict[str, Any],
        description: str,
        spans: tuple[SpanPrompt, ...],
        seed: int,
        inference_steps: int,
        chunk_duration: float = 10.0,
        chunk_overlap: float = 1.0,
    ):
        return _run_separation(
            pipeline,
            audio,
            description,
            seed,
            inference_steps,
            chunk_duration,
            chunk_overlap,
            spans=tuple(spans),
        )


class SAMAudioVisualSeparator:
    DESCRIPTION = (
        "Separates the sound associated with the white mask region across a sequence "
        "of video frames. A single mask is automatically repeated across all frames."
    )

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "pipeline": (PIPELINE_TYPE,),
            "audio": ("AUDIO",),
            "images": (
                "IMAGE",
                {"tooltip": "Video frames in chronological order."},
            ),
            "mask": (
                "MASK",
                {"tooltip": "White selects the visible object whose sound should be isolated."},
            ),
            "description": (
                "STRING",
                {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional text guidance to combine with the visual prompt.",
                },
            ),
        }
        required.update(_sampling_inputs())
        required.update(_chunking_inputs())
        return {"required": required}

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("target", "residual")
    FUNCTION = "separate"
    CATEGORY = CATEGORY

    def separate(
        self,
        pipeline: SAMAudioPipeline,
        audio: dict[str, Any],
        images: torch.Tensor,
        mask: torch.Tensor,
        description: str,
        seed: int,
        inference_steps: int,
        chunk_duration: float = 10.0,
        chunk_overlap: float = 1.0,
    ):
        visual_prompt = prepare_visual_prompt(images, mask)
        return _run_separation(
            pipeline,
            audio,
            description,
            seed,
            inference_steps,
            chunk_duration,
            chunk_overlap,
            visual_prompt=visual_prompt,
        )


class SAMAudioVideoSeparator:
    DESCRIPTION = (
        "Separates the sound associated with a white mask region from a native "
        "ComfyUI VIDEO value, using the video's embedded audio track."
    )

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "pipeline": (PIPELINE_TYPE,),
            "video": (
                "VIDEO",
                {
                    "tooltip": "A native ComfyUI VIDEO containing frames and an embedded audio track.",
                },
            ),
            "mask": (
                "MASK",
                {
                    "tooltip": "White selects the visible object whose sound should be isolated. Use one mask for all frames or one per frame.",
                },
            ),
            "description": (
                "STRING",
                {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional text guidance to combine with the visual prompt.",
                },
            ),
        }
        required.update(_sampling_inputs())
        required.update(_chunking_inputs())
        return {"required": required}

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("target", "residual")
    FUNCTION = "separate"
    CATEGORY = CATEGORY

    def separate(
        self,
        pipeline: SAMAudioPipeline,
        video: Any,
        mask: torch.Tensor,
        description: str,
        seed: int,
        inference_steps: int,
        chunk_duration: float = 10.0,
        chunk_overlap: float = 1.0,
    ):
        get_components = getattr(video, "get_components", None)
        if not callable(get_components):
            raise TypeError("video must be a native ComfyUI VIDEO value")

        components = get_components()
        images = getattr(components, "images", None)
        audio = getattr(components, "audio", None)
        if not isinstance(images, torch.Tensor) or images.shape[0] < 1:
            raise ValueError("VIDEO input does not contain any decoded frames")
        if audio is None:
            raise ValueError(
                "VIDEO input has no audio track; attach audio with ComfyUI's "
                "Create Video node or use SAM-Audio Visual Separate with separate "
                "IMAGE and AUDIO inputs"
            )

        visual_prompt = prepare_visual_prompt(images, mask)
        return _run_separation(
            pipeline,
            audio,
            description,
            seed,
            inference_steps,
            chunk_duration,
            chunk_overlap,
            visual_prompt=visual_prompt,
        )


NODE_CLASS_MAPPINGS = {
    "SAMAudioPipelineLoader": SAMAudioPipelineLoader,
    "SAMAudioSpanPrompt": SAMAudioSpanPrompt,
    "SAMAudioTextSeparator": SAMAudioTextSeparator,
    "SAMAudioSpanSeparator": SAMAudioSpanSeparator,
    "SAMAudioVisualSeparator": SAMAudioVisualSeparator,
    "SAMAudioVideoSeparator": SAMAudioVideoSeparator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAMAudioPipelineLoader": "SAM-Audio Model Loader",
    "SAMAudioSpanPrompt": "SAM-Audio Span Prompt",
    "SAMAudioTextSeparator": "SAM-Audio Text Separate",
    "SAMAudioSpanSeparator": "SAM-Audio Span Separate",
    "SAMAudioVisualSeparator": "SAM-Audio Visual Separate",
    "SAMAudioVideoSeparator": "SAM-Audio Video Separate",
}
