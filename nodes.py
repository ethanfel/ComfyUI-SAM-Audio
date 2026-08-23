from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import comfy.model_management as model_management
import comfy.model_patcher as model_patcher
import folder_paths
import torch

from .sam_audio_comfy.audio import (
    prepare_audio,
    prepare_visual_prompt,
    result_to_audio,
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


def _as_result_list(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value] if value.ndim == 1 else list(value)
    if isinstance(value, Sequence):
        return list(value)
    raise RuntimeError(f"SAM-Audio returned an unexpected result type: {type(value).__name__}")


def _run_separation(
    pipeline: SAMAudioPipeline,
    audio: dict[str, Any],
    description: str,
    seed: int,
    inference_steps: int,
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
    batch_size = len(waveforms)
    duration = target_length / pipeline.processor.audio_sampling_rate

    anchors = None
    if spans is not None:
        validate_spans(spans, duration)
        anchors = [list(spans) for _ in range(batch_size)]

    masked_videos = None
    if visual_prompt is not None:
        frames, mask = visual_prompt
        masked = pipeline.processor.mask_videos([frames], [mask])[0]
        masked_videos = [masked for _ in range(batch_size)]

    batch = pipeline.processor(
        audios=waveforms,
        descriptions=[description for _ in range(batch_size)],
        anchors=anchors,
        masked_videos=masked_videos,
    )

    pipeline.load(model_management)
    batch = batch.to(pipeline.device)

    with seeded_inference(seed, pipeline.device), torch.inference_mode():
        result = pipeline.model.separate(
            batch,
            ode_opt=sampling_options,
            reranking_candidates=1,
            predict_spans=predict_spans,
        )

    target = result_to_audio(
        _as_result_list(result.target),
        pipeline.processor.audio_sampling_rate,
        target_length,
    )
    residual = result_to_audio(
        _as_result_list(result.residual),
        pipeline.processor.audio_sampling_rate,
        target_length,
    )
    return target, residual


class SAMAudioPipelineLoader:
    DESCRIPTION = (
        "Loads an official gated SAM-Audio checkpoint or a local model from "
        "ComfyUI/models/sam_audio. The first official-model load downloads config.json "
        "and checkpoint.pt after Hugging Face access has been granted."
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
                        "tooltip": "Official models download into models/sam_audio; valid local model folders appear automatically.",
                    },
                )
            }
        }

    RETURN_TYPES = (PIPELINE_TYPE,)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load_model"
    CATEGORY = CATEGORY

    def load_model(self, model_name: str):
        return (
            load_pipeline(
                model_name,
                folder_paths,
                model_management,
                model_patcher,
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
    ):
        return _run_separation(
            pipeline,
            audio,
            description,
            seed,
            inference_steps,
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
    ):
        return _run_separation(
            pipeline,
            audio,
            description,
            seed,
            inference_steps,
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
    ):
        visual_prompt = prepare_visual_prompt(images, mask)
        return _run_separation(
            pipeline,
            audio,
            description,
            seed,
            inference_steps,
            visual_prompt=visual_prompt,
        )


NODE_CLASS_MAPPINGS = {
    "SAMAudioPipelineLoader": SAMAudioPipelineLoader,
    "SAMAudioSpanPrompt": SAMAudioSpanPrompt,
    "SAMAudioTextSeparator": SAMAudioTextSeparator,
    "SAMAudioSpanSeparator": SAMAudioSpanSeparator,
    "SAMAudioVisualSeparator": SAMAudioVisualSeparator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAMAudioPipelineLoader": "SAM-Audio Model Loader",
    "SAMAudioSpanPrompt": "SAM-Audio Span Prompt",
    "SAMAudioTextSeparator": "SAM-Audio Text Separate",
    "SAMAudioSpanSeparator": "SAM-Audio Span Separate",
    "SAMAudioVisualSeparator": "SAM-Audio Visual Separate",
}
