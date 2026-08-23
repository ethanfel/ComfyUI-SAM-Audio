import pytest
import torch

from sam_audio_comfy.audio import (
    AudioChunk,
    crossfade_audio_chunks,
    plan_audio_chunks,
    prepare_audio,
    prepare_visual_prompt,
    result_to_audio,
    slice_visual_frames,
)


def test_chunk_plan_covers_audio_with_requested_overlap():
    assert plan_audio_chunks(21, 1, 10.0, 2.0) == [
        AudioChunk(0, 10),
        AudioChunk(8, 18),
        AudioChunk(16, 21),
    ]
    assert plan_audio_chunks(21, 1, 0.0, 2.0) == [AudioChunk(0, 21)]
    assert plan_audio_chunks(5, 1, 10.0, 2.0) == [AudioChunk(0, 5)]

    with pytest.raises(ValueError, match="shorter than chunk_duration"):
        plan_audio_chunks(21, 1, 10.0, 10.0)


def test_crossfade_audio_chunks_normalizes_overlap():
    first = torch.ones(1, 1, 6)
    second = torch.full((1, 1, 6), 3.0)
    output = crossfade_audio_chunks(
        [(AudioChunk(0, 6), first), (AudioChunk(4, 10), second)], 10
    )

    assert torch.equal(output[..., :4], torch.ones(1, 1, 4))
    assert torch.allclose(output[0, 0, 4:6], torch.tensor([5 / 3, 7 / 3]))
    assert torch.equal(output[..., 6:], torch.full((1, 1, 4), 3.0))


def test_visual_frames_are_sliced_to_chunk_time_range():
    frames = torch.arange(10).view(10, 1)

    assert torch.equal(
        slice_visual_frames(frames, AudioChunk(2, 6), 10), frames[2:6]
    )
    assert torch.equal(
        slice_visual_frames(frames, AudioChunk(0, 10), 10), frames
    )


def test_prepare_audio_preserves_batch_and_resamples():
    waveform = torch.stack(
        [torch.ones(2, 8), torch.full((2, 8), 0.5)], dim=0
    )
    rows, length = prepare_audio(
        {"waveform": waveform, "sample_rate": 8}, target_sample_rate=4
    )

    assert len(rows) == 2
    assert rows[0].shape == (2, 4)
    assert rows[1].shape == (2, 4)
    assert length == 4


def test_result_to_audio_crops_and_pads_mono_batch():
    audio = result_to_audio(
        [torch.arange(8), torch.arange(3)], sample_rate=48_000, target_length=5
    )

    assert audio["sample_rate"] == 48_000
    assert audio["waveform"].shape == (2, 1, 5)
    assert torch.equal(audio["waveform"][0, 0], torch.arange(5).float())
    assert torch.equal(
        audio["waveform"][1, 0], torch.tensor([0.0, 1.0, 2.0, 0.0, 0.0])
    )


def test_prepare_visual_prompt_converts_layout_scale_and_repeats_mask():
    images = torch.zeros(3, 4, 6, 3)
    images[..., 0] = 1.0
    mask = torch.tensor([[0.0, 1.0], [0.0, 0.0]])

    frames, masks = prepare_visual_prompt(images, mask)

    assert frames.shape == (3, 3, 4, 6)
    assert frames.dtype == torch.uint8
    assert frames[:, 0].min().item() == 255
    assert frames[:, 1:].max().item() == 0
    assert masks.shape == (3, 1, 4, 6)
    assert masks.dtype == torch.bool
    assert torch.equal(masks[0], masks[1])


def test_prepare_visual_prompt_rejects_mismatched_frame_counts():
    images = torch.zeros(3, 4, 6, 3)
    mask = torch.zeros(2, 4, 6)

    try:
        prepare_visual_prompt(images, mask)
    except ValueError as error:
        assert "2 frames" in str(error)
    else:
        raise AssertionError("expected a frame-count validation error")
