import torch

from sam_audio_comfy.audio import prepare_audio, prepare_visual_prompt, result_to_audio


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
