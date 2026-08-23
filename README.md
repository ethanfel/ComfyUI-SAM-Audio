# ComfyUI SAM-Audio

ComfyUI nodes for [Meta SAM-Audio](https://github.com/facebookresearch/sam-audio), which separates a described sound from an audio mixture.

Supports text prompts, positive and negative time spans, and visual masks. Each separator outputs the isolated target and the remaining audio.

## Requirements

- Python 3.11 or newer
- A CUDA GPU is strongly recommended
- Access to a gated SAM-Audio checkpoint on Hugging Face

## Installation

Clone the repository into `ComfyUI/custom_nodes`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ethanfel/ComfyUI-SAM-Audio.git
```

Run the installer with the same Python environment used by ComfyUI:

```bash
python ComfyUI-SAM-Audio/install.py
```

Then restart ComfyUI. If installed through ComfyUI Manager, the installer should run automatically.

## Model access

Accept the license for the checkpoint you want to use, such as [facebook/sam-audio-large](https://huggingface.co/facebook/sam-audio-large), then authenticate in the ComfyUI environment:

```bash
hf auth login
```

Official checkpoints download on first use to `ComfyUI/models/sam_audio/`.

## Nodes

- **SAM-Audio Model Loader** — loads an official or local checkpoint
- **SAM-Audio Text Separate** — separates audio using a text description
- **SAM-Audio Span Prompt** — creates positive or negative time spans
- **SAM-Audio Span Separate** — separates audio using text and time spans
- **SAM-Audio Visual Separate** — separates audio using image frames and a mask

All nodes are in `audio/SAM-Audio`.

## Basic use

1. Load a model with **SAM-Audio Model Loader**.
2. Load audio with ComfyUI's audio loader.
3. Connect both to a separator and enter a short prompt such as `man speaking` or `dog barking`.
4. Preview or save the `target` and `residual` outputs.

Example workflows are included in the [`examples`](examples/) directory.

## Notes

- Output is mono at 48 kHz.
- White pixels in a visual mask identify the target object.
- Local model folders must contain `config.json` and `checkpoint.pt`.

## License

This ComfyUI integration is released under the MIT License. SAM-Audio and its checkpoints use Meta's separate [SAM License](https://github.com/facebookresearch/sam-audio/blob/main/LICENSE).
