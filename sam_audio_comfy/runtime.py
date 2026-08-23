from __future__ import annotations

import contextlib
import hashlib
import importlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, NamedTuple

import torch

from .audiotools_compat import audiotools_import_compatibility
from .xformers_compat import xformers_import_compatibility

LOGGER = logging.getLogger(__name__)

OFFICIAL_MODELS = (
    "facebook/sam-audio-small",
    "facebook/sam-audio-base",
    "facebook/sam-audio-large",
    "facebook/sam-audio-small-tv",
    "facebook/sam-audio-base-tv",
    "facebook/sam-audio-large-tv",
)

MODEL_FOLDER = "sam_audio"
MODEL_FILES = ("config.json", "checkpoint.pt")
MIRROR_MARKER = ".verified-mirror-checkpoint"


@dataclass(frozen=True)
class MirrorSpec:
    repo_id: str
    revision: str
    checkpoint_size: int
    checkpoint_sha256: str


# Each mirror revision is immutable, includes Meta's SAM License, and was checked
# against the corresponding facebook checkpoint.pt LFS SHA-256 on 2026-08-23.
MIRROR_MODELS = {
    "facebook/sam-audio-small": MirrorSpec(
        "mrfakename/sam-audio-small",
        "682824171fd5cfec47b88687f583a11050c1bf1d",
        5_100_547_943,
        "8c44fda9821fd9f2ec8977304e3c0f55290d9eacb6bbf25b4b8fb1f69c2a8c06",
    ),
    "facebook/sam-audio-base": MirrorSpec(
        "mrfakename/sam-audio-base",
        "351f6666d46357edea2149a338a680fcd0d7afdd",
        7_725_405_659,
        "b5f3e29ea7a9e80e90a00da495a8aafe890571f371c4bfb88c052c65a5636839",
    ),
    "facebook/sam-audio-large": MirrorSpec(
        "mrfakename/sam-audio-large",
        "497809f191dd4f673a2c93557176387cf2a06cf2",
        14_861_356_211,
        "ca55418b1d23e8c8a4dcc55f259d9801c8f79da0131a66e525d862c1289e3c4f",
    ),
    "facebook/sam-audio-small-tv": MirrorSpec(
        "mrfakename/sam-audio-small-tv",
        "2fee00b2836ede05fce503726ac9f908b1fc4aa9",
        5_100_547_943,
        "1a9693b235efc3176986664dac349eb64022c7468cf8d4195954490a2457e6a9",
    ),
    "facebook/sam-audio-base-tv": MirrorSpec(
        "mrfakename/sam-audio-base-tv",
        "ed6a64b2d3c6151276d81783becb8881d75efe4f",
        7_725_405_659,
        "569e3fecdefe267047b02d117acb5a67c03d2f73c4943d3d8bd43df9e9b9148d",
    ),
    "facebook/sam-audio-large-tv": MirrorSpec(
        "mrfakename/sam-audio-large-tv",
        "30eea7c915f43349f3b3f5c7e33e8658a01d7253",
        14_861_356_211,
        "90e047269238c498c5abe0da6e6ba40859d152111d4a09f582a209027a33b72f",
    ),
}


class SpanPrompt(NamedTuple):
    token: Literal["+", "-"]
    start: float
    end: float


@dataclass
class SAMAudioPipeline:
    model: torch.nn.Module
    processor: Any
    patcher: Any
    device: torch.device
    source: str

    def get_models(self) -> list[Any]:
        """Let ComfyUI track this model for the lifetime of the prompt."""
        return [self.patcher]

    def load(self, model_management: Any) -> None:
        model_management.load_models_gpu([self.patcher], force_full_load=True)
        self.model.eval()


def register_model_folder(folder_paths: Any) -> Path:
    model_dir = Path(folder_paths.models_dir) / MODEL_FOLDER
    try:
        folder_paths.add_model_folder_path(
            MODEL_FOLDER, str(model_dir), is_default=True
        )
    except TypeError:
        # ComfyUI versions before is_default support still accept the two-argument form.
        folder_paths.add_model_folder_path(MODEL_FOLDER, str(model_dir))
    return model_dir


def _is_model_directory(path: Path) -> bool:
    return path.is_dir() and all((path / name).is_file() for name in MODEL_FILES)


def discover_local_models(folder_paths: Any) -> dict[str, Path]:
    """Return stable UI labels for valid model directories in every configured root."""
    discovered: list[tuple[str, Path]] = []
    official_directory_names = {name.rsplit("/", 1)[-1] for name in OFFICIAL_MODELS}
    for raw_root in folder_paths.get_folder_paths(MODEL_FOLDER):
        root = Path(raw_root)
        if not root.is_dir():
            continue
        for current, directories, files in os.walk(root):
            current_path = Path(current)
            if all(name in files for name in MODEL_FILES):
                if current_path.parent == root and current_path.name in official_directory_names:
                    directories[:] = []
                    continue
                relative = current_path.relative_to(root).as_posix()
                label = relative if relative != "." else current_path.name
                discovered.append((f"local/{label}", current_path))
                directories[:] = []
                continue
            directories[:] = [
                name
                for name in directories
                if not name.startswith(".") and name != "__pycache__"
            ]

    counts: dict[str, int] = {}
    result: dict[str, Path] = {}
    for label, path in sorted(discovered, key=lambda item: (item[0].lower(), str(item[1]))):
        counts[label] = counts.get(label, 0) + 1
        unique_label = label if counts[label] == 1 else f"{label} [{counts[label]}]"
        result[unique_label] = path
    return result


def model_choices(folder_paths: Any) -> list[str]:
    return [*OFFICIAL_MODELS, *discover_local_models(folder_paths).keys()]


def _is_gate_error(error: Exception) -> bool:
    """Recognize Hugging Face gated/authorization failures across hub versions."""
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if type(current).__name__ == "GatedRepoError":
            return True
        response = getattr(current, "response", None)
        if getattr(response, "status_code", None) in (401, 403):
            return True
        current = current.__cause__ or current.__context__
    return False


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_mirror_checkpoint(destination: Path, spec: MirrorSpec) -> None:
    checkpoint = destination / "checkpoint.pt"
    marker = destination / MIRROR_MARKER
    expected_marker = f"{spec.checkpoint_size} {spec.checkpoint_sha256}\n"

    if (
        marker.is_file()
        and marker.read_text(encoding="utf-8") == expected_marker
        and checkpoint.stat().st_size == spec.checkpoint_size
    ):
        return

    actual_size = checkpoint.stat().st_size
    if actual_size != spec.checkpoint_size:
        raise RuntimeError(
            f"Mirror checkpoint size mismatch for {spec.repo_id}: expected "
            f"{spec.checkpoint_size} bytes, received {actual_size}"
        )

    actual_hash = _checkpoint_sha256(checkpoint)
    if actual_hash != spec.checkpoint_sha256:
        raise RuntimeError(
            f"Mirror checkpoint checksum mismatch for {spec.repo_id}; refusing to load it"
        )
    marker.write_text(expected_marker, encoding="utf-8")


def _download_mirror_model(
    huggingface_hub: Any,
    destination: Path,
    spec: MirrorSpec,
) -> Path:
    marker = destination / MIRROR_MARKER
    marker.write_text(f"pending {spec.checkpoint_sha256}\n", encoding="utf-8")
    snapshot = huggingface_hub.snapshot_download(
        repo_id=spec.repo_id,
        revision=spec.revision,
        local_dir=str(destination),
        allow_patterns=[*MODEL_FILES, "LICENSE"],
    )
    path = Path(snapshot)
    if not _is_model_directory(path) or not (path / "LICENSE").is_file():
        raise RuntimeError(
            f"The mirror snapshot for {spec.repo_id} is incomplete; expected "
            "config.json, checkpoint.pt, and LICENSE"
        )
    _verify_mirror_checkpoint(path, spec)
    return path


def _download_official_model(repo_id: str, model_root: Path) -> Path:
    destination = model_root / repo_id.rsplit("/", 1)[-1]
    if _is_model_directory(destination):
        marker = destination / MIRROR_MARKER
        if marker.exists() and (destination / "LICENSE").is_file():
            _verify_mirror_checkpoint(destination, MIRROR_MODELS[repo_id])
            return destination
        if not marker.exists():
            return destination

    destination.mkdir(parents=True, exist_ok=True)
    try:
        huggingface_hub = importlib.import_module("huggingface_hub")
    except Exception as error:
        raise RuntimeError(
            "huggingface-hub is required to download SAM-Audio checkpoints; "
            "run this node's install.py and restart ComfyUI"
        ) from error

    try:
        snapshot = huggingface_hub.snapshot_download(
            repo_id=repo_id,
            local_dir=str(destination),
            allow_patterns=list(MODEL_FILES),
        )
    except Exception as error:
        if _is_gate_error(error):
            spec = MIRROR_MODELS[repo_id]
            LOGGER.warning(
                "Access to %s was gated; downloading checksum-verified mirror %s",
                repo_id,
                spec.repo_id,
            )
            try:
                return _download_mirror_model(huggingface_hub, destination, spec)
            except Exception as mirror_error:
                raise RuntimeError(
                    f"Access to {repo_id} was denied and its verified public mirror "
                    f"{spec.repo_id} could not be downloaded"
                ) from mirror_error
        raise RuntimeError(
            f"Could not download {repo_id}. Accept the model license at "
            f"https://huggingface.co/{repo_id}, run `hf auth login` in ComfyUI's "
            "Python environment, and retry."
        ) from error

    path = Path(snapshot)
    if not _is_model_directory(path):
        raise RuntimeError(
            f"The downloaded snapshot for {repo_id} is incomplete; expected "
            f"{', '.join(MODEL_FILES)} in {path}"
        )
    (path / MIRROR_MARKER).unlink(missing_ok=True)
    return path


def resolve_model(model_name: str, folder_paths: Any) -> Path:
    local_models = discover_local_models(folder_paths)
    if model_name in local_models:
        return local_models[model_name]
    if model_name not in OFFICIAL_MODELS:
        raise ValueError(
            f"Unknown SAM-Audio model {model_name!r}. Refresh the browser after adding local models."
        )
    model_root = register_model_folder(folder_paths)
    return _download_official_model(model_name, model_root)


def import_sam_audio() -> tuple[Any, Any]:
    try:
        LOGGER.info(
            "Loading SAM-Audio through isolated inference-only compatibility layers"
        )
        with xformers_import_compatibility(), audiotools_import_compatibility():
            module = importlib.import_module("sam_audio")
        return module.SAMAudio, module.SAMAudioProcessor
    except Exception as error:
        raise RuntimeError(
            "SAM-Audio could not be imported. Install this node's requirements with "
            "ComfyUI's Python interpreter, then restart ComfyUI. This node does not "
            "require or install xFormers."
        ) from error


def load_pipeline(
    model_name: str,
    folder_paths: Any,
    model_management: Any,
    model_patcher: Any,
) -> SAMAudioPipeline:
    model_path = resolve_model(model_name, folder_paths)
    SAMAudio, SAMAudioProcessor = import_sam_audio()

    LOGGER.info("Loading SAM-Audio model from %s", model_path)
    try:
        model = SAMAudio.from_pretrained(
            str(model_path),
            map_location="cpu",
            strict=False,
            text_ranker=None,
            visual_ranker=None,
        )
        processor = SAMAudioProcessor.from_pretrained(str(model_path))
    except Exception as error:
        raise RuntimeError(
            f"Failed to load SAM-Audio from {model_path}. Upstream may also need its "
            "T5, Perception Encoder, and ranker assets available through Hugging Face."
        ) from error

    model = model.eval()
    # BaseModel exposes device() as a method, while ComfyUI's ModelPatcher tracks
    # residency through a mutable model.device attribute. Upstream does not call
    # that helper internally, so adopting ComfyUI's convention is safe here.
    model.device = torch.device("cpu")
    load_device = model_management.get_torch_device()
    offload_device = model_management.unet_offload_device()
    patcher = model_patcher.ModelPatcher(
        model, load_device=load_device, offload_device=offload_device
    )
    return SAMAudioPipeline(
        model=model,
        processor=processor,
        patcher=patcher,
        device=load_device,
        source=model_name,
    )


def ode_options(inference_steps: int) -> dict[str, Any]:
    if inference_steps < 2 or inference_steps % 2:
        raise ValueError("inference_steps must be an even integer of at least 2")
    return {"method": "midpoint", "options": {"step_size": 2.0 / inference_steps}}


@contextlib.contextmanager
def seeded_inference(seed: int, device: torch.device) -> Iterator[None]:
    cuda_devices: list[int] = []
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield


def validate_spans(spans: tuple[SpanPrompt, ...], duration: float) -> None:
    if not spans:
        raise ValueError("At least one span prompt is required")
    if not any(span.token == "+" for span in spans):
        raise ValueError("At least one positive span is required")
    for span in spans:
        if span.start < 0 or span.end <= span.start:
            raise ValueError(
                f"Invalid span {span.start:g}-{span.end:g}: end must be after start"
            )
        if span.end > duration + 1e-6:
            raise ValueError(
                f"Span ends at {span.end:g}s but the input audio is only {duration:.3f}s"
            )
