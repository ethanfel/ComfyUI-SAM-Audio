"""Install the tested SAM-Audio inference stack without forcing xFormers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# These are installed without dependency resolution. requirements.txt contains the
# inference dependencies; this avoids perception_models' large training dependency
# list, including a compiled xFormers wheel that can replace ComfyUI's torch stack.
UPSTREAM_PACKAGES = (
    "dacvae @ git+https://github.com/facebookresearch/dacvae.git@414c20785fc3a28373073ea8ef7a1316eeeaca6e",
    "perception-models @ git+https://github.com/facebookresearch/perception_models.git@e72b6810b1133e1c879f2cc965d276eb73803f1f",
    "sam-audio @ git+https://github.com/facebookresearch/sam-audio.git@bb4c6999d2677c7402360e426afc01ddfad6dce0",
)


def run_pip(*arguments: str) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *arguments]
    )


def main() -> None:
    run_pip("-r", str(ROOT / "requirements.txt"))
    for package in UPSTREAM_PACKAGES:
        run_pip("--no-deps", package)


if __name__ == "__main__":
    main()
