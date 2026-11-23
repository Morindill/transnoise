"""Run a command inside the train environment (train/pyproject.toml: Keras 3 + TensorFlow, CUDA 12), replacing the current process.

The pip-installed CUDA 12 libraries (nvidia/*/lib) are not on the system library search path; TF would silently fall back
to CPU if it cannot find them, so they are added to LD_LIBRARY_PATH before launch. The variable has to be set before the
process starts (changing os.environ from Python is too late), hence the exec restart.
"""

import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

from .paths import ROOT

WSL_LIBS = "/usr/lib/wsl/lib"  # WSL GPU driver libraries


def exec_in_train_env(cmd: Sequence[str]) -> None:
    uv = ["uv", "run", "--project", str(ROOT / "train"), "--no-sync"]
    site = subprocess.run([*uv, "python", "-c", "import site; print(site.getsitepackages()[0])"],
                          check=True, capture_output=True, text=True).stdout.strip()
    libs = [str(p) for p in sorted(Path(site).glob("nvidia/*/lib"))]
    old = os.environ.get("LD_LIBRARY_PATH")
    os.environ["LD_LIBRARY_PATH"] = ":".join([*libs, WSL_LIBS, *([old] if old else [])])
    os.execvp("uv", [*uv, *cmd])
