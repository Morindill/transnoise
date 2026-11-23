"""在 train 环境（train/pyproject.toml：Keras 3 + TensorFlow，CUDA 12）里运行命令，替换当前进程。

pip 装的 CUDA 12 库（nvidia/*/lib）不在系统库搜索路径里，TF 找不到就会悄悄退回 CPU，所以启动前把它们加进 LD_LIBRARY_PATH
（它必须在进程启动前设置，在 Python 里改 os.environ 来不及，因此用 exec 重启）。
"""

import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

from .paths import ROOT

WSL_LIBS = "/usr/lib/wsl/lib"  # WSL 的 GPU 驱动库


def exec_in_train_env(cmd: Sequence[str]) -> None:
    uv = ["uv", "run", "--project", str(ROOT / "train"), "--no-sync"]
    site = subprocess.run([*uv, "python", "-c", "import site; print(site.getsitepackages()[0])"],
                          check=True, capture_output=True, text=True).stdout.strip()
    libs = [str(p) for p in sorted(Path(site).glob("nvidia/*/lib"))]
    old = os.environ.get("LD_LIBRARY_PATH")
    os.environ["LD_LIBRARY_PATH"] = ":".join([*libs, WSL_LIBS, *([old] if old else [])])
    os.execvp("uv", [*uv, *cmd])
