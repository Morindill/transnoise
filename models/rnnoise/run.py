"""一次性工具（在 models/rnnoise 的 .venv 里运行）：用 RNNoise 增强一个目录里的所有 wav。
用法：models/rnnoise/.venv/bin/python models/rnnoise/run.py --in 带噪目录 --out 输出目录
每个文件用新的 RNNoise 实例（状态清零）。结束时打印一行耗时（仅供参考，不进评测）。"""

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
import soundfile as sf  # noqa: E402
from pyrnnoise import RNNoise  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--in", dest="src", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()

files = sorted(Path(a.src).glob("*.wav"))
Path(a.out).mkdir(parents=True, exist_ok=True)
total = audio = 0.0
for f in files:
    d = RNNoise(sample_rate=48000)
    t = time.perf_counter()
    for _ in d.denoise_wav(str(f), str(Path(a.out) / f.name)):
        pass
    total += time.perf_counter() - t
    audio += sf.info(f).duration
print(f"RNNoise: {len(files)} 个文件，音频 {audio:.0f}s，处理 {total:.1f}s，RTF≈{total / audio:.3f}（CPU，不含启动，仅供参考）")
