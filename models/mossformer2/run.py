"""一次性工具（在 models/mossformer2 的 .venv 里运行，必须以本目录为 cwd：权重路径 checkpoints/ 相对 cwd）：
用 MossFormer2_SE_48K 增强一个目录里的所有 wav。
用法：cd models/mossformer2 && .venv/bin/python run.py --in 带噪目录 --out 输出目录（用绝对路径）
第一个文件先预热一次（不计时）。结束时打印一行耗时（仅供参考，不进评测）。"""

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from clearvoice import ClearVoice

p = argparse.ArgumentParser()
p.add_argument("--in", dest="src", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()

files = sorted(Path(a.src).glob("*.wav"))
Path(a.out).mkdir(parents=True, exist_ok=True)
cv = ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])
cv(input_path=str(files[0]), online_write=False)  # 预热
total = audio = 0.0
for f in files:
    t = time.perf_counter()
    y = np.asarray(cv(input_path=str(f), online_write=False))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total += time.perf_counter() - t
    sf.write(Path(a.out) / f.name, np.clip(y.reshape(-1), -1, 1), 48000, subtype="PCM_16")
    audio += sf.info(f).duration
print(f"MossFormer2_SE_48K: {len(files)} 个文件，音频 {audio:.0f}s，处理 {total:.1f}s，RTF≈{total / audio:.3f}"
      f"（{'GPU' if torch.cuda.is_available() else 'CPU'}，不含加载和预热，仅供参考）")
