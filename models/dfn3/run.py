"""一次性工具：用 DeepFilterNet3 预编译二进制增强一个目录里的所有 wav（只用标准库）。
用法：python3 models/dfn3/run.py --in 带噪目录 --out 输出目录
不加 -D（它会丢掉开头来补偿延迟，输出变短）、不加 --pf；延迟由评测统一测量并对齐。"""

import argparse
import subprocess
import time
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHUNK = 200  # 每次命令行最多带多少个文件

p = argparse.ArgumentParser()
p.add_argument("--in", dest="src", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()

files = sorted(Path(a.src).glob("*.wav"))
Path(a.out).mkdir(parents=True, exist_ok=True)
cmd = [str(HERE / "bin" / "deep-filter"), "-m", str(HERE / "DeepFilterNet3_onnx.tar.gz"), "-o", a.out]
t = time.perf_counter()
for i in range(0, len(files), CHUNK):
    subprocess.run(cmd + [str(f) for f in files[i: i + CHUNK]], check=True, capture_output=True)
el = time.perf_counter() - t
audio = sum((lambda w: w.getnframes() / w.getframerate())(wave.open(str(f))) for f in files)
print(f"DeepFilterNet3: {len(files)} 个文件，音频 {audio:.0f}s，耗时 {el:.1f}s，RTF≈{el / audio:.3f}（含进程启动与读写，仅供参考）")
