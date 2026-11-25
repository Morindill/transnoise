"""One-off tool: enhance every wav in a directory with the precompiled DeepFilterNet3 binary (standard library only).
Usage: python3 models/dfn3/run.py --in <noisy dir> --out <output dir>
No -D (it drops the beginning to compensate for the delay, which shortens the output) and no --pf; the delay is measured and aligned uniformly by the evaluation."""

import argparse
import subprocess
import time
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHUNK = 200  # maximum number of files per command line

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
print(f"DeepFilterNet3: {len(files)} files, audio {audio:.0f}s, took {el:.1f}s, RTF≈{el / audio:.3f} (includes process start-up and I/O; for reference only)")
