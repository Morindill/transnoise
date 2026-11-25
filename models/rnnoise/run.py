"""One-off tool (run inside models/rnnoise's .venv): enhance every wav in a directory with RNNoise.
Usage: models/rnnoise/.venv/bin/python models/rnnoise/run.py --in <noisy dir> --out <output dir>
Each file gets a fresh RNNoise instance (state reset). Prints one timing line at the end (for reference only, not part of the evaluation)."""

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
print(f"RNNoise: {len(files)} files, audio {audio:.0f}s, processed in {total:.1f}s, RTF≈{total / audio:.3f} (CPU, excluding start-up; for reference only)")
