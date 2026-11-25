"""One-off tool (run inside models/mossformer2's .venv, with this directory as cwd: the weight path checkpoints/ is relative to cwd):
enhance every wav in a directory with MossFormer2_SE_48K.
Usage: cd models/mossformer2 && .venv/bin/python run.py --in <noisy dir> --out <output dir> (use absolute paths)
The first file is run once as a warm-up (not timed). Prints one timing line at the end (for reference only, not part of the evaluation)."""

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
cv(input_path=str(files[0]), online_write=False)  # warm-up
total = audio = 0.0
for f in files:
    t = time.perf_counter()
    y = np.asarray(cv(input_path=str(f), online_write=False))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total += time.perf_counter() - t
    sf.write(Path(a.out) / f.name, np.clip(y.reshape(-1), -1, 1), 48000, subtype="PCM_16")
    audio += sf.info(f).duration
print(f"MossFormer2_SE_48K: {len(files)} files, audio {audio:.0f}s, processed in {total:.1f}s, RTF≈{total / audio:.3f}"
      f" ({'GPU' if torch.cuda.is_available() else 'CPU'}, excluding loading and warm-up; for reference only)")
