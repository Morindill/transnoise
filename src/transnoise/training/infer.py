"""Inference tool: enhance every wav in a directory with a checkpoint; the output can be scored by the evaluation directly.
  uv run transnoise infer --ckpt data/checkpoints/baseline --in <noisy dir> --out <output dir>

--ckpt can be a run directory (uses latest), <run directory>/best, or a specific step_XXXXXXXX. EMA weights by default; --raw uses the raw weights.
The model architecture comes from the config_model.toml snapshot in the run directory (saved at training time).
Whole-clip offline processing: the output is time-aligned with the input (no streaming delay); input must be 48 kHz mono wav. Output is 48 kHz mono PCM16 with the same file names.
Then score it: uv run transnoise eval score noise <output dir> --name <name>
"""

import argparse
import time
from pathlib import Path

from ..paths import SR


def resolve(path: Path) -> Path:
    from . import checkpoint

    if (path / "meta.json").exists():
        return path
    found = checkpoint.latest(path)
    if found is None:
        raise SystemExit(f"{path} is not a checkpoint directory and has no latest")
    return found


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="run directory / best / step_XXXXXXXX")
    p.add_argument("--in", dest="src", required=True, help="directory of noisy wavs")
    p.add_argument("--out", required=True, help="output directory (when it is called wav, the evaluation result name defaults to its parent's name)")
    p.add_argument("--raw", action="store_true", help="use the raw weights instead of the EMA weights")
    p.add_argument("--limit", type=int, help="only process the first N files (for quick checks)")
    a = p.parse_args()

    import numpy as np
    import soundfile as sf

    from ..model.config import ModelConfig
    from ..model.denoiser import Denoiser
    from . import checkpoint
    from .trainer import setup_gpu

    setup_gpu()
    ckpt = resolve(Path(a.ckpt))
    model = Denoiser(ModelConfig.from_checkpoint(ckpt))
    model(np.zeros((1, SR), "float32"))  # build the weights
    meta = checkpoint.load_weights(ckpt, model, ema=not a.raw)
    print(f"checkpoint {ckpt} (step {meta['step']}, {'raw' if a.raw else 'EMA'} weights)")

    files = sorted(Path(a.src).glob("*.wav"))[: a.limit]
    if not files:
        raise SystemExit(f"no wav files in {a.src}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    audio = total = 0.0
    for f in files:
        x, sr = sf.read(f, dtype="float32")
        if sr != SR or x.ndim != 1:
            raise SystemExit(f"{f}: needs 48 kHz mono, got sr={sr} shape={x.shape}")
        t = time.perf_counter()
        y = np.asarray(model(x[None]))[0]
        total += time.perf_counter() - t
        audio += len(x) / sr
        sf.write(out / f.name, np.clip(y, -1.0, 1.0), SR, subtype="PCM_16")
    print(f"{len(files)} files → {out}; audio {audio:.0f}s, processed in {total:.1f}s (RTF≈{total / audio:.3f}, offline whole-clip, including first compile)")


if __name__ == "__main__":
    main()
