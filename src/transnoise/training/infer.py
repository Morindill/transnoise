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


def resolve(path: Path) -> Path:
    from . import checkpoint

    if (path / "meta.json").exists():
        return path
    found = checkpoint.latest(path)
    if found is None:
        raise SystemExit(f"{path} 不是检查点目录，也没有 latest")
    return found


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="运行目录 / best / step_XXXXXXXX")
    p.add_argument("--in", dest="src", required=True, help="带噪 wav 目录")
    p.add_argument("--out", required=True, help="输出目录（目录名叫 wav 时，评测结果名默认取它的上一级）")
    p.add_argument("--raw", action="store_true", help="用原始权重而不是 EMA 权重")
    p.add_argument("--limit", type=int, help="只处理前 N 个文件（快速检查用）")
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
    model(np.zeros((1, 48000), "float32"))  # build the weights
    meta = checkpoint.load_weights(ckpt, model, ema=not a.raw)
    print(f"检查点 {ckpt}（step {meta['step']}，{'原始' if a.raw else 'EMA'}权重）")

    files = sorted(Path(a.src).glob("*.wav"))[: a.limit]
    if not files:
        raise SystemExit(f"{a.src} 里没有 wav")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    audio = total = 0.0
    for f in files:
        x, sr = sf.read(f, dtype="float32")
        if sr != 48000 or x.ndim != 1:
            raise SystemExit(f"{f}：需要 48kHz 单声道，实际 sr={sr} 形状={x.shape}")
        t = time.perf_counter()
        y = np.asarray(model(x[None]))[0]
        total += time.perf_counter() - t
        audio += len(x) / sr
        sf.write(out / f.name, np.clip(y, -1.0, 1.0), 48000, subtype="PCM_16")
    print(f"{len(files)} 个文件 → {out}；音频 {audio:.0f}s，处理 {total:.1f}s（RTF≈{total / audio:.3f}，离线整段，含首次编译）")


if __name__ == "__main__":
    main()
