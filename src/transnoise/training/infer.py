"""推理工具：用检查点增强一个目录里的所有 wav，输出可以直接给评测打分。
  uv run transnoise infer --ckpt data/checkpoints/baseline --in 带噪目录 --out 输出目录

--ckpt 可以是运行目录（取 latest）、运行目录/best、或某个 step_XXXXXXXX。默认用 EMA 权重，--raw 用原始权重。
模型结构取自运行目录里的 config_model.toml 快照（没有就用 configs/model.toml）。
离线整段处理：输出和输入时间对齐（没有流式延迟）；输入需要 48kHz 单声道 wav。输出 48kHz 单声道 PCM16，文件名不变。
然后打分：uv run transnoise eval score noise <输出目录> --name 名字
"""

import argparse
import time
from pathlib import Path

from ..paths import ROOT


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
    snap = ckpt.parent / "config_model.toml"
    model = Denoiser(ModelConfig.load(snap if snap.exists() else ROOT / "configs" / "model.toml"))
    model(np.zeros((1, 48000), "float32"))  # 建好权重
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
