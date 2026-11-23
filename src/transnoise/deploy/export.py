"""把检查点导出成部署用的权重文件（只在 train 环境里运行：要建 Keras 模型来读权重）。
  uv run transnoise export --ckpt data/checkpoints/<名>/best [--out 文件] [--raw]

输出一个 npz：每个张量一个有名字的数组（blocks.0.attn.qkv.kernel …），加上 meta（JSON：全部模型超参 + ERB 频带宽度）、
window（Vorbis 窗）和每个 LayerNorm 的 eps。名字由代码结构决定，不依赖 Keras 自动编号，numpy 参考实现和 C++ 都按名字取。
默认用 EMA 权重（和 infer 一致）。LayerNorm 的 eps、GELU 的精确形式（erf）都以 Keras 实际行为为准，这里把 eps 一并导出。
"""

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from ..paths import DATA, ROOT


def collect(model) -> dict[str, np.ndarray]:
    """按结构（不是按变量顺序）收集全部权重。"""
    import keras

    to_np = keras.ops.convert_to_numpy
    out: dict[str, np.ndarray] = {}

    def dense(name, layer):
        out[f"{name}.kernel"], out[f"{name}.bias"] = to_np(layer.kernel), to_np(layer.bias)

    def norm(name, layer):
        out[f"{name}.gamma"], out[f"{name}.beta"] = to_np(layer.gamma), to_np(layer.beta)
        out[f"{name}.eps"] = np.array(layer.epsilon, "float32")
        assert layer.axis in (-1, [-1], (-1,), [2], (2,)), layer.axis  # 只支持在最后一维上做
        assert layer.center and layer.scale

    dense("erb_in", model.erb_in)
    dense("spec_in", model.spec_in)
    dense("mix", model.mix)
    for i, b in enumerate(model.blocks):
        p = f"blocks.{i}"
        norm(f"{p}.ln1", b.ln1)
        dense(f"{p}.attn.qkv", b.attn.qkv)
        out[f"{p}.attn.rel_bias"] = to_np(b.attn.rel_bias)
        dense(f"{p}.attn.out", b.attn.out)
        norm(f"{p}.ln2", b.ln2)
        dense(f"{p}.ff1", b.ff1)
        dense(f"{p}.ff2", b.ff2)
    norm("norm_out", model.norm_out)
    dense("gain", model.gain_head.fc)
    if model.df_head is not None:
        dense("df", model.df_head.fc)
    return out


def export(ckpt: Path, out_path: Path, ema: bool = True) -> dict:
    import keras

    from ..model.config import ModelConfig
    from ..model.denoiser import Denoiser
    from ..model.frontend import erb_widths, vorbis_window
    from ..training import checkpoint
    from ..training.infer import resolve

    ckpt = resolve(ckpt)
    snap = ckpt.parent / "config_model.toml"
    cfg = ModelConfig.load(snap if snap.exists() else ROOT / "configs" / "model.toml")
    model = Denoiser(cfg)
    model(np.zeros((1, cfg.hop * 8), "float32"))  # 建好权重
    meta = checkpoint.load_weights(ckpt, model, ema=ema)

    arrays = collect(model)
    n_params = sum(int(np.prod(a.shape)) for k, a in arrays.items() if not k.endswith(".eps"))
    assert n_params == sum(int(np.prod(w.shape)) for w in model.weights), "有权重没被导出"
    info = {"config": dataclasses.asdict(cfg), "erb_widths": erb_widths(cfg), "step": meta["step"],
            "ema": ema, "ckpt": str(ckpt), "keras": keras.__version__}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, meta=np.array(json.dumps(info)), window=vorbis_window(cfg.fft), **arrays)
    print(f"{out_path}：{len(arrays)} 个张量，{n_params:,} 个参数（step {meta['step']}，{'EMA' if ema else '原始'}权重）")
    return info


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="运行目录 / best / step_XXXXXXXX")
    p.add_argument("--out", help="默认 data/deploy/<运行名>/weights.npz")
    p.add_argument("--raw", action="store_true", help="用原始权重而不是 EMA 权重")
    a = p.parse_args()
    from ..training.infer import resolve

    ckpt = resolve(Path(a.ckpt))
    export(ckpt, Path(a.out) if a.out else DATA / "deploy" / ckpt.parent.name / "weights.npz", ema=not a.raw)


if __name__ == "__main__":
    main()
