"""Export a checkpoint as deployment weights (runs only in the train environment: a Keras model has to be built to read the weights).
  uv run transnoise export --ckpt data/checkpoints/<name>/best [--out FILE] [--raw]

Writes one npz: a named array per tensor (blocks.0.attn.qkv.kernel ...), plus meta (JSON: all model hyperparameters + ERB band widths),
window (the Vorbis window) and the eps of every LayerNorm. Names are determined by the code structure, independent of Keras auto-numbering;
the numpy reference and the C++ code both look tensors up by name.
EMA weights by default (same as infer). LayerNorm's eps and GELU's exact form (erf) follow Keras's actual behavior; the eps is exported along with the weights.
"""

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from ..paths import DATA


def collect(model) -> dict[str, np.ndarray]:
    """Collect all weights by structure (not by variable order)."""
    import keras

    to_np = keras.ops.convert_to_numpy
    out: dict[str, np.ndarray] = {}

    def dense(name, layer):
        out[f"{name}.kernel"], out[f"{name}.bias"] = to_np(layer.kernel), to_np(layer.bias)

    def norm(name, layer):
        out[f"{name}.gamma"], out[f"{name}.beta"] = to_np(layer.gamma), to_np(layer.beta)
        out[f"{name}.eps"] = np.array(layer.epsilon, "float32")
        assert layer.axis in (-1, [-1], (-1,), [2], (2,)), layer.axis  # only normalization over the last axis is supported
        assert layer.center and layer.scale

    dense("erb_in", model.erb_in)
    dense("spec_in", model.spec_in)
    dense("mix", model.mix)
    for i, b in enumerate(model.blocks):
        p = f"blocks.{i}"
        norm(f"{p}.ln_ffa", b.ln_ffa)
        dense(f"{p}.ffa1", b.ffa1)
        dense(f"{p}.ffa2", b.ffa2)
        norm(f"{p}.ln_att", b.ln_att)
        dense(f"{p}.attn.qkv", b.attn.qkv)
        out[f"{p}.attn.rel_bias"] = to_np(b.attn.rel_bias)
        dense(f"{p}.attn.out", b.attn.out)
        norm(f"{p}.ln_conv", b.ln_conv)
        dense(f"{p}.pw1", b.pw1)
        dense(f"{p}.dw", b.dw)
        norm(f"{p}.dw_ln", b.dw_ln)
        dense(f"{p}.pw2", b.pw2)
        norm(f"{p}.ln_ffb", b.ln_ffb)
        dense(f"{p}.ffb1", b.ffb1)
        dense(f"{p}.ffb2", b.ffb2)
    norm("norm_out", model.norm_out)
    dense("gain", model.gain_head.fc)
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
    cfg = ModelConfig.from_checkpoint(ckpt)
    model = Denoiser(cfg)
    model(np.zeros((1, cfg.hop * 8), "float32"))  # build the weights
    meta = checkpoint.load_weights(ckpt, model, ema=ema)

    arrays = collect(model)
    n_params = sum(int(np.prod(a.shape)) for k, a in arrays.items() if not k.endswith(".eps"))
    assert n_params == sum(int(np.prod(w.shape)) for w in model.weights), "some weights were not exported"
    info = {"config": dataclasses.asdict(cfg), "erb_widths": erb_widths(cfg), "step": meta["step"],
            "ema": ema, "ckpt": str(ckpt), "keras": keras.__version__}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, meta=np.array(json.dumps(info)), window=vorbis_window(cfg.fft), **arrays)
    print(f"{out_path}: {len(arrays)} tensors, {n_params:,} parameters (step {meta['step']}, {'EMA' if ema else 'raw'} weights)")
    return info


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="run directory / best / step_XXXXXXXX")
    p.add_argument("--out", help="default: data/deploy/<run name>/weights.npz")
    p.add_argument("--raw", action="store_true", help="use the raw weights instead of the EMA weights")
    a = p.parse_args()
    from ..training.infer import resolve

    ckpt = resolve(Path(a.ckpt))
    export(ckpt, Path(a.out) if a.out else DATA / "deploy" / ckpt.parent.name / "weights.npz", ema=not a.raw)


if __name__ == "__main__":
    main()
