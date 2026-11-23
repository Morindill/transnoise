"""部署参考实现自检：导出权重 → numpy 带 KV 缓存的流式实现 → 和 Keras 离线前向逐样本对拍（含负对照）。
用法：uv run transnoise check deploy [--ckpt data/checkpoints/baseline/best] [--clips 3] [--seconds 6]"""

import argparse
import time
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="data/checkpoints/baseline/best")
    p.add_argument("--clips", type=int, default=3)
    p.add_argument("--seconds", type=float, default=6.0)
    a = p.parse_args()

    import numpy as np
    import soundfile as sf

    from transnoise.deploy.export import export
    from transnoise.deploy.ref import RefStream
    from transnoise.model.config import ModelConfig
    from transnoise.model.denoiser import Denoiser
    from transnoise.paths import DATA, EVAL, ROOT
    from transnoise.training import checkpoint
    from transnoise.training.infer import resolve
    from transnoise.training.trainer import setup_gpu

    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'通过' if cond else '失败'}] {name} {detail}")

    import tensorflow as tf

    setup_gpu()
    tf.config.experimental.enable_tensor_float_32_execution(False)  # Ampere 以上默认用 TF32 做矩阵乘（尾数 10 位），会带来约 1e-3 的误差，对拍要用真正的 fp32
    ckpt = resolve(ROOT / a.ckpt if not Path(a.ckpt).is_absolute() else Path(a.ckpt))
    out = DATA / "deploy" / ckpt.parent.name / "weights.npz"
    info = export(ckpt, out)
    cfg_d = info["config"]

    snap = ckpt.parent / "config_model.toml"
    model = Denoiser(ModelConfig.load(snap if snap.exists() else ROOT / "configs" / "model.toml"))
    model(np.zeros((1, 48000), "float32"))
    checkpoint.load_weights(ckpt, model, ema=True)

    files = sorted((EVAL / "testsets" / "noise" / "noisy").glob("*.wav"))[:: max(1, 600 // a.clips)][: a.clips]
    hop, la = cfg_d["hop"], cfg_d["lookahead"]
    weights = np.load(out)
    worst_rel = worst_abs = 0.0
    for f in files:
        x, sr = sf.read(f, dtype="float32")
        assert sr == cfg_d["sr"]
        n = min(int(a.seconds * sr), len(x)) // hop * hop
        x = x[:n]
        y_off = np.asarray(model(x[None]))[0]
        ref = RefStream(weights)
        t0 = time.perf_counter()
        y_str = ref.process(x)
        dt = time.perf_counter() - t0
        L = len(x) - (la + 1) * hop
        assert len(y_str) == L
        err = np.abs(y_str - y_off[:L])
        rel = float(np.sqrt(np.mean((y_str - y_off[:L]) ** 2) / np.mean(y_off[:L] ** 2)))
        worst_rel, worst_abs = max(worst_rel, rel), max(worst_abs, float(err.max()))
        print(f"  {f.name}: 最大绝对误差 {err.max():.2e}，相对 RMS 误差 {rel:.2e}（numpy 参考 {dt / (len(x) / hop) * 1e3:.2f} ms/帧，仅供参考）")
        if f == files[0]:
            ctrl = {k: weights[k] for k in weights.files}
            ctrl["blocks.0.attn.rel_bias"] = ctrl["blocks.0.attn.rel_bias"][:, ::-1].copy()  # 负对照：偏置表反过来
            y_bad = RefStream(ctrl).process(x)
            rel_bad = float(np.sqrt(np.mean((y_bad - y_off[:L]) ** 2) / np.mean(y_off[:L] ** 2)))
    check("numpy 流式 = Keras 离线（相对 RMS 误差 < 1e-4）", worst_rel < 1e-4, f"最差 {worst_rel:.2e}，最大绝对误差 {worst_abs:.2e}")
    check("负对照：改坏一张偏置表后误差明显变大（≥ 100 倍）", rel_bad > 100 * max(worst_rel, 1e-7), f"{rel_bad:.2e}")

    # 起始段：序列开头的帧没有完整历史，最容易写错
    ref = RefStream(weights)
    x = np.random.default_rng(0).standard_normal(hop * 12).astype("float32") * 0.1
    y_off = np.asarray(model(x[None]))[0]
    y_str = ref.process(x)
    check("很短的输入（12 帧，窗口还没填满）也一致", np.abs(y_str - y_off[: len(y_str)]).max() < 1e-4 * max(1.0, np.abs(y_off).max()),
          f"最大绝对误差 {np.abs(y_str - y_off[: len(y_str)]).max():.2e}")

    print("\n全部通过" if ok else "\n有失败项")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
