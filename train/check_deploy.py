"""Deployment reference self-check: export weights -> numpy streaming implementation with a KV cache -> compare sample by sample with the Keras offline forward (including a negative control).
Usage: uv run transnoise check deploy [--ckpt data/checkpoints/baseline/best] [--clips 3] [--seconds 6]"""

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
    from transnoise.paths import DATA, EVAL, ROOT, SR
    from transnoise.training import checkpoint
    from transnoise.training.infer import resolve
    from transnoise.training.trainer import setup_gpu

    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

    import tensorflow as tf

    setup_gpu()
    tf.config.experimental.enable_tensor_float_32_execution(False)  # Ampere and newer use TF32 for matmuls by default (10-bit mantissa), which adds about 1e-3 of error; the comparison needs true fp32
    ckpt = resolve(ROOT / a.ckpt if not Path(a.ckpt).is_absolute() else Path(a.ckpt))
    out = DATA / "deploy" / ckpt.parent.name / "weights.npz"
    info = export(ckpt, out)
    cfg_d = info["config"]

    model = Denoiser(ModelConfig.from_checkpoint(ckpt))
    model(np.zeros((1, SR), "float32"))
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
        print(f"  {f.name}: max absolute error {err.max():.2e}, relative RMS error {rel:.2e} (numpy reference {dt / (len(x) / hop) * 1e3:.2f} ms/frame; for reference only)")
        if f == files[0]:
            ctrl = {k: weights[k] for k in weights.files}
            ctrl["blocks.0.attn.rel_bias"] = ctrl["blocks.0.attn.rel_bias"][:, ::-1].copy()  # negative control: reverse the bias table
            y_bad = RefStream(ctrl).process(x)
            rel_bad = float(np.sqrt(np.mean((y_bad - y_off[:L]) ** 2) / np.mean(y_off[:L] ** 2)))
    check("numpy streaming = Keras offline (relative RMS error < 1e-4)", worst_rel < 1e-4, f"worst {worst_rel:.2e}, max absolute error {worst_abs:.2e}")
    check("negative control: corrupting one bias table makes the error clearly larger (>= 100x)", rel_bad > 100 * max(worst_rel, 1e-7), f"{rel_bad:.2e}")

    # Start of the sequence: the first frames have no full history, the easiest place to get wrong
    ref = RefStream(weights)
    x = np.random.default_rng(0).standard_normal(hop * 12).astype("float32") * 0.1
    y_off = np.asarray(model(x[None]))[0]
    y_str = ref.process(x)
    check("very short input (12 frames, window not yet full) also matches", np.abs(y_str - y_off[: len(y_str)]).max() < 1e-4 * max(1.0, np.abs(y_off).max()),
          f"max absolute error {np.abs(y_str - y_off[: len(y_str)]).max():.2e}")

    print("\nall passed" if ok else "\nsome checks FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
