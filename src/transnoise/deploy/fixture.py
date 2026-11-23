"""生成 C++ 对拍用的夹具：输入波形 + numpy 参考实现的输出和每一帧的中间量（只依赖 numpy，主环境就能跑）。
  uv run python -m transnoise.deploy.fixture data/deploy/<名>/weights.npz [--wav 文件] [--seconds 3] [--out 路径]

格式（小端）：int32 hop, nframes, n_out, n_trace；float32 输入 [nframes·hop]；float32 期望输出 [n_out]；
然后每个 trace 帧依次：emb[D]、layers[n_layers][D]（每层输出）、xn[D]（最后一层输出的归一化，无仿射）、
gains[nb_erb]、coefs[nb_df·order·2]（深度滤波系数，tanh 之后）、spec_out 的实部[bins]和虚部[bins]。
"""

import argparse
from pathlib import Path

import numpy as np

from ..paths import EVAL
from .ref import RefStream, layer_norm


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("npz", type=Path)
    p.add_argument("--wav", type=Path, default=EVAL / "testsets" / "noise" / "noisy" / "en_0000.wav")
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    import soundfile as sf

    z = np.load(a.npz)
    ref = RefStream(z, trace=True)
    hop, la, d = ref.hop, ref.la, ref.d
    x, sr = sf.read(a.wav, dtype="float32")
    assert sr == ref.cfg["sr"]
    x = x[: min(int(a.seconds * sr), len(x)) // hop * hop]
    y = ref.process(x)
    eps = float(z["norm_out.eps"])
    out = a.out or a.npz.parent / "fixture.bin"
    with out.open("wb") as f:
        f.write(np.array([hop, len(x) // hop, len(y), len(ref.trace)], "<i4").tobytes())
        f.write(x.astype("<f4").tobytes())
        f.write(y.astype("<f4").tobytes())
        for t in ref.trace:
            xn = layer_norm(t["layers"][-1], np.ones(d, "float32"), np.zeros(d, "float32"), eps)
            parts = [t["emb"], *t["layers"], xn, t["gains"], t["coefs"].reshape(-1).astype("float32"),
                     t["spec_out"].real.astype("float32"), t["spec_out"].imag.astype("float32")]
            f.write(b"".join(np.asarray(q, "<f4").tobytes() for q in parts))
    print(f"{out}：{len(x) // hop} 帧输入，{len(ref.trace)} 个 trace 帧，{len(y)} 个输出样本（{out.stat().st_size / 2**20:.1f} MiB）")


if __name__ == "__main__":
    main()
