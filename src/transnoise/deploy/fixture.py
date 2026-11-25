"""Generate the fixture for C++ cross-checking: an input waveform + the numpy reference's output and per-frame intermediates (numpy only, runs in the main environment).
  uv run python -m transnoise.deploy.fixture data/deploy/<name>/weights.npz [--wav FILE] [--seconds 3] [--out PATH]

Format (little-endian): int32 hop, nframes, n_out, n_trace; float32 input [nframes*hop]; float32 expected output [n_out];
then for each trace frame in turn: pre[D] (the input of layer 0's attention), layers[n_layers][D] (every layer's output), xn[D] (normalized last-layer output, no affine),
gains[nb_erb], coefs[nb_df*order*2] (deep-filter coefficients, after tanh), the real part of spec_out [bins] and its imaginary part [bins].
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
            parts = [t["pre"], *t["layers"], xn, t["gains"], t["coefs"].reshape(-1).astype("float32"),
                     t["spec_out"].real.astype("float32"), t["spec_out"].imag.astype("float32")]
            f.write(b"".join(np.asarray(q, "<f4").tobytes() for q in parts))
    print(f"{out}: {len(x) // hop} input frames, {len(ref.trace)} trace frames, {len(y)} output samples ({out.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
