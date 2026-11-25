"""Model self-check (no training): python train/check_model.py. Verifies structural properties with random weights; the model pipeline is correct only if everything passes.
Usage: uv run transnoise check model"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np  # noqa: E402

from transnoise.model.config import ModelConfig  # noqa: E402
from transnoise.model.denoiser import Denoiser  # noqa: E402
from transnoise.model.frontend import erb_widths  # noqa: E402

cfg = ModelConfig.load()
rng = np.random.default_rng(0)
T = cfg.hop * 200
x = rng.standard_normal((1, T)).astype("float32") * 0.1
ok = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


m = Denoiser(cfg)
y = np.asarray(m(x))
n = sum(int(np.prod(w.shape)) for w in m.weights)
check("output shape = input shape", y.shape == x.shape, f"{y.shape}")
print(f"parameters {n / 1e6:.2f}M (DFN3 about 2.13M); ERB band widths {erb_widths(cfg)}")
print(f"latency: overall {cfg.bulk_delay_ms:.0f} ms, worst {cfg.latency_ms:.0f} ms (lookahead={cfg.lookahead})")

# 1) analysis -> synthesis reconstructs perfectly (the front and back end themselves lose nothing)
re, im = m.analysis(x)
rec = np.asarray(m.synthesis((re, im), length=T))
check("STFT analysis -> synthesis perfect reconstruction", np.abs(rec - x).max() < 1e-5, f"max error {np.abs(rec - x).max():.2e}")


# 2) causality: when the input is changed from sample s onward, the first affected output sample must not be earlier than 480*(floor(s/480) - lookahead - 1)
def first_change(model, s: int) -> int:
    x2 = x.copy()
    x2[:, s:] += 0.5
    d = np.abs(np.asarray(model(x2)) - np.asarray(model(x))).max(axis=0)
    return int(np.argmax(d > 1e-6)) if (d > 1e-6).any() else T


s = cfg.hop * 100
bound = cfg.hop * (s // cfg.hop - cfg.lookahead - 1)
n0 = first_change(m, s)
check("causality: the output never changes earlier than the lookahead budget allows", n0 >= bound, f"s={s}, earliest affected sample {n0}, lower bound {bound}")
check("lookahead is actually used and does not accumulate with depth (the bound is tight; window-edge weights are ≈0, so a few samples of slack are allowed)", n0 - bound < 16, f"layers={cfg.n_layers}, gap {n0 - bound} samples")

print("\nall passed" if ok else "\nsome checks FAILED")
raise SystemExit(0 if ok else 1)
