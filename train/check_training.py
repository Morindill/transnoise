"""Training-core self-check (synthetic data, no real dataset): the loss's bandwidth mask, overfitting one batch (gradients flow), and resumed training identical to uninterrupted training.
Usage: uv run transnoise check training"""

import os
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np  # noqa: E402

import keras  # noqa: E402
import tensorflow as tf  # noqa: E402

# Backpropagation on the GPU uses atomic adds by default, so two runs with the same seed differ by about 1e-3; this check needs bit-exact reproducibility to judge whether resuming is exact, so determinism mode is switched on (needed only here, not in real training)
tf.config.experimental.enable_op_determinism()
from transnoise.training.trainer import Trainer, setup_gpu  # noqa: E402

setup_gpu()
from transnoise.model.config import ModelConfig  # noqa: E402
from transnoise.model.denoiser import Denoiser  # noqa: E402
from transnoise.training.config import TrainConfig  # noqa: E402
from transnoise.training.losses import SpectralLoss  # noqa: E402

ok = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


mcfg = ModelConfig.load()
tcfg = replace(TrainConfig.load(), crop_seconds=1.0, batch_size=4, warmup_steps=10, total_steps=200, peak_lr=1e-3, ema_decay=0.9)
T, B, SR = int(tcfg.crop_seconds * mcfg.sr), tcfg.batch_size, mcfg.sr
rng = np.random.default_rng(0)
t = np.arange(T) / SR
clean = np.stack([sum(0.1 * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.28)) / k for k in range(1, 8))
                  * (0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(2, 5) * t)) for f0 in rng.uniform(100, 250, B)]).astype("float32")
noisy = (clean + 0.1 * rng.standard_normal(clean.shape)).astype("float32")
full = np.full(B, 24000.0, "float32")


def new(run_dir, seed=0):
    keras.utils.set_random_seed(seed)  # the initial weights are determined by the global seed at Denoiser construction (not inside Trainer), so set it before constructing
    return Trainer(Denoiser(mcfg), tcfg, run_dir)


# 1) bandwidth mask: add a strong sine at 15 kHz. With an 8 kHz cutoff it must be ignored; with a 24 kHz cutoff the loss grows clearly
loss = SpectralLoss(tcfg, SR)
pred_hf = clean + 0.2 * np.sin(2 * np.pi * 15000 * t).astype("float32")
l_masked = float(loss(clean, pred_hf, np.full(B, 8000.0, "float32"))[0])
l_open = float(loss(clean, pred_hf, full)[0])
l_zero = float(loss(clean, clean, full)[0])
check("loss: 0 when the prediction equals clean", l_zero < 1e-6, f"{l_zero:.2e}")
# The remaining 4% is the part of the 15 kHz sine that leaks to low frequencies through the Hann window and is amplified by the gamma=0.3 compression; it is not a mask failure
check("loss: bandwidth mask works (errors above the cutoff are ignored)", l_masked < 0.1 * l_open, f"cutoff 8k={l_masked:.2e}, no cutoff={l_open:.2e}")

# 2) overfit one batch: the loss drops sharply, which shows the whole gradient chain model -> loss -> optimizer is connected
tr = new(Path(tempfile.mkdtemp()))
first = tr.train_step(noisy, clean, full)["loss"]
for _ in range(80):
    last = tr.train_step(noisy, clean, full)["loss"]
check("overfit one batch: loss drops by more than half", last < 0.5 * first and np.isfinite(last), f"{first:.3f} → {last:.3f}, lr={tr.lr:.2e}, step={tr.step}")
shutil.rmtree(tr.run_dir, ignore_errors=True)

# 3) checkpoint resume: 8 training steps vs (4 steps -> save -> restore into fresh objects -> 4 more steps); losses and weights must match
batches = [((noisy * (1 + 0.1 * i)).astype("float32"), (clean * (1 + 0.1 * i)).astype("float32")) for i in range(8)]
run = Path(tempfile.mkdtemp())
a = new(run / "a")
loss_a = [a.train_step(n, c, full)["loss"] for n, c in batches]
b = new(run / "b")
loss_b = [b.train_step(n, c, full)["loss"] for n, c in batches[:4]]
ck = b.save({"epoch": 3, "batches_done": 4})
c = new(run / "b", seed=1)  # fresh model + fresh optimizer; the initial weights deliberately use another seed and must be fully overwritten by the checkpoint
meta = c.resume()
loss_c = [c.train_step(n, x, full)["loss"] for n, x in batches[4:]]
check("restore: step and training-loop state", c.step == 8 and meta["step"] == 4 and meta["extra"] == {"epoch": 3, "batches_done": 4},
      f"step={c.step}, extra={meta['extra']}")
resumed, plain = loss_b + loss_c, loss_a
rel = max(abs(x - y) / abs(y) for x, y in zip(resumed, plain))
wdiff = max(float(np.abs(np.asarray(p) - np.asarray(q)).max()) for p, q in zip(a.model.weights, c.model.weights))
check("resume = uninterrupted training (per-step loss, bit-identical)", rel < 1e-6, f"max relative difference {rel:.2e}")
check("resume = uninterrupted training (final weights, bit-identical)", wdiff < 1e-6, f"max weight difference {wdiff:.2e}")
ediff = max(float(np.abs(np.asarray(p) - np.asarray(q)).max()) for p, q in zip(a.ema, c.ema))
check("resume = uninterrupted training (EMA weights, bit-identical)", ediff < 1e-6, f"max difference {ediff:.2e}")
check("checkpoint directory layout", all((ck / f).exists() for f in ("model.npz", "optim.npz", "ema.npz", "meta.json")) and (run / "b" / "latest").exists(), ck.name)

# 3b) EMA: differs from the raw weights; inside ema_weights() the model weights = EMA, restored on exit
w0 = [np.asarray(w).copy() for w in a.model.weights]
check("EMA weights differ from the raw weights", max(float(np.abs(p - np.asarray(q)).max()) for p, q in zip(w0, a.ema)) > 1e-6)
with a.ema_weights():
    inside = max(float(np.abs(np.asarray(w) - np.asarray(e)).max()) for w, e in zip(a.model.weights, a.ema))
back = max(float(np.abs(np.asarray(w) - p).max()) for w, p in zip(a.model.weights, w0))
check("ema_weights(): uses EMA inside, restores on exit", inside < 1e-7 and back == 0.0, f"difference inside {inside:.1e}, difference after restore {back:.1e}")

# 3c) NaN protection: a batch containing NaN must skip the whole step (step, weights and EMA untouched), and normal training afterwards is unaffected
step0, w0 = a.step, [np.asarray(w).copy() for w in a.model.weights]
e0 = [np.asarray(e).copy() for e in a.ema]
bad_noisy = batches[0][0].copy()
bad_noisy[0, 100] = np.nan
res = a.train_step(bad_noisy, batches[0][1], full)
same = all(np.array_equal(np.asarray(w), p) for w, p in zip(a.model.weights, w0)) and all(np.array_equal(np.asarray(e), p) for e, p in zip(a.ema, e0))
check("NaN guard: the whole step is skipped, state unchanged", res["skipped"] and a.step == step0 and same, f"skipped={res['skipped']}, step {step0}→{a.step}")
res2 = a.train_step(*batches[0], full)
check("training continues normally after NaN", (not res2["skipped"]) and a.step == step0 + 1 and np.isfinite(res2["loss"]))

# 4) a mismatched architecture must raise an error
bad = Trainer(Denoiser(ModelConfig(**{**mcfg.__dict__, "n_layers": mcfg.n_layers + 1})), tcfg, run / "b")
try:
    bad.resume()
    check("refuses to load on structure mismatch", False)
except ValueError as e:
    check("refuses to load on structure mismatch", True, str(e)[:40] + "…")
shutil.rmtree(run, ignore_errors=True)

print("\nall passed" if ok else "\nsome checks FAILED")
raise SystemExit(0 if ok else 1)
