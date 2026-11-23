"""训练核心自检（合成数据，不用真实数据集）：损失的带宽掩码、过拟合一个 batch（梯度能流通）、断点续训与不中断完全一致。
用法：uv run transnoise check training"""

import os
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np  # noqa: E402

import keras  # noqa: E402
import tensorflow as tf  # noqa: E402

# GPU 上默认的反向传播有原子加，同样的种子跑两次也差 1e-3 量级；本检查需要逐位可复现才能判断续训是否精确，所以打开确定性模式（只有这里需要，真实训练不用）
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
    print(f"[{'通过' if cond else '失败'}] {name} {detail}")


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
    keras.utils.set_random_seed(seed)  # 初始权重由构造 Denoiser 时的全局种子决定（不是 Trainer 里），所以要在构造之前设
    return Trainer(Denoiser(mcfg), tcfg, run_dir)


# 1) 带宽掩码：在 15 kHz 处加一个强正弦。截止 8 kHz 时它必须被忽略；截止 24 kHz 时损失明显变大
loss = SpectralLoss(tcfg, SR)
pred_hf = clean + 0.2 * np.sin(2 * np.pi * 15000 * t).astype("float32")
l_masked = float(loss(clean, pred_hf, np.full(B, 8000.0, "float32"))[0])
l_open = float(loss(clean, pred_hf, full)[0])
l_zero = float(loss(clean, clean, full)[0])
check("损失：预测等于干净时为 0", l_zero < 1e-6, f"{l_zero:.2e}")
# 剩余的 4% 是 15kHz 正弦经 Hann 窗泄漏到低频的部分，在 γ=0.3 的压缩下被放大，不是掩码失效
check("损失：带宽掩码生效（高于截止频率的误差被忽略）", l_masked < 0.1 * l_open, f"截止8k={l_masked:.2e}，不截止={l_open:.2e}")

# 2) 过拟合一个 batch：损失能大幅下降，说明模型 → 损失 → 优化器整条梯度链是通的
tr = new(Path(tempfile.mkdtemp()))
first = tr.train_step(noisy, clean, full)["loss"]
for _ in range(80):
    last = tr.train_step(noisy, clean, full)["loss"]
check("过拟合一个 batch：损失下降超过一半", last < 0.5 * first and np.isfinite(last), f"{first:.3f} → {last:.3f}，lr={tr.lr:.2e}，step={tr.step}")
shutil.rmtree(tr.run_dir, ignore_errors=True)

# 3) 断点续训：训练 8 步 vs（训练 4 步 → 存档 → 全新对象恢复 → 再训练 4 步），损失和权重必须一致
batches = [((noisy * (1 + 0.1 * i)).astype("float32"), (clean * (1 + 0.1 * i)).astype("float32")) for i in range(8)]
run = Path(tempfile.mkdtemp())
a = new(run / "a")
loss_a = [a.train_step(n, c, full)["loss"] for n, c in batches]
b = new(run / "b")
loss_b = [b.train_step(n, c, full)["loss"] for n, c in batches[:4]]
ck = b.save({"epoch": 3, "batches_done": 4})
c = new(run / "b", seed=1)  # 全新模型 + 全新优化器，初始权重故意用另一个种子，必须被检查点完全覆盖
meta = c.resume()
loss_c = [c.train_step(n, x, full)["loss"] for n, x in batches[4:]]
check("恢复：step 和训练循环状态", c.step == 8 and meta["step"] == 4 and meta["extra"] == {"epoch": 3, "batches_done": 4},
      f"step={c.step}，extra={meta['extra']}")
resumed, plain = loss_b + loss_c, loss_a
rel = max(abs(x - y) / abs(y) for x, y in zip(resumed, plain))
wdiff = max(float(np.abs(np.asarray(p) - np.asarray(q)).max()) for p, q in zip(a.model.weights, c.model.weights))
check("断点续训 = 不中断训练（逐步损失，逐位一致）", rel < 1e-6, f"最大相对差 {rel:.2e}")
check("断点续训 = 不中断训练（最终权重，逐位一致）", wdiff < 1e-6, f"最大权重差 {wdiff:.2e}")
ediff = max(float(np.abs(np.asarray(p) - np.asarray(q)).max()) for p, q in zip(a.ema, c.ema))
check("断点续训 = 不中断训练（EMA 权重，逐位一致）", ediff < 1e-6, f"最大差 {ediff:.2e}")
check("存档目录结构", all((ck / f).exists() for f in ("model.npz", "optim.npz", "ema.npz", "meta.json")) and (run / "b" / "latest").exists(), ck.name)

# 3b) EMA：和原始权重不同；ema_weights() 内模型权重 = EMA，退出后还原
w0 = [np.asarray(w).copy() for w in a.model.weights]
check("EMA 权重与原始权重不同", max(float(np.abs(p - np.asarray(q)).max()) for p, q in zip(w0, a.ema)) > 1e-6)
with a.ema_weights():
    inside = max(float(np.abs(np.asarray(w) - np.asarray(e)).max()) for w, e in zip(a.model.weights, a.ema))
back = max(float(np.abs(np.asarray(w) - p).max()) for w, p in zip(a.model.weights, w0))
check("ema_weights()：内部用 EMA，退出后还原", inside < 1e-7 and back == 0.0, f"内部差 {inside:.1e}，还原差 {back:.1e}")

# 3c) NaN 保护：带 NaN 的 batch 必须整步跳过（step、权重、EMA 都不动），之后正常训练不受影响
step0, w0 = a.step, [np.asarray(w).copy() for w in a.model.weights]
e0 = [np.asarray(e).copy() for e in a.ema]
bad_noisy = batches[0][0].copy()
bad_noisy[0, 100] = np.nan
res = a.train_step(bad_noisy, batches[0][1], full)
same = all(np.array_equal(np.asarray(w), p) for w, p in zip(a.model.weights, w0)) and all(np.array_equal(np.asarray(e), p) for e, p in zip(a.ema, e0))
check("NaN 保护：整步跳过，状态不变", res["skipped"] and a.step == step0 and same, f"skipped={res['skipped']}，step {step0}→{a.step}")
res2 = a.train_step(*batches[0], full)
check("NaN 之后可以继续正常训练", (not res2["skipped"]) and a.step == step0 + 1 and np.isfinite(res2["loss"]))

# 4) 结构对不上时必须报错
bad = Trainer(Denoiser(ModelConfig(**{**mcfg.__dict__, "n_layers": 2})), tcfg, run / "b")
try:
    bad.resume()
    check("结构不一致时拒绝加载", False)
except ValueError as e:
    check("结构不一致时拒绝加载", True, str(e)[:40] + "…")
shutil.rmtree(run, ignore_errors=True)

print("\n全部通过" if ok else "\n有失败项")
raise SystemExit(0 if ok else 1)
