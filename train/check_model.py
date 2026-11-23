"""模型自检（不训练）：python train/check_model.py。用随机权重验证结构性质，全部通过才算模型管线正确。
用法：uv run transnoise check model"""

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
    print(f"[{'通过' if cond else '失败'}] {name} {detail}")


m = Denoiser(cfg)
y = np.asarray(m(x))
n = sum(int(np.prod(w.shape)) for w in m.weights)
check("输出形状 = 输入形状", y.shape == x.shape, f"{y.shape}")
print(f"参数量 {n / 1e6:.2f}M（DFN3 约 2.13M）；ERB 频带宽度 {erb_widths(cfg)}")
print(f"延迟：整体 {cfg.bulk_delay_ms:.0f} ms，最坏 {cfg.latency_ms:.0f} ms（lookahead={cfg.lookahead}）")

# 1) 分析→合成完美重建（前后端本身没有损耗）
re, im = m.analysis(x)
rec = np.asarray(m.synthesis((re, im), length=T))
check("STFT 分析→合成 完美重建", np.abs(rec - x).max() < 1e-5, f"最大误差 {np.abs(rec - x).max():.2e}")


# 2) 因果性：从样本 s 起改动输入，输出里受影响的第一个样本不能早于 480·(floor(s/480) - lookahead - 1)
def first_change(model, s: int) -> int:
    x2 = x.copy()
    x2[:, s:] += 0.5
    d = np.abs(np.asarray(model(x2)) - np.asarray(model(x))).max(axis=0)
    return int(np.argmax(d > 1e-6)) if (d > 1e-6).any() else T


s = cfg.hop * 100
bound = cfg.hop * (s // cfg.hop - cfg.lookahead - 1)
n0 = first_change(m, s)
check("因果性：输出不会比输入早变化超过前视预算", n0 >= bound, f"s={s}，最早受影响样本 {n0}，下界 {bound}")
check("前视确实被用到、且没有随层数累加（下界是紧的；窗口边缘权重≈0，容许差几个样本）", n0 - bound < 16, f"层数={cfg.n_layers}，差 {n0 - bound} 个样本")

# 3) 深度滤波开关
m0 = Denoiser(ModelConfig(**{**cfg.__dict__, "df_order": 0}))
check("df_order=0 可运行", np.asarray(m0(x)).shape == x.shape, f"参数量 {sum(int(np.prod(w.shape)) for w in m0.weights) / 1e6:.2f}M")

print("\n全部通过" if ok else "\n有失败项")
raise SystemExit(0 if ok else 1)
