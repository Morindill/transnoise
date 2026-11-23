"""多分辨率谱损失（参照 DFN3）+ 带宽掩码。只用 keras.ops。

每个 FFT 大小上：对压缩后的幅度 |X|^γ 和压缩后的复数谱 X·|X|^(γ-1)（保留相位）各算一个 MSE。
带宽掩码：每条样本给一个截止频率（干净语音的实测带宽），高于它的频点不计入损失——
16 kHz 以下带宽的语料在高频没有真实内容，不能让模型去"学"那里的静音。每条样本按有效频点数归一化，带宽窄的样本权重不会被稀释。
"""

import numpy as np
from keras import ops

from .config import TrainConfig


def _masked_mean(x, mask):
    """x: [B, F, K]，mask: [B, K]（0/1）→ 每条样本在有效频点上的均值，再对 batch 取平均。"""
    num = ops.sum(x * mask[:, None, :], axis=(1, 2))
    den = ops.sum(mask, axis=1) * ops.cast(ops.shape(x)[1], x.dtype)
    return ops.mean(num / ops.maximum(den, 1.0))


class SpectralLoss:
    def __init__(self, cfg: TrainConfig, sr: int = 48000):
        self.cfg, self.sr = cfg, sr
        # 每个分辨率的频点频率 [K]，用来和截止频率比较
        self.freqs = {n: np.arange(n // 2 + 1, dtype="float32") * sr / n for n in cfg.fft_sizes}

    def _spec(self, x, n):
        return ops.stft(x, n, n // 4, n, window="hann", center=True)

    def __call__(self, clean, pred, cutoff_hz):
        """clean/pred: [B, T]；cutoff_hz: [B]。返回 (总损失, {幅度项, 复数项})。"""
        g = self.cfg.gamma
        mags, cplxs = [], []
        for n in self.cfg.fft_sizes:
            mask = ops.cast(ops.convert_to_tensor(self.freqs[n])[None, :] <= cutoff_hz[:, None], "float32")  # [B, K]
            (cr, ci), (pr, pi) = self._spec(clean, n), self._spec(pred, n)
            cm, pm = ops.sqrt(cr * cr + ci * ci + 1e-12), ops.sqrt(pr * pr + pi * pi + 1e-12)
            mags.append(_masked_mean((ops.power(pm, g) - ops.power(cm, g)) ** 2, mask))
            sc, sp = ops.power(cm, g - 1.0), ops.power(pm, g - 1.0)
            cplxs.append(_masked_mean((pr * sp - cr * sc) ** 2 + (pi * sp - ci * sc) ** 2, mask))
        mag, cplx = sum(mags) / len(mags), sum(cplxs) / len(cplxs)
        return self.cfg.weight_mag * mag + self.cfg.weight_complex * cplx, {"mag": mag, "complex": cplx}
