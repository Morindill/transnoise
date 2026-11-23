"""STFT 前后端和 ERB 分带。没有可训练参数。

帧约定（流式和离线共用）：波形前面补 fft-hop 个零、后面补 fft 个零（再补到 hop 的整数倍），
第 t 帧覆盖补零后的 [t·hop, t·hop+fft)，即原始波形的 [(t-1)·hop, (t+1)·hop)。
合成用同一个 Vorbis 窗（满足 Princen-Bradley），重叠相加后丢掉前面的 fft-hop 个样本，分析→合成可完美重建。
"""

import keras
import numpy as np
from keras import ops

from .config import ModelConfig


def vorbis_window(n: int) -> np.ndarray:
    k = np.arange(n)
    return np.sin(np.pi / 2 * np.sin(np.pi * (k + 0.5) / n) ** 2).astype("float32")


def _erb(f):  # Hz -> ERB 刻度
    return 9.265 * np.log(1 + f / (24.7 * 9.265))


def _inv_erb(e):
    return 24.7 * 9.265 * (np.exp(e / 9.265) - 1)


def erb_widths(cfg: ModelConfig) -> list[int]:
    """每个 ERB 频带包含的 STFT 频点数：频带在 ERB 刻度上等分，至少 min_erb_bins 个频点，总和 = bins。"""
    freq_width = cfg.sr / cfg.fft
    lo, hi = _erb(0.0), _erb(cfg.sr / 2)
    step = (hi - lo) / cfg.nb_erb
    widths, prev, over = [], 0, 0
    for i in range(1, cfg.nb_erb + 1):
        edge = int(round(_inv_erb(lo + i * step) / freq_width))
        n = edge - prev - over
        over = max(cfg.min_erb_bins - n, 0)
        widths.append(max(n, cfg.min_erb_bins))
        prev = edge
    widths[-1] += 1
    widths[-1] -= max(sum(widths) - cfg.bins, 0)
    assert sum(widths) == cfg.bins, (sum(widths), cfg.bins)
    return widths


def erb_matrices(cfg: ModelConfig) -> tuple[np.ndarray, np.ndarray]:
    """forward [bins, nb_erb]：把功率谱按频带取平均；inverse [nb_erb, bins]：把频带增益复制回每个频点。"""
    widths = erb_widths(cfg)
    fwd = np.zeros((cfg.bins, cfg.nb_erb), "float32")
    inv = np.zeros((cfg.nb_erb, cfg.bins), "float32")
    start = 0
    for b, w in enumerate(widths):
        fwd[start:start + w, b] = 1.0 / w
        inv[b, start:start + w] = 1.0
        start += w
    return fwd, inv


class Analysis(keras.layers.Layer):
    """波形 [B, T] → 频谱 (实部, 虚部)，各 [B, F, bins]，F = T/hop + 2。T 必须是静态已知长度。"""

    def __init__(self, cfg: ModelConfig, **kw):
        super().__init__(**kw)
        self.cfg, self.win = cfg, vorbis_window(cfg.fft)

    def call(self, x):
        c = self.cfg
        t = x.shape[-1]
        extra = (-t) % c.hop
        x = ops.pad(x, [[0, 0], [c.fft - c.hop, extra + c.fft]])
        return ops.stft(x, c.fft, c.hop, c.fft, window=self.win, center=False)


class Synthesis(keras.layers.Layer):
    """频谱 (实部, 虚部) → 波形 [B, length]。"""

    def __init__(self, cfg: ModelConfig, **kw):
        super().__init__(**kw)
        self.cfg, self.win = cfg, vorbis_window(cfg.fft)

    def call(self, spec, length: int):
        c = self.cfg
        y = ops.istft(spec, c.fft, c.hop, c.fft, window=self.win, center=False)
        return y[:, c.fft - c.hop: c.fft - c.hop + length]
