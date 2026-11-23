"""输出头：ERB 增益头、深度滤波头，以及深度滤波本身。"""

import keras
from keras import ops

from .config import ModelConfig


class ErbGainHead(keras.layers.Layer):
    """embedding → nb_erb 个 0–1 的频带增益。"""

    def __init__(self, cfg: ModelConfig, **kw):
        super().__init__(**kw)
        self.fc = keras.layers.Dense(cfg.nb_erb, activation="sigmoid")

    def call(self, h):
        return self.fc(h)


class DeepFilterHead(keras.layers.Layer):
    """embedding → 每帧、每个低频频点、每个滤波阶数一个复数系数 [B, F, nb_df, order, 2]。"""

    def __init__(self, cfg: ModelConfig, **kw):
        super().__init__(**kw)
        self.cfg = cfg
        self.fc = keras.layers.Dense(cfg.nb_df * cfg.df_order * 2, activation="tanh")

    def call(self, h):
        c = self.cfg
        return ops.reshape(self.fc(h), (-1, ops.shape(h)[1], c.nb_df, c.df_order, 2))


def deep_filter(re, im, coefs, order: int, lookahead: int):
    """对每个频点在时间上做 order 阶复数 FIR：Y[t] = Σ_k W[t,k] · X[t + lookahead - k]（k = 0 … order-1）。
    re/im: [B, F, nb_df]（带噪谱的最低 nb_df 个频点）；coefs: [B, F, nb_df, order, 2]；不存在的帧按 0 处理。"""
    f = ops.shape(re)[1]
    left, right = order - 1 - lookahead, lookahead
    assert left >= 0, "df_order 必须 > lookahead"
    pad = [[0, 0], [left, right], [0, 0]]
    pr, pi = ops.pad(re, pad), ops.pad(im, pad)
    out_r = out_i = 0.0
    for k in range(order):
        xr, xi = pr[:, order - 1 - k: order - 1 - k + f], pi[:, order - 1 - k: order - 1 - k + f]
        wr, wi = coefs[..., k, 0], coefs[..., k, 1]
        out_r = out_r + wr * xr - wi * xi
        out_i = out_i + wr * xi + wi * xr
    return out_r, out_i
