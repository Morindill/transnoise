"""Output heads: the ERB gain head, the deep-filter head, and the deep filter itself."""

import keras
from keras import ops

from .config import ModelConfig


class ErbGainHead(keras.layers.Layer):
    """embedding -> nb_erb band gains in [0, 1]."""

    def __init__(self, cfg: ModelConfig, **kw):
        super().__init__(**kw)
        self.fc = keras.layers.Dense(cfg.nb_erb, activation="sigmoid")

    def call(self, h):
        return self.fc(h)


class DeepFilterHead(keras.layers.Layer):
    """embedding -> one complex coefficient per frame, low-frequency bin and filter tap: [B, F, nb_df, order, 2]."""

    def __init__(self, cfg: ModelConfig, **kw):
        super().__init__(**kw)
        self.cfg = cfg
        self.fc = keras.layers.Dense(cfg.nb_df * cfg.df_order * 2, activation="tanh")

    def call(self, h):
        c = self.cfg
        return ops.reshape(self.fc(h), (-1, ops.shape(h)[1], c.nb_df, c.df_order, 2))


def deep_filter(re, im, coefs, order: int, lookahead: int):
    """Per bin, an order-tap complex FIR over time: Y[t] = sum_k W[t,k] * X[t + lookahead - k] (k = 0 ... order-1).
    re/im: [B, F, nb_df] (the lowest nb_df bins of the noisy spectrum); coefs: [B, F, nb_df, order, 2]; missing frames count as 0."""
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
