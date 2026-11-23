"""Sliding-window self-attention + Transformer block.

Frame i only attends to frames [i-window, i+lookahead]; position is encoded by a learnable bias on the relative offset
instead of absolute positions, so the training and deployment lengths may differ and streaming does not need to know how
many frames were processed.
"""

import keras
from keras import ops


class BandedSelfAttention(keras.layers.Layer):
    def __init__(self, d_model: int, n_heads: int, window: int, lookahead: int, **kw):
        super().__init__(**kw)
        assert d_model % n_heads == 0
        self.d, self.h, self.window, self.lookahead = d_model, n_heads, window, lookahead
        self.qkv = keras.layers.Dense(3 * d_model)
        self.out = keras.layers.Dense(d_model)

    def build(self, input_shape):
        # one scalar bias per head and per relative offset (-window ... +lookahead)
        self.rel_bias = self.add_weight(name="rel_bias", shape=(self.h, self.window + self.lookahead + 1),
                                        initializer="zeros")

    def call(self, x):  # x: [B, F, D]
        f = ops.shape(x)[1]
        q, k, v = ops.split(self.qkv(x), 3, axis=-1)
        q, k, v = (ops.transpose(ops.reshape(t, (-1, f, self.h, self.d // self.h)), (0, 2, 1, 3)) for t in (q, k, v))
        scores = ops.einsum("bhid,bhjd->bhij", q, k) / (self.d // self.h) ** 0.5
        pos = ops.arange(f)
        off = pos[None, :] - pos[:, None]  # [F, F], off[i, j] = j - i
        allowed = (off >= -self.window) & (off <= self.lookahead)
        bias = ops.take(self.rel_bias, ops.clip(off + self.window, 0, self.window + self.lookahead), axis=1)  # [H,F,F]
        scores = ops.where(allowed[None, None], scores + bias[None], -1e9)
        ctx = ops.einsum("bhij,bhjd->bhid", ops.softmax(scores, axis=-1), v)
        return self.out(ops.reshape(ops.transpose(ctx, (0, 2, 1, 3)), (-1, f, self.d)))


class TransformerBlock(keras.layers.Layer):
    """Pre-LN: x + Attn(LN(x)), then x + FFN(LN(x)). LayerNorm acts on the feature axis only, never across frames, so causality is preserved."""

    def __init__(self, d_model, n_heads, ff_mult, window, lookahead, **kw):
        super().__init__(**kw)
        self.ln1, self.ln2 = keras.layers.LayerNormalization(), keras.layers.LayerNormalization()
        self.attn = BandedSelfAttention(d_model, n_heads, window, lookahead)
        self.ff1 = keras.layers.Dense(d_model * ff_mult, activation="gelu")
        self.ff2 = keras.layers.Dense(d_model)

    def call(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff2(self.ff1(self.ln2(x)))
