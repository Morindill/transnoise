"""Streaming Conformer building blocks: sliding-window self-attention, causal depthwise convolution and the macaron Conformer block.

Attention: frame i only attends to frames [i-window, i+lookahead]; position is encoded by a learnable bias on the relative offset
instead of absolute positions, so the training and deployment lengths may differ and streaming does not need to know how
many frames were processed.
Convolution: strictly causal, so it adds history but no lookahead.
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


class CausalDepthwiseConv(keras.layers.Layer):
    """Per-channel FIR over time: y[t] = bias + sum_k kernel[k] * x[t - size + 1 + k] (k = 0 ... size-1); frames before 0 count as 0."""

    def __init__(self, size: int, **kw):
        super().__init__(**kw)
        self.size = size

    def build(self, input_shape):
        d = input_shape[-1]
        self.kernel = self.add_weight(name="kernel", shape=(self.size, d), initializer="glorot_uniform")
        self.bias = self.add_weight(name="bias", shape=(d,), initializer="zeros")

    def call(self, x):  # x: [B, F, D]
        f = ops.shape(x)[1]
        p = ops.pad(x, [[0, 0], [self.size - 1, 0], [0, 0]])
        y = self.bias
        for k in range(self.size):
            y = y + p[:, k: k + f] * self.kernel[k]
        return y


class ConformerBlock(keras.layers.Layer):
    """Macaron Conformer block, pre-LN with residuals:
        x += 0.5 * FFN(LN(x))
        x += Attention(LN(x))
        x += Conv(LN(x))    # pointwise D -> 2D, GLU, causal depthwise conv, LayerNorm, Swish, pointwise D -> D
        x += 0.5 * FFN(LN(x))
    LayerNorm acts on the feature axis only, never across frames, so causality is preserved."""

    def __init__(self, d_model, n_heads, ff_mult, window, lookahead, conv_kernel, **kw):
        super().__init__(**kw)
        self.ln_ffa, self.ln_att, self.ln_conv, self.ln_ffb = (keras.layers.LayerNormalization() for _ in range(4))
        self.ffa1 = keras.layers.Dense(d_model * ff_mult, activation="silu")
        self.ffa2 = keras.layers.Dense(d_model)
        self.attn = BandedSelfAttention(d_model, n_heads, window, lookahead)
        self.pw1 = keras.layers.Dense(2 * d_model)
        self.dw = CausalDepthwiseConv(conv_kernel)
        self.dw_ln = keras.layers.LayerNormalization()
        self.pw2 = keras.layers.Dense(d_model)
        self.ffb1 = keras.layers.Dense(d_model * ff_mult, activation="silu")
        self.ffb2 = keras.layers.Dense(d_model)

    def call(self, x):
        x = x + 0.5 * self.ffa2(self.ffa1(self.ln_ffa(x)))
        x = x + self.attn(self.ln_att(x))
        a, g = ops.split(self.pw1(self.ln_conv(x)), 2, axis=-1)  # GLU
        x = x + self.pw2(ops.silu(self.dw_ln(self.dw(a * ops.sigmoid(g)))))
        return x + 0.5 * self.ffb2(self.ffb1(self.ln_ffb(x)))
