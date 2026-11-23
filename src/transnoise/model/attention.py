"""滑动窗口自注意力 + Transformer 块。

每一帧 i 只看 [i-window, i+lookahead] 这些帧；位置信息用"相对偏移"的可学习偏置，不用绝对位置，
所以训练长度和部署长度可以不同，流式时也不需要知道已经处理了多少帧。
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
        # 每个头、每个相对偏移（-window … +lookahead）一个标量偏置
        self.rel_bias = self.add_weight(name="rel_bias", shape=(self.h, self.window + self.lookahead + 1),
                                        initializer="zeros")

    def call(self, x):  # x: [B, F, D]
        f = ops.shape(x)[1]
        q, k, v = ops.split(self.qkv(x), 3, axis=-1)
        q, k, v = (ops.transpose(ops.reshape(t, (-1, f, self.h, self.d // self.h)), (0, 2, 1, 3)) for t in (q, k, v))
        scores = ops.einsum("bhid,bhjd->bhij", q, k) / (self.d // self.h) ** 0.5
        pos = ops.arange(f)
        off = pos[None, :] - pos[:, None]  # [F, F]，off[i, j] = j - i
        allowed = (off >= -self.window) & (off <= self.lookahead)
        bias = ops.take(self.rel_bias, ops.clip(off + self.window, 0, self.window + self.lookahead), axis=1)  # [H,F,F]
        scores = ops.where(allowed[None, None], scores + bias[None], -1e9)
        ctx = ops.einsum("bhij,bhjd->bhid", ops.softmax(scores, axis=-1), v)
        return self.out(ops.reshape(ops.transpose(ctx, (0, 2, 1, 3)), (-1, f, self.d)))


class TransformerBlock(keras.layers.Layer):
    """Pre-LN：x + Attn(LN(x))，再 x + FFN(LN(x))。LayerNorm 只在特征维上做，不跨帧，所以不破坏因果性。"""

    def __init__(self, d_model, n_heads, ff_mult, window, lookahead, **kw):
        super().__init__(**kw)
        self.ln1, self.ln2 = keras.layers.LayerNormalization(), keras.layers.LayerNormalization()
        self.attn = BandedSelfAttention(d_model, n_heads, window, lookahead)
        self.ff1 = keras.layers.Dense(d_model * ff_mult, activation="gelu")
        self.ff2 = keras.layers.Dense(d_model)

    def call(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff2(self.ff1(self.ln2(x)))
