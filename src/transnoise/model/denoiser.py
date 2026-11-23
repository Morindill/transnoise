"""降噪模型：波形 → STFT → 特征 → 编码 → 滑动窗口 Transformer → ERB 增益 + 深度滤波 → 逆 STFT → 波形。

enhance_frames 只处理频谱帧；call 是整段离线前向 = Analysis + enhance_frames + Synthesis。
"""

import keras
from keras import ops

from .attention import TransformerBlock
from .config import ModelConfig
from .frontend import Analysis, Synthesis, erb_matrices
from .heads import DeepFilterHead, ErbGainHead, deep_filter


class Denoiser(keras.Model):
    def __init__(self, cfg: ModelConfig, **kw):
        super().__init__(**kw)
        self.cfg = cfg
        self.analysis, self.synthesis = Analysis(cfg), Synthesis(cfg)
        self.erb_fwd, self.erb_inv = erb_matrices(cfg)
        d = cfg.d_model
        self.erb_in = keras.layers.Dense(d, activation="gelu")
        self.spec_in = keras.layers.Dense(d, activation="gelu")
        self.mix = keras.layers.Dense(d)
        # 前视只放在第一层（和深度滤波）：后面的层严格因果，总前视 = cfg.lookahead，不随层数累加
        self.blocks = [TransformerBlock(d, cfg.n_heads, cfg.ff_mult, cfg.window, cfg.lookahead if i == 0 else 0)
                       for i in range(cfg.n_layers)]
        self.norm_out = keras.layers.LayerNormalization()
        self.gain_head = ErbGainHead(cfg)
        self.df_head = DeepFilterHead(cfg) if cfg.df_order > 0 else None

    def features(self, re, im):
        """逐帧特征，不依赖其他帧（所以流式和离线一致）：
        ERB 特征 = 各频带平均功率的 dB 值做固定仿射；复数特征 = 最低 nb_df 个频点除以本帧平均幅度（与音量无关）。"""
        c = self.cfg
        erb = ops.matmul(re * re + im * im, self.erb_fwd)
        feat_erb = (10.0 * ops.log10(erb + 1e-10) + c.erb_db_offset) / c.erb_db_scale
        lr, li = re[..., :c.nb_df], im[..., :c.nb_df]
        mag = ops.mean(ops.sqrt(lr * lr + li * li + 1e-12), axis=-1, keepdims=True)
        return feat_erb, ops.concatenate([lr / (mag + 1e-6), li / (mag + 1e-6)], axis=-1)

    def embed(self, feat_erb, feat_spec):
        h = self.mix(ops.concatenate([self.erb_in(feat_erb), self.spec_in(feat_spec)], axis=-1))
        for blk in self.blocks:
            h = blk(h)
        return self.norm_out(h)

    def enhance_frames(self, re, im):
        """带噪谱帧 [B, F, bins] → 增强谱帧。"""
        c = self.cfg
        h = self.embed(*self.features(re, im))
        mask = ops.matmul(self.gain_head(h), self.erb_inv)  # [B, F, bins]
        out_r, out_i = re * mask, im * mask
        if self.df_head is not None:  # 低 nb_df 个频点：深度滤波（滤波对象是带噪谱）的输出再乘 ERB 增益，增益是 0–1 的门控
            dr, di = deep_filter(re[..., :c.nb_df], im[..., :c.nb_df], self.df_head(h), c.df_order, c.lookahead)
            gate = mask[..., :c.nb_df]
            out_r = ops.concatenate([dr * gate, out_r[..., c.nb_df:]], axis=-1)
            out_i = ops.concatenate([di * gate, out_i[..., c.nb_df:]], axis=-1)
        return out_r, out_i

    def call(self, x):
        """x: [B, T] 48kHz 波形（T 静态已知）→ 增强后的 [B, T]，与输入时间对齐（离线，没有流式延迟）。"""
        return self.synthesis(self.enhance_frames(*self.analysis(x)), length=x.shape[-1])
