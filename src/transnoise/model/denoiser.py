"""Denoising model: waveform -> STFT -> features -> encoder -> sliding-window Transformer -> ERB gains + deep filter -> inverse STFT -> waveform.

enhance_frames only processes spectral frames; call is the whole-clip offline forward = Analysis + enhance_frames + Synthesis.
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
        # Lookahead lives only in the first layer (and the deep filter): later layers are strictly causal, so the total
        # lookahead = cfg.lookahead and does not accumulate with depth
        self.blocks = [TransformerBlock(d, cfg.n_heads, cfg.ff_mult, cfg.window, cfg.lookahead if i == 0 else 0)
                       for i in range(cfg.n_layers)]
        self.norm_out = keras.layers.LayerNormalization()
        self.gain_head = ErbGainHead(cfg)
        self.df_head = DeepFilterHead(cfg)

    def features(self, re, im):
        """Per-frame features that do not depend on other frames (so streaming and offline agree):
        ERB feature = a fixed affine map of the per-band mean power in dB; complex feature = the lowest nb_df bins divided by this frame's mean magnitude (independent of volume)."""
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
        """Noisy spectral frames [B, F, bins] -> enhanced spectral frames."""
        c = self.cfg
        h = self.embed(*self.features(re, im))
        mask = ops.matmul(self.gain_head(h), self.erb_inv)  # [B, F, bins]
        # Lowest nb_df bins: the deep-filter output (the filter acts on the noisy spectrum) times the ERB gain, which is a 0-1 gate;
        # the remaining bins are simply multiplied by the gain
        dr, di = deep_filter(re[..., :c.nb_df], im[..., :c.nb_df], self.df_head(h), c.df_order, c.lookahead)
        gate = mask[..., :c.nb_df]
        out_r = ops.concatenate([dr * gate, (re * mask)[..., c.nb_df:]], axis=-1)
        out_i = ops.concatenate([di * gate, (im * mask)[..., c.nb_df:]], axis=-1)
        return out_r, out_i

    def call(self, x):
        """x: [B, T] 48 kHz waveform (T statically known) -> enhanced [B, T], time-aligned with the input (offline, no streaming delay)."""
        return self.synthesis(self.enhance_frames(*self.analysis(x)), length=x.shape[-1])
