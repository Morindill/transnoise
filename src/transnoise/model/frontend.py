"""STFT front and back end plus ERB banding. No trainable parameters.

Frame convention (shared by streaming and offline): fft-hop zeros are prepended and fft zeros appended (then padded to a
multiple of hop); frame t covers [t*hop, t*hop+fft) of the padded signal, i.e. [(t-1)*hop, (t+1)*hop) of the original waveform.
Synthesis uses the same Vorbis window (which satisfies Princen-Bradley); after overlap-add the first fft-hop samples are
dropped, so analysis -> synthesis reconstructs perfectly.
"""

import keras
import numpy as np
from keras import ops

from .config import ModelConfig


def vorbis_window(n: int) -> np.ndarray:
    k = np.arange(n)
    return np.sin(np.pi / 2 * np.sin(np.pi * (k + 0.5) / n) ** 2).astype("float32")


def _erb(f):  # Hz -> ERB scale
    return 9.265 * np.log(1 + f / (24.7 * 9.265))


def _inv_erb(e):
    return 24.7 * 9.265 * (np.exp(e / 9.265) - 1)


def erb_widths(cfg: ModelConfig) -> list[int]:
    """Number of STFT bins in each ERB band: bands are equally spaced on the ERB scale, at least min_erb_bins bins each, summing to bins."""
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
    """forward [bins, nb_erb]: average the power spectrum per band; inverse [nb_erb, bins]: copy each band gain back to its bins."""
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
    """Waveform [B, T] -> spectrum (real, imag), each [B, F, bins], F = T/hop + 2. T must have a statically known length."""

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
    """Spectrum (real, imag) -> waveform [B, length]."""

    def __init__(self, cfg: ModelConfig, **kw):
        super().__init__(**kw)
        self.cfg, self.win = cfg, vorbis_window(cfg.fft)

    def call(self, spec, length: int):
        c = self.cfg
        y = ops.istft(spec, c.fft, c.hop, c.fft, window=self.win, center=False)
        return y[:, c.fft - c.hop: c.fft - c.hop + length]
