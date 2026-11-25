"""Multi-resolution spectral loss (after DFN3) + bandwidth mask. Uses keras.ops only.

At each FFT size: one MSE on the compressed magnitude |X|^gamma and one on the compressed complex spectrum X*|X|^(gamma-1) (phase kept).
Bandwidth mask: each sample carries a cutoff frequency (the measured bandwidth of the clean speech); bins above it do not count in the loss.
Corpora with bandwidth below 16 kHz have no real content at high frequencies, and the model must not "learn" the silence there.
Each sample is normalized by its number of valid bins, so narrow-band samples are not diluted.
"""

import numpy as np
from keras import ops

from .config import TrainConfig


def _masked_mean(x, mask):
    """x: [B, F, K], mask: [B, K] (0/1) -> per-sample mean over the valid bins, then averaged over the batch."""
    num = ops.sum(x * mask[:, None, :], axis=(1, 2))
    den = ops.sum(mask, axis=1) * ops.cast(ops.shape(x)[1], x.dtype)
    return ops.mean(num / ops.maximum(den, 1.0))


class SpectralLoss:
    def __init__(self, cfg: TrainConfig, sr: int):
        self.cfg, self.sr = cfg, sr
        # bin frequencies [K] at each resolution, compared against the cutoff frequency
        self.freqs = {n: np.arange(n // 2 + 1, dtype="float32") * sr / n for n in cfg.fft_sizes}

    def _spec(self, x, n):
        return ops.stft(x, n, n // 4, n, window="hann", center=True)

    def __call__(self, clean, pred, cutoff_hz):
        """clean/pred: [B, T]; cutoff_hz: [B]. Returns (total loss, {magnitude term, complex term})."""
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
