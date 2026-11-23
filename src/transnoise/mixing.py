"""Noisy-speech synthesis shared by training (online mixing) and evaluation (fixed test sets), so both sides use the same
SNR, reverb and anti-clipping conventions. Depends on numpy only."""

import numpy as np


def fftconvolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    n = len(x) + len(h) - 1
    nfft = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(h, nfft), nfft)[:n]


def reverberate(x: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve speech with an RIR (peak-normalized first), truncated to the original length. The reference signal is this
    reverberant clean speech (only denoising is evaluated, not dereverberation)."""
    return fftconvolve(x, rir / np.abs(rir).max())[: len(x)].astype(np.float32)


def mix(clean: np.ndarray, noise: np.ndarray, snr_db) -> tuple[np.ndarray, np.ndarray]:
    """Scale the noise to the given SNR (over the whole-segment power) and add it. Returns (noisy, clean); both get the same
    anti-clipping scale (peak <= 0.95), so the SNR is unchanged."""
    g = np.sqrt((clean @ clean) / (noise @ noise) / 10 ** (snr_db / 10))
    noisy = clean + g * noise
    peak = max(np.abs(noisy).max(), np.abs(clean).max())
    scale = 0.95 / peak if peak > 0.95 else 1.0
    return noisy * scale, clean * scale
