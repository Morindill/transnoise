"""带噪语音的合成运算，训练（在线混合）和评测（固定测试集）共用同一份，保证两边的 SNR、混响、防削波口径一致。
只依赖 numpy（有 scipy 时卷积用 scipy，保证和已经生成的测试集逐位一致；没有时用 numpy 的 FFT 卷积）。"""

import numpy as np

try:
    from scipy.signal import fftconvolve
except ImportError:  # 训练环境没装 scipy
    def fftconvolve(x, h):
        n = len(x) + len(h) - 1
        nfft = 1 << (n - 1).bit_length()
        return np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(h, nfft), nfft)[:n]


def reverberate(x: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """语音和 RIR 卷积（RIR 先按峰值归一化），截到原长度。参考信号 = 这个带混响的干净语音（只评估去噪，不去混响）。"""
    return fftconvolve(x, rir / np.abs(rir).max())[: len(x)].astype(np.float32)


def mix(clean: np.ndarray, noise: np.ndarray, snr_db) -> tuple[np.ndarray, np.ndarray]:
    """按整段功率把噪声缩放到指定 SNR 后相加。返回 (带噪, 干净)，两者用同一个缩放防削波（峰值 ≤ 0.95），SNR 不变。"""
    g = np.sqrt((clean @ clean) / (noise @ noise) / 10 ** (snr_db / 10))
    noisy = clean + g * noise
    peak = max(np.abs(noisy).max(), np.abs(clean).max())
    scale = 0.95 / peak if peak > 0.95 else 1.0
    return noisy * scale, clean * scale
