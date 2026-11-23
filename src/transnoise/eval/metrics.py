import numpy as np
import soxr

from ..paths import ROOT, SR
from .config import CFG

try:
    from pesq import pesq
except ImportError:  # PESQ is an optional dependency (needs a C compiler); without it PESQ is recorded as NaN
    pesq = None

M = CFG["metrics"]
MS = M["metric_sr"]

# Non-personalized polynomial fit coefficients from the official DNSMOS P.835 dnsmos_local.py
_POLY = {
    "sig": np.poly1d([-0.08397278, 1.22083953, 0.0052439]),
    "bak": np.poly1d([-0.13166888, 1.60915514, -0.39604546]),
    "ovr": np.poly1d([-0.06766283, 1.11546468, 0.04602535]),
}
_DNSMOS_LEN = 9.01  # seconds; fixed input length of the model
_sess = None


def to16k(x: np.ndarray) -> np.ndarray:
    return soxr.resample(x, SR, MS, quality="HQ")


def estimate_lag(ref: np.ndarray, est: np.ndarray) -> int:
    """How many samples est lags behind ref (positive = est is later). Cross-correlation over the first 15 s only."""
    max_lag = int(M["max_lag_ms"] * SR / 1000)
    n = min(len(ref), len(est), 15 * SR)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    xc = np.fft.irfft(np.fft.rfft(est[:n], nfft) * np.conj(np.fft.rfft(ref[:n], nfft)), nfft)  # xc[k] = sum_t est[t+k]*ref[t]
    cand = np.concatenate([xc[nfft - max_lag:], xc[: max_lag + 1]])  # lag = -max_lag ... +max_lag
    return int(np.argmax(cand)) - max_lag


def align(est: np.ndarray, lag: int, n: int) -> np.ndarray:
    """Shift est left by lag samples (zero-pad the front if lag < 0), then truncate / zero-pad to length n."""
    est = est[lag:] if lag > 0 else np.concatenate([np.zeros(-lag, est.dtype), est])
    est = est[:n]
    return np.pad(est, (0, n - len(est)))


def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    ref, est = ref.astype(np.float64), est.astype(np.float64)
    ref, est = ref - ref.mean(), est - est.mean()
    t = (est @ ref) / (ref @ ref + 1e-12) * ref
    e = est - t
    return float(10 * np.log10((t @ t + 1e-12) / (e @ e + 1e-12)))


def _dnsmos_session():
    global _sess
    if _sess is None:
        import onnxruntime as ort

        ort.preload_dlls()  # load CUDA/cuDNN from the pip-installed nvidia-* packages
        _sess = ort.InferenceSession(
            str(ROOT / CFG["assets"]["dnsmos_p835"]), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        if _sess.get_providers()[0] != "CUDAExecutionProvider":  # onnxruntime silently falls back to CPU when the libraries are missing; say so
            print("警告：DNSMOS 没有用上 GPU，退回 CPU（会慢很多）")
    return _sess


def _dnsmos_windows(x16: np.ndarray) -> np.ndarray:
    """Official windowing: audio shorter than 9.01 s is repeated; one 9.01 s window per second. Returns (n_windows, window_length)."""
    n = int(_DNSMOS_LEN * MS)
    x16 = x16.astype(np.float32)
    while len(x16) < n:
        x16 = np.append(x16, x16)
    hops = int(np.floor(len(x16) / MS) - _DNSMOS_LEN) + 1
    segs = (x16[i * MS: int((i + _DNSMOS_LEN) * MS)] for i in range(hops))
    return np.stack([s for s in segs if len(s) == n])  # like the official code, drop the tail shorter than one window


def dnsmos_batch(xs: list[np.ndarray], batch: int = 8) -> list[tuple[float, float, float]]:
    """DNSMOS P.835 for several 16 kHz clips: all windows are batched for GPU inference; each window goes through the
    official polynomial mapping, then the windows of one clip are averaged. Keep the batch small (64 was measured 50x slower)."""
    sess = _dnsmos_session()
    wins = [_dnsmos_windows(x) for x in xs]
    allw = np.concatenate(wins)
    raw = np.concatenate([sess.run(None, {"input_1": allw[i: i + batch]})[0] for i in range(0, len(allw), batch)])
    cal = np.stack([_POLY["sig"](raw[:, 0]), _POLY["bak"](raw[:, 1]), _POLY["ovr"](raw[:, 2])], axis=1)
    ends = np.cumsum([len(w) for w in wins])
    return [tuple(float(v) for v in seg.mean(axis=0)) for seg in np.split(cal, ends[:-1])]


def score_cpu(ref: np.ndarray, est: np.ndarray) -> tuple[dict, np.ndarray]:
    """Metrics that can run in parallel on CPU. ref and est are 48 kHz mono, equal length, already aligned.
    Returns (si_sdr/pesq/stoi, 16 kHz version of est); the latter goes to dnsmos_batch."""
    from pystoi import stoi

    out = {"si_sdr": si_sdr(ref, est)}
    r16, e16 = to16k(ref), to16k(est)
    try:
        out["pesq"] = float(pesq(MS, r16, e16, "wb")) if pesq else float("nan")
    except Exception:  # e.g. PESQ finds no speech when the output is all silence
        out["pesq"] = float("nan")
    out["stoi"] = float(stoi(r16, e16, MS))
    return out, e16
