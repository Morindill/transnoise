import numpy as np
import soxr

from ..paths import ROOT, SR
from .config import CFG

try:
    from pesq import pesq
except ImportError:  # PESQ 是可选依赖（需要 C 编译器），缺了就记 NaN
    pesq = None

M = CFG["metrics"]
MS = M["metric_sr"]

# DNSMOS P.835 官方 dnsmos_local.py 的非个性化多项式拟合系数
_POLY = {
    "sig": np.poly1d([-0.08397278, 1.22083953, 0.0052439]),
    "bak": np.poly1d([-0.13166888, 1.60915514, -0.39604546]),
    "ovr": np.poly1d([-0.06766283, 1.11546468, 0.04602535]),
}
_DNSMOS_LEN = 9.01  # 秒，模型固定输入长度
_sess = None


def to16k(x: np.ndarray) -> np.ndarray:
    return soxr.resample(x, SR, MS, quality="HQ")


def estimate_lag(ref: np.ndarray, est: np.ndarray) -> int:
    """est 相对 ref 延迟了多少个样本（正数 = est 更晚）。用互相关，只看前 15 秒。"""
    max_lag = int(M["max_lag_ms"] * SR / 1000)
    n = min(len(ref), len(est), 15 * SR)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    xc = np.fft.irfft(np.fft.rfft(est[:n], nfft) * np.conj(np.fft.rfft(ref[:n], nfft)), nfft)  # xc[k] = Σ est[t+k]·ref[t]
    cand = np.concatenate([xc[nfft - max_lag:], xc[: max_lag + 1]])  # lag = -max_lag … +max_lag
    return int(np.argmax(cand)) - max_lag


def align(est: np.ndarray, lag: int, n: int) -> np.ndarray:
    """把 est 左移 lag 个样本（lag<0 则在前面补零），再截断/补零到长度 n。"""
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

        ort.preload_dlls()  # 从 pip 装的 nvidia-* 包里加载 CUDA/cuDNN
        _sess = ort.InferenceSession(
            str(ROOT / CFG["assets"]["dnsmos_p835"]), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        if _sess.get_providers()[0] != "CUDAExecutionProvider":  # 库缺失时 onnxruntime 会悄悄退回 CPU，这里说出来
            print("警告：DNSMOS 没有用上 GPU，退回 CPU（会慢很多）")
    return _sess


def _dnsmos_windows(x16: np.ndarray) -> np.ndarray:
    """官方切窗：不足 9.01 秒的重复拼接；每秒一个 9.01 秒窗口。返回 (窗口数, 窗口长度)。"""
    n = int(_DNSMOS_LEN * MS)
    x16 = x16.astype(np.float32)
    while len(x16) < n:
        x16 = np.append(x16, x16)
    hops = int(np.floor(len(x16) / MS) - _DNSMOS_LEN) + 1
    segs = (x16[i * MS: int((i + _DNSMOS_LEN) * MS)] for i in range(hops))
    return np.stack([s for s in segs if len(s) == n])  # 和官方一样，丢掉不足一窗的尾巴


def dnsmos_batch(xs: list[np.ndarray], batch: int = 8) -> list[tuple[float, float, float]]:
    """对多条 16kHz 音频算 DNSMOS P.835：所有窗口攒成 batch 在 GPU 上推理；
    每个窗口先做官方多项式映射，再对同一条音频的窗口取平均。batch 不要太大（实测 64 时反而慢 50 倍）。"""
    sess = _dnsmos_session()
    wins = [_dnsmos_windows(x) for x in xs]
    allw = np.concatenate(wins)
    raw = np.concatenate([sess.run(None, {"input_1": allw[i: i + batch]})[0] for i in range(0, len(allw), batch)])
    cal = np.stack([_POLY["sig"](raw[:, 0]), _POLY["bak"](raw[:, 1]), _POLY["ovr"](raw[:, 2])], axis=1)
    ends = np.cumsum([len(w) for w in wins])
    return [tuple(float(v) for v in seg.mean(axis=0)) for seg in np.split(cal, ends[:-1])]


def score_cpu(ref: np.ndarray, est: np.ndarray) -> tuple[dict, np.ndarray]:
    """CPU 上能并行的指标。ref、est 都是 48kHz 单声道、等长、已对齐。
    返回 (si_sdr/pesq/stoi, est 的 16kHz 版本)——后者交给 dnsmos_batch。"""
    from pystoi import stoi

    out = {"si_sdr": si_sdr(ref, est)}
    r16, e16 = to16k(ref), to16k(est)
    try:
        out["pesq"] = float(pesq(MS, r16, e16, "wb")) if pesq else float("nan")
    except Exception:  # 例如输出全静音时 PESQ 找不到语音
        out["pesq"] = float("nan")
    out["stoi"] = float(stoi(r16, e16, MS))
    return out, e16
