"""训练数据：在线混合。不依赖 TensorFlow/Keras（数据进程只 import numpy 和 soundfile）。

为什么在线做、不放进 data_pipeline：data_pipeline 只产出"干净的原子素材"（48kHz 单声道 FLAC + index），
而 语音 × 噪声 × RIR × SNR × 电平 的组合是无穷的，预先混好既占几百 GB，又把组合固定死，丢掉了随机增强。
混合运算本身（SNR 缩放、混响、防削波）在 transnoise/mixing.py，评测的固定测试集用的是同一份。

可复现：第 step 个 batch 完全由 (seed, step) 决定。所以断点续训只需要记住"已训练到第几步"，不用存索引或采样器状态。
"""

import json
import multiprocessing as mp
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf

from ..mixing import mix, reverberate
from ..paths import DATA, INDEX, SR
from .config import TrainConfig

MIN_SPEECH_RMS = 10 ** (-60 / 20)  # 裁出来的语音片段低于 -60 dBFS 视为静音，换一段
MIN_NOISE_RMS = 1e-4


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x)))


class Index:
    """index/<split>.jsonl 压成 numpy 数组（路径存成 bytes）。40 万行用 Python 对象会很占内存，每个数据进程还各复制一份。"""

    def __init__(self, split: str):
        rows = [json.loads(line) for line in (INDEX / f"{split}.jsonl").open()]
        speech = [r for r in rows if r["role"] == "speech"]
        self.speech_path = np.array([r["path"] for r in speech], dtype="S")
        self.speech_bw = np.array([r["bandwidth_hz"] or SR / 2 for r in speech], dtype="float32")
        # 噪声：所有来源并成一个集合，按文件时长加权抽文件（等价于在全部噪声音频的时间轴上均匀抽），来源不参与抽样
        noise = [r for r in rows if r["role"] == "noise"]
        self.noise_path = np.array([r["path"] for r in noise], dtype="S")
        dur = np.array([r["duration"] for r in noise], dtype="float64")
        self.noise_cdf = np.cumsum(dur) / dur.sum() if len(noise) else dur
        # RIR 按来源分组：先均匀选来源再选文件（各来源的文件数悬殊，直接按文件抽会被大来源淹没）
        self.rir = [np.array([r["path"] for r in rows if r["role"] == "rir" and r["source"] == s], dtype="S")
                    for s in sorted({r["source"] for r in rows if r["role"] == "rir"})]
        assert len(speech) and len(noise) and self.rir, f"{split}: 缺语音/噪声/RIR"


class BatchSource:
    def __init__(self, split: str, cfg: TrainConfig, seed: int | None = None):
        self.cfg, self.idx = cfg, Index(split)
        self.seed = cfg.seed if seed is None else seed

    # ---- 单个素材 ----
    def _speech(self, rng, n: int) -> tuple[np.ndarray, float]:
        for _ in range(5):
            i = int(rng.integers(len(self.idx.speech_path)))
            with sf.SoundFile(DATA / self.idx.speech_path[i].decode()) as f:
                if len(f) >= n:
                    f.seek(int(rng.integers(len(f) - n + 1)))
                    x = f.read(n, dtype="float32")
                else:  # 比片段短：放在随机位置，其余是静音（训练里也需要"只有噪声"的段落）
                    short = f.read(dtype="float32")
                    x = np.zeros(n, "float32")
                    o = int(rng.integers(n - len(short) + 1))
                    x[o: o + len(short)] = short
            if _rms(x) > MIN_SPEECH_RMS:
                break
        return x, float(self.idx.speech_bw[i])

    def _noise(self, rng, n: int) -> np.ndarray:
        for _ in range(10):
            i = min(int(np.searchsorted(self.idx.noise_cdf, rng.random(), side="right")), len(self.idx.noise_path) - 1)
            with sf.SoundFile(DATA / self.idx.noise_path[i].decode()) as f:
                if len(f) > n:
                    f.seek(int(rng.integers(len(f) - n + 1)))
                    x = f.read(n, dtype="float32")
                else:
                    x = np.resize(f.read(dtype="float32"), n)  # 比片段短：循环拼接
            if _rms(x) > MIN_NOISE_RMS:
                break
        return x

    def _rir(self, rng) -> np.ndarray:
        pool = self.idx.rir[int(rng.integers(len(self.idx.rir)))]
        rir, _ = sf.read(DATA / pool[int(rng.integers(len(pool)))].decode(), dtype="float32")
        return rir if rir.ndim == 1 else rir[:, 0]

    # ---- 样本 / batch ----
    def sample(self, rng) -> tuple[np.ndarray, np.ndarray, float, dict]:
        c, n = self.cfg, self.cfg.samples
        clean, bw = self._speech(rng, n)
        reverb = bool(rng.random() < c.p_reverb)
        if reverb:
            clean = reverberate(clean, self._rir(rng))  # 目标 = 带混响的干净语音
        clean *= 10 ** (rng.uniform(c.speech_db_min, c.speech_db_max) / 20) / (_rms(clean) + 1e-8)
        snr = float(rng.uniform(c.snr_min, c.snr_max))
        noisy, clean = mix(clean, self._noise(rng, n), snr)
        return noisy.astype("float32"), clean.astype("float32"), bw, {"snr_db": snr, "reverb": reverb}

    def batch(self, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """第 step 个 batch：(noisy [B,T], clean [B,T], cutoff_hz [B])。只由 (seed, step) 决定。"""
        rng = np.random.default_rng([self.seed, step])
        parts = [self.sample(rng) for _ in range(self.cfg.batch_size)]
        return (np.stack([p[0] for p in parts]), np.stack([p[1] for p in parts]),
                np.array([p[2] for p in parts], dtype="float32"))


_SRC: BatchSource | None = None


def _init(src: BatchSource) -> None:
    global _SRC
    _SRC = src


def _work(step: int):
    return _SRC.batch(step)


class Loader:
    """多进程预取：每个 batch 由一个数据进程整个做完，按 step 顺序依次返回，始终提前准备 prefetch 个。
    start 是起始 step，所以断点续训时从检查点的 step 接着取就行。用 spawn 启动数据进程，避免 fork 出带着 TF/GPU 状态的子进程。"""

    def __init__(self, src: BatchSource, start: int = 0, workers: int = 12, prefetch: int = 8):
        self.pool = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init, initargs=(src,))
        self.next, self.queue = start, deque()
        for _ in range(prefetch):
            self._submit()

    def _submit(self) -> None:
        self.queue.append(self.pool.submit(_work, self.next))
        self.next += 1

    def __iter__(self):
        return self

    def __next__(self):
        fut = self.queue.popleft()
        self._submit()
        return fut.result()

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
