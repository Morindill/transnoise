"""Training data: online mixing. Does not depend on TensorFlow/Keras (the data processes import only numpy and soundfile).

Why online and not part of data_pipeline: data_pipeline only produces clean atomic material (48 kHz mono FLAC + index),
while the combinations speech x noise x RIR x SNR x level are endless; pre-mixing them would take hundreds of GB, freeze
the combinations and lose the random augmentation.
The mixing arithmetic itself (SNR scaling, reverb, anti-clipping) lives in transnoise/mixing.py; the fixed evaluation test sets use the same code.

Reproducible: batch number step is fully determined by (seed, step). So resuming only needs to remember how many steps were
trained; no index or sampler state is stored.
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

MIN_SPEECH_RMS = 10 ** (-60 / 20)  # a cropped speech segment below -60 dBFS counts as silence; draw another one
MIN_NOISE_RMS = 1e-4


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x)))


class Index:
    """index/<split>.jsonl packed into numpy arrays (paths stored as bytes). 400k rows as Python objects would take a lot of memory, and every data process holds its own copy."""

    def __init__(self, split: str):
        rows = [json.loads(line) for line in (INDEX / f"{split}.jsonl").open()]
        speech = [r for r in rows if r["role"] == "speech"]
        self.speech_path = np.array([r["path"] for r in speech], dtype="S")
        self.speech_bw = np.array([r["bandwidth_hz"] or SR / 2 for r in speech], dtype="float32")
        # Noise: all sources form one pool, files are drawn weighted by duration (equivalent to sampling uniformly along the
        # time axis of all noise audio); the source takes no part in sampling
        noise = [r for r in rows if r["role"] == "noise"]
        self.noise_path = np.array([r["path"] for r in noise], dtype="S")
        dur = np.array([r["duration"] for r in noise], dtype="float64")
        self.noise_cdf = np.cumsum(dur) / dur.sum() if len(noise) else dur
        # RIRs are grouped by source: pick a source uniformly, then a file (file counts differ wildly across sources; drawing by file would be swamped by the big one)
        self.rir = [np.array([r["path"] for r in rows if r["role"] == "rir" and r["source"] == s], dtype="S")
                    for s in sorted({r["source"] for r in rows if r["role"] == "rir"})]
        assert len(speech) and len(noise) and self.rir, f"{split}: missing speech/noise/RIR"


class BatchSource:
    def __init__(self, split: str, cfg: TrainConfig, seed: int | None = None):
        self.cfg, self.idx = cfg, Index(split)
        self.seed = cfg.seed if seed is None else seed

    # ---- single clips ----
    def _speech(self, rng, n: int) -> tuple[np.ndarray, float]:
        for _ in range(5):
            i = int(rng.integers(len(self.idx.speech_path)))
            with sf.SoundFile(DATA / self.idx.speech_path[i].decode()) as f:
                if len(f) >= n:
                    f.seek(int(rng.integers(len(f) - n + 1)))
                    x = f.read(n, dtype="float32")
                else:  # shorter than the segment: place at a random position, the rest is silence (training also needs noise-only stretches)
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
                    x = np.resize(f.read(dtype="float32"), n)  # shorter than the segment: tile it
            if _rms(x) > MIN_NOISE_RMS:
                break
        return x

    def _rir(self, rng) -> np.ndarray:
        pool = self.idx.rir[int(rng.integers(len(self.idx.rir)))]
        rir, _ = sf.read(DATA / pool[int(rng.integers(len(pool)))].decode(), dtype="float32")
        return rir if rir.ndim == 1 else rir[:, 0]

    # ---- samples / batches ----
    def _kind(self, rng) -> str:
        u, c = rng.random(), self.cfg
        if u < c.p_silence:
            return "silence"
        if u < c.p_silence + c.p_noise_only:
            return "noise_only"
        if u < c.p_silence + c.p_noise_only + c.p_clean:
            return "clean_only"
        return "mixed"

    def sample(self, rng) -> tuple[np.ndarray, np.ndarray, float, dict]:
        """One (noisy, target, cutoff frequency, info). Kinds: mixed = speech + noise; noise_only = noise alone, target all zeros;
        clean_only = clean speech alone, target = input; silence = all zeros. Samples without speech have no bandwidth information, so the cutoff is the full band."""
        c, n = self.cfg, self.cfg.samples
        kind = self._kind(rng)
        if kind == "silence":
            z = np.zeros(n, "float32")
            return z, z, SR / 2, {"kind": kind, "reverb": False}
        level = rng.uniform(c.speech_db_min, c.speech_db_max)  # level of the clean speech; in noise_only the noise level = level - snr
        snr = float(rng.uniform(c.snr_min, c.snr_max))
        if kind == "noise_only":
            noise = self._noise(rng, n)
            noise *= 10 ** ((level - snr) / 20) / (_rms(noise) + 1e-8)
            noise *= min(1.0, 0.95 / np.abs(noise).max())
            return noise, np.zeros(n, "float32"), SR / 2, {"kind": kind, "snr_db": snr, "reverb": False}
        clean, bw = self._speech(rng, n)
        reverb = bool(rng.random() < c.p_reverb)
        if reverb:
            clean = reverberate(clean, self._rir(rng))  # target = reverberant clean speech
        clean *= 10 ** (level / 20) / (_rms(clean) + 1e-8)
        if kind == "clean_only":
            clean *= min(1.0, 0.95 / np.abs(clean).max())
            return clean, clean, bw, {"kind": kind, "reverb": reverb}
        noisy, clean = mix(clean, self._noise(rng, n), snr)
        return noisy.astype("float32"), clean.astype("float32"), bw, {"kind": kind, "snr_db": snr, "reverb": reverb}

    def batch(self, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batch number step: (noisy [B,T], clean [B,T], cutoff_hz [B]). Determined by (seed, step) alone."""
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
    """Multiprocess prefetch: each batch is produced entirely by one data process and returned in step order, always keeping prefetch batches ready.
    start is the starting step, so resuming simply continues from the checkpoint's step. Data processes are started with spawn to avoid forking children that carry TF/GPU state."""

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
