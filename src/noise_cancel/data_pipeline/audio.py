import hashlib
import io

import numpy as np
import soundfile as sf
import soxr

from ..paths import DATA, SR, STAGED
from .config import CFG

BW = CFG["bandwidth"]
CLEAN = CFG["clean"]


def bandwidth_hz(x: np.ndarray, sr: int) -> float:
    """谱滚降：活跃帧平均功率谱中，高于 (200-3000Hz 均值 + floor_db) 的最高频率。"""
    n = 2048
    x = x[: sr * 60]
    hop = n // 2
    nfr = (len(x) - n) // hop + 1
    frames = x[np.arange(n)[None, :] + hop * np.arange(nfr)[:, None]] * np.hanning(n).astype(np.float32)
    p = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    e = p.sum(1)
    psd = p[e >= np.median(e)].mean(0)
    psd = np.convolve(psd, np.ones(31) / 31, mode="same")  # 约 ±350Hz 平滑，抑制孤立音调
    db = 10 * np.log10(psd + 1e-20)
    freqs = np.arange(n // 2 + 1) * sr / n
    ref = db[(freqs >= 200) & (freqs <= 3000)].mean()
    return float(freqs[np.nonzero(db > ref + BW["floor_db"])[0].max()])


def tier(bw: float) -> str:
    return "A" if bw >= BW["tier_a_hz"] else "B" if bw >= BW["tier_b_hz"] else "C"


def make_id(source: str, member: str) -> str:
    return hashlib.sha1(f"{source}:{member}".encode()).hexdigest()[:16]


def process(task: dict) -> dict:
    """worker：解码 → 单声道 → 质量统计 → 重采样 48k → 写 FLAC。返回 manifest 行。"""
    role = task["role"]
    if "pcm" in task:
        mono, sr = task["pcm"].astype(np.float32) / 32768.0, SR
    else:
        x, sr = sf.read(io.BytesIO(task["data"]), dtype="float32", always_2d=True)
        mono = x.mean(1)

    dur = len(mono) / sr
    row = {
        "id": task["id"], "source": task["source"], "member": task["member"],
        "role": role, "lang": task["lang"], "speaker": task["speaker"], "split": task["split"],
        "duration": round(dur, 3), "orig_sr": sr,
        "path": None, "bandwidth_hz": None, "bw_tier": None, "rms_db": None,
        "clip_ratio": None, "keep": False, "drop_reason": None,
    }
    if dur < CLEAN["rir_min_duration" if role == "rir" else "min_duration"]:
        row["drop_reason"] = "too_short"
        return row

    row["rms_db"] = round(float(20 * np.log10(np.sqrt(np.mean(mono**2)) + 1e-9)), 2)
    row["clip_ratio"] = round(float(np.mean(np.abs(mono) >= 0.999)), 5)
    if role != "rir":
        row["bandwidth_hz"] = round(bandwidth_hz(mono, sr))
        row["bw_tier"] = tier(row["bandwidth_hz"])
        if row["rms_db"] < CLEAN["min_rms_db"][role]:
            row["drop_reason"] = "silent"
        elif role == "speech" and row["clip_ratio"] > CLEAN["speech_max_clip_ratio"]:
            row["drop_reason"] = "clipping"
        if row["drop_reason"]:
            return row

    if sr != SR:
        mono = soxr.resample(mono, sr, SR, quality="VHQ")
    out = STAGED / task["source"] / task["id"][:2] / (task["id"] + (".wav" if role == "rir" else ".flac"))
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, mono, SR, subtype="FLOAT" if role == "rir" else "PCM_16")
    row["path"] = out.relative_to(DATA).as_posix()
    row["keep"] = True
    return row
