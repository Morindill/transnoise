"""The fixed evaluation interface: input = a test set + a directory of enhanced wavs, output = results.jsonl (one row of scores per clip).

The evaluation neither knows nor cares which model produced the wavs or how. Requirement: the directory has an <id>.wav
for every clip of the test set, 48 kHz mono. Everything is recomputed and the results overwritten on every run.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import soundfile as sf

from ..paths import EVAL, SR
from . import metrics

KEEP = ("id", "lang", "snr_db", "reverb", "speech_source", "noise_source", "bw_tier", "speaker", "duration")


def _score_one(args: tuple) -> dict:
    """Worker: read clean and the enhanced result -> estimate the delay -> align -> score."""
    tdir, enhanced, row = args
    clean = sf.read(tdir / "clean" / f"{row['id']}.wav", dtype="float32")[0]
    est, sr = sf.read(enhanced / f"{row['id']}.wav", dtype="float32")
    if sr != SR or est.ndim != 1:
        raise ValueError(f"{enhanced / (row['id'] + '.wav')}: 需要 {SR}Hz 单声道，实际 sr={sr} 形状={est.shape}")
    lag = metrics.estimate_lag(clean, est)
    out = {k: row[k] for k in KEEP} | {"delay_samples": lag}
    cpu, e16 = metrics.score_cpu(clean, metrics.align(est, lag, len(clean)))
    return out | cpu, e16


def score(testset: str, enhanced: Path, name: str | None = None, workers: int | None = None) -> None:
    tdir = EVAL / "testsets" / testset
    if not tdir.exists():
        raise SystemExit(f"测试集 {tdir} 不存在，先 eval build-testset")
    enhanced = enhanced.resolve()
    meta = [json.loads(line) for line in (tdir / "meta.jsonl").open()]
    missing = [m["id"] for m in meta if not (enhanced / f"{m['id']}.wav").exists()]
    if missing:
        raise SystemExit(f"{enhanced} 缺 {len(missing)}/{len(meta)} 个文件，例如 {missing[0]}.wav")
    name = name or (enhanced.parent.name if enhanced.name == "wav" else enhanced.name)
    workers = workers or max(1, min(26, (os.cpu_count() or 4) - 2))
    if metrics.pesq is None:
        print("警告：没有安装 pesq，PESQ 这一列会是 NaN。安装：uv sync（pesq 在默认依赖组里，需要 C 编译器）")

    out = EVAL / "runs" / testset / name / "results.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with ProcessPoolExecutor(workers) as pool:  # CPU metrics run in parallel processes
        parts = list(pool.map(_score_one, [(tdir, enhanced, m) for m in meta], chunksize=4))
    mos = metrics.dnsmos_batch([e16 for _, e16 in parts])  # DNSMOS is batched on the GPU in the main process
    with tmp.open("w") as f:
        for (row, _), (sig, bak, ovr) in zip(parts, mos):
            row = row | {"dnsmos_sig": sig, "dnsmos_bak": bak, "dnsmos_ovr": ovr}
            f.write(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}) + "\n")
    tmp.replace(out)  # replace only once everything is computed, so a failure never leaves half a result
    print(f"[{testset}/{name}] 已打分 {len(meta)} 条 → {out}")
