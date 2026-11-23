import json
import random
import shutil

import numpy as np
import soundfile as sf

from ..mixing import mix, reverberate
from ..paths import DATA, EVAL, INDEX, SR
from .config import CFG

MIN_NOISE_RMS = 1e-4  # 低于它的噪声片段视为静音，换一段


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    return json.dumps(v) if isinstance(v, str) else repr(v)


def _by_source(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in sorted(rows, key=lambda r: r["id"]):  # 先排序，保证结果只取决于种子
        out.setdefault(r["source"], []).append(r)
    return out


def _pick_speech(pools: dict[str, list[dict]], n: int, rng: random.Random) -> list[dict]:
    """在各来源之间轮流抽，每个来源的数量接近；再打乱顺序。"""
    pools = {s: rng.sample(p, len(p)) for s, p in pools.items()}
    picked, sources, i = [], sorted(pools), 0
    while len(picked) < n and any(pools.values()):
        src = sources[i % len(sources)]
        if pools[src]:
            picked.append(pools[src].pop())
        i += 1
    rng.shuffle(picked)
    return picked


def _noise_segment(pools: dict[str, list[dict]], n: int, rng: random.Random) -> tuple[np.ndarray, dict]:
    for _ in range(20):
        row = rng.choice(pools[rng.choice(sorted(pools))])
        path = DATA / row["path"]
        frames = sf.info(path).frames
        off = rng.randrange(frames - n + 1) if frames > n else 0
        x, _ = sf.read(path, start=off, frames=min(n, frames), dtype="float32")
        if len(x) < n:
            x = np.resize(x, n)  # 噪声比语音短：循环拼接
        if np.sqrt(np.mean(x**2)) > MIN_NOISE_RMS:
            return x, {"noise_id": row["id"], "noise_source": row["source"], "noise_offset": off}
    raise RuntimeError("连续 20 次都抽到静音噪声，检查噪声数据")


def build(name: str, n_per_lang: int | None = None, out_name: str | None = None, force: bool = False) -> None:
    spec = dict(CFG["testsets"][name])
    if n_per_lang:
        spec["n_per_lang"] = n_per_lang
    out = EVAL / "testsets" / (out_name or name)
    if out.exists():
        if not force:
            raise SystemExit(f"{out} 已存在；测试集生成后不应改变，要重建请加 --force")
        shutil.rmtree(out)
        # 测试集一变，旧的评测结果和增强后的 wav 都对不上了（文件名 id 相同、内容不同，重打分会悄悄算出错的分数），一并删掉
        runs, reports = EVAL / "runs" / out.name, [EVAL / "reports" / f"{out.name}{s}" for s in (".md", ".csv")]
        stale = ([runs] if runs.exists() else []) + [p for p in reports if p.exists()]
        for p in stale:
            shutil.rmtree(p) if p.is_dir() else p.unlink()
        if stale:
            print("已删除过期的评测产物（需要重新推理并打分）：" + "，".join(str(p.relative_to(EVAL)) for p in stale))
    (out / "noisy").mkdir(parents=True)
    (out / "clean").mkdir()

    rows = [json.loads(line) for line in (INDEX / "test.jsonl").open()]
    noise_pools = _by_source([r for r in rows if r["role"] == "noise"])
    rir_pools = _by_source([r for r in rows if r["role"] == "rir"]) if spec["reverb"] else {}
    rng = random.Random(spec["seed"])
    meta = []

    for lang in sorted({r["lang"] for r in rows if r["role"] == "speech"}):
        cand = [r for r in rows if r["role"] == "speech" and r["lang"] == lang
                and spec["min_duration"] <= r["duration"] <= spec["max_duration"]]
        speech = _pick_speech(_by_source(cand), spec["n_per_lang"], rng)
        for k, s in enumerate(speech):
            x, sr = sf.read(DATA / s["path"], dtype="float32")
            assert sr == SR
            info = {"id": f"{lang}_{k:04d}", "lang": lang, "speech_id": s["id"], "speech_source": s["source"],
                    "speaker": s["speaker"], "bw_tier": s["bw_tier"], "reverb": spec["reverb"]}
            if spec["reverb"]:
                rsrc = rng.choice(sorted(rir_pools))
                rr = rng.choice(rir_pools[rsrc])
                rir, _ = sf.read(DATA / rr["path"], dtype="float32")
                x = reverberate(x, rir)  # 参考信号 = 带混响的干净语音
                info |= {"rir_id": rr["id"], "rir_source": rsrc, "rir_room": rr["speaker"]}
            noise, ninfo = _noise_segment(noise_pools, len(x), rng)
            snr = spec["snrs"][k % len(spec["snrs"])]
            noisy, x = mix(x, noise, snr)
            sf.write(out / "noisy" / f"{info['id']}.wav", noisy, SR, subtype="PCM_16")
            sf.write(out / "clean" / f"{info['id']}.wav", x, SR, subtype="PCM_16")
            meta.append(info | ninfo | {"snr_db": snr, "duration": round(len(x) / SR, 3)})

    with (out / "meta.jsonl").open("w") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    with (out / "spec.toml").open("w") as f:  # 生成参数的副本，用来复现
        f.write(f"# 由 configs/eval.toml 的 [testsets.{name}] 生成\n")
        for k, v in (spec | {"n_total": len(meta)}).items():
            f.write(f"{k} = {_toml_value(v)}\n")
    print(f"{out}: {len(meta)} 条，" + "，".join(f"{lang} {sum(m['lang'] == lang for m in meta)}" for lang in sorted({m['lang'] for m in meta})))
