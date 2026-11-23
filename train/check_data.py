"""Training-data self-check (reads the real data/index/train.jsonl, no training): shapes/values, whether SNR falls in the configured range, determinism, resume position, throughput and memory.
Usage: uv run transnoise check data"""

import time


def meminfo() -> float:
    kv = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
    return (kv["MemTotal"] - kv["MemAvailable"]) / 2**20  # GB


def main() -> None:
    import numpy as np

    from transnoise.training.config import TrainConfig
    from transnoise.training.dataset import BatchSource, Loader

    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'通过' if cond else '失败'}] {name} {detail}")

    cfg = TrainConfig.load()
    t0 = time.perf_counter()
    src = BatchSource("train", cfg)
    idx = src.idx
    print(f"索引加载 {time.perf_counter() - t0:.1f}s：语音 {len(idx.speech_path)} 条；噪声 {len(idx.noise_path)} 个文件；RIR {len(idx.rir)} 个来源 {[len(p) for p in idx.rir]}")

    # 1) shapes, values, peak
    noisy, clean, cutoff = src.batch(0)
    check("形状", noisy.shape == clean.shape == (cfg.batch_size, cfg.samples) and cutoff.shape == (cfg.batch_size,),
          f"{noisy.shape}，dtype={noisy.dtype}")
    check("数值有限、峰值 ≤ 0.95", np.isfinite(noisy).all() and np.isfinite(clean).all() and max(np.abs(noisy).max(), np.abs(clean).max()) <= 0.9501,
          f"峰值 {max(np.abs(noisy).max(), np.abs(clean).max()):.3f}")
    check("截止频率取自语料带宽", cutoff.min() >= 4000 and cutoff.max() <= 24000, f"范围 {cutoff.min():.0f}–{cutoff.max():.0f} Hz，"
          f"A/B/C 混合：{np.round(np.percentile(cutoff, [10, 50, 90]))}")

    # 2) sample kinds, SNR and reverb ratio: draw 2000 with sample(). The measured SNR of mixed samples must equal the drawn SNR (peak scaling does not change SNR);
    #    the reverb ratio is close to p_reverb (counted among samples with speech); noise-only targets are all zeros, clean_only target = input, silence all zeros
    rng = np.random.default_rng(7)
    n_total = 2000
    kinds, errs, rev, snrs, levels = {}, [], 0, [], []
    bad_target = 0
    for _ in range(n_total):
        n_, c_, _, info = src.sample(rng)
        k = info["kind"]
        kinds[k] = kinds.get(k, 0) + 1
        if k == "mixed":
            errs.append(abs(10 * np.log10((c_ @ c_) / (((n_ - c_) ** 2).sum())) - info["snr_db"]))
            snrs.append(info["snr_db"])
        if k in ("mixed", "clean_only"):
            rev += info["reverb"]
            levels.append(20 * np.log10(np.sqrt(np.mean(c_ ** 2)) + 1e-9))
        bad_target += (k in ("noise_only", "silence") and c_.any()) or (k == "clean_only" and not np.array_equal(n_, c_))
    n_speech = kinds["mixed"] + kinds.get("clean_only", 0)
    print("      样本类型：" + "，".join(f"{k} {v / n_total:.1%}" for k, v in sorted(kinds.items())))
    check("样本类型比例 ≈ 配置", all(abs(kinds.get(k, 0) / n_total - p) < 0.02 for k, p in
                               (("noise_only", cfg.p_noise_only), ("clean_only", cfg.p_clean), ("silence", cfg.p_silence))))
    check("目标正确（纯噪声/静音全零，clean_only 目标 = 输入）", bad_target == 0)
    check("实测 SNR = 抽到的 SNR", max(errs) < 0.05, f"最大偏差 {max(errs):.3f} dB；SNR 范围 {min(snrs):.1f}…{max(snrs):.1f}")
    check("混响比例 ≈ p_reverb", abs(rev / n_speech - cfg.p_reverb) < 0.03, f"{rev / n_speech:.3f} vs {cfg.p_reverb}")
    print(f"      干净语音电平 {np.percentile(levels, [5, 50, 95]).round(1)} dBFS（5/50/95 分位）")

    # 3) determinism: the same step always yields the same batch, different steps differ
    a, b = src.batch(5)[0], src.batch(5)[0]
    check("确定性：同 step 同 batch", np.array_equal(a, b))
    check("不同 step 不同 batch", not np.array_equal(a, src.batch(6)[0]))

    # 4) multiprocess Loader: starting from step 3 it must match the single-process batch(3) exactly (correct resume position); also measure throughput and memory
    base = meminfo()
    ld = Loader(src, start=3, workers=cfg.workers, prefetch=cfg.prefetch)
    first = next(ld)
    check("Loader 从指定 step 续上，内容一致", all(np.array_equal(x, y) for x, y in zip(first, src.batch(3))))
    for _ in range(cfg.prefetch):  # wait for the prefetch queue to settle
        next(ld)
    t0, n_batches, peak = time.perf_counter(), 40, 0.0
    for _ in range(n_batches):
        next(ld)
        peak = max(peak, meminfo() - base)
    dt = time.perf_counter() - t0
    rate = n_batches * cfg.batch_size / dt
    check("吞吐足以喂饱 GPU（实测 GPU 约 200 条/秒）", rate > 250, f"{rate:.0f} 条/秒（{cfg.workers} 个数据进程），系统内存比空闲时多用 {peak:.1f} GB")
    ld.close()

    # 5) noise source overview (reads the index): hours and bandwidth tiers per source in each split. With too little full-band noise in the validation set the validation loss comes out too low
    import json

    from transnoise.paths import DATA, INDEX

    hours: dict = {}
    wham_rows = []
    for split in ("train", "val", "test"):
        for line in (INDEX / f"{split}.jsonl").open():
            r = json.loads(line)
            if r["role"] != "noise":
                continue
            h = hours.setdefault((split, r["source"]), {"n": 0, "h": 0.0, "A": 0, "B": 0, "C": 0})
            h["n"] += 1
            h["h"] += r["duration"] / 3600
            h[r["bw_tier"]] += 1
            if r["source"] == "wham_noise":
                wham_rows.append((split, r))
    print("\n噪声来源概览：切分 / 来源 / 条数 / 小时 / 带宽分级 A,B,C")
    for (split, source), h in sorted(hours.items()):
        print(f"      {split:<6}{source:<16}{h['n']:>6}{h['h']:>8.1f} h   {h['A']},{h['B']},{h['C']}")
    full = {s: sum(h["h"] for (sp, _), h in hours.items() if sp == s) for s in ("train", "val", "test")}
    fullband = {s: sum(h["h"] * (h["A"] + h["B"]) / max(h["n"], 1) for (sp, _), h in hours.items() if sp == s) for s in ("train", "val", "test")}
    check("验证集里有足够的全频带噪声（≥ 3 小时 A/B 级，否则验证损失偏低、检测不出高频残留）", fullband["val"] >= 3.0,
          f"验证集 A/B 级约 {fullband['val']:.1f} h / 总 {full['val']:.1f} h；训练集 {fullband['train']:.1f} / {full['train']:.1f}")

    if wham_rows:  # WHAM! specifics: locations do not cross sets, no narrow-band files, all files exist
        loc_splits: dict = {}
        for split, r in wham_rows:
            loc_splits.setdefault(r["speaker"], set()).add(split)
        check("WHAM!：同一地点只在一个切分里（上游按地点划分）", all(len(s) == 1 for s in loc_splits.values()),
              f"{len(loc_splits)} 个地点，{len(wham_rows)} 段")
        check("WHAM!：全部是 A/B 级带宽（带宽门槛生效）", all(r["bw_tier"] in "AB" for _, r in wham_rows))
        missing = [r["path"] for _, r in wham_rows[:: max(1, len(wham_rows) // 200)] if not (DATA / r["path"]).exists()]
        check("WHAM!：抽样 200 段，staged 文件都存在", not missing, f"缺 {len(missing)}")
        durs = [r["duration"] for _, r in wham_rows]
        check("WHAM!：片段时长在 1–90 秒（长录音已切段）", 1.0 <= min(durs) and max(durs) <= 91.0, f"{min(durs):.1f}–{max(durs):.1f} s")

    print("\n全部通过" if ok else "\n有失败项")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
