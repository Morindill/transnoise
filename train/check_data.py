"""训练数据自检（读真实的 data/index/train.jsonl，不训练）：形状/数值、SNR 是否落在配置范围、确定性、续训位置、吞吐和内存。
用法：uv run transnoise check data"""

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

    # 1) 形状、数值、峰值
    noisy, clean, cutoff = src.batch(0)
    check("形状", noisy.shape == clean.shape == (cfg.batch_size, cfg.samples) and cutoff.shape == (cfg.batch_size,),
          f"{noisy.shape}，dtype={noisy.dtype}")
    check("数值有限、峰值 ≤ 0.95", np.isfinite(noisy).all() and np.isfinite(clean).all() and max(np.abs(noisy).max(), np.abs(clean).max()) <= 0.9501,
          f"峰值 {max(np.abs(noisy).max(), np.abs(clean).max()):.3f}")
    check("截止频率取自语料带宽", cutoff.min() >= 4000 and cutoff.max() <= 24000, f"范围 {cutoff.min():.0f}–{cutoff.max():.0f} Hz，"
          f"A/B/C 混合：{np.round(np.percentile(cutoff, [10, 50, 90]))}")

    # 2) SNR 和混响比例：用 sample() 抽 400 个，实测 SNR 必须等于抽到的 SNR（peak 缩放不改变 SNR），混响比例接近 p_reverb
    rng = np.random.default_rng(7)
    errs, rev, snrs, levels = [], 0, [], []
    for _ in range(400):
        n_, c_, _, info = src.sample(rng)
        errs.append(abs(10 * np.log10((c_ @ c_) / (((n_ - c_) ** 2).sum())) - info["snr_db"]))
        rev += info["reverb"]
        snrs.append(info["snr_db"])
        levels.append(20 * np.log10(np.sqrt(np.mean(c_ ** 2)) + 1e-9))
    check("实测 SNR = 抽到的 SNR", max(errs) < 0.05, f"最大偏差 {max(errs):.3f} dB；SNR 范围 {min(snrs):.1f}…{max(snrs):.1f}")
    check("混响比例 ≈ p_reverb", abs(rev / 400 - cfg.p_reverb) < 0.06, f"{rev / 400:.3f} vs {cfg.p_reverb}")
    print(f"      干净语音电平 {np.percentile(levels, [5, 50, 95]).round(1)} dBFS（5/50/95 分位）")

    # 3) 确定性：同一个 step 永远是同一个 batch，不同 step 不同
    a, b = src.batch(5)[0], src.batch(5)[0]
    check("确定性：同 step 同 batch", np.array_equal(a, b))
    check("不同 step 不同 batch", not np.array_equal(a, src.batch(6)[0]))

    # 4) 多进程 Loader：从 step 3 开始取，必须和单进程 batch(3) 完全一致（续训位置正确）；同时测吞吐和内存
    base = meminfo()
    ld = Loader(src, start=3, workers=cfg.workers, prefetch=cfg.prefetch)
    first = next(ld)
    check("Loader 从指定 step 续上，内容一致", all(np.array_equal(x, y) for x, y in zip(first, src.batch(3))))
    for _ in range(cfg.prefetch):  # 等预取队列稳定
        next(ld)
    t0, n_batches, peak = time.perf_counter(), 40, 0.0
    for _ in range(n_batches):
        next(ld)
        peak = max(peak, meminfo() - base)
    dt = time.perf_counter() - t0
    rate = n_batches * cfg.batch_size / dt
    check("吞吐足以喂饱 GPU（实测 GPU 约 200 条/秒）", rate > 250, f"{rate:.0f} 条/秒（{cfg.workers} 个数据进程），系统内存比空闲时多用 {peak:.1f} GB")
    ld.close()

    # 5) 噪声来源概览（读 index）：各切分里每个来源的小时数和带宽分级。验证集里全频带噪声太少，验证损失就会偏低
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

    if wham_rows:  # WHAM! 专项：地点不跨集合、没有窄带文件、文件都在
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
