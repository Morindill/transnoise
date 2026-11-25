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
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

    cfg = TrainConfig.load()
    t0 = time.perf_counter()
    src = BatchSource("train", cfg)
    idx = src.idx
    print(f"index loaded in {time.perf_counter() - t0:.1f}s: speech {len(idx.speech_path)} rows; noise {len(idx.noise_path)} files; RIR {len(idx.rir)} sources {[len(p) for p in idx.rir]}")

    # 1) shapes, values, peak
    noisy, clean, cutoff = src.batch(0)
    check("shape", noisy.shape == clean.shape == (cfg.batch_size, cfg.samples) and cutoff.shape == (cfg.batch_size,),
          f"{noisy.shape}, dtype={noisy.dtype}")
    check("values finite, peak <= 0.95", np.isfinite(noisy).all() and np.isfinite(clean).all() and max(np.abs(noisy).max(), np.abs(clean).max()) <= 0.9501,
          f"peak {max(np.abs(noisy).max(), np.abs(clean).max()):.3f}")
    check("cutoff frequency comes from the corpus bandwidth", cutoff.min() >= 4000 and cutoff.max() <= 24000, f"range {cutoff.min():.0f}–{cutoff.max():.0f} Hz, "
          f"A/B/C mix:{np.round(np.percentile(cutoff, [10, 50, 90]))}")

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
    print("      sample kinds: " + ", ".join(f"{k} {v / n_total:.1%}" for k, v in sorted(kinds.items())))
    check("sample-kind proportions ≈ config", all(abs(kinds.get(k, 0) / n_total - p) < 0.02 for k, p in
                               (("noise_only", cfg.p_noise_only), ("clean_only", cfg.p_clean), ("silence", cfg.p_silence))))
    check("targets correct (noise-only/silence all zero, clean_only target = input)", bad_target == 0)
    check("measured SNR = drawn SNR", max(errs) < 0.05, f"max deviation {max(errs):.3f} dB; SNR range {min(snrs):.1f}…{max(snrs):.1f}")
    check("reverb proportion ≈ p_reverb", abs(rev / n_speech - cfg.p_reverb) < 0.03, f"{rev / n_speech:.3f} vs {cfg.p_reverb}")
    print(f"      clean speech level {np.percentile(levels, [5, 50, 95]).round(1)} dBFS (5/50/95 percentiles)")

    # 3) determinism: the same step always yields the same batch, different steps differ
    a, b = src.batch(5)[0], src.batch(5)[0]
    check("deterministic: same step gives the same batch", np.array_equal(a, b))
    check("different steps give different batches", not np.array_equal(a, src.batch(6)[0]))

    # 4) multiprocess Loader: starting from step 3 it must match the single-process batch(3) exactly (correct resume position); also measure throughput and memory
    base = meminfo()
    ld = Loader(src, start=3, workers=cfg.workers, prefetch=cfg.prefetch)
    first = next(ld)
    check("Loader resumes from the given step with identical content", all(np.array_equal(x, y) for x, y in zip(first, src.batch(3))))
    for _ in range(cfg.prefetch):  # wait for the prefetch queue to settle
        next(ld)
    t0, n_batches, peak = time.perf_counter(), 40, 0.0
    for _ in range(n_batches):
        next(ld)
        peak = max(peak, meminfo() - base)
    dt = time.perf_counter() - t0
    rate = n_batches * cfg.batch_size / dt
    check("throughput is enough to feed the GPU (measured GPU: about 200 clips/s)", rate > 250, f"{rate:.0f} clips/s ({cfg.workers} data processes), system memory use {peak:.1f} GB above idle")
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
    print("\nnoise sources: split / source / count / hours / bandwidth tier A,B,C")
    for (split, source), h in sorted(hours.items()):
        print(f"      {split:<6}{source:<16}{h['n']:>6}{h['h']:>8.1f} h   {h['A']},{h['B']},{h['C']}")
    full = {s: sum(h["h"] for (sp, _), h in hours.items() if sp == s) for s in ("train", "val", "test")}
    fullband = {s: sum(h["h"] * (h["A"] + h["B"]) / max(h["n"], 1) for (sp, _), h in hours.items() if sp == s) for s in ("train", "val", "test")}
    check("enough full-band noise in the validation set (>= 3 hours of tier A/B; otherwise the validation loss is too low and high-frequency residue goes undetected)", fullband["val"] >= 3.0,
          f"validation tier A/B about {fullband['val']:.1f} h / total {full['val']:.1f} h; training set {fullband['train']:.1f} / {full['train']:.1f}")

    if wham_rows:  # WHAM! specifics: locations do not cross sets, no narrow-band files, all files exist
        loc_splits: dict = {}
        for split, r in wham_rows:
            loc_splits.setdefault(r["speaker"], set()).add(split)
        check("WHAM!: each location appears in only one split (the upstream split is by location)", all(len(s) == 1 for s in loc_splits.values()),
              f"{len(loc_splits)} locations, {len(wham_rows)} segments")
        check("WHAM!: all tier A/B bandwidth (the bandwidth gate works)", all(r["bw_tier"] in "AB" for _, r in wham_rows))
        missing = [r["path"] for _, r in wham_rows[:: max(1, len(wham_rows) // 200)] if not (DATA / r["path"]).exists()]
        check("WHAM!: 200 sampled segments, staged files all exist", not missing, f"missing {len(missing)}")
        durs = [r["duration"] for _, r in wham_rows]
        check("WHAM!: segment durations within 1–90 s (long recordings are already cut)", 1.0 <= min(durs) and max(durs) <= 91.0, f"{min(durs):.1f}–{max(durs):.1f} s")

    print("\nall passed" if ok else "\nsome checks FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
