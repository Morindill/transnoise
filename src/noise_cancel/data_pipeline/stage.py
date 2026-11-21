import hashlib
import json
import os
import re
import shutil
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from ..paths import MANIFESTS, SR, STAGED
from .archive import archives_in, iter_files
from .audio import make_id, process
from .config import CFG
from .progress import Bar
from .registry import Source

AUDIO = re.compile(r"\.(wav|flac)$", re.I)
UPSTREAM = {"train": "train", "dev": "val", "test": "test"}
SPLIT, RAW_SW, STAGE = CFG["split"], CFG["raw_sw"], CFG["stage"]


def hash_split(key: str) -> str:
    h = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 2**32
    return "train" if h < 1 - SPLIT["val"] - SPLIT["test"] else "val" if h < 1 - SPLIT["test"] else "test"


def _load_done(mf: Path) -> set[str]:
    """读取已处理条目的 id。被强杀时 manifest 末行可能只写了一半，这里把残缺的尾部截掉。"""
    ids: set[str] = set()
    if not mf.exists():
        return ids
    good_end = 0
    with mf.open("rb") as f:
        for line in f:
            try:
                ids.add(json.loads(line)["id"])
            except ValueError:
                break
            good_end = f.tell()
    if good_end < mf.stat().st_size:
        with mf.open("r+b") as f:
            f.truncate(good_end)
    return ids


def _tasks(src: Source, unit: Path, done: set[str], on_pos: Callable[[int], None]) -> Iterator[dict]:
    """遍历一个输入单元（一个压缩包，或一个 .sw 文件），产出待处理任务。on_pos 报告已读字节数。"""
    inc = re.compile(src.include) if src.include else None
    spk_re = re.compile(src.speaker) if src.speaker else None
    split_re = re.compile(src.split) if src.split else None

    def base(member: str, speaker: str | None, split_key: str) -> dict:
        m = split_re.search(member) if split_re else None
        split = UPSTREAM[m.group("split")] if m else hash_split(f"{src.name}:{split_key}")
        return {"id": make_id(src.name, member), "source": src.name, "member": member,
                "role": src.role, "lang": src.lang, "speaker": speaker, "split": split}

    if src.kind == "raw":
        step = SR * RAW_SW["chunk_sec"]
        pcm = np.memmap(unit, dtype="<i2", mode="r")
        for i, off in enumerate(range(0, len(pcm), step)):
            on_pos(off * 2)
            member = f"{unit.name}#{i}"
            if make_id(src.name, member) in done:
                continue
            t = base(member, None, f"{unit.name}#{i // RAW_SW['block']}")
            t["pcm"] = np.array(pcm[off: off + step])
            yield t
        return

    with unit.open("rb") as f:
        for member, read in iter_files(f, on_pos):
            if not AUDIO.search(member) or (inc and not inc.search(member)) or make_id(src.name, member) in done:
                continue
            sm = spk_re.search(member) if spk_re else None
            speaker = sm.group("spk") if sm else None
            t = base(member, speaker, speaker or member)
            t["data"] = read()
            yield t


def stage(src: Source, workers: int | None = None, limit: int | None = None, redo: bool = False) -> None:
    """主线程依次读压缩包（流式解压），workers 个进程做解码、清洗、重采样、写 FLAC。

    断点续跑：manifest 记已处理的条目；<源>.units.json 记已完整处理过的压缩包，整包直接跳过。
    """
    if not src.raw_dir.is_dir():
        raise SystemExit(f"[{src.name}] {src.raw_dir} 不存在，先 download")
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    mf, uf = MANIFESTS / f"{src.name}.jsonl", MANIFESTS / f"{src.name}.units.json"
    if redo:
        mf.unlink(missing_ok=True)
        uf.unlink(missing_ok=True)
        shutil.rmtree(STAGED / src.name, ignore_errors=True)

    all_units = sorted(src.raw_dir.glob("*.sw")) if src.kind == "raw" else archives_in(src.raw_dir)
    if not all_units:
        raise SystemExit(f"[{src.name}] {src.raw_dir} 里没有可处理的文件")
    units_done: dict[str, int] = json.loads(uf.read_text()) if uf.exists() else {}
    units = [u for u in all_units if units_done.get(u.name) != u.stat().st_size]
    if not units:
        print(f"[{src.name}] 已完成，跳过（{len(all_units)} 个包）")
        return

    done = _load_done(mf)
    if done or len(units) < len(all_units):
        print(f"[{src.name}] 续跑：已有 {len(done)} 条，剩 {len(units)}/{len(all_units)} 个包")
    workers = workers or STAGE["workers"] or max(1, min(16, (os.cpu_count() or 4) - 2))

    lock = threading.Lock()  # 回调在进程池的管理线程里执行，要和主线程互斥
    stop = threading.Event()
    slots = threading.BoundedSemaphore(workers * 4)  # 限制在途任务数，控制内存
    errors: list[BaseException] = []
    stats = {"submitted": 0, "n": 0, "kept": 0, "pos": 0}
    pos = {u.name: 0 for u in units}
    state = {u.name: {"pending": 0, "fed": False} for u in units}
    bar = Bar(src.name, sum(u.stat().st_size for u in units))

    def progress(final: bool = False) -> None:  # 在 lock 内调用
        bar.update(stats["pos"], f"  {stats['n']} 条 保留 {stats['kept']}", final)

    def set_pos(unit: Path, p: int) -> None:
        with lock:
            stats["pos"] += p - pos[unit.name]
            pos[unit.name] = p
            progress()

    def mark_if_complete(unit: Path) -> None:  # 在 lock 内调用：读完且在途任务全部写入后才标记
        s = state[unit.name]
        if s["fed"] and s["pending"] == 0 and not errors:
            units_done[unit.name] = unit.stat().st_size
            uf.write_text(json.dumps(units_done))

    with mf.open("a") as out, ProcessPoolExecutor(workers) as pool:  # 先退出 pool（等回调写完），再关文件
        def on_done(fut, unit: Path) -> None:
            slots.release()
            try:
                row = fut.result()
            except BaseException as e:
                errors.append(e)
                stop.set()
                return
            with lock:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                stats["n"] += 1
                stats["kept"] += row["keep"]
                state[unit.name]["pending"] -= 1
                mark_if_complete(unit)
                progress()

        def feed() -> None:
            for unit in units:
                for task in _tasks(src, unit, done, partial(set_pos, unit)):
                    if stop.is_set() or (limit is not None and stats["submitted"] >= limit):
                        return
                    stats["submitted"] += 1
                    with lock:
                        state[unit.name]["pending"] += 1
                    slots.acquire()
                    pool.submit(process, task).add_done_callback(partial(on_done, unit=unit))
                set_pos(unit, unit.stat().st_size)
                with lock:
                    state[unit.name]["fed"] = True
                    mark_if_complete(unit)

        try:
            feed()
        except BaseException:
            stop.set()
            pool.shutdown(wait=False, cancel_futures=True)
            bar.abort()
            raise
    if errors:
        raise errors[0]
    with lock:
        progress(final=True)
    print(f"[{src.name}] 完成：新增 {stats['n']} 条，保留 {stats['kept']}（此前已有 {len(done)}）")
