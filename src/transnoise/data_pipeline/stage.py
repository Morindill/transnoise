import csv
import hashlib
import io
import json
import os
import re
import shutil
import threading
import zipfile
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import soundfile as sf

from ..paths import MANIFESTS, SR, STAGED
from .archive import archives_in, iter_files
from .audio import make_id, process
from .config import CFG
from .progress import Bar
from .registry import Source

AUDIO = re.compile(r"\.(wav|flac)$", re.I)
UPSTREAM = {"train": "train", "dev": "val", "test": "test"}
META_SPLIT = {"train": "train", "valid": "val", "val": "val", "dev": "val", "test": "test"}  # split names in the in-archive CSV (lowercase)
SPLIT, RAW_SW, STAGE = CFG["split"], CFG["raw_sw"], CFG["stage"]


def hash_split(key: str) -> str:
    h = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 2**32
    return "train" if h < 1 - SPLIT["val"] - SPLIT["test"] else "val" if h < 1 - SPLIT["test"] else "test"


def _load_done(mf: Path) -> set[str]:
    """Read the ids of already processed entries. After a hard kill the last manifest line may be half written; truncate that broken tail."""
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


def _load_meta(src: Source, unit: Path) -> dict[str, tuple[str, str]]:
    """In-archive CSV -> {file name: (split, group)}. Reads the zip's central directory at the end of the file; no need to walk the whole archive."""
    fcol, scol, gcol = src.meta_cols
    with zipfile.ZipFile(unit) as z, z.open(src.meta_csv) as f:
        return {r[fcol]: (META_SPLIT[r[scol].strip().lower()], r[gcol])
                for r in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))}


def _segments(src: Source, t: dict, data: bytes, done: set[str]) -> Iterator[dict]:
    """Long recordings: decode and pick the channel in the main process, cut into long_noise_chunk_sec-second segments, one
    task per segment (workers never receive a whole file of hundreds of MB). A tail shorter than half a segment is merged
    into the last segment, so no fragments are produced. A segment's id is determined by <member>#<index>, so segments
    already done are skipped when resuming."""
    with sf.SoundFile(io.BytesIO(data)) as f:
        n, sr = len(f), f.samplerate
        step = int(sr * CFG["clean"]["long_noise_chunk_sec"])
        starts = list(range(0, n, step))
        if len(starts) > 1 and n - starts[-1] < step // 2:
            starts.pop()
        for k, s in enumerate(starts):
            member = f"{t['member']}#{k}"
            tid = make_id(src.name, member)
            if tid in done:
                continue
            f.seek(s)
            x = f.read((n if k == len(starts) - 1 else s + step) - s, dtype="float32", always_2d=True)[:, src.channel]
            yield t | {"id": tid, "member": member, "audio": np.ascontiguousarray(x), "sr": sr}


def _tasks(src: Source, unit: Path, done: set[str], on_pos: Callable[[int], None]) -> Iterator[dict]:
    """Walk one input unit (an archive or a .sw file) and yield the tasks to process. on_pos reports the bytes read so far."""
    inc = re.compile(src.include) if src.include else None
    spk_re = re.compile(src.speaker) if src.speaker else None
    split_re = re.compile(src.split) if src.split else None
    meta = _load_meta(src, unit) if src.meta_csv else {}

    def base(member: str, speaker: str | None, split_key: str) -> dict:
        m = split_re.search(member) if split_re else None
        split = UPSTREAM[m.group("split")] if m else hash_split(f"{src.name}:{split_key}")
        return {"id": make_id(src.name, member), "source": src.name, "member": member,
                "role": src.role, "lang": src.lang, "speaker": speaker, "split": split, "min_bw": src.min_bandwidth_hz}

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
            if meta:  # upstream gives each file's split and group (location): use it directly; a missing entry is an error, never a silent fallback to hashing
                name = member.rsplit("/", 1)[-1]
                if name not in meta:
                    raise SystemExit(f"[{src.name}] {member} is not in {src.meta_csv}, cannot determine its split")
                t["split"], t["speaker"] = meta[name]
            if src.channel is not None:
                yield from _segments(src, t, read(), done)
            else:
                t["data"] = read()
                yield t


def stage(src: Source, workers: int | None = None, redo: bool = False) -> None:
    """The main thread reads the archives one after another (streaming decompression); worker processes decode, clean, resample and write FLAC.

    Resumable: the manifest records processed entries; <source>.units.json records archives that were fully processed, which are skipped entirely.
    """
    if not src.raw_dir.is_dir():
        raise SystemExit(f"[{src.name}] {src.raw_dir} does not exist; run download first")
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    mf, uf = MANIFESTS / f"{src.name}.jsonl", MANIFESTS / f"{src.name}.units.json"
    if redo:
        mf.unlink(missing_ok=True)
        uf.unlink(missing_ok=True)
        shutil.rmtree(STAGED / src.name, ignore_errors=True)

    all_units = sorted(src.raw_dir.glob("*.sw")) if src.kind == "raw" else archives_in(src.raw_dir)
    if not all_units:
        raise SystemExit(f"[{src.name}] no files to process in {src.raw_dir}")
    units_done: dict[str, int] = json.loads(uf.read_text()) if uf.exists() else {}
    units = [u for u in all_units if units_done.get(u.name) != u.stat().st_size]
    if not units:
        print(f"[{src.name}] already done, skipping ({len(all_units)} packages)")
        return

    done = _load_done(mf)
    if done or len(units) < len(all_units):
        print(f"[{src.name}] resuming: {len(done)} rows done, {len(units)}/{len(all_units)} packages left")
    workers = workers or STAGE["workers"] or max(1, min(16, (os.cpu_count() or 4) - 2))

    lock = threading.Lock()  # callbacks run in the process pool's management thread and must be mutually exclusive with the main thread
    stop = threading.Event()
    slots = threading.BoundedSemaphore(workers * 4)  # bound the number of in-flight tasks to control memory
    errors: list[BaseException] = []
    stats = {"n": 0, "kept": 0, "pos": 0}
    pos = {u.name: 0 for u in units}
    state = {u.name: {"pending": 0, "fed": False} for u in units}
    bar = Bar(src.name, sum(u.stat().st_size for u in units))

    def progress(final: bool = False) -> None:  # call with the lock held
        bar.update(stats["pos"], f"  {stats['n']} rows, kept {stats['kept']}", final)

    def set_pos(unit: Path, p: int) -> None:
        with lock:
            stats["pos"] += p - pos[unit.name]
            pos[unit.name] = p
            progress()

    def mark_if_complete(unit: Path) -> None:  # call with the lock held: mark only after the unit is fully read and all its in-flight tasks are written
        s = state[unit.name]
        if s["fed"] and s["pending"] == 0 and not errors:
            units_done[unit.name] = unit.stat().st_size
            uf.write_text(json.dumps(units_done))

    with mf.open("a") as out, ProcessPoolExecutor(workers) as pool:  # leave the pool first (waits for the callbacks to finish writing), then close the file
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
                    if stop.is_set():
                        return
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
    print(f"[{src.name}] done: added {stats['n']} rows, kept {stats['kept']} ({len(done)} existed before)")
