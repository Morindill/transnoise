"""Segmented multi-connection download: a large file is cut into chunks fetched concurrently by several threads with HTTP Range; resumable.

Progress is kept in <file>.state (indices of finished chunks), data is written to <file>.part, and renamed once everything is complete.
"""

import http.client
import json
import os
import shutil
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import CFG
from .progress import Bar
from .registry import SOURCES, Source

DL = CFG["download"]
HEADERS = {"User-Agent": "transnoise/0.1"}
CHUNK = DL["chunk_mb"] << 20
BLOCK = DL["block_mb"] << 20


class _Stopped(Exception):
    pass


def _probe(url: str) -> int:
    """Return the file's total size in bytes; the server must support Range."""
    req = urllib.request.Request(url, headers={**HEADERS, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        if r.status != 206:
            raise OSError("server does not support Range requests")
        return int(r.headers["Content-Range"].rsplit("/", 1)[1])


def _get_range(url: str, fd: int, start: int, end: int, tick, stop: threading.Event) -> None:
    """Download [start, end] and write it at the matching offset of fd; on error, retry from the position already received."""
    pos = start
    for attempt in range(DL["retries"]):
        try:
            req = urllib.request.Request(url, headers={**HEADERS, "Range": f"bytes={pos}-{end}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status != 206:  # if the server returned the whole file with 200, the offsets written would be wrong
                    raise OSError(f"Range request returned {r.status}")
                while b := r.read(BLOCK):
                    if stop.is_set():
                        raise _Stopped
                    os.pwrite(fd, b, pos)
                    pos += len(b)
                    tick(len(b))
            if pos != end + 1:  # read(amt) just returns empty when the connection drops midway and raises nothing, so only the length reveals it
                raise OSError("incomplete data")
            return
        except (OSError, http.client.HTTPException):
            if attempt == DL["retries"] - 1:
                raise
            time.sleep(2**attempt)


def fetch(url: str, dest: Path, conns: int) -> None:
    size = _probe(url)
    if dest.exists() and dest.stat().st_size == size:
        print(f"  already complete, skipping: {dest.name}")
        return

    part = dest.with_name(dest.name + ".part")
    state = dest.with_name(dest.name + ".state")
    done: set[int] = set()
    if state.exists() and part.exists():
        s = json.loads(state.read_text())
        if s["size"] == size and s["chunk"] == CHUNK:
            done = set(s["done"])

    def save() -> None:
        tmp = state.with_suffix(".tmp")
        tmp.write_text(json.dumps({"size": size, "chunk": CHUNK, "done": sorted(done)}))
        tmp.replace(state)

    fd = os.open(part, os.O_RDWR | os.O_CREAT)
    os.ftruncate(fd, size)
    got = sum(min(CHUNK, size - i * CHUNK) for i in done)
    lock = threading.Lock()
    stop = threading.Event()
    bar = Bar(dest.name, size, got)

    def tick(k: int) -> None:
        nonlocal got
        with lock:
            got += k
            bar.update(got)

    todo = [i for i in range(-(-size // CHUNK)) if i not in done]
    try:
        with ThreadPoolExecutor(conns) as pool:
            futs = {pool.submit(_get_range, url, fd, i * CHUNK, min((i + 1) * CHUNK, size) - 1, tick, stop): i
                    for i in todo}
            try:
                for f in as_completed(futs):
                    f.result()
                    done.add(futs[f])
                    save()
            except BaseException:
                stop.set()  # make running download threads exit promptly so Ctrl+C does not wait for a whole chunk
                for f in futs:
                    f.cancel()
                bar.abort()
                raise
    finally:
        os.close(fd)
    bar.update(size, final=True)
    part.replace(dest)
    state.unlink(missing_ok=True)


def _unwrap(outer: Path, inner: Path) -> None:
    """Extract inner.name from the outer zip, writing .part first and then renaming (an interruption never leaves a broken file), then delete the outer archive."""
    print(f"  extracting {inner.name} …")
    part = inner.with_name(inner.name + ".part")
    with zipfile.ZipFile(outer) as z, z.open(inner.name) as s, part.open("wb") as o:
        shutil.copyfileobj(s, o, 1 << 24)
    part.replace(inner)
    outer.unlink()


def download(src: Source, conns: int) -> None:
    urls = src.urls
    if not urls and src.raw_name:  # shares the raw archive with another source (e.g. rirs_real uses rirs_simulated's archive)
        urls = next((s.urls for s in SOURCES.values() if s.raw_name == src.raw_name and s.urls), ())
    if not urls:
        raise SystemExit(f"[{src.name}] no download URL")
    src.raw_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for url in urls:
        dest = src.raw_dir / Path(url).name
        inner = src.raw_dir / src.unwrap if src.unwrap else None
        if inner and inner.exists():
            print(f"[{src.name}] already complete, skipping: {inner.name}")
            continue
        print(f"[{src.name}] {dest.name}")
        try:
            fetch(url, dest, conns)
            if inner:
                _unwrap(dest, inner)
        except Exception as e:
            print(f"  failed: {e} (re-run the same command to resume)")
            failed.append(url)
    if failed:
        raise SystemExit(f"[{src.name}] {len(failed)} file(s) failed:\n" + "\n".join(failed))
