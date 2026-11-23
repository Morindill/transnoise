"""分段多连接下载：把单个大文件切成块，多个线程用 HTTP Range 并发取，可断点续传。

进度记在 <文件>.state（已完成的块序号），数据写在 <文件>.part，全部完成后改名。
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
    """返回文件总字节数；要求服务器支持 Range。"""
    req = urllib.request.Request(url, headers={**HEADERS, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        if r.status != 206:
            raise OSError("服务器不支持 Range 请求")
        return int(r.headers["Content-Range"].rsplit("/", 1)[1])


def _get_range(url: str, fd: int, start: int, end: int, tick, stop: threading.Event) -> None:
    """下载 [start, end] 并写入 fd 的对应偏移；出错时从已收到的位置继续重试。"""
    pos = start
    for attempt in range(DL["retries"]):
        try:
            req = urllib.request.Request(url, headers={**HEADERS, "Range": f"bytes={pos}-{end}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status != 206:  # 返回 200 整文件的话，写进去的偏移就错了
                    raise OSError(f"Range 请求返回 {r.status}")
                while b := r.read(BLOCK):
                    if stop.is_set():
                        raise _Stopped
                    os.pwrite(fd, b, pos)
                    pos += len(b)
                    tick(len(b))
            if pos != end + 1:  # read(amt) 在连接中途断开时只返回空，不抛异常，只能靠长度发现
                raise OSError("数据不完整")
            return
        except (OSError, http.client.HTTPException):
            if attempt == DL["retries"] - 1:
                raise
            time.sleep(2**attempt)


def fetch(url: str, dest: Path, conns: int) -> None:
    size = _probe(url)
    if dest.exists() and dest.stat().st_size == size:
        print(f"  已完整，跳过: {dest.name}")
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
                stop.set()  # 让正在下载的线程尽快退出，Ctrl+C 不用等整块下完
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
    """从外层 zip 里取出 inner.name，先写 .part 再改名（中途被打断不会留下残缺文件），然后删掉外层包。"""
    print(f"  解出 {inner.name} …")
    part = inner.with_name(inner.name + ".part")
    with zipfile.ZipFile(outer) as z, z.open(inner.name) as s, part.open("wb") as o:
        shutil.copyfileobj(s, o, 1 << 24)
    part.replace(inner)
    outer.unlink()


def download(src: Source, conns: int, mirror: str = "") -> None:
    """mirror: 配置里 [download.mirrors] 的别名，或直接给主机名；只替换 OpenSLR 的 URL。"""
    host = DL["mirrors"].get(mirror, mirror)
    urls = src.urls
    if not urls and src.raw_name:  # 与别的源共用原始包（如 rirs_real 用 rirs_simulated 的包）
        urls = next((s.urls for s in SOURCES.values() if s.raw_name == src.raw_name and s.urls), ())
    if not urls:
        raise SystemExit(f"[{src.name}] 没有下载地址")
    src.raw_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for url in urls:
        dest = src.raw_dir / Path(url).name
        inner = src.raw_dir / src.unwrap if src.unwrap else None
        if inner and inner.exists():
            print(f"[{src.name}] 已完整，跳过: {inner.name}")
            continue
        print(f"[{src.name}] {dest.name}")
        if host:
            url = url.replace(DL["openslr_host"], host)
        try:
            fetch(url, dest, conns)
            if inner:
                _unwrap(dest, inner)
        except Exception as e:
            print(f"  失败: {e}（重跑同一命令可续传）")
            failed.append(url)
    if failed:
        raise SystemExit(f"[{src.name}] {len(failed)} 个文件失败:\n" + "\n".join(failed))
