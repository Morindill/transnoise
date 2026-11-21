"""流式遍历 tar / tgz / zip（含嵌套 tar），不解压到磁盘。"""

import re
import tarfile
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import BinaryIO

_TAR = re.compile(r"\.(tar|tar\.gz|tgz)$", re.I)
ARCHIVE_EXT = (".tar", ".tar.gz", ".tgz", ".zip")

Entry = tuple[str, Callable[[], bytes]]


def archives_in(raw_dir: Path) -> list[Path]:
    return sorted(p for p in raw_dir.iterdir() if p.name.lower().endswith(ARCHIVE_EXT))


def iter_files(f: BinaryIO, on_pos: Callable[[int], None]) -> Iterator[Entry]:
    """产出 (包内路径, 读取函数)。读取函数必须在迭代器前进之前调用。

    on_pos(已读取的压缩字节数) 每产出一个文件调用一次，用作进度。zip 不能用 f.tell()：
    ZipFile 一上来就读文件末尾的目录，指针会先跳到末尾，所以 zip 用条目在文件里的偏移量。
    """
    if zipfile.is_zipfile(f):
        with zipfile.ZipFile(f) as z:
            for info in z.infolist():
                if not info.is_dir():
                    on_pos(info.header_offset)
                    yield info.filename, lambda i=info: z.read(i)
    else:
        f.seek(0)
        with tarfile.open(fileobj=f, mode="r|*") as tf:
            for entry in _iter_tar(tf, ""):
                on_pos(f.tell())
                yield entry


def _iter_tar(tf: tarfile.TarFile, prefix: str) -> Iterator[Entry]:
    for m in tf:
        if not m.isfile():
            continue
        inner_f = tf.extractfile(m)
        name = prefix + m.name
        if _TAR.search(m.name):
            with tarfile.open(fileobj=inner_f, mode="r|*") as inner:
                yield from _iter_tar(inner, name + "/")
        else:
            yield name, inner_f.read
