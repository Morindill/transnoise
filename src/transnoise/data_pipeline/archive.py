"""Stream through tar / tgz / tar.bz2 / zip archives (including nested tars) without extracting to disk."""

import re
import tarfile
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import BinaryIO

_TAR = re.compile(r"\.(tar|tar\.gz|tgz|tar\.bz2)$", re.I)
ARCHIVE_EXT = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".zip")

Entry = tuple[str, Callable[[], bytes]]


def archives_in(raw_dir: Path) -> list[Path]:
    return sorted(p for p in raw_dir.iterdir() if p.name.lower().endswith(ARCHIVE_EXT))


def iter_files(f: BinaryIO, on_pos: Callable[[int], None]) -> Iterator[Entry]:
    """Yield (path inside the archive, read function). The read function must be called before the iterator advances.

    on_pos(compressed bytes read so far) is called once per yielded file and drives the progress bar. zip cannot use
    f.tell(): ZipFile reads the central directory at the end of the file first, so the pointer jumps to the end; for zip
    the entry's offset in the file is used instead.
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
