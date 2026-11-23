import sys
import time
from collections import deque


class Bar:
    """按字节数的进度条：终端里原地刷新，重定向时每 10 秒一行。非线程安全，调用方自己加锁。"""

    def __init__(self, label: str, total: int, start: int = 0):
        self.label, self.total = label, total
        self.samples: deque = deque([(time.time(), start)])  # 近一段时间的 (时间, 已完成字节)，用来算瞬时速度
        self.last = 0.0
        self.tty = sys.stdout.isatty()
        self.interval = 0.5 if self.tty else 10  # 刷新间隔（秒）
        self.window = 5 if self.tty else 30  # 速度窗口必须比刷新间隔长，否则窗口里只剩一个样本，速度恒为 0

    def update(self, done: int, extra: str = "", final: bool = False) -> None:
        now = time.time()
        if not final and now - self.last < self.interval:
            return
        self.last = now
        self.samples.append((now, done))
        while len(self.samples) > 1 and now - self.samples[0][0] > self.window:
            self.samples.popleft()
        t_old, d_old = self.samples[0]
        speed = (done - d_old) / max(now - t_old, 1e-6)
        left = int((self.total - done) / speed) if speed > 1e3 else None
        eta = "0s" if final else "--" if left is None else f"{left // 60}m{left % 60:02d}s"
        frac = done / self.total if self.total else 1.0
        bar = "#" * int(24 * frac) + "." * (24 - int(24 * frac))
        line = (f"  {self.label} [{bar}] {frac:6.1%}  {done / 1e9:.2f}/{self.total / 1e9:.2f} GB  "
                f"{speed / 1e6:5.1f} MB/s  ETA {eta}{extra}")
        print("\r" + line + "\033[K" if self.tty else line, end="\n" if final or not self.tty else "", flush=True)

    def abort(self) -> None:
        if self.tty:
            print()
