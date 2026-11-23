import sys
import time
from collections import deque


class Bar:
    """Byte-based progress bar: redraws in place on a terminal, one line every 10 s when redirected. Not thread-safe; the caller locks."""

    def __init__(self, label: str, total: int, start: int = 0):
        self.label, self.total = label, total
        self.samples: deque = deque([(time.time(), start)])  # recent (time, bytes done) pairs, used for the instantaneous speed
        self.last = 0.0
        self.tty = sys.stdout.isatty()
        self.interval = 0.5 if self.tty else 10  # refresh interval (seconds)
        self.window = 5 if self.tty else 30  # the speed window must be longer than the interval, otherwise only one sample is left and the speed is always 0

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
