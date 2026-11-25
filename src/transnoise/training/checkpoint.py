"""Checkpoints: model weights + optimizer state (including the iteration count, so the learning-rate schedule resumes) + EMA weights + the training loop's own state (best validation loss etc.).

Layout <run>/step_00001000/{model.npz, optim.npz, ema.npz, meta.json}, written to a temporary directory and then renamed, so an interruption never leaves a broken checkpoint;
<run>/latest points to the newest one; the most recent keep_last are kept on a rolling basis; best/ is stored separately (the caller decides when).
On restore the number and shapes of the weights are checked; a mismatched architecture raises an error instead of silently loading half of it.
"""

import json
import os
import shutil
import time
from pathlib import Path

import keras
import numpy as np


def _arrays(variables) -> dict[str, np.ndarray]:
    return {f"v{i}": keras.ops.convert_to_numpy(v) for i, v in enumerate(variables)}


def save(run_dir: Path, step: int, model, optimizer, ema, extra: dict | None = None, keep_last: int = 3,
         name: str | None = None) -> Path:
    """name=None: save as step_XXXXXXXX, update latest and prune old ones; name="best": save as best/, overwriting the old one."""
    run_dir.mkdir(parents=True, exist_ok=True)
    final = run_dir / (name or f"step_{step:08d}")
    tmp = run_dir / f".tmp_{final.name}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    np.savez(tmp / "model.npz", **_arrays(model.weights))
    np.savez(tmp / "optim.npz", **_arrays(optimizer.variables))
    np.savez(tmp / "ema.npz", **_arrays(ema))
    (tmp / "meta.json").write_text(json.dumps({
        "step": step, "extra": extra or {}, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_shapes": [list(w.shape) for w in model.weights],
        "optim_shapes": [list(v.shape) for v in optimizer.variables]}))
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)
    if name is None:
        pointer = run_dir / ".latest.tmp"
        pointer.write_text(final.name)
        os.replace(pointer, run_dir / "latest")
        for old in sorted(run_dir.glob("step_*"))[:-keep_last]:
            shutil.rmtree(old, ignore_errors=True)
    return final


def latest(run_dir: Path) -> Path | None:
    p = run_dir / "latest"
    return run_dir / p.read_text().strip() if p.exists() and (run_dir / p.read_text().strip()).exists() else None


def load(path: Path, model, optimizer, ema) -> dict:
    """Write the state in path back into the already-built model, optimizer and EMA variables; returns meta (including step and extra)."""
    meta = json.loads((path / "meta.json").read_text())
    for label, vars_, shapes, file in (("model", model.weights, meta["model_shapes"], "model.npz"),
                                       ("optimizer", optimizer.variables, meta["optim_shapes"], "optim.npz"),
                                       ("EMA", ema, meta["model_shapes"], "ema.npz")):
        have = [list(v.shape) for v in vars_]
        if have != shapes:
            raise ValueError(f"{path}: {label} variables do not match the checkpoint ({len(have)} vs {len(shapes)} stored); was the structure or config changed?")
        with np.load(path / file) as z:
            for i, v in enumerate(vars_):
                v.assign(z[f"v{i}"])
    return meta


def load_weights(path: Path, model, ema: bool = True) -> dict:
    """For inference: write only the weights into the already-built model (EMA weights by default); no optimizer needed. A shape mismatch raises an error."""
    meta = json.loads((path / "meta.json").read_text())
    have = [list(w.shape) for w in model.weights]
    if have != meta["model_shapes"]:
        raise ValueError(f"{path}: model structure does not match the checkpoint ({len(have)} weights vs {len(meta['model_shapes'])} stored); was model.toml changed?")
    with np.load(path / ("ema.npz" if ema else "model.npz")) as z:
        for i, w in enumerate(model.weights):
            w.assign(z[f"v{i}"])
    return meta
