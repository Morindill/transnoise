"""断点存档：模型权重 + 优化器状态（含迭代数，学习率进度随之恢复）+ EMA 权重 + 训练循环自己的状态（best 验证损失等）。

目录 <run>/step_00001000/{model.npz, optim.npz, ema.npz, meta.json}，先写到临时目录再改名，所以写到一半被打断不会留下坏的检查点；
<run>/latest 指向最新一个；滚动保留最近 keep_last 个；best/ 另存（由调用方决定什么时候存）。
恢复时校验权重个数和形状，结构对不上会直接报错，而不是悄悄加载一半。
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
    """name=None：存成 step_XXXXXXXX 并更新 latest、清理旧的；name="best"：存成 best/，覆盖旧的。"""
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
    """把 path 里的状态写回已经 build 过的 model、optimizer 和 EMA 变量，返回 meta（含 step 和 extra）。"""
    meta = json.loads((path / "meta.json").read_text())
    for label, vars_, shapes, file in (("模型", model.weights, meta["model_shapes"], "model.npz"),
                                       ("优化器", optimizer.variables, meta["optim_shapes"], "optim.npz"),
                                       ("EMA", ema, meta["model_shapes"], "ema.npz")):
        have = [list(v.shape) for v in vars_]
        if have != shapes:
            raise ValueError(f"{path}：{label}的变量和检查点对不上（{len(have)} 个 vs 存档 {len(shapes)} 个），结构或配置改过？")
        with np.load(path / file) as z:
            for i, v in enumerate(vars_):
                v.assign(z[f"v{i}"])
    return meta


def load_weights(path: Path, model, ema: bool = True) -> dict:
    """推理用：只把权重写进已经 build 过的 model（默认用 EMA 权重），不需要优化器。形状对不上会报错。"""
    meta = json.loads((path / "meta.json").read_text())
    have = [list(w.shape) for w in model.weights]
    if have != meta["model_shapes"]:
        raise ValueError(f"{path}：模型结构和检查点对不上（{len(have)} 个权重 vs 存档 {len(meta['model_shapes'])} 个），model.toml 改过？")
    with np.load(path / ("ema.npz" if ema else "model.npz")) as z:
        for i, w in enumerate(model.weights):
            w.assign(z[f"v{i}"])
    return meta
