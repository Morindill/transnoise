"""Training loop:  uv run transnoise train [--name NAME] [--set field=value ...] [--max-steps N] [--rerun [--yes]]

- Resumes automatically when a checkpoint exists; --rerun first deletes everything in the run directory (checkpoints, best, logs, TensorBoard curves) and trains from scratch.
- Ctrl+C / SIGTERM saves a checkpoint before exiting. A tqdm progress bar sits at the bottom of the terminal (disabled automatically when redirected to a file, leaving only the log lines).
- Logs: one line every log_every steps to the terminal, <run>/logs.csv and TensorBoard (written only if installed); curves: tensorboard --logdir <run>/tb
- Every eval_every steps the raw and EMA weights are evaluated on the fixed validation set; best/ is saved when the EMA validation loss reaches a new low.
- After nan_patience consecutive NaN/Inf steps (those steps are already skipped) it rolls back to the previous checkpoint; more than max_rollbacks rollbacks stop with an error.

Note: data processes are started with spawn and re-import this module, so only light imports sit at the top level; TF/Keras are imported inside main().
"""

import argparse
import csv
import shutil
import signal
import sys
import time
from dataclasses import fields, replace
from pathlib import Path

from ..paths import ROOT
from .config import TrainConfig

try:
    from tqdm import tqdm
except ImportError:  # without tqdm there is no progress bar; everything else works as usual
    tqdm = None


def say(msg: str = "") -> None:
    """Print one line. With a progress bar use tqdm.write so it does not overwrite the bar at the bottom."""
    (tqdm.write if tqdm else print)(msg)


class _NoBar:
    n = 0

    def update(self, n=1): ...
    def set_postfix(self, *a, **k): ...
    def close(self): ...


class Logger:
    """Terminal + CSV + TensorBoard (optional)."""

    def __init__(self, run_dir: Path):
        run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path, self.keys = run_dir / "logs.csv", {}
        self.tb = None
        try:
            import tensorboard  # noqa: F401
            import tensorflow as tf

            self.tf, self.tb = tf, tf.summary.create_file_writer(str(run_dir / "tb"))
        except ImportError:
            say("没装 tensorboard，只写 CSV（装法：uv sync --project train）")

    def log(self, step: int, values: dict) -> None:
        row = {"step": step, "time": time.strftime("%H:%M:%S"), **values}
        path = self.csv_path.with_name("val.csv" if "val_ema" in values else "logs.csv")  # training rows and validation rows have different columns, so they go to two files
        keys = self.keys.setdefault(path, list(row))
        new = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)
        if self.tb is not None:
            with self.tb.as_default():
                for k, v in values.items():
                    self.tf.summary.scalar(k, v, step=step)
            self.tb.flush()
        say(f"[{row['time']}] step {step:>7} " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in values.items()))


def parse_overrides(cfg: TrainConfig, items: list[str]) -> TrainConfig:
    types = {f.name: f.type for f in fields(cfg)}
    upd = {}
    for it in items:
        k, v = it.split("=", 1)
        if k not in types:
            raise SystemExit(f"--set {k}：TrainConfig 没有这个字段")
        upd[k] = type(getattr(cfg, k))(v) if not isinstance(getattr(cfg, k), tuple) else tuple(int(x) for x in v.split(","))
    return replace(cfg, **upd)


def wipe(run_dir: Path, yes: bool) -> None:
    """--rerun: delete the whole run directory. First list what will be deleted; an interactive terminal asks for confirmation, a non-interactive environment needs an explicit --yes."""
    if not run_dir.exists():
        say(f"{run_dir} 不存在，没有可删除的存档，直接从头训练")
        return
    files = [f for f in run_dir.rglob("*") if f.is_file()]
    steps = sorted(d.name for d in run_dir.glob("step_*"))
    say(f"将删除 {run_dir}：{len(files)} 个文件，{sum(f.stat().st_size for f in files) / 2**20:.0f} MB；"
        f"检查点 {steps[-1] if steps else '无'}（共 {len(steps)} 个），best {'有' if (run_dir / 'best').exists() else '无'}，日志和曲线一并删除")
    if not yes:
        if not sys.stdin.isatty():
            raise SystemExit("非交互环境下删除存档需要加 --yes")
        if input("确认删除并从头训练？[y/N] ").strip().lower() != "y":
            raise SystemExit("已取消，什么都没有删除")
    shutil.rmtree(run_dir)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", help="运行名（检查点目录 data/checkpoints/<名>/），默认取 configs/train.toml 的 run.name")
    p.add_argument("--set", action="append", default=[], metavar="字段=值", help="覆盖 TrainConfig 字段，可多次，如 --set batch_size=8")
    p.add_argument("--max-steps", type=int, help="这次运行最多训练多少步就退出（冒烟测试用）")
    p.add_argument("--rerun", action="store_true", help="删除这个运行目录里的全部存档，从头重新训练")
    p.add_argument("--yes", action="store_true", help="--rerun 时跳过确认（nohup/脚本里用）")
    a = p.parse_args()

    cfg = parse_overrides(TrainConfig.load(), a.set)
    if a.name:
        cfg = replace(cfg, name=a.name)
    run_dir = cfg.run_dir
    if a.rerun:
        wipe(run_dir, a.yes)

    import numpy as np

    from ..model.config import ModelConfig
    from ..model.denoiser import Denoiser
    from .dataset import BatchSource, Loader
    from .trainer import Trainer, setup_gpu

    setup_gpu()
    run_dir.mkdir(parents=True, exist_ok=True)
    for f in ("model.toml", "train.toml"):  # config snapshot: which hyperparameters this run used
        shutil.copy(ROOT / "configs" / f, run_dir / f"config_{f}")

    trainer = Trainer(Denoiser(ModelConfig.load()), cfg, run_dir)
    best, meta = float("inf"), trainer.resume()
    if meta:
        best = meta["extra"].get("best_val", best)
        say(f"从检查点续训：step {trainer.step}，best_val={best:.4g}")
    else:
        say(f"从头开始训练：{run_dir}")
    params = sum(int(np.prod(w.shape)) for w in trainer.model.weights)
    say(f"模型 {params / 1e6:.2f}M 参数；batch {cfg.batch_size} × {cfg.crop_seconds}s；总步数 {cfg.total_steps}；lr 峰值 {cfg.peak_lr}")

    val_src = BatchSource("val", cfg, seed=cfg.seed + 10**6)  # fixed validation set: the same batches, so evaluations are comparable
    val = [val_src.batch(i) for i in range(cfg.val_batches)]
    logger = Logger(run_dir)
    train_src = BatchSource("train", cfg)
    loader = Loader(train_src, start=trainer.step, workers=cfg.workers, prefetch=cfg.prefetch)

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

    def extra() -> dict:
        return {"best_val": best}

    bar = (tqdm(total=cfg.total_steps, initial=trainer.step, unit="step", dynamic_ncols=True, smoothing=0.05, disable=None)
           if tqdm else _NoBar())  # disable=None: switched off automatically when not on an interactive terminal
    start_step, last_saved = trainer.step, trainer.step
    window = {"n": 0, "t": time.perf_counter(), "wait": 0.0, "skipped": 0, "loss": 0.0, "mag": 0.0, "complex": 0.0, "gn": 0.0}
    streak = rollbacks = 0
    try:
        while trainer.step < cfg.total_steps and not stop["flag"] and (a.max_steps is None or trainer.step - start_step < a.max_steps):
            t = time.perf_counter()
            noisy, clean, cutoff = next(loader)
            window["wait"] += time.perf_counter() - t
            m = trainer.train_step(noisy, clean, cutoff)
            window["n"] += 1
            if m["skipped"]:
                window["skipped"] += 1
                streak += 1
                say(f"step {trainer.step}：损失/梯度出现 NaN/Inf，这一步已跳过（连续 {streak} 次）")
                if streak >= cfg.nan_patience:
                    rollbacks += 1
                    if rollbacks > cfg.max_rollbacks:
                        raise SystemExit(f"回滚超过 {cfg.max_rollbacks} 次仍然出现 NaN，停止。检查学习率或数据。")
                    if trainer.resume() is None:
                        raise SystemExit("连续出现 NaN，但没有检查点可回滚。")
                    say(f"回滚到检查点 step {trainer.step}（第 {rollbacks} 次）")
                    bar.n = trainer.step
                    loader.close()
                    loader, streak = Loader(train_src, start=trainer.step, workers=cfg.workers, prefetch=cfg.prefetch), 0
                continue
            streak = 0
            for k, kk in (("loss", "loss"), ("mag", "mag"), ("complex", "complex"), ("gn", "grad_norm")):
                window[k] += m[kk]
            step = trainer.step
            bar.update(step - bar.n)
            bar.set_postfix(loss=f"{m['loss']:.4g}", refresh=False)

            if step % cfg.log_every == 0:
                dt, n_ok = time.perf_counter() - window["t"], max(window["n"] - window["skipped"], 1)
                logger.log(step, {"loss": window["loss"] / n_ok, "mag": window["mag"] / n_ok, "complex": window["complex"] / n_ok,
                                  "grad_norm": window["gn"] / n_ok, "lr": trainer.lr, "clips_per_s": window["n"] * cfg.batch_size / dt,
                                  "data_wait": window["wait"] / dt, "skipped": window["skipped"]})
                window.update(n=0, t=time.perf_counter(), wait=0.0, skipped=0, loss=0.0, mag=0.0, complex=0.0, gn=0.0)
            if step % cfg.eval_every == 0:
                raw = trainer.evaluate(val)
                with trainer.ema_weights():
                    ema = trainer.evaluate(val)
                if ema < best:
                    best = ema
                    trainer.save(extra(), name="best")
                logger.log(step, {"val_raw": raw, "val_ema": ema, "val_best": best})
            if step % cfg.every_steps == 0:
                trainer.save(extra())
                last_saved = step
    except KeyboardInterrupt:
        say("收到 Ctrl+C，存档后退出")
    finally:
        loader.close()
        if trainer.step > last_saved:
            say(f"存档 step {trainer.step} → {trainer.save(extra())}")
        bar.close()
    say(f"结束：step {trainer.step}")


if __name__ == "__main__":
    main()
