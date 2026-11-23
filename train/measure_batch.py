"""Measure the GPU memory peak and speed of one training step (forward + backward + optimizer update) for different segment lengths x batch sizes, to choose batch_size in configs/train.toml.
Usage: uv run transnoise check batch [--crops 3 4 5] [--batches 8 16 24 32 48 64] [--budget-gb 8]
Random data, the dataset is not read. The memory peak is the largest amount the TF allocator actually allocated; batches beyond --budget-gb are not measured."""

import argparse
import os
import time
from dataclasses import replace

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from transnoise.training.trainer import Trainer, setup_gpu  # noqa: E402

setup_gpu()
from transnoise.model.config import ModelConfig  # noqa: E402
from transnoise.model.denoiser import Denoiser  # noqa: E402
from transnoise.training.config import TrainConfig  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--crops", type=float, nargs="+", default=[3.0, 4.0, 5.0])
p.add_argument("--batches", type=int, nargs="+", default=[8, 16, 24, 32, 48, 64])
p.add_argument("--budget-gb", type=float, default=8.0)
a = p.parse_args()

mcfg = ModelConfig.load()
total = tf.config.experimental.get_memory_info("GPU:0")  # triggers GPU initialization
print(f"{'片段(s)':>7} {'batch':>6} {'峰值显存':>9} {'ms/步':>8} {'条/秒':>8}  备注")
for crop in a.crops:
    tcfg = replace(TrainConfig.load(), crop_seconds=crop)
    tr = Trainer(Denoiser(mcfg), tcfg)
    rng = np.random.default_rng(0)
    for b in a.batches:
        x = rng.standard_normal((b, tr.samples)).astype("float32") * 0.1
        y = x + 0.05 * rng.standard_normal(x.shape).astype("float32")
        cutoff = np.full(b, 24000.0, "float32")
        tf.config.experimental.reset_memory_stats("GPU:0")
        try:
            tr.train_step(y, x, cutoff)  # the first step includes compilation
            t0 = time.perf_counter()
            for _ in range(5):
                tr.train_step(y, x, cutoff)
            dt = (time.perf_counter() - t0) / 5
        except tf.errors.ResourceExhaustedError:
            print(f"{crop:>7.1f} {b:>6} {'OOM':>9}")
            break
        peak = tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2**30
        note = "超出预算，停止" if peak > a.budget_gb else ""
        print(f"{crop:>7.1f} {b:>6} {peak:>7.2f}GB {dt * 1000:>8.0f} {b / dt:>8.1f}  {note}")
        if note:
            break
