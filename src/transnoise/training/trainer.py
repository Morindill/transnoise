"""训练核心：一步训练（含 NaN 保护和 EMA）、评估、断点存档/恢复。训练步用 tf.GradientTape（后端固定为 TensorFlow，和 train 环境一致），
模型和损失只用 keras.ops。数据由调用方提供：noisy/clean [B, T] 48kHz、cutoff_hz [B]（干净语音的带宽，用于损失掩码）。

NaN 保护：损失或梯度范数不是有限值时，这一步整个跳过（权重、优化器状态、EMA、迭代数都不动），返回 skipped=True；
连续跳过多少步后回滚到检查点由训练循环决定。
"""

from contextlib import contextmanager
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from ..model.denoiser import Denoiser
from . import checkpoint
from .config import TrainConfig
from .losses import SpectralLoss
from .optim import make_optimizer


def setup_gpu() -> None:
    """显存按需增长（默认 TF 会一口气占满）。必须在第一次使用 GPU 之前调用。"""
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)


class Trainer:
    def __init__(self, model: Denoiser, cfg: TrainConfig, run_dir: Path | None = None):
        keras.utils.set_random_seed(cfg.seed)
        self.model, self.cfg, self.run_dir = model, cfg, run_dir or cfg.run_dir
        self.loss_fn, self.optimizer = SpectralLoss(cfg, model.cfg.sr), make_optimizer(cfg)
        t = int(round(cfg.crop_seconds * model.cfg.sr / model.cfg.hop)) * model.cfg.hop  # 片段长度取 hop 的整数倍
        self.samples = t
        model(np.zeros((1, t), "float32"))  # 建好所有权重
        self.optimizer.build(model.trainable_variables)
        self.ema = [tf.Variable(keras.ops.convert_to_numpy(w), trainable=False) for w in model.weights]
        sig = [tf.TensorSpec([None, t], tf.float32), tf.TensorSpec([None, t], tf.float32), tf.TensorSpec([None], tf.float32)]
        self._train = tf.function(self._train_fn, input_signature=sig)
        self._eval = tf.function(self._eval_fn, input_signature=sig)

    def _update_ema(self):
        n = tf.cast(self.optimizer.iterations, tf.float32)
        d = tf.minimum(self.cfg.ema_decay, (1.0 + n) / (10.0 + n))
        for e, w in zip(self.ema, self.model.weights):
            e.assign(d * e + (1.0 - d) * tf.convert_to_tensor(w))

    def _train_fn(self, noisy, clean, cutoff):
        vars_ = self.model.trainable_variables
        with tf.GradientTape() as tape:
            loss, parts = self.loss_fn(clean, self.model(noisy), cutoff)
        grads = tape.gradient(loss, vars_)
        gnorm = tf.linalg.global_norm(grads)
        ok = tf.math.is_finite(loss) & tf.math.is_finite(gnorm)

        def update():
            self.optimizer.apply(grads, vars_)
            self._update_ema()
            return tf.constant(True)

        tf.cond(ok, update, lambda: tf.constant(False))
        return loss, parts, gnorm, ok

    def _eval_fn(self, noisy, clean, cutoff):
        return self.loss_fn(clean, self.model(noisy), cutoff)

    @property
    def step(self) -> int:
        return int(keras.ops.convert_to_numpy(self.optimizer.iterations))

    @property
    def lr(self) -> float:
        return float(keras.ops.convert_to_numpy(self.optimizer.learning_rate))

    def train_step(self, noisy, clean, cutoff_hz) -> dict:
        loss, parts, gnorm, ok = self._train(noisy, clean, cutoff_hz)
        return {"loss": float(loss), **{k: float(v) for k, v in parts.items()}, "grad_norm": float(gnorm),
                "skipped": not bool(ok)}

    @contextmanager
    def ema_weights(self):
        """临时把模型权重换成 EMA 权重（评估、导出用），退出时换回。"""
        backup = [keras.ops.convert_to_numpy(w) for w in self.model.weights]
        for w, e in zip(self.model.weights, self.ema):
            w.assign(e)
        try:
            yield
        finally:
            for w, b in zip(self.model.weights, backup):
                w.assign(b)

    def evaluate(self, batches) -> float:
        """batches: 可迭代，每项 (noisy, clean, cutoff_hz)；返回按条数加权的平均损失（用当前模型权重）。"""
        total = n = 0.0
        for noisy, clean, cutoff in batches:
            loss, _ = self._eval(noisy, clean, cutoff)
            total, n = total + float(loss) * len(noisy), n + len(noisy)
        return total / max(n, 1)

    def save(self, extra: dict | None = None, name: str | None = None) -> Path:
        return checkpoint.save(self.run_dir, self.step, self.model, self.optimizer, self.ema, extra, self.cfg.keep_last, name)

    def resume(self, path: Path | None = None) -> dict | None:
        """从 path（默认最新的）恢复；没有检查点返回 None。返回 meta（meta["extra"] 是当时存的训练循环状态）。"""
        path = path or checkpoint.latest(self.run_dir)
        return checkpoint.load(path, self.model, self.optimizer, self.ema) if path else None
