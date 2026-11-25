"""Training core: one training step (with NaN protection and EMA), evaluation, checkpoint save/restore. The training step uses
tf.GradientTape (the backend is fixed to TensorFlow, matching the train environment); the model and loss use keras.ops only.
Data is supplied by the caller: noisy/clean [B, T] at 48 kHz, cutoff_hz [B] (bandwidth of the clean speech, used for the loss mask).

NaN protection: when the loss or gradient norm is not finite, the whole step is skipped (weights, optimizer state, EMA and
iteration count are untouched) and skipped=True is returned; how many consecutive skips trigger a rollback to a checkpoint is decided by the training loop.
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
    """Grow GPU memory on demand (TF grabs it all at once by default). Must be called before the GPU is first used."""
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)


class Trainer:
    def __init__(self, model: Denoiser, cfg: TrainConfig, run_dir: Path | None = None):
        keras.utils.set_random_seed(cfg.seed)
        self.model, self.cfg, self.run_dir = model, cfg, run_dir or cfg.run_dir
        self.loss_fn, self.optimizer = SpectralLoss(cfg, model.cfg.sr), make_optimizer(cfg)
        self.samples = t = cfg.samples  # segment length: a multiple of hop, the same value the data loader uses
        model(np.zeros((1, t), "float32"))  # build all weights
        self.optimizer.build(model.trainable_variables)
        self.ema = [tf.Variable(keras.ops.convert_to_numpy(w), trainable=False) for w in model.weights]
        sig = [tf.TensorSpec([None, t], tf.float32), tf.TensorSpec([None, t], tf.float32), tf.TensorSpec([None], tf.float32)]
        # XLA-compiled: it fuses the many small elementwise operations of the model and the loss (about 4x faster than eager graphs on the same GPU)
        self._train = tf.function(self._train_fn, input_signature=sig, jit_compile=True)
        self._eval = tf.function(self._eval_fn, input_signature=sig, jit_compile=True)

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
        """Temporarily swap the model weights for the EMA weights (for evaluation and export); swapped back on exit."""
        backup = [keras.ops.convert_to_numpy(w) for w in self.model.weights]
        for w, e in zip(self.model.weights, self.ema):
            w.assign(e)
        try:
            yield
        finally:
            for w, b in zip(self.model.weights, backup):
                w.assign(b)

    def evaluate(self, batches) -> float:
        """batches: an iterable of (noisy, clean, cutoff_hz); returns the mean loss weighted by clip count (with the current model weights)."""
        total = n = 0.0
        for noisy, clean, cutoff in batches:
            loss, _ = self._eval(noisy, clean, cutoff)
            total, n = total + float(loss) * len(noisy), n + len(noisy)
        return total / max(n, 1)

    def save(self, extra: dict | None = None, name: str | None = None) -> Path:
        return checkpoint.save(self.run_dir, self.step, self.model, self.optimizer, self.ema, extra, self.cfg.keep_last, name)

    def resume(self, path: Path | None = None) -> dict | None:
        """Restore from path (the latest by default); returns None when there is no checkpoint. Returns meta (meta["extra"] is the training-loop state stored then)."""
        path = path or checkpoint.latest(self.run_dir)
        return checkpoint.load(path, self.model, self.optimizer, self.ema) if path else None
