"""AdamW + linear warmup -> cosine decay. The learning rate is a function of optimizer.iterations, so after restoring a
checkpoint (the optimizer state includes the iteration count) the schedule resumes where it left off."""

import keras
from keras import ops

from .config import TrainConfig


class WarmupCosine(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak: float, minimum: float, warmup: int, total: int):
        super().__init__()
        self.peak, self.minimum, self.warmup, self.total = peak, minimum, warmup, total

    def __call__(self, step):
        step = ops.cast(step, "float32")
        warm = self.peak * (step + 1.0) / self.warmup
        t = ops.clip((step - self.warmup) / max(self.total - self.warmup, 1), 0.0, 1.0)
        cos = self.minimum + 0.5 * (self.peak - self.minimum) * (1.0 + ops.cos(3.141592653589793 * t))
        return ops.where(step < self.warmup, warm, cos)


def make_optimizer(cfg: TrainConfig) -> keras.optimizers.AdamW:
    opt = keras.optimizers.AdamW(
        learning_rate=WarmupCosine(cfg.peak_lr, cfg.min_lr, cfg.warmup_steps, cfg.total_steps),
        weight_decay=cfg.weight_decay, beta_2=cfg.beta2, global_clipnorm=cfg.clip_norm)
    opt.exclude_from_weight_decay(var_names=["bias", "gamma", "beta", "rel_bias"])  # no decay on biases, LayerNorm, relative position bias
    return opt
