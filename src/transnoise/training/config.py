import tomllib
from dataclasses import dataclass
from pathlib import Path

from ..paths import DATA, ROOT


@dataclass(frozen=True)
class TrainConfig:
    """训练超参，取值全部来自 configs/train.toml（这里不设默认值，避免两处不一致）。"""

    name: str
    seed: int
    crop_seconds: float
    batch_size: int
    # loss
    gamma: float
    fft_sizes: tuple[int, ...]
    weight_mag: float
    weight_complex: float
    # optim
    peak_lr: float
    min_lr: float
    warmup_steps: int
    total_steps: int
    weight_decay: float
    beta2: float
    clip_norm: float
    # checkpoint
    every_steps: int
    keep_last: int
    # data
    p_reverb: float
    snr_min: float
    snr_max: float
    speech_db_min: float
    speech_db_max: float
    workers: int
    prefetch: int
    val_batches: int
    # loop
    log_every: int
    eval_every: int
    ema_decay: float
    nan_patience: int
    max_rollbacks: int

    @property
    def run_dir(self) -> Path:
        return DATA / "checkpoints" / self.name

    @property
    def samples(self) -> int:
        """片段长度（样本数）。取 480 的整数倍（模型 hop = 10 ms）。"""
        return int(round(self.crop_seconds * 48000 / 480)) * 480

    @classmethod
    def load(cls, path: Path = ROOT / "configs" / "train.toml") -> "TrainConfig":
        with path.open("rb") as f:
            t = tomllib.load(f)
        loss = {**t["loss"], "fft_sizes": tuple(t["loss"]["fft_sizes"])}
        return cls(name=t["run"]["name"], seed=t["run"]["seed"], crop_seconds=t["batch"]["crop_seconds"],
                   batch_size=t["batch"]["batch_size"], **loss, **t["optim"], **t["checkpoint"], **t["data"], **t["loop"])
