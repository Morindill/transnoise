import tomllib
from dataclasses import dataclass
from pathlib import Path

from ..paths import DATA, ROOT


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters; every value comes from configs/train.toml (no defaults here, so the two cannot disagree)."""

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
    p_noise_only: float
    p_clean: float
    p_silence: float
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
        """Segment length in samples, rounded to a multiple of 480 (the model hop = 10 ms)."""
        return int(round(self.crop_seconds * 48000 / 480)) * 480

    @classmethod
    def load(cls, path: Path = ROOT / "configs" / "train.toml") -> "TrainConfig":
        with path.open("rb") as f:
            t = tomllib.load(f)
        loss = {**t["loss"], "fft_sizes": tuple(t["loss"]["fft_sizes"])}
        return cls(name=t["run"]["name"], seed=t["run"]["seed"], crop_seconds=t["batch"]["crop_seconds"],
                   batch_size=t["batch"]["batch_size"], **loss, **t["optim"], **t["checkpoint"], **t["data"], **t["loop"])
