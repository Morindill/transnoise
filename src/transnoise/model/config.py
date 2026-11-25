import tomllib
from dataclasses import dataclass
from pathlib import Path

from ..paths import ROOT


@dataclass(frozen=True)
class ModelConfig:
    """Model hyperparameters; every value comes from configs/model.toml (no defaults here, so the two cannot disagree)."""

    # frontend
    sr: int
    fft: int
    hop: int
    nb_erb: int
    min_erb_bins: int
    nb_df: int
    erb_db_offset: float
    erb_db_scale: float
    # conformer
    d_model: int
    n_layers: int
    n_heads: int
    ff_mult: int
    conv_kernel: int
    window: int
    lookahead: int
    # deep filter
    df_order: int

    @property
    def bins(self) -> int:
        return self.fft // 2 + 1

    @property
    def bulk_delay_ms(self) -> float:
        """Overall delay of the streaming output relative to the input (what the evaluation's cross-correlation measures): (lookahead + 1) frames."""
        return (self.lookahead + 1) * self.hop / self.sr * 1000

    @property
    def latency_ms(self) -> float:
        """Worst-case algorithmic delay: a sample waits at most (lookahead + 2) frames = STFT window + lookahead before it can be output."""
        return (self.lookahead + 2) * self.hop / self.sr * 1000

    @classmethod
    def load(cls, path: Path = ROOT / "configs" / "model.toml") -> "ModelConfig":
        with path.open("rb") as f:
            t = tomllib.load(f)
        return cls(**t["frontend"], **t["conformer"], df_order=t["deep_filter"]["order"])

    @classmethod
    def from_checkpoint(cls, ckpt: Path) -> "ModelConfig":
        """The model hyperparameter snapshot saved in the run directory at training time; it matches the checkpoint's weight shapes."""
        return cls.load(ckpt.parent / "config_model.toml")
