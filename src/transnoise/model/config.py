import tomllib
from dataclasses import dataclass
from pathlib import Path

from ..paths import ROOT


@dataclass(frozen=True)
class ModelConfig:
    """模型超参，取值全部来自 configs/model.toml（这里不设默认值，避免两处不一致）。"""

    # frontend
    sr: int
    fft: int
    hop: int
    nb_erb: int
    min_erb_bins: int
    nb_df: int
    erb_db_offset: float
    erb_db_scale: float
    # transformer
    d_model: int
    n_layers: int
    n_heads: int
    ff_mult: int
    window: int
    lookahead: int
    # deep filter
    df_order: int

    @property
    def bins(self) -> int:
        return self.fft // 2 + 1

    @property
    def bulk_delay_ms(self) -> float:
        """流式输出相对输入的整体延迟（评测里互相关测到的就是它）：(前视 + 1) 帧。"""
        return (self.lookahead + 1) * self.hop / self.sr * 1000

    @property
    def latency_ms(self) -> float:
        """最坏算法延迟：一个样本从进入到能输出，最多等 (前视 + 2) 帧 = STFT 窗 + 前视。"""
        return (self.lookahead + 2) * self.hop / self.sr * 1000

    @classmethod
    def load(cls, path: Path = ROOT / "configs" / "model.toml") -> "ModelConfig":
        with path.open("rb") as f:
            t = tomllib.load(f)
        return cls(**t["frontend"], **t["transformer"], df_order=t["deep_filter"]["order"])
