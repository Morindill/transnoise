import tomllib

from ..paths import ROOT

with (ROOT / "configs" / "data_pipeline.toml").open("rb") as f:
    CFG = tomllib.load(f)
