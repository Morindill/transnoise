import tomllib

from ..paths import ROOT

with (ROOT / "configs" / "eval.toml").open("rb") as f:
    CFG = tomllib.load(f)
