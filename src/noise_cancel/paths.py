import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("NC_DATA", ROOT / "data"))

RAW = DATA / "raw"  # 原始下载，不改动
STAGED = DATA / "staged"  # 48kHz 单声道 FLAC（RIR 为 float32 WAV）
MANIFESTS = DATA / "manifests"  # 每个数据源一个 jsonl（stage 的断点记录，含被丢弃的行）
INDEX = DATA / "index"  # 汇总索引 train/val/test.jsonl，只含保留的行，训练和评测读这里

SR = 48000
