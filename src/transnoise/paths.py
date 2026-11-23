from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

RAW = DATA / "raw"  # original downloads, never modified
STAGED = DATA / "staged"  # 48 kHz mono FLAC (RIRs are float32 WAV)
MANIFESTS = DATA / "manifests"  # one jsonl per source (stage's resume record, includes dropped rows)
INDEX = DATA / "index"  # merged train/val/test.jsonl, kept rows only; training and evaluation read from here
EVAL = DATA / "eval"  # evaluation: assets/ (DNSMOS model), testsets/<name>/, runs/<testset>/<model>/, reports/

SR = 48000
