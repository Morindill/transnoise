import json
from collections import defaultdict

from ..paths import INDEX, MANIFESTS

SPLITS = ("train", "val", "test")
KEYS = ("id", "path", "source", "role", "lang", "speaker", "duration", "orig_sr", "bandwidth_hz", "bw_tier")


def build_index() -> None:
    """把各数据源的 manifest 汇总成 index/{train,val,test}.jsonl（只含 keep 的行）。可随时重建。"""
    INDEX.mkdir(parents=True, exist_ok=True)
    tmp = {s: INDEX / f"{s}.jsonl.tmp" for s in SPLITS}
    hours: dict = defaultdict(float)
    count: dict = defaultdict(int)
    outs = {s: tmp[s].open("w") for s in SPLITS}
    try:
        for mf in sorted(MANIFESTS.glob("*.jsonl")):
            for line in mf.open():
                r = json.loads(line)
                if r["keep"]:
                    outs[r["split"]].write(json.dumps({k: r[k] for k in KEYS}, ensure_ascii=False) + "\n")
                    key = (r["split"], r["role"], r["lang"])
                    hours[key] += r["duration"] / 3600
                    count[key] += 1
    finally:
        for f in outs.values():
            f.close()
    for s in SPLITS:
        tmp[s].replace(INDEX / f"{s}.jsonl")

    print(f"{'split':<7}{'role':<8}{'lang':<7}{'条数':>9}{'小时':>9}")
    for key in sorted(count, key=lambda k: (SPLITS.index(k[0]), k[1], k[2])):
        print(f"{key[0]:<7}{key[1]:<8}{key[2]:<7}{count[key]:>9}{hours[key]:>9.1f}")
    print(f"索引写入 {INDEX}")
