import argparse
import json
from collections import defaultdict

from .paths import MANIFESTS


def _status() -> None:
    agg = defaultdict(lambda: [0.0, 0, 0])  # 小时, 保留数, 丢弃数
    drops = defaultdict(int)
    for mf in sorted(MANIFESTS.glob("*.jsonl")):
        for line in mf.open():
            r = json.loads(line)
            a = agg[(r["source"], r["lang"], r["bw_tier"] or "-", r["split"])]
            if r["keep"]:
                a[0] += r["duration"] / 3600
                a[1] += 1
            else:
                a[2] += 1
                drops[(r["source"], r["drop_reason"])] += 1
    print(f"{'source':<14}{'lang':<7}{'tier':<5}{'split':<7}{'hours':>9}{'kept':>9}{'dropped':>9}")
    for (s, lg, t, sp), (h, k, d) in sorted(agg.items()):
        print(f"{s:<14}{lg:<7}{t:<5}{sp:<7}{h:>9.1f}{k:>9}{d:>9}")
    for (s, why), c in sorted(drops.items()):
        print(f"  丢弃 {s}: {why} x {c}")


def _add_data(sub: argparse._SubParsersAction) -> None:
    from .data_pipeline.config import CFG
    from .data_pipeline.registry import SETS

    d = sub.add_parser("data", help="数据管线").add_subparsers(dest="action", required=True)
    d.add_parser("list", help="列出数据源")
    for c in ("download", "stage"):
        s = d.add_parser(c)
        s.add_argument("names", nargs="+", help="数据源名或集合: " + ", ".join(SETS))
        if c == "download":
            s.add_argument("--conns", type=int, default=CFG["download"]["conns"], help="单个文件的并发连接数")
            s.add_argument("--mirror", default="",
                           help="OpenSLR 镜像：" + ", ".join(CFG["download"]["mirrors"]) + "，或直接给主机名")
        else:
            s.add_argument("--workers", type=int, help="解码/重采样进程数，默认见配置")
            s.add_argument("--limit", type=int, help="只处理前 N 条，用于试跑")
            s.add_argument("--redo", action="store_true",
                           help="清空该源的 manifest、完成标记和 data/staged/<源>/ 后重新处理（raw 不动）")
    d.add_parser("status", help="按来源/语言/带宽/切分统计小时数")
    d.add_parser("index", help="把各源 manifest 汇总成 data/index/{train,val,test}.jsonl")


def _run_data(a: argparse.Namespace) -> None:
    from .data_pipeline.registry import SOURCES, resolve

    if a.action == "list":
        for s in SOURCES.values():
            print(f"{s.name:<16}{s.role:<8}{s.lang:<7}{len(s.urls):>3} 个文件  {s.license}  {s.note}")
    elif a.action == "download":
        from .data_pipeline.download import download
        for s in resolve(a.names):
            download(s, a.conns, a.mirror)
    elif a.action == "stage":
        from .data_pipeline.stage import stage
        for s in resolve(a.names):
            stage(s, a.workers, a.limit, a.redo)
    elif a.action == "status":
        _status()
    elif a.action == "index":
        from .data_pipeline.index import build_index
        build_index()


def main() -> None:
    p = argparse.ArgumentParser(prog="noise-cancel")
    sub = p.add_subparsers(dest="cmd", required=True)
    _add_data(sub)
    a = p.parse_args()
    if a.cmd == "data":
        _run_data(a)
