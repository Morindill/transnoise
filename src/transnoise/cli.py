import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from .paths import DATA, MANIFESTS, ROOT

# 在 train 环境里运行的命令：子命令名 → (说明, 底层命令)。参数原样传给底层命令
_TRAIN_ENV = {
    "train": ("训练（有检查点自动续训）", ("python", "-m", "transnoise.training.run")),
    "infer": ("用检查点增强一个目录的 wav", ("python", "-m", "transnoise.training.infer")),
    "export": ("把检查点导出成部署权重", ("python", "-m", "transnoise.deploy.export")),
    "tensorboard": ("看训练曲线", ("tensorboard", "--logdir", str(DATA / "checkpoints"))),
}
_CHECKS = {"model": "check_model", "data": "check_data", "training": "check_training", "deploy": "check_deploy",
           "batch": "measure_batch"}  # train/ 下的自检和实测脚本


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
    from .data_pipeline.registry import SOURCES

    d = sub.add_parser("data", help="数据管线").add_subparsers(dest="action", required=True)
    d.add_parser("list", help="列出数据源")
    for c in ("download", "stage"):
        s = d.add_parser(c)
        s.add_argument("names", nargs="+", choices=list(SOURCES), metavar="源", help="数据源名: " + ", ".join(SOURCES))
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
    from .data_pipeline.registry import SOURCES

    if a.action == "list":
        for s in SOURCES.values():
            print(f"{s.name:<16}{s.role:<8}{s.lang:<7}{len(s.urls):>3} 个文件  {s.license}  {s.note}")
    elif a.action == "download":
        from .data_pipeline.download import download
        for n in a.names:
            download(SOURCES[n], a.conns, a.mirror)
    elif a.action == "stage":
        from .data_pipeline.stage import stage
        for n in a.names:
            stage(SOURCES[n], a.workers, a.limit, a.redo)
    elif a.action == "status":
        _status()
    elif a.action == "index":
        from .data_pipeline.index import build_index
        build_index()


def _add_eval(sub: argparse._SubParsersAction) -> None:
    from .eval.config import CFG

    e = sub.add_parser("eval", help="评测").add_subparsers(dest="action", required=True)
    b = e.add_parser("build-testset", help="按 configs/eval.toml 合成测试集（固定种子，生成后不应改变）")
    b.add_argument("name", choices=list(CFG["testsets"]))
    b.add_argument("--n", type=int, help="覆盖每种语言的条数（试跑用）")
    b.add_argument("--as", dest="out_name", help="输出到另一个名字（试跑用，避免覆盖正式测试集）")
    b.add_argument("--force", action="store_true", help="已存在时删除重建")
    s = e.add_parser("score", help="给一个增强后的 wav 目录打分（输入：测试集 + 目录；输出：results.jsonl）")
    s.add_argument("testset")
    s.add_argument("enhanced", type=Path, help="含 <id>.wav 的目录，48kHz 单声道；testset 自己的 noisy/ 就是基线")
    s.add_argument("--name", help="结果名字，默认取目录名（目录叫 wav 时取上一级目录名）")
    s.add_argument("--workers", type=int, help="打分进程数")
    p = e.add_parser("report", help="汇总几个结果（表格 + csv）")
    p.add_argument("testset")
    p.add_argument("names", nargs="*", help="要对比的结果名字，默认该测试集下全部；名为 noisy 的是基线")
    p.add_argument("--by", default="lang,snr_db,speech_source,noise_source", help="逗号分隔的切片字段")
    p.add_argument("--metric", default="si_sdr_imp", help="切片表里的指标：si_sdr_imp/si_sdr/pesq/stoi/dnsmos_ovr 等")


def _run_eval(a: argparse.Namespace) -> None:
    if a.action == "build-testset":
        from .eval.testset import build
        build(a.name, a.n, a.out_name, a.force)
    elif a.action == "score":
        from .eval.scorer import score
        score(a.testset, a.enhanced, a.name, a.workers)
    elif a.action == "report":
        from .eval.report import report
        report(a.testset, a.names or None, a.by.split(","), a.metric)


def _add_train_env(sub: argparse._SubParsersAction) -> None:
    """只为 -h 的列表登记；这些子命令的参数不经过 argparse，在 main 里原样转给底层命令。"""
    for name, (help_, _) in _TRAIN_ENV.items():
        sub.add_parser(name, help=help_ + "（参数原样传给底层命令，加 -h 看它的用法）", add_help=False)
    sub.add_parser("check", help="train 环境里的自检和实测脚本：check {" + ",".join(_CHECKS) + "} [脚本参数]", add_help=False)


def _run_train_env(argv: list[str]) -> None:
    from .train_env import exec_in_train_env

    if argv[0] != "check":
        exec_in_train_env([*_TRAIN_ENV[argv[0]][1], *argv[1:]])
    if len(argv) < 2 or argv[1] not in _CHECKS:
        raise SystemExit(f"用法: transnoise check {{{','.join(_CHECKS)}}} [脚本参数]")
    exec_in_train_env(["python", str(ROOT / "train" / f"{_CHECKS[argv[1]]}.py"), *argv[2:]])


def main() -> None:
    argv = sys.argv[1:]
    if argv and (argv[0] in _TRAIN_ENV or argv[0] == "check"):
        _run_train_env(argv)
    p = argparse.ArgumentParser(prog="transnoise")
    sub = p.add_subparsers(dest="cmd", required=True)
    _add_data(sub)
    _add_eval(sub)
    _add_train_env(sub)
    a = p.parse_args()
    {"data": _run_data, "eval": _run_eval}[a.cmd](a)
