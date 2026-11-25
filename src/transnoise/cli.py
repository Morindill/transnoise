import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from .paths import DATA, MANIFESTS, ROOT

# Commands run inside the train environment: subcommand name -> (description, underlying command). Arguments are passed through unchanged
_TRAIN_ENV = {
    "train": ("train (resumes automatically from a checkpoint)", ("python", "-m", "transnoise.training.run")),
    "infer": ("enhance a directory of wavs with a checkpoint", ("python", "-m", "transnoise.training.infer")),
    "export": ("export a checkpoint to deployment weights", ("python", "-m", "transnoise.deploy.export")),
    "tensorboard": ("view training curves", ("tensorboard", "--logdir", str(DATA / "checkpoints"))),
}
_CHECKS = {"model": "check_model", "data": "check_data", "training": "check_training", "deploy": "check_deploy",
           "batch": "measure_batch"}  # self-check and measurement scripts under train/


def _status() -> None:
    agg = defaultdict(lambda: [0.0, 0, 0])  # hours, kept, dropped
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
        print(f"  dropped {s}: {why} x {c}")


def _add_data(sub: argparse._SubParsersAction) -> None:
    from .data_pipeline.config import CFG
    from .data_pipeline.registry import SOURCES

    d = sub.add_parser("data", help="data pipeline").add_subparsers(dest="action", required=True)
    d.add_parser("list", help="list data sources")
    for c in ("download", "stage"):
        s = d.add_parser(c)
        s.add_argument("names", nargs="+", choices=list(SOURCES), metavar="source", help="data source name: " + ", ".join(SOURCES))
        if c == "download":
            s.add_argument("--conns", type=int, default=CFG["download"]["conns"], help="concurrent connections per file")
        else:
            s.add_argument("--workers", type=int, help="number of decode/resample processes (default: from config)")
            s.add_argument("--redo", action="store_true",
                           help="clear the source's manifest, completion marker and data/staged/<source>/, then re-process (raw is left untouched)")
    d.add_parser("status", help="hours by source / language / bandwidth / split")
    d.add_parser("index", help="merge the per-source manifests into data/index/{train,val,test}.jsonl")


def _run_data(a: argparse.Namespace) -> None:
    from .data_pipeline.registry import SOURCES

    if a.action == "list":
        for s in SOURCES.values():
            print(f"{s.name:<16}{s.role:<8}{s.lang:<7}{len(s.urls):>3} files  {s.license}  {s.note}")
    elif a.action == "download":
        from .data_pipeline.download import download
        for n in a.names:
            download(SOURCES[n], a.conns)
    elif a.action == "stage":
        from .data_pipeline.stage import stage
        for n in a.names:
            stage(SOURCES[n], a.workers, a.redo)
    elif a.action == "status":
        _status()
    elif a.action == "index":
        from .data_pipeline.index import build_index
        build_index()


def _add_eval(sub: argparse._SubParsersAction) -> None:
    from .eval.config import CFG

    e = sub.add_parser("eval", help="evaluation").add_subparsers(dest="action", required=True)
    b = e.add_parser("build-testset", help="synthesize a test set from configs/eval.toml (fixed seed; must not change once generated)")
    b.add_argument("name", choices=list(CFG["testsets"]))
    b.add_argument("--force", action="store_true", help="delete and rebuild if it already exists")
    s = e.add_parser("score", help="score a directory of enhanced wavs (input: test set + directory; output: results.jsonl)")
    s.add_argument("testset")
    s.add_argument("enhanced", type=Path, help="directory containing <id>.wav files, 48 kHz mono; the test set's own noisy/ is the baseline")
    s.add_argument("--name", help="result name; default: the directory name (the parent's name when the directory is called wav)")
    s.add_argument("--workers", type=int, help="number of scoring processes")
    p = e.add_parser("report", help="compare several results (table + csv)")
    p.add_argument("testset")
    p.add_argument("names", nargs="*", help="result names to compare; default: all under the test set; the one named noisy is the baseline")
    p.add_argument("--by", default="lang,snr_db,speech_source,noise_source", help="comma-separated fields to slice by")
    p.add_argument("--metric", default="si_sdr_imp", help="metric in the sliced tables: si_sdr_imp/si_sdr/pesq/stoi/dnsmos_ovr etc.")


def _run_eval(a: argparse.Namespace) -> None:
    if a.action == "build-testset":
        from .eval.testset import build
        build(a.name, a.force)
    elif a.action == "score":
        from .eval.scorer import score
        score(a.testset, a.enhanced, a.name, a.workers)
    elif a.action == "report":
        from .eval.report import report
        report(a.testset, a.names or None, a.by.split(","), a.metric)


def _add_train_env(sub: argparse._SubParsersAction) -> None:
    """Registered only so they appear in -h; these subcommands' arguments bypass argparse and main passes them to the underlying command unchanged."""
    for name, (help_, _) in _TRAIN_ENV.items():
        sub.add_parser(name, help=help_ + " (arguments are passed to the underlying command unchanged; add -h for its usage)", add_help=False)
    sub.add_parser("check", help="checks and measurement scripts in the train environment: check {" + ",".join(_CHECKS) + "} [script arguments]", add_help=False)


def _run_train_env(argv: list[str]) -> None:
    from .train_env import exec_in_train_env

    if argv[0] != "check":
        exec_in_train_env([*_TRAIN_ENV[argv[0]][1], *argv[1:]])
    if len(argv) < 2 or argv[1] not in _CHECKS:
        raise SystemExit(f"usage: transnoise check {{{','.join(_CHECKS)}}} [script arguments]")
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
