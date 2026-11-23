"""Summary: read runs/<testset>/<name>/results.jsonl and compare the given names (all by default).
Convention: the result named noisy (the test set's noisy/ scored as if it were an enhanced directory) is the baseline for dSI-SDR."""

import csv
import json

import numpy as np

from ..paths import EVAL, SR

METRICS = [("si_sdr_imp", "ΔSI-SDR"), ("si_sdr", "SI-SDR"), ("pesq", "PESQ"), ("stoi", "STOI"),
           ("dnsmos_sig", "SIG"), ("dnsmos_bak", "BAK"), ("dnsmos_ovr", "OVRL")]
BASELINE = "noisy"
SELF_CHECK = "clean"  # the reference scored against itself: zero error, so SI-SDR only reflects float precision (about 145 dB), not quality; excluded from comparison


def _mean_ci(vals) -> tuple[float, float, int]:
    v = np.array([x for x in vals if x == x], dtype=float)  # drop NaN
    if not len(v):
        return float("nan"), float("nan"), 0
    return float(v.mean()), float(1.96 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0, len(v)


def _load(testset: str, names: list[str] | None) -> dict[str, list[dict]]:
    base = EVAL / "runs" / testset
    found = [p.name for p in sorted(base.iterdir()) if (p / "results.jsonl").exists()] if base.exists() else []
    names = names or found
    if missing := [n for n in names if n not in found]:
        raise SystemExit(f"{base} 下没有 {missing} 的结果（先 eval score）；现有: {found}")
    rows = {n: [json.loads(line) for line in (base / n / "results.jsonl").open()] for n in names}
    if BASELINE in found:
        ref = {r["id"]: r["si_sdr"] for r in (json.loads(line) for line in (base / BASELINE / "results.jsonl").open())}
        for rs in rows.values():
            for r in rs:
                r["si_sdr_imp"] = r["si_sdr"] - ref[r["id"]]
    return rows


def _table(header: list[str], body: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
                     + ["| " + " | ".join(r) + " |" for r in body])


def _overall(rows: dict) -> tuple[str, list[dict]]:
    cols = [(k, n) for k, n in METRICS if any(k in r for rs in rows.values() for r in rs)]
    body, summary = [], []
    for name, rs in rows.items():
        stats = {k: _mean_ci([r[k] for r in rs if k in r]) for k, _ in cols}
        delay = float(np.median([r["delay_samples"] for r in rs])) / SR * 1000
        # no valid values (n=0, e.g. pesq was not installed at the time): show a dash, not nan±nan
        cells = [("—" if (name == SELF_CHECK and k in ("si_sdr", "si_sdr_imp")) or stats[k][2] == 0 else f"{stats[k][0]:.2f}±{stats[k][1]:.2f}")
                 for k, _ in cols]
        body.append([name, str(len(rs))] + cells + [f"{delay:.1f}"])
        summary.append({"name": name, "n": len(rs), "delay_ms": round(delay, 2),
                        **{k: round(stats[k][0], 4) for k, _ in cols}, **{k + "_ci": round(stats[k][1], 4) for k, _ in cols}})
    return _table(["名字", "n"] + [n for _, n in cols] + ["延迟 ms"], body), summary


def _sliced(rows: dict, by: str, metric: str) -> str:
    groups = sorted({r[by] for rs in rows.values() for r in rs}, key=lambda v: (isinstance(v, str), v))
    body = []
    for g in groups:
        cells = [f"{_mean_ci([r[metric] for r in rs if r[by] == g and metric in r])[0]:.2f}" for rs in rows.values()]
        body.append([str(g), str(sum(1 for r in next(iter(rows.values())) if r[by] == g))] + cells)
    return _table([by, "n"] + list(rows), body)


def report(testset: str, names: list[str] | None, by: list[str], metric: str) -> None:
    rows = _load(testset, names)
    if not rows:
        raise SystemExit(f"{EVAL / 'runs' / testset} 下没有结果，先 eval score")
    table, summary = _overall(rows)
    label = dict(METRICS).get(metric, metric)
    parts = [f"# 评测报告：{testset}", "", "均值±95% 置信区间。ΔSI-SDR = 相对 `noisy`（带噪输入本身）的改善，没有 noisy 结果就不显示。"
             "延迟是对齐时用互相关测得的中位数。`clean` 是自检（参考对自身），SI-SDR 不显示，其余指标是各指标的上界参照。", "", "## 总体", table]
    for b in by:
        sl = {n: rs for n, rs in rows.items() if n != SELF_CHECK or metric not in ("si_sdr", "si_sdr_imp")}
        parts += ["", f"## 按 {b} 切片（{label} 均值）", _sliced(sl, b, metric)]
    text = "\n".join(parts)
    out = EVAL / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{testset}.md").write_text(text + "\n")
    with (out / f"{testset}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(text)
    print(f"\n已保存到 {out}/{testset}.md / .csv")
