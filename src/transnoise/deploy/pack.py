"""把 weights.npz 打包成 C++ 用的东西（只依赖 numpy，主环境就能跑）：
  uv run python -m transnoise.deploy.pack data/deploy/<名>/weights.npz [--out-dir cpp/gen]

输出：
  weights.bin   C++ 加载的权重（float32，带名字表和模型常量，C++ 加载时逐项校验，对不上就拒绝加载）
  config.h      模型常量（constexpr）。C++ 里所有形状都来自这里，编译期写死；换模型结构要重新生成并重新编译

打包时做的折叠（数学上等价，只带来 1e-7 量级的舍入差）：
  - 每个 LayerNorm 的 γ、β 折进它后面的线性层：(x̂·γ+β)·W+b = x̂·(diag(γ)·W) + (β·W+b)，C++ 里 LN 只做归一化
  - 最后的 norm_out 同理折进 ERB 增益头和深度滤波头
矩阵保持 Keras 的 [输入, 输出] 行主序，C++ 里按"输出维向量化"的循环直接用。
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"NCWGT001"
ALIGN = 64
# 常量表的顺序：config.h 里的 CONSTS 和 weights.bin 的头用同一个顺序，C++ 加载时逐项比较
CONST_KEYS = ["sr", "fft", "hop", "nb_erb", "nb_df", "d_model", "n_layers", "n_heads", "ff_mult", "window", "lookahead", "df_order"]


def fold(gamma, beta, w, b):
    g, be, w, b = (np.asarray(a, np.float64) for a in (gamma, beta, w, b))
    return (g[:, None] * w).astype(np.float32), (be @ w + b).astype(np.float32)


def tensors(z) -> tuple[dict, list[tuple[str, np.ndarray]]]:
    info = json.loads(str(z["meta"]))
    c = info["config"]
    eps = {float(z[k]) for k in z.files if k.endswith(".eps")}
    assert len(eps) == 1, f"各 LayerNorm 的 eps 不一致：{eps}"  # C++ 里 LN_EPS 是一个常量
    out: list[tuple[str, np.ndarray]] = []

    def add(name, a):
        out.append((name, np.ascontiguousarray(a, np.float32)))

    for n in ("erb_in", "spec_in", "mix"):
        add(f"{n}.w", z[f"{n}.kernel"]), add(f"{n}.b", z[f"{n}.bias"])
    for i in range(c["n_layers"]):
        p = f"blocks.{i}"
        w, b = fold(z[f"{p}.ln1.gamma"], z[f"{p}.ln1.beta"], z[f"{p}.attn.qkv.kernel"], z[f"{p}.attn.qkv.bias"])
        add(f"b{i}.qkv.w", w), add(f"b{i}.qkv.b", b)
        add(f"b{i}.rel", z[f"{p}.attn.rel_bias"])
        add(f"b{i}.out.w", z[f"{p}.attn.out.kernel"]), add(f"b{i}.out.b", z[f"{p}.attn.out.bias"])
        w, b = fold(z[f"{p}.ln2.gamma"], z[f"{p}.ln2.beta"], z[f"{p}.ff1.kernel"], z[f"{p}.ff1.bias"])
        add(f"b{i}.ff1.w", w), add(f"b{i}.ff1.b", b)
        add(f"b{i}.ff2.w", z[f"{p}.ff2.kernel"]), add(f"b{i}.ff2.b", z[f"{p}.ff2.bias"])
    for n in ("gain", "df"):
        w, b = fold(z["norm_out.gamma"], z["norm_out.beta"], z[f"{n}.kernel"], z[f"{n}.bias"])
        add(f"{n}.w", w), add(f"{n}.b", b)
    return {"eps": eps.pop(), "info": info}, out


def write_bin(path: Path, consts: list[int], items: list[tuple[str, np.ndarray]]) -> None:
    offs, pos = [], 0
    for _, a in items:
        offs.append(pos)
        pos += a.size
    head = MAGIC + struct.pack("<IIII", 1, len(consts), len(items), 0) + struct.pack(f"<{len(consts)}i", *consts)
    for (name, a), o in zip(items, offs):
        assert len(name) < 40, name
        head += name.encode().ljust(40, b"\0") + struct.pack("<QQ", o, a.size)
    head += b"\0" * (-len(head) % ALIGN)
    path.write_bytes(head + b"".join(a.tobytes() for _, a in items))


def write_config(path: Path, c: dict, widths: list[int], eps: float, step: int) -> None:
    assert c["fft"] == 2 * c["hop"], "C++ 实现假设 fft == 2·hop"
    assert c["d_model"] % c["n_heads"] == 0 and sum(widths) == c["fft"] // 2 + 1
    consts = ", ".join(str(c[k]) for k in CONST_KEYS)
    path.write_text(f"""// 由 transnoise.deploy.pack 生成，不要手改。模型形状全部是编译期常量；weights.bin 的头里带同一组常量，加载时逐项校验。
// 来源：检查点 step {step}
#pragma once
#include <cstdint>

namespace tn {{
inline constexpr int SR = {c['sr']}, FFT = {c['fft']}, HOP = {c['hop']}, BINS = FFT / 2 + 1;
inline constexpr int NB_ERB = {c['nb_erb']}, NB_DF = {c['nb_df']}, ORDER = {c['df_order']};
inline constexpr int D = {c['d_model']}, LAYERS = {c['n_layers']}, HEADS = {c['n_heads']}, DH = D / HEADS, FF = D * {c['ff_mult']};
inline constexpr int WINDOW = {c['window']}, LA = {c['lookahead']};
inline constexpr float ERB_DB_OFFSET = {c['erb_db_offset']!r}f, ERB_DB_SCALE = {c['erb_db_scale']!r}f, LN_EPS = {eps!r}f;
inline constexpr int ERB_WIDTHS[NB_ERB] = {{{', '.join(map(str, widths))}}};
inline constexpr int32_t CONSTS[] = {{{consts}}};  // 顺序：{' '.join(CONST_KEYS)}
}}  // namespace tn
""")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("npz", type=Path)
    p.add_argument("--out-dir", type=Path, default=None, help="默认放在 npz 旁边")
    a = p.parse_args()
    out_dir = a.out_dir or a.npz.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(a.npz)
    extra, items = tensors(z)
    info = extra["info"]
    c = info["config"]
    write_bin(out_dir / "weights.bin", [c[k] for k in CONST_KEYS], items)
    write_config(out_dir / "config.h", c, info["erb_widths"], extra["eps"], info["step"])
    n = sum(a.size for _, a in items)
    print(f"{out_dir}/weights.bin：{len(items)} 个张量，{n:,} 个 float（{n * 4 / 2**20:.2f} MiB）；config.h 已生成（d={c['d_model']}，前视={c['lookahead']}）")


if __name__ == "__main__":
    main()
