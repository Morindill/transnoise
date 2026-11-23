"""流式推理的 numpy 参考实现（带 KV 缓存）：每收到 hop 个样本只算新的一帧，不重算历史。只依赖 numpy，不依赖 Keras。
它是 C++ 实现的"黄金参照"：C++ 的每一层都要和这里（以及 trace）对拍；而它自己和 Keras 离线前向、stream.py 对拍（train/check_deploy.py）。

流水线（lookahead = la，一帧编号 n 到达时）：
  帧 n：特征 → 嵌入 → 第 0 层的 LN/QKV，把 K/V 放进缓存，把 (嵌入, Q) 放进待处理队列
  帧 u = n - la 的第 0 层注意力现在才有它需要的全部键（u-window … u+la = n），算完第 0 层；
  再依次算第 1、2 层（严格因果：键 = u-window … u，K/V 在这里生成并缓存）、输出头、ERB 增益、深度滤波、逆 STFT 重叠相加。
这样每帧每层的计算量是常数，和 stream.py 的"重算最近 keep 帧"逐样本等价（只差浮点误差）。
"""

import json
import math

import numpy as np

try:
    from scipy.special import erf as _erf
except ImportError:  # 没装 scipy 时退回 math.erf（慢一点，结果一样）
    _erf = np.vectorize(math.erf, otypes=[np.float32])


def gelu(x: np.ndarray) -> np.ndarray:  # Keras Dense(activation="gelu") 默认是精确的 erf 形式
    return (0.5 * x * (1.0 + _erf(x / np.float32(math.sqrt(2.0))))).astype(np.float32)


def layer_norm(x, gamma, beta, eps):
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps) * gamma + beta).astype(np.float32)


class _Ring:
    """定长环形缓冲：push 新的一行；last(m) 按时间从旧到新取最近 m 行。"""

    def __init__(self, cap: int, *shape: int):
        self.buf, self.cap, self.n = np.zeros((cap, *shape), np.float32), cap, 0

    def push(self, x) -> None:
        self.buf[self.n % self.cap] = x
        self.n += 1

    def last(self, m: int) -> np.ndarray:
        return self.buf[(self.n - m + np.arange(m)) % self.cap]


class RefStream:
    def __init__(self, weights, trace: bool = False):
        w = {k: weights[k] for k in weights.files} if hasattr(weights, "files") else dict(weights)
        info = json.loads(str(w.pop("meta")))
        self.cfg, self.widths = info["config"], np.asarray(info["erb_widths"])
        self.win = w.pop("window").astype(np.float32)
        self.w = w
        c = self.cfg
        self.d, self.h = c["d_model"], c["n_heads"]
        self.dh = self.d // self.h
        self.la, self.W, self.order, self.hop = c["lookahead"], c["window"], c["df_order"], c["hop"]
        assert self.order > self.la, "df_order 必须 > lookahead"
        self.starts = np.concatenate([[0], np.cumsum(self.widths)[:-1]])
        self.trace_on = trace
        self.reset()

    def reset(self) -> None:
        c = self.cfg
        self.prev = np.zeros(self.hop, np.float32)  # 第 0 帧前面的补零
        self.tail = np.zeros(self.hop, np.float32)  # 重叠相加还没完成的后半帧
        self.t = -1  # 已收到的帧编号
        self.hist = np.zeros((self.order, c["fft"] // 2 + 1), np.complex64)  # hist[k] = 第 n-k 帧的频谱（负编号的帧是 0）
        self.pending: list[tuple[np.ndarray, np.ndarray]] = []  # 第 0 层还没算的 (嵌入, Q)
        la_of = [self.la if i == 0 else 0 for i in range(c["n_layers"])]
        self.la_of = la_of
        self.k = [_Ring(self.W + la + 1, self.h, self.dh) for la in la_of]
        self.v = [_Ring(self.W + la + 1, self.h, self.dh) for la in la_of]
        self.trace = []

    # ---- 小积木 ----
    def lin(self, x, name):
        return (x @ self.w[f"{name}.kernel"] + self.w[f"{name}.bias"]).astype(np.float32)

    def ln(self, x, name):
        return layer_norm(x, self.w[f"{name}.gamma"], self.w[f"{name}.beta"], self.w[f"{name}.eps"])

    def qkv(self, i, x):
        """LN → QKV 投影，K/V 入缓存，返回 Q [h, dh]。"""
        q, k, v = np.split(self.lin(self.ln(x, f"blocks.{i}.ln1"), f"blocks.{i}.attn.qkv"), 3)
        self.k[i].push(k.reshape(self.h, self.dh))
        self.v[i].push(v.reshape(self.h, self.dh))
        return q.reshape(self.h, self.dh)

    def attend(self, i, x, q, u):
        """第 i 层对帧 u 的注意力 + 前馈（K/V 缓存里最新的一帧就是 u + 该层前视）。"""
        la, W = self.la_of[i], self.W
        m = min(u, W) + la + 1  # 存在的键：帧 u-min(u,W) … u+la
        K, V = self.k[i].last(m), self.v[i].last(m)
        bias = self.w[f"blocks.{i}.attn.rel_bias"][:, W + la - m + 1: W + la + 1]  # 偏移 -(m-1-la) … +la
        s = np.einsum("hd,mhd->hm", q, K) / np.float32(math.sqrt(self.dh)) + bias
        s = np.exp(s - s.max(-1, keepdims=True))
        ctx = np.einsum("hm,mhd->hd", s / s.sum(-1, keepdims=True), V).reshape(self.d)
        x = x + self.lin(ctx, f"blocks.{i}.attn.out")
        return x + self.lin(gelu(self.lin(self.ln(x, f"blocks.{i}.ln2"), f"blocks.{i}.ff1")), f"blocks.{i}.ff2")

    def embed(self, spec: np.ndarray) -> np.ndarray:
        c = self.cfg
        re, im = spec.real.astype(np.float32), spec.imag.astype(np.float32)
        erb = np.add.reduceat(re * re + im * im, self.starts) / self.widths  # 每个频带的平均功率
        feat_erb = ((10.0 * np.log10(erb + 1e-10) + c["erb_db_offset"]) / c["erb_db_scale"]).astype(np.float32)
        lr, li = re[: c["nb_df"]], im[: c["nb_df"]]
        mag = np.sqrt(lr * lr + li * li + 1e-12).mean()
        feat_spec = np.concatenate([lr / (mag + 1e-6), li / (mag + 1e-6)]).astype(np.float32)
        h = np.concatenate([gelu(self.lin(feat_erb, "erb_in")), gelu(self.lin(feat_spec, "spec_in"))])
        return self.lin(h, "mix")

    # ---- 流式接口 ----
    def push(self, block: np.ndarray) -> np.ndarray | None:
        """喂入 hop 个样本；返回 hop 个输出样本，预热阶段（前 la+1 次）返回 None。"""
        c = self.cfg
        assert block.shape == (self.hop,)
        spec = np.fft.rfft(np.concatenate([self.prev, block]).astype(np.float32) * self.win).astype(np.complex64)
        self.prev = block.astype(np.float32)
        self.hist = np.roll(self.hist, 1, axis=0)
        self.hist[0] = spec
        self.t += 1
        emb = self.embed(spec)
        self.pending.append((emb, self.qkv(0, emb)))
        u = self.t - self.la
        if u < 0:
            return None
        x, q = self.pending.pop(0)
        emb_u = x  # 帧 u 的嵌入（emb 是刚到的帧 n 的）
        layers = []
        x = self.attend(0, x, q, u)
        layers.append(x)
        for i in range(1, c["n_layers"]):
            x = self.attend(i, x, self.qkv(i, x), u)
            layers.append(x)
        h = self.ln(x, "norm_out")
        gains = (1.0 / (1.0 + np.exp(-self.lin(h, "gain")))).astype(np.float32)
        mask = np.repeat(gains, self.widths)
        out = self.hist[self.la] * mask  # 第 u 帧 = n - la 帧的频谱
        coefs = None
        if "df.kernel" in self.w:
            coefs = np.tanh(self.lin(h, "df")).reshape(c["nb_df"], self.order, 2)
            wc = coefs[..., 0] + 1j * coefs[..., 1]  # [nb_df, order]
            # Y[f] = g[f] · Σ_k W[f,k] · X[u+la-k][f]，u+la-k = n-k，即 hist[k]；g 是该频点所在频带的增益
            out[: c["nb_df"]] = (wc.T * self.hist[: self.order, : c["nb_df"]]).sum(0) * mask[: c["nb_df"]]
        if self.trace_on:
            self.trace.append({"u": u, "emb": emb_u, "layers": layers, "h": h, "gains": gains, "coefs": coefs,
                               "spec_out": out.copy()})
        seg = np.fft.irfft(out, c["fft"]).astype(np.float32) * self.win
        y = self.tail + seg[: self.hop]
        self.tail = seg[self.hop:]
        return None if u == 0 else y  # u=0 对应第 0 帧前面的补零区，丢掉

    def process(self, x: np.ndarray) -> np.ndarray:
        """整段喂入（长度是 hop 的整数倍）；返回已输出的样本，长度 = len(x) - (la+1)·hop，和离线输出的前这么多个样本对齐。"""
        self.reset()
        outs = [self.push(x[i: i + self.hop]) for i in range(0, len(x), self.hop)]
        return np.concatenate([o for o in outs if o is not None])
