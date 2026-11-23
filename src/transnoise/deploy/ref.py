"""Numpy reference of streaming inference (with a KV cache): every hop samples received compute only the new frame, never recomputing history.
Depends on numpy only, not on Keras. It is the golden reference of the C++ implementation: every C++ layer is cross-checked against it (and its trace),
while it is itself cross-checked against the Keras offline forward (train/check_deploy.py).

Pipeline (lookahead = la, when frame n arrives):
  frame n: features -> embedding -> layer 0's LN/QKV; K/V go into the cache and (embedding, Q) into the pending queue
  frame u = n - la now has all the keys its layer-0 attention needs (u-window ... u+la = n), so layer 0 is finished;
  then layers 1 and 2 (strictly causal: keys = u-window ... u, K/V are produced and cached here), the output heads, ERB gains, deep filter and inverse-STFT overlap-add.
This way the cost per frame and layer is constant, and the result equals the offline forward sample for sample (up to floating-point error).
"""

import json
import math

import numpy as np

_erf = np.vectorize(math.erf, otypes=[np.float32])


def gelu(x: np.ndarray) -> np.ndarray:  # Keras Dense(activation="gelu") defaults to the exact erf form
    return (0.5 * x * (1.0 + _erf(x / np.float32(math.sqrt(2.0))))).astype(np.float32)


def layer_norm(x, gamma, beta, eps):
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps) * gamma + beta).astype(np.float32)


class _Ring:
    """Fixed-length ring buffer: push appends a new row; last(m) returns the most recent m rows in time order, oldest first."""

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
        self.prev = np.zeros(self.hop, np.float32)  # zero padding in front of frame 0
        self.tail = np.zeros(self.hop, np.float32)  # second half of a frame whose overlap-add is not finished
        self.t = -1  # index of the last frame received
        self.hist = np.zeros((self.order, c["fft"] // 2 + 1), np.complex64)  # hist[k] = spectrum of frame n-k (frames with negative index are 0)
        self.pending: list[tuple[np.ndarray, np.ndarray]] = []  # (embedding, Q) pairs layer 0 has not processed yet
        la_of = [self.la if i == 0 else 0 for i in range(c["n_layers"])]
        self.la_of = la_of
        self.k = [_Ring(self.W + la + 1, self.h, self.dh) for la in la_of]
        self.v = [_Ring(self.W + la + 1, self.h, self.dh) for la in la_of]
        self.trace = []

    # ---- building blocks ----
    def lin(self, x, name):
        return (x @ self.w[f"{name}.kernel"] + self.w[f"{name}.bias"]).astype(np.float32)

    def ln(self, x, name):
        return layer_norm(x, self.w[f"{name}.gamma"], self.w[f"{name}.beta"], self.w[f"{name}.eps"])

    def qkv(self, i, x):
        """LN -> QKV projection, K/V go into the cache, returns Q [h, dh]."""
        q, k, v = np.split(self.lin(self.ln(x, f"blocks.{i}.ln1"), f"blocks.{i}.attn.qkv"), 3)
        self.k[i].push(k.reshape(self.h, self.dh))
        self.v[i].push(v.reshape(self.h, self.dh))
        return q.reshape(self.h, self.dh)

    def attend(self, i, x, q, u):
        """Layer i's attention + feed-forward for frame u (the newest frame in the K/V cache is u + that layer's lookahead)."""
        la, W = self.la_of[i], self.W
        m = min(u, W) + la + 1  # keys that exist: frames u-min(u,W) ... u+la
        K, V = self.k[i].last(m), self.v[i].last(m)
        bias = self.w[f"blocks.{i}.attn.rel_bias"][:, W + la - m + 1: W + la + 1]  # offsets -(m-1-la) ... +la
        s = np.einsum("hd,mhd->hm", q, K) / np.float32(math.sqrt(self.dh)) + bias
        s = np.exp(s - s.max(-1, keepdims=True))
        ctx = np.einsum("hm,mhd->hd", s / s.sum(-1, keepdims=True), V).reshape(self.d)
        x = x + self.lin(ctx, f"blocks.{i}.attn.out")
        return x + self.lin(gelu(self.lin(self.ln(x, f"blocks.{i}.ln2"), f"blocks.{i}.ff1")), f"blocks.{i}.ff2")

    def embed(self, spec: np.ndarray) -> np.ndarray:
        c = self.cfg
        re, im = spec.real.astype(np.float32), spec.imag.astype(np.float32)
        erb = np.add.reduceat(re * re + im * im, self.starts) / self.widths  # mean power of each band
        feat_erb = ((10.0 * np.log10(erb + 1e-10) + c["erb_db_offset"]) / c["erb_db_scale"]).astype(np.float32)
        lr, li = re[: c["nb_df"]], im[: c["nb_df"]]
        mag = np.sqrt(lr * lr + li * li + 1e-12).mean()
        feat_spec = np.concatenate([lr / (mag + 1e-6), li / (mag + 1e-6)]).astype(np.float32)
        h = np.concatenate([gelu(self.lin(feat_erb, "erb_in")), gelu(self.lin(feat_spec, "spec_in"))])
        return self.lin(h, "mix")

    # ---- streaming interface ----
    def push(self, block: np.ndarray) -> np.ndarray | None:
        """Feed hop samples; returns hop output samples, or None during warm-up (the first la+1 calls)."""
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
        emb_u = x  # frame u's embedding (emb belongs to the just-arrived frame n)
        layers = []
        x = self.attend(0, x, q, u)
        layers.append(x)
        for i in range(1, c["n_layers"]):
            x = self.attend(i, x, self.qkv(i, x), u)
            layers.append(x)
        h = self.ln(x, "norm_out")
        gains = (1.0 / (1.0 + np.exp(-self.lin(h, "gain")))).astype(np.float32)
        mask = np.repeat(gains, self.widths)
        out = self.hist[self.la] * mask  # spectrum of frame u = n - la
        coefs = np.tanh(self.lin(h, "df")).reshape(c["nb_df"], self.order, 2)
        wc = coefs[..., 0] + 1j * coefs[..., 1]  # [nb_df, order]
        # Y[f] = g[f] * sum_k W[f,k] * X[u+la-k][f], where u+la-k = n-k, i.e. hist[k]; g is the gain of the band containing bin f
        out[: c["nb_df"]] = (wc.T * self.hist[: self.order, : c["nb_df"]]).sum(0) * mask[: c["nb_df"]]
        if self.trace_on:
            self.trace.append({"u": u, "emb": emb_u, "layers": layers, "h": h, "gains": gains, "coefs": coefs,
                               "spec_out": out.copy()})
        seg = np.fft.irfft(out, c["fft"]).astype(np.float32) * self.win
        y = self.tail + seg[: self.hop]
        self.tail = seg[self.hop:]
        return None if u == 0 else y  # u=0 corresponds to the zero-padded region before frame 0; drop it

    def process(self, x: np.ndarray) -> np.ndarray:
        """Feed a whole clip (length a multiple of hop); returns the samples output so far, of length len(x) - (la+1)*hop, aligned with the first that many samples of the offline output."""
        self.reset()
        outs = [self.push(x[i: i + self.hop]) for i in range(0, len(x), self.hop)]
        return np.concatenate([o for o in outs if o is not None])
