// C++ implementation of the streaming denoising model: all shapes come from config.h (compile-time constants), the weights from weights.bin (validated on load).
// Frame-by-frame processing (HOP samples at a time) with a KV cache and a convolution history per layer; the real-time path (push) does not allocate memory, take locks or throw exceptions.
// Numerically it corresponds one-to-one to the numpy reference (transnoise/deploy/ref.py); see there for the algorithm description.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>

#include "config.h"
#include "fft.h"

namespace tn {

// ---------------------------------------------------------------- weights
class Weights {
 public:
  struct Lin {
    const float *w, *b;
  };
  struct Block {
    Lin ffa1, ffa2, qkv, out, pw1, pw2, ffb1, ffb2;
    const float* rel;                // [HEADS][WINDOW + la_i + 1]
    const float *dw_w, *dw_b;        // depthwise convolution: [CK][D] taps (tap k multiplies the frame CK-1-k steps back), [D] bias
    const float *dwln_g, *dwln_b;    // [D] gamma and beta of the LayerNorm after the depthwise convolution
  };

  Lin erb_in, spec_in, mix, gain, df;
  Block blk[LAYERS];

  // Construct from the contents of weights.bin (copied into 64-byte aligned internal storage). Throws on a shape/constant/name mismatch (at load time only).
  Weights(const uint8_t* data, size_t size) {
    constexpr size_t kHead = 8 + 8;
    if (size < kHead || std::memcmp(data, "TNWEIGHT", 8) != 0) throw std::runtime_error("weights.bin: bad magic number");
    uint32_t n_const, n_tensors;
    std::memcpy(&n_const, data + 8, 4);
    std::memcpy(&n_tensors, data + 12, 4);
    constexpr size_t n_expect = sizeof(CONSTS) / sizeof(CONSTS[0]);
    if (n_const != n_expect) throw std::runtime_error("weights.bin: wrong number of constants");
    const uint8_t* p = data + kHead;
    for (size_t i = 0; i < n_expect; ++i) {
      int32_t v;
      std::memcpy(&v, p + 4 * i, 4);
      if (v != CONSTS[i]) throw std::runtime_error("weights.bin: model constant #" + std::to_string(i) + " differs from the compiled-in config.h; re-pack and recompile");
    }
    p += 4 * n_const;
    struct Entry {
      std::string name;
      uint64_t off, count;
    };
    std::unique_ptr<Entry[]> table(new Entry[n_tensors]);
    uint64_t total = 0;
    for (uint32_t i = 0; i < n_tensors; ++i, p += 56) {
      table[i].name = std::string(reinterpret_cast<const char*>(p), strnlen(reinterpret_cast<const char*>(p), 40));
      std::memcpy(&table[i].off, p + 40, 8);
      std::memcpy(&table[i].count, p + 48, 8);
      total = std::max(total, table[i].off + table[i].count);
    }
    size_t data_off = static_cast<size_t>(p - data);
    data_off += (64 - data_off % 64) % 64;
    if (size != data_off + total * 4) throw std::runtime_error("weights.bin: wrong size");
    store_ = static_cast<float*>(::operator new(((total * 4 + 63) / 64) * 64, std::align_val_t(64)));  // std::aligned_alloc is not available on MSVC
    std::memcpy(store_, data + data_off, total * 4);

    auto take = [&](const std::string& name, size_t count) -> const float* {
      for (uint32_t i = 0; i < n_tensors; ++i)
        if (table[i].name == name) {
          if (table[i].count != count) throw std::runtime_error("weights.bin: wrong element count for " + name);
          return store_ + table[i].off;
        }
      throw std::runtime_error("weights.bin: missing " + name);
    };
    auto lin = [&](const std::string& n, int in, int out) { return Lin{take(n + ".w", size_t(in) * out), take(n + ".b", out)}; };
    erb_in = lin("erb_in", NB_ERB, D);
    spec_in = lin("spec_in", 2 * NB_DF, D);
    mix = lin("mix", 2 * D, D);
    for (int i = 0; i < LAYERS; ++i) {
      const std::string b = "b" + std::to_string(i);
      blk[i].ffa1 = lin(b + ".ffa1", D, FF);
      blk[i].ffa2 = lin(b + ".ffa2", FF, D);
      blk[i].qkv = lin(b + ".qkv", D, 3 * D);
      blk[i].rel = take(b + ".rel", size_t(HEADS) * (WINDOW + (i == 0 ? LA : 0) + 1));
      blk[i].out = lin(b + ".out", D, D);
      blk[i].pw1 = lin(b + ".pw1", D, 2 * D);
      blk[i].dw_w = take(b + ".dw.w", size_t(CK) * D);
      blk[i].dw_b = take(b + ".dw.b", D);
      blk[i].dwln_g = take(b + ".dwln.g", D);
      blk[i].dwln_b = take(b + ".dwln.b", D);
      blk[i].pw2 = lin(b + ".pw2", D, D);
      blk[i].ffb1 = lin(b + ".ffb1", D, FF);
      blk[i].ffb2 = lin(b + ".ffb2", FF, D);
    }
    gain = lin("gain", D, NB_ERB);
    df = lin("df", D, NB_DF * ORDER * 2);
  }
  Weights(const Weights&) = delete;
  Weights& operator=(const Weights&) = delete;
  ~Weights() { ::operator delete(store_, std::align_val_t(64)); }

 private:
  float* store_ = nullptr;
};

// ---------------------------------------------------------------- kernels
namespace kern {

// Compile-time square root (Newton iteration in double): std::sqrt is not constexpr on MSVC, and GCC's __builtin_sqrtf does not exist on MSVC
constexpr double csqrt(double x) {
  double g = x > 1.0 ? x / 2 : 1.0;
  for (int i = 0; i < 60; ++i) g = 0.5 * (g + x / g);
  return g;
}

// y = b + x*W, with W in [IN][OUT] row-major order. The outer loop runs over the input dimension and the inner one over the output dimension:
// every output element accumulates in a fixed order (no floating-point reordering needed) and the compiler can vectorize the inner loop directly.
template <int IN, int OUT>
inline void linear(const float* __restrict x, const Weights::Lin& l, float* __restrict y) {
  for (int j = 0; j < OUT; ++j) y[j] = l.b[j];
  for (int i = 0; i < IN; ++i) {
    const float xi = x[i];
    const float* __restrict wr = l.w + size_t(i) * OUT;
    for (int j = 0; j < OUT; ++j) y[j] += xi * wr[j];
  }
}

inline float gelu(float x) { return 0.5f * x * (1.0f + std::erf(x / 1.41421356237f)); }

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

// Normalization only (gamma and beta are already folded into the following linear layer)
inline void ln_norm(const float* x, float* y) {
  float mean = 0.f;
  for (int i = 0; i < D; ++i) mean += x[i];
  mean /= D;
  float var = 0.f;
  for (int i = 0; i < D; ++i) {
    const float t = x[i] - mean;
    var += t * t;
  }
  var /= D;
  const float inv = 1.0f / std::sqrt(var + LN_EPS);
  for (int i = 0; i < D; ++i) y[i] = (x[i] - mean) * inv;
}

// Normalization with gamma and beta, for the LayerNorm that cannot be folded into a linear layer
inline void ln_affine(const float* x, const float* gamma, const float* beta, float* y) {
  ln_norm(x, y);
  for (int i = 0; i < D; ++i) y[i] = y[i] * gamma[i] + beta[i];
}

}  // namespace kern

// ---------------------------------------------------------------- streaming denoiser (mono; one instance = all the state of one audio stream)
struct Trace {  // for cross-checking: filled once per output frame
  float pre[D], layer[LAYERS][D], xn[D], gains[NB_ERB], coefs[NB_DF * ORDER * 2], re[BINS], im[BINS];
};

class Denoiser {
 public:
  static_assert(FFT == 2 * HOP, "the implementation assumes fft == 2*hop");
  static_assert(ORDER > LA, "df_order must be > lookahead");
  static_assert(CK >= 1, "conv_kernel must be >= 1");

  explicit Denoiser(const Weights& w) : w_(w) {
    for (int k = 0; k < FFT; ++k) win_[k] = static_cast<float>(std::sin(kPi / 2 * std::pow(std::sin(kPi * (k + 0.5) / FFT), 2)));
    reset();
  }

  void reset() {
    std::memset(prev_, 0, sizeof prev_);
    std::memset(tail_, 0, sizeof tail_);
    std::memset(hist_re_, 0, sizeof hist_re_);
    std::memset(hist_im_, 0, sizeof hist_im_);
    std::memset(k_, 0, sizeof k_);
    std::memset(v_, 0, sizeof v_);
    std::memset(cv_, 0, sizeof cv_);
    for (int i = 0; i < LAYERS; ++i) kn_[i] = cn_[i] = 0;
    hist_head_ = 0;
    pend_head_ = pend_n_ = 0;
    t_ = -1;
  }

  Trace* trace = nullptr;  // when non-null, every output frame is written into it

  // Feed HOP samples; when output is ready, write HOP samples to out and return true (the first LA+1 calls return false). in and out may be the same memory.
  bool push(const float* in, float* out) {
    // --- STFT: frame = [previous block, this block] x window
    cf z[FFT], spec[FFT];
    for (int i = 0; i < HOP; ++i) {
      z[i] = {prev_[i] * win_[i], 0.f};
      z[HOP + i] = {in[i] * win_[HOP + i], 0.f};
    }
    std::memcpy(prev_, in, sizeof prev_);
    fft_.forward(z, spec);
    hist_head_ = (hist_head_ + 1) % ORDER;
    float* hre = hist_re_[hist_head_];
    float* him = hist_im_[hist_head_];
    for (int k = 0; k < BINS; ++k) hre[k] = spec[k].re, him[k] = spec[k].im;
    ++t_;

    // --- features -> embedding -> layer 0's first feed-forward half-step and QKV (K/V go into the cache, x and Q into the pending queue)
    float emb[D], q[D];
    embed(hre, him, emb);
    ffn(w_.blk[0].ffa1, w_.blk[0].ffa2, emb);
    qkv(0, emb, q);
    const int slot = (pend_head_ + pend_n_) % (LA + 1);
    std::memcpy(pend_x_[slot], emb, sizeof emb);
    std::memcpy(pend_q_[slot], q, sizeof q);
    ++pend_n_;

    const int u = static_cast<int>(t_ - LA);  // the frame that can be finalized now
    if (u < 0) return false;
    float x[D];
    std::memcpy(x, pend_x_[pend_head_], sizeof x);
    std::memcpy(q, pend_q_[pend_head_], sizeof q);
    pend_head_ = (pend_head_ + 1) % (LA + 1);
    --pend_n_;
    if (trace) std::memcpy(trace->pre, x, sizeof x);

    finish(0, x, q, u);
    if (trace) std::memcpy(trace->layer[0], x, sizeof x);
    for (int i = 1; i < LAYERS; ++i) {
      ffn(w_.blk[i].ffa1, w_.blk[i].ffa2, x);
      qkv(i, x, q);
      finish(i, x, q, u);
      if (trace) std::memcpy(trace->layer[i], x, sizeof x);
    }

    // --- output heads
    float xn[D], gains[NB_ERB], coefs[NB_DF * ORDER * 2];
    kern::ln_norm(x, xn);
    kern::linear<D, NB_ERB>(xn, w_.gain, gains);
    for (int b = 0; b < NB_ERB; ++b) gains[b] = 1.0f / (1.0f + std::exp(-gains[b]));
    kern::linear<D, NB_DF * ORDER * 2>(xn, w_.df, coefs);
    for (int i = 0; i < NB_DF * ORDER * 2; ++i) coefs[i] = std::tanh(coefs[i]);

    // The spectrum of frame u = n - LA times the ERB gain; the lowest NB_DF bins use the deep filter instead, Y[f] = g[f] * sum_k W[f,k] * X[n-k][f] (g is the gain of the band containing bin f)
    cf* o = spec;  // reuse
    float gate[NB_DF];
    {
      const float* sre = hist_re_[(hist_head_ - LA + ORDER) % ORDER];
      const float* sim = hist_im_[(hist_head_ - LA + ORDER) % ORDER];
      int k = 0;
      for (int b = 0; b < NB_ERB; ++b)
        for (int e = 0; e < ERB_WIDTHS[b]; ++e, ++k) {
          o[k] = {sre[k] * gains[b], sim[k] * gains[b]};
          if (k < NB_DF) gate[k] = gains[b];
        }
    }
    for (int f = 0; f < NB_DF; ++f) {
      float yr = 0.f, yi = 0.f;
      for (int k = 0; k < ORDER; ++k) {
        const int s = (hist_head_ - k + ORDER) % ORDER;
        const float wr = coefs[(f * ORDER + k) * 2], wi = coefs[(f * ORDER + k) * 2 + 1];
        const float xr = hist_re_[s][f], xi = hist_im_[s][f];
        yr += wr * xr - wi * xi;
        yi += wr * xi + wi * xr;
      }
      o[f] = {yr * gate[f], yi * gate[f]};
    }
    if (trace) {
      std::memcpy(trace->xn, xn, sizeof xn);
      std::memcpy(trace->gains, gains, sizeof gains);
      std::memcpy(trace->coefs, coefs, sizeof coefs);
      for (int k = 0; k < BINS; ++k) trace->re[k] = o[k].re, trace->im[k] = o[k].im;
    }

    // --- inverse STFT: fill out a Hermitian-symmetric full spectrum, invert with a conj-FFT, apply the window, overlap-add
    cf full[FFT], tmp[FFT];
    full[0] = {o[0].re, 0.f};  // like numpy.fft.irfft, ignore the imaginary parts of DC and Nyquist
    full[FFT / 2] = {o[FFT / 2].re, 0.f};
    for (int k = 1; k < FFT / 2; ++k) {
      full[k] = {o[k].re, -o[k].im};  // store the conjugate first; taking the conjugate again after forward = inverse transform
      full[FFT - k] = {o[k].re, o[k].im};
    }
    fft_.forward(full, tmp);
    const float scale = 1.0f / FFT;
    float seg[FFT];
    for (int i = 0; i < FFT; ++i) seg[i] = tmp[i].re * scale * win_[i];
    if (u == 0) {  // the zero-padded region before frame 0: drop the output, but keep the second half for the next overlap-add
      std::memcpy(tail_, seg + HOP, sizeof tail_);
      return false;
    }
    for (int i = 0; i < HOP; ++i) out[i] = tail_[i] + seg[i];
    std::memcpy(tail_, seg + HOP, sizeof tail_);
    return true;
  }

 private:
  static constexpr int CAP = WINDOW + LA + 1;  // capacity of each layer's K/V ring buffer (not fully used from layer 1 on)

  void embed(const float* re, const float* im, float* emb) {
    float feat_erb[NB_ERB], feat_spec[2 * NB_DF];
    int k = 0;
    for (int b = 0; b < NB_ERB; ++b) {
      float s = 0.f;
      for (int e = 0; e < ERB_WIDTHS[b]; ++e, ++k) s += re[k] * re[k] + im[k] * im[k];
      feat_erb[b] = (10.0f * std::log10(s / ERB_WIDTHS[b] + 1e-10f) + ERB_DB_OFFSET) / ERB_DB_SCALE;
    }
    float mag = 0.f;
    for (int f = 0; f < NB_DF; ++f) mag += std::sqrt(re[f] * re[f] + im[f] * im[f] + 1e-12f);
    mag /= NB_DF;
    const float inv = 1.0f / (mag + 1e-6f);  // the numpy version divides; this multiplies by the reciprocal, within 1 ulp
    for (int f = 0; f < NB_DF; ++f) feat_spec[f] = re[f] * inv, feat_spec[NB_DF + f] = im[f] * inv;
    float h[2 * D];
    kern::linear<NB_ERB, D>(feat_erb, w_.erb_in, h);
    kern::linear<2 * NB_DF, D>(feat_spec, w_.spec_in, h + D);
    for (int i = 0; i < 2 * D; ++i) h[i] = kern::gelu(h[i]);
    kern::linear<2 * D, D>(h, w_.mix, emb);
  }

  // LN -> QKV; K/V are written into layer i's cache, Q is returned.
  void qkv(int i, const float* x, float* q) {
    float xn[D], r[3 * D];
    kern::ln_norm(x, xn);
    kern::linear<D, 3 * D>(xn, w_.blk[i].qkv, r);
    std::memcpy(q, r, D * sizeof(float));
    const int pos = static_cast<int>(kn_[i] % CAP);
    std::memcpy(k_[i][pos], r + D, D * sizeof(float));
    std::memcpy(v_[i][pos], r + 2 * D, D * sizeof(float));
    ++kn_[i];
  }

  // Half-step feed-forward module: x += 0.5 * FFN(LN(x)) (the LN affine is folded into l1).
  void ffn(const Weights::Lin& l1, const Weights::Lin& l2, float* x) {
    float xn[D], f1[FF], f2[D];
    kern::ln_norm(x, xn);
    kern::linear<D, FF>(xn, l1, f1);
    for (int j = 0; j < FF; ++j) f1[j] = kern::silu(f1[j]);
    kern::linear<FF, D>(f1, l2, f2);
    for (int d = 0; d < D; ++d) x[d] += 0.5f * f2[d];
  }

  // Layer i's attention for frame u, updating x in place.
  void attend(int i, float* x, const float* q, int u) {
    const int la = i == 0 ? LA : 0;
    const int m = std::min(u, WINDOW) + la + 1;  // keys that exist: frames u-min(u,W) ... u+la
    const int first = static_cast<int>((kn_[i] - m) % CAP + CAP) % CAP;
    const float* rel = w_.blk[i].rel;
    const int rel_n = WINDOW + la + 1, rel0 = WINDOW + la - m + 1;
    constexpr float inv_sqrt_dh = 1.0f / static_cast<float>(kern::csqrt(DH));
    constexpr int DH8 = DH / 8 * 8;  // the dot product accumulates in 8 lanes; when DH is not a multiple of 8, the remaining dimensions are added to the first lanes in turn
    float ctx[D];
    for (int h = 0; h < HEADS; ++h) {
      float s[CAP];
      float smax = -3.4e38f;
      for (int j = 0, pos = first; j < m; ++j, pos = pos + 1 == CAP ? 0 : pos + 1) {
        const float* kk = k_[i][pos] + h * DH;
        const float* qq = q + h * DH;
        float acc[8] = {};  // fixed 8-lane accumulation: a deterministic order that also vectorizes easily
        for (int d = 0; d < DH8; d += 8)
          for (int l = 0; l < 8; ++l) acc[l] += qq[d + l] * kk[d + l];
        for (int d = DH8; d < DH; ++d) acc[d - DH8] += qq[d] * kk[d];
        const float dot = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
        s[j] = dot * inv_sqrt_dh + rel[h * rel_n + rel0 + j];
        smax = std::max(smax, s[j]);
      }
      float sum = 0.f;
      for (int j = 0; j < m; ++j) sum += (s[j] = std::exp(s[j] - smax));
      const float inv = 1.0f / sum;
      float* c = ctx + h * DH;
      for (int d = 0; d < DH; ++d) c[d] = 0.f;
      for (int j = 0, pos = first; j < m; ++j, pos = pos + 1 == CAP ? 0 : pos + 1) {
        const float p = s[j] * inv;
        const float* vv = v_[i][pos] + h * DH;
        for (int d = 0; d < DH; ++d) c[d] += p * vv[d];
      }
    }
    float o[D];
    kern::linear<D, D>(ctx, w_.blk[i].out, o);
    for (int d = 0; d < D; ++d) x[d] += o[d];
  }

  // Layer i's convolution module for the next frame, updating x in place: pointwise, GLU, causal depthwise convolution over the last CK GLU outputs, LN, Swish, pointwise.
  void conv(int i, float* x) {
    const Weights::Block& b = w_.blk[i];
    float xn[D], h[2 * D];
    kern::ln_norm(x, xn);
    kern::linear<D, 2 * D>(xn, b.pw1, h);
    float* newest = cv_[i][cn_[i] % CK];
    for (int d = 0; d < D; ++d) newest[d] = h[d] / (1.0f + std::exp(-h[D + d]));  // GLU
    ++cn_[i];
    float y[D];
    std::memcpy(y, b.dw_b, sizeof y);
    int slot = static_cast<int>(cn_[i] % CK);  // the oldest of the last CK frames; before frame 0 the rows are still zero
    for (int k = 0; k < CK; ++k, slot = slot + 1 == CK ? 0 : slot + 1) {
      const float* row = cv_[i][slot];
      const float* tap = b.dw_w + size_t(k) * D;
      for (int d = 0; d < D; ++d) y[d] += tap[d] * row[d];
    }
    kern::ln_affine(y, b.dwln_g, b.dwln_b, xn);
    for (int d = 0; d < D; ++d) xn[d] = kern::silu(xn[d]);
    kern::linear<D, D>(xn, b.pw2, y);
    for (int d = 0; d < D; ++d) x[d] += y[d];
  }

  // Everything of layer i that follows the QKV projection, for frame u.
  void finish(int i, float* x, const float* q, int u) {
    attend(i, x, q, u);
    conv(i, x);
    ffn(w_.blk[i].ffb1, w_.blk[i].ffb2, x);
  }

  const Weights& w_;
  Fft fft_;
  float win_[FFT];
  float prev_[HOP], tail_[HOP];
  float hist_re_[ORDER][BINS], hist_im_[ORDER][BINS];
  int hist_head_;
  alignas(64) float k_[LAYERS][CAP][D], v_[LAYERS][CAP][D];
  alignas(64) float cv_[LAYERS][CK][D];  // the last CK GLU outputs of each layer's convolution module (ring buffer)
  int64_t kn_[LAYERS], cn_[LAYERS];  // long is only 32 bits on Windows, so use int64_t
  alignas(64) float pend_x_[LA + 1][D], pend_q_[LA + 1][D];
  int pend_head_, pend_n_;
  int64_t t_;
};

}  // namespace tn
