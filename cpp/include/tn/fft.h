// Complex FFT of the fixed length tn::FFT (mixed radix 2/3/4/5, recursive DIT, table-driven twiddles). The tables are computed once at construction and no memory is allocated afterwards.
// Real input is treated as complex (imaginary part 0).
#pragma once
#include <cmath>
#include <stdexcept>

#include "config.h"

namespace tn {

inline constexpr double kPi = 3.14159265358979323846;  // MSVC has no M_PI by default, so it is not used

struct cf {
  float re, im;
};

class Fft {
 public:
  static constexpr int N = FFT;

  Fft() {
    for (int k = 0; k < N; ++k) {
      const double a = -2.0 * kPi * k / N;
      tw_[k] = {static_cast<float>(std::cos(a)), static_cast<float>(std::sin(a))};
    }
    int n = N;
    for (int r : {4, 2, 3, 5}) {
      while (n % r == 0 && nfac_ < kMaxFac) {
        fac_[nfac_++] = r;
        n /= r;
      }
    }
    if (n != 1) throw std::runtime_error("FFT 长度只支持 2/3/5 的因子");
  }

  // out = DFT(in); in and out must not overlap
  void forward(const cf* in, cf* out) const { rec(in, out, N, 1, 0); }

 private:
  static constexpr int kMaxFac = 16;

  void rec(const cf* in, cf* out, int n, int stride, int fi) const {
    if (n == 1) {
      out[0] = in[0];
      return;
    }
    const int r = fac_[fi], m = n / r;
    for (int q = 0; q < r; ++q) rec(in + q * stride, out + q * m, m, stride * r, fi + 1);
    const int step = N / n, rstep = N / r;
    for (int k = 0; k < m; ++k) {
      cf t[5];
      for (int q = 0; q < r; ++q) {
        const cf a = out[q * m + k], w = tw_[q * k * step];
        t[q] = {a.re * w.re - a.im * w.im, a.re * w.im + a.im * w.re};
      }
      for (int p = 0; p < r; ++p) {
        cf s = t[0];
        for (int q = 1; q < r; ++q) {
          const cf w = tw_[((p * q) % r) * rstep];
          s.re += t[q].re * w.re - t[q].im * w.im;
          s.im += t[q].re * w.im + t[q].im * w.re;
        }
        out[k + p * m] = s;
      }
    }
  }

  cf tw_[N];
  int fac_[kMaxFac] = {};
  int nfac_ = 0;
};

}  // namespace tn
