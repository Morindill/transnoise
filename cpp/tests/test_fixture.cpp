// Cross-check: the C++ implementation vs the numpy reference's fixture (input waveform + expected output + per-frame intermediates).
// Usage: test_fixture <weights.bin> <fixture.bin>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "common.h"
#include "tn/model.h"

struct Err {
  double worst = 0;  // max|a-b| / max(max|b|, 1e-3), worst over frames
  void add(const float* a, const float* b, int n) {
    double d = 0, s = 1e-3;
    for (int i = 0; i < n; ++i) d = std::max(d, double(std::fabs(a[i] - b[i]))), s = std::max(s, double(std::fabs(b[i])));
    worst = std::max(worst, d / s);
  }
};

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s weights.bin fixture.bin\n", argv[0]);
    return 2;
  }
  const auto wb = read_file(argv[1]);
  const auto fx = read_file(argv[2]);
  tn::Weights weights(wb.data(), wb.size());
  auto model = std::make_unique<tn::Denoiser>(weights);
  auto trace = std::make_unique<tn::Trace>();
  model->trace = trace.get();

  const int32_t* h = reinterpret_cast<const int32_t*>(fx.data());
  const int hop = h[0], nframes = h[1], n_out = h[2], n_trace = h[3];
  if (hop != tn::HOP) {
    std::fprintf(stderr, "fixture hop=%d differs from the compiled-in HOP=%d\n", hop, tn::HOP);
    return 2;
  }
  const float* in = reinterpret_cast<const float*>(fx.data() + 16);
  const float* want = in + size_t(nframes) * hop;
  const float* tr = want + n_out;
  constexpr int kTraceLen = tn::D * (2 + tn::LAYERS) + tn::NB_ERB + tn::NB_DF * tn::ORDER * 2 + 2 * tn::BINS;
  if (size_t(16) + 4 * (size_t(nframes) * hop + n_out + size_t(n_trace) * kTraceLen) != fx.size()) {
    std::fprintf(stderr, "fixture size does not match config.h\n");
    return 2;
  }

  Err e_pre, e_xn, e_gain, e_coef, e_re, e_im;
  Err e_layer[tn::LAYERS];
  std::vector<float> got(size_t(n_out), 0.f);
  int produced = 0, tframe = 0;
  float out[tn::HOP];
  for (int f = 0; f < nframes; ++f) {
    const bool ready = model->push(in + size_t(f) * hop, out);
    if (f < tn::LA) continue;  // warm-up: no frame is finished yet (the trace starts at u=0, a frame that produces no samples)
    if (ready) {
      std::copy(out, out + hop, got.begin() + size_t(produced) * hop);
      ++produced;
    }
    const float* t = tr + size_t(tframe) * kTraceLen;
    e_pre.add(trace->pre, t, tn::D);
    t += tn::D;
    for (int i = 0; i < tn::LAYERS; ++i, t += tn::D) e_layer[i].add(trace->layer[i], t, tn::D);
    e_xn.add(trace->xn, t, tn::D), t += tn::D;
    e_gain.add(trace->gains, t, tn::NB_ERB), t += tn::NB_ERB;
    e_coef.add(trace->coefs, t, tn::NB_DF * tn::ORDER * 2), t += tn::NB_DF * tn::ORDER * 2;
    e_re.add(trace->re, t, tn::BINS), t += tn::BINS;
    e_im.add(trace->im, t, tn::BINS);
    ++tframe;
  }
  if (produced != n_out / hop || tframe != n_trace) {
    std::fprintf(stderr, "wrong number of output frames: produced %d, expected %d; trace %d, expected %d\n", produced, n_out / hop, tframe, n_trace);
    return 1;
  }
  double d2 = 0, s2 = 0, dmax = 0;
  for (int i = 0; i < n_out; ++i) {
    const double d = got[i] - want[i];
    d2 += d * d, s2 += double(want[i]) * want[i], dmax = std::max(dmax, std::fabs(d));
  }
  const double rel = std::sqrt(d2 / s2);

  struct Row {
    std::string name;
    double err, tol;
  };
  std::vector<Row> rows = {{"layer 0 attention input", e_pre.worst, 1e-4}};
  for (int i = 0; i < tn::LAYERS; ++i) rows.push_back({"layer " + std::to_string(i) + " output", e_layer[i].worst, 1e-4});
  rows.push_back({"norm_out", e_xn.worst, 1e-4});
  // Output heads: a wide linear layer + the rounding difference from folding LN, then sigmoid/tanh; the output waveform error (below) is the final criterion
  rows.push_back({"ERB gain", e_gain.worst, 5e-5});
  rows.push_back({"deep-filter coefs", e_coef.worst, 5e-5});
  rows.push_back({"enhanced spec real", e_re.worst, 1e-4});
  rows.push_back({"enhanced spec imag", e_im.worst, 1e-4});
  bool ok = true;
  std::printf("per-frame intermediates (relative max error = max|diff| / max(max|ref|, 1e-3), worst over all frames):\n");
  for (const auto& r : rows) {
    const bool pass = r.err < r.tol;
    ok &= pass;
    std::printf("  [%s] %-24s %.2e (tolerance %.0e)\n", pass ? "PASS" : "FAIL", r.name.c_str(), r.err, r.tol);
  }
  const bool pass_out = rel < 1e-5 && dmax < 1e-5;
  ok &= pass_out;
  std::printf("output waveform: [%s] relative RMS error %.2e, max absolute error %.2e (%d samples)\n", pass_out ? "PASS" : "FAIL", rel, dmax, n_out);

  // Speed: replay the whole input a few times and collect the per-frame time (only a rough reference; the real p99 has to be measured in the host)
  std::vector<double> us;
  model->trace = nullptr;
  for (int rep = 0; rep < 5; ++rep) {
    model->reset();
    for (int f = 0; f < nframes; ++f) {
      const auto t0 = std::chrono::steady_clock::now();
      model->push(in + size_t(f) * hop, out);
      us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
    }
  }
  std::sort(us.begin(), us.end());
  const auto q = [&](double p) { return us[size_t(p * (us.size() - 1))]; };
  double mean = 0;
  for (double v : us) mean += v;
  mean /= us.size();
  std::printf("time per frame (%zu runs): mean %.1f µs, p50 %.1f, p99 %.1f, max %.1f (budget %.0f µs, mean uses %.2f%%)\n", us.size(), mean, q(0.5), q(0.99), us.back(),
              1e6 * tn::HOP / tn::SR, 100 * mean / (1e6 * tn::HOP / tn::SR));
  std::printf(ok ? "all passed\n" : "some checks FAILED\n");
  return ok ? 0 : 1;
}
