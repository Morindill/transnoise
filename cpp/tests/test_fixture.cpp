// 对拍：C++ 实现 vs numpy 参考实现的夹具（输入波形 + 期望输出 + 每帧中间量）。
// 用法：test_fixture <weights.bin> <fixture.bin>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "common.h"
#include "tn/model.h"

struct Err {
  double worst = 0;  // max|a-b| / max(max|b|, 1e-3)，按帧取最差
  void add(const float* a, const float* b, int n) {
    double d = 0, s = 1e-3;
    for (int i = 0; i < n; ++i) d = std::max(d, double(std::fabs(a[i] - b[i]))), s = std::max(s, double(std::fabs(b[i])));
    worst = std::max(worst, d / s);
  }
};

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "用法：%s weights.bin fixture.bin\n", argv[0]);
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
    std::fprintf(stderr, "夹具 hop=%d，和编译进来的 HOP=%d 不一致\n", hop, tn::HOP);
    return 2;
  }
  const float* in = reinterpret_cast<const float*>(fx.data() + 16);
  const float* want = in + size_t(nframes) * hop;
  const float* tr = want + n_out;
  constexpr int kTraceLen = tn::D * (2 + tn::LAYERS) + tn::NB_ERB + tn::NB_DF * tn::ORDER * 2 + 2 * tn::BINS;
  if (size_t(16) + 4 * (size_t(nframes) * hop + n_out + size_t(n_trace) * kTraceLen) != fx.size()) {
    std::fprintf(stderr, "夹具大小和 config.h 对不上\n");
    return 2;
  }

  Err e_emb, e_xn, e_gain, e_coef, e_re, e_im;
  Err e_layer[tn::LAYERS];
  std::vector<float> got(size_t(n_out), 0.f);
  int produced = 0, tframe = 0;
  float out[tn::HOP];
  for (int f = 0; f < nframes; ++f) {
    const bool ready = model->push(in + size_t(f) * hop, out);
    if (f < tn::LA) continue;  // 预热：还没有任何帧算完（trace 从 u=0 起，那一帧不产出样本）
    if (ready) {
      std::copy(out, out + hop, got.begin() + size_t(produced) * hop);
      ++produced;
    }
    const float* t = tr + size_t(tframe) * kTraceLen;
    e_emb.add(trace->emb, t, tn::D);
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
    std::fprintf(stderr, "输出帧数不对：产出 %d 帧，期望 %d；trace %d，期望 %d\n", produced, n_out / hop, tframe, n_trace);
    return 1;
  }
  double d2 = 0, s2 = 0, dmax = 0;
  for (int i = 0; i < n_out; ++i) {
    const double d = got[i] - want[i];
    d2 += d * d, s2 += double(want[i]) * want[i], dmax = std::max(dmax, std::fabs(d));
  }
  const double rel = std::sqrt(d2 / s2);

  struct Row {
    const char* name;
    double err, tol;
  };
  std::vector<Row> rows = {{"嵌入 emb", e_emb.worst, 1e-4}};
  static const char* layer_names[] = {"第 0 层输出", "第 1 层输出", "第 2 层输出", "第 3 层输出"};
  for (int i = 0; i < tn::LAYERS; ++i) rows.push_back({layer_names[i], e_layer[i].worst, 1e-4});
  rows.push_back({"norm_out 归一化", e_xn.worst, 1e-4});
  // 输出头：宽线性层 + LN 折叠的舍入差，再经 sigmoid/tanh；输出波形的误差（下面）才是最终判据
  rows.push_back({"ERB 增益", e_gain.worst, 5e-5});
  rows.push_back({"深度滤波系数", e_coef.worst, 5e-5});
  rows.push_back({"增强谱实部", e_re.worst, 1e-4});
  rows.push_back({"增强谱虚部", e_im.worst, 1e-4});
  bool ok = true;
  std::printf("逐帧中间量（相对最大误差 = max|差| / max(max|参考|, 1e-3)，取所有帧最差）：\n");
  for (const auto& r : rows) {
    const bool pass = r.err < r.tol;
    ok &= pass;
    std::printf("  [%s] %-14s %.2e（阈值 %.0e）\n", pass ? "通过" : "失败", r.name, r.err, r.tol);
  }
  const bool pass_out = rel < 1e-5 && dmax < 1e-5;
  ok &= pass_out;
  std::printf("输出波形：[%s] 相对 RMS 误差 %.2e，最大绝对误差 %.2e（%d 个样本）\n", pass_out ? "通过" : "失败", rel, dmax, n_out);

  // 速度：重放整段输入几遍，统计每帧耗时（只是粗略的参考，真正的 p99 要在宿主里量）
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
  std::printf("每帧耗时（%zu 次）：均值 %.1f µs，p50 %.1f，p99 %.1f，最大 %.1f（预算 %.0f µs，均值占 %.2f%%）\n", us.size(), mean, q(0.5), q(0.99), us.back(),
              1e6 * tn::HOP / tn::SR, 100 * mean / (1e6 * tn::HOP / tn::SR));
  std::printf(ok ? "全部通过\n" : "有失败项\n");
  return ok ? 0 : 1;
}
