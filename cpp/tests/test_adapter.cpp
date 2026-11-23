// BlockAdapter 自检：任意分块方式（固定/随机，1 … 数千样本）的输出必须逐比特相同，且等于"整段按 HOP 喂入"的输出延迟 latency() 个样本。
// 用法：test_adapter <weights.bin> <fixture.bin>
#include <cstdio>
#include <random>
#include <vector>

#include "common.h"
#include "tn/stream_adapter.h"

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  const auto wb = read_file(argv[1]), fx = read_file(argv[2]);
  tn::Weights weights(wb.data(), wb.size());
  const int32_t* h = reinterpret_cast<const int32_t*>(fx.data());
  const int nframes = h[1], n_out = h[2];
  const float* in = reinterpret_cast<const float*>(fx.data() + 16);
  const float* want = in + size_t(nframes) * tn::HOP;  // numpy 参考的输出（已对齐，长度 n_out）
  const size_t n = size_t(nframes) * tn::HOP;

  // 参照：不经适配器，整段按 HOP 喂入，得到对齐的输出
  auto ref_model = std::make_unique<tn::Denoiser>(weights);
  std::vector<float> ref;
  float o[tn::HOP];
  for (int f = 0; f < nframes; ++f)
    if (ref_model->push(in + size_t(f) * tn::HOP, o)) ref.insert(ref.end(), o, o + tn::HOP);
  bool ok = ref.size() == size_t(n_out);
  for (int i = 0; ok && i < n_out; ++i) ok = ref[i] == want[i] || std::fabs(ref[i] - want[i]) < 1e-5f;
  std::printf("[%s] 直接按 HOP 喂入的输出和夹具一致（%zu 个样本）\n", ok ? "通过" : "失败", ref.size());

  struct Case {
    const char* name;
    int fixed;  // 传给 BlockAdapter 的 fixed_block
    std::vector<int> sizes;  // 循环使用的块大小
  };
  std::mt19937 rng(1);
  std::vector<int> rnd(997);
  for (auto& v : rnd) v = 1 + int(rng() % 3000);
  std::vector<Case> cases = {{"块=1", 1, {1}},       {"块=32", 32, {32}},     {"块=64", 64, {64}},       {"块=128", 128, {128}},
                             {"块=256", 256, {256}}, {"块=480", 480, {480}},  {"块=512", 512, {512}},    {"块=960", 960, {960}},
                             {"块=1024", 1024, {1024}}, {"块=4800", 4800, {4800}}, {"随机 1…3000", 0, rnd}};
  for (const auto& c : cases) {
    auto m = std::make_unique<tn::Denoiser>(weights);
    tn::BlockAdapter ad(*m, c.fixed);
    const int lat = ad.latency();
    std::vector<float> x(in, in + n), y(n, 0.f);
    x.resize(n + lat + 5000, 0.f);  // 补零把尾巴冲出来
    y.resize(x.size(), 0.f);
    size_t pos = 0, k = 0;
    while (pos < n + lat) {
      const int b = int(std::min<size_t>(c.sizes[k++ % c.sizes.size()], x.size() - pos));
      ad.process(x.data() + pos, y.data() + pos, b);
      pos += b;
    }
    // 输出 y[lat + i] 应该等于参照的第 i 个样本，逐比特相同
    size_t bad = 0;
    for (size_t i = 0; i < ref.size() && lat + i < y.size(); ++i) bad += y[lat + i] != ref[i];
    const bool pass = bad == 0 && ad.underruns() == 0;
    ok &= pass;
    std::printf("[%s] %-12s 延迟 %4d 样本（%.2f ms），不一致的样本 %zu，underrun %d\n", pass ? "通过" : "失败", c.name, lat, 1e3 * lat / tn::SR, bad, ad.underruns());
  }
  std::printf(ok ? "全部通过\n" : "有失败项\n");
  return ok ? 0 : 1;
}
