// 命令行：用 C++ 核心（经 BlockAdapter 中间件）增强一个目录里的所有 wav，输出和输入时间对齐、等长，可以直接给评测打分。
//   tn_wav --weights gen/weights.bin --in 带噪目录 --out 输出目录 [--block 480] [--limit N]
// --block 模拟宿主块大小（默认 480；0 = 块大小不固定的最坏情况延迟）。输出是 PCM16 单声道 48kHz，文件名不变。
// 然后打分：uv run transnoise eval score noise <输出目录> --name 名字
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "tn/stream_adapter.h"
#include "tn/wav.h"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
  std::string weights_path, in_dir, out_dir;
  int block = 480;
  size_t limit = SIZE_MAX;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s 需要一个参数\n", a.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--weights") weights_path = val();
    else if (a == "--in") in_dir = val();
    else if (a == "--out") out_dir = val();
    else if (a == "--block") block = std::atoi(val().c_str());
    else if (a == "--limit") limit = std::strtoul(val().c_str(), nullptr, 10);
    else {
      std::fprintf(stderr, "未知参数 %s\n", a.c_str());
      return 2;
    }
  }
  if (weights_path.empty() || in_dir.empty() || out_dir.empty() || block < 0) {
    std::fprintf(stderr, "用法：%s --weights weights.bin --in 目录 --out 目录 [--block 480] [--limit N]\n", argv[0]);
    return 2;
  }
  try {
    std::ifstream wf(weights_path, std::ios::binary);
    if (!wf) throw std::runtime_error("打不开 " + weights_path);
    const std::vector<uint8_t> wb((std::istreambuf_iterator<char>(wf)), std::istreambuf_iterator<char>());
    tn::Weights weights(wb.data(), wb.size());
    auto model = std::make_unique<tn::Denoiser>(weights);
    tn::BlockAdapter ad(*model, block);

    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(in_dir))
      if (e.path().extension() == ".wav") files.push_back(e.path());
    std::sort(files.begin(), files.end());
    if (files.size() > limit) files.resize(limit);
    if (files.empty()) throw std::runtime_error(in_dir + " 里没有 wav");
    fs::create_directories(out_dir);

    const int lat = ad.latency(), chunk = block > 0 ? block : 480;
    std::printf("权重 %s；%zu 个文件；宿主块 %d 样本；延迟 %d 样本 = %.2f ms\n", weights_path.c_str(), files.size(), block, lat, 1e3 * lat / tn::SR);
    double audio_s = 0, wall_s = 0;
    for (const auto& p : files) {
      const auto a = tn::wav::read(p.string());
      if (a.sr != tn::SR || a.channels != 1) throw std::runtime_error(p.string() + "：需要 48kHz 单声道");
      const size_t n = a.x.size();
      std::vector<float> x(a.x), y(n + lat + chunk, 0.f);
      x.resize(n + lat + chunk, 0.f);  // 后面补零，把延迟的尾巴冲出来
      ad.reset();
      const auto t0 = std::chrono::steady_clock::now();
      for (size_t pos = 0; pos < n + lat; pos += chunk) ad.process(x.data() + pos, y.data() + pos, chunk);
      wall_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      audio_s += double(n) / tn::SR;
      if (ad.underruns()) throw std::runtime_error("输出不够（underrun），延迟推导有误");
      tn::wav::write_pcm16((fs::path(out_dir) / p.filename()).string(), y.data() + lat, n, tn::SR);
    }
    std::printf("音频 %.0f s，处理 %.2f s（RTF≈%.4f，单线程，含 BlockAdapter）\n", audio_s, wall_s, wall_s / audio_s);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "错误：%s\n", e.what());
    return 1;
  }
  return 0;
}
