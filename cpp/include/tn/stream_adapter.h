// 中间件：任意宿主块大小（1 … 任意个样本，每次可以不同）↔ 模型的 HOP 样本一帧。
// 所有使用 Denoiser 的外壳（tn_wav 命令行、plugin/ 里的插件）都通过它调用模型，宿主块大小的细节只在这里处理。
//
// 输入按样本进 FIFO，攒够 HOP 个就 push 一帧；输出从一个预先填了 latency() 个零的 FIFO 里按样本出。
// 输出 = 输入延迟 latency() 个样本（整个流是时间对齐的 + 固定延迟），与宿主怎么切块无关：同一段输入用任何分块方式，
// 输出逐比特相同（test_adapter 验证）。
//
// 延迟：模型本身要等 LA+1 帧；宿主块不是 HOP 的整数倍时，还要多等一点才能保证每次调用都有足够的输出样本。
//   固定块大小 B：latency = (LA+1)·HOP + HOP − gcd(B, HOP)（B 是 HOP 的倍数时 = (LA+1)·HOP，最小）
//   块大小不固定：latency = (LA+2)·HOP − 1（最坏情况）
// 这个数就是应该报告给宿主的延迟（setLatencySamples）。
#pragma once
#include <numeric>
#include <vector>

#include "model.h"

namespace tn {

class BlockAdapter {
 public:
  // fixed_block > 0：宿主块大小固定，用最小延迟；0：块大小不固定，用最坏情况延迟。构造时分配内存，之后 process 不分配。
  // max_block：宿主一次调用最多给多少样本（超过的调用不处理，输出静音）。
  explicit BlockAdapter(Denoiser& model, int fixed_block = 0, int max_block = 16384)
      : m_(model),
        max_block_(max_block),
        latency_(fixed_block > 0 ? (LA + 1) * HOP + HOP - std::gcd(fixed_block, HOP) : (LA + 2) * HOP - 1),
        cap_(latency_ + max_block + 2 * HOP + 1),  // 一次调用里先产出再取走，峰值 = 调用前的存量（≤ latency）+ 本块产出（≤ n + HOP）
        ring_(cap_, 0.f) {
    reset();
  }

  int latency() const { return latency_; }
  int underruns() const { return underruns_; }  // 应该始终是 0；非 0 说明延迟的推导有漏洞

  void reset() {
    m_.reset();
    std::fill(ring_.begin(), ring_.end(), 0.f);
    head_ = 0;
    count_ = latency_;  // 预填 latency_ 个零
    in_fill_ = 0;
    underruns_ = 0;
  }

  // n 个样本进，n 个样本出；in 和 out 可以是同一块内存（先把输入全部读完，再写输出）。实时安全。
  // 必须"先喂完这一块输入、再取这一块输出"：延迟的推导只在调用边界上成立，块内逐样本交错会在块中间缺输出。
  void process(const float* in, float* out, int n) {
    if (n > max_block_) {  // 超出构造时声明的上限：不处理，输出静音（不能在实时线程里扩容）
      std::fill(out, out + n, 0.f);
      return;
    }
    for (int i = 0; i < n; ++i) {
      in_[in_fill_++] = in[i];
      if (in_fill_ == HOP) {
        float o[HOP];
        if (m_.push(in_, o))
          for (int k = 0; k < HOP; ++k) {
            ring_[(head_ + count_) % cap_] = o[k];
            ++count_;
          }
        in_fill_ = 0;
      }
    }
    for (int i = 0; i < n; ++i) {
      if (count_ > 0) {
        out[i] = ring_[head_];
        head_ = head_ + 1 == cap_ ? 0 : head_ + 1;
        --count_;
      } else {
        out[i] = 0.f;
        ++underruns_;
      }
    }
  }

 private:
  Denoiser& m_;
  const int max_block_, latency_, cap_;
  std::vector<float> ring_;
  int head_ = 0, count_ = 0, in_fill_ = 0, underruns_ = 0;
  float in_[HOP] = {};
};

}  // namespace tn
