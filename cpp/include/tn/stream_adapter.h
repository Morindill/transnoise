// Middleware: any host block size (1 ... any number of samples, possibly different on every call) <-> the model's frames of HOP samples.
// Every shell that uses Denoiser (the tn_wav command line, the plugin in plugin/) calls the model through it, so host block-size details are handled only here.
//
// Input goes sample by sample into a FIFO, and a frame is pushed whenever HOP samples have accumulated; output comes sample by sample from a FIFO pre-filled with latency() zeros.
// Output = input delayed by latency() samples (the whole stream stays time-aligned apart from a fixed delay), independent of how the host cuts blocks: the same input
// gives bit-identical output with any blocking (verified by test_adapter).
//
// Latency: the model itself waits for LA+1 frames; when the host block is not a multiple of HOP, a little more waiting is needed so that every call has enough output samples.
//   Fixed block size B: latency = (LA+1)*HOP + HOP - gcd(B, HOP) (when B is a multiple of HOP this is (LA+1)*HOP, the minimum)
//   Variable block size: latency = (LA+2)*HOP - 1 (worst case)
// This number is the latency to report to the host (setLatencySamples).
#pragma once
#include <numeric>
#include <vector>

#include "model.h"

namespace tn {

class BlockAdapter {
 public:
  // fixed_block > 0: the host block size is fixed, use the minimum latency; 0: block size varies, use the worst-case latency. Memory is allocated at construction, process allocates nothing afterwards.
  // max_block: the most samples the host passes in one call (calls above it are not processed and output silence).
  explicit BlockAdapter(Denoiser& model, int fixed_block = 0, int max_block = 16384)
      : m_(model),
        max_block_(max_block),
        latency_(fixed_block > 0 ? (LA + 1) * HOP + HOP - std::gcd(fixed_block, HOP) : (LA + 2) * HOP - 1),
        cap_(latency_ + max_block + 2 * HOP + 1),  // within one call output is produced before it is taken, so the peak = stock before the call (<= latency) + output of this block (<= n + HOP)
        ring_(cap_, 0.f) {
    reset();
  }

  int latency() const { return latency_; }
  int underruns() const { return underruns_; }  // should always be 0; non-zero means the latency derivation has a hole

  void reset() {
    m_.reset();
    std::fill(ring_.begin(), ring_.end(), 0.f);
    head_ = 0;
    count_ = latency_;  // pre-fill latency_ zeros
    in_fill_ = 0;
    underruns_ = 0;
  }

  // n samples in, n samples out; in and out may be the same memory (all input is read before any output is written). Real-time safe.
  // The block's input must be fed completely before its output is taken: the latency derivation holds only at call boundaries, and interleaving sample by sample inside a block would run short of output mid-block.
  void process(const float* in, float* out, int n) {
    if (n > max_block_) {  // above the limit declared at construction: do not process, output silence (the real-time thread cannot grow the buffers)
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
