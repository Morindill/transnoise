// Minimal WAV reading and writing (for the command-line tool and tests, not the real-time path): reads PCM16 / PCM24 / PCM32 / float32, writes PCM16.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tn::wav {

struct Audio {
  int sr = 0, channels = 0;
  std::vector<float> x;  // interleaved samples
};

inline Audio read(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("打不开 " + path);
  auto u32 = [&] { uint32_t v = 0; f.read(reinterpret_cast<char*>(&v), 4); return v; };
  auto u16 = [&] { uint16_t v = 0; f.read(reinterpret_cast<char*>(&v), 2); return v; };
  char tag[4];
  f.read(tag, 4);
  if (std::memcmp(tag, "RIFF", 4) != 0) throw std::runtime_error(path + "：不是 RIFF/WAV");
  u32();
  f.read(tag, 4);
  if (std::memcmp(tag, "WAVE", 4) != 0) throw std::runtime_error(path + "：不是 WAVE");
  Audio a;
  int fmt = 0, bits = 0;
  while (f.read(tag, 4)) {
    const uint32_t size = u32();
    if (std::memcmp(tag, "fmt ", 4) == 0) {
      fmt = u16();
      a.channels = u16();
      a.sr = static_cast<int>(u32());
      u32(), u16();
      bits = u16();
      if (fmt == 0xFFFE && size >= 26) {  // WAVE_FORMAT_EXTENSIBLE: the real format is in the first two bytes of the sub-format
        u16(), u16(), u32();
        fmt = u16();
        f.seekg(size - 26 + 14, std::ios::cur);
      } else {
        f.seekg(size - 16, std::ios::cur);
      }
    } else if (std::memcmp(tag, "data", 4) == 0) {
      if (!fmt || !bits) throw std::runtime_error(path + "：data 在 fmt 之前");
      const int bytes = bits / 8;
      const size_t n = size / bytes;
      std::vector<uint8_t> raw(size);
      f.read(reinterpret_cast<char*>(raw.data()), size);
      a.x.resize(n);
      for (size_t i = 0; i < n; ++i) {
        const uint8_t* p = raw.data() + i * bytes;
        if (fmt == 3 && bits == 32) {
          std::memcpy(&a.x[i], p, 4);
        } else if (fmt == 1) {
          int32_t v = 0;
          if (bits == 16) v = static_cast<int16_t>(p[0] | (p[1] << 8));
          else if (bits == 24) v = (static_cast<int32_t>(p[0] << 8 | p[1] << 16 | p[2] << 24)) >> 8;
          else if (bits == 32) std::memcpy(&v, p, 4);
          else throw std::runtime_error(path + "：不支持的位深");
          a.x[i] = static_cast<float>(v) / static_cast<float>(1u << (bits - 1));
        } else {
          throw std::runtime_error(path + "：不支持的 WAV 格式");
        }
      }
      return a;
    } else {
      f.seekg(size + (size & 1), std::ios::cur);
    }
  }
  throw std::runtime_error(path + "：没有 data 块");
}

// Mono PCM16 (clipped and rounded, matching how soundfile writes PCM_16)
inline void write_pcm16(const std::string& path, const float* x, size_t n, int sr) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("打不开 " + path);
  auto w32 = [&](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
  auto w16 = [&](uint16_t v) { f.write(reinterpret_cast<const char*>(&v), 2); };
  f.write("RIFF", 4), w32(36 + uint32_t(n) * 2), f.write("WAVEfmt ", 8);
  w32(16), w16(1), w16(1), w32(sr), w32(sr * 2), w16(2), w16(16);
  f.write("data", 4), w32(uint32_t(n) * 2);
  for (size_t i = 0; i < n; ++i) {
    const float v = std::fmin(std::fmax(x[i], -1.0f), 32767.0f / 32768.0f);
    const int16_t s = static_cast<int16_t>(std::lrintf(v * 32768.0f));
    f.write(reinterpret_cast<const char*>(&s), 2);
  }
}

}  // namespace tn::wav
