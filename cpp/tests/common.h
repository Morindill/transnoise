// 测试程序共用：把整个文件读进内存（权重、夹具）。
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <vector>

inline std::vector<uint8_t> read_file(const char* path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "打不开 %s\n", path);
    std::exit(2);
  }
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}
