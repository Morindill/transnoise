# C++ 核心的构建、对拍和命令行工具。只依赖 g++（C++20）和 make，不依赖任何第三方库。
#   make test      # 编译并跑全部对拍（需要先生成 gen/ 和夹具，见下）
#   make tn_wav    # 命令行工具 build/tn_wav（增强一个目录的 wav，用来给评测打分）
# 生成权重和常量：
#   uv run transnoise export --ckpt data/checkpoints/<名>/best
#   uv run --no-sync python -m transnoise.deploy.pack data/deploy/<名>/weights.npz --out-dir cpp/gen
#   uv run --no-sync python -m transnoise.deploy.fixture data/deploy/<名>/weights.npz --out cpp/gen/fixture.bin
# 用 C++ 输出打分（评测流水线不用改）：
#   cpp/build/tn_wav --weights cpp/gen/weights.bin --in data/eval/testsets/noise/noisy --out data/eval/runs/noise/cpp/wav
#   uv run transnoise eval score noise data/eval/runs/noise/cpp/wav
CXX      ?= g++
# 不开 -ffast-math：保持浮点语义，结果与参考实现一致；-march=native 只用于本机对拍，插件用固定的指令集（见 plugin/CMakeLists.txt）
CXXFLAGS ?= -std=c++20 -O3 -march=native -Wall -Wextra -Wno-unused-parameter
GEN      ?= gen
INC      := -I$(GEN) -Iinclude
HDRS     := $(wildcard include/tn/*.h tests/*.h) $(GEN)/config.h

.PHONY: test tn_wav clean
test: build/test_fixture build/test_adapter
	build/test_fixture $(GEN)/weights.bin $(GEN)/fixture.bin
	build/test_adapter $(GEN)/weights.bin $(GEN)/fixture.bin

tn_wav: build/tn_wav

build/%: tests/%.cpp $(HDRS)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(INC) $< -o $@

build/tn_wav: tools/tn_wav.cpp $(HDRS)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(INC) $< -o $@

clean:
	rm -rf build
