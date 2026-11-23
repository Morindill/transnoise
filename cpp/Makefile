# Build, cross-check and command-line tool of the C++ core. Needs only g++ (C++20) and make, no third-party libraries.
#   make test      # compile and run all cross-checks (gen/ and the fixture must be generated first, see below)
#   make tn_wav    # command-line tool build/tn_wav (enhances a directory of wavs, for scoring by the evaluation)
# Generate the weights and constants:
#   uv run transnoise export --ckpt data/checkpoints/<name>/best
#   uv run --no-sync python -m transnoise.deploy.pack data/deploy/<name>/weights.npz --out-dir cpp/gen
#   uv run --no-sync python -m transnoise.deploy.fixture data/deploy/<name>/weights.npz --out cpp/gen/fixture.bin
# Score the C++ output (the evaluation pipeline needs no change):
#   cpp/build/tn_wav --weights cpp/gen/weights.bin --in data/eval/testsets/noise/noisy --out data/eval/runs/noise/cpp/wav
#   uv run transnoise eval score noise data/eval/runs/noise/cpp/wav
CXX      ?= g++
# No -ffast-math: keep floating-point semantics so results match the reference; -march=native is for local cross-checks only, the plugin uses a fixed instruction set (see plugin/CMakeLists.txt)
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
