# Transnoise VST3 插件（JUCE，AGPLv3 下使用）。在 Windows 上用 build_win.bat 构建。
#   依赖：../cpp/include（模型核心，零依赖）、../cpp/gen/{config.h,weights.bin}（由 deploy.pack 生成，weights.bin 会被编进插件）
cmake_minimum_required(VERSION 3.22)
project(Transnoise VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(TN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../cpp" CACHE PATH "模型核心目录（含 include/ 和 gen/）")

include(FetchContent)
# 用源码压缩包而不是 git clone：工程目录在 WSL 文件系统里，Windows 的 git 会因为文件属主不同拒绝在里面操作（dubious ownership）
FetchContent_Declare(juce
  URL https://github.com/juce-framework/JUCE/archive/refs/tags/8.0.17-security.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(juce)

juce_add_plugin(Transnoise
  COMPANY_NAME "Morindill"
  IS_SYNTH FALSE
  NEEDS_MIDI_INPUT FALSE
  NEEDS_MIDI_OUTPUT FALSE
  PLUGIN_MANUFACTURER_CODE Mrnd
  PLUGIN_CODE Trns
  FORMATS VST3 Standalone
  VST3_CATEGORIES Fx
  PRODUCT_NAME "Transnoise"
  COPY_PLUGIN_AFTER_BUILD FALSE)

# 权重编进插件（BinaryData::weights_bin）
juce_add_binary_data(TnWeights SOURCES "${TN_ROOT}/gen/weights.bin")

target_sources(Transnoise PRIVATE Source/PluginProcessor.cpp Source/PluginProcessor.h)
target_include_directories(Transnoise PRIVATE "${TN_ROOT}/include" "${TN_ROOT}/gen")
target_compile_definitions(Transnoise PUBLIC JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0 JUCE_VST3_CAN_REPLACE_VST2=0)
target_link_libraries(Transnoise
  PRIVATE TnWeights juce::juce_audio_utils
  PUBLIC juce::juce_recommended_config_flags juce::juce_recommended_warning_flags)

# 模型核心的矩阵乘向量靠编译器自动向量化；默认只有 SSE2，开 AVX2 约快一倍（需要 2013 年以后的 CPU）
# /utf-8：我们的源码是 UTF-8（带中文注释），MSVC 在中文系统上默认按 GBK(936) 读，会把注释里的汉字吞掉换行/引号
if(MSVC)
  target_compile_options(Transnoise PRIVATE /arch:AVX2 /utf-8)
endif()
