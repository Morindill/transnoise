#pragma once
#include <juce_audio_processors/juce_audio_processors.h>

#include <memory>
#include <vector>

#include "tn/stream_adapter.h"

// 降噪插件：每个声道一个模型实例（共用一份权重），经 BlockAdapter 把任意宿主块大小接到模型的 10 ms 帧上。
// 参数：Mix（干湿，0–100%）、Bypass。干声按插件延迟对齐，所以 Mix 和 Bypass 都不会引入相位/时间错位。
// 只支持 48 kHz：其它采样率下插件直通（不处理，也不报延迟）。
class TransnoiseProcessor : public juce::AudioProcessor {
 public:
  TransnoiseProcessor();

  void prepareToPlay(double sampleRate, int samplesPerBlock) override;
  void releaseResources() override {}
  bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
  void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
  void processBlockBypassed(juce::AudioBuffer<float>& b, juce::MidiBuffer& m) override { processBlock(b, m); }  // 旁通也要走延迟对齐

  juce::AudioProcessorEditor* createEditor() override;
  bool hasEditor() const override { return true; }
  const juce::String getName() const override { return JucePlugin_Name; }
  bool acceptsMidi() const override { return false; }
  bool producesMidi() const override { return false; }
  double getTailLengthSeconds() const override { return 0.0; }
  int getNumPrograms() override { return 1; }
  int getCurrentProgram() override { return 0; }
  void setCurrentProgram(int) override {}
  const juce::String getProgramName(int) override { return {}; }
  void changeProgramName(int, const juce::String&) override {}
  void getStateInformation(juce::MemoryBlock& dest) override;
  void setStateInformation(const void* data, int size) override;
  juce::AudioProcessorParameter* getBypassParameter() const override { return bypass_; }

 private:
  static juce::AudioProcessorValueTreeState::ParameterLayout createLayout();

  struct Channel {
    std::unique_ptr<tn::Denoiser> model;
    std::unique_ptr<tn::BlockAdapter> adapter;
    std::vector<float> ring;  // 干声延迟线
    int w = 0;
  };

  static constexpr int kChunk = 2048;  // 一次最多处理这么多样本；宿主的块更大就分段（非固定块模式下分段不影响延迟）

  juce::AudioProcessorValueTreeState apvts_;
  juce::AudioParameterBool* bypass_ = nullptr;
  std::atomic<float>* mix_param_ = nullptr;
  juce::SmoothedValue<float> mix_;

  std::unique_ptr<tn::Weights> weights_;
  std::vector<Channel> ch_;
  std::vector<float> wet_, dry_, gains_;
  bool model_ok_ = false, rate_ok_ = false;
  int latency_ = 0;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TransnoiseProcessor)
};
