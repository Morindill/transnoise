#pragma once
#include <juce_audio_processors/juce_audio_processors.h>

#include <memory>
#include <vector>

#include "tn/stream_adapter.h"

// Denoising plugin: one model instance per channel (sharing one set of weights); a BlockAdapter connects any host block size to the model's 10 ms frames.
// Parameters: Mix (dry/wet, 0-100%) and Bypass. The dry signal is aligned to the plugin latency, so neither Mix nor Bypass introduces a phase/time misalignment.
// 48 kHz only: at other sample rates the plugin passes audio through (no processing, no latency reported).
class TransnoiseProcessor : public juce::AudioProcessor {
 public:
  TransnoiseProcessor();

  void prepareToPlay(double sampleRate, int samplesPerBlock) override;
  void releaseResources() override {}
  bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
  void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
  void processBlockBypassed(juce::AudioBuffer<float>& b, juce::MidiBuffer& m) override { processBlock(b, m); }  // bypass must go through the delay alignment too

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
    std::vector<float> ring;  // dry-signal delay line
    int w = 0;
  };

  static constexpr int kChunk = 2048;  // process at most this many samples at a time; larger host blocks are split (in variable-block mode splitting does not affect the latency)

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
