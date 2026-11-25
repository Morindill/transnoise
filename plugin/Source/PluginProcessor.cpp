#include "PluginProcessor.h"

#include <BinaryData.h>

#include <algorithm>
#include <cmath>

TransnoiseProcessor::TransnoiseProcessor()
    : AudioProcessor(BusesProperties()
                         .withInput("Input", juce::AudioChannelSet::stereo(), true)
                         .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      apvts_(*this, nullptr, "STATE", createLayout()) {
  bypass_ = dynamic_cast<juce::AudioParameterBool*>(apvts_.getParameter("bypass"));
  mix_param_ = apvts_.getRawParameterValue("mix");
  try {
    weights_ = std::make_unique<tn::Weights>(reinterpret_cast<const uint8_t*>(BinaryData::weights_bin),
                                             static_cast<size_t>(BinaryData::weights_binSize));
    model_ok_ = true;
  } catch (const std::exception& e) {
    juce::ignoreUnused(e);  // DBG is empty in Release
    DBG("Transnoise: failed to load weights:" << e.what());  // the plugin passes audio through instead of crashing
  }
}

juce::AudioProcessorValueTreeState::ParameterLayout TransnoiseProcessor::createLayout() {
  juce::AudioProcessorValueTreeState::ParameterLayout layout;
  layout.add(std::make_unique<juce::AudioParameterFloat>(
      juce::ParameterID{"mix", 1}, "Mix", juce::NormalisableRange<float>(0.0f, 1.0f, 0.001f), 1.0f,
      juce::AudioParameterFloatAttributes().withStringFromValueFunction([](float v, int) { return juce::String(juce::roundToInt(v * 100.0f)) + " %"; })));
  layout.add(std::make_unique<juce::AudioParameterBool>(juce::ParameterID{"bypass", 1}, "Bypass", false));
  return layout;
}

bool TransnoiseProcessor::isBusesLayoutSupported(const BusesLayout& l) const {
  const auto& out = l.getMainOutputChannelSet();
  return (out == juce::AudioChannelSet::mono() || out == juce::AudioChannelSet::stereo()) && out == l.getMainInputChannelSet();
}

void TransnoiseProcessor::prepareToPlay(double sampleRate, int /*samplesPerBlock*/) {
  rate_ok_ = std::abs(sampleRate - tn::SR) < 0.5;
  ch_.clear();
  latency_ = 0;
  if (model_ok_ && rate_ok_) {
    for (int c = 0; c < getTotalNumInputChannels(); ++c) {
      Channel s;
      s.model = std::make_unique<tn::Denoiser>(*weights_);
      s.adapter = std::make_unique<tn::BlockAdapter>(*s.model, /*fixed_block=*/0, kChunk);  // variable block size: use the worst-case latency, so chunked calls are safe too
      latency_ = s.adapter->latency();
      s.ring.assign(static_cast<size_t>(latency_ + kChunk + 1), 0.0f);
      ch_.push_back(std::move(s));
    }
  }
  setLatencySamples(latency_);  // 0 when the sample rate is wrong or the model failed to load: the plugin passes audio through
  wet_.assign(kChunk, 0.0f);
  dry_.assign(kChunk, 0.0f);
  gains_.assign(kChunk, 0.0f);
  mix_.reset(sampleRate, 0.02);
  mix_.setCurrentAndTargetValue(bypass_->get() ? 0.0f : mix_param_->load());
}

void TransnoiseProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) {
  juce::ScopedNoDenormals noDenormals;
  if (ch_.empty()) return;  // pass-through
  const int n = buffer.getNumSamples();
  const int nch = std::min(buffer.getNumChannels(), static_cast<int>(ch_.size()));
  mix_.setTargetValue(bypass_->get() ? 0.0f : mix_param_->load());

  for (int pos = 0; pos < n; pos += kChunk) {
    const int m = std::min(kChunk, n - pos);
    for (int i = 0; i < m; ++i) gains_[static_cast<size_t>(i)] = mix_.getNextValue();
    for (int c = 0; c < nch; ++c) {
      float* x = buffer.getWritePointer(c) + pos;
      Channel& s = ch_[static_cast<size_t>(c)];
      const int cap = static_cast<int>(s.ring.size());
      for (int i = 0; i < m; ++i) {  // the dry signal goes into the delay line; read back the sample from latency_ samples ago
        s.ring[static_cast<size_t>(s.w)] = x[i];
        int r = s.w - latency_;
        if (r < 0) r += cap;
        dry_[static_cast<size_t>(i)] = s.ring[static_cast<size_t>(r)];
        s.w = s.w + 1 == cap ? 0 : s.w + 1;
      }
      s.adapter->process(x, wet_.data(), m);
      for (int i = 0; i < m; ++i) {
        const float d = dry_[static_cast<size_t>(i)];
        x[i] = d + gains_[static_cast<size_t>(i)] * (wet_[static_cast<size_t>(i)] - d);
      }
    }
  }
}

juce::AudioProcessorEditor* TransnoiseProcessor::createEditor() { return new juce::GenericAudioProcessorEditor(*this); }

void TransnoiseProcessor::getStateInformation(juce::MemoryBlock& dest) {
  if (auto xml = apvts_.copyState().createXml()) copyXmlToBinary(*xml, dest);
}

void TransnoiseProcessor::setStateInformation(const void* data, int size) {
  if (auto xml = getXmlFromBinary(data, size))
    if (xml->hasTagName(apvts_.state.getType())) apvts_.replaceState(juce::ValueTree::fromXml(*xml));
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new TransnoiseProcessor(); }
