// sherpa-onnx/csrc/online-stream.cc
//
// Copyright (c)  2023  Xiaomi Corporation
#include "sherpa-onnx/csrc/online-stream.h"

#include <memory>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/features.h"

namespace sherpa_onnx {

class OnlineStream::Impl {
 public:
  explicit Impl(const FeatureExtractorConfig &config,
                ContextGraphPtr context_graph)
      : feat_extractor_(config), context_graph_(context_graph) {}

  void AcceptWaveform(int32_t sampling_rate, const float *waveform, int32_t n) {
    feat_extractor_.AcceptWaveform(sampling_rate, waveform, n);
  }

  void InputFinished() const { feat_extractor_.InputFinished(); }

  int32_t NumFramesReady() const {
    return feat_extractor_.NumFramesReady() - start_frame_index_;
  }

  bool IsLastFrame(int32_t frame) const {
    return feat_extractor_.IsLastFrame(frame);
  }

  std::vector<float> GetFrames(int32_t frame_index, int32_t n) const {
    return feat_extractor_.GetFrames(frame_index + start_frame_index_, n);
  }

  void Reset() {
    // we don't reset the feature extractor
    start_frame_index_ += num_processed_frames_;
    num_processed_frames_ = 0;
  }

  int32_t &GetNumProcessedFrames() { return num_processed_frames_; }

  int32_t GetNumFramesSinceStart() const { return start_frame_index_; }

  int32_t &GetCurrentSegment() { return segment_; }

  void SetResult(const OnlineTransducerDecoderResult &r) { result_ = r; }

  OnlineTransducerDecoderResult &GetResult() { return result_; }

  OnlineCtcDecoderResult &GetCtcResult() { return ctc_result_; }

  void SetCtcResult(const OnlineCtcDecoderResult &r) { ctc_result_ = r; }

  int32_t FeatureDim() const { return feat_extractor_.FeatureDim(); }

  void SetStates(std::vector<Ort::Value> states) {
    states_ = std::move(states);
  }

  std::vector<Ort::Value> &GetStates() { return states_; }

  const ContextGraphPtr &GetContextGraph() const { return context_graph_; }

  void SetFasterDecoder(std::unique_ptr<kaldi_decoder::FasterDecoder> decoder) {
    faster_decoder_ = std::move(decoder);
  }

  kaldi_decoder::FasterDecoder *GetFasterDecoder() const {
    return faster_decoder_.get();
  }

  int32_t &GetFasterDecoderProcessedFrames() {
    return faster_decoder_processed_frames_;
  }

 private:
  FeatureExtractor feat_extractor_;
  /// For contextual-biasing
  ContextGraphPtr context_graph_;
  int32_t num_processed_frames_ = 0;  // before subsampling
  int32_t start_frame_index_ = 0;     // never reset
  int32_t segment_ = 0;
  OnlineTransducerDecoderResult result_;
  OnlineCtcDecoderResult ctc_result_;
  std::vector<Ort::Value> states_;  // states for transducer or ctc models
  std::unique_ptr<kaldi_decoder::FasterDecoder> faster_decoder_;
  int32_t faster_decoder_processed_frames_ = 0;
};

OnlineStream::OnlineStream(const FeatureExtractorConfig &config /*= {}*/,
                           ContextGraphPtr context_graph /*= nullptr */)
    : impl_(std::make_unique<Impl>(config, context_graph)) {}

OnlineStream::~OnlineStream() = default;

void OnlineStream::AcceptWaveform(int32_t sampling_rate, const float *waveform,
                                  int32_t n) const {
  impl_->AcceptWaveform(sampling_rate, waveform, n);
}

void OnlineStream::InputFinished() const { impl_->InputFinished(); }

int32_t OnlineStream::NumFramesReady() const { return impl_->NumFramesReady(); }

bool OnlineStream::IsLastFrame(int32_t frame) const {
  return impl_->IsLastFrame(frame);
}

std::vector<float> OnlineStream::GetFrames(int32_t frame_index,
                                           int32_t n) const {
  return impl_->GetFrames(frame_index, n);
}

void OnlineStream::Reset() { impl_->Reset(); }

int32_t OnlineStream::FeatureDim() const { return impl_->FeatureDim(); }

int32_t &OnlineStream::GetNumProcessedFrames() {
  return impl_->GetNumProcessedFrames();
}

int32_t OnlineStream::GetNumFramesSinceStart() const {
  return impl_->GetNumFramesSinceStart();
}

int32_t &OnlineStream::GetCurrentSegment() {
  return impl_->GetCurrentSegment();
}

void OnlineStream::SetResult(const OnlineTransducerDecoderResult &r) {
  impl_->SetResult(r);
}

OnlineTransducerDecoderResult &OnlineStream::GetResult() {
  return impl_->GetResult();
}


OnlineCtcDecoderResult &OnlineStream::GetCtcResult() {
  return impl_->GetCtcResult();
}

void OnlineStream::SetCtcResult(const OnlineCtcDecoderResult &r) {
  impl_->SetCtcResult(r);
}

void OnlineStream::SetStates(std::vector<Ort::Value> states) {
  impl_->SetStates(std::move(states));
}

std::vector<Ort::Value> &OnlineStream::GetStates() {
  return impl_->GetStates();
}

const ContextGraphPtr &OnlineStream::GetContextGraph() const {
  return impl_->GetContextGraph();
}

void OnlineStream::SetFasterDecoder(
    std::unique_ptr<kaldi_decoder::FasterDecoder> decoder) {
  impl_->SetFasterDecoder(std::move(decoder));
}

kaldi_decoder::FasterDecoder *OnlineStream::GetFasterDecoder() const {
  return impl_->GetFasterDecoder();
}

int32_t &OnlineStream::GetFasterDecoderProcessedFrames() {
  return impl_->GetFasterDecoderProcessedFrames();
}

}  // namespace sherpa_onnx