// sherpa-onnx/csrc/online-recognizer.cc
//
// Copyright (c)  2023  Xiaomi Corporation
// Copyright (c)  2023  Pingfeng Luo

#include "sherpa-onnx/csrc/online-recognizer.h"

#include <assert.h>

#include <algorithm>
#include <iomanip>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/online-recognizer-impl.h"

namespace sherpa_onnx {

/// Helper for `OnlineRecognizerResult::AsJsonString()`
template<typename T>
std::string VecToString(const std::vector<T>& vec, int32_t precision = 6) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision);
  oss << "[ ";
  std::string sep = "";
  for (const auto& item : vec) {
    oss << sep << item;
    sep = ", ";
  }
  oss << " ]";
  return oss.str();
}

/// Helper for `OnlineRecognizerResult::AsJsonString()`
template<>  // explicit specialization for T = std::string
std::string VecToString<std::string>(const std::vector<std::string>& vec,
                                     int32_t) {  // ignore 2nd arg
  std::ostringstream oss;
  oss << "[ ";
  std::string sep = "";
  for (const auto& item : vec) {
    oss << sep << "\"" << item << "\"";
    sep = ", ";
  }
  oss << " ]";
  return oss.str();
}

std::string OnlineRecognizerResult::AsJsonString() const {
  std::ostringstream os;
  os << "{ ";
  os << "\"text\": " << "\"" << text << "\"" << ", ";
  os << "\"tokens\": " << VecToString(tokens) << ", ";
  os << "\"timestamps\": " << VecToString(timestamps, 2) << ", ";
  os << "\"ys_probs\": " << VecToString(ys_probs, 6) << ", ";
  os << "\"context_scores\": " << VecToString(context_scores, 6) << ", ";
  os << "\"segment\": " << segment << ", ";
  os << "\"start_time\": " << std::fixed << std::setprecision(2)
     << start_time  << ", ";
  os << "\"is_final\": " << (is_final ? "true" : "false");
  os << "}";
  return os.str();
}

void OnlineRecognizerConfig::Register(ParseOptions *po) {
  feat_config.Register(po);
  model_config.Register(po);
  endpoint_config.Register(po);

  po->Register("enable-endpoint", &enable_endpoint,
               "True to enable endpoint detection. False to disable it.");
  po->Register("blank-penalty", &blank_penalty,
               "The penalty applied on blank symbol during decoding. "
               "Note: It is a positive value. "
               "Increasing value will lead to lower deletion at the cost"
               "of higher insertions. "
               "Currently only applicable for transducer models.");\
  po->Register(
      "hotwords-file", &hotwords_file,
      "The file containing hotwords, one words/phrases per line, and for each"
      "phrase the bpe/cjkchar are separated by a space. For example: "
      "▁HE LL O ▁WORLD"
      "你 好 世 界");
  po->Register("decoding-method", &decoding_method,
               "decoding method,"
               "now support greedy_search.");
}

bool OnlineRecognizerConfig::Validate() const {
  if (!hotwords_file.empty()) {
    return false;
  }

  return model_config.Validate();
}

std::string OnlineRecognizerConfig::ToString() const {
  std::ostringstream os;

  os << "OnlineRecognizerConfig(";
  os << "feat_config=" << feat_config.ToString() << ", ";
  os << "model_config=" << model_config.ToString() << ", ";
  os << "endpoint_config=" << endpoint_config.ToString() << ", ";
  os << "enable_endpoint=" << (enable_endpoint ? "True" : "False") << ", ";
  os << "max_active_paths=" << max_active_paths << ", ";
  os << "hotwords_score=" << hotwords_score << ", ";
  os << "hotwords_file=\"" << hotwords_file << "\", ";
  os << "decoding_method=\"" << decoding_method << "\", ";
  os << "blank_penalty=" << blank_penalty << ")";

  return os.str();
}

OnlineRecognizer::OnlineRecognizer(const OnlineRecognizerConfig &config)
    : impl_(OnlineRecognizerImpl::Create(config)) {}


OnlineRecognizer::~OnlineRecognizer() = default;

std::unique_ptr<OnlineStream> OnlineRecognizer::CreateStream() const {
  return impl_->CreateStream();
}

std::unique_ptr<OnlineStream> OnlineRecognizer::CreateStream(
    const std::string &hotwords) const {
  return impl_->CreateStream(hotwords);
}

bool OnlineRecognizer::IsReady(OnlineStream *s) const {
  return impl_->IsReady(s);
}

void OnlineRecognizer::DecodeStreams(OnlineStream **ss, int32_t n) const {
  impl_->DecodeStreams(ss, n);
}

OnlineRecognizerResult OnlineRecognizer::GetResult(OnlineStream *s) const {
  return impl_->GetResult(s);
}

bool OnlineRecognizer::IsEndpoint(OnlineStream *s) const {
  return impl_->IsEndpoint(s);
}

void OnlineRecognizer::Reset(OnlineStream *s) const { impl_->Reset(s); }

}  // namespace sherpa_onnx
