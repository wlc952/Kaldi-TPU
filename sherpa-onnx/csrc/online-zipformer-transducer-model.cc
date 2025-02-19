// sherpa-onnx/csrc/online-zipformer-transducer-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-zipformer-transducer-model.h"

#include <assert.h>
#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include "sherpa-onnx/csrc/cat.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-transducer-decoder.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/unbind.h"


using namespace bmruntime;

namespace sherpa_onnx {

static std::string shape_to_str(const bm_shape_t& shape) {
    std::string str ="[ ";
    for(int i=0; i<shape.num_dims; i++){
      str += std::to_string(shape.dims[i]) + " ";
    }
    str += "]";
    return str;
}

void showInfo(const bm_net_info_t* m_netinfo)
{
    const char* dtypeMap[] = {
    "FLOAT32",
    "FLOAT16",
    "INT8",
    "UINT8",
    "INT16",
    "UINT16",
    "INT32",
    "UINT32",
    };
    printf("\n########################\n");
    printf("NetName: %s\n", m_netinfo->name);
    for(int s=0; s<m_netinfo->stage_num; s++){
        printf("---- stage %d ----\n", s);
        for(int i=0; i<m_netinfo->input_num; i++){
            auto shapeStr = shape_to_str(m_netinfo->stages[s].input_shapes[i]);
            printf("  Input %d) '%s' shape=%s dtype=%s scale=%g\n",
                i,
                m_netinfo->input_names[i],
                shapeStr.c_str(),
                dtypeMap[m_netinfo->input_dtypes[i]],
                m_netinfo->input_scales[i]);
        }
        for(int i=0; i<m_netinfo->output_num; i++){
            auto shapeStr = shape_to_str(m_netinfo->stages[s].output_shapes[i]);
            printf("  Output %d) '%s' shape=%s dtype=%s scale=%g\n",
                i,
                m_netinfo->output_names[i],
                shapeStr.c_str(),
                dtypeMap[m_netinfo->output_dtypes[i]],
                m_netinfo->output_scales[i]);
        }
    }
    printf("########################\n\n");

}

OnlineZipformerTransducerModel::OnlineZipformerTransducerModel(
    const OnlineModelConfig &config)
    : env_(ORT_LOGGING_LEVEL_WARNING),
      config_(config),
      sess_opts_(GetSessionOptions(config)),
      allocator_{} {
  {
    InitEncoder(config.transducer.encoder);
    // showInfo(encoder_net->info());
  }

  {
    InitDecoder(config.transducer.decoder);
    // showInfo(decoder_net->info());
  }

  {
    InitJoiner(config.transducer.joiner);
    // showInfo(joiner_net->info());
  }
}

void OnlineZipformerTransducerModel::InitEncoder(const std::string &model_path) {
    encoder_ctx = std::make_shared<Context>(dev_id);
    bm_status_t status = encoder_ctx->load_bmodel(model_path.c_str());
    assert(BM_SUCCESS == status);

    // create Network
    std::vector<const char *> network_names;
    encoder_ctx->get_network_names(&network_names);
    encoder_net = std::make_shared<Network>(*encoder_ctx, network_names[0], 0); // use stage[0]
    assert(encoder_net->info()->input_num == enc_in_num && encoder_net->info()->output_num == enc_out_num);

    // set meta data
    encoder_dims_ = {384, 384, 384, 384, 384};
    attention_dims_ = {192, 192, 192, 192, 192};
    num_encoder_layers_ = {2, 4, 3, 2, 4};
    cnn_module_kernels_ = {31, 31, 31, 31, 31};
    left_context_len_ = {64, 32, 16, 8, 32};
    T_ = 39;
    decode_chunk_len_ = 32;
}

void OnlineZipformerTransducerModel::InitDecoder(const std::string &model_path) {
    decoder_ctx = std::make_shared<Context>(dev_id);
    bm_status_t status = decoder_ctx->load_bmodel(model_path.c_str());
    assert(BM_SUCCESS == status);

    // create Network
    std::vector<const char *> network_names;
    decoder_ctx->get_network_names(&network_names);
    decoder_net = std::make_shared<Network>(*decoder_ctx, network_names[0], 0); // use stage[0]
    assert(decoder_net->info()->input_num == dec_in_num && decoder_net->info()->output_num == dec_out_num);

    // set meta data
    vocab_size_ = 6254;
    context_size_ = 2;
}

void OnlineZipformerTransducerModel::InitJoiner(const std::string &model_path) {
    joiner_ctx = std::make_shared<Context>(dev_id);
    bm_status_t status = joiner_ctx->load_bmodel(model_path.c_str());
    assert(BM_SUCCESS == status);

    // create Network
    std::vector<const char *> network_names;
    joiner_ctx->get_network_names(&network_names);
    joiner_net = std::make_shared<Network>(*joiner_ctx, network_names[0], 0); // use stage[0]
    assert(joiner_net->info()->input_num == jo_in_num && joiner_net->info()->output_num == jo_out_num);

    // set meta data
    // joiner_dim = 512
}

void OnlineZipformerTransducerModel::ReleaseModels() {
  // free_runtime(encoder_);
  // free_runtime(decoder_sess_);
  // free_runtime(joiner_sess_);
}

OnlineZipformerTransducerModel::~OnlineZipformerTransducerModel() {
  ReleaseModels();
}

std::vector<Ort::Value> OnlineZipformerTransducerModel::StackStates(
    const std::vector<std::vector<Ort::Value>> &states) const {
  int32_t batch_size = static_cast<int32_t>(states.size());
  int32_t num_encoders = static_cast<int32_t>(num_encoder_layers_.size());

  std::vector<const Ort::Value *> buf(batch_size);

  std::vector<Ort::Value> ans;
  ans.reserve(states[0].size());

  // cached_len
  for (int32_t i = 0; i != num_encoders; ++i) {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = &states[n][i];
    }
    auto v = Cat<int64_t>(allocator_, buf, 1);  // (num_layers, 1)
    ans.push_back(std::move(v));
  }

  // cached_avg
  for (int32_t i = 0; i != num_encoders; ++i) {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = &states[n][num_encoders + i];
    }
    auto v = Cat(allocator_, buf, 1);  // (num_layers, 1, encoder_dims)
    ans.push_back(std::move(v));
  }

  // cached_key
  for (int32_t i = 0; i != num_encoders; ++i) {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = &states[n][num_encoders * 2 + i];
    }
    // (num_layers, left_context_len, 1, attention_dims)
    auto v = Cat(allocator_, buf, 2);
    ans.push_back(std::move(v));
  }

  // cached_val
  for (int32_t i = 0; i != num_encoders; ++i) {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = &states[n][num_encoders * 3 + i];
    }
    // (num_layers, left_context_len, 1, attention_dims/2)
    auto v = Cat(allocator_, buf, 2);
    ans.push_back(std::move(v));
  }

  // cached_val2
  for (int32_t i = 0; i != num_encoders; ++i) {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = &states[n][num_encoders * 4 + i];
    }
    // (num_layers, left_context_len, 1, attention_dims/2)
    auto v = Cat(allocator_, buf, 2);
    ans.push_back(std::move(v));
  }

  // cached_conv1
  for (int32_t i = 0; i != num_encoders; ++i) {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = &states[n][num_encoders * 5 + i];
    }
    // (num_layers, 1, encoder_dims, cnn_module_kernels-1)
    auto v = Cat(allocator_, buf, 1);
    ans.push_back(std::move(v));
  }

  // cached_conv2
  for (int32_t i = 0; i != num_encoders; ++i) {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = &states[n][num_encoders * 6 + i];
    }
    // (num_layers, 1, encoder_dims, cnn_module_kernels-1)
    auto v = Cat(allocator_, buf, 1);
    ans.push_back(std::move(v));
  }

  return ans;
}

std::vector<std::vector<Ort::Value>>
OnlineZipformerTransducerModel::UnStackStates(
    const std::vector<Ort::Value> &states) const {
  assert(states.size() == num_encoder_layers_.size() * 7);

  int32_t batch_size = states[0].GetTensorTypeAndShapeInfo().GetShape()[1];
  int32_t num_encoders = num_encoder_layers_.size();

  std::vector<std::vector<Ort::Value>> ans;
  ans.resize(batch_size);

  // cached_len
  for (int32_t i = 0; i != num_encoders; ++i) {
    auto v = Unbind<int64_t>(allocator_, &states[i], 1);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  // cached_avg
  for (int32_t i = num_encoders; i != 2 * num_encoders; ++i) {
    auto v = Unbind(allocator_, &states[i], 1);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  // cached_key
  for (int32_t i = 2 * num_encoders; i != 3 * num_encoders; ++i) {
    auto v = Unbind(allocator_, &states[i], 2);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  // cached_val
  for (int32_t i = 3 * num_encoders; i != 4 * num_encoders; ++i) {
    auto v = Unbind(allocator_, &states[i], 2);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  // cached_val2
  for (int32_t i = 4 * num_encoders; i != 5 * num_encoders; ++i) {
    auto v = Unbind(allocator_, &states[i], 2);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  // cached_conv1
  for (int32_t i = 5 * num_encoders; i != 6 * num_encoders; ++i) {
    auto v = Unbind(allocator_, &states[i], 1);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  // cached_conv2
  for (int32_t i = 6 * num_encoders; i != 7 * num_encoders; ++i) {
    auto v = Unbind(allocator_, &states[i], 1);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  return ans;
}

std::vector<Ort::Value> OnlineZipformerTransducerModel::GetEncoderInitStates() {
  // Please see
  // https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless7_streaming/zipformer.py#L673
  // for details

  int32_t n = static_cast<int32_t>(encoder_dims_.size());
  std::vector<Ort::Value> cached_len_vec;
  std::vector<Ort::Value> cached_avg_vec;
  std::vector<Ort::Value> cached_key_vec;
  std::vector<Ort::Value> cached_val_vec;
  std::vector<Ort::Value> cached_val2_vec;
  std::vector<Ort::Value> cached_conv1_vec;
  std::vector<Ort::Value> cached_conv2_vec;

  cached_len_vec.reserve(n);
  cached_avg_vec.reserve(n);
  cached_key_vec.reserve(n);
  cached_val_vec.reserve(n);
  cached_val2_vec.reserve(n);
  cached_conv1_vec.reserve(n);
  cached_conv2_vec.reserve(n);

  for (int32_t i = 0; i != n; ++i) {
    {
      std::array<int64_t, 2> s{num_encoder_layers_[i], 1};
      auto v =
          Ort::Value::CreateTensor<int64_t>(allocator_, s.data(), s.size());
      Fill<int64_t>(&v, 0);
      cached_len_vec.push_back(std::move(v));
    }

    {
      std::array<int64_t, 3> s{num_encoder_layers_[i], 1, encoder_dims_[i]};
      auto v = Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
      Fill(&v, 0);
      cached_avg_vec.push_back(std::move(v));
    }

    {
      std::array<int64_t, 4> s{num_encoder_layers_[i], left_context_len_[i], 1,
                               attention_dims_[i]};
      auto v = Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
      Fill(&v, 0);
      cached_key_vec.push_back(std::move(v));
    }

    {
      std::array<int64_t, 4> s{num_encoder_layers_[i], left_context_len_[i], 1,
                               attention_dims_[i] / 2};
      auto v = Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
      Fill(&v, 0);
      cached_val_vec.push_back(std::move(v));
    }

    {
      std::array<int64_t, 4> s{num_encoder_layers_[i], left_context_len_[i], 1,
                               attention_dims_[i] / 2};
      auto v = Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
      Fill(&v, 0);
      cached_val2_vec.push_back(std::move(v));
    }

    {
      std::array<int64_t, 4> s{num_encoder_layers_[i], 1, encoder_dims_[i],
                               cnn_module_kernels_[i] - 1};
      auto v = Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
      Fill(&v, 0);
      cached_conv1_vec.push_back(std::move(v));
    }

    {
      std::array<int64_t, 4> s{num_encoder_layers_[i], 1, encoder_dims_[i],
                               cnn_module_kernels_[i] - 1};
      auto v = Ort::Value::CreateTensor<float>(allocator_, s.data(), s.size());
      Fill(&v, 0);
      cached_conv2_vec.push_back(std::move(v));
    }
  }

  std::vector<Ort::Value> ans;
  ans.reserve(n * 7);

  for (auto &v : cached_len_vec) ans.push_back(std::move(v));
  for (auto &v : cached_avg_vec) ans.push_back(std::move(v));
  for (auto &v : cached_key_vec) ans.push_back(std::move(v));
  for (auto &v : cached_val_vec) ans.push_back(std::move(v));
  for (auto &v : cached_val2_vec) ans.push_back(std::move(v));
  for (auto &v : cached_conv1_vec) ans.push_back(std::move(v));
  for (auto &v : cached_conv2_vec) ans.push_back(std::move(v));

  return ans;
}

std::pair<Ort::Value, std::vector<Ort::Value>>
OnlineZipformerTransducerModel::RunEncoder(Ort::Value features,
                                           std::vector<Ort::Value> states,
                                           Ort::Value /* processed_frames */) {
  
  auto &bm_encoder_inputs = encoder_net->Inputs();
  auto &bm_encoder_outputs = encoder_net->Outputs();

  std::vector<Ort::Value> encoder_inputs;
  encoder_inputs.reserve(1 + states.size());

  std::cout << "&&& RunEncoder ========================\n features( in 0 )" << std::endl;
  // Print3(&features)

  encoder_inputs.push_back(std::move(features));
  for (auto &v : states) {
    encoder_inputs.push_back(std::move(v));
  }

  LoadOrtValuesToBMTensors(encoder_inputs, bm_encoder_inputs, enc_in_num);
  
  bool status = encoder_net->Forward();
  assert(BM_SUCCESS == status);

  std::vector<Ort::Value> next_states = GetOrtValuesFromBMTensors(bm_encoder_outputs.begin()+1, enc_out_num-1);

  return {std::move(GetOrtValueFromBMTensor(bm_encoder_outputs[0])), std::move(next_states)};
}

Ort::Value OnlineZipformerTransducerModel::RunDecoder(Ort::Value decoder_input) {
  ConvertOrtValueToBMTensor(decoder_input, decoder_net->Inputs()[0]);
  bool status = decoder_net->Forward();
  assert(BM_SUCCESS == status);
  return std::move(GetOrtValueFromBMTensor(decoder_net->Outputs()[0]));
}

Ort::Value OnlineZipformerTransducerModel::RunJoiner(Ort::Value encoder_out,
                                                     Ort::Value decoder_out) {
  std::vector<Ort::Value> temp = {};
  temp.push_back(std::move(encoder_out));
  temp.push_back(std::move(decoder_out));
  LoadOrtValuesToBMTensors(temp, joiner_net->Inputs(), jo_in_num);
  bool status = joiner_net->Forward();
  assert(BM_SUCCESS == status);
  return std::move(GetOrtValueFromBMTensor(joiner_net->Outputs()[0]));
}

}  // namespace sherpa_onnx
