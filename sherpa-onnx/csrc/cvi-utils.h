#ifndef SHERPA_CVIRTL_UTILS_H_
#define SHERPA_CVIRTL_UTILS_H_

#include <cassert>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "cviruntime.h"  // NOLINT

namespace sherpa_onnx {

/**
 * Get the input names of a model.
 *
 * @param model A cviruntime model handle.
 * @param input_names. On return, it contains the input names of the model.
 * @param input_names_ptr. On return, input_names_ptr[i] contains
 *                         input_names[i].c_str()
 */
void GetInputNames(CVI_MODEL_HANDLE model, std::vector<std::string> *input_names,
                   std::vector<const char *> *input_names_ptr);

/**
 * Get the output names of a model.
 *
 * @param model A cviruntime model handle.
 * @param output_names. On return, it contains the output names of the model.
 * @param output_names_ptr. On return, output_names_ptr[i] contains
 *                         output_names[i].c_str()
 */
void GetOutputNames(CVI_MODEL_HANDLE model, std::vector<std::string> *output_names,
                    std::vector<const char *> *output_names_ptr);

/**
 * Get the output frame of Encoder
 *
 * @param allocator allocator of cviruntime
 * @param encoder_out encoder out tensor
 * @param t frame_index
 */
CVI_TENSOR GetEncoderOutFrame(CVI_MODEL_HANDLE model, CVI_TENSOR *encoder_out,
                              int32_t t);

/**
 * Print model metadata.
 *
 * @param os The output stream.
 * @param model The cviruntime model handle.
 */
void PrintModelMetadata(std::ostream &os, CVI_MODEL_HANDLE model);

// Return a deep copy of v
CVI_TENSOR Clone(CVI_MODEL_HANDLE model, const CVI_TENSOR *v);

// Return a shallow copy
CVI_TENSOR View(const CVI_TENSOR *v);

// Print a 1-D tensor to stderr
void Print1D(const CVI_TENSOR *v);

// Print a 2-D tensor to stderr
template <typename T = float>
void Print2D(const CVI_TENSOR *v);

// Print a 3-D tensor to stderr
void Print3D(const CVI_TENSOR *v);

// Print a 4-D tensor to stderr
void Print4D(const CVI_TENSOR *v);

template <typename T = float>
void Fill(CVI_TENSOR *tensor, T value) {
    auto n = CVI_NN_TensorCount(tensor);
    auto p = CVI_NN_TensorPtr(tensor);
    std::fill(p, p + n * sizeof(T), value);
}

std::vector<CVI_TENSOR> ReadFile(const std::string &filename);

// TODO(fangjun): Document it
CVI_TENSOR Repeat(CVI_MODEL_HANDLE model, const CVI_TENSOR *cur_encoder_out,
                  const std::vector<int32_t> &hyps_num_split);

}  // namespace sherpa_onnx

#endif  // SHERPA_CVIRTL_UTILS_H_