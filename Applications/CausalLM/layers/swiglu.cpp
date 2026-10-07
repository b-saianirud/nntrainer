// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   swiglu.cpp
 * @date   14 July 2023
 * @brief  Implementation of SwiGLU activation function
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>
#include <cstring>
#include <util_simd.h>

#include "swiglu.h"

namespace causallm {

static constexpr size_t OUT_IDX = 0;
static constexpr size_t INPUT_IDX_1 = 0;
static constexpr size_t INPUT_IDX_2 = 1;

namespace ActivationOp {
/**
 * @brief activation function swiglu
 * @param x input
 * @retval swiglu(x)
 */
float swiglu(float x) { return x / (1 + nntrainer::exp_util(-x)); }
} // namespace ActivationOp

void SwiGLULayer::finalize(nntrainer::InitLayerContext &context) {
  context.setOutputDimensions({context.getInputDimensions()[0]});

  if (!std::get<nntrainer::props::SkipPrefill>(swiglu_props).empty())
    skip_prefill = std::get<nntrainer::props::SkipPrefill>(swiglu_props).get();

  cache_inputs =
    (context.getExecutionMode() == ml::train::ExecutionMode::TRAIN);
  if (cache_inputs) {
    nntrainer::TensorDim cache_dim(context.getInputDimensions()[0]);
    cache_dim.setTensorType(
      {context.getFormat(), nntrainer::TensorDim::DataType::FP32});
    cache_idx[0] = context.requestTensor(
      cache_dim, "gate_cache", nntrainer::Initializer::NONE, false,
      nntrainer::TensorLifespan::ITERATION_LIFESPAN);
    cache_idx[1] = context.requestTensor(
      cache_dim, "up_cache", nntrainer::Initializer::NONE, false,
      nntrainer::TensorLifespan::ITERATION_LIFESPAN);
  }
}

void SwiGLULayer::forwarding(nntrainer::RunLayerContext &context,
                             bool training) {
  nntrainer::Tensor &in1 = context.getInput(INPUT_IDX_1);
  computeSwiGLU(context, 0, in1.getDim().height());
}

void SwiGLULayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) {
  bool is_prefill = !from || (to - from) > 1;
  if (skip_prefill && is_prefill)
    return;

  computeSwiGLU(context, from, to);
}

void SwiGLULayer::computeSwiGLU(nntrainer::RunLayerContext &context,
                                unsigned int from, unsigned int to) {
  nntrainer::Tensor &in1 = context.getInput(INPUT_IDX_1);
  nntrainer::Tensor &in2 = context.getInput(INPUT_IDX_2);
  nntrainer::Tensor &out = context.getOutput(OUT_IDX);

  int iter = to - from;

  if (in1.getDataType() == ml::train::TensorDim::DataType::FP32) {
    for (unsigned int b = 0; b < in1.batch(); b++) {
      for (unsigned int c = 0; c < in1.channel(); c++) {
        for (unsigned int h = 0; h < iter; h++) {
          nntrainer::swiglu(in1.width(),
                            out.getData<float>() + out.getIndex(b, c, h, 0),
                            in1.getData<float>() + in1.getIndex(b, c, h, 0),
                            in2.getData<float>() + in2.getIndex(b, c, h, 0));
        }
        if (cache_inputs) {
          // keep a private copy of gate/up for calcDerivative(); see
          // cache_inputs in swiglu.h.
          nntrainer::Tensor &gate_cache = context.getTensor(cache_idx[0]);
          nntrainer::Tensor &up_cache = context.getTensor(cache_idx[1]);
          for (unsigned int h = 0; h < iter; h++) {
            std::memcpy(gate_cache.getData<float>() +
                          gate_cache.getIndex(b, c, h, 0),
                        in1.getData<float>() + in1.getIndex(b, c, h, 0),
                        sizeof(float) * in1.width());
            std::memcpy(up_cache.getData<float>() +
                          up_cache.getIndex(b, c, h, 0),
                        in2.getData<float>() + in2.getIndex(b, c, h, 0),
                        sizeof(float) * in2.width());
          }
        }
      }
    }
  } else if (in1.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    for (unsigned int b = 0; b < in1.batch(); b++) {
      for (unsigned int c = 0; c < in1.channel(); c++) {
        for (unsigned int h = 0; h < iter; h++) {
          nntrainer::swiglu(in1.width(),
                            out.getData<_FP16>() + out.getIndex(b, c, h, 0),
                            in1.getData<_FP16>() + in1.getIndex(b, c, h, 0),
                            in2.getData<_FP16>() + in2.getIndex(b, c, h, 0));
        }
      }
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

void SwiGLULayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  ml::train::TensorDim input_dim1 = context.getInput(INPUT_IDX_1).getDim();
  ml::train::TensorDim input_dim2 = context.getInput(INPUT_IDX_2).getDim();
  ml::train::TensorDim output_dim = context.getOutput(OUT_IDX).getDim();

  input_dim1.height(input_dimensions[0].height());
  input_dim2.height(input_dimensions[0].height());
  output_dim.height(input_dimensions[0].height());

  context.updateInput(INPUT_IDX_1, input_dim1);
  context.updateInput(INPUT_IDX_2, input_dim2);
  context.updateOutput(OUT_IDX, output_dim);

  if (cache_inputs) {
    nntrainer::TensorDim cache_dim = input_dim1;
    cache_dim.setTensorType(
      {input_dim1.getFormat(), nntrainer::TensorDim::DataType::FP32});
    context.updateTensor(cache_idx[0], cache_dim);
    context.updateTensor(cache_idx[1], cache_dim);
  }
}

void SwiGLULayer::setBatch(nntrainer::RunLayerContext &context,
                           unsigned int batch) {
  if (cache_inputs) {
    context.updateTensor(cache_idx[0], batch);
    context.updateTensor(cache_idx[1], batch);
  }
}

/**
 * @brief calcDerivative for SwiGLU.
 * @details out = silu(gate) * up, where gate = input[0], up = input[1] and
 *          silu(g) = g * sigmoid(g).
 *          d(out)/d(gate) = up * silu'(gate),
 *            silu'(g) = sigmoid(g) * (1 + g * (1 - sigmoid(g)))
 *          d(out)/d(up) = silu(gate)
 *          SwiGLU has no weights of its own, so there is no
 *          calcGradient to implement.
 *
 * @note nntrainer aliases each outgoing derivative onto its input's own
 *       buffer (the activation's storage is recycled to hold its gradient),
 *       so `d_gate` IS `gate` and `d_up` IS `up`. Every input element must
 *       therefore be read into a local before *either* output element at
 *       that index is written — otherwise writing d_up[i] destroys up[i]
 *       before it is used to form d_gate[i].
 */
void SwiGLULayer::calcDerivative(nntrainer::RunLayerContext &context) {
  // gate/up come from the private caches filled in forwarding(): the input
  // buffers of this weightless layer are not guaranteed to be alive here.
  nntrainer::Tensor &gate = context.getTensor(cache_idx[0]);
  nntrainer::Tensor &up = context.getTensor(cache_idx[1]);
  const nntrainer::Tensor &dy = context.getIncomingDerivative(OUT_IDX);
  nntrainer::Tensor &d_gate = context.getOutgoingDerivative(INPUT_IDX_1);
  nntrainer::Tensor &d_up = context.getOutgoingDerivative(INPUT_IDX_2);

  NNTR_THROW_IF(gate.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "[swiglu] calcDerivative only supports FP32 for now";

  const size_t len = gate.size();
  const float *g = gate.getData<float>();
  const float *u = up.getData<float>();
  const float *dy_ = dy.getData<float>();
  float *dg = d_gate.getData<float>();
  float *du = d_up.getData<float>();

  for (size_t i = 0; i < len; ++i) {
    // Snapshot all inputs at this index first; see the aliasing note above.
    const float gi = g[i];
    const float ui = u[i];
    const float dyi = dy_[i];

    const float sig = 1.0f / (1.0f + std::exp(-gi));
    const float silu = gi * sig;
    const float dsilu = sig * (1.0f + gi * (1.0f - sig));

    du[i] = dyi * silu;
    dg[i] = dyi * ui * dsilu;
  }
}

#ifdef PLUGGABLE

nntrainer::Layer *create_swiglu_layer() {
  auto layer = new SwiGLULayer();
  return layer;
}

void destroy_swiglu_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_swiglu_layer,
                                                   destroy_swiglu_layer};
}

#endif

} // namespace causallm
