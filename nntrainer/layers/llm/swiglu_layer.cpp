// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Niket Agarwal <niket.a@samsung.com>
 *
 * @file   swiglu_layer.cpp
 * @date   6 June 2024
 * @see    https://github.com/nntrainer/nntrainer
 * @author Niket Agarwal <niket.a@samsung.com>
 * @author Eunju Yang <ej.yang@samsung.com>
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Backend-neutral SwiGLU activation: out = silu(gate) * up.
 */

#include "swiglu_layer.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <nntrainer_error.h>
#include <node_exporter.h>
#include <tensor.h>
#include <thread_manager.h>

namespace nntrainer {

static constexpr size_t OUT_IDX = 0;
static constexpr size_t INPUT_IDX_1 = 0; /**< gate */
static constexpr size_t INPUT_IDX_2 = 1; /**< up */

void SwiGLULayer::finalize(InitLayerContext &context) {
  context.setOutputDimensions({context.getInputDimensions()[0]});

  if (!std::get<props::SkipPrefill>(swiglu_props).empty())
    skip_prefill = std::get<props::SkipPrefill>(swiglu_props).get();

  cache_inputs =
    (context.getExecutionMode() == ml::train::ExecutionMode::TRAIN);
  if (cache_inputs) {
    TensorDim cache_dim(context.getInputDimensions()[0]);
    cache_dim.setTensorType({context.getFormat(), TensorDim::DataType::FP32});
    cache_idx[0] =
      context.requestTensor(cache_dim, "gate_cache", Initializer::NONE, false,
                            TensorLifespan::ITERATION_LIFESPAN);
    cache_idx[1] =
      context.requestTensor(cache_dim, "up_cache", Initializer::NONE, false,
                            TensorLifespan::ITERATION_LIFESPAN);
  }
}

void SwiGLULayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, swiglu_props);
  if (!remain_props.empty()) {
    std::string msg = "[SwiGLULayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

void SwiGLULayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &in1 = context.getInput(INPUT_IDX_1);
  Tensor &in2 = context.getInput(INPUT_IDX_2);
  Tensor &out = context.getOutput(OUT_IDX);

  if (cache_inputs) {
    // Private copy of gate/up for calcDerivative(); taken before the op in case
    // the output aliases an input. shortcut: whole-tensor copy, FP32 only (the
    // training path), see calcDerivative().
    std::memcpy(context.getTensor(cache_idx[0]).getData<float>(),
                in1.getData<float>(), sizeof(float) * in1.size());
    std::memcpy(context.getTensor(cache_idx[1]).getData<float>(),
                in2.getData<float>(), sizeof(float) * in2.size());
  }

  // Rows are the (batch, channel, height) axes flattened; width() is the
  // per-row element count.
  in1.getOps()->swiglu(in1, in2, out,
                       in1.batch() * in1.channel() * in1.height(),
                       /*row_offset=*/0);
}

void SwiGLULayer::incremental_forwarding(RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) {
  // A multi-token step is a prefill step, so a KV-shared layer skips it here
  // exactly as it skips the from == 0 one: the branch produces nothing any
  // later layer reads.
  if (skip_prefill && (from == 0 || (to - from) > 1))
    return;

  Tensor &in1 = context.getInput(INPUT_IDX_1);
  Tensor &in2 = context.getInput(INPUT_IDX_2);
  Tensor &out = context.getOutput(OUT_IDX);

  // Every producer writes the live rows starting at the buffer base on every
  // backend, so the offset is always 0 and the count is the live-row window.
  // Rows are (batch, channel, height) flattened, so the window has to be
  // scaled by batch*channel or a batch>1 shape only computes its first
  // (to - from) rows.
  const unsigned int bc = in1.batch() * in1.channel();
  in1.getOps()->swiglu(in1, in2, out, (to - from) * bc, /*row_offset=*/0);
}

void SwiGLULayer::updateTensorsByInputDimensions(
  RunLayerContext &context, std::vector<TensorDim> input_dimensions) {
  TensorDim input_dim1 = context.getInput(INPUT_IDX_1).getDim();
  TensorDim input_dim2 = context.getInput(INPUT_IDX_2).getDim();
  TensorDim output_dim = context.getOutput(OUT_IDX).getDim();

  input_dim1.height(input_dimensions[0].height());
  input_dim2.height(input_dimensions[0].height());
  output_dim.height(input_dimensions[0].height());

  context.updateInput(INPUT_IDX_1, input_dim1);
  context.updateInput(INPUT_IDX_2, input_dim2);
  context.updateOutput(OUT_IDX, output_dim);

  if (cache_inputs) {
    TensorDim cache_dim = input_dim1;
    cache_dim.setTensorType(
      {input_dim1.getFormat(), TensorDim::DataType::FP32});
    context.updateTensor(cache_idx[0], cache_dim);
    context.updateTensor(cache_idx[1], cache_dim);
  }
}

void SwiGLULayer::setBatch(RunLayerContext &context, unsigned int batch) {
  if (cache_inputs) {
    context.updateTensor(cache_idx[0], batch);
    context.updateTensor(cache_idx[1], batch);
  }
}

/**
 * @brief calcDerivative for SwiGLU.
 * @details out = silu(gate) * up, gate = input[0], up = input[1],
 *          silu(g) = g * sigmoid(g).
 *          d(out)/d(gate) = up * silu'(gate),
 *            silu'(g) = sigmoid(g) * (1 + g * (1 - sigmoid(g)))
 *          d(out)/d(up) = silu(gate)
 *          SwiGLU has no weights, so there is no calcGradient.
 *
 * @note nntrainer aliases each outgoing derivative onto its input's own buffer,
 *       so d_gate IS gate and d_up IS up. Every input element must therefore be
 *       read into a local before either output element at that index is
 * written. gate/up come from the private caches filled in forwarding(), because
 * the input buffers of this weightless layer are not guaranteed alive here.
 */
void SwiGLULayer::calcDerivative(RunLayerContext &context) {
  NNTR_THROW_IF(!cache_inputs, std::runtime_error)
    << "[swiglu] calcDerivative needs the layer to be finalized in TRAIN mode";

  Tensor &gate = context.getTensor(cache_idx[0]);
  Tensor &up = context.getTensor(cache_idx[1]);
  const Tensor &dy = context.getIncomingDerivative(OUT_IDX);
  Tensor &d_gate = context.getOutgoingDerivative(INPUT_IDX_1);
  Tensor &d_up = context.getOutgoingDerivative(INPUT_IDX_2);

  NNTR_THROW_IF(gate.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "[swiglu] calcDerivative only supports FP32 for now";

  const size_t len = gate.size();
  const float *g = gate.getData<float>();
  const float *u = up.getData<float>();
  const float *dy_ = dy.getData<float>();
  float *dg = d_gate.getData<float>();
  float *du = d_up.getData<float>();

  // Element-wise and independent per index (gate/up are the private caches, so
  // the outgoing buffers cannot alias what is read), hence safe to split.
  constexpr size_t kChunk = 16384;
  const size_t chunks = (len + kChunk - 1) / kChunk;
  ThreadManager::Global().parallel_for(0, chunks, [&](size_t c) {
    const size_t lo = c * kChunk;
    const size_t hi = std::min(len, lo + kChunk);
    for (size_t i = lo; i < hi; ++i) {
      const float gi = g[i];
      const float ui = u[i];
      const float dyi = dy_[i];

      const float sig = 1.0f / (1.0f + std::exp(-gi));
      const float silu = gi * sig;
      const float dsilu = sig * (1.0f + gi * (1.0f - sig));

      du[i] = dyi * silu;
      dg[i] = dyi * ui * dsilu;
    }
  });
}

} // namespace nntrainer
