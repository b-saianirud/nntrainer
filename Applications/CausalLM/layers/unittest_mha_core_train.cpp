// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_mha_core_train.cpp
 * @date   18 November 2026
 * @brief  mha_core training path: calcDerivative() against finite
 *         differences, and the training forward against the inference
 *         prefill it must reproduce.
 * @bug    No known bugs except for NYI items
 */

#include <layer_context.h>
#include <var_grad.h>
#include <weight.h>

#include <mha_core.h>

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

namespace {

std::vector<nntrainer::Weight *>
makeWeightView(std::vector<nntrainer::Weight> &weights) {
  std::vector<nntrainer::Weight *> view;
  view.reserve(weights.size());
  for (auto &weight : weights)
    view.push_back(&weight);
  return view;
}

std::vector<nntrainer::Var_Grad *>
makeVarGradView(std::vector<nntrainer::Var_Grad> &vars) {
  std::vector<nntrainer::Var_Grad *> view;
  view.reserve(vars.size());
  for (auto &var : vars)
    view.push_back(&var);
  return view;
}

nntrainer::RunLayerContext
makeRunContext(const std::string &name, std::vector<nntrainer::Weight> &weights,
               std::vector<nntrainer::Var_Grad> &inputs,
               std::vector<nntrainer::Var_Grad> &outputs,
               std::vector<nntrainer::Var_Grad> &tensors) {
  return nntrainer::RunLayerContext(
    name, true, 0.0f, false, 1.0f, nullptr, false, makeWeightView(weights),
    makeVarGradView(inputs), makeVarGradView(outputs),
    makeVarGradView(tensors));
}

/**
 * @brief Central-difference numerical gradient check: perturbs each element
 *        of `x`, re-runs `forward`, and compares (dL/dx)_i against
 *        `analytic[i]`, where L(x) := dot(forward(x), dy) so that the exact
 *        gradient of L w.r.t. x is what a correct backward pass should
 *        produce for the fixed incoming derivative `dy`.
 */
void checkGradient(float *x, const float *analytic, const float *dy, size_t n,
                   size_t out_n, const std::function<const float *()> &forward,
                   float eps = 1e-3f, float tol = 5e-2f) {
  auto loss = [&](const float *out) {
    double l = 0.0;
    for (size_t j = 0; j < out_n; ++j)
      l += static_cast<double>(out[j]) * static_cast<double>(dy[j]);
    return l;
  };

  for (size_t i = 0; i < n; ++i) {
    float orig = x[i];

    x[i] = orig + eps;
    double lp = loss(forward());

    x[i] = orig - eps;
    double lm = loss(forward());

    x[i] = orig;

    double numeric = (lp - lm) / (2.0 * eps);
    EXPECT_NEAR(analytic[i], numeric, tol)
      << "gradient mismatch at index " << i << " (analytic=" << analytic[i]
      << ", numeric=" << numeric << ")";
  }
}

} // namespace

TEST(MhaCoreTrain, MhaCoreCalcDerivativeMatchesFiniteDifference) {
  // 4 Q heads, 2 KV heads (GQA group size 2), head_dim=4, seq_len=3.
  const unsigned int seq_len = 3, num_heads_q = 4, num_heads_kv = 2,
                     head_dim = 4;
  const unsigned int q_width = num_heads_q * head_dim;   // 16
  const unsigned int kv_width = num_heads_kv * head_dim; // 8

  causallm::MHACoreLayer layer;
  ASSERT_NO_THROW(layer.setProperty(
    {"num_heads=" + std::to_string(num_heads_q),
     "num_heads_KV=" + std::to_string(num_heads_kv), "max_timestep=8"}));

  nntrainer::InitLayerContext init_context(
    {nntrainer::TensorDim({1, 1, seq_len, q_width}),
     nntrainer::TensorDim({1, 1, seq_len, kv_width}),
     nntrainer::TensorDim({1, 1, seq_len, kv_width})},
    {true}, false, "mha_core_gradcheck", "", 0.0f, {"NCHW", "FP32", "FP32"});
  ASSERT_NO_THROW(layer.finalize(init_context));

  std::vector<nntrainer::Weight> weights; // no weights (use_sink=false)
  std::vector<nntrainer::Var_Grad> inputs, outputs, tensors;

  for (const auto &dim : init_context.getInputDimensions())
    inputs.emplace_back(dim, nntrainer::Initializer::NONE, true, true, "in");
  outputs.emplace_back(init_context.getOutSpecs()[0].variable_spec.dim,
                       nntrainer::Initializer::NONE, true, true, "output");
  // Build the 5 requested tensors (cache_key, cache_value, train_q_roped,
  // train_k_roped, train_attn_wt) directly from the specs finalize()
  // produced, so their exact dims/dtypes always match what the layer
  // actually requested.
  for (const auto &spec : init_context.getTensorsSpec())
    tensors.emplace_back(spec, true);

  auto run_context =
    makeRunContext("mha_core_gradcheck", weights, inputs, outputs, tensors);

  float *q = inputs[0].getVariableRef().getData<float>();
  float *k = inputs[1].getVariableRef().getData<float>();
  float *v = inputs[2].getVariableRef().getData<float>();
  for (unsigned int i = 0; i < seq_len * q_width; ++i)
    q[i] = std::sin(0.7f * i + 1.0f) * 0.5f;
  for (unsigned int i = 0; i < seq_len * kv_width; ++i)
    k[i] = std::cos(0.5f * i + 0.3f) * 0.5f;
  for (unsigned int i = 0; i < seq_len * kv_width; ++i)
    v[i] = std::sin(0.3f * i + 0.9f) * 0.4f;

  std::vector<float> dy(seq_len * q_width);
  for (unsigned int i = 0; i < dy.size(); ++i)
    dy[i] = std::cos(0.9f * i + 0.2f) * 0.3f;
  std::copy(dy.begin(), dy.end(),
            run_context.getOutputGradUnsafe(0).getData<float>());

  layer.forwarding(run_context, true);
  layer.calcDerivative(run_context);

  std::vector<float> analytic_q(seq_len * q_width),
    analytic_k(seq_len * kv_width), analytic_v(seq_len * kv_width);
  std::copy(run_context.getOutgoingDerivative(0).getData<float>(),
            run_context.getOutgoingDerivative(0).getData<float>() +
              seq_len * q_width,
            analytic_q.begin());
  std::copy(run_context.getOutgoingDerivative(1).getData<float>(),
            run_context.getOutgoingDerivative(1).getData<float>() +
              seq_len * kv_width,
            analytic_k.begin());
  std::copy(run_context.getOutgoingDerivative(2).getData<float>(),
            run_context.getOutgoingDerivative(2).getData<float>() +
              seq_len * kv_width,
            analytic_v.begin());

  auto forward = [&]() -> const float * {
    layer.forwarding(run_context, true);
    return run_context.getOutput(0).getData<float>();
  };

  checkGradient(q, analytic_q.data(), dy.data(), seq_len * q_width,
                seq_len * q_width, forward, 1e-3f, 5e-3f);
  checkGradient(k, analytic_k.data(), dy.data(), seq_len * kv_width,
                seq_len * q_width, forward, 1e-3f, 5e-3f);
  checkGradient(v, analytic_v.data(), dy.data(), seq_len * kv_width,
                seq_len * q_width, forward, 1e-3f, 5e-3f);
}

/**
 * @brief mha_core's training forward (full-sequence, internal-cache) must
 *        agree with the trusted inference prefill path
 *        (incremental_forwarding over [0, seq_len)) for identical Q/K/V,
 *        since both compute the same dense causal GQA attention.
 */
TEST(MhaCoreTrain, MhaCoreTrainForwardMatchesInferencePrefill) {
  const unsigned int seq_len = 5, num_heads_q = 4, num_heads_kv = 2,
                     head_dim = 4;
  const unsigned int q_width = num_heads_q * head_dim;
  const unsigned int kv_width = num_heads_kv * head_dim;

  std::vector<float> q(seq_len * q_width), k(seq_len * kv_width),
    v(seq_len * kv_width);
  for (unsigned int i = 0; i < q.size(); ++i)
    q[i] = std::sin(0.7f * i + 1.0f) * 0.5f;
  for (unsigned int i = 0; i < k.size(); ++i)
    k[i] = std::cos(0.5f * i + 0.3f) * 0.5f;
  for (unsigned int i = 0; i < v.size(); ++i)
    v[i] = std::sin(0.3f * i + 0.9f) * 0.4f;

  auto run = [&](ml::train::ExecutionMode mode) -> std::vector<float> {
    causallm::MHACoreLayer layer;
    layer.setProperty({"num_heads=" + std::to_string(num_heads_q),
                       "num_heads_KV=" + std::to_string(num_heads_kv),
                       "max_timestep=" + std::to_string(seq_len)});
    nntrainer::InitLayerContext init_context(
      {nntrainer::TensorDim({1, 1, seq_len, q_width}),
       nntrainer::TensorDim({1, 1, seq_len, kv_width}),
       nntrainer::TensorDim({1, 1, seq_len, kv_width})},
      {true}, false, "mha_cmp", "", 0.0f, {"NCHW", "FP32", "FP32"}, 1.0f, mode);
    layer.finalize(init_context);

    std::vector<nntrainer::Weight> weights;
    std::vector<nntrainer::Var_Grad> inputs, outputs, tensors;
    for (const auto &d : init_context.getInputDimensions())
      inputs.emplace_back(d, nntrainer::Initializer::NONE, true, true, "in");
    outputs.emplace_back(init_context.getOutSpecs()[0].variable_spec.dim,
                         nntrainer::Initializer::NONE, true, true, "out");
    for (const auto &spec : init_context.getTensorsSpec())
      tensors.emplace_back(spec, true);

    std::copy(q.begin(), q.end(), inputs[0].getVariableRef().getData<float>());
    std::copy(k.begin(), k.end(), inputs[1].getVariableRef().getData<float>());
    std::copy(v.begin(), v.end(), inputs[2].getVariableRef().getData<float>());

    auto rc = makeRunContext("mha_cmp", weights, inputs, outputs, tensors);
    rc.getOutput(0).setZero();
    if (mode == ml::train::ExecutionMode::TRAIN)
      layer.forwarding(rc, true);
    else
      layer.incremental_forwarding(rc, 0, seq_len, false);

    const float *o = rc.getOutput(0).getData<float>();
    return std::vector<float>(o, o + seq_len * q_width);
  };

  auto inference_out = run(ml::train::ExecutionMode::INFERENCE);
  auto training_out = run(ml::train::ExecutionMode::TRAIN);

  ASSERT_EQ(inference_out.size(), training_out.size());
  // Tolerance is set by the inference path storing its KV cache in 16-bit
  // (FP16/UINT16) while the training path keeps Q/K/V in FP32, so the two
  // agree only to FP16 precision.
  for (size_t i = 0; i < inference_out.size(); ++i)
    EXPECT_NEAR(inference_out[i], training_out[i], 2e-3)
      << "attention output mismatch at " << i << " (row " << i / q_width
      << ", head " << (i % q_width) / head_dim << ")";
}

/**
 * @brief Regression test: nntrainer aliases each outgoing derivative onto its
 *        input's own buffer, so a calcDerivative that writes one output
 *        before reading every input it needs will silently consume its own
 *        output. SwiGLU is the case that matters (d_up aliases up, which is
 *        still needed to form d_gate).
 *
 * @note The other gradient checks in this file build each Var_Grad with
 *       independent variable/gradient storage, which does NOT reproduce the
 *       aliasing the real graph uses — hence this dedicated test.
 */
