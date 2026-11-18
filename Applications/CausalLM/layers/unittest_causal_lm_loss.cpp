// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_causal_lm_loss.cpp
 * @date   18 November 2026
 * @brief  causal_lm_loss: per-position next-token NLL, the mean over labelled
 *         positions, its gradient, and the validation statistics.
 * @bug    No known bugs except for NYI items
 */

#include <layer_context.h>
#include <var_grad.h>

#include <causal_lm_loss.h>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

constexpr unsigned int kBatch = 2, kSeq = 3, kVocab = 5;

/** target id per position; -1 carries no loss */
const float kLabels[kBatch * kSeq] = {1, 4, -1, 0, -1, 2};

struct LossFixture {
  causallm::CausalLMLossLayer layer;
  std::vector<nntrainer::Var_Grad> inputs, outputs;
  std::vector<nntrainer::Weight> weights;
  std::vector<nntrainer::Var_Grad> tensors;
  nntrainer::RunLayerContext rc;

  LossFixture() : rc(build()) {}

  nntrainer::RunLayerContext build() {
    nntrainer::InitLayerContext init(
      {nntrainer::TensorDim({kBatch, 1, kSeq, kVocab})}, {true}, false, "loss",
      "", 0.0f, {"NCHW", "FP32", "FP32"});
    layer.finalize(init);

    inputs.emplace_back(init.getInputDimensions()[0],
                        nntrainer::Initializer::NONE, true, true, "logits");
    for (unsigned i = 0; i < kBatch * kSeq * kVocab; ++i)
      inputs[0].getVariableRef().getData<float>()[i] =
        std::sin(0.9f * i + 0.2f) * 2.0f;
    inputs[0].getGradientRef().setZero();

    outputs.emplace_back(init.getOutSpecs()[0].variable_spec.dim,
                         nntrainer::Initializer::NONE, true, true, "nll");
    // the label travels in the output's gradient tensor
    for (unsigned i = 0; i < kBatch * kSeq; ++i)
      outputs[0].getGradientRef().getData<float>()[i] = kLabels[i];

    auto view = [](auto &v) {
      std::vector<std::remove_reference_t<decltype(v[0])> *> out;
      for (auto &e : v)
        out.push_back(&e);
      return out;
    };
    return nntrainer::RunLayerContext("loss", true, 0.0f, false, 1.0f, nullptr,
                                      false, view(weights), view(inputs),
                                      view(outputs), view(tensors));
  }

  float *logits() { return inputs[0].getVariableRef().getData<float>(); }
  float *nll() { return outputs[0].getVariableRef().getData<float>(); }
  float *dx() { return inputs[0].getGradientRef().getData<float>(); }
};

double rowNll(const float *row, int target) {
  double mx = row[0];
  for (unsigned v = 1; v < kVocab; ++v)
    mx = std::max<double>(mx, row[v]);
  double sum = 0;
  for (unsigned v = 0; v < kVocab; ++v)
    sum += std::exp(row[v] - mx);
  return mx + std::log(sum) - row[target];
}

} // namespace

TEST(CausalLmLoss, ForwardIsTheMeanNllOverLabelledPositions) {
  LossFixture fx;
  fx.layer.forwarding(fx.rc, true);

  double total = 0;
  unsigned count = 0;
  for (unsigned i = 0; i < kBatch * kSeq; ++i) {
    if (kLabels[i] < 0) {
      EXPECT_EQ(fx.nll()[i], 0.0f) << "unlabelled position " << i;
      continue;
    }
    const double want = rowNll(fx.logits() + i * kVocab, (int)kLabels[i]);
    EXPECT_NEAR(fx.nll()[i], want, 1e-5) << "position " << i;
    total += want;
    ++count;
  }
  ASSERT_EQ(count, 4u);
  EXPECT_NEAR(fx.rc.getLoss(), total / count, 1e-5);
}

/** d(mean NLL)/d logits by central differences, zero on unlabelled rows */
TEST(CausalLmLoss, DerivativeMatchesFiniteDifference) {
  LossFixture fx;
  fx.layer.forwarding(fx.rc, true);
  fx.layer.calcDerivative(fx.rc);
  std::vector<float> analytic(fx.dx(), fx.dx() + kBatch * kSeq * kVocab);

  auto meanNll = [&]() {
    fx.layer.forwarding(fx.rc, true);
    return (double)fx.rc.getLoss();
  };
  const float eps = 1e-3f;
  for (unsigned i = 0; i < analytic.size(); ++i) {
    const float orig = fx.logits()[i];
    fx.logits()[i] = orig + eps;
    const double lp = meanNll();
    fx.logits()[i] = orig - eps;
    const double lm = meanNll();
    fx.logits()[i] = orig;
    EXPECT_NEAR(analytic[i], (lp - lm) / (2 * eps), 2e-3) << "i=" << i;
  }
  for (unsigned v = 0; v < kVocab; ++v)
    EXPECT_EQ(analytic[2 * kVocab + v], 0.0f) << "unlabelled row 2";
}

/** validation (training = false) accumulates token-weighted NLL and top-k */
TEST(CausalLmLoss, EvalStatsCountTokensAndTopK) {
  LossFixture fx;
  causallm::g_causal_lm_eval_stats.reset();
  fx.layer.forwarding(fx.rc, false);

  const auto &st = causallm::g_causal_lm_eval_stats;
  EXPECT_EQ(st.tokens, 4u);
  EXPECT_LE(st.top1, st.top5);
  EXPECT_LE(st.top5, st.tokens);

  double nll = 0;
  size_t top1 = 0;
  for (unsigned i = 0; i < kBatch * kSeq; ++i) {
    if (kLabels[i] < 0)
      continue;
    const float *row = fx.logits() + i * kVocab;
    nll += rowNll(row, (int)kLabels[i]);
    unsigned arg = 0;
    for (unsigned v = 1; v < kVocab; ++v)
      if (row[v] > row[arg])
        arg = v;
    top1 += (arg == (unsigned)kLabels[i]);
  }
  EXPECT_NEAR(st.nll, nll, 1e-4);
  EXPECT_EQ(st.top1, top1);
  // with 5 classes top-5 is always a hit
  EXPECT_EQ(st.top5, 4u);
}
