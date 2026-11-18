// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 nntrainer contributors
 *
 * @file   unittest_layers_fully_connected_lora.cpp
 * @date   18 November 2026
 * @brief  FullyConnectedLayer with LoRA: forward and the two backward
 *         products, against hand-computed math and finite differences.
 * @bug    No known bugs except for NYI items
 */

#include <layer_context.h>
#include <var_grad.h>
#include <weight.h>

#include <fc_layer.h>
#include <quantizer.h>

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <vector>

namespace {

// kIn and kUnit are multiples of 32 so the same fixture can hold a Q4_0 base
// weight (Q4_0 tensors need both dimensions in whole 32-element blocks).
constexpr unsigned int kBatch = 2, kHeight = 3, kIn = 32, kUnit = 32, kRank = 4;
constexpr float kAlpha = 8.0f; // scaling = alpha / rank = 2

/**
 * @brief A bias-free FC layer with LoRA on top of a frozen base weight (FP32,
 *        or Q4_0 when `quant`), wired to a RunLayerContext the way the graph
 *        does it.
 */
struct LoraFcFixture {
  nntrainer::FullyConnectedLayer layer;
  nntrainer::InitLayerContext init_context;
  std::vector<nntrainer::Weight> weights;
  std::vector<nntrainer::Var_Grad> inputs, outputs, tensors;
  /** the base weight as the layer multiplies by it (dequantized if Q4_0) */
  std::vector<float> base;
  // rc's initializer calls build(), which fills the members above: keep it
  // declared last.
  nntrainer::RunLayerContext rc;

  explicit LoraFcFixture(bool quant = false) :
    init_context({nntrainer::TensorDim({kBatch, 1, kHeight, kIn})}, {true},
                 false, "fc_lora", "", 0.0f,
                 {"NCHW", quant ? "Q4_0" : "FP32", "FP32"}),
    rc(build(quant)) {}

  static float pattern(float seed, unsigned i, float amp) {
    return std::sin(seed * i + 0.3f) * amp;
  }

  nntrainer::RunLayerContext build(bool quant) {
    layer.setProperty(
      {"unit=" + std::to_string(kUnit), "lora_rank=" + std::to_string(kRank),
       "lora_alpha=" + std::to_string((int)kAlpha), "disable_bias=true"});
    layer.finalize(init_context);

    auto addWeight = [&](unsigned r, unsigned c, const char *name, float seed,
                         float amp, bool trainable) {
      nntrainer::Tensor v(nntrainer::TensorDim({1, 1, r, c}), true);
      for (unsigned i = 0; i < r * c; ++i)
        v.getData<float>()[i] = pattern(seed, i, amp);
      nntrainer::Tensor g;
      if (trainable) {
        g = nntrainer::Tensor(nntrainer::TensorDim({1, 1, r, c}), true);
        g.setZero();
      }
      weights.emplace_back(v, g, nntrainer::Tensor(), name);
    };
    {
      nntrainer::Tensor w(
        1, 1, kIn, kUnit,
        {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32});
      for (unsigned i = 0; i < kIn * kUnit; ++i)
        w.getData<float>()[i] = pattern(0.41f, i, 0.5f);
      if (quant) {
        auto q =
          nntrainer::Quantization::createQuantizer(nntrainer::QScheme::Q4_0);
        nntrainer::Tensor wq = q->quantize(w, nntrainer::Tdatatype::Q4_0);
        nntrainer::Tensor wd = q->dequantize(wq, nntrainer::Tdatatype::FP32);
        base.assign(wd.getData<float>(), wd.getData<float>() + kIn * kUnit);
        weights.emplace_back(wq, nntrainer::Tensor(), nntrainer::Tensor(),
                             "weight");
      } else {
        base.assign(w.getData<float>(), w.getData<float>() + kIn * kUnit);
        weights.emplace_back(w, nntrainer::Tensor(), nntrainer::Tensor(),
                             "weight");
      }
    }
    addWeight(kIn, kRank, "loraA", 0.77f, 0.6f, true);
    addWeight(kRank, kUnit, "loraB", 0.53f, 0.4f, true);

    tensors.emplace_back(nntrainer::TensorDim({kBatch, 1, kHeight, kRank}),
                         nntrainer::Initializer::NONE, true, true,
                         "hidden_tmp_lora");
    tensors.emplace_back(nntrainer::TensorDim({kBatch, 1, kHeight, kUnit}),
                         nntrainer::Initializer::NONE, true, true,
                         "hidden_lora");

    inputs.emplace_back(init_context.getInputDimensions()[0],
                        nntrainer::Initializer::NONE, true, true, "input");
    for (unsigned i = 0; i < kBatch * kHeight * kIn; ++i)
      inputs[0].getVariableRef().getData<float>()[i] = pattern(0.71f, i, 0.8f);
    inputs[0].getGradientRef().setZero();

    outputs.emplace_back(init_context.getOutSpecs()[0].variable_spec.dim,
                         nntrainer::Initializer::NONE, true, true, "output");

    auto view = [](auto &v) {
      std::vector<std::remove_reference_t<decltype(v[0])> *> out;
      for (auto &e : v)
        out.push_back(&e);
      return out;
    };
    return nntrainer::RunLayerContext(
      "fc_lora", true, 0.0f, false, 1.0f, nullptr, false, view(weights),
      view(inputs), view(outputs), view(tensors));
  }

  float *w(unsigned i) { return weights[i].getVariableRef().getData<float>(); }
  float *x() { return inputs[0].getVariableRef().getData<float>(); }
  float *y() { return outputs[0].getVariableRef().getData<float>(); }
  float *dy() { return outputs[0].getGradientRef().getData<float>(); }
  float *dx() { return inputs[0].getGradientRef().getData<float>(); }
  float *dA() { return weights[1].getGradientRef().getData<float>(); }
  float *dB() { return weights[2].getGradientRef().getData<float>(); }

  void setDy() {
    for (unsigned i = 0; i < kBatch * kHeight * kUnit; ++i)
      dy()[i] = pattern(0.29f, i, 1.0f);
  }
};

constexpr float kScale = kAlpha / kRank;

/** y = x W + scale * (x A) B, rows flattened */
std::vector<double> referenceForward(LoraFcFixture &fx) {
  const unsigned rows = kBatch * kHeight;
  std::vector<double> y(rows * kUnit, 0.0);
  for (unsigned r = 0; r < rows; ++r)
    for (unsigned n = 0; n < kUnit; ++n) {
      double acc = 0;
      for (unsigned k = 0; k < kIn; ++k)
        acc += fx.x()[r * kIn + k] * fx.base[k * kUnit + n];
      for (unsigned j = 0; j < kRank; ++j) {
        double xa = 0;
        for (unsigned k = 0; k < kIn; ++k)
          xa += fx.x()[r * kIn + k] * fx.w(1)[k * kRank + j];
        acc += kScale * xa * fx.w(2)[j * kUnit + n];
      }
      y[r * kUnit + n] = acc;
    }
  return y;
}

} // namespace

class FullyConnectedLora : public ::testing::TestWithParam<bool> {};

TEST_P(FullyConnectedLora, ForwardAddsTheScaledAdapter) {
  LoraFcFixture fx(GetParam());
  fx.layer.forwarding(fx.rc, true);
  auto ref = referenceForward(fx);
  for (unsigned i = 0; i < ref.size(); ++i)
    EXPECT_NEAR(fx.y()[i], ref[i], 2e-4) << "i=" << i;
}

TEST_P(FullyConnectedLora, CalcDerivativeIsDyTimesWeightPlusAdapterTransposed) {
  LoraFcFixture fx(GetParam());
  fx.layer.forwarding(fx.rc, true);
  fx.setDy();
  fx.layer.calcDerivative(fx.rc);

  // dx = dy (W + scale A B)^T
  const unsigned rows = kBatch * kHeight;
  for (unsigned r = 0; r < rows; ++r)
    for (unsigned k = 0; k < kIn; ++k) {
      double acc = 0;
      for (unsigned n = 0; n < kUnit; ++n) {
        double wn = fx.base[k * kUnit + n];
        for (unsigned j = 0; j < kRank; ++j)
          wn += kScale * fx.w(1)[k * kRank + j] * fx.w(2)[j * kUnit + n];
        acc += fx.dy()[r * kUnit + n] * wn;
      }
      EXPECT_NEAR(fx.dx()[r * kIn + k], acc, 1e-4) << "r=" << r << " k=" << k;
    }
}

/**
 * @brief dA and dB match central differences of L = sum(y * dy)
 */
TEST_P(FullyConnectedLora, CalcGradientMatchesFiniteDifference) {
  LoraFcFixture fx(GetParam());
  fx.layer.forwarding(fx.rc, true);
  fx.setDy();
  fx.layer.calcGradient(fx.rc);

  auto loss = [&]() {
    fx.layer.forwarding(fx.rc, true);
    double l = 0;
    for (unsigned i = 0; i < kBatch * kHeight * kUnit; ++i)
      l += (double)fx.y()[i] * fx.dy()[i];
    return l;
  };
  const float eps = 1e-3f;
  auto check = [&](float *param, const float *analytic, unsigned n) {
    for (unsigned i = 0; i < n; ++i) {
      const float orig = param[i];
      param[i] = orig + eps;
      const double lp = loss();
      param[i] = orig - eps;
      const double lm = loss();
      param[i] = orig;
      EXPECT_NEAR(analytic[i], (lp - lm) / (2.0 * eps), 2e-3) << "i=" << i;
    }
  };
  // snapshot: the loss() forwards leave the gradients untouched
  std::vector<float> dA(fx.dA(), fx.dA() + kIn * kRank);
  std::vector<float> dB(fx.dB(), fx.dB() + kRank * kUnit);
  check(fx.w(1), dA.data(), kIn * kRank);
  check(fx.w(2), dB.data(), kRank * kUnit);
}

TEST_P(FullyConnectedLora, BaseWeightIsFrozen) {
  LoraFcFixture fx(GetParam());
  EXPECT_FALSE(fx.weights[0].hasGradient());
  EXPECT_TRUE(fx.weights[1].hasGradient());
  EXPECT_TRUE(fx.weights[2].hasGradient());
}

INSTANTIATE_TEST_SUITE_P(BaseWeight, FullyConnectedLora, ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool> &info) {
                           return info.param ? "Q4_0" : "FP32";
                         });
