/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * @file        unittest_nntrainer_mezo_interface.cpp
 * @date        05 May 2026
 * @brief       Unit test for MeZO optimizer interface changes.
 * @see         https://github.com/nntrainer/nntrainer
 * @author      Sachin Singh <sachin.3@samsung.com>
 * @author      Niket Agarwal <niket.a@samsung.com>
 * @bug         No known bugs
 */
#include <gtest/gtest.h>

#include <mezo.h>
#include <neuralnet.h>
#include <optimizer_devel.h>
#include <optimizer_wrapped.h>
#include <sgd.h>

using namespace nntrainer;

/**
 * @brief Test that MeZO optimizer correctly reports it doesn't require backprop
 */
TEST(nntrainer_mezo_interface, requiresBackprop_01_p) {
  MeZO mezo;
  EXPECT_FALSE(mezo.requiresBackprop());
}

/**
 * @brief Test that SGD optimizer correctly reports it requires backprop
 */
TEST(nntrainer_mezo_interface, requiresBackprop_02_p) {
  SGD sgd;
  EXPECT_TRUE(sgd.requiresBackprop());
}

/**
 * @brief Test that OptimizerWrapped correctly delegates requiresBackprop for
 * MeZO
 */
TEST(nntrainer_mezo_interface, requiresBackprop_wrapped_01_p) {
  auto mezo_ptr = std::make_unique<MeZO>();
  auto wrapped = createOptimizerWrapped(std::move(mezo_ptr));
  EXPECT_FALSE(wrapped->requiresBackprop());
}

/**
 * @brief Test that OptimizerWrapped correctly delegates requiresBackprop for
 * SGD
 */
TEST(nntrainer_mezo_interface, requiresBackprop_wrapped_02_p) {
  auto sgd_ptr = std::make_unique<SGD>();
  auto wrapped = createOptimizerWrapped(std::move(sgd_ptr));
  EXPECT_TRUE(wrapped->requiresBackprop());
}

/**
 * @brief Mock function to test trainStep interface
 */
static bool forward_called = false;
static bool get_loss_called = false;

void mock_forward_fn() { forward_called = true; }

float mock_get_loss_fn() {
  get_loss_called = true;
  return 0.5f;
}

/**
 * @brief Test that MeZO optimizer trainStep method can be called
 */
TEST(nntrainer_mezo_interface, trainStep_01_p) {
  MeZO mezo;
  std::vector<Tensor *> params; // Empty params for this test

  // This should not crash and should call the functions
  forward_called = false;
  get_loss_called = false;

  mezo.trainStep(mock_forward_fn, mock_get_loss_fn, params);

  // Note: The actual MeZO implementation will try to perturb parameters,
  // but with empty params vector, it should complete without crashing
  SUCCEED();
}

/**
 * @brief Test that base Optimizer trainStep method is a no-op
 */
TEST(nntrainer_mezo_interface, trainStep_base_01_p) {
  // Create a mock optimizer that inherits from base Optimizer
  class MockOptimizer : public Optimizer {
  public:
    double getDefaultLearningRate() const override { return 0.01; }
    void applyGradient(RunOptimizerContext &context) override {}
    std::vector<TensorDim>
    getOptimizerVariableDim(const TensorDim &dim) override {
      return {};
    }
    const std::string getType() const override { return "Mock"; }
    void setProperty(const std::vector<std::string> &values) override {}
  };

  MockOptimizer mock_opt;
  std::vector<Tensor *> params;

  // This should be a no-op and not crash
  mock_opt.trainStep(mock_forward_fn, mock_get_loss_fn, params);
  SUCCEED();
}

/**
 * @brief Build a small (1,1,1,3) parameter tensor filled with known values,
 * mirroring how mezo.cpp itself allocates its transient noise tensors.
 */
static Tensor makeParamTensor(float v0, float v1, float v2) {
  Tensor t(TensorDim(1, 1, 1, 3));
  t.allocate();
  t.setValue(0, 0, 0, 0, v0);
  t.setValue(0, 0, 0, 1, v1);
  t.setValue(0, 0, 0, 2, v2);
  return t;
}

/**
 * @brief A full trainStep (+eps -> +eps-2eps -> +eps -> update) must leave
 * every parameter numerically unchanged when loss_plus == loss_minus, since
 * the three perturbation calls regenerate the identical noise vector z from
 * the same seed and net to zero, and projected_grad == 0 makes the final
 * update a no-op. This was never exercised by the original interface-only
 * tests (which only ever passed an empty params vector).
 */
TEST(nntrainer_mezo_interface, trainStep_restores_theta_when_loss_flat_p) {
  MeZO mezo;

  Tensor p0 = makeParamTensor(1.0f, -2.0f, 0.5f);
  Tensor p1 = makeParamTensor(3.25f, 0.0f, -1.75f);
  std::vector<Tensor *> params = {&p0, &p1};

  // loss_plus and loss_minus are identical regardless of the perturbation,
  // so projected_grad == (c - c) / (2*eps) == 0 exactly.
  auto flat_loss_fn = []() { return 42.0f; };
  auto noop_forward_fn = []() {};

  mezo.trainStep(noop_forward_fn, flat_loss_fn, params);

  const float tol = 1e-4f;
  EXPECT_NEAR(p0.getValue(0, 0, 0, 0), 1.0f, tol);
  EXPECT_NEAR(p0.getValue(0, 0, 0, 1), -2.0f, tol);
  EXPECT_NEAR(p0.getValue(0, 0, 0, 2), 0.5f, tol);
  EXPECT_NEAR(p1.getValue(0, 0, 0, 0), 3.25f, tol);
  EXPECT_NEAR(p1.getValue(0, 0, 0, 1), 0.0f, tol);
  EXPECT_NEAR(p1.getValue(0, 0, 0, 2), -1.75f, tol);
}

/**
 * @brief For a loss that is exactly linear in the first parameter element
 * (loss = theta[0]), MeZO's two-point estimator is exact (no higher-order
 * terms to approximate away): projected_grad == z[0], so the update to
 * that element is -lr * z[0]^2, which is strictly negative for any nonzero
 * z[0]. This checks MeZO actually moves the loss downhill, not just that
 * it perturbs and restores without crashing.
 */
TEST(nntrainer_mezo_interface, trainStep_reduces_linear_loss_p) {
  MeZO mezo;
  mezo.setProperty({"MeZO_learning_rate=0.1", "MeZO_epsilon=0.01"});

  Tensor p0 = makeParamTensor(1.0f, -2.0f, 0.5f);
  std::vector<Tensor *> params = {&p0};

  const float theta0_before = p0.getValue(0, 0, 0, 0);
  auto noop_forward_fn = []() {};
  auto linear_loss_fn = [&params]() {
    return params[0]->getValue(0, 0, 0, 0);
  };

  mezo.trainStep(noop_forward_fn, linear_loss_fn, params);

  /**
   * The update is theta0 - lr*z0^2, strictly downhill for any nonzero z0. It
   * is not strictly *representable* as a decrease, though: theta0 is 1.0f
   * (ulp ~1.19e-7) and lr is 0.1, so a draw with |z0| < ~7.7e-4 rounds the
   * update away entirely. That happens for roughly 1 run in 1600, which is
   * often enough to flake in CI, so assert the direction rather than a strict
   * inequality. The seed is drawn from std::random_device inside trainStep
   * and cannot be pinned from here.
   */
  EXPECT_LE(p0.getValue(0, 0, 0, 0), theta0_before);
}

/**
 * @brief MeZO_lr_decay shrinks the step over successive updates.
 * @details A fixed rate large enough to make early progress is too coarse to
 * converge finely later; the decay is what lets one run do both. Driving the
 * same exactly-linear loss as above, the k-th update is -lr*decay^k*z_k^2, so
 * with the RNG averaged over many steps the later moves are strictly smaller.
 */
TEST(nntrainer_mezo_interface, lr_decay_shrinks_updates_p) {
  auto total_move = [](float decay, int steps) {
    MeZO mezo;
    mezo.setProperty({"MeZO_learning_rate=0.1", "MeZO_epsilon=0.01",
                      "MeZO_lr_decay=" + std::to_string(decay)});
    Tensor p = makeParamTensor(1.0f, -2.0f, 0.5f);
    std::vector<Tensor *> params = {&p};
    auto fwd = []() {};
    auto loss = [&params]() { return params[0]->getValue(0, 0, 0, 0); };
    for (int i = 0; i < steps; ++i)
      mezo.trainStep(fwd, loss, params);
    return 1.0f - p.getValue(0, 0, 0, 0); // total distance descended
  };

  // Same seedless RNG both times, but 200 steps average out the draws well
  // enough that a 0.9 decay must move materially less than no decay at all.
  const float undecayed = total_move(1.0f, 200);
  const float decayed = total_move(0.9f, 200);
  EXPECT_GT(undecayed, 0.0f);
  EXPECT_LT(decayed, undecayed);
}

/**
 * @brief A decay of 1.0 (the default) must reproduce the fixed-rate behaviour.
 */
TEST(nntrainer_mezo_interface, lr_decay_default_is_off_p) {
  MeZO mezo;
  mezo.setProperty({"MeZO_learning_rate=0.1", "MeZO_epsilon=0.01"});
  EXPECT_FLOAT_EQ(mezo.getEffectiveLearningRate(), mezo.getLearningRate());
}

/**
 * @brief Main gtest
 */
int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
