// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Sachin Singh <sachin.3@samsung.com>
 *
 * @file   mezo.cpp
 * @date   17 March 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Sachin Singh <sachin.3@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the MeZO optimizer.
 */

#include <mezo.h>
#include <node_exporter.h>
#include <random>
#include <stdexcept>
#include <util_func.h>

namespace nntrainer {

MeZO::MeZO() :
  mezo_props(MeZOEpsilon(), MeZOLearningRate(), MeZOLearningRateDecay(),
             MeZOMinLearningRate()) {

  auto &[epsilon, learningRate, lrDecay, minLearningRate] = mezo_props;
  epsilon.set(0.01f);
  learningRate.set(0.0001f);
  /** 1.0 keeps the rate fixed, i.e. the pre-decay behaviour */
  lrDecay.set(1.0f);
  minLearningRate.set(0.0f);
}

namespace {
/**
 * @brief Reject any weight MeZO cannot perturb correctly.
 * @details Both weight loops below reach for a raw float* via getData<float>()
 * and walk getDataLen() elements. Tensor::getData<T>() is an unchecked cast, so
 * on a half-precision or quantized weight that pointer reinterprets the buffer
 * and the loop runs past its end -- silent weight corruption plus a heap
 * overflow. FP32 is the only layout this arithmetic is valid for, so say so
 * loudly rather than corrupting memory. Reachable today via
 * model_tensor_type=FP16-FP16 / fc_layer_dtype=FP16 in nntr_config.json.
 */
void assertPerturbableFP32(const nntrainer::Tensor *t) {
  if (t->getDataType() != ml::train::TensorDim::DataType::FP32)
    throw std::invalid_argument(
      "[MeZO] only FP32 weights can be perturbed; got a non-FP32 tensor. "
      "Load the model with model_tensor_type=FP32-FP32.");
}
} // namespace

void MeZO::perturbParameters(std::vector<Tensor *> &params, float epsilon,
                             int seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> normal_dist(0.0f, 1.0f);

  /**
   * @note z is applied element-by-element straight into the weight rather
   * than materializing a full noise tensor first. Holding one is pointless
   * here - each element is consumed immediately - and for a large embedding
   * table the temporary is hundreds of MB, allocated and freed four times per
   * step. Keeping it out preserves MeZO's "no model-sized buffers" property.
   * The RNG is drawn in the same order as before, so the regenerated z is
   * bit-identical to the one used by the other perturbations of this step.
   */
  for (auto *ptr : params) {
    assertPerturbableFP32(ptr);
    float *data = ptr->getData<float>();
    size_t param_size = ptr->getDim().getDataLen();
    for (size_t i = 0; i < param_size; ++i) {
      // θi ← θi + ϵz
      data[i] += epsilon * normal_dist(gen);
    }
  }
}

float MeZO::trainStep(std::function<void()> forward_fn,
                      std::function<float()> get_loss_fn,
                      std::vector<Tensor *> &params) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> distrib(0, 1000000000 - 1);
  int seed = distrib(gen);

  float mezo_epsilon = getEpsilon();

  // Perturb parameters with +epsilon
  perturbParameters(params, mezo_epsilon, seed);

  // Forward pass with positive perturbation
  forward_fn();
  float loss_plus = get_loss_fn();

  // Perturb parameters with -2*epsilon (to get from +epsilon to -epsilon)
  perturbParameters(params, -2 * mezo_epsilon, seed);

  // Forward pass with negative perturbation
  forward_fn();
  float loss_minus = get_loss_fn();

  // Restoring weight to original state
  perturbParameters(params, mezo_epsilon, seed);

  float projected_grad = (loss_plus - loss_minus) / (2.0f * mezo_epsilon);

  updateWeightsMeZO(params, seed, projected_grad);
  ++step_count;

  /**
   * Report the midpoint of the two probe losses rather than whatever the last
   * forward left behind. The caller's get_loss_fn() reads the loss of the most
   * recent forward pass, which here is the θ-εz probe - a randomly perturbed
   * model, not the weights being trained. That turns the reported curve into
   * noise. (L₊ + L₋)/2 cancels the ±εz terms and equals L(θ) + O(ε²), so it
   * tracks the real loss and costs nothing extra.
   */
  return 0.5f * (loss_plus + loss_minus);
}

void MeZO::setProperty(const std::vector<std::string> &values) {

  auto remain_props = loadProperties(values, mezo_props);

  if (!remain_props.empty()) {
    std::string msg = "[MeZO Optimizer] Unknown Properties count " +
                      std::to_string(remain_props.size());
    throw exception::not_supported(msg);
  }
}

void MeZO::updateWeightsMeZO(std::vector<nntrainer::Tensor *> &weights,
                             int seed, float projected_grad) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> normal_dist(0.0f, 1.0f);

  const float lr = getEffectiveLearningRate();
  const float scale = -lr * projected_grad;

  /** @note in-place for the same reason as perturbParameters() */
  for (auto *ptr : weights) {
    assertPerturbableFP32(ptr);
    float *data = ptr->getData<float>();
    size_t param_size = ptr->getDim().getDataLen();
    for (size_t i = 0; i < param_size; ++i) {
      // θi ← θi −ηt ∗ projected_grad ∗ z
      data[i] += scale * normal_dist(gen);
    }
  }
}

} // namespace nntrainer
