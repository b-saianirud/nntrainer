// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Sachin Singh <sachin.3@samsung.com>
 *
 * @file   mezo.h
 * @date   17 March 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Sachin Singh <sachin.3@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the MeZO optimizer.
 */
#ifndef __MeZO_H__
#define __MeZO_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <cmath>

#include <optimizer_devel.h>

namespace nntrainer {

/**
 * @class MeZOEpsilon
 * @brief Property class for MeZO epsilon (perturbation magnitude)
 */
class MeZOEpsilon : public Property<float> {
public:
  static constexpr const char *key =
    "MeZO_epsilon";                /**< unique key to access */
  using prop_tag = float_prop_tag; /**< property type */
};

/**
 * @class MeZOLearningRate
 * @brief Property class for MeZO learning rate
 */
class MeZOLearningRate : public Property<float> {
public:
  static constexpr const char *key =
    "MeZO_learning_rate";          /**< unique key to access */
  using prop_tag = float_prop_tag; /**< property type */
};

/**
 * @class MeZOLearningRateDecay
 * @brief Per-step multiplicative decay applied to the MeZO learning rate
 * @details MeZO takes one update per step from a single scalar estimate of a
 * very high-dimensional gradient, so a rate large enough to make early
 * progress is too coarse to converge finely later: the loss descends, then
 * random-walks around a floor. Decaying the rate lets the same run do both.
 * 1.0 (the default) reproduces the previous fixed-rate behaviour exactly.
 */
class MeZOLearningRateDecay : public Property<float> {
public:
  MeZOLearningRateDecay(float value = 1.0f) { set(value); }
  static constexpr const char *key =
    "MeZO_lr_decay";               /**< unique key to access */
  using prop_tag = float_prop_tag; /**< property type */
};

/**
 * @class MeZOMinLearningRate
 * @brief Floor the decayed learning rate so it never reaches zero
 */
class MeZOMinLearningRate : public Property<float> {
public:
  MeZOMinLearningRate(float value = 0.0f) { set(value); }
  static constexpr const char *key =
    "MeZO_min_learning_rate";      /**< unique key to access */
  using prop_tag = float_prop_tag; /**< property type */
};

/**
 * @class   MeZO optimizer class
 * @brief   MeZO (Zero'th order) optimizer class
 */
class MeZO : public Optimizer {
public:
  /**
   * @brief Construct a new MeZO object
   *
   */
  MeZO();

  /**
   * @copydoc Optimizer::getDefaultLearningRate()
   *
   */
  double getDefaultLearningRate() const override { return 0.0002; }

  /**
   * @copydoc applyGradient(RunOptimizerContext &context)
   */
  void applyGradient(RunOptimizerContext &context) override { return; }

  /**
   * @brief     Check if this optimizer requires backpropagation
   * @retval    true if backprop is required, false otherwise
   */
  bool requiresBackprop() const override { return false; }

  /**
   * @brief     Custom training step for MeZO optimizer
   * @param[in] forward_fn Function to perform forward pass
   * @param[in] get_loss_fn Function to get current loss
   * @param[in] params Vector of parameter tensors to update
   */
  float trainStep(std::function<void()> forward_fn,
                  std::function<float()> get_loss_fn,
                  std::vector<Tensor *> &params) override;

  /**
   * @copydoc Optimizer::getType()
   */
  const std::string getType() const override { return MeZO::type; }

  /**
   * @copydoc Optimizer::getOptimizerVariableDim(const TensorDim &dim)
   */
  std::vector<TensorDim>
  getOptimizerVariableDim(const TensorDim &dim) override {
    return {};
  }

  /**
   * @brief Set Optimizer Parameters
   * @param[in] values Optimizer Parameter list
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @brief Get the epsilon value for perturbation
   * @return Perturbation magnitude
   */
  float getEpsilon() const { return std::get<MeZOEpsilon>(mezo_props).get(); }

  /**
   * @brief Get the learning rate
   * @return Learning rate value
   */
  float getLearningRate() const {
    return std::get<MeZOLearningRate>(mezo_props).get();
  }

  /**
   * @brief Learning rate for the update about to be applied
   * @return base rate decayed by MeZO_lr_decay^step_count, floored at
   *         MeZO_min_learning_rate
   */
  float getEffectiveLearningRate() const {
    const float decay = std::get<MeZOLearningRateDecay>(mezo_props).get();
    const float base = getLearningRate();
    if (decay >= 1.0f)
      return base;
    const float lr = base * std::pow(decay, static_cast<float>(step_count));
    return std::max(lr, std::get<MeZOMinLearningRate>(mezo_props).get());
  }

  /**
   * @brief Update multiple weights using MeZO gradient estimation
   * @param weights Vector of weights to update
   * @param seed Random seed for reproducibility
   * @param projected_grad Estimated gradient computed as (loss_plus -
   * loss_minus) / (2 * epsilon)
   */
  void updateWeightsMeZO(std::vector<nntrainer::Tensor *> &weights, int seed,
                         float projected_grad);

  /**
   * @brief Perturb parameters using normal distribution
   * @param params Vector of parameter tensors to perturb
   * @param epsilon Perturbation magnitude. It is multipled with the direction
   * of perturbation (+1, -1, or -2 etc)
   * @param seed Random seed for reproducibility
   */
  void perturbParameters(std::vector<nntrainer::Tensor *> &params,
                         float epsilon, int seed);

  static constexpr const char *type = "MeZO";

private:
  std::tuple<MeZOEpsilon, MeZOLearningRate, MeZOLearningRateDecay,
             MeZOMinLearningRate>
    mezo_props;          /**< MeZO epsilon, learning rate and its decay */
  size_t step_count = 0; /**< updates applied so far, drives the decay */
};
} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __MeZO_H__ */
