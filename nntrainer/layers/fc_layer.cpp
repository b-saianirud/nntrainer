/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
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
 *
 *
 * @file	fc_layer.cpp
 * @date	14 May 2020
 * @brief	This is Fully Connected Layer Class for Neural Network
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <common_properties.h>
#include <compute_ops.h>
#include <fc_layer.h>
#include <layer_context.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <quantizer.h>
#include <util_func.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum FCParams { weight, bias };
enum LORAParams { loraA, loraB, loraTmp, loraOut };

namespace {
/** a Q4_0/Q6_K base weight kept frozen under LoRA */
bool isFrozenQuantizedBase(const Tensor &weight) {
  return weight.getDataType() == TensorDim::DataType::Q4_0 ||
         weight.getDataType() == TensorDim::DataType::Q6_K;
}

/** the backend ops table of the layer's context (CPU table when none) */
ComputeOps *frozenOps(RunLayerContext &context) {
  ComputeOps *ops = context.getComputeOps();
  return ops != nullptr ? ops : get_cpu_ops();
}
} // namespace

FullyConnectedLayer::FullyConnectedLayer() :
  LayerImpl(),
  lora_scaling(1.0f),
  fc_props(props::Unit(), props::LoraRank(), props::LoraAlpha(),
           props::FusedActivation(), props::PlanLastRowOnly()),
  quantizer(nullptr) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
  lora_idx.fill(std::numeric_limits<unsigned>::max());
}

void FullyConnectedLayer::finalize(InitLayerContext &context) {
  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<props::BiasDecay>(*layer_impl_props);
  auto &bias_initializer = std::get<props::BiasInitializer>(*layer_impl_props);
  auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);

  const auto &unit = std::get<props::Unit>(fc_props).get();
  const auto &lora_rank = (std::get<props::LoraRank>(fc_props).empty())
                            ? 0
                            : std::get<props::LoraRank>(fc_props).get();
  lora_scaling = (lora_rank && !std::get<props::LoraAlpha>(fc_props).empty())
                   ? (float)std::get<props::LoraAlpha>(fc_props) / lora_rank
                   : 1;
  if (!std::get<props::SkipPrefill>(*layer_impl_props).empty())
    skip_prefill = std::get<props::SkipPrefill>(*layer_impl_props).get();

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Fully connected layer takes only one input";

  std::vector<TensorDim> output_dims(1);

  /// @todo fc actaully supports multidimensions. EffDimFlag shouldn't be fixed
  /// like this.
  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  bool is_nchw = (context.getFormat() == Tformat::NCHW);
  /** set output dimensions */
  auto const &in_dim = context.getInputDimensions()[0];

  /// The CPU kernels take the input in the activation dtype. With an FP32
  /// input in an FP16-activation model the dot is dispatched on the FP32
  /// input and writes FP32 results into the FP16 output, so the layer returns
  /// NaN or garbage without an error. The input layer keeps the dtype it is
  /// declared with, so an FP16 model fed straight from an input layer has to
  /// declare input_dtype=FP16 there.
  const auto act_dtype = context.getActivationDataType();
  const auto in_dtype = in_dim.getDataType();
  const auto float_name = [](TensorDim::DataType t) -> const char * {
    return t == TensorDim::DataType::FP32   ? "FP32"
           : t == TensorDim::DataType::FP16 ? "FP16"
                                            : nullptr;
  };
  NNTR_THROW_IF(float_name(act_dtype) && float_name(in_dtype) &&
                  in_dtype != act_dtype,
                std::invalid_argument)
    << "fully_connected layer '" << context.getName() << "': input dtype is "
    << float_name(in_dtype) << " but the model activation dtype is "
    << float_name(act_dtype) << "; feed it in " << float_name(act_dtype)
    << " (e.g. input_dtype=" << float_name(act_dtype) << " on the input layer)";

  output_dims[0] = in_dim;
  is_nchw ? output_dims[0].width(unit) : output_dims[0].channel(unit);

  // Same planning contract the neutral FC states, so the property reads the
  // same whichever engine resolves "fully_connected": a head the model has
  // declared to produce only its last row is planned at height 1 instead of
  // the graph-build height, whose rows above the first are allocated and never
  // written. Read with skip_prefill, which is what makes the rest of the plane
  // dead in the first place.
  auto &plan_last_row_only = std::get<props::PlanLastRowOnly>(fc_props);
  if (skip_prefill && !plan_last_row_only.empty() && plan_last_row_only.get())
    output_dims[0].height(1);

  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  context.setOutputDimensions(output_dims);

  /** set weight specifications */
  // @todo : This NCHW format setting is just temporal, it needs to be set by
  // global configuration

  /** Bias Dimension : (1, 1, 1, unit) */
  /// @note Bias is un-quantized and added directly to the activation. Its
  /// storage dtype must match how it is laid out on disk:
  ///  - float weight (FP16/FP32): bias is stored in the activation dtype, so
  ///    request it as such (no cast needed at the add site).
  ///  - quantized weight (Q4_0/Q6_K/QINT*/...): bias is stored FP32 on disk;
  ///    requesting it as the (possibly FP16) activation dtype would reinterpret
  ///    the FP32 bytes and corrupt it. Request FP32 and cast to the activation
  ///    dtype at the add site below.
  const auto weight_dtype = context.getWeightDataType();
  const bool weight_is_float = (weight_dtype == TensorDim::DataType::FP32 ||
                                weight_dtype == TensorDim::DataType::FP16);
  const auto bias_dtype = weight_is_float ? context.getActivationDataType()
                                          : TensorDim::DataType::FP32;
  TensorDim bias_dim(1, is_nchw ? 1 : unit, 1, is_nchw ? unit : 1,
                     TensorDim::TensorType(context.getFormat(), bias_dtype),
                     is_nchw ? 0b0001 : 0b0100);

  /** Weight Dimension : (1, 1, in_dim.width(), unit)*/
  TensorDim weight_dim(
    1, is_nchw ? 1 : unit, is_nchw ? in_dim.width() : 1,
    is_nchw ? unit : in_dim.channel(),
    TensorDim::TensorType(context.getFormat(), context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  weight_idx[FCParams::weight] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "weight", (lora_rank == 0));

  if (disable_bias.empty() || disable_bias.get() == false) {
    weight_idx[FCParams::bias] =
      context.requestWeight(bias_dim, bias_initializer, WeightRegularizer::NONE,
                            1.0f, bias_decay, "bias", (lora_rank == 0));
  }

  /** create weights for LoRA */
  if (lora_rank) {

    // LoRA adapters are always FP32: gradients and the optimizer need the
    // precision, and they are tiny next to the Q4_0 base weight.
    const auto lora_dtype = TensorDim::DataType::FP32;

    /** loraA Dimension : (1, 1, in_dim.width, lora_rank) */
    TensorDim loraA_dim(1, is_nchw ? 1 : lora_rank,
                        is_nchw ? in_dim.width() : 1,
                        is_nchw ? lora_rank : in_dim.channel(),
                        TensorDim::TensorType(context.getFormat(), lora_dtype),
                        is_nchw ? 0b0011 : 0b0101);

    /** loraB Dimension : (1, 1, lora_rank, unit) */
    TensorDim loraB_dim(1, is_nchw ? 1 : unit, is_nchw ? lora_rank : 1,
                        is_nchw ? unit : lora_rank,
                        TensorDim::TensorType(context.getFormat(), lora_dtype),
                        is_nchw ? 0b0011 : 0b0101);

    /** loraTmp Dimension : (B, 1, in_dim.height(), lora_rank) */
    TensorDim loraTmp_dim(
      in_dim.batch(), is_nchw ? 1 : lora_rank, is_nchw ? in_dim.height() : 1,
      is_nchw ? lora_rank : in_dim.width(),
      TensorDim::TensorType(context.getFormat(),
                            context.getActivationDataType()),
      is_nchw ? 0b1011 : 0b1101);

    /** loraTmp Dimension : (B, 1, in_dim.height(), unit) */
    TensorDim loraOut_dim(
      in_dim.batch(), is_nchw ? 1 : unit, is_nchw ? in_dim.height() : 1,
      is_nchw ? unit : in_dim.width(),
      TensorDim::TensorType(context.getFormat(),
                            context.getActivationDataType()),
      is_nchw ? 0b1011 : 0b1101);

    /**
     * @note Standard LoRA initialization (Hu et al. 2021): A is drawn from a
     * zero-mean distribution and B is zero, so the adapter contributes
     * B*A == 0 at step 0 and the pretrained model is reproduced exactly.
     *
     * The reverse assignment (A zero, B random) also gives B*A == 0, but the
     * first update to A is then steered entirely by the random B, which made
     * training diverge above lr ~1e-6 on Qwen3-0.6B; this ordering is stable
     * at 1e-4.
     */
    lora_idx[LORAParams::loraA] = context.requestWeight(
      loraA_dim, Initializer::LECUN_NORMAL, weight_regularizer,
      weight_regularizer_constant, weight_decay, "loraA", true);

    lora_idx[LORAParams::loraB] = context.requestWeight(
      loraB_dim, Initializer::ZEROS, weight_regularizer,
      weight_regularizer_constant, weight_decay, "loraB", true);

    lora_idx[LORAParams::loraTmp] =
      context.requestTensor(loraTmp_dim, "hidden_tmp_lora", Initializer::NONE,
                            true, TensorLifespan::FORWARD_GRAD_LIFESPAN);

    lora_idx[LORAParams::loraOut] =
      context.requestTensor(loraOut_dim, "hidden_lora", Initializer::NONE, true,
                            TensorLifespan::FORWARD_FUNC_LIFESPAN);
  }

  ///@todo this quantizaer should be moved to tensor, not layer!
  switch (context.getWeightDataType()) {
  case ml::train::TensorDim::DataType::QINT4:
  case ml::train::TensorDim::DataType::QINT8:
  case ml::train::TensorDim::DataType::QINT16:
    quantizer =
      Quantization::createQuantizer(nntrainer::QScheme::PER_TENSOR_AFFINE);
    break;
  default:
    quantizer = nullptr;
    break;
  }

  // inline fused activation: if the FusionRealizer moved an activation onto
  // this FC (fused_activation set), build the SAME ActiFunc the standalone
  // ActivationLayer would use, so the fused forward is value-identical.
  auto &fused_act = std::get<props::FusedActivation>(fc_props);
  rejectFusedActivationOnTrainingGraph(fused_act, context.getExecutionMode(),
                                       "FC");
  if (!fused_act.empty() && fused_act.get() != ActivationType::ACT_NONE) {
    if (context.getActivationDataType() == TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
      acti_func.setActiFunc<_FP16>(fused_act.get());
#else
      throw std::invalid_argument(
        "[FC] fused fp16 activation needs enable-fp16");
#endif
    } else {
      acti_func.setActiFunc<float>(fused_act.get());
    }
  }
}

void FullyConnectedLayer::exportTo(
  Exporter &exporter, const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(fc_props, method, this);
}

void FullyConnectedLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, fc_props);
  LayerImpl::setProperty(remain_props);
}

void FullyConnectedLayer::setBatch(nntrainer::RunLayerContext &context,
                                   unsigned int batch) {
  if (!std::get<props::LoraRank>(fc_props).empty()) {
    // update Lora Tensor's batch info.
    context.updateTensor(lora_idx[LORAParams::loraTmp], batch);
    context.updateTensor(lora_idx[LORAParams::loraOut], batch);
  }
}

void FullyConnectedLayer::pack(RunLayerContext &context) {
  for (auto &w : context.getWeights()) {
    Tensor &var = w->getVariableRef();
    if (var.getDataType() == TensorDim::DataType::QS4CX)
      var.pack();
  }
}

void FullyConnectedLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  ///@todo This dequantization action should be moved to tensor.dot()
  if (quantizer != nullptr) {
    Tensor weight_ = quantizer->dequantize(weight, input_.getDataType());
    input_.dot(weight_, hidden_, false, false);
  } else if (isFrozenQuantizedBase(weight)) {
    // LoRA on a quantized base: the frozen weight is multiplied at the
    // activation's full precision (the fused W4A8 kernel would quantize the
    // activation as well, and that noise stops training from converging).
    // The op table owns the matmul, so an accelerator can keep the weight
    // resident and run it there.
    frozenOps(context)->fc_frozen(input_, weight, hidden_);
  } else {
    input_.dot(weight, hidden_, false, false);
  }

  if (!std::get<props::LoraRank>(fc_props).empty()) {
    Tensor &loraA = context.getWeight(lora_idx[LORAParams::loraA]);
    Tensor &loraB = context.getWeight(lora_idx[LORAParams::loraB]);
    Tensor &hidden_tmp_lora = context.getTensor(lora_idx[LORAParams::loraTmp]);
    Tensor &hidden_out_lora = context.getTensor(lora_idx[LORAParams::loraOut]);

    // Scale the small [rows x rank] intermediate instead of the full
    // [rows x out] LoRA output. loraTmp therefore holds scaling * (X*A),
    // which is what calcGradient expects.
    input_.dot(loraA, hidden_tmp_lora, false, false);
    hidden_tmp_lora.multiply_i(lora_scaling);
    hidden_tmp_lora.dot(loraB, hidden_out_lora, false, false);

    hidden_.add_i(hidden_out_lora);
  }

  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &bias = context.getWeight(weight_idx[FCParams::bias]);
    if (bias.getDataType() != hidden_.getDataType()) {
      Tensor bias_cast = bias.clone(hidden_.getDataType());
      hidden_.add_i(bias_cast);
    } else {
      hidden_.add_i(bias);
    }
  }

  // fused activation epilogue: apply the activation inline on the GEMM
  // output, eliminating the separate ActivationLayer node. In-place when the
  // activation supports it (relu/sigmoid/tanh); otherwise via a temp input
  // copy, exactly mirroring ActivationLayer's run_fn(input, output) ->
  // value-identical.
  auto &fused_act = std::get<props::FusedActivation>(fc_props);
  if (!fused_act.empty() && fused_act.get() != ActivationType::ACT_NONE) {
    if (acti_func.supportInPlace()) {
      acti_func.run_fn(hidden_, hidden_);
    } else {
      Tensor in_copy = hidden_.clone();
      acti_func.run_fn(in_copy, hidden_);
    }
  }
}

void FullyConnectedLayer::incremental_forwarding(RunLayerContext &context,
                                                 unsigned int from,
                                                 unsigned int to,
                                                 bool training) {
  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor loraA, loraB;
  TensorDim hidden_tmp_lora_dim, hidden_out_lora_dim;

  bool is_prefill = !from || (to - from) > 1;
  if (skip_prefill && is_prefill)
    return;

  const bool has_lora = !std::get<props::LoraRank>(fc_props).empty();
  if (has_lora) {
    loraA = context.getWeight(lora_idx[LORAParams::loraA]);
    loraB = context.getWeight(lora_idx[LORAParams::loraB]);
    /**
     * @note loraTmp/loraOut are requested with FORWARD_GRAD_LIFESPAN /
     * FORWARD_FUNC_LIFESPAN, which are not allocated when the graph is
     * compiled without a backward pass (pure inference). This function is
     * the inference/decode path, so only shape metadata is read from them
     * here (always valid, independent of data allocation); the actual
     * scratch computation below uses freshly-allocated local tensors rather
     * than a shared-data view into this possibly-unallocated storage.
     */
    hidden_tmp_lora_dim =
      context.getTensor(lora_idx[LORAParams::loraTmp]).getDim();
    hidden_out_lora_dim =
      context.getTensor(lora_idx[LORAParams::loraOut]).getDim();
  }

  TensorDim input_dim = input_.getDim();
  TensorDim hidden_dim = hidden_.getDim();

  TensorDim input_step_dim = input_dim;
  TensorDim hidden_step_dim = hidden_dim;

  input_step_dim.batch(1);
  if (input_dim.height() > 1)
    input_step_dim.height(to - from);
  hidden_step_dim.batch(1);
  if (hidden_dim.height() > 1)
    hidden_step_dim.height(to - from);

  auto &fused_act = std::get<props::FusedActivation>(fc_props);

  // @todo make it parallelized with batch axis
  for (unsigned int b = 0; b < hidden_.batch(); ++b) {
    // input_ and hidden_ (output) generally have different widths (e.g. Q/K/V
    // projections, up/down/gate projections), so each tensor's per-batch
    // offset must use its own featureLen. Using hidden_dim's featureLen for
    // input_ (as before) either overflows input_'s backing memory when
    // hidden_'s width is larger (thrown by getSharedDataTensor's bounds
    // check), or silently reads from the wrong offset when it is smaller.
    Tensor input_step = input_.getSharedDataTensor(
      input_step_dim, b * input_dim.getFeatureLen(), true);
    Tensor hidden_step = hidden_.getSharedDataTensor(
      hidden_step_dim, b * hidden_dim.getFeatureLen(), true);

    input_step.dot(weight, hidden_step, false, false);

    if (has_lora) {
      nntrainer::TensorDim hidden_tmp_lora_step_dim = hidden_tmp_lora_dim;
      hidden_tmp_lora_step_dim.batch(1);
      if (hidden_tmp_lora_step_dim.height() > 1)
        hidden_tmp_lora_step_dim.height(to - from);

      nntrainer::TensorDim hidden_out_lora_step_dim = hidden_out_lora_dim;
      hidden_out_lora_step_dim.batch(1);
      if (hidden_out_lora_step_dim.height() > 1)
        hidden_out_lora_step_dim.height(to - from);

      nntrainer::Tensor hidden_tmp_lora_step(hidden_tmp_lora_step_dim, true);
      nntrainer::Tensor hidden_out_lora_step(hidden_out_lora_step_dim, true);

      input_step.dot(loraA, hidden_tmp_lora_step, false, false);
      hidden_tmp_lora_step.dot(loraB, hidden_out_lora_step, false, false);
      hidden_out_lora_step.multiply_i(lora_scaling);
      hidden_step.add_i(hidden_out_lora_step);
    }

    if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
        disable_bias.empty() || disable_bias.get() == false) {
      Tensor &bias = context.getWeight(weight_idx[FCParams::bias]);
      if (bias.getDataType() != hidden_step.getDataType()) {
        Tensor bias_cast = bias.clone(hidden_step.getDataType());
        hidden_step.add_i(bias_cast);
      } else {
        hidden_step.add_i(bias);
      }
    }

    // The same epilogue forwarding() applies. Without it one layer with one
    // fused_activation value would produce two different results depending on
    // which entry point ran, and the incremental path -- the one an LLM graph
    // takes -- would return pre-activation values.
    if (!fused_act.empty() && fused_act.get() != ActivationType::ACT_NONE) {
      if (acti_func.supportInPlace()) {
        acti_func.run_fn(hidden_step, hidden_step);
      } else {
        Tensor in_copy = hidden_step.clone();
        acti_func.run_fn(in_copy, hidden_step);
      }
    }
  }
}

void FullyConnectedLayer::calcDerivative(RunLayerContext &context) {
  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);

  const Tensor &derivative_ = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &ret_ = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  if (!std::get<props::LoraRank>(fc_props).empty()) {
    Tensor &lora_A = context.getWeight(lora_idx[LORAParams::loraA]);
    Tensor &lora_B = context.getWeight(lora_idx[LORAParams::loraB]);

    if (quantizer == nullptr && isFrozenQuantizedBase(weight)) {
      // dx = dy * (W + s*A*B)^T. The frozen base term goes through the op table
      // (exact dequantization on the CPU, resident weight on an accelerator);
      // the LoRA term is (dy * B^T) * A^T * s, which avoids forming the dense
      // [in x out] product A*B.
      frozenOps(context)->fc_frozen_dgrad(derivative_, weight, ret_);
      Tensor tmp = derivative_.dot(lora_B, false, true);
      tmp.multiply_i(lora_scaling);
      // accumulate into ret_ (beta = 1) instead of building a full-size
      // temporary and adding it
      tmp.dot(lora_A, ret_, false, true, 1.0f);
    } else {
      Tensor w_deq = quantizer != nullptr
                       ? quantizer->dequantize(weight, ret_.getDataType())
                       : weight;
      Tensor lora_contrib = lora_A.dot(lora_B).multiply(lora_scaling);
      ret_.dot_deriv_wrt_1(w_deq.add(lora_contrib), derivative_, false, false);
    }
  } else {
    ret_.dot_deriv_wrt_1(weight, derivative_, false, false);
  }
}

void FullyConnectedLayer::calcGradient(RunLayerContext &context) {

  /** (default) calcGradient - compute gradient of weight and bias */
  if (std::get<props::LoraRank>(fc_props).empty()) {
    Tensor &djdw = context.getWeightGrad(weight_idx[FCParams::weight]);
    djdw.setZero();

    const Tensor &derivative_ = context.getIncomingDerivative(SINGLE_INOUT_IDX);
    Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

    if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
        disable_bias.empty() || disable_bias.get() == false) {
      Tensor &djdb = context.getWeightGrad(weight_idx[FCParams::bias]);
      djdb.setZero();

      if (context.isGradientFirstAccess(weight_idx[FCParams::bias])) {
        derivative_.sum({0, 1, 2}, djdb);
      } else {
        /// @todo optimize below by adding beta to Tensor::sum
        Tensor t = derivative_.sum({0, 1, 2});
        djdb.add_i(t);
      }
    }

    input_.dot_deriv_wrt_2(
      djdw, derivative_, false, false,
      !context.isGradientFirstAccess(weight_idx[FCParams::weight]));
  } else {
    /** (lora) calcGradient - compute gradients of LoRA params only */
    Tensor &djdla = context.getWeightGrad(lora_idx[LORAParams::loraA]);
    Tensor &djdlb = context.getWeightGrad(lora_idx[LORAParams::loraB]);
    Tensor &djdtmp = context.getTensorGrad(lora_idx[LORAParams::loraTmp]);

    const Tensor &derivative_ = context.getIncomingDerivative(SINGLE_INOUT_IDX);
    Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
    Tensor &loraB = context.getWeight(lora_idx[LORAParams::loraB]);
    Tensor &loraTmp = context.getTensor(lora_idx[LORAParams::loraTmp]);
    // The forward already folded the scaling into loraTmp, so dB needs no
    // extra factor and the scaling is applied to the small d(loraTmp) below
    // instead of copying and scaling the full dY.
    const Tensor &lora_derivative_ = derivative_;

    loraTmp.dot_deriv_wrt_2(
      djdlb, lora_derivative_, false, false,
      !context.isGradientFirstAccess(lora_idx[LORAParams::loraB]));

    if (context.isGradientFirstAccess(lora_idx[LORAParams::loraTmp])) {
      djdtmp.dot_deriv_wrt_1(loraB, lora_derivative_, false, false, false);
      djdtmp.multiply_i(lora_scaling);
    } else {
      Tensor dtmp(djdtmp.getDim(), true);
      dtmp.dot_deriv_wrt_1(loraB, lora_derivative_, false, false, false);
      dtmp.multiply_i(lora_scaling);
      djdtmp.add_i(dtmp);
    }
    input_.dot_deriv_wrt_2(
      djdla, djdtmp, false, false,
      !context.isGradientFirstAccess(lora_idx[LORAParams::loraA]));
  }
}

} /* namespace nntrainer */
