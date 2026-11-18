// SPDX-License-Identifier: Apache-2.0
/**
 * @file   causal_lm_loss.cpp
 * @brief  Masked per-token cross-entropy loss for full-sequence causal LM
 *         training. See causal_lm_loss.h for the label convention.
 * @bug    No known bugs except for NYI items
 */

#include <algorithm>
#include <cmath>
#include <limits>

#include "causal_lm_loss.h"

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

CausalLMEvalStats g_causal_lm_eval_stats;

namespace {

/** @brief Target id of position s, or -1 if it carries no loss. */
inline int targetOf(const float *label_b, unsigned int s, unsigned int vocab) {
  const float t = label_b[s];
  if (!(t >= 0.0f) || t >= static_cast<float>(vocab))
    return -1;
  return static_cast<int>(t);
}

/** @brief logsumexp of one logits row (numerically stable). */
inline float logSumExp(const float *row, unsigned int vocab) {
  float mx = row[0];
  for (unsigned int v = 1; v < vocab; ++v)
    mx = std::max(mx, row[v]);
  double sum = 0.0;
  for (unsigned int v = 0; v < vocab; ++v)
    sum += std::exp(static_cast<double>(row[v] - mx));
  return mx + static_cast<float>(std::log(sum));
}

} // namespace

void CausalLMLossLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "[causal_lm_loss] takes exactly one input";

  const nntrainer::TensorDim in_dim = context.getInputDimensions()[0];
  NNTR_THROW_IF(in_dim.channel() != 1, std::invalid_argument)
    << "[causal_lm_loss] logits must be [B,1,S,V]";

  // The output is [B,1,1,S] FP32 and doubles as the label shape: one target
  // id per position. forwarding() fills it with each position's NLL.
  nntrainer::TensorDim out_dim = in_dim;
  out_dim.height(1);
  out_dim.width(in_dim.height());
  out_dim.setDataType(nntrainer::TensorDim::DataType::FP32);
  context.setOutputDimensions({out_dim});
}

void CausalLMLossLayer::forwarding(nntrainer::RunLayerContext &context,
                                   bool training) {
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  out.setZero();
  if (!context.isLabelAvailable(SINGLE_INOUT_IDX))
    return;

  const nntrainer::Tensor &logits = context.getInput(SINGLE_INOUT_IDX);
  const nntrainer::Tensor &label = context.getLabel(SINGLE_INOUT_IDX);
  const unsigned int batch = logits.batch();
  const unsigned int seq = logits.height();
  const unsigned int vocab = logits.width();

  const float *x = logits.getData<float>();
  const float *y = label.getData<float>();
  float *nll = out.getData<float>();

  double total = 0.0;
  size_t count = 0;
  for (unsigned int b = 0; b < batch; ++b) {
    const float *y_b = y + static_cast<size_t>(b) * seq;
    for (unsigned int s = 0; s < seq; ++s) {
      const int t = targetOf(y_b, s, vocab);
      if (t < 0)
        continue;
      const float *row = x + (static_cast<size_t>(b) * seq + s) * vocab;
      const float l = logSumExp(row, vocab) - row[t];
      nll[static_cast<size_t>(b) * seq + s] = l;
      total += l;
      ++count;
      if (!training) {
        // Rank of the target = number of logits strictly above it.
        unsigned int rank = 0;
        for (unsigned int v = 0; v < vocab; ++v)
          rank += row[v] > row[t];
        auto &st = g_causal_lm_eval_stats;
        st.nll += l;
        ++st.tokens;
        st.top1 += rank < 1;
        st.top5 += rank < 5;
      }
    }
  }
  context.setLoss(count ? static_cast<float>(total / count) : 0.0f);
}

void CausalLMLossLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  nntrainer::Tensor &dx = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  // For a label layer the incoming derivative *is* the label tensor.
  const nntrainer::Tensor &label =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);
  const nntrainer::Tensor &logits = context.getInput(SINGLE_INOUT_IDX);
  const unsigned int batch = logits.batch();
  const unsigned int seq = logits.height();
  const unsigned int vocab = logits.width();

  const float *x = logits.getData<float>();
  const float *y = label.getData<float>();
  float *g = dx.getData<float>();

  // dx may share memory with the logits (the stock loss layers note the same),
  // so never bulk-zero it: each row is read fully before it is overwritten,
  // and unlabelled rows are zeroed individually.
  size_t count = 0;
  for (unsigned int b = 0; b < batch; ++b)
    for (unsigned int s = 0; s < seq; ++s)
      if (targetOf(y + static_cast<size_t>(b) * seq, s, vocab) >= 0)
        ++count;
  const float inv_n = count ? 1.0f / static_cast<float>(count) : 0.0f;

  for (unsigned int b = 0; b < batch; ++b) {
    const float *y_b = y + static_cast<size_t>(b) * seq;
    for (unsigned int s = 0; s < seq; ++s) {
      const int t = targetOf(y_b, s, vocab);
      const size_t off = (static_cast<size_t>(b) * seq + s) * vocab;
      const float *row = x + off;
      float *grow = g + off;
      if (t < 0) {
        std::fill(grow, grow + vocab, 0.0f);
        continue;
      }
      const float lse = logSumExp(row, vocab);
      for (unsigned int v = 0; v < vocab; ++v)
        grow[v] = std::exp(row[v] - lse) * inv_n;
      grow[t] -= inv_n;
    }
  }
}

void CausalLMLossLayer::setProperty(const std::vector<std::string> &values) {
  NNTR_THROW_IF(!values.empty(), std::invalid_argument)
    << "[causal_lm_loss] takes no properties, got " << values.size();
}

} // namespace causallm
