// SPDX-License-Identifier: Apache-2.0
/**
 * @file   causal_lm_loss.h
 * @brief  Masked per-token cross-entropy loss for full-sequence causal LM
 *         training
 * @bug    No known bugs except for NYI items
 *
 * @details Input is the logits of every sequence position, [B,1,S,V]. The
 *          label is NOT one-hot: it is [B,1,1,S] holding the target token id
 *          of each position as a float, with a negative value marking a
 *          position that carries no loss (padding, or a position whose
 *          target is unknown). That keeps the label at S floats per sample
 *          instead of S*V.
 *
 *          loss = mean over all labelled positions of the whole batch of
 *                 -log softmax(logits)[target]
 *          which matches the default reduction of the HF/PyTorch causal-LM
 *          loss. d logits = (softmax - onehot) / n_labelled on labelled
 *          positions and exactly 0 elsewhere.
 */

#ifndef __CAUSAL_LM_LOSS_LAYER_H__
#define __CAUSAL_LM_LOSS_LAYER_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>

namespace causallm {

/**
 * @brief Token-level validation statistics accumulated by every non-training
 *        forwarding() (i.e. validation / eval) of the loss layer. Unlike the
 *        framework's valid_loss (mean of per-batch means) this is weighted by
 *        the number of labelled tokens, and it also tracks top-1/top-5
 *        next-token accuracy. The trainer reads and resets it per epoch.
 */
struct CausalLMEvalStats {
  double nll = 0.0;
  size_t tokens = 0;
  size_t top1 = 0;
  size_t top5 = 0;
  void reset() { *this = CausalLMEvalStats(); }
};
WIN_EXPORT extern CausalLMEvalStats g_causal_lm_eval_stats;

WIN_EXPORT class CausalLMLossLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT CausalLMLossLayer() : Layer() {}
  WIN_EXPORT ~CausalLMLossLayer() = default;

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;

  WIN_EXPORT bool supportBackwarding() const override { return true; }
  WIN_EXPORT bool requireLabel() const override { return true; }

  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override {}

  WIN_EXPORT const std::string getType() const override {
    return CausalLMLossLayer::type;
  }

  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "causal_lm_loss";
};

} // namespace causallm

#endif /* __CAUSAL_LM_LOSS_LAYER_H__ */
