// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   mha_core_train.cpp
 * @brief  Training path of MHACoreLayer (full-sequence forward with private
 *         activation caches, and the attention backward), kept out of
 *         mha_core.cpp so that file stays close to upstream. mha_core.cpp calls
 *         requestTrainingTensors(), trainForwarding(), resizeTrainingTensors()
 *         and updateTrainingTensors() from small hooks.
 *
 *         The attention-backward GEMMs and the training flash-attention forward
 *         are dispatched to the Hexagon DSP through the libggml-hexagon bridge
 *         when it is loadable (engine=htp), and run on the CPU otherwise.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fp16.h>
#include <layer_context.h>
#include <limits>
#include <mha_core.h>
#include <mutex>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <sstream>
#include <string>
#include <thread>
#include <thread_manager.h>
#include <unordered_map>
#include <util_func.h>
#include <vector>

namespace {

/**
 * @brief Flash attention bridge function type.
 * Dispatches Q/K/V/mask to DSP via nntr_htp_bridge_flash_attn.
 */
using flash_attn_fn = int (*)(const void *, const void *, const void *,
                              const void *, void *, unsigned int, unsigned int,
                              unsigned int, unsigned int, unsigned int, float,
                              int, int);

flash_attn_fn get_flash_attn_bridge() {
  static flash_attn_fn fn = []() -> flash_attn_fn {
    void *handle = dlopen("libggml-hexagon.so", RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
      ml_logw("MHACore: dlopen(libggml-hexagon.so) failed: %s "
              "(flash attention disabled, using CPU path)",
              dlerror());
      return nullptr;
    }
    void *s = dlsym(handle, "nntr_htp_bridge_flash_attn");
    if (!s) {
      ml_logw("MHACore: dlsym(nntr_htp_bridge_flash_attn) failed: %s "
              "(flash attention disabled, using CPU path)",
              dlerror());
      return nullptr;
    }
    ml_logi("MHACore: flash attention bridge loaded successfully");
    return reinterpret_cast<flash_attn_fn>(s);
  }();
  return fn;
}

/**
 * @brief Training variant of the flash attention bridge: [token][head][dim]
 * Q/K/V/out with K/V in plain heap memory. Null if libggml-hexagon.so is an
 * older build without nntr_htp_bridge_flash_attn_tokmajor, in which case
 * trainForwarding keeps its CPU attention loop.
 */
const flash_attn_fn &get_flash_attn_train_bridge() {
  static const flash_attn_fn fn = []() -> flash_attn_fn {
    void *handle = dlopen("libggml-hexagon.so", RTLD_NOW | RTLD_GLOBAL);
    if (!handle)
      return nullptr;
    void *s = dlsym(handle, "nntr_htp_bridge_flash_attn_tokmajor");
    if (!s) {
      ml_logw("MHACore: dlsym(nntr_htp_bridge_flash_attn_tokmajor) failed: %s "
              "(training uses the CPU attention loop)",
              dlerror());
      return nullptr;
    }
    return reinterpret_cast<flash_attn_fn>(s);
  }();
  return fn;
}

/**
 * @brief Build causal mask for flash attention.
 * mask[i][j] = 0 if j < cache_from + i + 1, else -INF (0xFC00).
 * Layout: [n_tokens, n_kv] F16, row-major.
 */
void build_causal_mask(std::vector<uint16_t> &mask, unsigned int n_tokens,
                       unsigned int n_kv, unsigned int cache_from) {
  mask.resize(n_tokens * n_kv);
  for (unsigned int i = 0; i < n_tokens; ++i) {
    const unsigned int valid_to = cache_from + i + 1;
    for (unsigned int j = 0; j < n_kv; ++j) {
      mask[i * n_kv + j] = (j < valid_to) ? 0x0000 : 0xFC00;
    }
  }
}

// Causal mask cache: key = (cache_from << 40) | (n_kv << 20) | n_tokens
static std::unordered_map<uint64_t, std::vector<uint16_t>> g_causal_mask_cache;

const std::vector<uint16_t> &get_cached_causal_mask(unsigned int n_tokens,
                                                    unsigned int n_kv,
                                                    unsigned int cache_from) {
  uint64_t key =
    ((uint64_t)cache_from << 40) | ((uint64_t)n_kv << 20) | (uint64_t)n_tokens;
  auto it = g_causal_mask_cache.find(key);
  if (it != g_causal_mask_cache.end()) {
    return it->second;
  }
  std::vector<uint16_t> &mask = g_causal_mask_cache[key];
  build_causal_mask(mask, n_tokens, n_kv, cache_from);
  return mask;
}

/**
 * @brief Check if flash attention should be used.
 * Enabled for prefill (step_size > 1) with head_dim a multiple of 64
 * (HMX fast path). The bridge (nntr_htp_bridge_flash_attn) and the DSP
 * kernel (hmx_flash_attn_ext in flash-attn-ops.c) both require
 * head_dim % 64 == 0.
 *
 * Two env knobs (mirroring the ones this was originally validated with,
 * see FLASH_ATTN_STATUS.md) let us bisect flash-attn from the rest of the
 * htp path without a rebuild:
 *  - NNTR_HEXAGON_FLASH_ATTN=0   force-disable (default: enabled)
 *  - NNTR_HEXAGON_FLASH_ATTN_MIN_TOKENS=<n>  below this step_size, fall
 *    back to the CPU attention path even if the bridge is available
 *    (default 1: training batches whole sequences, so the DSP takes them
 *    all; raise it if a very short sequence is dispatch-bound).
 */
bool should_use_flash_attn(unsigned int step_size, unsigned int head_dim,
                           bool is_prefill) {
  static const bool enabled = [] {
    const char *v = std::getenv("NNTR_HEXAGON_FLASH_ATTN");
    return !(v && std::string(v) == "0");
  }();
  static const unsigned int min_tokens = [] {
    const char *v = std::getenv("NNTR_HEXAGON_FLASH_ATTN_MIN_TOKENS");
    return v ? static_cast<unsigned int>(std::atoi(v)) : 1u;
  }();

  if (!enabled)
    return false;
  if (!is_prefill || step_size <= 1)
    return false;
  if (step_size < min_tokens)
    return false;
  if (head_dim % 64 != 0)
    return false;
  const flash_attn_fn &fn = get_flash_attn_bridge();
  return (fn != nullptr);
}

/**
 * @brief Batched FP32 SGEMM bridge function type.
 * Fuses many independent, tightly-packed row-major GEMMs into a single
 * FastRPC round trip. See nntr_htp_bridge_sgemm_batch_fp32 in
 * ggml-hexagon's nntr-htp-bridge.cpp for the full contract.
 */
using sgemm_batch_fn = int (*)(const float *const *, const float *const *,
                               float *const *, const unsigned int *,
                               const unsigned int *, const unsigned int *,
                               const int *, const int *, unsigned int);

sgemm_batch_fn get_sgemm_batch_bridge() {
  static sgemm_batch_fn fn = []() -> sgemm_batch_fn {
    void *handle = dlopen("libggml-hexagon.so", RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
      ml_logw("MHACore: dlopen(libggml-hexagon.so) failed: %s "
              "(attention-backward GEMM dispatch disabled, using CPU path)",
              dlerror());
      return nullptr;
    }
    void *s = dlsym(handle, "nntr_htp_bridge_sgemm_batch_fp32");
    if (!s) {
      ml_logw("MHACore: dlsym(nntr_htp_bridge_sgemm_batch_fp32) failed: %s "
              "(attention-backward GEMM dispatch disabled, using CPU path)",
              dlerror());
      return nullptr;
    }
    ml_logi("MHACore: sgemm_batch bridge loaded successfully");
    return reinterpret_cast<sgemm_batch_fn>(s);
  }();
  return fn;
}

/**
 * @brief Whether attention-backward should dispatch its reformulated GEMMs
 * to the DSP batch bridge instead of running them on the CPU.
 *
 *  - NNTR_HEXAGON_ATTN_BWD=0   force-disable (default: enabled)
 *  - NNTR_HEXAGON_ATTN_BWD_MIN_TOKENS=<n>  below this seq_len, fall back to
 *    CPU even if the bridge is available (default 1, like
 *    NNTR_HEXAGON_FLASH_ATTN_MIN_TOKENS).
 */
bool should_use_attn_bwd_gemm(unsigned int seq_len) {
  static const bool enabled = [] {
    const char *v = std::getenv("NNTR_HEXAGON_ATTN_BWD");
    return !(v && std::string(v) == "0");
  }();
  static const unsigned int min_tokens = [] {
    const char *v = std::getenv("NNTR_HEXAGON_ATTN_BWD_MIN_TOKENS");
    return v ? static_cast<unsigned int>(std::atoi(v)) : 1u;
  }();

  if (!enabled)
    return false;
  if (seq_len < min_tokens)
    return false;
  return get_sgemm_batch_bridge() != nullptr;
}

/**
 * @brief One tightly-packed row-major FP32 GEMM: C[M,N] = op(A) @ op(B),
 * where op(X) is a transpose iff the corresponding trans flag is set. No
 * lda/ldb/ldc - all three buffers must be tightly packed, matching the
 * nntr_htp_bridge_sgemm_(batch_)fp32 contract exactly so the two executors
 * are drop-in interchangeable.
 */
struct GemmDesc {
  const float *A;
  const float *B;
  float *C;
  unsigned int M, N, K;
  int transA, transB;
};

/**
 * @brief Plain triple-loop CPU implementation of the same contract as
 * nntr_htp_bridge_sgemm_fp32, used as the fallback executor for
 * run_batched_gemm() (and hence also as the only executor exercised by
 * host-side unit tests, which have no DSP to dlopen).
 */
void cpu_sgemm_fp32(const float *A, const float *B, float *C, unsigned int M,
                    unsigned int N, unsigned int K, int transA, int transB) {
  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (unsigned int k = 0; k < K; ++k) {
        const float a = transA ? A[k * M + m] : A[m * K + k];
        const float b = transB ? B[n * K + k] : B[k * N + n];
        sum += a * b;
      }
      C[m * N + n] = sum;
    }
  }
}

/**
 * @brief Hard cap on GEMMs per nntr_htp_bridge_sgemm_batch_fp32 FastRPC call
 * (bridge-side stack-allocated tensor arrays are fixed at 16 - see
 * nntr_htp_bridge_sgemm_batch_fp32 in nntr-htp-bridge.cpp).
 */
constexpr unsigned int kMaxBridgeGemmBatch = 16;

/**
 * @brief Runs a batch of independent GEMMs, dispatching chunks of up to
 * kMaxBridgeGemmBatch to the DSP in one FastRPC round trip each when
 * available and worthwhile (should_use_attn_bwd_gemm), falling back to
 * per-GEMM CPU execution for any chunk the DSP call fails on (or for
 * everything, when the layer is not tagged engine=htp).
 */
void run_batched_gemm(std::vector<GemmDesc> &descs, unsigned int seq_len,
                      bool on_dsp) {
  if (descs.empty())
    return;

  const sgemm_batch_fn fn = on_dsp && should_use_attn_bwd_gemm(seq_len)
                              ? get_sgemm_batch_bridge()
                              : nullptr;

  for (unsigned int base = 0; base < descs.size();
       base += kMaxBridgeGemmBatch) {
    const unsigned int n =
      std::min<unsigned int>(kMaxBridgeGemmBatch, descs.size() - base);
    bool dsp_ok = false;
    if (fn) {
      std::vector<const float *> a_ptrs(n), b_ptrs(n);
      std::vector<float *> c_ptrs(n);
      std::vector<unsigned int> ms(n), ns(n), ks(n);
      std::vector<int> tas(n), tbs(n);
      for (unsigned int i = 0; i < n; ++i) {
        const GemmDesc &d = descs[base + i];
        a_ptrs[i] = d.A;
        b_ptrs[i] = d.B;
        c_ptrs[i] = d.C;
        ms[i] = d.M;
        ns[i] = d.N;
        ks[i] = d.K;
        tas[i] = d.transA;
        tbs[i] = d.transB;
      }
      int rc = fn(a_ptrs.data(), b_ptrs.data(), c_ptrs.data(), ms.data(),
                  ns.data(), ks.data(), tas.data(), tbs.data(), n);
      if (rc == 0) {
        dsp_ok = true;
      } else {
        ml_logw("MHACore: nntr_htp_bridge_sgemm_batch_fp32 failed (rc=%d), "
                "falling back to CPU for this chunk of %u GEMMs",
                rc, n);
      }
    }
    if (!dsp_ok) {
      for (unsigned int i = 0; i < n; ++i) {
        const GemmDesc &d = descs[base + i];
        cpu_sgemm_fp32(d.A, d.B, d.C, d.M, d.N, d.K, d.transA, d.transB);
      }
    }
  }
}

} // namespace

namespace causallm {

void MHACoreLayer::requestTrainingTensors(nntrainer::InitLayerContext &context,
                                          unsigned int batch_size,
                                          unsigned int train_seq_len) {
  train_tensors_requested = true;

  /**
   * @note Sized from the actual training sequence length (query height), NOT
   * max_timestep: max_timestep budgets for prompt + all generated tokens, which
   * training never grows into, and the attention-weight cache is quadratic in
   * it.
   */
  ml::train::TensorDim q_roped_dim(
    {batch_size, 1, train_seq_len, num_heads_Q * head_dim},
    {context.getFormat(), ml::train::TensorDim::DataType::FP32});
  ml::train::TensorDim k_roped_dim(
    {batch_size, 1, train_seq_len, num_heads_KV * head_dim},
    {context.getFormat(), ml::train::TensorDim::DataType::FP32});
  ml::train::TensorDim attn_wt_dim(
    {batch_size, num_heads_Q, train_seq_len, train_seq_len},
    {context.getFormat(), ml::train::TensorDim::DataType::FP32});

  tensor_idx[AttentionParams::train_q_roped] = context.requestTensor(
    q_roped_dim, "train_q_roped", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::ITERATION_LIFESPAN);
  tensor_idx[AttentionParams::train_k_roped] = context.requestTensor(
    k_roped_dim, "train_k_roped", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::ITERATION_LIFESPAN);
  tensor_idx[AttentionParams::train_attn_wt] = context.requestTensor(
    attn_wt_dim, "train_attn_wt", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::ITERATION_LIFESPAN);
  tensor_idx[AttentionParams::train_v] =
    context.requestTensor(k_roped_dim, "train_v", nntrainer::Initializer::NONE,
                          false, nntrainer::TensorLifespan::ITERATION_LIFESPAN);
}

void MHACoreLayer::resizeTrainingTensors(nntrainer::RunLayerContext &context,
                                         unsigned int batch) {
  // A batch-size change alone must resize the caches too, or every batch index
  // beyond the one finalize() ran with reads/writes past its backing tensor.
  if (!train_tensors_requested)
    return;
  context.updateTensor(tensor_idx[AttentionParams::train_q_roped], batch);
  context.updateTensor(tensor_idx[AttentionParams::train_k_roped], batch);
  context.updateTensor(tensor_idx[AttentionParams::train_attn_wt], batch);
  context.updateTensor(tensor_idx[AttentionParams::train_v], batch);
}

void MHACoreLayer::updateTrainingTensors(
  nntrainer::RunLayerContext &context,
  const std::vector<nntrainer::TensorDim> &input_dimensions) {
  if (!train_tensors_requested)
    return;
  const unsigned int height = input_dimensions[0].height();
  ml::train::TensorDim kv_dim = input_dimensions[0];
  kv_dim.width(kv_dim.width() / (num_heads_Q / num_heads_KV));

  // Sized from the actual sequence length, not max_timestep — see the
  // matching note in finalize().
  ml::train::TensorDim q_roped_dim = input_dimensions[0];
  q_roped_dim.setDataType(ml::train::TensorDim::DataType::FP32);

  ml::train::TensorDim k_roped_dim = kv_dim;
  k_roped_dim.setDataType(ml::train::TensorDim::DataType::FP32);

  ml::train::TensorDim attn_wt_dim(
    {input_dimensions[0].batch(), num_heads_Q, height, height},
    {input_dimensions[0].getFormat(), ml::train::TensorDim::DataType::FP32});

  context.updateTensor(tensor_idx[AttentionParams::train_q_roped], q_roped_dim);
  context.updateTensor(tensor_idx[AttentionParams::train_k_roped], k_roped_dim);
  context.updateTensor(tensor_idx[AttentionParams::train_attn_wt], attn_wt_dim);
  context.updateTensor(tensor_idx[AttentionParams::train_v], k_roped_dim);
}

/**
 * @brief Full-sequence dense causal multi-head attention with GQA, used as
 *        the training forward for the internal-cache (!use_external_cache)
 *        path. See header for scope limitations.
 */
void MHACoreLayer::trainForwarding(nntrainer::RunLayerContext &context) {
  NNTR_THROW_IF(!train_tensors_requested, std::invalid_argument)
    << "[mha_core] training forward requires the layer to have been "
       "finalized in ExecutionMode::TRAIN (the Q/K/attention caches it "
       "needs are only requested then)";
  NNTR_THROW_IF(!is_causal, std::invalid_argument)
    << "[mha_core] training forward only supports causal attention for now";
  NNTR_THROW_IF(local_window_size != std::numeric_limits<unsigned int>::max(),
                std::invalid_argument)
    << "[mha_core] training forward does not support sliding window "
       "attention yet";
  NNTR_THROW_IF(use_sink, std::invalid_argument)
    << "[mha_core] training forward does not support attention sink yet";
  NNTR_THROW_IF(attn_logit_softcapping > 0.0f, std::invalid_argument)
    << "[mha_core] training forward does not support logit softcapping yet";

  nntrainer::Tensor &query = context.getInput(INOUT_INDEX::QUERY);
  nntrainer::Tensor &key = context.getInput(INOUT_INDEX::KEY);
  nntrainer::Tensor &value = context.getInput(INOUT_INDEX::VALUE);
  nntrainer::Tensor &output = context.getOutput(INOUT_INDEX::OUTPUT);

  NNTR_THROW_IF(query.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "[mha_core] training forward only supports FP32 for now";

  const unsigned int batch = query.getDim().batch();
  const unsigned int seq_len = query.getDim().height();
  const unsigned int gqa_size = num_heads_Q / num_heads_KV;
  const float scale = 1.0f / std::sqrt((float)head_dim);

  nntrainer::Tensor &train_q =
    context.getTensor(tensor_idx[AttentionParams::train_q_roped]);
  nntrainer::Tensor &train_k =
    context.getTensor(tensor_idx[AttentionParams::train_k_roped]);
  nntrainer::Tensor &train_attn =
    context.getTensor(tensor_idx[AttentionParams::train_attn_wt]);

  // Apply RoPE to Q, K into the persistent training caches, leaving the
  // original input tensors (query/key) untouched — calcDerivative reads raw
  // V straight from context.getInput(VALUE), and undoes RoPE on dQ/dK
  // itself rather than relying on any in-place mutation of the inputs.
  apply_rotary_emb_tensor_v2(query, train_q, head_dim, 0,
                             /*apply_rope=*/theta > 0.0f);
  apply_rotary_emb_tensor_v2(key, train_k, head_dim, 0,
                             /*apply_rope=*/theta > 0.0f);

  // Keep a private copy of V for calcDerivative(): the VALUE input buffer
  // is only guaranteed alive through forwarding().
  {
    nntrainer::Tensor &train_v =
      context.getTensor(tensor_idx[AttentionParams::train_v]);
    std::memcpy(train_v.getData<float>(), value.getData<float>(),
                sizeof(float) * value.size());
  }

  const ml::train::TensorDim q_dim = train_q.getDim(); // (B,1,max_ts,HQ*D)
  const ml::train::TensorDim k_dim = train_k.getDim(); // (B,1,max_ts,HKV*D)
  const ml::train::TensorDim v_dim = value.getDim();   // (B,1,seq,HKV*D)
  const ml::train::TensorDim o_dim = output.getDim();  // (B,1,seq,HQ*D)
  const ml::train::TensorDim attn_dim =
    train_attn.getDim(); // (B,HQ,max_ts,max_ts)

  const float *q_data = train_q.getData<float>();
  const float *k_data = train_k.getData<float>();
  const float *v_data = value.getData<float>();
  float *out_data = output.getData<float>();
  float *attn_data = train_attn.getData<float>();

  const size_t q_row_stride = q_dim.width();
  const size_t k_row_stride = k_dim.width();
  const size_t v_row_stride = v_dim.width();
  const size_t o_row_stride = o_dim.width();
  const size_t attn_row_stride = attn_dim.width();

  std::vector<float> scores(seq_len);

  // Plain O(S^2) CPU causal attention for one batch element - the original
  // trainForwarding body, factored out so it can serve both as the only
  // path (no DSP bridge) and as the exact-correctness fallback when the
  // flash-attention bridge call below fails partway through a batch.
  auto cpu_forward_batch = [&](unsigned int b) {
    const float *q_b = q_data + b * q_dim.getFeatureLen();
    const float *k_b = k_data + b * k_dim.getFeatureLen();
    const float *v_b = v_data + b * v_dim.getFeatureLen();
    float *o_b = out_data + b * o_dim.getFeatureLen();
    float *attn_b = attn_data + b * attn_dim.getFeatureLen();

    for (unsigned int qh = 0; qh < num_heads_Q; ++qh) {
      const unsigned int kv = qh / gqa_size;
      float *attn_h = attn_b + qh * attn_dim.height() * attn_row_stride;

      for (unsigned int i = 0; i < seq_len; ++i) {
        const float *q_row = q_b + i * q_row_stride + qh * head_dim;
        float *attn_row = attn_h + i * attn_row_stride;

        float max_score = -std::numeric_limits<float>::infinity();
        for (unsigned int j = 0; j <= i; ++j) {
          const float *k_row = k_b + j * k_row_stride + kv * head_dim;
          float s = 0.0f;
          for (unsigned int d = 0; d < head_dim; ++d)
            s += q_row[d] * k_row[d];
          s *= scale;
          scores[j] = s;
          if (s > max_score)
            max_score = s;
        }

        float sum = 0.0f;
        for (unsigned int j = 0; j <= i; ++j) {
          float e = std::exp(scores[j] - max_score);
          attn_row[j] = e;
          sum += e;
        }
        float inv_sum = 1.0f / sum;
        for (unsigned int j = 0; j <= i; ++j)
          attn_row[j] *= inv_sum;

        float *out_row = o_b + i * o_row_stride + qh * head_dim;
        for (unsigned int d = 0; d < head_dim; ++d)
          out_row[d] = 0.0f;
        for (unsigned int j = 0; j <= i; ++j) {
          const float *v_row = v_b + j * v_row_stride + kv * head_dim;
          float w = attn_row[j];
          for (unsigned int d = 0; d < head_dim; ++d)
            out_row[d] += w * v_row[d];
        }
      }
    }
  };

  last_forward_used_flash = false;

  // Training always processes the full sequence in one shot, so it is
  // "prefill" for should_use_flash_attn()'s purposes - same gate
  // (NNTR_HEXAGON_FLASH_ATTN / _MIN_TOKENS) as the inference path.
  // The DSP is used only for a layer tagged engine=htp: without the tag this
  // is the CPU reference path, whether or not libggml-hexagon.so is loadable.
  if (context.isComputeEngine("htp") &&
      should_use_flash_attn(seq_len, head_dim, /*is_prefill=*/true) &&
      get_flash_attn_train_bridge() != nullptr) {
    const flash_attn_fn &fn = get_flash_attn_train_bridge();
    // cache_from=0: training has no KV-cache prefix, the whole sequence is
    // fresh every step.
    const std::vector<uint16_t> &mask =
      get_cached_causal_mask(seq_len, seq_len, 0);

    // The bridge wants Q/K/V/out laid out as [token][head][dim] (see
    // nntr_htp_bridge_flash_attn's doc comment in nntr-htp-bridge.cpp) -
    // exactly how train_q/train_k/value/output already store them, and the
    // inference-path call above passes its tensors straight through with no
    // reshuffling either. train_q's and train_k's *_row_stride is their
    // max-timestep-sized tensor's width (HQ*D / HKV*D respectively) with no
    // extra padding, so the first seq_len rows of q_b/k_b are one flat
    // contiguous block; v_dim's height is exactly seq_len already. So Q and
    // out can be passed straight through; only K/V need converting from
    // FP32 to FP16 (the DSP flash-attention kernel always reads K/V as
    // FP16, matching the inference KV-cache layout, independent of
    // Q/output dtype) - a flat elementwise convert, no gathering.
    const size_t kv_elems = (size_t)seq_len * num_heads_KV * head_dim;
    std::vector<uint16_t> k_gather(kv_elems);
    std::vector<uint16_t> v_gather(kv_elems);

    bool all_ok = true;
    for (unsigned int b = 0; b < batch && all_ok; ++b) {
      const float *q_b = q_data + b * q_dim.getFeatureLen();
      const float *k_b = k_data + b * k_dim.getFeatureLen();
      const float *v_b = v_data + b * v_dim.getFeatureLen();
      float *o_b = out_data + b * o_dim.getFeatureLen();

      for (size_t idx = 0; idx < kv_elems; ++idx) {
        k_gather[idx] = nntrainer::compute_fp32_to_fp16(k_b[idx]);
        v_gather[idx] = nntrainer::compute_fp32_to_fp16(v_b[idx]);
      }

      int rc = fn(q_b, k_gather.data(), v_gather.data(), mask.data(), o_b,
                  seq_len, num_heads_Q, num_heads_KV, head_dim, seq_len, scale,
                  /*q_is_fp16=*/0, /*out_is_fp16=*/0);
      if (rc != 0) {
        ml_logw("MHACore: trainForwarding flash_attn returned %d (batch %u"
                " of %u) - falling back to CPU for the whole forward pass",
                rc, b, batch);
        all_ok = false;
      }
    }

    if (all_ok) {
      // train_attn_wt is intentionally left unpopulated: flash attention
      // never materializes the [S,S] softmax matrix. calcDerivative()
      // checks last_forward_used_flash and recomputes it from
      // train_q_roped/train_k_roped instead of reading this cache.
      last_forward_used_flash = true;
      return;
    }
    // Bridge failed partway through a batch: some batches' `output` may
    // already hold DSP results and none have train_attn_wt populated.
    // Redo every batch on the CPU path for a fully consistent result.
  }

  for (unsigned int b = 0; b < batch; ++b) {
    cpu_forward_batch(b);
  }
}

void MHACoreLayer::apply_inverse_rotary_emb(float *grad, unsigned int width,
                                            unsigned int dim,
                                            unsigned int height,
                                            unsigned int from) {
  if (!(theta > 0.0f))
    return; // no RoPE on this layer

  // The forward pass (apply_rotary_emb_tensor_v2) bound the FP32 cos/sin
  // tables; they are flat [seq_len, dim / 2] rows.
  const std::vector<float> *cos_tab =
    cached_freqs_cos != nullptr ? cached_freqs_cos : freqs_cos;
  const std::vector<float> *sin_tab =
    cached_freqs_sin != nullptr ? cached_freqs_sin : freqs_sin;
  NNTR_THROW_IF(cos_tab == nullptr || sin_tab == nullptr, std::runtime_error)
    << "[mha_core] RoPE tables not initialized before the attention backward";

  const unsigned int half_ = dim / 2;
  for (unsigned int h = 0; h < height; ++h) {
    const float *cos_ = cos_tab->data() + static_cast<size_t>(from + h) * half_;
    const float *sin_ = sin_tab->data() + static_cast<size_t>(from + h) * half_;
    float *row = grad + h * width;
    for (unsigned int w = 0; w < width; w += dim) {
      for (unsigned int k = 0; k < half_; ++k) {
        unsigned int i0 = w + k;
        unsigned int i1 = w + k + half_;
        float d_i0 = row[i0];
        float d_i1 = row[i1];
        row[i0] = cos_[k] * d_i0 + sin_[k] * d_i1;
        row[i1] = -sin_[k] * d_i0 + cos_[k] * d_i1;
      }
    }
  }
}

/**
 * @brief calcDerivative for the internal-cache (training) path.
 * @details Standard scaled-dot-product-attention backward (with GQA
 *          grouping and causal masking) against the Q/K (post-RoPE) and
 *          attention-weight caches populated by trainForwarding(), followed
 *          by undoing RoPE on dQ/dK (dV needs no inverse rotation since no
 *          rotation was applied to V). No-op in external-cache (inference)
 *          mode, where backward is never invoked.
 *
 * @note nntrainer aliases each outgoing derivative onto its input's own
 *       buffer, so dV IS the V input. V is still needed to form d_attn, so
 *       it must be snapshotted before dV is zeroed. dQ/dK are safe to zero
 *       directly because the backward pass reads the post-RoPE Q/K from the
 *       layer's own training caches rather than from those inputs.
 */
void MHACoreLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  if (use_external_cache || !train_tensors_requested)
    return;

  const bool on_dsp = context.isComputeEngine("htp");
  const nntrainer::Tensor &dy =
    context.getIncomingDerivative(INOUT_INDEX::OUTPUT);
  nntrainer::Tensor &dQ = context.getOutgoingDerivative(INOUT_INDEX::QUERY);
  nntrainer::Tensor &dK = context.getOutgoingDerivative(INOUT_INDEX::KEY);
  nntrainer::Tensor &dV = context.getOutgoingDerivative(INOUT_INDEX::VALUE);

  nntrainer::Tensor &train_q =
    context.getTensor(tensor_idx[AttentionParams::train_q_roped]);
  nntrainer::Tensor &train_k =
    context.getTensor(tensor_idx[AttentionParams::train_k_roped]);
  nntrainer::Tensor &train_attn =
    context.getTensor(tensor_idx[AttentionParams::train_attn_wt]);
  // V comes from the private copy taken in trainForwarding(); the VALUE
  // input buffer is not guaranteed alive here (and dV may alias it).
  const nntrainer::Tensor &value =
    context.getTensor(tensor_idx[AttentionParams::train_v]);

  const unsigned int batch = dQ.getDim().batch();
  const unsigned int seq_len = dQ.getDim().height();
  const unsigned int gqa_size = num_heads_Q / num_heads_KV;
  const float scale = 1.0f / std::sqrt((float)head_dim);
  const unsigned int S = seq_len;
  const unsigned int D = head_dim;

  dQ.setZero();
  dK.setZero();
  dV.setZero();

  const ml::train::TensorDim dy_dim = dy.getDim();     // (B,1,seq,HQ*D)
  const ml::train::TensorDim dq_dim = dQ.getDim();     // (B,1,seq,HQ*D)
  const ml::train::TensorDim dk_dim = dK.getDim();     // (B,1,seq,HKV*D)
  const ml::train::TensorDim dv_dim = dV.getDim();     // (B,1,seq,HKV*D)
  const ml::train::TensorDim q_dim = train_q.getDim(); // (B,1,max_ts,HQ*D)
  const ml::train::TensorDim k_dim = train_k.getDim(); // (B,1,max_ts,HKV*D)
  const ml::train::TensorDim v_dim = value.getDim();   // (B,1,seq,HKV*D)
  // (B,HQ,seq,seq): each (batch,head) block is tightly packed [seq,seq],
  // see updateTensorsByInputDimensions - sized from seq_len, not max_ts.
  const ml::train::TensorDim attn_dim = train_attn.getDim();

  const float *dy_data = dy.getData<float>();
  const float *q_data = train_q.getData<float>();
  const float *k_data = train_k.getData<float>();
  const float *v_data = value.getData<float>();
  float *attn_data = train_attn.getData<float>();
  float *dq_data = dQ.getData<float>();
  float *dk_data = dK.getData<float>();
  float *dv_data = dV.getData<float>();

  const size_t dy_row_stride = dy_dim.width();
  const size_t q_row_stride = q_dim.width();
  const size_t k_row_stride = k_dim.width();
  const size_t v_row_stride = v_dim.width();

  // Gather Q/K/V/dY per-head slices into contiguous scratch buffers (the
  // bridge GEMMs require tightly-packed [M,K]/[K,N] input, but these
  // tensors interleave all heads on the width dimension), run the
  // reformulated attention backward as two batches of GEMMs dispatched
  // through run_batched_gemm() (DSP when available and worthwhile, CPU
  // otherwise), then scatter/accumulate the results back.
  //
  // Indexing: per-(b,qh) buffers use idx = b * num_heads_Q + qh; per-(b,kv)
  // buffers use kv_idx = b * num_heads_KV + kv.
  // These buffers total ~50 MB per call and calcDerivative runs once per layer
  // per step, so allocating them fresh made the allocator map and unmap fresh
  // pages every time (~1.5 GB of page faults per step, over 20% of the process
  // time in the kernel). Keep them alive across calls and re-zero in place:
  // same contents the zero-initialized std::vectors had.
  auto reuse_zeroed = [](size_t slot, size_t n) -> std::vector<float> & {
    thread_local std::vector<float> bufs[10];
    bufs[slot].assign(n, 0.0f);
    return bufs[slot];
  };
  auto &q_gather = reuse_zeroed(0, batch * num_heads_Q * S * D);
  auto &dy_gather = reuse_zeroed(1, batch * num_heads_Q * S * D);
  auto &dq_scratch = reuse_zeroed(2, batch * num_heads_Q * S * D);
  auto &dk_partial = reuse_zeroed(3, batch * num_heads_Q * S * D);
  auto &dv_partial = reuse_zeroed(4, batch * num_heads_Q * S * D);
  // d_attn (GEMM1 output), overwritten in place with d_score (softmax
  // backward output) once GEMM1 completes - the two are never live at once.
  auto &score = reuse_zeroed(5, batch * num_heads_Q * S * S);

  auto &k_gather = reuse_zeroed(6, batch * num_heads_KV * S * D);
  auto &v_gather = reuse_zeroed(7, batch * num_heads_KV * S * D);
  auto &dk_accum = reuse_zeroed(8, batch * num_heads_KV * S * D);
  auto &dv_accum = reuse_zeroed(9, batch * num_heads_KV * S * D);

  auto gather_head = [](const float *src, float *dst, unsigned int rows,
                        size_t row_stride, unsigned int head_offset,
                        unsigned int width) {
    for (unsigned int r = 0; r < rows; ++r)
      std::memcpy(dst + r * width, src + r * row_stride + head_offset,
                  width * sizeof(float));
  };
  auto scatter_head = [](const float *src, float *dst, unsigned int rows,
                         size_t row_stride, unsigned int head_offset,
                         unsigned int width) {
    for (unsigned int r = 0; r < rows; ++r)
      std::memcpy(dst + r * row_stride + head_offset, src + r * width,
                  width * sizeof(float));
  };

  for (unsigned int b = 0; b < batch; ++b) {
    const float *k_b = k_data + b * k_dim.getFeatureLen();
    const float *v_b = v_data + b * v_dim.getFeatureLen();
    for (unsigned int kv = 0; kv < num_heads_KV; ++kv) {
      const unsigned int kv_idx = b * num_heads_KV + kv;
      gather_head(k_b, k_gather.data() + kv_idx * S * D, S, k_row_stride,
                  kv * D, D);
      gather_head(v_b, v_gather.data() + kv_idx * S * D, S, v_row_stride,
                  kv * D, D);
    }
  }

  // Round 1: d_attn[S,S] = dY_h[S,D] @ V_kv[S,D]^T.
  std::vector<GemmDesc> round1;
  round1.reserve(batch * num_heads_Q);
  for (unsigned int b = 0; b < batch; ++b) {
    const float *q_b = q_data + b * q_dim.getFeatureLen();
    const float *dy_b = dy_data + b * dy_dim.getFeatureLen();
    for (unsigned int qh = 0; qh < num_heads_Q; ++qh) {
      const unsigned int kv = qh / gqa_size;
      const unsigned int idx = b * num_heads_Q + qh;
      const unsigned int kv_idx = b * num_heads_KV + kv;
      gather_head(q_b, q_gather.data() + idx * S * D, S, q_row_stride, qh * D,
                  D);
      gather_head(dy_b, dy_gather.data() + idx * S * D, S, dy_row_stride,
                  qh * D, D);
      round1.push_back({dy_gather.data() + idx * S * D,
                        v_gather.data() + kv_idx * S * D,
                        score.data() + idx * S * S, S, S, D, 0, 1});
    }
  }
  run_batched_gemm(round1, S, on_dsp);

  // If trainForwarding() dispatched through the flash-attention bridge,
  // train_attn was never populated (flash attention never materializes the
  // [S,S] softmax matrix) - reconstruct it here from train_q_roped/
  // train_k_roped before the masking/softmax-backward step below reads it.
  // q_gather/k_gather are already fully populated by the loops above.
  if (last_forward_used_flash) {
    std::vector<GemmDesc> qk;
    qk.reserve(batch * num_heads_Q);
    for (unsigned int b = 0; b < batch; ++b) {
      float *attn_b = attn_data + b * attn_dim.getFeatureLen();
      for (unsigned int qh = 0; qh < num_heads_Q; ++qh) {
        const unsigned int kv = qh / gqa_size;
        const unsigned int idx = b * num_heads_Q + qh;
        const unsigned int kv_idx = b * num_heads_KV + kv;
        qk.push_back({q_gather.data() + idx * S * D,
                      k_gather.data() + kv_idx * S * D, attn_b + qh * S * S, S,
                      S, D, 0, 1});
      }
    }
    run_batched_gemm(qk, S, on_dsp);

    // qk left raw QK^T in attn_data (unscaled, unmasked). Apply scale and a
    // causal row-softmax in place, same math as trainForwarding's CPU path,
    // so the result is bit-for-bit what the non-flash path would have
    // cached here.
    for (unsigned int b = 0; b < batch; ++b) {
      float *attn_b = attn_data + b * attn_dim.getFeatureLen();
      for (unsigned int qh = 0; qh < num_heads_Q; ++qh) {
        float *attn_h = attn_b + qh * S * S;
        for (unsigned int i = 0; i < S; ++i) {
          float *row = attn_h + i * S;
          float max_score = -std::numeric_limits<float>::infinity();
          for (unsigned int j = 0; j <= i; ++j) {
            row[j] *= scale;
            if (row[j] > max_score)
              max_score = row[j];
          }
          float sum = 0.0f;
          for (unsigned int j = 0; j <= i; ++j) {
            row[j] = std::exp(row[j] - max_score);
            sum += row[j];
          }
          const float inv_sum = 1.0f / sum;
          for (unsigned int j = 0; j <= i; ++j)
            row[j] *= inv_sum;
          for (unsigned int j = i + 1; j < S; ++j)
            row[j] = 0.0f;
        }
      }
    }
  }

  // Zero-mask the strictly-upper triangle of each (b, qh) attn block
  // in-place (trainForwarding only ever wrote j <= i; the rest is
  // uninitialized tensor-pool memory), then softmax backward elementwise,
  // overwriting `score` (currently d_attn) with d_score * scale in place.
  // Because attn[i][j] is now 0 for j > i, d_score is automatically 0
  // there too - no separate causal mask is needed on d_score itself.
  for (unsigned int b = 0; b < batch; ++b) {
    float *attn_b = attn_data + b * attn_dim.getFeatureLen();
    for (unsigned int qh = 0; qh < num_heads_Q; ++qh) {
      float *attn_h = attn_b + qh * S * S;
      float *d_attn_h = score.data() + (b * num_heads_Q + qh) * S * S;

      for (unsigned int i = 0; i < S; ++i) {
        float *attn_row = attn_h + i * S;
        float *d_attn_row = d_attn_h + i * S;
        for (unsigned int j = i + 1; j < S; ++j)
          attn_row[j] = 0.0f;

        float dot_sum = 0.0f;
        for (unsigned int j = 0; j <= i; ++j)
          dot_sum += attn_row[j] * d_attn_row[j];

        for (unsigned int j = 0; j < S; ++j)
          d_attn_row[j] = attn_row[j] * (d_attn_row[j] - dot_sum) * scale;
      }
    }
  }

  // Round 2: dQ_h = d_score @ K_kv; dK_partial_h = d_score^T @ Q_h;
  // dV_partial_h = attn_masked^T @ dY_h.
  std::vector<GemmDesc> round2;
  round2.reserve(batch * num_heads_Q * 3);
  for (unsigned int b = 0; b < batch; ++b) {
    const float *attn_b = attn_data + b * attn_dim.getFeatureLen();
    for (unsigned int qh = 0; qh < num_heads_Q; ++qh) {
      const unsigned int kv = qh / gqa_size;
      const unsigned int idx = b * num_heads_Q + qh;
      const unsigned int kv_idx = b * num_heads_KV + kv;
      const float *attn_h = attn_b + qh * S * S;
      const float *d_score_h = score.data() + idx * S * S;

      round2.push_back({d_score_h, k_gather.data() + kv_idx * S * D,
                        dq_scratch.data() + idx * S * D, S, D, S, 0, 0});
      round2.push_back({d_score_h, q_gather.data() + idx * S * D,
                        dk_partial.data() + idx * S * D, S, D, S, 1, 0});
      round2.push_back({attn_h, dy_gather.data() + idx * S * D,
                        dv_partial.data() + idx * S * D, S, D, S, 1, 0});
    }
  }
  run_batched_gemm(round2, S, on_dsp);

  // Scatter dQ directly (disjoint per-Q-head output, no accumulation);
  // accumulate dK/dV partials across each KV-head's gqa_size Q-heads before
  // scattering.
  for (unsigned int b = 0; b < batch; ++b) {
    float *dq_b = dq_data + b * dq_dim.getFeatureLen();
    for (unsigned int qh = 0; qh < num_heads_Q; ++qh) {
      const unsigned int idx = b * num_heads_Q + qh;
      scatter_head(dq_scratch.data() + idx * S * D, dq_b, S, q_row_stride,
                   qh * D, D);
    }
  }

  for (unsigned int b = 0; b < batch; ++b) {
    for (unsigned int qh = 0; qh < num_heads_Q; ++qh) {
      const unsigned int kv = qh / gqa_size;
      const unsigned int idx = b * num_heads_Q + qh;
      const unsigned int kv_idx = b * num_heads_KV + kv;
      float *dk_acc = dk_accum.data() + kv_idx * S * D;
      float *dv_acc = dv_accum.data() + kv_idx * S * D;
      const float *dk_p = dk_partial.data() + idx * S * D;
      const float *dv_p = dv_partial.data() + idx * S * D;
      for (unsigned int i = 0; i < S * D; ++i) {
        dk_acc[i] += dk_p[i];
        dv_acc[i] += dv_p[i];
      }
    }
  }

  for (unsigned int b = 0; b < batch; ++b) {
    float *dk_b = dk_data + b * dk_dim.getFeatureLen();
    float *dv_b = dv_data + b * dv_dim.getFeatureLen();
    for (unsigned int kv = 0; kv < num_heads_KV; ++kv) {
      const unsigned int kv_idx = b * num_heads_KV + kv;
      scatter_head(dk_accum.data() + kv_idx * S * D, dk_b, S, k_row_stride,
                   kv * D, D);
      scatter_head(dv_accum.data() + kv_idx * S * D, dv_b, S, v_row_stride,
                   kv * D, D);
    }
  }

  // undo RoPE on dQ, dK (dV needs no inverse rotation: no rotation was
  // applied to V in trainForwarding).
  for (unsigned int b = 0; b < batch; ++b) {
    apply_inverse_rotary_emb(dq_data + b * dq_dim.getFeatureLen(),
                             dq_dim.width(), head_dim, seq_len, 0);
    apply_inverse_rotary_emb(dk_data + b * dk_dim.getFeatureLen(),
                             dk_dim.width(), head_dim, seq_len, 0);
  }
}

void MHACoreLayer::calcGradient(nntrainer::RunLayerContext &context) {}

} // namespace causallm
