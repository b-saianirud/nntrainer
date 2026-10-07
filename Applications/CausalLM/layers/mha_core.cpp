// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   mha_core.cpp
 * @date   11 July 2025
 * @see    https://github.com/nntrainer/nntrainer
 *         https://arxiv.org/abs/1706.03762
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This code is based on custom_multi_head_attention_layer.cpp.
 *         This code is a part of the break down version of the mha layer.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

static std::mutex rope_init_mtx;

#include <fp16.h>
#include <layer_context.h>
#include <mha_core.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <thread_manager.h>
#include <util_func.h>

#include <cstdint>

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
 * cdsp path without a rebuild:
 *  - NNTR_HEXAGON_FLASH_ATTN=0   force-disable (default: enabled)
 *  - NNTR_HEXAGON_FLASH_ATTN_MIN_TOKENS=<n>  below this step_size, fall
 *    back to the CPU attention path even if the bridge is available
 *    (default 160 - below that the FastRPC round trip isn't worth it).
 */
bool should_use_flash_attn(unsigned int step_size, unsigned int head_dim,
                           bool is_prefill) {
  static const bool enabled = [] {
    const char *v = std::getenv("NNTR_HEXAGON_FLASH_ATTN");
    return !(v && std::string(v) == "0");
  }();
  static const unsigned int min_tokens = [] {
    const char *v = std::getenv("NNTR_HEXAGON_FLASH_ATTN_MIN_TOKENS");
    return v ? static_cast<unsigned int>(std::atoi(v)) : 160u;
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
 *    CPU even if the bridge is available (default 160, mirroring
 *    NNTR_HEXAGON_FLASH_ATTN_MIN_TOKENS - below that the FastRPC round trip
 *    isn't worth it).
 */
bool should_use_attn_bwd_gemm(unsigned int seq_len) {
  static const bool enabled = [] {
    const char *v = std::getenv("NNTR_HEXAGON_ATTN_BWD");
    return !(v && std::string(v) == "0");
  }();
  static const unsigned int min_tokens = [] {
    const char *v = std::getenv("NNTR_HEXAGON_ATTN_BWD_MIN_TOKENS");
    return v ? static_cast<unsigned int>(std::atoi(v)) : 160u;
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
 * everything, when the DSP path isn't used at all).
 */
void run_batched_gemm(std::vector<GemmDesc> &descs, unsigned int seq_len) {
  if (descs.empty())
    return;

  const sgemm_batch_fn fn =
    should_use_attn_bwd_gemm(seq_len) ? get_sgemm_batch_bridge() : nullptr;

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

inline float convert_scalar(uint16_t h) {
  return nntrainer::compute_fp16_to_fp32(h);
}

namespace causallm {

#define tile_size 4

static void compute_kcaches_fp32_reference(
  const float *in, const float *kcache, float *output, int num_rows,
  int num_cache_head, int head_dim, int gqa_size, size_t local_window_size,
  int head_start = 0, int head_end = -1) {
  const int actual_head_end = (head_end < 0) ? num_cache_head : head_end;
  NNTR_THROW_IF(head_start >= actual_head_end, std::invalid_argument)
    << "head_start (" << head_start << ") must be less than head_end ("
    << actual_head_end << ")";

  const int window = static_cast<int>(
    std::min(static_cast<size_t>(num_rows), local_window_size));
  const int start_row = num_rows - window;
  const float inv_sqrt_head_dim =
    1.0f / std::sqrt(static_cast<float>(head_dim));

  for (int n = head_start; n < actual_head_end; ++n) {
    for (int g = 0; g < gqa_size; ++g) {
      const float *query = in + (n * gqa_size + g) * head_dim;
      for (int row = start_row; row < num_rows; ++row) {
        const float *key = kcache + (row * num_cache_head + n) * head_dim;
        float sum = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
          sum += query[d] * key[d];
        }
        output[(row - start_row) * num_cache_head * gqa_size + n * gqa_size +
               g] = sum * inv_sqrt_head_dim;
      }
    }
  }
}

static void compute_vcache_fp32_transposed_reference(
  int row_num, const float *in, const float *vcache, float *output,
  int num_cache_head, int gqa_size, int head_dim, size_t local_window_size,
  int head_start = 0, int head_end = -1) {
  const int actual_head_end = (head_end < 0) ? num_cache_head : head_end;
  NNTR_THROW_IF(head_start >= actual_head_end, std::invalid_argument)
    << "head_start (" << head_start << ") must be less than head_end ("
    << actual_head_end << ")";

  const int window = static_cast<int>(
    std::min(static_cast<size_t>(row_num + 1), local_window_size));
  const int start_row = row_num + 1 - window;

  for (int n = head_start; n < actual_head_end; ++n) {
    for (int h = 0; h < gqa_size; ++h) {
      float *out = output + (n * gqa_size + h) * head_dim;
      std::fill(out, out + head_dim, 0.0f);

      for (int row = start_row; row <= row_num; ++row) {
        const int attn_row = row - start_row;
        const float a_val =
          in[attn_row * (num_cache_head * gqa_size) + n * gqa_size + h];
        const float *value = vcache + (row * num_cache_head + n) * head_dim;
        for (int d = 0; d < head_dim; ++d) {
          out[d] += a_val * value[d];
        }
      }
    }
  }
}

/************************************************************** */

/**
 * @brief constructor of MHACoreLayer
 */
MHACoreLayer::MHACoreLayer() :
  mha_core_props(
    nntrainer::props::NumHeads(), props::NumHeads_KV(),
    nntrainer::props::ProjectedKeyDim(), nntrainer::props::ProjectedValueDim(),
    nntrainer::props::OutputShape(), nntrainer::props::DropOutRate(),
    nntrainer::props::ReturnAttentionWeight(),
    nntrainer::props::AverageAttentionWeight(), nntrainer::props::MaxTimestep(),
    props::SlidingWindow(), props::MaxNewTokens(), props::RopeTheta(),
    props::UseRope(), props::MaxPositionEmbeddings(), props::UseSink(),
    props::RopeScalingType(), props::RopeScalingFactor(),
    props::RopePartialRotaryFactor(), props::RopeScalingMaxPositionEmbeddings(),
    props::AttnLogitSoftcapping(), props::IsCausal()),
  sm(nntrainer::ActivationType::ACT_SOFTMAX),
  epsilon(1e-3),
  cache_index(0),
  num_heads_Q(0),
  num_heads_KV(0),
  head_dim(0),
  cache_shift(false) {
  tensor_idx.fill(std::numeric_limits<unsigned>::max());
}

MHACoreLayer::~MHACoreLayer() {}

/************************************************************** */

void MHACoreLayer::finalize(nntrainer::InitLayerContext &context) {

  NNTR_THROW_IF(context.getNumInputs() < 3 || context.getNumInputs() > 5,
                std::invalid_argument)
    << "Multi head Attention layer needs 3, 4, or 5 inputs. "
       "(query, key, value; mask is optional; external cache_key + cache_value "
       "for external cache mode)";

  use_external_cache = (context.getNumInputs() >= 5);
  ml::train::TensorDim::TensorType activation_type = {
    context.getFormat(), context.getActivationDataType()};
  ml::train::TensorDim empty_dim(activation_type);

  const std::vector<ml::train::TensorDim> &input_dims =
    context.getInputDimensions();
  const ml::train::TensorDim &query_dim = input_dims[INOUT_INDEX::QUERY];
  const ml::train::TensorDim &key_dim = input_dims[INOUT_INDEX::KEY];

  /** max time step of this model */
  const unsigned int max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();

  /** max position embeddings */
  max_position_embeddings =
    std::get<props::MaxPositionEmbeddings>(mha_core_props).get();

  /** local window size */
  local_window_size = std::get<props::SlidingWindow>(mha_core_props).get();

  /** use rope */
  use_rope = std::get<props::UseRope>(mha_core_props).get();

  /** attention scaling computation */
  rope_scaling_type = std::get<props::RopeScalingType>(mha_core_props).get();
  scale = std::get<props::RopeScalingFactor>(mha_core_props).get();
  rope_partial_rotary_factor =
    std::get<props::RopePartialRotaryFactor>(mha_core_props).get();
  if (rope_scaling_type == "yarn")
    original_max_position_embeddings =
      std::get<props::RopeScalingMaxPositionEmbeddings>(mha_core_props).get();

  /** query_dim = (B, 1, seq_len, H_Q * Head_Dim ) */
  const unsigned int batch_size = query_dim.batch();
  const unsigned int query_width = query_dim.width();
  /** key_dim = (B, 1, max_seq_len, H_KV * Head_Dim ) */
  const unsigned int key_width = key_dim.width();

  /**
   *  @note If NumHeads_KV is set, then use the value. Otherwise,
   *        we initialize num_heads_KV with num_heads_Q.
   */
  num_heads_Q = static_cast<size_t>(
    std::get<nntrainer::props::NumHeads>(mha_core_props).get());
  num_heads_KV =
    std::get<props::NumHeads_KV>(mha_core_props).empty()
      ? num_heads_Q
      : static_cast<size_t>(std::get<props::NumHeads_KV>(mha_core_props).get());

  // head_dim
  head_dim = static_cast<size_t>(query_width) / num_heads_Q;
  NNTR_THROW_IF(head_dim != key_width / num_heads_KV, std::invalid_argument)
    << "num_heads_Q and num_heads_KV are not properly given. Please check the "
       "num_heads_* are set correctly so that the `head_dim`s are all same for "
       "query / key / value";

  /** Weight for Sink */
  use_sink = std::get<props::UseSink>(mha_core_props).get();
  if (use_sink) {
#if ENABLE_FP16 && defined(__ANDROID__)
    nntrainer::TensorDim sink_dim(
      1, 1, 1, num_heads_Q,
      nntrainer::TensorDim::TensorType(context.getFormat(),
                                       ml::train::TensorDim::DataType::FP16));
#else
    nntrainer::TensorDim sink_dim(
      1, 1, 1, num_heads_Q,
      nntrainer::TensorDim::TensorType(context.getFormat(),
                                       context.getActivationDataType()));
#endif
    sink_idx = context.requestWeight(sink_dim, nntrainer::Initializer::ZEROS,
                                     nntrainer::WeightRegularizer::NONE, 0.0f,
                                     0.0f, "sink");
  }

  attn_logit_softcapping =
    std::get<props::AttnLogitSoftcapping>(mha_core_props).get();

  /** Is Causal */
  is_causal = std::get<props::IsCausal>(mha_core_props).get();

  if (!std::get<nntrainer::props::SkipPrefill>(*layer_impl_props).empty())
    skip_prefill =
      std::get<nntrainer::props::SkipPrefill>(*layer_impl_props).get();

  /** Tensor for KV-Cache (only allocate internally when not using external
   * cache) */
  if (!use_external_cache) {
    const bool for_training =
      (context.getExecutionMode() == ml::train::ExecutionMode::TRAIN);

    /**
     * @note The KV cache exists purely to serve incremental (one-token-at-a-
     * time) decoding. Training always consumes the whole fixed-length
     * sequence in a single forward pass (see trainForwarding()), so the
     * cache is dead weight there — skip it rather than reserving
     * max_timestep * num_heads_KV * head_dim per layer for nothing.
     */
    if (!for_training) {
#ifdef ENABLE_FP16
      ml::train::TensorDim cache_key_dim(
        {batch_size, 1, max_timestep, num_heads_KV * head_dim},
        {context.getFormat(), ml::train::TensorDim::DataType::FP16});
      ml::train::TensorDim cache_value_dim(
        {batch_size, 1, max_timestep, num_heads_KV * head_dim},
        {context.getFormat(), ml::train::TensorDim::DataType::FP16});
#else
      ml::train::TensorDim cache_key_dim(
        {batch_size, 1, max_timestep, num_heads_KV * head_dim},
        {context.getFormat(), ml::train::TensorDim::DataType::UINT16});
      ml::train::TensorDim cache_value_dim(
        {batch_size, 1, max_timestep, num_heads_KV * head_dim},
        {context.getFormat(), ml::train::TensorDim::DataType::UINT16});
#endif

      tensor_idx[AttentionParams::cache_key] = context.requestTensor(
        cache_key_dim, "cache_key", nntrainer::Initializer::NONE, false,
        nntrainer::TensorLifespan::MAX_LIFESPAN);
      tensor_idx[AttentionParams::cache_value] = context.requestTensor(
        cache_value_dim, "cache_value", nntrainer::Initializer::NONE, false,
        nntrainer::TensorLifespan::MAX_LIFESPAN);
      internal_cache_requested = true;
    }

    if (for_training) {
      train_tensors_requested = true;

      /**
       * @note Sized from the actual training sequence length (query height),
       * NOT max_timestep. max_timestep budgets for prompt + all generated
       * tokens during decode, which training never grows into — and the
       * attention-weight cache is quadratic in that length, so using
       * max_timestep here would waste
       * num_heads_Q * (max_timestep^2 - seq_len^2) * 4 bytes per layer.
       */
      const unsigned int train_seq_len = query_dim.height();

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
      tensor_idx[AttentionParams::train_v] = context.requestTensor(
        k_roped_dim, "train_v", nntrainer::Initializer::NONE, false,
        nntrainer::TensorLifespan::ITERATION_LIFESPAN);
    }
  }

  theta = (float)std::get<props::RopeTheta>(mha_core_props).get();

  /** set Output dimension! - one output */
  std::vector<nntrainer::TensorDim> output_dims(1);
  output_dims[0] = input_dims[0];
  output_dims[0].width(head_dim * num_heads_Q);
  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});
  context.setOutputDimensions(output_dims);
}

/************************************************************** */

/**
 * @note In external KV cache mode (use_external_cache == true), this
 *       implements the inference forward pass using cache tensors supplied
 *       as input[3] (cache_key) and input[4] (cache_value). The host (e.g.
 *       KVCacheManager via setExternalTensors) is responsible for owning
 *       these buffers and for calling setCacheIndex() before each step to
 *       set the write position. After this call cache_index is advanced by
 *       input.height().
 *
 *       In legacy 3/4-input mode (use_external_cache == false) training is
 *       NYI and incremental_forwarding() is the inference path.
 *
 *       Input layout for external cache mode:
 *         input[0] = Q   (B, 1, step_size, num_heads_Q  * head_dim)
 *         input[1] = K   (B, 1, step_size, num_heads_KV * head_dim)
 *         input[2] = V   (B, 1, step_size, num_heads_KV * head_dim)
 *         input[3] = cache_key   (B, 1, max_seq_len, num_heads_KV * head_dim)
 *         input[4] = cache_value (B, 1, max_seq_len, num_heads_KV * head_dim)
 */
void MHACoreLayer::forwarding(nntrainer::RunLayerContext &context,
                              bool training) {
  if (!use_external_cache) {
    /**
     * @note Gated on the training tensors existing, NOT on the `training`
     * flag: validation also comes through forwarding() (with
     * training=false) and must still compute attention, otherwise it scores
     * a stale output buffer. The caches are only requested when the graph
     * was finalized for TRAIN, and inference never uses forwarding() at all
     * (it goes through incremental_forwarding), so this is the complete
     * condition.
     */
    if (train_tensors_requested) {
      trainForwarding(context);
    }
    return;
  }

  nntrainer::Tensor &query = context.getInput(INOUT_INDEX::QUERY);
  nntrainer::Tensor &key = context.getInput(INOUT_INDEX::KEY);
  nntrainer::Tensor &value = context.getInput(INOUT_INDEX::VALUE);
  nntrainer::Tensor &output = context.getOutput(INOUT_INDEX::OUTPUT);

  nntrainer::Tensor &cache_key = context.getInput(3);
  nntrainer::Tensor &cache_value = context.getInput(4);

  nntrainer::Tensor sink;
  if (use_sink) {
    sink = context.getWeight(sink_idx);
  }

  unsigned int step_size = (incremental_step_size > 0)
                             ? incremental_step_size
                             : (unsigned int)query.height();
  unsigned int from = cache_index;
  unsigned int to = cache_index + step_size;

  auto get_step_dim = [step_size](const ml::train::TensorDim &dim) {
    auto step_dim = dim;
    step_dim.batch(1);
    step_dim.height(step_size);
    return step_dim;
  };

  ml::train::TensorDim query_dim = query.getDim();
  ml::train::TensorDim key_dim = key.getDim();
  ml::train::TensorDim value_dim = value.getDim();
  ml::train::TensorDim output_dim = output.getDim();
  ml::train::TensorDim cache_key_dim = cache_key.getDim();
  ml::train::TensorDim cache_value_dim = cache_value.getDim();

  ml::train::TensorDim query_step_dim = get_step_dim(query_dim);
  ml::train::TensorDim key_step_dim = get_step_dim(key_dim);
  ml::train::TensorDim value_step_dim = get_step_dim(value_dim);
  ml::train::TensorDim output_step_dim = get_step_dim(output_dim);
  ml::train::TensorDim cache_key_step_dim = get_step_dim(cache_key_dim);
  ml::train::TensorDim cache_value_step_dim = get_step_dim(cache_value_dim);

  unsigned int batch_size = query_dim.batch();
  for (unsigned int batch = 0; batch < batch_size; ++batch) {
    nntrainer::Tensor query_step = query.getSharedDataTensor(
      query_step_dim, batch * query_dim.getFeatureLen(), true);
    nntrainer::Tensor key_step = key.getSharedDataTensor(
      key_step_dim, batch * key_dim.getFeatureLen(), true);
    nntrainer::Tensor value_step = value.getSharedDataTensor(
      value_step_dim, batch * value_dim.getFeatureLen(), true);
    nntrainer::Tensor output_step = output.getSharedDataTensor(
      output_step_dim, batch * output_dim.getFeatureLen(), true);

    if (query_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
#if ENABLE_FP16 && defined(__ANDROID__)
      nntrainer::TensorDim Q_step_dim = query_step_dim;
      nntrainer::TensorDim K_step_dim = key_step_dim;
      nntrainer::TensorDim V_step_dim = value_step_dim;
      nntrainer::TensorDim O_step_dim = output_step_dim;
      Q_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);
      K_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);
      V_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);
      O_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);

      nntrainer::Tensor Q_step = nntrainer::Tensor(Q_step_dim, true);
      nntrainer::Tensor K_step = nntrainer::Tensor(K_step_dim, true);
      nntrainer::Tensor V_step = nntrainer::Tensor(V_step_dim, true);
      nntrainer::Tensor O_step = nntrainer::Tensor(O_step_dim, true);

      Q_step.copyData(query_step);
      K_step.copyData(key_step);
      V_step.copyData(value_step);

      if (use_sink) {
        one_batch_incremental_forwarding(
          batch, from, from, to, Q_step, K_step, V_step, O_step, cache_key,
          cache_value, cache_key_dim, cache_key_step_dim, cache_value_dim,
          cache_value_step_dim, sink);
      } else {
        one_batch_incremental_forwarding(batch, from, from, to, Q_step, K_step,
                                         V_step, O_step, cache_key, cache_value,
                                         cache_key_dim, cache_key_step_dim,
                                         cache_value_dim, cache_value_step_dim);
      }
      output_step.copyData(O_step);
#else
      if (use_sink) {
        one_batch_incremental_forwarding(
          batch, from, from, to, query_step, key_step, value_step, output_step,
          cache_key, cache_value, cache_key_dim, cache_key_step_dim,
          cache_value_dim, cache_value_step_dim, sink);
      } else {
        one_batch_incremental_forwarding(
          batch, from, from, to, query_step, key_step, value_step, output_step,
          cache_key, cache_value, cache_key_dim, cache_key_step_dim,
          cache_value_dim, cache_value_step_dim);
      }
#endif
    } else {
      one_batch_incremental_forwarding(
        batch, from, from, to, query_step, key_step, value_step, output_step,
        cache_key, cache_value, cache_key_dim, cache_key_step_dim,
        cache_value_dim, cache_value_step_dim);
    }
  }

  cache_index += step_size;
}

/**
 * @note This incremental_forwarding method is invoked for inference mode.
 *       Please note that Transformer Decoder's MHA takes only one sequence at a
 * step. Incremental forwarding function is used for this.
 */
void MHACoreLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                          unsigned int _from, unsigned int _to,
                                          bool training) {
  // External KV cache path: from/to are interpreted as the absolute write
  // position; route through forwarding() which reads cache_key/cache_value
  // from input slots 3/4. forwarding() advances cache_index internally.
  if (use_external_cache) {
    cache_index = _from;
    incremental_step_size = _to - _from;
    forwarding(context, training);
    incremental_step_size = 0;
    return;
  }

  /// @todo replace step_size into input height
  unsigned int step_size = _to - _from;

  unsigned int max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();

  unsigned int from = _from;
  unsigned int to = _to;

  if (to > max_timestep) {
    // initial forwarding
    if (!_from) {
      throw std::invalid_argument(
        "to shouldn't greater than max_timestep for initial forwarding");
    } else {
      throw std::runtime_error("NYI: cache shift is not available");
      // exceeds the kv_cache size
      // KV_cache is shifted!
      cache_shift = true;
      from = max_timestep - 1;
      to = max_timestep;
    }
  }

  // util fn to compute tensor dimension for one step.
  auto get_step_dim = [step_size](const ml::train::TensorDim &dim) {
    auto step_dim = dim;
    step_dim.batch(1);
    step_dim.height(step_size);
    return step_dim;
  };

  /** incremental forwarding for each batch */
  nntrainer::Tensor &query =
    context.getInput(INOUT_INDEX::QUERY); // projected query
  nntrainer::Tensor &key = context.getInput(INOUT_INDEX::KEY); // projected key
  nntrainer::Tensor &value =
    context.getInput(INOUT_INDEX::VALUE); // projected value
  nntrainer::Tensor &output =
    context.getOutput(INOUT_INDEX::OUTPUT); // output to be projected

  nntrainer::Tensor &cache_key =
    context.getTensor(tensor_idx[AttentionParams::cache_key]);
  nntrainer::Tensor &cache_value =
    context.getTensor(tensor_idx[AttentionParams::cache_value]);

  nntrainer::Tensor sink;
  if (use_sink) {
    sink = context.getWeight(sink_idx);
  }

  ml::train::TensorDim query_dim =
    query.getDim(); // (B, 1, seq_len, n_heads_Q * head_dim)
  ml::train::TensorDim key_dim =
    key.getDim(); // (B, 1, seq_len, n_heads_KV * head_dim)
  ml::train::TensorDim value_dim =
    value.getDim(); // (B, 1, seq_len, n_heads_KV * head_dim)
  ml::train::TensorDim output_dim =
    output.getDim(); // (B, 1, seq_len, n_heads_Q * head_dim)
  ml::train::TensorDim cache_key_dim =
    cache_key.getDim(); // (B, 1, max_timestep, n_heads_KV * head_dim)
  ml::train::TensorDim cache_value_dim =
    cache_value.getDim(); // (B, 1, max_timestep, n_heads_KV * head_dim)

  ml::train::TensorDim query_step_dim =
    get_step_dim(query_dim); // (1, 1, step_size, n_heads_Q * head_dim)
  ml::train::TensorDim key_step_dim = get_step_dim(key_dim);
  ml::train::TensorDim value_step_dim = get_step_dim(value_dim);
  ml::train::TensorDim output_step_dim =
    get_step_dim(output_dim); // (1, 1, step_size, n_heads_Q * head_dim)
  ml::train::TensorDim cache_key_step_dim =
    get_step_dim(cache_key_dim); // (1, 1, step_size, n_heads_KV * head_dim)

  ml::train::TensorDim cache_value_step_dim =
    get_step_dim(cache_value_dim); // (1, 1, step_size, n_heads_KV * head_dim)

  unsigned int batch_size = query_dim.batch();
  // do the incremental forwarding
  for (unsigned int batch = 0; batch < batch_size; ++batch) {

    // preparing step tensors
    nntrainer::Tensor query_step = query.getSharedDataTensor(
      query_step_dim, batch * query_dim.getFeatureLen(), true);
    nntrainer::Tensor key_step = key.getSharedDataTensor(
      key_step_dim, batch * key_dim.getFeatureLen(), true);
    nntrainer::Tensor value_step = value.getSharedDataTensor(
      value_step_dim, batch * value_dim.getFeatureLen(), true);
    nntrainer::Tensor output_step = output.getSharedDataTensor(
      output_step_dim, batch * output_dim.getFeatureLen(), true);

    if (query_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
#if ENABLE_FP16 && defined(__ANDROID__)
      nntrainer::TensorDim Q_step_dim = query_step_dim;
      nntrainer::TensorDim K_step_dim = key_step_dim;
      nntrainer::TensorDim V_step_dim = value_step_dim;
      nntrainer::TensorDim O_step_dim = output_step_dim;
      Q_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);
      K_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);
      V_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);
      O_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);

      nntrainer::Tensor Q_step = nntrainer::Tensor(Q_step_dim, true);
      nntrainer::Tensor K_step = nntrainer::Tensor(K_step_dim, true);
      nntrainer::Tensor V_step = nntrainer::Tensor(V_step_dim, true);
      nntrainer::Tensor O_step = nntrainer::Tensor(O_step_dim, true);

      Q_step.copyData(query_step);
      K_step.copyData(key_step);
      V_step.copyData(value_step);
      if (use_sink) {
        one_batch_incremental_forwarding(
          batch, _from, from, to, Q_step, K_step, V_step, O_step, cache_key,
          cache_value, cache_key_dim, cache_key_step_dim, cache_value_dim,
          cache_value_step_dim, sink);
      } else {
        one_batch_incremental_forwarding(batch, _from, from, to, Q_step, K_step,
                                         V_step, O_step, cache_key, cache_value,
                                         cache_key_dim, cache_key_step_dim,
                                         cache_value_dim, cache_value_step_dim);
      }
      output_step.copyData(O_step);
#else
      if (use_sink) {
        one_batch_incremental_forwarding(
          batch, _from, from, to, query_step, key_step, value_step, output_step,
          cache_key, cache_value, cache_key_dim, cache_key_step_dim,
          cache_value_dim, cache_value_step_dim, sink);
      } else {
        one_batch_incremental_forwarding(
          batch, _from, from, to, query_step, key_step, value_step, output_step,
          cache_key, cache_value, cache_key_dim, cache_key_step_dim,
          cache_value_dim, cache_value_step_dim);
      }
#endif
    } else {
      one_batch_incremental_forwarding(
        batch, _from, from, to, query_step, key_step, value_step, output_step,
        cache_key, cache_value, cache_key_dim, cache_key_step_dim,
        cache_value_dim, cache_value_step_dim);
    }
  }

  // increase cache size
  cache_index += step_size;
}

/**
 * @brief Function to compute Attention Scores using Tensor inputs. Wrapper
 * around nntrainer::compute_kcaches with multi-threading support
 *
 * Expected Input Shapes:
 * @param in (Query): [Batch, 1, sequence_len, Num_Heads_Q * Head_Dim]
 * @param cache (Key Cache): [Batch, 1, Max_Timestep, Num_Heads_KV * Head_Dim]
 * @param out (Attention Score): [Batch, 1, 1, Num_Heads_Q * Context_Len]
 *            where Context_Len is usually the current timestep 'to'.
 *
 */
void MHACoreLayer::compute_kcaches(nntrainer::Tensor &in,
                                   nntrainer::Tensor &cache,
                                   nntrainer::Tensor &out, unsigned int from,
                                   size_t sequence_len, unsigned int num_head,
                                   unsigned int group_size,
                                   unsigned int head_dim) {

  // Dispatch based on data type (FP32 or FP16)
  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    if (sequence_len == 1) {
      // Single token processing (common during generation)
      // Parallelize over KV heads for decoding since Q direction is always 1
      int row_to_compute = is_causal ? from + 1 : from + sequence_len;
      unsigned int num_cache_head = num_head / group_size;

      // Use ThreadManager for lower overhead parallelization during decoding
      const float *in_data = in.getData<float>();
      float *out_data = out.getData<float>();

      auto &tm = nntrainer::ThreadManager::Global();
      if (cache.getDataType() == ml::train::TensorDim::DataType::FP32) {
        const float *cache_data = cache.getData<float>();
        tm.parallel_for(
          0, static_cast<size_t>(num_cache_head), [=](size_t head_kv) {
            compute_kcaches_fp32_reference(
              in_data, cache_data, out_data, row_to_compute, num_cache_head,
              head_dim, group_size, local_window_size, head_kv, head_kv + 1);
          });
      } else {
        const uint16_t *cache_data = cache.getData<uint16_t>();
        tm.parallel_for(0, static_cast<size_t>(num_cache_head),
                        [=](size_t head_kv) {
                          nntrainer::compute_kcaches<uint16_t>(
                            in_data, cache_data, out_data, row_to_compute,
                            num_cache_head, head_dim, group_size, tile_size,
                            local_window_size, head_kv, head_kv + 1);
                        });
      }

    } else {
      // Sequence processing (prefill or chunked)
      // Iterate over ALL query rows so that no row is skipped even when
      // sequence_len > local_window_size.
      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, static_cast<size_t>(sequence_len), [=](size_t i) {
        float *input_addr = in.getData<float>() + num_head * head_dim * i;
        int row_to_compute = is_causal ? from + i + 1 : from + sequence_len;
        // Windowed cumulative offset so that each row's scores are placed
        // contiguously after the previous row's scores (respecting the window).
        size_t out_start_row = is_causal ? calc_windowed_attn_index(from + i) -
                                             calc_windowed_attn_index(from)
                                         : i * (from + sequence_len);
        float *output_addr = out.getData<float>() + out_start_row * num_head;

        if (cache.getDataType() == ml::train::TensorDim::DataType::FP32) {
          float *cache_addr = cache.getData<float>();
          compute_kcaches_fp32_reference(
            input_addr, cache_addr, output_addr, row_to_compute,
            num_head / group_size, head_dim, group_size, local_window_size);
        } else {
          uint16_t *cache_addr = cache.getData<uint16_t>();
          nntrainer::compute_kcaches<uint16_t>(
            input_addr, cache_addr, output_addr, row_to_compute,
            num_head / group_size, head_dim, group_size, tile_size,
            local_window_size);
        }
      });
    }
  } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    if (sequence_len == 1) {
      // Single token processing (common during generation)
      // Parallelize over KV heads for decoding since Q direction is always 1
      int num_rows = is_causal ? from + 1 : from + sequence_len;
      unsigned int num_cache_head = num_head / group_size;

      // Use ThreadManager for lower overhead parallelization during decoding
      const _FP16 *in_data = in.getData<_FP16>();
      const _FP16 *cache_data = cache.getData<_FP16>();
      _FP16 *out_data = out.getData<_FP16>();

      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(
        0, static_cast<size_t>(num_cache_head), [=](size_t head_kv) {
          nntrainer::compute_kcaches(
            in_data, cache_data, out_data, num_rows, num_cache_head, head_dim,
            group_size, tile_size, local_window_size, head_kv, head_kv + 1);
        });
    } else {
      // Iterate over ALL query rows so that no row is skipped even when
      // sequence_len > local_window_size.
      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, static_cast<size_t>(sequence_len), [=](size_t i) {
        _FP16 *input_addr = in.getData<_FP16>() + num_head * head_dim * i;
        _FP16 *cache_addr = cache.getData<_FP16>();
        int row_to_compute = is_causal ? from + i + 1 : from + sequence_len;
        // Windowed cumulative offset so that each row's scores are placed
        // contiguously after the previous row's scores (respecting the window).
        size_t out_start_row = is_causal ? calc_windowed_attn_index(from + i) -
                                             calc_windowed_attn_index(from)
                                         : i * (from + sequence_len);

        _FP16 *output_addr = out.getData<_FP16>() + out_start_row * num_head;

        nntrainer::compute_kcaches(input_addr, cache_addr, output_addr,
                                   row_to_compute, num_head / group_size,
                                   head_dim, group_size, tile_size,
                                   local_window_size);
      });
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

void MHACoreLayer::one_batch_incremental_forwarding(
  const unsigned int batch, const unsigned int _from, const unsigned int from,
  const unsigned int to, nntrainer::Tensor &query_step,
  nntrainer::Tensor &key_step, nntrainer::Tensor &value_step,
  nntrainer::Tensor &attention_output_step, nntrainer::Tensor &cache_key,
  nntrainer::Tensor &cache_value, ml::train::TensorDim &cache_key_dim,
  ml::train::TensorDim &cache_key_step_dim,
  ml::train::TensorDim &cache_value_dim,
  ml::train::TensorDim &cache_value_step_dim) {

  /**
   *
   *  cache_key
   *  +------------------------------------------+
   *  |<--cache_index-->|<--b_cache_value_step-->|
   *  +------------------------------------------+
   *                    |<-------key_step------->|
   *  |<-------------b_cached_key--------------->|
   */

  // Load Input Tensors of this batch : b_ denotes a Tensor for this batch
  nntrainer::Tensor b_cache_key_step = cache_key.getSharedDataTensor(
    cache_key_step_dim,
    batch * cache_key_dim.getFeatureLen() + cache_index * cache_key_dim.width(),
    true);
  nntrainer::Tensor b_cache_value_step =
    cache_value.getSharedDataTensor(cache_value_step_dim,
                                    batch * cache_value_dim.getFeatureLen() +
                                      cache_index * cache_value_dim.width(),
                                    true);

  // append kcache with or without rotary embedding
  apply_rotary_emb_tensor_v2(key_step, b_cache_key_step, head_dim, cache_index,
                             !use_rope);

  // append vcache without rotary embedding
  if (query_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
    apply_rotary_emb_tensor_v2(value_step, b_cache_value_step, head_dim,
                               cache_index, true);
  } else if (query_step.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    b_cache_value_step.copyData(value_step);
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }

  unsigned int step_size = to - from;
  bool is_prefill = !from || step_size > 1;
  if (skip_prefill && is_prefill)
    return;

  // apply rotary embedding for query
  if (use_rope) {
    apply_rotary_emb_tensor_v2(query_step, query_step, head_dim, cache_index,
                               false);
  }

  /// @todo replace step_size into input height
  unsigned int cache_from = cache_index;
  unsigned int cache_to = cache_from + step_size;

  // batch(1): this is a single batch's windowed view into the cache, not
  // the whole cache - without resetting it, the requested dim keeps
  // cache_key_dim's full batch count while the offset only accounts for
  // one batch slot, so any batch > 0 reads/overflows into other batches'
  // cache entries.
  ml::train::TensorDim cached_key_dim = cache_key_dim;
  ml::train::TensorDim cached_value_dim = cache_value_dim;
  cached_key_dim.batch(1);
  cached_value_dim.batch(1);
  cached_key_dim.height(cache_to);
  cached_value_dim.height(cache_to);

  nntrainer::Tensor b_cached_key = cache_key.getSharedDataTensor(
    cached_key_dim, batch * cache_key_dim.getFeatureLen(), true);
  nntrainer::Tensor b_cached_value = cache_value.getSharedDataTensor(
    cached_value_dim, batch * cache_value_dim.getFeatureLen(), true);

  // out_ stores the output of Q * K
  nntrainer::Tensor out_(1, 1,
                         is_causal ? (calc_windowed_attn_index(cache_to) -
                                      calc_windowed_attn_index(cache_from))
                                   : (step_size * cache_to),
                         num_heads_Q, query_step.getTensorType());

  unsigned int gqa_size = num_heads_Q / num_heads_KV;

  // Try flash attention dispatch (DSP offload) for prefill
  bool use_flash = should_use_flash_attn(step_size, head_dim, is_prefill);

  if (use_flash) {
    const std::vector<uint16_t> &mask =
      get_cached_causal_mask(step_size, cache_to, cache_from);

    bool q_is_fp16 =
      (query_step.getDataType() == ml::train::TensorDim::DataType::FP16);
    bool out_is_fp16 = (attention_output_step.getDataType() ==
                        ml::train::TensorDim::DataType::FP16);

    const flash_attn_fn &fn = get_flash_attn_bridge();
    int rc =
      fn(query_step.getData(), b_cached_key.getData(), b_cached_value.getData(),
         mask.data(), attention_output_step.getData(), step_size, num_heads_Q,
         num_heads_KV, head_dim, cache_to, 1.0f / sqrtf((float)head_dim),
         q_is_fp16, out_is_fp16);

    if (rc != 0) {
      ml_logw("MHACore: flash_attn returned %d, falling back to CPU", rc);
      use_flash = false;
    }
  }

  if (!use_flash) {
    // Original CPU path
    compute_kcaches(query_step, b_cached_key, out_, cache_from,
                    cache_to - cache_from, num_heads_Q, gqa_size, head_dim);

    softmax_triangle(out_, step_size, num_heads_Q, cache_from);

    compute_fp16vcache_transposed(out_, b_cached_value, attention_output_step,
                                  cache_from, num_heads_KV, gqa_size, head_dim,
                                  cache_to);
  }
}

void MHACoreLayer::one_batch_incremental_forwarding(
  const unsigned int batch, const unsigned int _from, const unsigned int from,
  const unsigned int to, nntrainer::Tensor &query_step,
  nntrainer::Tensor &key_step, nntrainer::Tensor &value_step,
  nntrainer::Tensor &attention_output_step, nntrainer::Tensor &cache_key,
  nntrainer::Tensor &cache_value, ml::train::TensorDim &cache_key_dim,
  ml::train::TensorDim &cache_key_step_dim,
  ml::train::TensorDim &cache_value_dim,
  ml::train::TensorDim &cache_value_step_dim, nntrainer::Tensor &sink_step) {
  /// @todo replace from, to into cache_index, input height
  /// @note currently, only gpt-oss uses this method

  /**
   *  cache_key
   *  +--------+                        ->
   *  |        |                        ->
   *  |        |                        ->
   *  |........| from                   ->
   *  |........| to -> b_cache_key_step -> b_cached_key
   *  |        |
   *  +--------+
   *
   */

  /** 1. Load Input Tensors of this batch : b_ denotes a Tensor for this batch
   * **/
  nntrainer::Tensor b_cache_key_step = cache_key.getSharedDataTensor(
    cache_key_step_dim,
    batch * cache_key_dim.getFeatureLen() + from * cache_key_dim.width(), true);
  nntrainer::Tensor b_cache_value_step = cache_value.getSharedDataTensor(
    cache_value_step_dim,
    batch * cache_value_dim.getFeatureLen() + from * cache_value_dim.width(),
    true);

  if (use_rope) {
    apply_rotary_emb_tensor_v2(query_step, query_step, head_dim, _from, false);
  }

  apply_rotary_emb_tensor_v2(key_step, b_cache_key_step, head_dim, _from,
                             !use_rope);

  if (query_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
    apply_rotary_emb_tensor_v2(value_step, b_cache_value_step, head_dim, _from,
                               true);
  } else if (query_step.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    b_cache_value_step.copyData(value_step);
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }

  // batch(1): see the non-sink overload above for why this is required.
  ml::train::TensorDim cached_key_dim = cache_key_dim;
  ml::train::TensorDim cached_value_dim = cache_value_dim;
  cached_key_dim.batch(1);
  cached_value_dim.batch(1);
  cached_key_dim.height(to);
  cached_value_dim.height(to);

  nntrainer::Tensor b_cached_key = cache_key.getSharedDataTensor(
    cached_key_dim, batch * cache_key_dim.getFeatureLen(), true);
  nntrainer::Tensor b_cached_value = cache_value.getSharedDataTensor(
    cached_value_dim, batch * cache_value_dim.getFeatureLen(), true);

  nntrainer::Tensor out_(1, 1,
                         is_causal ? (((to - from) == 1)
                                        ? to
                                        : calc_windowed_attn_index(to) -
                                            calc_windowed_attn_index(from))
                                   : ((to - from) * to),
                         num_heads_Q, query_step.getTensorType());

  unsigned int gqa_size = num_heads_Q / num_heads_KV;

  compute_kcaches(query_step, b_cached_key, out_, _from, to - from, num_heads_Q,
                  gqa_size, head_dim);

  softmax_triangle(out_, to - from, num_heads_Q, from, sink_step);

  compute_fp16vcache_transposed(out_, b_cached_value, attention_output_step,
                                from, num_heads_KV, gqa_size, head_dim, to);
}

/************************************************************** */

/**
 * @brief rotary embedding-related member function
 * @note seq_len -> max_position_embeddings
 */
void MHACoreLayer::precompute_freqs(int head_dim, unsigned int seq_len,
                                    float theta, bool is_fp16) {
  const std::string rope_cache_key = getRopeCacheKey(head_dim, seq_len, theta);
  thetas.clear();
  if (rope_scaling_type == "default")
    _compute_default_parameters(head_dim, theta);
  else if (rope_scaling_type == "yarn")
    _compute_yarn_parameters(head_dim, theta);
  else if (rope_scaling_type == "proportional")
    _compute_proportional_parameters(head_dim, theta);
  else
    NNTR_THROW_IF(true, std::invalid_argument) << "Unsupported rope type!";

  unsigned int half_ = head_dim / 2;

  if (!is_fp16) {
    auto it = rope_cache_fp32.find(rope_cache_key);
    if (it != rope_cache_fp32.end()) {
      freqs_fp32 = it->second;
      return;
    }

    auto cached = std::make_shared<RopeCacheFP32>();
    cached->cos.assign(seq_len, std::vector<float>(head_dim, 0));
    cached->sin.assign(seq_len, std::vector<float>(head_dim, 0));

    for (unsigned int i = 0; i < seq_len; ++i) {
      nntrainer::calc_trigonometric_vals_dup(
        half_, thetas.data(), cached->cos[i].data(), cached->sin[i].data(), i,
        attention_scaling);
    }
    rope_cache_fp32[rope_cache_key] = cached;
    freqs_fp32 = cached;
  }

#ifdef ENABLE_FP16
  if (is_fp16) {
    auto it = rope_cache_fp16.find(rope_cache_key);
    if (it != rope_cache_fp16.end()) {
      freqs_fp16 = it->second;
      return;
    }

    auto cached = std::make_shared<RopeCacheFP16>();
    cached->cos.assign(seq_len, std::vector<_FP16>(head_dim, 0));
    cached->sin.assign(seq_len, std::vector<_FP16>(head_dim, 0));

    std::vector<float> cos_tmp(head_dim);
    std::vector<float> sin_tmp(head_dim);

    for (unsigned int i = 0; i < seq_len; ++i) {
      nntrainer::calc_trigonometric_vals_dup(half_, thetas.data(),
                                             cos_tmp.data(), sin_tmp.data(), i,
                                             attention_scaling);
      for (unsigned int j = 0; j < head_dim; ++j) {
        cached->cos[i][j] = (_FP16)cos_tmp[j];
        cached->sin[i][j] = (_FP16)sin_tmp[j];
      }
    }
    rope_cache_fp16[rope_cache_key] = cached;
    freqs_fp16 = cached;
  }
#endif
}

std::string MHACoreLayer::getRopeCacheKey(int head_dim, unsigned int seq_len,
                                          float theta) const {
  std::ostringstream ss;
  ss << rope_scaling_type << "|" << head_dim << "|" << seq_len << "|" << theta
     << "|" << scale << "|" << rope_partial_rotary_factor << "|"
     << original_max_position_embeddings;
  return ss.str();
}

void MHACoreLayer::_compute_default_parameters(int head_dim, float theta) {

  // no attention scaling
  attention_scaling = 1.0f;

  // theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... , dim/2]
  // head_dim should be divisible by 2
  unsigned int half_ = head_dim / 2;
  for (unsigned int i = 0; i < half_; ++i) {
    thetas.push_back(1.0 /
                     (std::pow(theta, (2 * i) / static_cast<float>(head_dim))));
  }
}

void MHACoreLayer::_compute_proportional_parameters(int head_dim, float theta) {
  attention_scaling = 1.0f;
  const int half_dim = static_cast<int>(head_dim / 2);
  const int rope_angles =
    static_cast<int>((rope_partial_rotary_factor * head_dim) / 2.0f);

  thetas.reserve(half_dim);
  for (int i = 0; i < rope_angles; ++i) {
    thetas.push_back(1.0f /
                     (std::pow(theta, (2 * i) / static_cast<float>(head_dim))));
  }

  for (int i = rope_angles; i < half_dim; ++i) {
    thetas.push_back(0.0f);
  }

  for (auto &val : thetas) {
    val /= scale;
  }
}

void MHACoreLayer::_compute_yarn_parameters(int head_dim, float theta) {

  // Config parameters
  ///@todo partial_rotary_factor should be generalized to fully support
  /// transformers's implementation
  // const float partial_rotary_factor = has_partial_rotary_factor ?
  // config_partial_rotary_factor : 1.0f;
  const float partial_rotary_factor = 1.0f;
  const int dim = static_cast<int>(head_dim * partial_rotary_factor);
  const float base = theta;

  // Handle max position embeddings

  // Attention scaling calculation (simplified from Python version)
  auto get_mscale = [](float scale, float mscale = 1.0f) {
    return (scale <= 1.0f) ? 1.0f : (0.1f * mscale * std::log(scale) + 1.0f);
  };

  ///@todo attention_scaling should be generalized to fully support
  /// transformers's implementation
  // if (has_mscale && has_mscale_all_dim) {
  // attention_scaling = get_mscale(factor, mscale) / get_mscale(factor,
  // mscale_all_dim);
  // } else {
  // attention_scaling = get_mscale(factor);
  // }
  attention_scaling = get_mscale(scale);

  ///@todo attention_scaling should be generalized to fully support
  /// transformers's implementation
  // const float beta_fast = has_beta_fast ? config_beta_fast : 32.0f;
  // const float beta_slow = has_beta_slow ? config_beta_slow : 1.0f;
  // const bool truncate = has_truncate ? config_truncate : true;
  // Beta parameters
  const float beta_fast = 32.0f;
  const float beta_slow = 1.0f;
  const bool truncate = false;

  // Helper functions
  auto find_correction_dim = [&](float num_rotations) {
    return (dim * std::log(original_max_position_embeddings /
                           (num_rotations * 2 * M_PI))) /
           (2 * std::log(base));
  };

  auto [low, high] = [&]() {
    float low_val = find_correction_dim(beta_fast);
    float high_val = find_correction_dim(beta_slow);
    if (truncate) {
      low_val = std::floor(low_val);
      high_val = std::ceil(high_val);
    }
    return std::make_pair(low_val, high_val);
  }();

  // Compute position frequencies
  thetas.resize(dim / 2);

  // Compute interpolation and extrapolation frequencies
  std::vector<float> inv_freq_interpolation;
  std::vector<float> inv_freq_extrapolation;
  for (size_t i = 0; i < dim / 2; ++i) {
    inv_freq_extrapolation.push_back(
      1.0 / (std::pow(theta, (2 * i) / static_cast<float>(head_dim))));
    inv_freq_interpolation.push_back(
      1.0 / (scale * std::pow(theta, (2 * i) / static_cast<float>(head_dim))));
  }

  auto linear_ramp_factor = [](float min, float max, int size) {
    if (min == max) {
      max += 0.001f; // Prevent singularity
    }
    std::vector<float> ramp(size);
    for (int i = 0; i < size; ++i) {
      float val = (i - min) / (max - min);
      ramp[i] = std::clamp(val, 0.0f, 1.0f);
    }
    return ramp;
  };

  std::vector<float> inv_freq_extrapolation_factor =
    linear_ramp_factor(low, high, dim / 2);
  for (auto &val : inv_freq_extrapolation_factor) {
    val = 1.0f - val;
  }

  // Combine frequencies
  for (size_t i = 0; i < thetas.size(); ++i) {
    thetas[i] =
      inv_freq_extrapolation[i] * inv_freq_extrapolation_factor[i] +
      inv_freq_interpolation[i] * (1.0f - inv_freq_extrapolation_factor[i]);
  }
}

void MHACoreLayer::apply_rotary_emb_tensor_v2(nntrainer::Tensor &in,
                                              nntrainer::Tensor &out,
                                              unsigned int dim,
                                              unsigned int from,
                                              bool convert_only) {
  if (!use_rope) {
    if (&in != &out) {
      out.copyData(in);
    }
    return;
  }
  unsigned int half_ = dim / 2;
  unsigned int max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();

  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    if (freqs_fp32 == nullptr) {
      const std::lock_guard<std::mutex> lock(rope_init_mtx);
      if (freqs_fp32 == nullptr) {
        precompute_freqs(head_dim, max_position_embeddings, theta, false);
      }
    }
    std::vector<float> *cos_ = nullptr;
    std::vector<float> *sin_ = nullptr;

    for (unsigned int b = 0; b < in.batch(); b++) {
      for (unsigned int c = 0; c < in.channel(); c++) {
        for (unsigned int h = 0; h < in.height(); h++) {
          if (from < max_timestep) {
            cos_ = &freqs_fp32->cos[from + h];
            sin_ = &freqs_fp32->sin[from + h];
          }
          float *in_ptr = in.getData<float>() +
                          b * in.channel() * in.height() * in.width() +
                          c * in.height() * in.width() + h * in.width();

          if (out.getDataType() == ml::train::TensorDim::DataType::FP32) {
            float *out_ptr = out.getData<float>() +
                             b * out.channel() * out.height() * out.width() +
                             c * out.height() * out.width() + h * out.width();

            if (out_ptr != in_ptr) {
              std::memcpy(out_ptr, in_ptr, sizeof(float) * in.width());
            }
            if (!convert_only) {
              nntrainer::compute_rotary_emb_value(
                in.width(), dim, half_, out_ptr, nullptr, cos_->data(),
                sin_->data(), false);
            }
          } else if (out.getDataType() ==
                       ml::train::TensorDim::DataType::UINT16 ||
                     out.getDataType() ==
                       ml::train::TensorDim::DataType::FP16) {
            uint16_t *out_ptr = out.getData<uint16_t>() +
                                b * out.channel() * out.height() * out.width() +
                                c * out.height() * out.width() +
                                h * out.width();

            nntrainer::compute_rotary_emb_value(in.width(), dim, half_, in_ptr,
                                                out_ptr, cos_->data(),
                                                sin_->data(), convert_only);
          }
        }
      }
    }
  } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    if (freqs_fp16 == nullptr) {
      const std::lock_guard<std::mutex> lock(rope_init_mtx);
      if (freqs_fp16 == nullptr) {
        precompute_freqs(head_dim, max_position_embeddings, theta, true);
      }
    }
    std::vector<_FP16> *cos_ = nullptr;
    std::vector<_FP16> *sin_ = nullptr;

    for (unsigned int b = 0; b < in.batch(); b++) {
      for (unsigned int c = 0; c < in.channel(); c++) {
        for (unsigned int h = 0; h < in.height(); h++) {
          if (from < max_timestep) {
            cos_ = &freqs_fp16->cos[from + h];
            sin_ = &freqs_fp16->sin[from + h];
          }
          _FP16 *in_ptr = in.getData<_FP16>() +
                          b * in.channel() * in.height() * in.width() +
                          c * in.height() * in.width() + h * in.width();
          _FP16 *out_ptr = out.getData<_FP16>() +
                           b * out.channel() * out.height() * out.width() +
                           c * out.height() * out.width() + h * out.width();

          nntrainer::compute_rotary_emb_value(in.width(), dim, half_, in_ptr,
                                              out_ptr, cos_->data(),
                                              sin_->data());
        }
      }
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

void MHACoreLayer::softmax_triangle(nntrainer::Tensor &qk_out, size_t row,
                                    size_t num_head, unsigned int from) {
  if (qk_out.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *qk_out_ = qk_out.getData<float>();

    if (attn_logit_softcapping > 0.0f) {
      size_t len =
        qk_out.batch() * qk_out.height() * qk_out.width() * qk_out.channel();
      float inv_softcapping = 1.0f / attn_logit_softcapping;
      for (size_t i = 0; i < len; ++i) {
        qk_out_[i] =
          std::tanh(qk_out_[i] * inv_softcapping) * attn_logit_softcapping;
      }
    }

    if (row == 1) {
      size_t start_row = 0;
      size_t end_row = 0;
      if (is_causal) {
        end_row = from < local_window_size ? from + 1 : local_window_size;
      } else {
        end_row = from + row; // end_row = to
      }
      nntrainer::softmax_row_inplace(qk_out_, start_row, end_row, num_head);
    } else {
      // Iterate over ALL rows (not just min(row, window)) so that every query
      // row in a long prefill gets softmaxed over the correct windowed range.
      size_t total_rows = row;
      if (!is_causal)
        total_rows = row;

      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, total_rows, [=](size_t i) {
        size_t start_row, end_row;
        if (is_causal) {
          start_row =
            calc_windowed_attn_index(from + i) - calc_windowed_attn_index(from);
          end_row = calc_windowed_attn_index(from + i + 1) -
                    calc_windowed_attn_index(from);
        } else {
          unsigned int to = from + row;
          start_row = i * to;
          end_row = (i + 1) * to;
        }
        nntrainer::softmax_row(qk_out_, start_row, end_row, num_head);
      });
    }
  } else if (qk_out.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    _FP16 *qk_out_ = qk_out.getData<_FP16>();

    if (attn_logit_softcapping > 0.0f) {
      size_t len =
        qk_out.batch() * qk_out.height() * qk_out.width() * qk_out.channel();
      float inv_softcapping = 1.0f / attn_logit_softcapping;
      for (size_t i = 0; i < len; ++i) {
        qk_out_[i] = (_FP16)(std::tanh((float)qk_out_[i] * inv_softcapping) *
                             attn_logit_softcapping);
      }
    }

    if (row == 1) {
      size_t start_row = 0;
      size_t end_row = 0;
      if (is_causal) {
        end_row = from < local_window_size ? from + 1 : local_window_size;
      } else {
        end_row = from + row; // end_row = to
      }
      nntrainer::softmax_row_inplace(qk_out_, start_row, end_row, num_head);
    } else {
      // Iterate over ALL rows (not just min(row, window)) so that every query
      // row in a long prefill gets softmaxed over the correct windowed range.
      size_t total_rows = row;
      if (!is_causal)
        total_rows = row;

      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, total_rows, [=](size_t i) {
        size_t start_row, end_row;
        if (is_causal) {
          start_row =
            calc_windowed_attn_index(from + i) - calc_windowed_attn_index(from);
          end_row = calc_windowed_attn_index(from + i + 1) -
                    calc_windowed_attn_index(from);
        } else {
          unsigned int to = from + row;
          start_row = i * to;
          end_row = (i + 1) * to;
        }
        nntrainer::softmax_row_inplace(qk_out_, start_row, end_row, num_head);
      });
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

void MHACoreLayer::softmax_triangle(nntrainer::Tensor &qk_out, size_t row,
                                    size_t num_head, unsigned int from,
                                    nntrainer::Tensor &sink_step) {
  if (qk_out.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *qk_out_ = qk_out.getData<float>();

    if (attn_logit_softcapping > 0.0f) {
      size_t len =
        qk_out.batch() * qk_out.height() * qk_out.width() * qk_out.channel();
      float inv_softcapping = 1.0f / attn_logit_softcapping;
      for (size_t i = 0; i < len; ++i) {
        qk_out_[i] =
          std::tanh(qk_out_[i] * inv_softcapping) * attn_logit_softcapping;
      }
    }

    if (row == 1) {
      size_t start_row = 0;
      size_t end_row = 0;
      if (is_causal) {
        end_row = from < local_window_size ? from + 1 : local_window_size;
      } else {
        unsigned int to = from + row;
        end_row = to;
      }
      nntrainer::softmax_row_inplace(qk_out_, start_row, end_row, num_head,
                                     sink_step.getData());
    } else {
      // Iterate over ALL rows (not just min(row, window)) for correct windowed
      // prefill when sequence_len > local_window_size.
      size_t total_rows = row;
      if (!is_causal)
        total_rows = row;

      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, total_rows, [=](size_t i) {
        size_t start_row, end_row;
        if (is_causal) {
          start_row =
            calc_windowed_attn_index(i + from) - calc_windowed_attn_index(from);
          end_row = calc_windowed_attn_index(from + i + 1) -
                    calc_windowed_attn_index(from);
        } else {
          unsigned int to = from + row;
          start_row = i * to;
          end_row = (i + 1) * to;
        }
        nntrainer::softmax_row(qk_out_, start_row, end_row, num_head,
                               sink_step.getData());
      });
    }
  } else if (qk_out.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    _FP16 *qk_out_ = qk_out.getData<_FP16>();
    _FP16 *sink_step_ = sink_step.getData<_FP16>();

    if (attn_logit_softcapping > 0.0f) {
      size_t len =
        qk_out.batch() * qk_out.height() * qk_out.width() * qk_out.channel();
      float inv_softcapping = 1.0f / attn_logit_softcapping;
      for (size_t i = 0; i < len; ++i) {
        qk_out_[i] = (_FP16)(std::tanh((float)qk_out_[i] * inv_softcapping) *
                             attn_logit_softcapping);
      }
    }

    if (row == 1) {
      size_t start_row = 0;
      size_t end_row = 0;
      if (is_causal) {
        end_row = from < local_window_size ? from + 1 : local_window_size;
      } else {
        end_row = from + row; // end_row = to
      }
      nntrainer::softmax_row_inplace(qk_out_, start_row, end_row, num_head,
                                     sink_step_);
    } else {
      // Iterate over ALL rows (not just min(row, window)) for correct windowed
      // prefill when sequence_len > local_window_size.
      size_t total_rows = row;
      if (!is_causal)
        total_rows = row;

      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, total_rows, [=](size_t i) {
        size_t start_row, end_row;
        if (is_causal) {
          start_row =
            calc_windowed_attn_index(i + from) - calc_windowed_attn_index(from);
          end_row = calc_windowed_attn_index(from + i + 1) -
                    calc_windowed_attn_index(from);
        } else {
          unsigned int to = from + row;
          start_row = i * to;
          end_row = (i + 1) * to;
        }
        nntrainer::softmax_row(qk_out_, start_row, end_row, num_head,
                               sink_step_);
      });
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

void MHACoreLayer::compute_fp16vcache_transposed(
  nntrainer::Tensor &in, nntrainer::Tensor &vcache, nntrainer::Tensor &output,
  int from, int num_cache_head, int gqa_size, int head_dim, int to) {

  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    if ((to - from) != 1) {
      // Iterate over ALL output rows so every query row gets an output even
      // when (to - from) > local_window_size.
      int total = to - from;
      if (!is_causal)
        total = to - from;

      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, static_cast<size_t>(total), [=](size_t i) {
        size_t start_idx;
        if (is_causal) {
          start_idx =
            calc_windowed_attn_index(from + i) - calc_windowed_attn_index(from);
        } else {
          start_idx = i * to; // linear index
        }
        const float *input =
          in.getData<float>() + start_idx * num_cache_head * gqa_size;
        float *out =
          output.getData<float>() + i * (num_cache_head * gqa_size * head_dim);

        int row_num = is_causal ? (from + (int)i) : to - 1;
        if (vcache.getDataType() == ml::train::TensorDim::DataType::FP32) {
          compute_vcache_fp32_transposed_reference(
            row_num, input, vcache.getData<float>(), out, num_cache_head,
            gqa_size, head_dim, local_window_size);
        } else {
          nntrainer::compute_fp16vcache_fp32_transposed(
            row_num, input, vcache.getData<uint16_t>(), out, num_cache_head,
            gqa_size, head_dim, local_window_size);
        }
      });
    } else {
      // Single token processing (common during generation)
      // Parallelize over KV heads for decoding since Q direction is always 1
      int row_num = to - 1;

      // Use OpenMP for lower overhead parallelization during decoding
      const float *in_data = in.getData<float>();
      float *output_data = output.getData<float>();

      auto &tm = nntrainer::ThreadManager::Global();
      if (vcache.getDataType() == ml::train::TensorDim::DataType::FP32) {
        const float *vcache_data = vcache.getData<float>();
        tm.parallel_for(
          0, static_cast<size_t>(num_cache_head), [=](size_t head_kv) {
            compute_vcache_fp32_transposed_reference(
              row_num, in_data, vcache_data, output_data, num_cache_head,
              gqa_size, head_dim, local_window_size, head_kv, head_kv + 1);
          });
      } else {
        const uint16_t *vcache_data = vcache.getData<uint16_t>();
        tm.parallel_for(
          0, static_cast<size_t>(num_cache_head), [=](size_t head_kv) {
            nntrainer::compute_fp16vcache_fp32_transposed(
              row_num, in_data, vcache_data, output_data, num_cache_head,
              gqa_size, head_dim, local_window_size, head_kv, head_kv + 1);
          });
      }
    }
  } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    if ((to - from) != 1) {
      // Iterate over ALL output rows so every query row gets an output even
      // when (to - from) > local_window_size.
      int total = to - from;
      if (!is_causal)
        total = to - from;

      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, static_cast<size_t>(total), [=](size_t i) {
        size_t start_idx;
        if (is_causal) {
          start_idx =
            calc_windowed_attn_index(from + i) - calc_windowed_attn_index(from);
        } else {
          start_idx = i * to;
        }
        const _FP16 *input =
          in.getData<_FP16>() + start_idx * num_cache_head * gqa_size;
        _FP16 *out =
          output.getData<_FP16>() + i * (num_cache_head * gqa_size * head_dim);
        int row_num = is_causal ? (from + (int)i) : to - 1;
        nntrainer::compute_fp16vcache_transposed(
          row_num, input, vcache.getData<_FP16>(), out, num_cache_head,
          gqa_size, head_dim, local_window_size);
      });
    } else {
      // Single token processing (common during generation)
      // Parallelize over KV heads for decoding since Q direction is always 1
      int row_num = to - 1;

      // Use OpenMP for lower overhead parallelization during decoding
      const _FP16 *in_data = in.getData<_FP16>();
      const _FP16 *vcache_data = vcache.getData<_FP16>();
      _FP16 *output_data = output.getData<_FP16>();

      auto &tm_fp16 = nntrainer::ThreadManager::Global();
      tm_fp16.parallel_for(
        0, static_cast<size_t>(num_cache_head), [=](size_t head_kv) {
          nntrainer::compute_fp16vcache_transposed(
            row_num, in_data, vcache_data, output_data, num_cache_head,
            gqa_size, head_dim, local_window_size, head_kv, head_kv + 1);
        });
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

void MHACoreLayer::setBatch(nntrainer::RunLayerContext &context,
                            unsigned int batch) {

  const float dropout_rate =
    std::get<nntrainer::props::DropOutRate>(mha_core_props).get();
  if (internal_cache_requested) {
    context.updateTensor(tensor_idx[AttentionParams::cache_key], batch);
    context.updateTensor(tensor_idx[AttentionParams::cache_value], batch);
  }
  // context.updateTensor(tensor_idx[AttentionParams::attention_weight], batch);
  if (dropout_rate > epsilon) {
    context.updateTensor(tensor_idx[AttentionParams::dropout_mask], batch);
  }
  // Training caches (populated by trainForwarding(), consumed by
  // calcDerivative()): only resized in updateTensorsByInputDimensions()
  // (sequence-length changes) until this fix -- a batch-size change alone
  // (this function) left them sized for whatever batch finalize() ran with,
  // so every batch index beyond that read/wrote past the end of its backing
  // tensor once the real batch size was set. Same root cause as
  // RMSNormLayer::setBatch's note.
  if (train_tensors_requested) {
    context.updateTensor(tensor_idx[AttentionParams::train_q_roped], batch);
    context.updateTensor(tensor_idx[AttentionParams::train_k_roped], batch);
    context.updateTensor(tensor_idx[AttentionParams::train_attn_wt], batch);
    context.updateTensor(tensor_idx[AttentionParams::train_v], batch);
  }
}

void MHACoreLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  unsigned int height = input_dimensions[0].height();
  unsigned int &max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();
  unsigned int &max_new_tokens =
    std::get<props::MaxNewTokens>(mha_core_props).get();
  max_position_embeddings =
    std::get<props::MaxPositionEmbeddings>(mha_core_props).get();
  max_timestep = height + max_new_tokens;

  ml::train::TensorDim kv_dim = input_dimensions[0];
  kv_dim.width(kv_dim.width() / (num_heads_Q / num_heads_KV));

  ml::train::TensorDim kv_cache_dim = kv_dim;
#ifdef ENABLE_FP16
  kv_cache_dim.setDataType(ml::train::TensorDim::DataType::FP16);
#else
  kv_cache_dim.setDataType(ml::train::TensorDim::DataType::UINT16);
#endif
  kv_cache_dim.height(max_timestep);

  context.updateInput(INOUT_INDEX::QUERY, input_dimensions[0]);
  context.updateInput(INOUT_INDEX::KEY, kv_dim);
  context.updateInput(INOUT_INDEX::VALUE, kv_dim);
  context.updateOutput(0, input_dimensions[0]);

  if (internal_cache_requested) {
    context.updateTensor(tensor_idx[AttentionParams::cache_key], kv_cache_dim);
    context.updateTensor(tensor_idx[AttentionParams::cache_value],
                         kv_cache_dim);
  }

  if (train_tensors_requested) {
    // Sized from the actual sequence length, not max_timestep — see the
    // matching note in finalize().
    ml::train::TensorDim q_roped_dim = input_dimensions[0];
    q_roped_dim.setDataType(ml::train::TensorDim::DataType::FP32);

    ml::train::TensorDim k_roped_dim = kv_dim;
    k_roped_dim.setDataType(ml::train::TensorDim::DataType::FP32);

    ml::train::TensorDim attn_wt_dim(
      {input_dimensions[0].batch(), num_heads_Q, height, height},
      {input_dimensions[0].getFormat(), ml::train::TensorDim::DataType::FP32});

    context.updateTensor(tensor_idx[AttentionParams::train_q_roped],
                         q_roped_dim);
    context.updateTensor(tensor_idx[AttentionParams::train_k_roped],
                         k_roped_dim);
    context.updateTensor(tensor_idx[AttentionParams::train_attn_wt],
                         attn_wt_dim);
    context.updateTensor(tensor_idx[AttentionParams::train_v], k_roped_dim);
  }
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

  {
    static bool printed = false;
    if (!printed) {
      auto scan = [&](const char *label, nntrainer::Tensor &t) {
        const float *d = t.getData<float>();
        const size_t n = t.size();
        for (size_t k = 0; k < n && !printed; ++k) {
          uint32_t bits;
          float v = d[k];
          std::memcpy(&bits, &v, sizeof(v));
          if ((bits & 0x7F800000u) == 0x7F800000u) {
            printed = true;
            const auto &td = t.getDim();
            fprintf(stderr,
                    "HEXDIAG mha_in layer=%s tensor=%s flatidx=%zu BAD "
                    "(dim=%zu,%zu,%zu,%zu)\n",
                    context.getName().c_str(), label, k, td.batch(),
                    td.channel(), td.height(), td.width());
          }
        }
      };
      scan("query", query);
      scan("key", key);
      scan("value", value);
    }
  }

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
  apply_rotary_emb_tensor_v2(query, train_q, head_dim, 0, false);
  apply_rotary_emb_tensor_v2(key, train_k, head_dim, 0, false);

  // Keep a private copy of V for calcDerivative(): the VALUE input buffer
  // is only guaranteed alive through forwarding().
  {
    nntrainer::Tensor &train_v =
      context.getTensor(tensor_idx[AttentionParams::train_v]);
    std::memcpy(train_v.getData<float>(), value.getData<float>(),
                sizeof(float) * value.size());
  }

  const ml::train::TensorDim q_dim = train_q.getDim();    // (B,1,max_ts,HQ*D)
  const ml::train::TensorDim k_dim = train_k.getDim();    // (B,1,max_ts,HKV*D)
  const ml::train::TensorDim v_dim = value.getDim();      // (B,1,seq,HKV*D)
  const ml::train::TensorDim o_dim = output.getDim();     // (B,1,seq,HQ*D)
  const ml::train::TensorDim attn_dim = train_attn.getDim(); // (B,HQ,max_ts,max_ts)

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
  if (should_use_flash_attn(seq_len, head_dim, /*is_prefill=*/true) &&
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
                  seq_len, num_heads_Q, num_heads_KV, head_dim, seq_len,
                  scale, /*q_is_fp16=*/0, /*out_is_fp16=*/0);
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

  {
    static bool printed = false;
    if (!printed) {
      for (unsigned int b = 0; b < batch && !printed; ++b) {
        const float *o_b = out_data + b * o_dim.getFeatureLen();
        for (unsigned int i = 0; i < seq_len && !printed; ++i) {
          const float *row = o_b + i * o_row_stride;
          for (unsigned int k = 0; k < o_row_stride; ++k) {
            uint32_t bits;
            float v = row[k];
            std::memcpy(&bits, &v, sizeof(v));
            if ((bits & 0x7F800000u) == 0x7F800000u) {
              printed = true;
              fprintf(stderr,
                      "HEXDIAG mha_out layer=%s b=%u row=%u k=%u BAD\n",
                      context.getName().c_str(), b, i, k);
              break;
            }
          }
        }
      }
    }
  }
}

void MHACoreLayer::apply_inverse_rotary_emb(float *grad, unsigned int width,
                                            unsigned int dim,
                                            unsigned int height,
                                            unsigned int from) {
  if (!use_rope)
    return;

  const unsigned int half_ = dim / 2;
  for (unsigned int h = 0; h < height; ++h) {
    const std::vector<float> &cos_ = freqs_fp32->cos[from + h];
    const std::vector<float> &sin_ = freqs_fp32->sin[from + h];
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

  const ml::train::TensorDim dy_dim = dy.getDim(); // (B,1,seq,HQ*D)
  const ml::train::TensorDim dq_dim = dQ.getDim(); // (B,1,seq,HQ*D)
  const ml::train::TensorDim dk_dim = dK.getDim(); // (B,1,seq,HKV*D)
  const ml::train::TensorDim dv_dim = dV.getDim(); // (B,1,seq,HKV*D)
  const ml::train::TensorDim q_dim = train_q.getDim();       // (B,1,max_ts,HQ*D)
  const ml::train::TensorDim k_dim = train_k.getDim();       // (B,1,max_ts,HKV*D)
  const ml::train::TensorDim v_dim = value.getDim();         // (B,1,seq,HKV*D)
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
  std::vector<float> q_gather(batch * num_heads_Q * S * D);
  std::vector<float> dy_gather(batch * num_heads_Q * S * D);
  std::vector<float> dq_scratch(batch * num_heads_Q * S * D);
  std::vector<float> dk_partial(batch * num_heads_Q * S * D);
  std::vector<float> dv_partial(batch * num_heads_Q * S * D);
  // d_attn (GEMM1 output), overwritten in place with d_score (softmax
  // backward output) once GEMM1 completes - the two are never live at once.
  std::vector<float> score(batch * num_heads_Q * S * S);

  std::vector<float> k_gather(batch * num_heads_KV * S * D);
  std::vector<float> v_gather(batch * num_heads_KV * S * D);
  std::vector<float> dk_accum(batch * num_heads_KV * S * D, 0.0f);
  std::vector<float> dv_accum(batch * num_heads_KV * S * D, 0.0f);

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
  run_batched_gemm(round1, S);

  if (context.getName() == "layer1_attention") {
    static bool printed = false;
    if (!printed) {
      printed = true;
      auto minmax = [](const char *label, const float *d, size_t n) {
        float mn = std::numeric_limits<float>::infinity();
        float mx = -std::numeric_limits<float>::infinity();
        for (size_t k = 0; k < n; ++k) {
          if (d[k] < mn)
            mn = d[k];
          if (d[k] > mx)
            mx = d[k];
        }
        fprintf(stderr, "HEXDIAG mag_%s layer1 min=%g max=%g\n", label, mn, mx);
      };
      minmax("dy", dy_data, dy.size());
      minmax("q_gather", q_gather.data(), q_gather.size());
      minmax("k_gather", k_gather.data(), k_gather.size());
      minmax("v_gather", v_gather.data(), v_gather.size());
      minmax("round1_dattn", score.data(), score.size());
      minmax("train_attn", attn_data, train_attn.size());
    }
  }

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
                      k_gather.data() + kv_idx * S * D, attn_b + qh * S * S,
                      S, S, D, 0, 1});
      }
    }
    run_batched_gemm(qk, S);

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
  run_batched_gemm(round2, S);

  if (context.getName() == "layer1_attention") {
    static bool printed = false;
    if (!printed) {
      printed = true;
      auto minmax = [](const char *label, const float *d, size_t n) {
        float mn = std::numeric_limits<float>::infinity();
        float mx = -std::numeric_limits<float>::infinity();
        for (size_t k = 0; k < n; ++k) {
          if (d[k] < mn)
            mn = d[k];
          if (d[k] > mx)
            mx = d[k];
        }
        fprintf(stderr, "HEXDIAG mag_%s layer1 min=%g max=%g\n", label, mn, mx);
      };
      minmax("d_score", score.data(), score.size());
      minmax("dq_scratch", dq_scratch.data(), dq_scratch.size());
      minmax("dk_partial", dk_partial.data(), dk_partial.size());
      minmax("dv_partial", dv_partial.data(), dv_partial.size());
    }
  }

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

  if (context.getName() == "layer1_attention") {
    static bool printed_pre = false;
    if (!printed_pre) {
      printed_pre = true;
      auto minmax = [](const char *label, const float *d, size_t n) {
        float mn = std::numeric_limits<float>::infinity();
        float mx = -std::numeric_limits<float>::infinity();
        for (size_t k = 0; k < n; ++k) {
          if (d[k] < mn)
            mn = d[k];
          if (d[k] > mx)
            mx = d[k];
        }
        fprintf(stderr, "HEXDIAG mag_%s layer1 min=%g max=%g\n", label, mn, mx);
      };
      minmax("dQ_prerope", dq_data, dQ.size());
      minmax("dK_prerope", dk_data, dK.size());
      minmax("dV_final", dv_data, dV.size());
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

  if (context.getName() == "layer1_attention") {
    static bool printed_post = false;
    if (!printed_post) {
      printed_post = true;
      auto minmax = [](const char *label, const float *d, size_t n) {
        float mn = std::numeric_limits<float>::infinity();
        float mx = -std::numeric_limits<float>::infinity();
        for (size_t k = 0; k < n; ++k) {
          if (d[k] < mn)
            mn = d[k];
          if (d[k] > mx)
            mx = d[k];
        }
        fprintf(stderr, "HEXDIAG mag_%s layer1 min=%g max=%g\n", label, mn, mx);
      };
      minmax("dQ_postrope", dq_data, dQ.size());
      minmax("dK_postrope", dk_data, dK.size());
    }
  }

  {
    static bool printed_dy = false;
    if (!printed_dy) {
      const float *d = dy_data;
      const size_t n = dy.size();
      for (size_t k = 0; k < n; ++k) {
        uint32_t bits;
        float v = d[k];
        std::memcpy(&bits, &v, sizeof(v));
        if ((bits & 0x7F800000u) == 0x7F800000u) {
          printed_dy = true;
          fprintf(stderr,
                  "HEXDIAG bwd_dy_firstbad layer=%s flatidx=%zu BAD "
                  "val_bits=0x%08x flash=%d\n",
                  context.getName().c_str(), k, bits,
                  (int)last_forward_used_flash);
          break;
        }
      }
    }
  }

  if (context.getName() == "layer0_attention") {
    static int call_count = 0;
    if (call_count < 3) {
      ++call_count;
      auto scan = [&](const char *label, const float *d, size_t n) {
        for (size_t k = 0; k < n; ++k) {
          uint32_t bits;
          float v = d[k];
          std::memcpy(&bits, &v, sizeof(v));
          if ((bits & 0x7F800000u) == 0x7F800000u) {
            fprintf(stderr,
                    "HEXDIAG bwd_%s layer=%s call=%d flatidx=%zu BAD "
                    "val_bits=0x%08x flash=%d\n",
                    label, context.getName().c_str(), call_count, k, bits,
                    (int)last_forward_used_flash);
            return;
          }
        }
        fprintf(stderr, "HEXDIAG bwd_%s layer=%s call=%d CLEAN n=%zu flash=%d\n",
                label, context.getName().c_str(), call_count, n,
                (int)last_forward_used_flash);
      };
      scan("dy", dy_data, dy.size());
      scan("score_dattn", score.data(), score.size());
      scan("dq", dq_data, dQ.size());
      scan("dk", dk_data, dK.size());
      scan("dv", dv_data, dV.size());
    }
  }
}

void MHACoreLayer::calcGradient(nntrainer::RunLayerContext &context) {}

void MHACoreLayer::exportTo(nntrainer::Exporter &exporter,
                            const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(mha_core_props, method, this);
}

void MHACoreLayer::setProperty(const std::vector<std::string> &values) {
  std::vector<std::string> props;
  props.reserve(values.size());
  for (const auto &value : values) {
    std::string key;
    std::string parsed_value;
    if (nntrainer::getKeyValue(value, key, parsed_value) == ML_ERROR_NONE &&
        key == "cache_index") {
      setCacheIndex(static_cast<unsigned int>(std::stoul(parsed_value)));
    } else {
      props.push_back(value);
    }
  }

  auto remain_props = loadProperties(props, mha_core_props);
  LayerImpl::setProperty(remain_props);
}

size_t MHACoreLayer::calc_attn_index(size_t i) { return (i * (i + 1)) / 2; };

size_t MHACoreLayer::calc_windowed_attn_index(size_t i) {
  // S(i) = sum_{k=0}^{i-1} min(k+1, W)
  // For i <= W:  S(i) = i*(i+1)/2   (same as full-attention triangular index)
  // For i >  W:  S(i) = W*(W+1)/2 + (i - W)*W
  // When W == UINT_MAX, i <= W is always true, so we never evaluate
  // W*(W+1)/2 and there is no overflow.
  if (i <= local_window_size) {
    return (i * (i + 1)) / 2;
  } else {
    return (local_window_size * (local_window_size + 1)) / 2 +
           (i - local_window_size) * local_window_size;
  }
};

#ifdef PLUGGABLE

nntrainer::Layer *create_mha_core_layer() {
  auto layer = new MHACoreLayer();
  return layer;
}

void destroy_mha_core_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_mha_core_layer,
                                                   destroy_mha_core_layer};
}

#endif

} // namespace causallm
