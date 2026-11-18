// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   hexagon_compute_ops.cpp
 * @date   23 July 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @brief  Hexagon HTP ComputeOps table (FastRPC bridge).
 *
 * The matmuls run on the DSP through the nntr_htp_bridge_* entry points of
 * libggml-hexagon.so (ggml-hexagon's nntr-htp-bridge.cpp), which reuse its
 * cDSP session machinery directly: nntrainer has no ggml graph to hand it.
 * The library is dlopen()ed on first use; if it is missing, every op falls
 * back to the CPU.
 *
 * Weights stay in the ordinary ARM q4_0x4 layout on the host (the CPU kernels
 * and the CPU fallbacks below read the same bytes). The first time a weight
 * is seen it is converted q4_0x4 -> block_q4_0 -> q4x4x2 and uploaded to a
 * persistent rpcmem arena keyed on the host pointer, where the bridge tiles
 * it for the HMX kernels. The weight pool has MAX_LIFESPAN, so the address is
 * stable for the life of the process.
 *
 * Every op not overridden here is inherited from CpuComputeOps: the table is
 * attached to a whole context, so each tensor created under engine=htp
 * reaches it for every op it runs, not only for the accelerated ones.
 */

#include <compute_ops.h>
#include <cpu_ops_table.h>
#include <fp16.h>
#include <hexagon_compute_ops.h>
#include <hexagon_repack.h>
#include <quantizer.h>
#include <tensor.h>

#include <nntrainer_log.h>

#include <dlfcn.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace nntrainer {

namespace {

using nntr_htp_bridge_upload_fn = int (*)(const void *, const void *,
                                          unsigned int, unsigned int);
using nntr_htp_bridge_gemm_q4_0_fn = int (*)(const void *, const float *,
                                             float *, unsigned int,
                                             unsigned int, unsigned int);
using nntr_htp_bridge_sgemm_fn = int (*)(const float *, const float *, float *,
                                         unsigned int, unsigned int,
                                         unsigned int, int, int);
using nntr_htp_bridge_gemm_f16w_fn = int (*)(const void *, const void *,
                                             const float *, float *,
                                             unsigned int, unsigned int,
                                             unsigned int);

struct BridgeApi {
  nntr_htp_bridge_upload_fn upload = nullptr;
  nntr_htp_bridge_gemm_q4_0_fn gemm = nullptr; /**< Q4_0 weight x FP32 act */
  nntr_htp_bridge_sgemm_fn sgemm = nullptr;    /**< general FP32 GEMM */
  nntr_htp_bridge_gemm_f16w_fn gemm_f16w = nullptr; /**< resident FP16 weight */
};

/**
 * @brief Lazily dlopen libggml-hexagon.so and dlsym the bridge entry points.
 * Cached for the process lifetime: ggml-hexagon's session (kept alive across
 * calls on its side) assumes one long-lived library handle.
 */
const BridgeApi &get_bridge_api() {
  static const BridgeApi api = [] {
    void *handle = dlopen("libggml-hexagon.so", RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
      ml_logw("HexagonComputeOps: dlopen(libggml-hexagon.so) failed: %s "
              "- all ops will fall back to CPU",
              dlerror());
      return BridgeApi{};
    }

    auto sym = [handle](const char *name) {
      void *s = dlsym(handle, name);
      if (!s) {
        std::ostringstream oss;
        oss << "HexagonComputeOps: dlsym(" << name << ") failed: " << dlerror();
        throw std::runtime_error(oss.str());
      }
      return s;
    };
    // The FP32 GEMM and the resident-weight GEMM are optional: an older
    // libggml-hexagon.so without them just leaves those ops on the CPU.
    auto sym_optional = [handle](const char *name) -> void * {
      return dlsym(handle, name);
    };

    BridgeApi a;
    a.upload = reinterpret_cast<nntr_htp_bridge_upload_fn>(
      sym("nntr_htp_bridge_upload_weight_q4x4x2"));
    a.gemm = reinterpret_cast<nntr_htp_bridge_gemm_q4_0_fn>(
      sym("nntr_htp_bridge_gemm_q4_0"));
    a.sgemm = reinterpret_cast<nntr_htp_bridge_sgemm_fn>(
      sym_optional("nntr_htp_bridge_sgemm_fp32"));
    a.gemm_f16w = reinterpret_cast<nntr_htp_bridge_gemm_f16w_fn>(
      sym_optional("nntr_htp_bridge_gemm_f16w"));
    return a;
  }();
  return api;
}

/** rows of a tensor with batch and channel folded in (NCHW) */
unsigned int rowsOf(const Tensor &t) {
  return t.getDim().batch() * t.getDim().channel() * t.getDim().height();
}

/**
 * HMX accumulates and returns FP16. A few layers of Qwen3 (the
 * massive-activation ffn_down outputs) exceed its range, which shows up as
 * saturated or non-finite values.
 */
bool fitsFp16(const float *d, size_t n) {
  for (size_t i = 0; i < n; ++i)
    if (!(std::fabs(d[i]) < 65504.0f))
      return false;
  return true;
}

} // namespace

class HexagonComputeOps : public CpuComputeOps {
public:
  /** Q4_0 GEMM for any tensor-level dot() against a Q4_0 weight. */
  bool supports_gemm_q4_0_accel_fp32() const override {
    return get_bridge_api().gemm != nullptr;
  }

  void gemm_q4_0_accel_fp32(void *matAdata, float *matBdata, float *matCdata,
                            unsigned int M, unsigned int N,
                            unsigned int K) override {
    const BridgeApi &api = get_bridge_api();
    if (api.gemm) {
      ensure_uploaded(api, matAdata, N, K);
      if (api.gemm(matAdata, matBdata, matCdata, M, N, K) == 0)
        return;
      // The bridge declines rather than crashes for shapes it cannot take
      // (e.g. more than 1024 activation rows).
      ml_logw("HexagonComputeOps::gemm_q4_0_accel_fp32: bridge declined "
              "(M=%u N=%u K=%u) - using the CPU kernel",
              M, N, K);
    }
    cpu_->gemm_q4_0_fp32(M, N, K, matBdata, K, matAdata, N, matCdata, N);
  }

  /**
   * output = input * W for a frozen Q4_0 weight. The DSP dequantizes the
   * weight to FP16 tiles and converts the FP32 activation to FP16 itself, so
   * there is none of the Q8_0 activation noise of the CPU kernel. If the
   * result leaves the FP16 range, the weight is remembered and this call and
   * every later one take the exact CPU path.
   */
  void fc_frozen(Tensor &input, Tensor &weight, Tensor &output) override {
    const BridgeApi &api = get_bridge_api();
    void *key = weight.getData();
    if (api.gemm && onDsp(weight, input, output) && !overflowed(key)) {
      const unsigned int M = rowsOf(input);
      const unsigned int K = input.width();
      const unsigned int N = output.width();
      ensure_uploaded(api, key, N, K);
      if (api.gemm(key, input.getData<float>(), output.getData<float>(), M, N,
                   K) == 0) {
        if (fitsFp16(output.getData<float>(), output.size()))
          return;
        mark_overflowed(key);
        ml_logw("HexagonComputeOps::fc_frozen: FP16 overflow, this weight now "
                "runs on the CPU");
      } else {
        ml_logw("HexagonComputeOps::fc_frozen: bridge declined (M=%u N=%u "
                "K=%u) - using the CPU path",
                M, N, K);
      }
    }
    CpuComputeOps::fc_frozen(input, weight, output);
  }

  /**
   * dx = dy * W^T against a resident FP16 copy of the dequantized weight:
   * nntrainer's [in x out] weight layout is exactly the DSP's [rows x cols]
   * layout for this product, so no transpose is needed. Weights whose forward
   * overflowed FP16 stay on the CPU.
   */
  void fc_frozen_dgrad(const Tensor &dy, Tensor &weight, Tensor &dx) override {
    const BridgeApi &api = get_bridge_api();
    void *key = weight.getData();
    if (api.gemm_f16w && onDsp(weight, dy, dx) && !overflowed(key)) {
      const unsigned int M = rowsOf(dy);
      const unsigned int N_out = dy.width();
      const unsigned int K_in = dx.width();

      std::vector<uint16_t> w16;
      if (!has_f16_copy(key)) {
        Tensor w = Quantization::createQuantizer(QScheme::Q4_0)
                     ->dequantize(weight, Tdatatype::FP32);
        const float *wd = w.getData<float>();
        w16.resize(static_cast<size_t>(K_in) * N_out);
        for (size_t i = 0; i < w16.size(); ++i)
          w16[i] = compute_fp32_to_fp16(wd[i]);
      }
      const int rc =
        api.gemm_f16w(key, w16.empty() ? nullptr : w16.data(),
                      dy.getData<float>(), dx.getData<float>(), M, K_in, N_out);
      if (rc == 0) {
        mark_f16_copy(key);
        return;
      }
      ml_logw("HexagonComputeOps::fc_frozen_dgrad: bridge failed (rc=%d) - "
              "using the CPU path",
              rc);
    }
    CpuComputeOps::fc_frozen_dgrad(dy, weight, dx);
  }

  /**
   * FP32 GEMM, used by the large training matmuls that have no quantized
   * weight (alpha = 1, beta = 0, tight leading dimensions). Skinny GEMMs
   * (LoRA rank-sized N or K) are dispatch-bound and the DSP MUL_MAT path
   * hangs on them, so they stay on the CPU. NNTR_HEXAGON_SGEMM_MIN_DIM
   * overrides the threshold.
   */
  void sgemm_fp32(const unsigned int TStorageOrder, bool TransA, bool TransB,
                  const unsigned int M, const unsigned int N,
                  const unsigned int K, const float alpha, const float *A,
                  const unsigned int lda, const float *B,
                  const unsigned int ldb, const float beta, float *C,
                  const unsigned int ldc) override {
    const BridgeApi &api = get_bridge_api();
    static const unsigned int min_dim = [] {
      const char *v = std::getenv("NNTR_HEXAGON_SGEMM_MIN_DIM");
      return v ? static_cast<unsigned int>(std::strtoul(v, nullptr, 10)) : 64u;
    }();
    const bool big_enough = M >= min_dim && N >= min_dim && K >= min_dim;
    const bool tight = (TransA ? lda == M : lda == K) &&
                       (TransB ? ldb == K : ldb == N) && ldc == N;
    if (api.sgemm && big_enough && tight && alpha == 1.0f && beta == 0.0f) {
      const int rc =
        api.sgemm(A, B, C, M, N, K, TransA ? 1 : 0, TransB ? 1 : 0);
      if (rc == 0)
        return;
      ml_logw("HexagonComputeOps::sgemm_fp32: bridge failed (rc=%d, M=%u N=%u "
              "K=%u) - using the CPU",
              rc, M, N, K);
    }
    cpu_->sgemm_fp32(TStorageOrder, TransA, TransB, M, N, K, alpha, A, lda, B,
                     ldb, beta, C, ldc);
  }

private:
  /** only a Q4_0 weight with FP32 activations is taken by the DSP kernels */
  static bool onDsp(const Tensor &weight, const Tensor &a, const Tensor &b) {
    return weight.getDataType() == Tdatatype::Q4_0 &&
           a.getDataType() == Tdatatype::FP32 &&
           b.getDataType() == Tdatatype::FP32;
  }

  bool overflowed(const void *key) {
    std::lock_guard<std::mutex> lk(mtx_);
    return overflowed_.count(key) != 0;
  }
  void mark_overflowed(const void *key) {
    std::lock_guard<std::mutex> lk(mtx_);
    overflowed_.insert(key);
  }
  bool has_f16_copy(const void *key) {
    std::lock_guard<std::mutex> lk(mtx_);
    return f16_resident_.count(key) != 0;
  }
  void mark_f16_copy(const void *key) {
    std::lock_guard<std::mutex> lk(mtx_);
    f16_resident_.insert(key);
  }

  /**
   * Convert the ARM q4_0x4 weight to q4x4x2 and hand it to the bridge, once
   * per weight. All three layouts take 18 bytes per 32 values, so the two
   * scratch buffers are exactly one weight each and are reused across
   * weights: peak extra host memory is twice the largest weight, not the
   * model.
   */
  void ensure_uploaded(const BridgeApi &api, void *matAdata, unsigned int N,
                       unsigned int K) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (uploaded_.count(matAdata))
      return;

    const size_t nbytes = (size_t)N * ((size_t)K / 32) * 18;
    if (unpack_scratch_.size() < nbytes) {
      unpack_scratch_.resize(nbytes);
      htp_scratch_.resize(nbytes);
    }
    cpu_->unpack_q4_0(matAdata, unpack_scratch_.data(), nbytes, N, K);
    repack_q4_0_to_htp_q4x4x2(htp_scratch_.data(), unpack_scratch_.data(),
                              nbytes, N, K);
    if (api.upload(matAdata, htp_scratch_.data(), N, K) != 0) {
      throw std::runtime_error(
        "HexagonComputeOps::ensure_uploaded: "
        "nntr_htp_bridge_upload_weight_q4x4x2 failed (see log for details)");
    }
    uploaded_.insert(matAdata);
  }

  ComputeOps *cpu_ = get_cpu_ops();

  std::mutex mtx_;
  std::unordered_set<const void *> uploaded_;     /**< q4x4x2 on the DSP */
  std::unordered_set<const void *> f16_resident_; /**< FP16 copy on the DSP */
  std::unordered_set<const void *> overflowed_; /**< forward left FP16 range */
  std::vector<uint8_t> unpack_scratch_;
  std::vector<uint8_t> htp_scratch_;
};

ComputeOps *get_hexagon_ops() {
  static HexagonComputeOps instance;
  return &instance;
}

} // namespace nntrainer
