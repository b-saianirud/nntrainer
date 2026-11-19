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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
   * logits = input * W^T for a frozen FP32 weight stored [out x in] (tied
   * embedding / lm_head: 151936 x 1024). The weight is far larger than one
   * bridge arena, so it is uploaded as FP16 row chunks and the product is
   * computed chunk by chunk; each chunk's [M x chunk] result is scattered
   * into its columns of the output. NNTR_HEXAGON_NT=0 disables, and any
   * bridge failure falls back to the exact CPU product.
   */
  void fc_frozen_nt(Tensor &input, Tensor &weight, Tensor &output) override {
    const BridgeApi &api = get_bridge_api();
    const unsigned int M = rowsOf(input);
    const unsigned int K = input.width();
    const unsigned int N = output.width();
    if (api.gemm_f16w && ntOnDsp(weight, input, output, M, K, N)) {
      std::lock_guard<std::mutex> lk(nt_mtx_);
      const float *w = weight.getData<float>();
      const unsigned int chunk = ntChunk();
      const char *base = reinterpret_cast<const char *>(weight.getData());
      bool ok = true;
      unsigned int idx = 0;
      for (unsigned int c0 = 0; c0 < N && ok; c0 += chunk, ++idx) {
        const unsigned int nc = std::min(chunk, N - c0);
        const void *key = base + 1 + idx;
        std::vector<uint16_t> w16;
        if (!has_f16_copy(key)) {
          w16.resize(static_cast<size_t>(nc) * K);
          const float *src = w + static_cast<size_t>(c0) * K;
          for (size_t i = 0; i < w16.size(); ++i)
            w16[i] = compute_fp32_to_fp16(src[i]);
        }
        nt_tmp_.resize(static_cast<size_t>(M) * nc);
        ok =
          api.gemm_f16w(key, w16.empty() ? nullptr : w16.data(),
                        input.getData<float>(), nt_tmp_.data(), M, nc, K) == 0;
        if (!ok)
          break;
        mark_f16_copy(key);
        float *out = output.getData<float>();
        for (unsigned int m = 0; m < M; ++m)
          std::memcpy(out + static_cast<size_t>(m) * N + c0,
                      nt_tmp_.data() + static_cast<size_t>(m) * nc,
                      sizeof(float) * nc);
      }
      if (ok) {
        if (ntCheck())
          ntCheckForward(input, weight, output, M, K, N);
        return;
      }
      ml_logw("HexagonComputeOps::fc_frozen_nt: bridge failed (M=%u N=%u K=%u)"
              " - using the CPU path",
              M, N, K);
    }
    CpuComputeOps::fc_frozen_nt(input, weight, output);
  }

  /**
   * dx = dy * W for the same weight: a sum over the vocabulary, done as one
   * product per chunk against a resident FP16 copy of that chunk transposed
   * to [in x chunk]. dy is scaled by a power of two before the FP16 conversion
   * (softmax gradients are tiny and would flush to zero) and the result is
   * scaled back exactly.
   */
  void fc_frozen_nt_dgrad(const Tensor &dy, Tensor &weight,
                          Tensor &dx) override {
    const BridgeApi &api = get_bridge_api();
    const unsigned int M = rowsOf(dy);
    const unsigned int N = dy.width();
    const unsigned int K = dx.width(); /* hidden size */
    if (api.gemm_f16w && ntOnDsp(weight, dx, dy, M, K, N)) {
      std::lock_guard<std::mutex> lk(nt_mtx_);
      const float *w = weight.getData<float>();
      const float *g = dy.getData<float>();
      const unsigned int chunk = ntBackChunk();
      const char *base = reinterpret_cast<const char *>(weight.getData());

      float maxabs = 0.0f;
      const size_t total = static_cast<size_t>(M) * N;
      for (size_t i = 0; i < total; ++i)
        maxabs = std::max(maxabs, std::fabs(g[i]));
      float *out = dx.getData<float>();
      if (std::isfinite(maxabs) && maxabs == 0.0f) {
        std::memset(out, 0, sizeof(float) * static_cast<size_t>(M) * K);
        return;
      }
      if (std::isfinite(maxabs)) {
        const int e = static_cast<int>(std::floor(std::log2(8.0f / maxabs)));
        const float scale = std::ldexp(1.0f, e);
        const float inv = std::ldexp(1.0f, -e);

        bool ok = true;
        unsigned int idx = 0;
        std::vector<float> acc(static_cast<size_t>(M) * K, 0.0f);
        for (unsigned int c0 = 0; c0 < N && ok; c0 += chunk, ++idx) {
          const unsigned int nc = std::min(chunk, N - c0);
          const void *key = base + 1000 + idx;
          std::vector<uint16_t> w16;
          if (!has_f16_copy(key)) {
            w16.resize(static_cast<size_t>(K) * nc);
            for (unsigned int j = 0; j < nc; ++j) {
              const float *row = w + static_cast<size_t>(c0 + j) * K;
              for (unsigned int h = 0; h < K; ++h)
                w16[static_cast<size_t>(h) * nc + j] =
                  compute_fp32_to_fp16(row[h]);
            }
          }
          nt_tmp_.resize(static_cast<size_t>(M) * nc);
          for (unsigned int m = 0; m < M; ++m) {
            const float *src = g + static_cast<size_t>(m) * N + c0;
            float *dst = nt_tmp_.data() + static_cast<size_t>(m) * nc;
            for (unsigned int j = 0; j < nc; ++j)
              dst[j] = src[j] * scale;
          }
          nt_out_.resize(static_cast<size_t>(M) * K);
          ok = api.gemm_f16w(key, w16.empty() ? nullptr : w16.data(),
                             nt_tmp_.data(), nt_out_.data(), M, K, nc) == 0;
          if (!ok)
            break;
          mark_f16_copy(key);
          for (size_t i = 0; i < acc.size(); ++i)
            acc[i] += nt_out_[i];
        }
        if (ok) {
          for (size_t i = 0; i < acc.size(); ++i)
            out[i] = acc[i] * inv;
          if (ntCheck())
            ntCheckBackward(dy, weight, dx, M, K, N);
          return;
        }
        ml_logw("HexagonComputeOps::fc_frozen_nt_dgrad: bridge failed (M=%u "
                "N=%u K=%u) - using the CPU path",
                M, N, K);
      }
    }
    CpuComputeOps::fc_frozen_nt_dgrad(dy, weight, dx);
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

  /** FP32 weight [N x K] against FP32 activations, shapes the DSP can take */
  static bool ntOnDsp(const Tensor &weight, const Tensor &a, const Tensor &b,
                      unsigned int M, unsigned int K, unsigned int N) {
    static const bool off = [] {
      const char *v = std::getenv("NNTR_HEXAGON_NT");
      return v != nullptr && v[0] == '0';
    }();
    return !off && weight.getDataType() == Tdatatype::FP32 &&
           a.getDataType() == Tdatatype::FP32 &&
           b.getDataType() == Tdatatype::FP32 && M >= 1 && M <= 1024 &&
           K % 32 == 0 && N % 32 == 0 &&
           weight.size() == static_cast<size_t>(N) * K;
  }
  /** rows of the vocabulary uploaded per FP16 chunk (multiple of 32) */
  static unsigned int ntChunk() {
    static const unsigned int c = [] {
      const char *v = std::getenv("NNTR_HEXAGON_NT_CHUNK");
      const unsigned long n = v ? std::strtoul(v, nullptr, 10) : 32768ul;
      return static_cast<unsigned int>(n >= 32 ? (n / 32) * 32 : 32768ul);
    }();
    return c;
  }
  /**
   * Vocabulary rows per backward chunk. The backward product reduces over
   * the chunk, and the DSP matmul needs the padded activation tile
   * (rows x chunk FP16) to fit its 8 MB of VTCM, so this is much smaller than
   * the forward chunk (32768 failed to solve for more than 32 rows).
   */
  static unsigned int ntBackChunk() {
    static const unsigned int c = [] {
      const char *v = std::getenv("NNTR_HEXAGON_NT_BCHUNK");
      const unsigned long n = v ? std::strtoul(v, nullptr, 10) : 8192ul;
      return static_cast<unsigned int>(n >= 32 ? (n / 32) * 32 : 8192ul);
    }();
    return c;
  }
  static bool ntCheck() {
    static const bool on = std::getenv("NNTR_HEXAGON_NT_CHECK") != nullptr;
    return on;
  }
  /** first call only: compare the DSP result with the CPU on a few rows */
  void ntCheckForward(Tensor &input, Tensor &weight, Tensor &output,
                      unsigned int M, unsigned int K, unsigned int N) {
    if (nt_checked_fwd_++)
      return;
    const unsigned int m = std::min(M, 2u);
    std::vector<float> ref(static_cast<size_t>(m) * N);
    cpu_->sgemm_fp32(0, false, true, m, N, K, 1.0f, input.getData<float>(), K,
                     weight.getData<float>(), K, 0.0f, ref.data(), N);
    double maxerr = 0, maxref = 0;
    const float *o = output.getData<float>();
    for (size_t i = 0; i < ref.size(); ++i) {
      maxerr = std::max(maxerr, static_cast<double>(std::fabs(o[i] - ref[i])));
      maxref = std::max(maxref, static_cast<double>(std::fabs(ref[i])));
    }
    ml_logi("HexagonComputeOps NT check fwd: M=%u N=%u K=%u max|err|=%.5g "
            "max|ref|=%.5g",
            M, N, K, maxerr, maxref);
  }
  void ntCheckBackward(const Tensor &dy, Tensor &weight, Tensor &dx,
                       unsigned int M, unsigned int K, unsigned int N) {
    if (nt_checked_bwd_++)
      return;
    const unsigned int m = std::min(M, 2u);
    std::vector<float> ref(static_cast<size_t>(m) * K);
    cpu_->sgemm_fp32(0, false, false, m, K, N, 1.0f, dy.getData<float>(), N,
                     weight.getData<float>(), K, 0.0f, ref.data(), K);
    double maxerr = 0, maxref = 0;
    const float *o = dx.getData<float>();
    for (size_t i = 0; i < ref.size(); ++i) {
      maxerr = std::max(maxerr, static_cast<double>(std::fabs(o[i] - ref[i])));
      maxref = std::max(maxref, static_cast<double>(std::fabs(ref[i])));
    }
    ml_logi("HexagonComputeOps NT check bwd: M=%u N=%u K=%u max|err|=%.5g "
            "max|ref|=%.5g",
            M, N, K, maxerr, maxref);
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
  std::mutex nt_mtx_;
  std::vector<float> nt_tmp_, nt_out_; /**< chunk staging for the NT ops */
  int nt_checked_fwd_ = 0, nt_checked_bwd_ = 0;
  std::vector<uint8_t> unpack_scratch_;
  std::vector<uint8_t> htp_scratch_;
};

ComputeOps *get_hexagon_ops() {
  static HexagonComputeOps instance;
  return &instance;
}

} // namespace nntrainer
