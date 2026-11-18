// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 j2z0.lee <j2z0.lee@ax.samsung.com>
 *
 * @file   htp_context.cpp
 * @date   19 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author j2z0.lee <j2z0.lee@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  HTP compute Context implementation.
 */

#if defined(ENABLE_HEXKL) || defined(ENABLE_HEXAGON_CDSP)

#include <htp_context.h>

#include <app_context.h>
#include <compute_ops.h>
#include <context_data.h>
#ifdef ENABLE_HEXKL
#include <htp_backend.h>
#endif
#ifdef ENABLE_HEXAGON_CDSP
#include <hexagon_compute_ops.h>
#include <hexagon_rpc_allocator.h>
#endif

#include <cstdlib>
#include <cstring>

namespace nntrainer {

std::unique_ptr<nntrainer::Layer>
HtpContext::createLayerObject(const std::string &type,
                              const std::vector<std::string> &properties) {
  // HTP has no custom layers; reuse the CPU app context's factories.
  return AppContext::Global().createLayerObject(type, properties);
}

std::unique_ptr<nntrainer::Layer>
HtpContext::createLayerObject(const int int_key,
                              const std::vector<std::string> &properties) {
  return AppContext::Global().createLayerObject(int_key, properties);
}

#ifdef ENABLE_HEXAGON_CDSP
namespace {
/**
 * @brief Whether the FastRPC bridge runs instead of HexKL. With only the
 *        bridge compiled in it always does; with both, HexKL (inference) is the
 *        default and NNTR_HTP_BACKEND=bridge selects the bridge (LoRA
 * training).
 */
bool useBridgeBackend() {
#ifdef ENABLE_HEXKL
  const char *backend = std::getenv("NNTR_HTP_BACKEND");
  return backend != nullptr && std::strcmp(backend, "bridge") == 0;
#else
  return true;
#endif
}
} // namespace
#endif

void HtpContext::initialize() noexcept {
  // Ensure the CPU backend ops table exists — it is both the fallback
  // and the base class of HtpComputeOps. Go through ensureComputeOps()
  // rather than init_backend() directly: init_backend() is not
  // idempotent (__ggml_init, __openblas_set_num_threads), and
  // ensureComputeOps() is the call_once funnel every other Context
  // routes through.
  ensureComputeOps();

  auto cd = getContextData();
  if (!cd)
    return;

  // Wrap the NPU ops installation in try/catch so that a late exception from
  // HtpBackend::global() or get_htp_ops() (e.g., missing skel, driver error)
  // degrades to CPU rather than propagating out of this noexcept function
  // and hitting std::terminate.  Pattern mirrors ClContext::initialize().
  try {
#ifdef ENABLE_HEXAGON_CDSP
    if (useBridgeBackend()) {
      // FastRPC bridge (libggml-hexagon.so, loaded with dlopen at first use;
      // without it every op below runs on the CPU). Q4_0 / FP16 weight GEMMs,
      // flash attention and the training whole-ops run on the DSP, and the
      // activation pool is rpcmem-backed so the bridge can map it directly.
      cd->setComputeOps(get_hexagon_ops());
      cd->setMemAllocator(std::make_shared<HexagonRpcAllocator>());
      return;
    }
#endif
#ifdef ENABLE_HEXKL
    // NPU up -> HTP ops (shgemm on NPU, rest delegated to CPU via
    // inheritance). NPU down -> plain CPU ops, so an engine="htp" model
    // still runs, transparently, on the CPU.
    if (HtpBackend::global().enabled()) {
      cd->setComputeOps(get_htp_ops());
      // Model tensors (weights, activations) keep the default CPU allocator.
      // The kernels stage into NPU-accessible buffers per call via
      // sdkl_npu_alloc; routing every model tensor through an NPU allocator
      // exhausts the DMA pool on a full transformer model.
    } else {
      cd->setComputeOps(get_cpu_ops());
    }
#else
    cd->setComputeOps(get_cpu_ops());
#endif
  } catch (...) {
    // HtpBackend::global() or get_htp_ops() threw — degrade to CPU.
    cd->setComputeOps(get_cpu_ops());
  }
}

} // namespace nntrainer

#endif // ENABLE_HEXKL || ENABLE_HEXAGON_CDSP
