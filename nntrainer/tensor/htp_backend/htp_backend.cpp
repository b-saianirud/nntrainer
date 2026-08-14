// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   htp_backend.cpp
 * @date   18 Jun 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  HTP backend lifecycle implementation (HexKL sdkl.h).
 */

#ifdef ENABLE_HEXKL

#include <htp_backend.h>

#include <nntrainer_log.h>

// Hexagon SDK headers: remote.h defines CDSP_DOMAIN_ID used by sdkl.h.
#include <sdkl_compat.h>

#include <atomic>

namespace nntrainer {

namespace {
std::atomic<bool> g_npu_alive{false};
} // namespace

bool npuAlive() { return g_npu_alive.load(std::memory_order_acquire); }

HtpBackend &HtpBackend::global() {
  static HtpBackend instance;
  return instance;
}

HtpBackend::HtpBackend() {
  domain_ = CDSP_DOMAIN_ID;

  int err = sdkl_npu_initialize(domain_, nullptr, nullptr);
  if (err != 0) {
    // Graceful disable: leave enabled_ = false so supports_*() reports
    // false and callers fall back to CPU. Not fatal.
    ml_logw("HexKL NPU init failed (err=%d); HTP backend disabled, "
            "falling back to CPU.",
            err);
    return;
  }

  enabled_ = true;
  g_npu_alive.store(true, std::memory_order_release);

  char version[SDKL_VERSION_STR_LEN] = {0};
  if (sdkl_npu_get_version(domain_, version) == 0) {
    ml_logi("HexKL NPU initialized (domain=%d, version=%s)", domain_, version);
  } else {
    ml_logi("HexKL NPU initialized (domain=%d)", domain_);
  }
}

HtpBackend::~HtpBackend() {
  if (enabled_) {
    g_npu_alive.store(false, std::memory_order_release);
    sdkl_npu_finalize(domain_);
    enabled_ = false;
  }
}

} // namespace nntrainer

#endif // ENABLE_HEXKL
