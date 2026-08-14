// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   htp_compute_ops.cpp
 * @date   18 Jun 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  HTP (Hexagon/HMX) ComputeOps entry point.
 *
 * Compiled only when ENABLE_HEXKL is defined.
 *
 * This commit registers the vendor and the accessor its context resolves
 * through; no kernel is wired to the NPU yet, so the table it hands back
 * *is* the CPU table. An engine="htp" model therefore runs correctly,
 * just without acceleration.
 *
 * A dedicated HtpComputeOps subclass arrives with the first accelerated
 * kernel, which is also the commit that makes one necessary: with nothing
 * overridden, a subclass would only duplicate get_cpu_ops(). Actual HexKL
 * calls live in hmx_ops/hexkl_mm.
 */

#ifdef ENABLE_HEXKL

#include <compute_ops.h>
#include <htp_backend.h>

namespace nntrainer {

ComputeOps *get_htp_ops() { return get_cpu_ops(); }

} // namespace nntrainer

#endif // ENABLE_HEXKL
