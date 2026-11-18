// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   hexagon_compute_ops.h
 * @see    https://github.com/nntrainer/nntrainer
 * @brief  Accessor for the Hexagon cDSP ComputeOps table. Mirrors
 * get_cl_ops()/get_cpu_ops() - see compute_ops.h for the dispatch contract.
 */

#ifndef __HEXAGON_COMPUTE_OPS_H__
#define __HEXAGON_COMPUTE_OPS_H__

namespace nntrainer {

class ComputeOps;

/**
 * @brief Get the Hexagon cDSP ComputeOps table.
 *
 * The table derives from the CPU table, so every op it does not override runs
 * on the host exactly as under engine=cpu. It overrides the matmuls that
 * LoRA training spends its time in (frozen-weight forward / input gradient,
 * Q4_0 GEMM, large FP32 GEMM) and runs them on the DSP through the
 * nntr_htp_bridge_* entry points of libggml-hexagon.so, which it loads at
 * run time; without that library every op falls back to the CPU.
 *
 * @return ComputeOps* pointer to the ops table
 */
ComputeOps *get_hexagon_ops();

} // namespace nntrainer

#endif /* __HEXAGON_COMPUTE_OPS_H__ */
