// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 nntrainer contributors
 *
 * @file   unittest_compute_ops_frozen_fc.cpp
 * @date   18 November 2026
 * @brief  ComputeOps::fc_frozen / ::fc_frozen_dgrad -- the contract every
 *         backend table is measured against.
 * @see    https://github.com/nntrainer/nntrainer
 * @bug    No known bugs except for NYI items
 *
 * @details LoRA training multiplies against a frozen Q4_0 base weight. The
 * ops must dequantize the weight exactly and multiply at the activation
 * precision (no Q8_0 activation quantization), so the expectation is the
 * dequantized weight run through the ordinary FP32 dot, element for element.
 */

#include <gtest/gtest.h>

#include <compute_ops.h>
#include <cpu_backend.h>
#include <quantizer.h>
#include <tensor.h>

namespace {

constexpr unsigned int B = 2, H = 6, K = 128, N = 64;

nntrainer::Tensor makeQ4Weight(nntrainer::Tensor &w_fp32) {
  w_fp32 = nntrainer::Tensor(
    1, 1, K, N, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32});
  w_fp32.setRandNormal(0.0f, 0.05f);
  return nntrainer::Quantization::createQuantizer(nntrainer::QScheme::Q4_0)
    ->quantize(w_fp32, nntrainer::Tdatatype::Q4_0);
}

nntrainer::Tensor dequant(const nntrainer::Tensor &q) {
  return nntrainer::Quantization::createQuantizer(nntrainer::QScheme::Q4_0)
    ->dequantize(q, nntrainer::Tdatatype::FP32);
}

} // namespace

/**
 * @brief fc_frozen multiplies by the exactly dequantized Q4_0 weight
 */
TEST(ComputeOpsFrozenFc, forwardUsesDequantizedWeight_p) {
  nntrainer::init_backend();
  nntrainer::Tensor w_fp32;
  nntrainer::Tensor w_q4 = makeQ4Weight(w_fp32);

  nntrainer::Tensor x(B, 1, H, K,
                      {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32});
  x.setRandNormal(0.0f, 1.0f);
  nntrainer::Tensor y(B, 1, H, N,
                      {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32});

  nntrainer::get_cpu_ops()->fc_frozen(x, w_q4, y);

  nntrainer::Tensor ref = x.dot(dequant(w_q4));
  for (unsigned int b = 0; b < B; ++b)
    for (unsigned int h = 0; h < H; ++h)
      for (unsigned int n = 0; n < N; ++n)
        ASSERT_FLOAT_EQ(y.getValue(b, 0, h, n), ref.getValue(b, 0, h, n))
          << "b=" << b << " h=" << h << " n=" << n;
}

/**
 * @brief fc_frozen_dgrad is dy * dequantize(W)^T for every batch
 */
TEST(ComputeOpsFrozenFc, dgradUsesDequantizedWeight_p) {
  nntrainer::init_backend();
  nntrainer::Tensor w_fp32;
  nntrainer::Tensor w_q4 = makeQ4Weight(w_fp32);

  nntrainer::Tensor dy(B, 1, H, N,
                       {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32});
  dy.setRandNormal(0.0f, 1.0f);
  nntrainer::Tensor dx(B, 1, H, K,
                       {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32});

  nntrainer::get_cpu_ops()->fc_frozen_dgrad(dy, w_q4, dx);

  // dx = dy * W^T, W being K x N
  nntrainer::Tensor ref = dy.dot(dequant(w_q4), false, true);
  for (unsigned int b = 0; b < B; ++b)
    for (unsigned int h = 0; h < H; ++h)
      for (unsigned int k = 0; k < K; ++k)
        ASSERT_FLOAT_EQ(dx.getValue(b, 0, h, k), ref.getValue(b, 0, h, k))
          << "b=" << b << " h=" << h << " k=" << k;
}

/**
 * @brief a float weight is used as it is
 */
TEST(ComputeOpsFrozenFc, floatWeightPassesThrough_p) {
  nntrainer::init_backend();
  nntrainer::Tensor w(1, 1, K, N,
                      {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32});
  w.setRandNormal(0.0f, 0.05f);
  nntrainer::Tensor x(B, 1, H, K,
                      {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32});
  x.setRandNormal(0.0f, 1.0f);
  nntrainer::Tensor y(B, 1, H, N,
                      {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::FP32});

  nntrainer::get_cpu_ops()->fc_frozen(x, w, y);

  nntrainer::Tensor ref = x.dot(w);
  for (unsigned int i = 0; i < B * H * N; ++i)
    ASSERT_FLOAT_EQ(y.getData<float>()[i], ref.getData<float>()[i]);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
