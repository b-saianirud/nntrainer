// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 nntrainer contributors
 *
 * @file   unittest_hexagon_repack.cpp
 * @date   18 November 2026
 * @brief  block_q4_0 <-> Hexagon q4x4x2 repack: lossless round trip.
 * @bug    No known bugs except for NYI items
 *
 * @details A host-only check of the layout conversion HexagonComputeOps runs
 * before it uploads a weight to the DSP. Every byte of the weight must
 * survive repack followed by unpack, and the repacked buffer must differ from
 * the plain one (so the test cannot pass on a no-op).
 */

#include <gtest/gtest.h>

#include <hexagon_repack.h>

#include <cstdint>
#include <random>
#include <vector>

namespace {

constexpr size_t kBlockBytes = 18; // fp16 scale + 16 bytes of nibbles per 32

std::vector<uint8_t> randomBlocks(unsigned rows, unsigned cols) {
  std::vector<uint8_t> w(static_cast<size_t>(rows) * (cols / 32) * kBlockBytes);
  std::mt19937 gen(1234);
  for (auto &b : w)
    b = static_cast<uint8_t>(gen());
  return w;
}

} // namespace

TEST(HexagonRepack, RoundTripIsLossless) {
  for (auto [rows, cols] :
       {std::pair<unsigned, unsigned>{64, 256}, {128, 512}, {32, 1024}}) {
    auto plain = randomBlocks(rows, cols);
    std::vector<uint8_t> packed(plain.size()), back(plain.size());

    nntrainer::repack_q4_0_to_htp_q4x4x2(packed.data(), plain.data(),
                                         plain.size(), rows, cols);
    nntrainer::unpack_htp_q4x4x2_to_q4_0(back.data(), packed.data(),
                                         packed.size(), rows, cols);

    EXPECT_NE(packed, plain) << rows << "x" << cols << ": repack is a no-op";
    EXPECT_EQ(back, plain) << rows << "x" << cols;
  }
}

TEST(HexagonRepack, RejectsAWidthThatIsNotWholeQ4x4x2Groups) {
  auto plain = randomBlocks(32, 96);
  std::vector<uint8_t> packed(plain.size());
  EXPECT_THROW(nntrainer::repack_q4_0_to_htp_q4x4x2(packed.data(), plain.data(),
                                                    plain.size(), 32, 96),
               std::invalid_argument);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
