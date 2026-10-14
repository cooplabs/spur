// hg message structs: sizes and offsets equal the struct formats unitree_sdk2_python packs for the CRC (the
// static_asserts in hg_msgs.hpp pin them at compile time; this test documents them at run time too).
#include <gtest/gtest.h>

#include <cstddef>

#include "r1rt/hg_msgs.hpp"
#include "r1rt/types.h"

namespace {

using namespace r1rt::hg;

TEST(HgMsgs, Sizes) {
  EXPECT_EQ(sizeof(MotorCmd), 28u);
  EXPECT_EQ(sizeof(LowCmd), 1004u);    // '<2B2x' + 'B3x5fI'*35 + '5I'
  EXPECT_EQ(sizeof(MotorState), 56u);  // 'B3x4f2hf7I'
  EXPECT_EQ(sizeof(IMUState), 56u);    // '13fh2x'
  EXPECT_EQ(sizeof(LowState), 2092u);  // '<2I2B2xI' + '13fh2x' + 'B3x4f2hf7I'*35 + '40B5I'
  EXPECT_EQ(sizeof(PressSensorState), 104u);
  EXPECT_EQ(crc_words<LowCmd>(), 250u);
  EXPECT_EQ(crc_words<LowState>(), 522u);
}

TEST(HgMsgs, SlotCountsMatchTheCore) {
  EXPECT_EQ(kNumMotorSlots, R1RT_HG_NUM_SLOTS);
  EXPECT_EQ(kRemoteBytes, R1RT_REMOTE_BYTES);
  const std::uint32_t slots[R1RT_NUM_JOINTS] = R1RT_HG_SLOT_MAP_INIT;
  for (std::uint32_t s : slots) EXPECT_LT(s, static_cast<std::uint32_t>(kNumMotorSlots));
}

TEST(HgMsgs, ClearZeroesPadding) {
  LowCmd m;
  auto* raw = reinterpret_cast<unsigned char*>(&m);
  for (std::size_t i = 0; i < sizeof(m); ++i) raw[i] = 0xFF;
  clear(m);
  for (std::size_t i = 0; i < sizeof(m); ++i) ASSERT_EQ(raw[i], 0u) << i;
}

}  // namespace
