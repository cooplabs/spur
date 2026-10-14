// Unitree CRC32 for hg LowCmd_ / LowState_ against the golden file written by tools/crc_golden.py (a re-implementation
// of unitree_sdk2_python's CRC and packing): every message's CRC, its bytes, and a field-by-field repack must match.
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "r1rt/crc32.hpp"
#include "r1rt/hg_msgs.hpp"
#include "r1rt/types.h"

namespace {

using r1rt::hg::LowCmd;
using r1rt::hg::LowState;

struct Golden {
  std::string name, kind;
  std::uint32_t crc = 0;
  std::vector<std::uint8_t> bytes;
};

std::vector<Golden> load_golden() {
  std::vector<Golden> out;
  FILE* f = std::fopen(R1RT_CRC_GOLDEN, "r");
  if (f == nullptr) return out;
  char name[64], kind[16], crc[16];
  static char hex[8192];
  while (std::fscanf(f, "%63s %15s %15s %8191s", name, kind, crc, hex) == 4) {
    Golden g;
    g.name = name;
    g.kind = kind;
    g.crc = static_cast<std::uint32_t>(std::strtoul(crc, nullptr, 16));
    const std::size_t n = std::strlen(hex) / 2;
    g.bytes.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      char b[3] = {hex[2 * i], hex[2 * i + 1], 0};
      g.bytes[i] = static_cast<std::uint8_t>(std::strtoul(b, nullptr, 16));
    }
    out.push_back(std::move(g));
  }
  std::fclose(f);
  return out;
}

template <typename Msg>
std::vector<std::uint32_t> words_of(const std::vector<std::uint8_t>& bytes) {
  std::vector<std::uint32_t> w(sizeof(Msg) / 4);
  std::memcpy(w.data(), bytes.data(), sizeof(Msg));
  return w;
}

// Copy every field of `src` into a cleared message (the way a writer fills a LowCmd_ from its own command).
LowCmd repack(const LowCmd& src) {
  LowCmd m;
  r1rt::hg::clear(m);
  m.mode_pr = src.mode_pr;
  m.mode_machine = src.mode_machine;
  for (int i = 0; i < r1rt::hg::kNumMotorSlots; ++i) {
    m.motor_cmd[i].mode = src.motor_cmd[i].mode;
    m.motor_cmd[i].q = src.motor_cmd[i].q;
    m.motor_cmd[i].dq = src.motor_cmd[i].dq;
    m.motor_cmd[i].tau = src.motor_cmd[i].tau;
    m.motor_cmd[i].kp = src.motor_cmd[i].kp;
    m.motor_cmd[i].kd = src.motor_cmd[i].kd;
    m.motor_cmd[i].reserve = src.motor_cmd[i].reserve;
  }
  for (int i = 0; i < 4; ++i) m.reserve[i] = src.reserve[i];
  m.crc = src.crc;
  return m;
}

LowState repack(const LowState& src) {
  LowState m;
  r1rt::hg::clear(m);
  m.version[0] = src.version[0];
  m.version[1] = src.version[1];
  m.mode_pr = src.mode_pr;
  m.mode_machine = src.mode_machine;
  m.tick = src.tick;
  for (int i = 0; i < 4; ++i) m.imu_state.quaternion[i] = src.imu_state.quaternion[i];
  for (int i = 0; i < 3; ++i) {
    m.imu_state.gyroscope[i] = src.imu_state.gyroscope[i];
    m.imu_state.accelerometer[i] = src.imu_state.accelerometer[i];
    m.imu_state.rpy[i] = src.imu_state.rpy[i];
  }
  m.imu_state.temperature = src.imu_state.temperature;
  for (int i = 0; i < r1rt::hg::kNumMotorSlots; ++i) {
    const auto& s = src.motor_state[i];
    auto& d = m.motor_state[i];
    d.mode = s.mode;
    d.q = s.q;
    d.dq = s.dq;
    d.ddq = s.ddq;
    d.tau_est = s.tau_est;
    d.temperature[0] = s.temperature[0];
    d.temperature[1] = s.temperature[1];
    d.vol = s.vol;
    d.sensor[0] = s.sensor[0];
    d.sensor[1] = s.sensor[1];
    d.motorstate = s.motorstate;
    for (int k = 0; k < 4; ++k) d.reserve[k] = s.reserve[k];
  }
  for (int i = 0; i < r1rt::hg::kRemoteBytes; ++i) m.wireless_remote[i] = src.wireless_remote[i];
  for (int i = 0; i < 4; ++i) m.reserve[i] = src.reserve[i];
  m.crc = src.crc;
  return m;
}

template <typename Msg>
void check_message(const Golden& g) {
  SCOPED_TRACE(g.name);
  ASSERT_EQ(g.bytes.size(), sizeof(Msg));
  Msg m;
  std::memcpy(static_cast<void*>(&m), g.bytes.data(), sizeof(Msg));
  EXPECT_EQ(m.crc, g.crc) << "the golden message carries its CRC";
  // 1. the CRC over the raw words, bit-wise port and table version
  const auto w = words_of<Msg>(g.bytes);
  EXPECT_EQ(r1rt::crc32_core_ref(w.data(), r1rt::hg::crc_words<Msg>()), g.crc);
  EXPECT_EQ(r1rt::crc32_words(w.data(), r1rt::hg::crc_words<Msg>()), g.crc);
  // 2. the message API
  EXPECT_EQ(r1rt::hg_crc(m), g.crc);
  // 3. the canonical image is byte-identical to Python's struct.pack (field offsets + zero padding)
  std::vector<std::uint8_t> img(sizeof(Msg));
  r1rt::hg::crc_image(m, img.data());
  EXPECT_EQ(img, g.bytes);
  // 4. field-by-field repack into a cleared message gives the same CRC and bytes
  const Msg r = repack(m);
  EXPECT_EQ(r1rt::hg_crc(r), g.crc);
  EXPECT_EQ(std::memcmp(&r, g.bytes.data(), sizeof(Msg)), 0);
}

TEST(Crc32, GoldenFileLoads) {
  const auto gs = load_golden();
  ASSERT_EQ(gs.size(), 8u) << "missing or truncated " << R1RT_CRC_GOLDEN;
}

TEST(Crc32, GoldenMessages) {
  int n = 0;
  for (const auto& g : load_golden()) {
    if (g.kind == "lowcmd") {
      check_message<LowCmd>(g);
    } else {
      ASSERT_EQ(g.kind, "lowstate");
      check_message<LowState>(g);
    }
    ++n;
    std::printf("[ info ] %-18s %-8s crc %08x OK\n", g.name.c_str(), g.kind.c_str(), g.crc);
  }
  EXPECT_EQ(n, 8);
}

// The R1 stand command built here from the slot map (types.h) must equal Python's lowcmd_r1_stand byte for byte.
TEST(Crc32, StandCommandBuiltInCpp) {
  // POSE_STAND / KP_STIFF / KD_STIFF of r1sim.joints (the golden generator reads them from there).
  const double pose[R1RT_NUM_JOINTS] = {-0.300, 0.025, -0.130, 0.455, -0.140, -0.050, -0.300, -0.025, 0.130,
                                        0.455,  -0.140, 0.050, 0.0,   0.0,    0.22,   0.193,  0.039,  0.76,
                                        0.0,    0.22,   -0.194, -0.033, 0.76, 0.0,    0.0,    0.0};
  const double kp[R1RT_NUM_JOINTS] = {200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 300,
                                      300, 100, 100, 100, 100, 50,  100, 100, 100, 100, 50,  50,  10};
  const double kd[R1RT_NUM_JOINTS] = {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0.1};
  const std::uint32_t slots[R1RT_NUM_JOINTS] = R1RT_HG_SLOT_MAP_INIT;
  LowCmd m;
  r1rt::hg::clear(m);
  m.mode_pr = R1RT_MODE_PR;
  m.mode_machine = R1RT_MODE_MACHINE_R1;
  for (int i = 0; i < R1RT_NUM_JOINTS; ++i) {
    auto& c = m.motor_cmd[slots[i]];
    c.mode = 1;
    c.q = static_cast<float>(pose[i]);
    c.kp = static_cast<float>(kp[i]);
    c.kd = static_cast<float>(kd[i]);
  }
  m.crc = r1rt::hg_crc(m);
  for (const auto& g : load_golden()) {
    if (g.name != "lowcmd_r1_stand") continue;
    EXPECT_EQ(m.crc, g.crc);
    EXPECT_EQ(std::memcmp(&m, g.bytes.data(), sizeof(LowCmd)), 0);
    return;
  }
  FAIL() << "lowcmd_r1_stand not in the golden file";
}

// Stale padding bytes must not change the CRC (the canonical image zeroes them); a naive CRC over the raw memory
// would differ - that is why hg_crc packs.
TEST(Crc32, PaddingDoesNotChangeCrc) {
  LowCmd m;
  r1rt::hg::clear(m);
  m.mode_machine = 1;
  m.motor_cmd[3].mode = 1;
  m.motor_cmd[3].q = 0.5f;
  const std::uint32_t clean = r1rt::hg_crc(m);
  auto* raw = reinterpret_cast<std::uint8_t*>(&m);
  raw[2] = 0xAB;  // LowCmd padding after mode_machine
  raw[offsetof(LowCmd, motor_cmd) + 1] = 0xCD;  // MotorCmd padding after mode
  EXPECT_EQ(r1rt::hg_crc(m), clean);
  std::uint32_t w[sizeof(LowCmd) / 4];
  std::memcpy(w, &m, sizeof(LowCmd));
  EXPECT_NE(r1rt::crc32_core_ref(w, r1rt::hg::crc_words<LowCmd>()), clean);
}

TEST(Crc32, Mpeg2CheckValue) {
  const char* s = "123456789";
  EXPECT_EQ(r1rt::crc32_mpeg2_bytes(reinterpret_cast<const std::uint8_t*>(s), 9), 0x0376E6E7u);
}

// The table version equals the bit-wise port on random word streams (lengths 0..300).
TEST(Crc32, TableEqualsBitwise) {
  std::mt19937 rng(12345);
  std::vector<std::uint32_t> w(300);
  for (int len = 0; len <= 300; len += 7) {
    for (auto& x : w) x = static_cast<std::uint32_t>(rng());
    EXPECT_EQ(r1rt::crc32_words(w.data(), static_cast<std::size_t>(len)),
              r1rt::crc32_core_ref(w.data(), static_cast<std::size_t>(len)))
        << len;
  }
  EXPECT_EQ(r1rt::crc32_words(nullptr, 0), 0xFFFFFFFFu);
}

// Any single-bit change in the covered words changes the CRC; the crc word itself is not covered.
TEST(Crc32, CoverageOfFields) {
  LowCmd m;
  r1rt::hg::clear(m);
  const std::uint32_t base = r1rt::hg_crc(m);
  m.motor_cmd[34].reserve = 1;  // the last covered field before reserve[]
  EXPECT_NE(r1rt::hg_crc(m), base);
  m.motor_cmd[34].reserve = 0;
  m.reserve[3] = 1;  // the last covered word
  EXPECT_NE(r1rt::hg_crc(m), base);
  m.reserve[3] = 0;
  m.crc = 0xDEADBEEF;  // not covered
  EXPECT_EQ(r1rt::hg_crc(m), base);
}

}  // namespace
