// Unitree's CRC32 and the hg CRC images (crc32.hpp, hg_msgs.hpp).
#include "r1rt/crc32.hpp"

#include <array>
#include <cstring>

namespace r1rt {

namespace {

constexpr std::uint32_t kPoly = 0x04C11DB7u;

constexpr std::array<std::uint32_t, 256> make_table() {
  std::array<std::uint32_t, 256> t{};
  for (std::uint32_t i = 0; i < 256; ++i) {
    std::uint32_t c = i << 24;
    for (int b = 0; b < 8; ++b) c = (c & 0x80000000u) ? (c << 1) ^ kPoly : (c << 1);
    t[i] = c;
  }
  return t;
}

constexpr std::array<std::uint32_t, 256> kTable = make_table();

inline std::uint32_t step_byte(std::uint32_t crc, std::uint8_t byte) noexcept {
  return (crc << 8) ^ kTable[((crc >> 24) ^ byte) & 0xFFu];
}

template <typename T>
inline void put(std::uint8_t* out, std::size_t offset, const T& v) noexcept {
  std::memcpy(out + offset, &v, sizeof(T));
}

}  // namespace

// Port of unitree_sdk2 crc.h crc32_core (BSD-3-Clause, Copyright (c) Unitree Robotics): same variables, same steps.
std::uint32_t crc32_core_ref(const std::uint32_t* words, std::size_t n) noexcept {
  std::uint32_t xbit = 0;
  std::uint32_t data = 0;
  std::uint32_t crc32 = 0xFFFFFFFFu;
  for (std::size_t i = 0; i < n; ++i) {
    xbit = 1u << 31;
    data = words[i];
    for (std::uint32_t bits = 0; bits < 32; ++bits) {
      if (crc32 & 0x80000000u) {
        crc32 <<= 1;
        crc32 ^= kPoly;
      } else {
        crc32 <<= 1;
      }
      if (data & xbit) crc32 ^= kPoly;
      xbit >>= 1;
    }
  }
  return crc32;
}

std::uint32_t crc32_words(const std::uint32_t* words, std::size_t n) noexcept {
  std::uint32_t crc = 0xFFFFFFFFu;
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint32_t w = words[i];
    crc = step_byte(crc, static_cast<std::uint8_t>(w >> 24));
    crc = step_byte(crc, static_cast<std::uint8_t>(w >> 16));
    crc = step_byte(crc, static_cast<std::uint8_t>(w >> 8));
    crc = step_byte(crc, static_cast<std::uint8_t>(w));
  }
  return crc;
}

std::uint32_t crc32_mpeg2_bytes(const std::uint8_t* data, std::size_t n) noexcept {
  std::uint32_t crc = 0xFFFFFFFFu;
  for (std::size_t i = 0; i < n; ++i) crc = step_byte(crc, data[i]);
  return crc;
}

namespace hg {

void crc_image(const LowCmd& m, std::uint8_t* out) noexcept {
  std::memset(out, 0, sizeof(LowCmd));
  put(out, offsetof(LowCmd, mode_pr), m.mode_pr);
  put(out, offsetof(LowCmd, mode_machine), m.mode_machine);
  for (int i = 0; i < kNumMotorSlots; ++i) {
    const std::size_t b = offsetof(LowCmd, motor_cmd) + static_cast<std::size_t>(i) * sizeof(MotorCmd);
    const MotorCmd& c = m.motor_cmd[i];
    put(out, b + offsetof(MotorCmd, mode), c.mode);
    put(out, b + offsetof(MotorCmd, q), c.q);
    put(out, b + offsetof(MotorCmd, dq), c.dq);
    put(out, b + offsetof(MotorCmd, tau), c.tau);
    put(out, b + offsetof(MotorCmd, kp), c.kp);
    put(out, b + offsetof(MotorCmd, kd), c.kd);
    put(out, b + offsetof(MotorCmd, reserve), c.reserve);
  }
  put(out, offsetof(LowCmd, reserve), m.reserve);
  put(out, offsetof(LowCmd, crc), m.crc);
}

void crc_image(const LowState& m, std::uint8_t* out) noexcept {
  std::memset(out, 0, sizeof(LowState));
  put(out, offsetof(LowState, version), m.version);
  put(out, offsetof(LowState, mode_pr), m.mode_pr);
  put(out, offsetof(LowState, mode_machine), m.mode_machine);
  put(out, offsetof(LowState, tick), m.tick);
  const std::size_t ib = offsetof(LowState, imu_state);
  put(out, ib + offsetof(IMUState, quaternion), m.imu_state.quaternion);
  put(out, ib + offsetof(IMUState, gyroscope), m.imu_state.gyroscope);
  put(out, ib + offsetof(IMUState, accelerometer), m.imu_state.accelerometer);
  put(out, ib + offsetof(IMUState, rpy), m.imu_state.rpy);
  put(out, ib + offsetof(IMUState, temperature), m.imu_state.temperature);
  for (int i = 0; i < kNumMotorSlots; ++i) {
    const std::size_t b = offsetof(LowState, motor_state) + static_cast<std::size_t>(i) * sizeof(MotorState);
    const MotorState& s = m.motor_state[i];
    put(out, b + offsetof(MotorState, mode), s.mode);
    put(out, b + offsetof(MotorState, q), s.q);
    put(out, b + offsetof(MotorState, dq), s.dq);
    put(out, b + offsetof(MotorState, ddq), s.ddq);
    put(out, b + offsetof(MotorState, tau_est), s.tau_est);
    put(out, b + offsetof(MotorState, temperature), s.temperature);
    put(out, b + offsetof(MotorState, vol), s.vol);
    put(out, b + offsetof(MotorState, sensor), s.sensor);
    put(out, b + offsetof(MotorState, motorstate), s.motorstate);
    put(out, b + offsetof(MotorState, reserve), s.reserve);
  }
  put(out, offsetof(LowState, wireless_remote), m.wireless_remote);
  put(out, offsetof(LowState, reserve), m.reserve);
  put(out, offsetof(LowState, crc), m.crc);
}

}  // namespace hg

namespace {

template <typename Msg>
std::uint32_t crc_of(const Msg& m) noexcept {
  std::uint32_t words[sizeof(Msg) / 4];
  hg::crc_image(m, reinterpret_cast<std::uint8_t*>(words));
  return crc32_words(words, hg::crc_words<Msg>());
}

}  // namespace

std::uint32_t hg_crc(const hg::LowCmd& m) noexcept { return crc_of(m); }
std::uint32_t hg_crc(const hg::LowState& m) noexcept { return crc_of(m); }

}  // namespace r1rt
