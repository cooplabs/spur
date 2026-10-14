// r1rt/hg_msgs.hpp - Unitree "hg" low-level messages (R1 / G1 / H1-2 family) as plain C++ structs (C++17).
//
// Field order and types: unitree_sdk2 @63096d0 include/unitree/idl/hg/{MotorCmd_,LowCmd_,MotorState_,IMUState_,
// LowState_,PressSensorState_}.hpp (CycloneDDS idlcxx output of Unitree's hg IDL; copies + sha256 in rt/idl/).
//
// THIS IS THE CRC LAYOUT, NOT THE WIRE LAYOUT. Unitree computes the LowCmd_ / LowState_ CRC over the in-memory
// image of the C++ message (natural alignment, little-endian; the padding bytes zero):
//   unitree_sdk2 examples: crc = crc32_core((uint32_t*)&msg, (sizeof(msg) >> 2) - 1);
//   unitree_sdk2_python utils/crc.py packs the same image with struct formats '<2B2x' 'B3x5fI'*35 '5I' (LowCmd_,
//   1004 B) and '<2I2B2xI' '13fh2x' 'B3x4f2hf7I'*35 '40B5I' (LowState_, 2092 B).
// The DDS wire encoding (XCDR) aligns members differently (e.g. motor_cmd[0].mode sits at byte 2 on the wire, at
// byte 4 here); serialisation belongs to the DDS layer (the CycloneDDS types generated from rt/idl/src, built with
// -DR1RT_WITH_CYCLONEDDS=ON; their classes have exactly this layout - test_hg_dds.cpp), which copies the fields into
// these structs for the CRC. The static_asserts below pin every offset the CRC depends on.
//
// HandCmd_ / HandState_ (Dex3) carry variable-length sequences (motor_cmd, motor_state, press_sensor_state) and no
// CRC: only their fixed-size element types are here; the Dex3-1 instances are 7 motors and 9 pads.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#if !defined(__BYTE_ORDER__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "hg_msgs.hpp: the CRC image is defined for little-endian hosts (the R1's Orin and the Mac are)"
#endif

namespace r1rt::hg {

inline constexpr int kNumMotorSlots = 35;   // LowCmd_.motor_cmd / LowState_.motor_state length
inline constexpr int kRemoteBytes = 40;     // LowState_.wireless_remote
inline constexpr int kDex3Motors = 7;       // HandCmd_.motor_cmd length for a Dex3-1 hand
inline constexpr int kDex3Pads = 9;         // HandState_.press_sensor_state length for a Dex3-1 hand
inline constexpr int kPadSlots = 12;        // PressSensorState_.pressure / temperature length

struct MotorCmd {        // unitree_hg::msg::dds_::MotorCmd_
  std::uint8_t mode;     // 0 = off (damped by the board?), 1 = enabled (FOC)
  float q;               // rad
  float dq;              // rad/s
  float tau;             // N m feed-forward
  float kp;
  float kd;
  std::uint32_t reserve;
};

struct LowCmd {          // unitree_hg::msg::dds_::LowCmd_
  std::uint8_t mode_pr;       // 0 = ankles in pitch/roll (PR) mode
  std::uint8_t mode_machine;  // must echo LowState_.mode_machine (R1 26-DoF: 1)
  MotorCmd motor_cmd[kNumMotorSlots];
  std::uint32_t reserve[4];
  std::uint32_t crc;          // over every 32-bit word before it
};

struct MotorState {      // unitree_hg::msg::dds_::MotorState_
  std::uint8_t mode;
  float q;
  float dq;
  float ddq;
  float tau_est;
  std::int16_t temperature[2];
  float vol;
  std::uint32_t sensor[2];
  std::uint32_t motorstate;   // error word
  std::uint32_t reserve[4];
};

struct IMUState {        // unitree_hg::msg::dds_::IMUState_
  float quaternion[4];   // w x y z
  float gyroscope[3];
  float accelerometer[3];
  float rpy[3];
  std::int16_t temperature;
};

struct LowState {        // unitree_hg::msg::dds_::LowState_
  std::uint32_t version[2];
  std::uint8_t mode_pr;
  std::uint8_t mode_machine;
  std::uint32_t tick;    // board tick (~1 kHz)
  IMUState imu_state;
  MotorState motor_state[kNumMotorSlots];
  std::uint8_t wireless_remote[kRemoteBytes];
  std::uint32_t reserve[4];
  std::uint32_t crc;
};

struct PressSensorState {  // unitree_hg::msg::dds_::PressSensorState_ (one Dex3 pad: 3 x 4 slots)
  float pressure[kPadSlots];
  float temperature[kPadSlots];
  std::uint32_t lost;
  std::uint32_t reserve;
};

// Layout pins (= unitree_sdk2_python's struct formats above; any change breaks the CRC against the robot).
static_assert(std::is_trivially_copyable_v<LowCmd> && std::is_standard_layout_v<LowCmd>, "POD");
static_assert(std::is_trivially_copyable_v<LowState> && std::is_standard_layout_v<LowState>, "POD");
static_assert(sizeof(MotorCmd) == 28 && offsetof(MotorCmd, q) == 4 && offsetof(MotorCmd, reserve) == 24, "MotorCmd_");
static_assert(sizeof(LowCmd) == 1004, "LowCmd_ = 1004 B ('<2B2x' + 'B3x5fI'*35 + '5I')");
static_assert(offsetof(LowCmd, motor_cmd) == 4 && offsetof(LowCmd, reserve) == 984 && offsetof(LowCmd, crc) == 1000,
              "LowCmd_ offsets");
static_assert(sizeof(MotorState) == 56 && offsetof(MotorState, temperature) == 20 && offsetof(MotorState, vol) == 24 &&
                  offsetof(MotorState, motorstate) == 36 && offsetof(MotorState, reserve) == 40,
              "MotorState_ ('B3x4f2hf7I')");
static_assert(sizeof(IMUState) == 56 && offsetof(IMUState, temperature) == 52, "IMUState_ ('13fh2x')");
static_assert(sizeof(LowState) == 2092, "LowState_ = 2092 B");
static_assert(offsetof(LowState, tick) == 12 && offsetof(LowState, imu_state) == 16 &&
                  offsetof(LowState, motor_state) == 72 && offsetof(LowState, wireless_remote) == 2032 &&
                  offsetof(LowState, reserve) == 2072 && offsetof(LowState, crc) == 2088,
              "LowState_ offsets");
static_assert(sizeof(PressSensorState) == 104, "PressSensorState_");

// Words covered by the CRC: all but the trailing crc word.
template <typename Msg>
constexpr std::size_t crc_words() noexcept {
  static_assert(sizeof(Msg) % 4 == 0 && offsetof(Msg, crc) == sizeof(Msg) - 4, "crc must be the last word");
  return sizeof(Msg) / 4 - 1;
}

// Zero every byte, padding included (the CRC image needs zero padding; value-initialisation does not promise it).
template <typename Msg>
inline void clear(Msg& m) noexcept {
  std::memset(static_cast<void*>(&m), 0, sizeof(Msg));
}

// The canonical CRC image of a message: a zeroed buffer with every field copied in at its offset (so stale
// padding in `m` can never change the CRC). `out` must hold sizeof(Msg) bytes. No allocation.
void crc_image(const LowCmd& m, std::uint8_t* out) noexcept;
void crc_image(const LowState& m, std::uint8_t* out) noexcept;

}  // namespace r1rt::hg
