// The processes' DDS layer (dds_io.hpp): the lowstate conversion (slot map, IMU, Unitree's CRC -> crc_ok; a corrupted
// message keeps crc_ok = 0 so the core counts it and does not use it), our command -> LowCmd_ (= r1rt_fill_hg_lowcmd's
// image, CRC) and back, and over a real CycloneDDS loopback: the latest lowstate reaches the writer's seqlock, and the
// built-in's rt/lowcmd is passed on from FOREIGN writers only (our r1rt / r1guard writers carry user_data "r1mc3:").
#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "dds/dds.hpp"
#include "r1rt/crc32.hpp"
#include "r1rt/dds_io.hpp"
#include "r1rt/hg_msgs.hpp"
#include "r1rt/rt_thread.hpp"
#include "unitree/idl/hg/LowCmd_.hpp"
#include "unitree/idl/hg/LowState_.hpp"

namespace {
namespace uhg = unitree_hg::msg::dds_;
using namespace r1rt;

constexpr std::uint32_t kSlot[R1RT_NUM_JOINTS] = R1RT_HG_SLOT_MAP_INIT;
constexpr int kDomain = 57;  // proc's range 50-59 (CONTRACT.md)

void loopback_env() {
  // The loopback is "lo0" on macOS, "lo" on Linux (CycloneDDS refuses a domain whose named interface does not exist).
#if defined(__APPLE__)
#define R1RT_TEST_LOOPBACK "lo0"
#else
#define R1RT_TEST_LOOPBACK "lo"
#endif
  setenv("CYCLONEDDS_URI",
         "<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"" R1RT_TEST_LOOPBACK "\"/></Interfaces>"
         "<AllowMulticast>false</AllowMulticast></General><Discovery><ParticipantIndex>auto</ParticipantIndex>"
         "<Peers><Peer address=\"127.0.0.1\"/></Peers></Discovery></Domain></CycloneDDS>",
         1);
}

uhg::LowState_ sample_state() {
  uhg::LowState_ m;
  m.mode_machine(1);
  m.tick(4242);
  for (std::size_t k = 0; k < 35; ++k) {
    auto& s = m.motor_state()[k];
    s.q(0.01f * static_cast<float>(k));
    s.dq(-0.001f * static_cast<float>(k));
    s.tau_est(0.5f);
    s.temperature()[0] = static_cast<int16_t>(30 + k);
    s.temperature()[1] = 31;
    s.vol(48.0f);
    s.motorstate(k == 22 ? 0x4u : 0u);
  }
  m.imu_state().quaternion() = {0.999f, 0.01f, -0.02f, 0.0f};
  m.imu_state().gyroscope() = {0.1f, 0.2f, 0.3f};
  m.imu_state().accelerometer() = {0.0f, 0.1f, 9.8f};
  m.wireless_remote()[3] = 0x20;
  m.crc(dds_io::lowstate_crc(m));
  return m;
}

TEST(ProcDds, LowstateConversionMapsSlotsAndChecksTheCrc) {
  const uhg::LowState_ m = sample_state();
  r1rt_lowstate_in in;
  dds_io::to_lowstate_in(m, 123, &in);
  EXPECT_EQ(in.crc_ok, 1);
  EXPECT_EQ(in.t_rx_ns, 123);
  EXPECT_EQ(in.board_tick, 4242u);
  EXPECT_EQ(in.mode_machine, 1);
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    EXPECT_FLOAT_EQ(static_cast<float>(in.q[j]), 0.01f * static_cast<float>(kSlot[j])) << j;
    EXPECT_EQ(in.temperature[j][0], static_cast<int16_t>(30 + kSlot[j]));
  }
  EXPECT_EQ(in.motor_state[19], 0x4u);  // joint 19 sits in hg slot 22
  EXPECT_FLOAT_EQ(static_cast<float>(in.imu_quat[0]), 0.999f);
  EXPECT_FLOAT_EQ(static_cast<float>(in.imu_acc[2]), 9.8f);
  EXPECT_EQ(in.wireless_remote[3], 0x20);
  // The same CRC as r1rt::hg_crc over the canonical image (the board's algorithm, golden-tested in r1rt_io_tests).
  uhg::LowState_ bad = m;
  bad.motor_state()[7].q(bad.motor_state()[7].q() + 1e-3f);  // one field changed in transit
  dds_io::to_lowstate_in(bad, 124, &in);
  EXPECT_EQ(in.crc_ok, 0);
}

TEST(ProcDds, CommandToDdsIsTheFillImageAndRoundTrips) {
  r1rt_lowcmd_out c;
  std::memset(&c, 0, sizeof c);
  c.publish = 1;
  c.mode_machine = 1;
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    c.enable[j] = 1;
    c.q[j] = 0.1 * j;
    c.dq[j] = -0.01 * j;
    c.kp[j] = 84.302;
    c.kd[j] = 5.367;
    c.tau_ff[j] = 0.25;
  }
  uhg::LowCmd_ d;
  const std::uint32_t crc = dds_io::to_dds(c, &d);
  hg::LowCmd img;
  EXPECT_EQ(r1rt_fill_hg_lowcmd(&c, &img, sizeof img), crc);
  EXPECT_EQ(d.crc(), crc);
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) EXPECT_FLOAT_EQ(d.motor_cmd()[kSlot[j]].q(), static_cast<float>(c.q[j]));
  EXPECT_EQ(d.motor_cmd()[14].mode(), 0);  // unused slot
  r1rt_lowcmd_out back;
  EXPECT_TRUE(dds_io::to_lowcmd_out(d, &back));
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    EXPECT_FLOAT_EQ(static_cast<float>(back.kp[j]), 84.302f);
    EXPECT_EQ(back.enable[j], 1);
  }
  d.motor_cmd()[3].kp(0.0f);  // corrupted: the CRC no longer matches
  EXPECT_FALSE(dds_io::to_lowcmd_out(d, &back));
}

struct Seen {
  std::atomic<int> n{0};
  std::atomic<float> kp0{0.0f};
};
void on_builtin(void* ctx, const r1rt_lowcmd_out& cmd, std::int64_t) {
  auto* s = static_cast<Seen*>(ctx);
  s->kp0.store(static_cast<float>(cmd.kp[0]));
  s->n.fetch_add(1);
}

TEST(ProcDds, LoopbackLatestStateAndOnlyForeignLowcmdIsTheBuiltin) {
  loopback_env();
  Seen seen;
  dds_io::BusOptions a;
  a.domain = kDomain;
  a.role = "r1rt";
  a.read_builtin = true;
  a.on_builtin = on_builtin;
  a.builtin_ctx = &seen;
  dds_io::Bus core(a);
  ASSERT_TRUE(core.ok()) << core.error();
  dds_io::BusOptions b;
  b.domain = kDomain;
  b.role = "r1guard";
  dds_io::Bus guard(b);
  ASSERT_TRUE(guard.ok()) << guard.error();
  std::atomic<bool> stop{false};
  std::thread rx([&] { core.run_receive(stop); });

  // a foreign participant: the robot's state + Unitree's built-in controller (no user_data)
  dds::domain::DomainParticipant dp(kDomain);
  dds::topic::Topic<uhg::LowState_> ts(dp, dds_io::kTopicLowState);
  dds::topic::Topic<uhg::LowCmd_> tc(dp, dds_io::kTopicLowCmd);
  dds::pub::Publisher pub(dp);
  dds::pub::DataWriter<uhg::LowState_> ws(pub, ts);
  dds::pub::DataWriter<uhg::LowCmd_> wc(pub, tc);

  r1rt_lowcmd_out ours;
  std::memset(&ours, 0, sizeof ours);
  ours.publish = 1;
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) ours.kp[j] = 31.0;
  uhg::LowCmd_ foreign;
  r1rt_lowcmd_out f;
  std::memset(&f, 0, sizeof f);
  f.publish = 1;
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) f.kp[j] = 200.0;  // Unitree's stand lock
  dds_io::to_dds(f, &foreign);

  const std::int64_t t_end = proc::now_ns() + 5'000'000'000;
  r1rt_lowstate_in in;
  std::uint64_t cursor = 0;
  bool got_state = false;
  while (proc::now_ns() < t_end && !(got_state && seen.n.load() >= 5 && core.counters().own_lowcmd_rx.load() >= 5)) {
    ws.write(sample_state());
    wc.write(foreign);
    guard.write(ours);  // r1guard's writer: ours, must never be taken for the built-in
    core.write(ours);   // r1rt's own writer: ours too
    proc::sleep_for(10'000'000);
    if (core.latest(&in, &cursor)) got_state = true;
  }
  stop.store(true);
  rx.join();
  ASSERT_TRUE(got_state) << "no lowstate reached the seqlock over the loopback";
  EXPECT_EQ(in.crc_ok, 1);
  EXPECT_EQ(in.board_tick, 4242u);
  EXPECT_GE(seen.n.load(), 5);
  EXPECT_FLOAT_EQ(seen.kp0.load(), 200.0f) << "only the foreign (built-in) command may reach the core";
  EXPECT_GE(core.counters().own_lowcmd_rx.load(), 5u);
  EXPECT_EQ(core.counters().builtin_crc_bad.load(), 0u);
  EXPECT_GE(core.counters().foreign_writers.load(), 1u);
  EXPECT_FALSE(core.latest(&in, &cursor)) << "no new state since the last read";
}

}  // namespace
