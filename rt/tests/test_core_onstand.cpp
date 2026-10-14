// ON_STAND in r1rt::Core (session 5, ABI 5; DESIGN §3.5): the robot hanging on its stand. R1RT_REQ_ON_STAND from
// MONITOR (the takeover's gates that make sense for a hanging robot; no policy needed; the built-in fresh but not
// necessarily stiff), MIRROR -> RELEASED -> BLEND into a stiff hold of the pose measured at RELEASED -> ON_STAND; then
// per-joint L0 for legs + waist through the upper_cmd ring's leg / waist groups (HOLD, STREAM with TTL, TRAJECTORY),
// the arms through the upper-body executor; STOP decelerates every group to HOLD; only the operator's DAMP leaves it.
// Also R1RT_FAULT_GUARD_ABSENT wired to r1guard's heartbeat (an alarm only). Synthetic state stream; the simulated
// hanging robot: tests/integration/test_v3_core_onstand.py.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>

#include "r1rt/arm_model.hpp"
#include "r1rt/shm_layout.h"
#include "r1rt/step_api.h"
#include "test_core_rig.hpp"

namespace {

constexpr std::int64_t kBlendNs = 1000 * kMs;
constexpr double kOnStandKp = 150.0;
constexpr double kOnStandKd = 4.0;

void on_stand_gains(r1rt_config& c) {
  for (int j = 0; j < 14; ++j) {
    c.on_stand_kp[j] = kOnStandKp;
    c.on_stand_kd[j] = kOnStandKd;
  }
}

// Unitree's command on the stand: damping (leg kp 0) or a stand lock, every motor enabled, `dev` from the pose.
r1rt_lowcmd_out builtin(const r1rt_lowstate_in& st, double leg_kp, double dev = 0.01) {
  r1rt_lowcmd_out b{};
  b.publish = 1;
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    b.enable[j] = 1;
    b.q[j] = st.q[j] + dev;
    b.kp[j] = j < 12 ? leg_kp : 0.0;
    b.kd[j] = 3.0;
  }
  return b;
}

std::int32_t req(Rig& r, std::int32_t q) { return r1rt_core_request(r.core, q, r.t); }

std::int32_t push(Rig& r, const r1rt_upper_cmd& c) { return r1rt_segment_push_upper_cmd(r.seg, &c); }

void feed(Rig& r, const r1rt_lowcmd_out& b) { ASSERT_EQ(r1rt_core_set_builtin_cmd(r.core, &b, r.t), R1RT_OK); }

// MONITOR -> ON_STAND through MIRROR (0.1 s) and the 1 s blend; the built-in damping the legs (as on the stand).
void enter_on_stand(Rig& r, double leg_kp = 0.0) {
  r.tick();
  feed(r, builtin(r.st, leg_kp));
  ASSERT_EQ(req(r, R1RT_REQ_ON_STAND), R1RT_OK);
  for (int i = 0; i < 50; ++i) {
    r.tick();
    feed(r, builtin(r.st, leg_kp));
  }
  ASSERT_EQ(r1rt_core_mode(r.core), R1RT_MODE_TAKEOVER_MIRROR);
  ASSERT_EQ(req(r, R1RT_REQ_TAKEOVER_RELEASED), R1RT_OK);
  for (int i = 0; i < 520; ++i) r.tick();
  ASSERT_EQ(r1rt_core_mode(r.core), R1RT_MODE_ON_STAND);
}

r1rt_upper_cmd lower_stream(std::uint8_t group, std::uint64_t id, const double* q, int n, std::int64_t t,
                            std::uint64_t epoch, std::int64_t ttl = 300 * kMs) {
  r1rt_upper_cmd c{};
  c.cmd_id = id;
  c.stop_epoch = epoch;
  c.t_issue_ns = t;
  c.ttl_ns = ttl;
  c.group = group;
  c.source = R1RT_SOURCE_STREAM;
  c.n_joints = static_cast<std::uint8_t>(n);
  for (int k = 0; k < n; ++k) c.q[k] = q[k];
  return c;
}

TEST(CoreOnStand, TheGatesForAHangingRobot) {
  // no policy is needed on the stand (TAKEOVER needs one: refused NO_POLICY with the same rig)
  Rig r(false, on_stand_gains);
  if (!r.ok) FAIL() << r.why;
  r.tick();
  feed(r, builtin(r.st, 0.0));
  EXPECT_EQ(req(r, R1RT_REQ_TAKEOVER), R1RT_E_REFUSED);
  EXPECT_EQ(r.status().takeover_refusal, R1RT_TAKEOVER_NO_POLICY);
  // Unitree damping the legs (kp 0): a takeover refuses (not visibly holding), ON_STAND accepts (the stand holds)
  EXPECT_EQ(req(r, R1RT_REQ_ON_STAND), R1RT_OK);
  EXPECT_EQ(r.status().on_stand_refusal, R1RT_TAKEOVER_OK);
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_TAKEOVER_MIRROR);
  r.tick();
  EXPECT_EQ(r.status().takeover_target, R1RT_MODE_ON_STAND);
  EXPECT_EQ(r.out.publish, 1);
  for (int j = 0; j < 12; ++j) EXPECT_EQ(r.out.kp[j], 0.0);  // MIRROR republishes the damped command as it is
}

TEST(CoreOnStand, GatesThatStillRefuse) {
  Rig r(true, on_stand_gains);
  NEED_RIG(r);
  r.tick();  // no built-in command at all: nothing to mirror / release
  EXPECT_EQ(req(r, R1RT_REQ_ON_STAND), R1RT_E_REFUSED);
  EXPECT_EQ(r.status().on_stand_refusal, R1RT_TAKEOVER_BUILTIN);
  r.st.dq[2] = 0.7;  // a swinging robot
  r.tick();
  feed(r, builtin(r.st, 0.0));
  EXPECT_EQ(req(r, R1RT_REQ_ON_STAND), R1RT_E_REFUSED);
  EXPECT_EQ(r.status().on_stand_refusal, R1RT_TAKEOVER_MOVING);
  r.st.dq[2] = 0.0;
  ASSERT_TRUE(r.enter());  // BALANCE: ON_STAND only from MONITOR
  r.tick();
  feed(r, builtin(r.st, 0.0));
  EXPECT_EQ(req(r, R1RT_REQ_ON_STAND), R1RT_E_REFUSED);
  EXPECT_EQ(r.status().on_stand_refusal, R1RT_TAKEOVER_NOT_MONITOR);
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_BALANCE);
}

TEST(CoreOnStand, BlendsIntoAStiffHoldOfTheMeasuredPoseNoPolicy) {
  Rig r(true, on_stand_gains);
  NEED_RIG(r);
  r.tick();
  feed(r, builtin(r.st, 0.0, 0.03));
  ASSERT_EQ(req(r, R1RT_REQ_ON_STAND), R1RT_OK);
  for (int i = 0; i < 50; ++i) {
    r.tick();
    feed(r, builtin(r.st, 0.0, 0.03));
  }
  const r1rt_lowcmd_out snap = r.out;  // MIRROR's last command = the built-in's (damped legs, q 0.03 off)
  const std::uint64_t commits0 = r.status().policy_commits;
  ASSERT_EQ(req(r, R1RT_REQ_TAKEOVER_RELEASED), R1RT_OK);
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_TAKEOVER_BLEND);
  double max_step = 0.0;
  r1rt_lowcmd_out prev = snap;
  for (int i = 0; i < 520; ++i) {
    r.tick();
    ASSERT_EQ(r.out.publish, 1);
    for (int j = 0; j < R1RT_NUM_JOINTS; ++j) max_step = std::fmax(max_step, std::fabs(r.out.q[j] - prev.q[j]));
    if (i == 249) {  // halfway: (1 - a) snapshot + a hold
      for (int j = 0; j < 12; ++j) EXPECT_NEAR(r.out.kp[j], 0.5 * kOnStandKp, 0.5) << j;
    }
    prev = r.out;
  }
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_ON_STAND);
  EXPECT_LT(max_step, 0.0005);  // 0.03 rad over 1 s at 500 Hz: 0.00006 per tick
  for (int j = 0; j < 14; ++j) {
    EXPECT_NEAR(r.out.q[j], r.st.q[j], 1e-9) << j;  // the hold of the pose measured at RELEASED
    EXPECT_EQ(r.out.kp[j], kOnStandKp) << j;
    EXPECT_EQ(r.out.kd[j], kOnStandKd) << j;
  }
  for (int j = 14; j < R1RT_NUM_JOINTS; ++j) EXPECT_EQ(r.out.kp[j], 60.0) << j;  // arms / head: the upper HOLD gains
  EXPECT_EQ(r.status().policy_commits, commits0);  // no policy step at all (thread idle, no entry action)
  EXPECT_EQ(r.status().inline_runs, 0u);
  EXPECT_EQ(r1rt::shm::OwnershipToken::owner_of(r.segment().token.load()), R1RT_OWNER_R1RT);
}

TEST(CoreOnStand, LegsAndWaistFollowL0StreamsAndHoldOnTtl) {
  Rig r(true, on_stand_gains);
  NEED_RIG(r);
  enter_on_stand(r);
  const double q0 = r.out.q[3];  // left knee (joint 3), held
  const std::uint64_t ep = r1rt_segment_stop_epoch(r.seg);
  double q6[6];
  for (int k = 0; k < 6; ++k) q6[k] = r.out.q[k];
  q6[3] = q0 + 0.2;
  ASSERT_EQ(push(r, lower_stream(R1RT_GROUP_LEG_LEFT, 7, q6, 6, r.t, ep)),
            R1RT_OK);
  for (int i = 0; i < 100; ++i) r.tick();  // 0.2 s: the target moves at dq_max (2 rad/s) -> reached in 0.1 s
  EXPECT_EQ(r.status().lower_accepted, 1u);
  EXPECT_EQ(r.status().lower_source[0], R1RT_SOURCE_STREAM);
  EXPECT_NEAR(r.out.q[3], q0 + 0.2, 1e-9);
  EXPECT_EQ(r.out.kp[3], kOnStandKp);  // record gains 0 = the hold gains
  for (int i = 0; i < 150; ++i) r.tick();  // TTL 0.3 s expired: decelerate (at rest already) and HOLD there
  EXPECT_EQ(r.status().lower_source[0], R1RT_SOURCE_HOLD);
  EXPECT_NEAR(r.out.q[3], q0 + 0.2, 1e-9);
  // the waist group too
  double w2[2] = {r.out.q[12] + 0.05, r.out.q[13] - 0.05};
  ASSERT_EQ(push(r, lower_stream(R1RT_GROUP_WAIST, 8, w2, 2, r.t, ep)),
            R1RT_OK);
  for (int i = 0; i < 50; ++i) r.tick();
  EXPECT_NEAR(r.out.q[12], w2[0], 1e-9);
  EXPECT_NEAR(r.out.q[13], w2[1], 1e-9);
}

TEST(CoreOnStand, LegRecordsAreRefusedWhileThePolicyOwnsTheLegs) {
  Rig r(true, on_stand_gains);
  NEED_RIG(r);
  ASSERT_TRUE(r.enter());
  for (int i = 0; i < 20; ++i) r.tick();
  const double before = r.out.q[3];
  double q6[6] = {0, 0, 0, 1.0, 0, 0};
  ASSERT_EQ(push(r, lower_stream(R1RT_GROUP_LEG_LEFT, 9, q6, 6, r.t,
                                                                         r1rt_segment_stop_epoch(r.seg))),
            R1RT_OK);
  for (int i = 0; i < 20; ++i) r.tick();
  EXPECT_EQ(r.status().lower_rejected, 1u);
  EXPECT_EQ(r.status().lower_accepted, 0u);
  EXPECT_LT(std::fabs(r.out.q[3] - before), 0.05);  // the policy's knee, not 1.0
}

TEST(CoreOnStand, StopHoldsEveryGroupOnlyTheOperatorDampsAndNothingBalances) {
  Rig r(true, on_stand_gains);
  NEED_RIG(r);
  enter_on_stand(r);
  const std::uint64_t ep = r1rt_segment_stop_epoch(r.seg);
  double q6[6];
  for (int k = 0; k < 6; ++k) q6[k] = r.out.q[6 + k];
  q6[0] += 0.5;  // right hip pitch: a long move (0.25 s at 2 rad/s)
  ASSERT_EQ(push(r, lower_stream(R1RT_GROUP_LEG_RIGHT, 11, q6, 6, r.t, ep,
                                                                         300 * kMs)),
            R1RT_OK);
  for (int i = 0; i < 40; ++i) r.tick();  // moving
  const double mid = r.out.q[6];
  ASSERT_EQ(req(r, R1RT_REQ_STOP), R1RT_OK);
  for (int i = 0; i < 200; ++i) r.tick();
  EXPECT_EQ(r.status().lower_source[1], R1RT_SOURCE_HOLD);
  EXPECT_LT(r.out.q[6], q6[0] - 0.05);  // stopped short of the target (decelerated within the stop time)
  EXPECT_GT(r.out.q[6], mid);
  for (int j = 0; j < 14; ++j) EXPECT_EQ(r.out.kp[j], kOnStandKp);  // a stop is never limp
  // never a balance / freeze on the hanging robot
  EXPECT_EQ(req(r, R1RT_REQ_ENTER_BALANCE), R1RT_E_REFUSED);
  EXPECT_EQ(req(r, R1RT_REQ_STAND_FREEZE), R1RT_E_REFUSED);
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_ON_STAND);
  // the session end: the operator's damp (the mailbox refuses it from anyone else)
  std::uint64_t seq = 0;
  ASSERT_EQ(r1rt_segment_request_mode(r.seg, R1RT_REQ_DAMP, R1RT_REQUESTER_R1D, r.t, &seq), R1RT_OK);
  r.tick();
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_ON_STAND);
  ASSERT_EQ(r1rt_segment_request_mode(r.seg, R1RT_REQ_DAMP, R1RT_REQUESTER_OPERATOR, r.t, &seq), R1RT_OK);
  r.tick();
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_DAMP);
}

TEST(CoreOnStand, AStateGapResendsTheLastCommand) {
  Rig r(true, on_stand_gains);
  NEED_RIG(r);
  enter_on_stand(r);
  const r1rt_lowcmd_out last = r.out;
  for (int i = 0; i < 100; ++i) r.tick(true);  // 200 ms without lowstate: STALE
  EXPECT_EQ(r.out.publish, 1);
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    EXPECT_EQ(r.out.q[j], last.q[j]) << j;
    EXPECT_EQ(r.out.kp[j], last.kp[j]) << j;
  }
  EXPECT_GT(r.status().stale_resends, 0u);
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_ON_STAND);
}

void require_guard(r1rt_config& c) {
  on_stand_gains(c);
  c.takeover_require_guard = 1;
}

void beat_guard(Rig& r) {
  r1rt_heartbeat hb{};
  static std::uint64_t seq = 0;
  hb.seq = ++seq;
  hb.t_ns = r.t;
  hb.rate_hz = 1000;
  r.segment().guard_heartbeat.write(hb);
}

TEST(CoreGuardAbsent, AnAlarmInAPublishingModeOnlyWithAGuardProfile) {
  Rig r(true, require_guard);
  NEED_RIG(r);
  r.tick();
  feed(r, builtin(r.st, 0.0));
  EXPECT_EQ(req(r, R1RT_REQ_ON_STAND), R1RT_E_REFUSED);  // no guard heartbeat: refused (as a takeover)
  EXPECT_EQ(r.status().on_stand_refusal, R1RT_TAKEOVER_GUARD);
  EXPECT_EQ(r.status().faults & R1RT_FAULT_GUARD_ABSENT, 0u);  // MONITOR publishes nothing: no alarm
  // with a beating guard: in, and no alarm
  for (int i = 0; i < 5; ++i) {
    beat_guard(r);
    r.tick();
    feed(r, builtin(r.st, 0.0));
  }
  ASSERT_EQ(req(r, R1RT_REQ_ON_STAND), R1RT_OK);
  ASSERT_EQ(req(r, R1RT_REQ_TAKEOVER_RELEASED), R1RT_OK);
  for (int i = 0; i < 600; ++i) {
    beat_guard(r);
    r.tick();
  }
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_ON_STAND);
  EXPECT_EQ(r.status().faults_seen & R1RT_FAULT_GUARD_ABSENT, 0u);
  // the guard falls silent: GUARD_ABSENT after guard_max_age (50 ms), an alarm - the command does not change
  const r1rt_lowcmd_out before = r.out;
  int ticks_to_alarm = -1;
  for (int i = 0; i < 100; ++i) {
    r.tick();
    if (ticks_to_alarm < 0 && (r.status().faults & R1RT_FAULT_GUARD_ABSENT)) ticks_to_alarm = i + 1;
  }
  EXPECT_GE(ticks_to_alarm, 25);  // 50 ms at 2 ms per tick
  EXPECT_LE(ticks_to_alarm, 27);
  EXPECT_EQ(r.out.publish, 1);
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    EXPECT_EQ(r.out.q[j], before.q[j]) << j;
    EXPECT_EQ(r.out.kp[j], before.kp[j]) << j;
  }
  EXPECT_GT(r.status().guard_absent_ticks, 70u);
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_ON_STAND);
  // back: the alarm clears
  beat_guard(r);
  r.tick();
  EXPECT_EQ(r.status().faults & R1RT_FAULT_GUARD_ABSENT, 0u);
}

TEST(CoreGuardAbsent, NoAlarmWithoutAGuardProfile) {
  Rig r;  // the simulator's single-core profile (takeover_require_guard 0)
  NEED_RIG(r);
  ASSERT_TRUE(r.enter());
  for (int i = 0; i < 100; ++i) r.tick();
  EXPECT_EQ(r.status().faults_seen & R1RT_FAULT_GUARD_ABSENT, 0u);
  EXPECT_EQ(r.status().guard_age_ns, INT64_MAX);
}

TEST(CoreMirror, CarriesTheBlendFreezeAndTarget) {
  Rig r(true, on_stand_gains);
  NEED_RIG(r);
  r.tick();
  feed(r, builtin(r.st, 200.0));
  ASSERT_EQ(r1rt_core_request(r.core, R1RT_REQ_TAKEOVER, r.t), R1RT_OK);
  r.tick();
  ASSERT_EQ(req(r, R1RT_REQ_TAKEOVER_RELEASED), R1RT_OK);
  const std::int64_t t_rel = r.t;
  for (int i = 0; i < 100; ++i) r.tick();
  r1rt_mirror mi{};
  ASSERT_TRUE(r.segment().mirror.try_read(mi));
  EXPECT_EQ(mi.mode, R1RT_MODE_TAKEOVER_BLEND);
  EXPECT_EQ(mi.takeover_target, R1RT_MODE_BALANCE);
  EXPECT_EQ(mi.blend_t0_ns, t_rel);
  EXPECT_EQ(mi.blend_ns, kBlendNs);
  EXPECT_EQ(mi.freeze_t0_ns, 0);
  ASSERT_EQ(req(r, R1RT_REQ_STAND_FREEZE), R1RT_OK);  // the operator's freeze from BLEND
  const std::int64_t t_fr = r.t;
  r.tick();
  ASSERT_TRUE(r.segment().mirror.try_read(mi));
  EXPECT_EQ(mi.mode, R1RT_MODE_STAND_FREEZE);
  EXPECT_EQ(mi.freeze_t0_ns, t_fr);
  EXPECT_EQ(mi.freeze_ramp_ns, 400 * kMs);
  EXPECT_EQ(mi.freeze_invalid, 0);
  EXPECT_EQ(mi.blend_t0_ns, 0);
}

TEST(CoreHandsPose, TheArmModelUsesTheHandsJointsAndPayloadsWhileFresh) {
  // session 5 (the surface guard's / arm model's input, DESIGN §3.4): r1hands' finger joints + declared payloads in the
  // segment's hands_pose region change the arms' gravity feed-forward exactly as the arm model says; a stale or invalid
  // pose falls back to the open hand without payload (session 4's behaviour)
  Rig r;
  NEED_RIG(r);
  ASSERT_TRUE(r.enter());
  for (int i = 0; i < 10; ++i) r.tick();
  double open_ff[R1RT_NUM_JOINTS];
  ASSERT_TRUE(r1rt::arm::gravity_ff26(r.st.q, r.st.imu_quat, nullptr, nullptr, nullptr, open_ff));
  for (int j = 14; j < 24; ++j) EXPECT_NEAR(r.out.tau_ff[j], open_ff[j], 1e-9) << j;  // HOLD: tau_ff = gravity ff
  EXPECT_EQ(r.status().hands_pose_used, 0);
  r1rt_hands_pose hp{};
  hp.seq = 1;
  hp.t_ns = r.t;
  for (int s = 0; s < 2; ++s) {
    for (int k = 0; k < 7; ++k) hp.q[s][k] = 0.4 * (k % 2 == 0 ? 1.0 : -1.0);
    hp.payload_kg[s] = 0.5;
  }
  hp.valid = 1;
  r.segment().hands_pose.write(hp);
  r.tick();
  double held_ff[R1RT_NUM_JOINTS];
  ASSERT_TRUE(r1rt::arm::gravity_ff26(r.st.q, r.st.imu_quat, hp.q[0], hp.q[1], hp.payload_kg, held_ff));
  double diff = 0.0;
  for (int j = 14; j < 24; ++j) {
    EXPECT_NEAR(r.out.tau_ff[j], held_ff[j], 1e-9) << j;
    diff = std::fmax(diff, std::fabs(held_ff[j] - open_ff[j]));
  }
  EXPECT_GT(diff, 0.5);  // 0.5 kg in each hand: a visible shoulder torque
  EXPECT_EQ(r.status().hands_pose_used, 1);
  for (int i = 0; i < 60; ++i) r.tick();  // 120 ms without a new pose: stale -> the open hand again
  for (int j = 14; j < 24; ++j) EXPECT_NEAR(r.out.tau_ff[j], open_ff[j], 1e-9) << j;
  EXPECT_EQ(r.status().hands_pose_used, 0);
}

void require_rt(r1rt_config& c) {
  on_stand_gains(c);
  c.takeover_require_rt = 1;  // a robot profile; rt_granted stays 0 (as the process found it)
}

TEST(CoreProcessFacts, NoTakeoverOrOnStandWithoutTheRealTimeGrantsTheProfileRequires) {
  Rig r(true, require_rt);
  NEED_RIG(r);
  r.tick();
  feed(r, builtin(r.st, 200.0));
  EXPECT_NE(r.status().faults & R1RT_FAULT_RT_NOT_GRANTED, 0u);  // an alarm while not granted
  EXPECT_EQ(req(r, R1RT_REQ_TAKEOVER), R1RT_E_REFUSED);
  EXPECT_EQ(r.status().takeover_refusal, R1RT_TAKEOVER_RT);
  EXPECT_EQ(req(r, R1RT_REQ_ON_STAND), R1RT_E_REFUSED);
  EXPECT_EQ(r.status().on_stand_refusal, R1RT_TAKEOVER_RT);
  r1rt_heartbeat hb{};
  ASSERT_TRUE(r.segment().rt_heartbeat.try_read(hb));
  EXPECT_EQ(hb.rt_granted, 0);
  ASSERT_EQ(r1rt_core_set_rt_granted(r.core, 1), R1RT_OK);
  r.tick();
  feed(r, builtin(r.st, 200.0));
  EXPECT_EQ(r.status().faults & R1RT_FAULT_RT_NOT_GRANTED, 0u);
  ASSERT_TRUE(r.segment().rt_heartbeat.try_read(hb));
  EXPECT_EQ(hb.rt_granted, 1);
  EXPECT_EQ(r.status().rt_granted, 1);
  EXPECT_EQ(req(r, R1RT_REQ_TAKEOVER), R1RT_OK);
}

TEST(CoreProcessFacts, NoRtGateInTheSimulatorsProfile) {
  Rig r(true, on_stand_gains);  // takeover_require_rt 0, rt_granted 0 (lockstep)
  NEED_RIG(r);
  r.tick();
  feed(r, builtin(r.st, 200.0));
  EXPECT_EQ(r.status().faults & R1RT_FAULT_RT_NOT_GRANTED, 0u);
  EXPECT_EQ(req(r, R1RT_REQ_TAKEOVER), R1RT_OK);
}

TEST(CoreProcessFacts, TheLoopTimingLandsInTheTelemetryRecord) {
  Rig r;
  NEED_RIG(r);
  r.tick();
  EXPECT_EQ(r.tel.wake_lateness_ns, 0);
  EXPECT_EQ(r.tel.iteration_ns, 0);
  ASSERT_EQ(r1rt_core_set_loop_timing(r.core, 41'000, 230'000), R1RT_OK);
  r.tick();
  EXPECT_EQ(r.tel.wake_lateness_ns, 41'000);
  EXPECT_EQ(r.tel.iteration_ns, 230'000);
  ASSERT_EQ(r1rt_core_set_loop_timing(r.core, 12'000, 90'000), R1RT_OK);
  r.tick();
  EXPECT_EQ(r.status().max_wake_lateness_ns, 41'000);
  EXPECT_EQ(r.status().max_iteration_ns, 230'000);
  r1rt_telemetry_record rec{};
  std::int32_t last_wake = -1;
  while (r.segment().telemetry.try_pop(rec)) last_wake = rec.wake_lateness_ns;  // the ring r1log drains
  EXPECT_EQ(last_wake, 12'000);
}

}  // namespace

// ---- session 6: what Unitree's controller does on the stand (the user, 10 Oct: it starts in ZERO TORQUE; the user
// can switch it to a stiff STAND LOCK, not balancing). Whether zero torque publishes rt/lowcmd is unknown (HW1 B1.2).
// No gate changed here: these pin today's behaviour for each case.

namespace s6 {

// MIRROR (0.1 s) with the given built-in command, then RELEASED; returns the largest per-tick change of q and kp over
// the MIRROR -> BLEND -> ON_STAND sequence (from the first MIRROR tick on) and the MIRROR's first command.
struct Seq {
  double max_dq = 0.0, max_dkp = 0.0;
  r1rt_lowcmd_out first_mirror{};
  std::uint8_t used_mirror = 0;
};

Seq run(Rig& r, const r1rt_lowcmd_out& b) {
  Seq out;
  r.tick();
  feed(r, b);
  EXPECT_EQ(req(r, R1RT_REQ_ON_STAND), R1RT_OK);
  EXPECT_EQ(r.status().on_stand_refusal, R1RT_TAKEOVER_OK);
  out.used_mirror = r.status().takeover_used_mirror;
  r1rt_lowcmd_out prev = b;  // the built-in's command: what the motors had before our first packet
  bool first = true;
  auto track = [&]() {
    if (first) {
      out.first_mirror = r.out;
      first = false;
    }
    for (int j = 0; j < 14; ++j) {
      out.max_dq = std::fmax(out.max_dq, std::fabs(r.out.q[j] - prev.q[j]));
      out.max_dkp = std::fmax(out.max_dkp, std::fabs(r.out.kp[j] - prev.kp[j]));
    }
    prev = r.out;
  };
  for (int i = 0; i < 50; ++i) {
    r.tick();
    feed(r, b);
    track();
  }
  EXPECT_EQ(req(r, R1RT_REQ_TAKEOVER_RELEASED), R1RT_OK);
  for (int i = 0; i < 520; ++i) {
    r.tick();
    track();
  }
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_ON_STAND);
  return out;
}

r1rt_lowcmd_out unitree(const r1rt_lowstate_in& st, double leg_kp, double kd, std::uint8_t motor_mode) {
  r1rt_lowcmd_out b{};
  b.publish = 1;
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    b.enable[j] = motor_mode;  // hg MotorCmd_.mode as the DDS layer hands it on (dds_io.cpp: enable = mode)
    b.q[j] = st.q[j] + 0.01;
    b.kp[j] = j < 14 ? leg_kp : 0.0;  // legs + waist (the stand lock holds the waist too; an estimate)
    b.kd[j] = kd;
  }
  return b;
}

}  // namespace s6

TEST(CoreOnStand, S6_AStiffStandLockIsAcceptedAndBlendsWithoutAStep) {
  // (a) stand lock (legs kp 200, every motor enabled): accepted, MIRROR republishes it, the 1 s blend moves q by
  // 0.01 rad and kp 200 -> 150 smoothly (per 2 ms tick: <= 0.01 / 500 rad, <= 50 / 500 kp, + rounding).
  Rig r(true, on_stand_gains);
  NEED_RIG(r);
  const s6::Seq q = s6::run(r, s6::unitree(r.st, 200.0, 3.0, 1));
  EXPECT_EQ(q.used_mirror, 1);
  for (int j = 0; j < 12; ++j) EXPECT_EQ(q.first_mirror.kp[j], 200.0) << j;
  EXPECT_LT(q.max_dq, 0.01 / 500.0 * 1.05);
  EXPECT_LT(q.max_dkp, 50.0 / 500.0 * 1.05);
}

TEST(CoreOnStand, S6_ZeroTorquePublishedWithMotorsEnabledIsAcceptedAndRampsTheGainsUp) {
  // (b) zero torque PUBLISHED as kp 0 / kd 0 with every motor enabled (mode 1): accepted (U13 / D22: soft is fine).
  // Session 6 (R7): the soft slots' targets mean nothing, so the snapshot is the measured pose at kp 0 / kd 0 (not the
  // built-in's targets, 0.01 rad off here) and the blend ramps the legs 0 -> 150 over 1 s (<= 0.3 per tick).
  Rig r(true, on_stand_gains);
  NEED_RIG(r);
  const s6::Seq q = s6::run(r, s6::unitree(r.st, 0.0, 0.0, 1));
  EXPECT_EQ(q.used_mirror, 2);
  for (int j = 0; j < 14; ++j) {
    EXPECT_EQ(q.first_mirror.kp[j], 0.0) << j;
    EXPECT_NEAR(q.first_mirror.q[j], r.st.q[j], 1e-12) << j;
  }
  EXPECT_LT(q.max_dkp, kOnStandKp / 500.0 * 1.05);
  EXPECT_LT(q.max_dq, 0.0101);  // only the first command: the measured pose instead of the kp-0 target (no torque)
}

TEST(CoreOnStand, S6_ZeroTorqueWithMotorsDisabledBlendsTheGainsUpFromZero) {
  // (b') zero torque published with the motors DISABLED (hg mode 0): the snapshot is unusable as a command (a motor not
  // enabled). Until 6c59cd6 MIRROR then held the measured pose at the ON_STAND gains from its first tick (legs 0 -> 150
  // in ONE tick; rehearse R7). Now (U13 / D22: a 1 s blend into the stiff hold): the snapshot is the MEASURED pose at
  // the built-in's own gains (a disabled slot = kp 0 / kd 0), and the blend ramps the gains up (fails on 6c59cd6).
  Rig r(true, on_stand_gains);
  NEED_RIG(r);
  const s6::Seq q = s6::run(r, s6::unitree(r.st, 0.0, 0.0, 0));
  EXPECT_EQ(q.used_mirror, 2);
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    EXPECT_EQ(q.first_mirror.kp[j], 0.0) << j;
    EXPECT_EQ(q.first_mirror.kd[j], 0.0) << j;
    EXPECT_NEAR(q.first_mirror.q[j], r.st.q[j], 1e-12) << j;  // the measured pose
  }
  EXPECT_LT(q.max_dkp, kOnStandKp / 500.0 * 1.5);  // <= hold / (1 s x 500 Hz) per tick (+ margin)
  EXPECT_LT(q.max_dq, 0.02);  // the first command: the measured pose instead of the built-in's (kp 0) targets
}

TEST(CoreOnStand, S6_ZeroTorqueWithFarTargetsBlendsTheGainsUpFromZero) {
  // Unitree's zero torque as the bridge emulates it: kp 0 / kd 0, motors enabled, its targets far from the hanging pose
  // (0.3 rad: beyond mirror_max_dev, so the built-in's command is not mirrored as it is). The same soft snapshot: the
  // measured pose at kp 0 / kd 0, gains ramped over the blend (6c59cd6: a one-tick step to 150; fails there).
  Rig r(true, on_stand_gains);
  NEED_RIG(r);
  r1rt_lowcmd_out b = s6::unitree(r.st, 0.0, 0.0, 1);
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) b.q[j] = r.st.q[j] + 0.3;
  const s6::Seq q = s6::run(r, b);
  EXPECT_EQ(q.used_mirror, 2);
  for (int j = 0; j < 14; ++j) {
    EXPECT_EQ(q.first_mirror.kp[j], 0.0) << j;
    EXPECT_NEAR(q.first_mirror.q[j], r.st.q[j], 1e-12) << j;
  }
  EXPECT_LT(q.max_dkp, kOnStandKp / 500.0 * 1.5);
  // a damping built-in (kp 0, kd 3) keeps its kd in the snapshot
  Rig d(true, on_stand_gains);
  NEED_RIG(d);
  r1rt_lowcmd_out bd = s6::unitree(d.st, 0.0, 3.0, 1);
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) bd.q[j] = d.st.q[j] + 0.3;
  const s6::Seq qd = s6::run(d, bd);
  EXPECT_EQ(qd.used_mirror, 2);
  for (int j = 0; j < 14; ++j) EXPECT_EQ(qd.first_mirror.kd[j], 3.0) << j;
}

TEST(CoreOnStand, S6_AStiffBuiltinFarFromThePoseKeepsTheMeasuredHoldFallback) {
  // unchanged: a STIFF built-in whose targets are not usable (0.8 rad off) -> our hold of the measured pose at the
  // ON_STAND gains (200 -> 150: no softening, no stiffening step from zero)
  Rig r(true, on_stand_gains);
  NEED_RIG(r);
  r1rt_lowcmd_out b = s6::unitree(r.st, 200.0, 3.0, 1);
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) b.q[j] = r.st.q[j] + 0.8;  // beyond mirror_max_dev (0.6 rad)
  const s6::Seq q = s6::run(r, b);
  EXPECT_EQ(q.used_mirror, 0);
  for (int j = 0; j < 12; ++j) EXPECT_EQ(q.first_mirror.kp[j], kOnStandKp) << j;
}

TEST(CoreOnStand, S6_ASilentBuiltinIsRefusedBuiltin) {
  // (c) zero torque that publishes NOTHING: ON_STAND refused with R1RT_TAKEOVER_BUILTIN (nothing to mirror, no live
  // writer to release); nothing published, MONITOR kept. The options are the user's (S6.md).
  Rig r(true, on_stand_gains);
  NEED_RIG(r);
  for (int i = 0; i < 5; ++i) r.tick();
  EXPECT_EQ(req(r, R1RT_REQ_ON_STAND), R1RT_E_REFUSED);
  EXPECT_EQ(r.status().on_stand_refusal, R1RT_TAKEOVER_BUILTIN);
  r.tick();
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_MONITOR);
  EXPECT_EQ(r.out.publish, 0);
}

// ---- session 6: the operator's STAND_FREEZE on the stand (HW1 runbook S10, U3) and STAND_HOLD back ----

namespace s6 {

std::pair<std::int32_t, std::int32_t> mb(Rig& r, std::uint32_t req, std::uint32_t requester) {
  std::uint64_t seq = 0;
  EXPECT_EQ(r1rt_segment_request_mode(r.seg, req, requester, r.t, &seq), R1RT_OK);
  std::int32_t status = 1, refusal = 0;
  std::uint32_t after = 0;
  for (int i = 0; i < 5; ++i) {
    r.tick();
    if (r1rt_segment_mode_reply(r.seg, seq, &status, &refusal, &after) == R1RT_OK) break;
  }
  return {status, refusal};
}

}  // namespace s6

TEST(CoreOnStand, S6_TheOperatorsFreezeOnTheStandRampsToTheStandPoseHoldsAndReturnsToTheHold) {
  // Session 5 refused STAND_FREEZE in ON_STAND (fails on 26e8bce). Now: the operator's freeze ramps legs + waist from
  // the hold to the stand pose over stand_ramp_s (0.4 s) at the stand gains, never leads to BALANCE, and STAND_HOLD
  // blends back into the ON_STAND hold of the stand pose without a position step.
  Rig r(true, on_stand_gains);
  NEED_RIG(r);
  enter_on_stand(r);
  const r1rt_lowcmd_out hold = r.out;
  auto [st0, rf0] = s6::mb(r, R1RT_REQ_STAND_FREEZE, R1RT_REQUESTER_R1D);
  EXPECT_EQ(st0, R1RT_E_REFUSED);
  EXPECT_EQ(rf0, R1RT_REFUSE_NOT_OPERATOR);
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_ON_STAND);
  auto [st1, rf1] = s6::mb(r, R1RT_REQ_STAND_FREEZE, R1RT_REQUESTER_OPERATOR);
  ASSERT_EQ(st1, R1RT_OK) << rf1;
  ASSERT_EQ(r1rt_core_mode(r.core), R1RT_MODE_STAND_FREEZE);
  EXPECT_EQ(r.status().freeze_from_stand, 1);
  r1rt_lowcmd_out prev = r.out;
  double max_dq = 0.0;
  for (int i = 0; i < 250; ++i) {  // 0.5 s: past the 0.4 s ramp
    r.tick();
    ASSERT_EQ(r.out.publish, 1);
    for (int j = 0; j < 14; ++j) max_dq = std::fmax(max_dq, std::fabs(r.out.q[j] - prev.q[j]));
    prev = r.out;
  }
  for (int j = 0; j < 14; ++j) {
    EXPECT_NEAR(r.out.q[j], r.cfg.stand_q[j], 1e-12) << j;  // at the stand pose
    EXPECT_NEAR(r.out.kp[j], r.cfg.stand_kp[j], 1e-12) << j;  // at the stand gains
  }
  double span = 0.0;
  for (int j = 0; j < 14; ++j) span = std::fmax(span, std::fabs(r.cfg.stand_q[j] - hold.q[j]));
  EXPECT_LT(max_dq, span / 200.0 * 1.05);  // a linear 0.4 s ramp: span / 200 per 2 ms tick
  EXPECT_EQ(req(r, R1RT_REQ_ENTER_BALANCE), R1RT_E_REFUSED);  // never balance the hanging robot
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_STAND_FREEZE);
  auto [st2, rf2] = s6::mb(r, R1RT_REQ_STAND_HOLD, R1RT_REQUESTER_OPERATOR);
  ASSERT_EQ(st2, R1RT_OK) << rf2;
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_TAKEOVER_BLEND);
  prev = r.out;
  max_dq = 0.0;
  for (int i = 0; i < 520; ++i) {
    r.tick();
    for (int j = 0; j < 14; ++j) max_dq = std::fmax(max_dq, std::fabs(r.out.q[j] - prev.q[j]));
    prev = r.out;
  }
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_ON_STAND);
  EXPECT_LT(max_dq, 1e-9);  // a hold of the stand pose: no position change at all
  for (int j = 0; j < 12; ++j) {
    EXPECT_NEAR(r.out.q[j], r.cfg.stand_q[j], 1e-9) << j;
    EXPECT_NEAR(r.out.kp[j], kOnStandKp, 1e-9) << j;
  }
  EXPECT_EQ(req(r, R1RT_REQ_STAND_HOLD), R1RT_E_REFUSED);  // only from the freeze on the stand
  EXPECT_EQ(r.status().freeze_refusal, R1RT_REFUSE_MODE);
}

TEST(CoreOnStand, S6_TheFreezeOnTheStandIsRefusedFarFromTheStandPose) {
  Rig r(true, on_stand_gains);
  NEED_RIG(r);
  enter_on_stand(r);
  // the left knee 0.3 rad off through an L0 stream (the lower executor), then the freeze: refused FAR, nothing moves
  double q6[6];
  for (int k = 0; k < 6; ++k) q6[k] = r.out.q[k];
  q6[3] += 0.3;
  std::uint64_t id = 1;
  for (int i = 0; i < 150; ++i) {
    if (i % 25 == 0) {
      ASSERT_EQ(push(r, lower_stream(R1RT_GROUP_LEG_LEFT, id++, q6, 6, r.t, r1rt_segment_stop_epoch(r.seg))), R1RT_OK);
    }
    r.tick();
  }
  ASSERT_NEAR(r.out.q[3], q6[3], 1e-6);
  auto [st, rf] = s6::mb(r, R1RT_REQ_STAND_FREEZE, R1RT_REQUESTER_OPERATOR);
  EXPECT_EQ(st, R1RT_E_REFUSED);
  EXPECT_EQ(rf, R1RT_REFUSE_FAR);
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_ON_STAND);
}

// ---- session 6: the stand session (r1rt --stand-session; DESIGN §3.5 start flag) ----

TEST(CoreOnStand, S6_AStandSessionRefusesEveryEntryIntoBalanceButNotOnStand) {
  // The user's OK (10 Oct): with Unitree in stand lock a takeover was ACCEPTED on the hanging robot (the balance policy
  // in the air) - nothing locked it out (fails on 934666c: the request / setter do not exist there).
  Rig r(true, on_stand_gains);
  NEED_RIG(r);
  ASSERT_EQ(r1rt_core_set_stand_session(r.core, 1), R1RT_OK);
  EXPECT_EQ(r1rt_core_set_stand_session(r.core, 0), R1RT_E_REFUSED);  // never cleared at run time
  r.tick();
  EXPECT_EQ(r.status().stand_session, 1);
  r1rt_heartbeat hb{};
  ASSERT_TRUE(r.segment().rt_heartbeat.try_read(hb));
  EXPECT_EQ(hb.guard_flags & R1RT_RT_FLAG_STAND_SESSION, R1RT_RT_FLAG_STAND_SESSION);
  feed(r, s6::unitree(r.st, 200.0, 3.0, 1));  // Unitree's stand lock: a takeover's gates would pass
  EXPECT_EQ(req(r, R1RT_REQ_TAKEOVER), R1RT_E_REFUSED);
  EXPECT_EQ(r.status().takeover_refusal, R1RT_TAKEOVER_STAND_SESSION);
  EXPECT_EQ(req(r, R1RT_REQ_ENTER_BALANCE), R1RT_E_REFUSED);
  auto [st, rf] = s6::mb(r, R1RT_REQ_ENTER_BALANCE, R1RT_REQUESTER_OPERATOR);
  EXPECT_EQ(st, R1RT_E_REFUSED);
  EXPECT_EQ(rf, R1RT_TAKEOVER_STAND_SESSION);
  auto [st2, rf2] = s6::mb(r, R1RT_REQ_TAKEOVER, R1RT_REQUESTER_R1D);
  EXPECT_EQ(st2, R1RT_E_REFUSED);
  EXPECT_EQ(rf2, R1RT_TAKEOVER_STAND_SESSION);
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_MONITOR);
  EXPECT_EQ(r.out.publish, 0);
  enter_on_stand(r, 200.0);  // ON_STAND unchanged
  EXPECT_EQ(req(r, R1RT_REQ_ENTER_BALANCE), R1RT_E_REFUSED);
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_ON_STAND);
}

TEST(CoreOnStand, S6_WithoutAStandSessionTheTakeoverIsUnchanged) {
  Rig r(true, on_stand_gains);
  NEED_RIG(r);
  EXPECT_EQ(r1rt_core_set_stand_session(r.core, 0), R1RT_OK);  // clearing what was never set: fine
  r.tick();
  EXPECT_EQ(r.status().stand_session, 0);
  feed(r, s6::unitree(r.st, 200.0, 3.0, 1));
  EXPECT_EQ(req(r, R1RT_REQ_TAKEOVER), R1RT_OK);
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_TAKEOVER_MIRROR);
}
