// The takeover sequence in r1rt::Core (ABI 4; DESIGN §3.5, v2 r1mc2/runtime.py:335-412): MONITOR -> TAKEOVER_MIRROR
// (the built-in's last rt/lowcmd republished) -> TAKEOVER_BLEND (linear blend into our BALANCE command) -> BALANCE,
// on a synthetic state stream. Every gate refuses with nothing changed; v3's failure branches (DESIGN §3.5): a fault
// in MIRROR keeps mirroring (alarm), ABORT goes back to MONITOR and returns the token, a fault in BLEND finishes the
// blend (alarm), an invalid action in BLEND freezes from the blended command; nothing ever damps. The same sequence
// on the simulated robot (the S1 fixture as the built-in): tests/integration/test_v3_core_takeover.py.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>

#include "r1rt/shm_layout.h"
#include "r1rt/step_api.h"
#include "test_core_rig.hpp"

namespace {

constexpr std::int64_t kBlendNs = 1000 * kMs;  // the default takeover_blend_s (1.0 s, v2's takeover default)

// The built-in controller's command as the DDS layer hands it over: a stand lock (kp 200 legs) `dev` rad from the
// measured pose on every joint, every motor enabled.
r1rt_lowcmd_out stand_lock(const r1rt_lowstate_in& st, double dev = 0.02, double leg_kp = 200.0) {
  r1rt_lowcmd_out b{};
  b.publish = 1;
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    b.enable[j] = 1;
    b.q[j] = st.q[j] + dev;
    b.kp[j] = j < 12 ? leg_kp : 60.0;
    b.kd[j] = 3.0;
  }
  return b;
}

void feed(Rig& r, const r1rt_lowcmd_out& b, std::int64_t age_ns = 0) {
  ASSERT_EQ(r1rt_core_set_builtin_cmd(r.core, &b, r.t - age_ns), R1RT_OK);
}

std::int32_t takeover(Rig& r) { return r1rt_core_request(r.core, R1RT_REQ_TAKEOVER, r.t); }

r1rt::shm::OwnershipToken& token(Rig& r) { return r.segment().token; }

r1rt_owner owner(Rig& r) { return r1rt::shm::OwnershipToken::owner_of(token(r).load()); }

// MONITOR with a fresh stand lock: tick once, feed, request.
std::int32_t try_takeover(Rig& r, const r1rt_lowcmd_out& b) {
  r.tick();
  feed(r, b);
  return takeover(r);
}

TEST(CoreTakeover, EveryGateRefusesAndNothingChanges) {
  struct Case {
    const char* name;
    int why;
    void (*spoil)(Rig&);
    void (*mend)(Rig&);
  };
  const Case cases[] = {
      {"tilt 15 deg", R1RT_TAKEOVER_TILT,
       [](Rig& r) {
         r.st.imu_quat[0] = std::cos(0.5 * 15.0 * M_PI / 180.0);
         r.st.imu_quat[2] = std::sin(0.5 * 15.0 * M_PI / 180.0);
       },
       [](Rig& r) {
         r.st.imu_quat[0] = 1.0;
         r.st.imu_quat[2] = 0.0;
       }},
      {"moving 0.6 rad/s", R1RT_TAKEOVER_MOVING, [](Rig& r) { r.st.dq[3] = 0.6; }, [](Rig& r) { r.st.dq[3] = 0.0; }},
      {"motor error", R1RT_TAKEOVER_MOTOR_ERROR, [](Rig& r) { r.st.motor_state[5] = 0x20; },
       [](Rig& r) { r.st.motor_state[5] = 0; }},
      {"hot 60 C", R1RT_TAKEOVER_HOT, [](Rig& r) { r.st.temperature[7][1] = 60; },
       [](Rig& r) { r.st.temperature[7][1] = 45; }},
  };
  for (const Case& c : cases) {
    Rig r;
    NEED_RIG(r);
    c.spoil(r);
    EXPECT_EQ(try_takeover(r, stand_lock(r.st)), R1RT_E_REFUSED) << c.name;
    EXPECT_EQ(r.status().takeover_refusal, c.why) << c.name;
    EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_MONITOR) << c.name;
    EXPECT_EQ(owner(r), R1RT_OWNER_NONE) << c.name;
    r.tick();
    EXPECT_EQ(r.out.publish, 0) << c.name;
    c.mend(r);  // the positive control: the same rig is accepted once the gate is met
    EXPECT_EQ(try_takeover(r, stand_lock(r.st)), R1RT_OK) << c.name;
    EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_TAKEOVER_MIRROR) << c.name;
  }
}

TEST(CoreTakeover, TheBuiltinMustBeFreshAndStiffTheStateFreshTheTokenFree) {
  Rig r;
  NEED_RIG(r);
  r.tick();
  EXPECT_EQ(takeover(r), R1RT_E_REFUSED);  // nothing received from the built-in
  EXPECT_EQ(r.status().takeover_refusal, R1RT_TAKEOVER_BUILTIN);
  r.tick();
  feed(r, stand_lock(r.st), 150 * kMs);  // received 150 ms ago (> 100 ms)
  EXPECT_EQ(takeover(r), R1RT_E_REFUSED);
  EXPECT_EQ(r.status().takeover_refusal, R1RT_TAKEOVER_BUILTIN);
  EXPECT_EQ(try_takeover(r, stand_lock(r.st, 0.02, 8.0)), R1RT_E_REFUSED);  // soft legs (kp 8 <= 10): not holding
  EXPECT_EQ(r.status().takeover_refusal, R1RT_TAKEOVER_BUILTIN);
  for (int i = 0; i < 30; ++i) r.tick(/*withhold=*/true);  // 60 ms without lowstate (> 50 ms)
  feed(r, stand_lock(r.st));
  EXPECT_EQ(takeover(r), R1RT_E_REFUSED);
  EXPECT_EQ(r.status().takeover_refusal, R1RT_TAKEOVER_STATE);
  std::uint64_t w = token(r).load();
  ASSERT_TRUE(token(r).try_transfer(w, R1RT_OWNER_R1GUARD));  // the guard holds the token
  EXPECT_EQ(try_takeover(r, stand_lock(r.st)), R1RT_E_REFUSED);
  EXPECT_EQ(r.status().takeover_refusal, R1RT_TAKEOVER_TOKEN);
  EXPECT_EQ(owner(r), R1RT_OWNER_R1GUARD);
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_MONITOR);
  EXPECT_EQ(r.status().takeover_requests, 5u);

  Rig np(/*with_policy=*/false);
  np.tick();
  feed(np, stand_lock(np.st));
  EXPECT_EQ(takeover(np), R1RT_E_REFUSED);
  EXPECT_EQ(np.status().takeover_refusal, R1RT_TAKEOVER_NO_POLICY);

  Rig b;
  NEED_RIG(b);
  ASSERT_TRUE(b.enter());
  EXPECT_EQ(try_takeover(b, stand_lock(b.st)), R1RT_E_REFUSED);
  EXPECT_EQ(b.status().takeover_refusal, R1RT_TAKEOVER_NOT_MONITOR);
  EXPECT_EQ(r1rt_core_mode(b.core), R1RT_MODE_BALANCE);
}

TEST(CoreTakeover, TheGuardGateWhenTheProfileRequiresIt) {
  Rig r(true, [](r1rt_config& c) { c.takeover_require_guard = 1; });
  NEED_RIG(r);
  EXPECT_EQ(try_takeover(r, stand_lock(r.st)), R1RT_E_REFUSED);
  EXPECT_EQ(r.status().takeover_refusal, R1RT_TAKEOVER_GUARD);
  r1rt_heartbeat hb{};
  hb.seq = 1;
  hb.t_ns = r.t - 80 * kMs;  // a guard heartbeat 80 ms old (> 50 ms): absent
  r.segment().guard_heartbeat.write(hb);
  EXPECT_EQ(try_takeover(r, stand_lock(r.st)), R1RT_E_REFUSED);
  EXPECT_EQ(r.status().takeover_refusal, R1RT_TAKEOVER_GUARD);
  r.tick();
  hb.seq = 2;
  hb.t_ns = r.t;
  r.segment().guard_heartbeat.write(hb);
  feed(r, stand_lock(r.st));
  EXPECT_EQ(takeover(r), R1RT_OK);
}

TEST(CoreTakeover, MirrorRepublishesTheSnapshotThenBlendsExactlyIntoBalance) {
  Rig r;   // the takeover
  Rig r2;  // the same core entering BALANCE directly at the release instant: "ours", exactly
  NEED_RIG(r);
  NEED_RIG(r2);
  const r1rt_lowcmd_out b = stand_lock(r.st, 0.03);
  ASSERT_EQ(try_takeover(r, b), R1RT_OK);
  r1rt_core_status s = r.status();
  EXPECT_EQ(s.takeover_refusal, R1RT_TAKEOVER_OK);
  EXPECT_EQ(s.takeover_used_mirror, 1);
  EXPECT_NEAR(s.mirror_dev_rad, 0.03, 1e-12);
  EXPECT_EQ(owner(r), R1RT_OWNER_R1RT);
  r1rt_lowcmd_out later = stand_lock(r.st, 0.1);  // the built-in moves on during MIRROR: the snapshot stays
  for (int i = 0; i < 50; ++i) {
    r.tick();
    feed(r, later);  // the built-in still publishes during MIRROR (it has not released yet): expected, no alarm
    ASSERT_EQ(r.out.publish, 1);
    for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
      ASSERT_EQ(r.out.q[j], b.q[j]) << j;
      ASSERT_EQ(r.out.kp[j], b.kp[j]) << j;
      ASSERT_EQ(r.out.kd[j], b.kd[j]) << j;
      ASSERT_EQ(r.out.enable[j], 1);
    }
    ASSERT_EQ(r.status().faults & (R1RT_FAULT_TAKEOVER_ALARM | R1RT_FAULT_BUILTIN_ACTIVE), 0u);
    ASSERT_EQ(r.status().blend_progress, 0.0);
  }
  // ReleaseMode succeeded (r1d): BLEND from the snapshot; r2 enters BALANCE at the same instant on the same state
  r2.t = r.t - kTick;
  ASSERT_TRUE(r2.enter());
  ASSERT_EQ(r2.t, r.t);
  ASSERT_EQ(r1rt_core_request(r.core, R1RT_REQ_TAKEOVER_RELEASED, r.t), R1RT_OK);
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_TAKEOVER_BLEND);
  const std::int64_t t0 = r.t;
  r.k = 0;
  const int n_blend = static_cast<int>(kBlendNs / kTick);
  double max_step = 0.0;
  double prev[R1RT_NUM_JOINTS];
  std::memcpy(prev, b.q, sizeof prev);
  for (int i = 1; i <= n_blend + 20; ++i) {
    r.tick();
    r2.tick();
    ASSERT_EQ(r.out.publish, 1);
    ASSERT_EQ(r2.out.publish, 1);
    const double a = std::fmin(std::fmax(static_cast<double>(r.t - t0) / static_cast<double>(kBlendNs), 0.0), 1.0);
    for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
      const double want = (1.0 - a) * b.q[j] + a * r2.out.q[j];
      ASSERT_NEAR(r.out.q[j], want, 1e-12) << "tick " << i << " joint " << j;
      ASSERT_NEAR(r.out.kp[j], (1.0 - a) * b.kp[j] + a * r2.out.kp[j], 1e-9) << j;
      ASSERT_NEAR(r.out.kd[j], (1.0 - a) * b.kd[j] + a * r2.out.kd[j], 1e-12) << j;
      max_step = std::fmax(max_step, std::fabs(r.out.q[j] - prev[j]));
      prev[j] = r.out.q[j];
    }
    s = r.status();
    ASSERT_NE(s.mode, R1RT_MODE_DAMP);
    ASSERT_NEAR(s.blend_progress, a, 1e-12);
    r1rt_heartbeat hb{};
    ASSERT_TRUE(r.segment().rt_heartbeat.try_read(hb));
    ASSERT_NEAR(hb.blend_progress, a, 1e-6);
    r1rt_mirror mi{};
    ASSERT_TRUE(r.segment().mirror.try_read(mi));
    ASSERT_NEAR(mi.blend_progress, a, 1e-6);
    ASSERT_EQ(mi.mode, s.mode);
    if (i < n_blend) {
      ASSERT_EQ(s.mode, R1RT_MODE_TAKEOVER_BLEND) << i;
    } else {
      ASSERT_EQ(s.mode, R1RT_MODE_BALANCE) << i;
    }
  }
  // bumpless: the largest per-tick change of a commanded joint over the whole sequence (the policy's own steps
  // included) stays small; the snapshot -> ours distance is spread over blend_s
  EXPECT_LT(max_step, 0.02);
  s = r.status();
  EXPECT_GT(s.policy_thread_runs, 10u);  // the policy thread ran during the blend and after
  EXPECT_EQ(s.faults_seen & (R1RT_FAULT_TAKEOVER_ALARM | R1RT_FAULT_POLICY_INVALID), 0u);
  EXPECT_EQ(s.takeover_alarm_ticks, 0u);
}

TEST(CoreTakeover, WithoutAUsableBuiltinCommandMirrorHoldsTheMeasuredPose) {
  Rig r;
  NEED_RIG(r);
  r1rt_lowcmd_out far = stand_lock(r.st, 0.7);  // 0.7 rad from the measured pose (> 0.6): not mirrored
  ASSERT_EQ(try_takeover(r, far), R1RT_OK);
  EXPECT_EQ(r.status().takeover_used_mirror, 0);
  r.tick();
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    EXPECT_EQ(r.out.q[j], r.st.q[j]);
    EXPECT_EQ(r.out.kp[j], r.cfg.stand_kp[j]);
    EXPECT_EQ(r.out.kd[j], r.cfg.stand_kd[j]);
  }
  Rig d;
  NEED_RIG(d);
  r1rt_lowcmd_out off = stand_lock(d.st);
  off.enable[20] = 0;  // a motor disabled in the built-in's command: not mirrored
  ASSERT_EQ(try_takeover(d, off), R1RT_OK);
  EXPECT_EQ(d.status().takeover_used_mirror, 0);
}

TEST(CoreTakeover, AbortGoesBackToMonitorReturnsTheTokenAndRaisesTheAlarm) {
  Rig r;
  NEED_RIG(r);
  ASSERT_EQ(try_takeover(r, stand_lock(r.st)), R1RT_OK);
  for (int i = 0; i < 10; ++i) r.tick();
  EXPECT_EQ(r.out.publish, 1);
  ASSERT_EQ(r1rt_core_request(r.core, R1RT_REQ_TAKEOVER_ABORT, r.t), R1RT_OK);
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_MONITOR);
  EXPECT_EQ(owner(r), R1RT_OWNER_NONE);
  r.tick();
  EXPECT_EQ(r.out.publish, 0);  // nothing of ours competes with the built-in that still drives the motors
  r1rt_core_status s = r.status();
  EXPECT_NE(s.faults & R1RT_FAULT_TAKEOVER_ALARM, 0u);
  EXPECT_NE(s.faults_seen & R1RT_FAULT_TAKEOVER_ALARM, 0u);
  EXPECT_EQ(s.takeover_aborts, 1u);
  r.tick();
  EXPECT_EQ(r.status().faults & R1RT_FAULT_TAKEOVER_ALARM, 0u);  // one-shot
  // a second takeover works after an abort
  ASSERT_EQ(try_takeover(r, stand_lock(r.st)), R1RT_OK);
  ASSERT_EQ(r1rt_core_request(r.core, R1RT_REQ_TAKEOVER_RELEASED, r.t), R1RT_OK);
  // ABORT is refused once the built-in has released (going back to MONITOR would leave the robot limp)
  EXPECT_EQ(r1rt_core_request(r.core, R1RT_REQ_TAKEOVER_ABORT, r.t), R1RT_E_REFUSED);
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_TAKEOVER_BLEND);
  // and the takeover requests are refused out of sequence
  EXPECT_EQ(r1rt_core_request(r.core, R1RT_REQ_TAKEOVER_RELEASED, r.t), R1RT_E_REFUSED);
  EXPECT_EQ(r1rt_core_request(r.core, R1RT_REQ_ENTER_BALANCE, r.t), R1RT_E_REFUSED);
}

TEST(CoreTakeover, AFaultDuringMirrorKeepsMirroringWithAnAlarm) {
  Rig r(true, [](r1rt_config& c) { c.takeover_mirror_alarm_ns = 400 * kMs; });
  NEED_RIG(r);
  const r1rt_lowcmd_out b = stand_lock(r.st);
  ASSERT_EQ(try_takeover(r, b), R1RT_OK);
  for (int i = 0; i < 75; ++i) {  // 150 ms without lowstate: STALE
    r.tick(/*withhold=*/true);
    ASSERT_EQ(r.out.publish, 1);
    for (int j = 0; j < R1RT_NUM_JOINTS; ++j) ASSERT_EQ(r.out.q[j], b.q[j]);
  }
  r1rt_core_status s = r.status();
  EXPECT_NE(s.faults & R1RT_FAULT_LOWSTATE_STALE, 0u);
  EXPECT_NE(s.faults & R1RT_FAULT_TAKEOVER_ALARM, 0u);
  EXPECT_EQ(s.mode, R1RT_MODE_TAKEOVER_MIRROR);
  r.tick();  // state back: no alarm, still mirroring
  EXPECT_EQ(r.status().faults & R1RT_FAULT_TAKEOVER_ALARM, 0u);
  for (int i = 0; i < 150; ++i) r.tick();  // MIRROR for > 400 ms without RELEASED / ABORT: alarm, nothing else
  s = r.status();
  EXPECT_NE(s.faults & R1RT_FAULT_TAKEOVER_ALARM, 0u);
  EXPECT_EQ(s.mode, R1RT_MODE_TAKEOVER_MIRROR);
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) EXPECT_EQ(r.out.q[j], b.q[j]);
  ASSERT_EQ(r1rt_core_request(r.core, R1RT_REQ_TAKEOVER_RELEASED, r.t), R1RT_OK);
  for (int i = 0; i < 520; ++i) r.tick();
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_BALANCE);
}

TEST(CoreTakeover, FaultsDuringTheBlendFinishItIntoBalance) {
  Rig r;
  NEED_RIG(r);
  ASSERT_EQ(try_takeover(r, stand_lock(r.st)), R1RT_OK);
  r.tick();
  ASSERT_EQ(r1rt_core_request(r.core, R1RT_REQ_TAKEOVER_RELEASED, r.t), R1RT_OK);
  r.k = 0;
  for (int i = 0; i < 40; ++i) r.tick(false, /*thread=*/false);  // the policy thread stalls 80 ms into the blend
  r1rt_core_status s = r.status();
  EXPECT_EQ(s.mode, R1RT_MODE_TAKEOVER_BLEND);
  EXPECT_NE(s.faults & R1RT_FAULT_POLICY_LATE, 0u);  // rung 1: inline
  EXPECT_NE(s.faults & R1RT_FAULT_TAKEOVER_ALARM, 0u);
  EXPECT_GE(s.inline_entries, 1u);
  r1rt_lowcmd_out before{};
  for (int i = 0; i < 60; ++i) {  // 120 ms without lowstate: the last blended command is resent
    r.tick(/*withhold=*/true);
    if (i == 49) before = r.out;
  }
  EXPECT_NE(r.status().faults & R1RT_FAULT_LOWSTATE_STALE, 0u);
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) EXPECT_EQ(r.out.q[j], before.q[j]);
  EXPECT_EQ(r.out.publish, 1);
  for (int i = 0; i < 450; ++i) r.tick();  // state and the thread back; the blend ends at 1 s
  s = r.status();
  EXPECT_EQ(s.mode, R1RT_MODE_BALANCE);
  EXPECT_EQ(s.faults_seen & R1RT_FAULT_POLICY_INVALID, 0u);
  EXPECT_GT(s.takeover_alarm_ticks, 0u);
}

TEST(CoreTakeover, AnInvalidActionDuringTheBlendFreezesFromTheBlendedCommandNeverDamps) {
  Rig r;
  NEED_RIG(r);
  ASSERT_EQ(try_takeover(r, stand_lock(r.st)), R1RT_OK);
  r.tick();
  ASSERT_EQ(r1rt_core_request(r.core, R1RT_REQ_TAKEOVER_RELEASED, r.t), R1RT_OK);
  r.k = 0;
  ASSERT_EQ(r1rt_core_inject(r.core, R1RT_INJECT_POLICY_NAN, r.t + 50 * kMs, r.t + 10'000 * kMs, 0.0), R1RT_OK);
  r1rt_lowcmd_out last_blend{};
  bool froze = false;
  for (int i = 0; i < 300 && !froze; ++i) {
    r.tick();
    if (r1rt_core_mode(r.core) == R1RT_MODE_STAND_FREEZE) {
      froze = true;
      // continuity: the freeze ramp starts at the last blended command (one ramp step away at most)
      for (int j = 0; j < 14; ++j) EXPECT_NEAR(r.out.q[j], last_blend.q[j], 0.01) << j;
    } else {
      last_blend = r.out;
    }
  }
  ASSERT_TRUE(froze);
  for (int i = 0; i < 300; ++i) {
    r.tick();
    ASSERT_NE(r1rt_core_mode(r.core), R1RT_MODE_DAMP);
    ASSERT_EQ(r.out.publish, 1);
  }
  const r1rt_core_status s = r.status();
  EXPECT_NE(s.faults_seen & R1RT_FAULT_POLICY_INVALID, 0u);
  EXPECT_NE(s.faults_seen & R1RT_FAULT_TAKEOVER_ALARM, 0u);
  EXPECT_EQ(s.mode, R1RT_MODE_STAND_FREEZE);
}

TEST(CoreTakeover, TheBuiltinStillWritingAfterTheReleaseIsAnAlarmOnly) {
  Rig r;
  NEED_RIG(r);
  const r1rt_lowcmd_out b = stand_lock(r.st);
  ASSERT_EQ(try_takeover(r, b), R1RT_OK);
  r.tick();
  feed(r, b);
  ASSERT_EQ(r1rt_core_request(r.core, R1RT_REQ_TAKEOVER_RELEASED, r.t), R1RT_OK);
  r.k = 0;
  for (int i = 0; i < 520; ++i) {
    r.tick();
    // the release did not take: two writers on rt/lowcmd (a sample received after the release; the one fed at the
    // release instant itself is the last of the built-in's own, not two writers)
    if (i > 0) {
      ASSERT_NE(r.status().faults & R1RT_FAULT_BUILTIN_ACTIVE, 0u) << i;
    }
    feed(r, b);
  }
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_BALANCE);  // nothing automatic: the operator decides
  for (int i = 0; i < 60; ++i) r.tick();  // the built-in stops (> 100 ms): the alarm clears
  EXPECT_EQ(r.status().faults & R1RT_FAULT_BUILTIN_ACTIVE, 0u);
}

TEST(CoreTakeover, TheOperatorCanDampDuringTheTakeoverNothingElseDoes) {
  Rig r;
  NEED_RIG(r);
  ASSERT_EQ(try_takeover(r, stand_lock(r.st)), R1RT_OK);
  r.tick();
  ASSERT_EQ(r1rt_core_request(r.core, R1RT_REQ_DAMP, r.t), R1RT_OK);  // the operator's damp wins (v2 too)
  r.tick();
  EXPECT_EQ(r1rt_core_mode(r.core), R1RT_MODE_DAMP);
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) EXPECT_EQ(r.out.kp[j], 0.0);
}

}  // namespace
