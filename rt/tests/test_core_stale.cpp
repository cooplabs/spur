// Session 9 (U19 / D25): the state-age limit + the "hold the stand" rung (docs/sim/STATE_AGE.md).
// r1rt::ContentAgeClock reads a late sample's real age from the board tick; the core raises R1RT_FAULT_STATE_OLD and
// counts sustained episodes in every setting, and only with stale_rung 1 / 2 freezes a BALANCING robot into the stand
// (never limp), back to BALANCE by the operator (1) or by itself once the state is live again (2).
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include "r1rt/stand_pose.hpp"
#include "r1rt/state_age.hpp"
#include "test_core_rig.hpp"

namespace {

// A 1 kHz board tick received every 2 ms, `late_ms` behind (a backlog: fresh receive stamps, old content).
void feed(r1rt::ContentAgeClock& c, std::int64_t t0, int ms, int late_ms = 0, double ppm = 0.0) {
  for (int k = 0; k <= ms; k += 2) {
    const std::int64_t t = t0 + k * kMs;
    const double board_ms = static_cast<double>(t / kMs - late_ms) * (1.0 + ppm * 1e-6);
    c.sample(static_cast<std::uint32_t>(board_ms), t);
  }
}

void stale_on(r1rt_config& c) { c.stale_rung = 1; }
void stale_return(r1rt_config& c) { c.stale_rung = 2; }

// One writer tick with the content `late_ms` old (fresh receive stamp, the board tick behind).
void tick_late(Rig& r, int late_ms) {
  r.t += kTick;
  r.st.t_rx_ns = r.t;
  r.st.board_tick = static_cast<std::uint32_t>((r.t - late_ms * kMs) / kMs);
  r1rt_core_step(r.core, &r.st, r.t, &r.out, &r.tel);
  ++r.k;
  if (r.k % 10 == 0) r1rt_core_policy_step(r.core, r.t);
}

bool legs_stiff(const r1rt_lowcmd_out& o) {
  for (int j = 0; j < 12; ++j)
    if (!(o.kp[j] > 10.0)) return false;
  return o.publish != 0;
}

}  // namespace

TEST(ContentAgeClock, FreshSamplesReadTheirReceiveAgeABacklogReadsItsLateness) {
  r1rt::ContentAgeClock c;
  c.configure(1000.0, 5 * kMs);
  feed(c, 0, 1000);
  ASSERT_TRUE(c.valid());
  EXPECT_NEAR(static_cast<double>(c.age_ns(1000 * kMs)), 0.0, 1.0 * kMs);
  EXPECT_NEAR(static_cast<double>(c.age_ns(1007 * kMs)), 7.0 * kMs, 1.0 * kMs);  // 7 ms with no new sample
  feed(c, 1002 * kMs, 1500, 30);                                                  // from here 30 ms late, 1.5 s
  EXPECT_NEAR(static_cast<double>(c.age_ns(2502 * kMs)), 30.0 * kMs, 1.0 * kMs);  // not absorbed as drift
}

TEST(ContentAgeClock, AStableStepIsRelearnedAfterTwoSecondsAGrowingLagIsNot) {
  r1rt::ContentAgeClock c;
  c.configure(1000.0, 5 * kMs);
  feed(c, 0, 1000);
  feed(c, 1002 * kMs, 1900, 30);  // 1.9 s of a constant 30 ms: still read old
  EXPECT_NEAR(static_cast<double>(c.age_ns(2902 * kMs)), 30.0 * kMs, 1.0 * kMs);
  feed(c, 2904 * kMs, 200, 30);  // past 2 s: a new time base
  EXPECT_LT(c.age_ns(3104 * kMs), 2 * kMs);
  EXPECT_EQ(c.relearns(), 1u);
  r1rt::ContentAgeClock g;  // a lag that keeps growing (a reader slower than the samples): never relearned
  g.configure(1000.0, 5 * kMs);
  feed(g, 0, 1000);
  for (int k = 0; k <= 4000; k += 2) {
    const std::int64_t t = (1002 + k) * kMs;
    g.sample(static_cast<std::uint32_t>(1002 + k - 10 - k / 100), t);  // 10 ms late, +1 ms every 100 ms
  }
  EXPECT_GT(g.age_ns(5002 * kMs), 45 * kMs);
  EXPECT_EQ(g.relearns(), 0u);
}

TEST(ContentAgeClock, ClockDriftAndDuplicateTicksStayNearZero) {
  r1rt::ContentAgeClock c;
  c.configure(1000.0, 5 * kMs);
  feed(c, 0, 20000, 0, -500.0);  // the board 500 ppm slow: +0.5 ms/s of offset, followed by the baseline
  EXPECT_LT(c.age_ns(20000 * kMs), 2 * kMs);
  c.sample(19999, 20001 * kMs);  // a duplicate tick (r1rt receives ~1052 samples/s of a 1 kHz tick)
  EXPECT_TRUE(c.valid());
}

TEST(ContentAgeClock, AStuckTickIsNoClockAndAWildJumpRebases) {
  r1rt::ContentAgeClock c;
  c.configure(1000.0, 5 * kMs);
  feed(c, 0, 500);
  ASSERT_TRUE(c.valid());
  for (int k = 2; k <= 40; k += 2) c.sample(500, (500 + k) * kMs);  // the tick stops while samples arrive
  EXPECT_FALSE(c.valid());
  r1rt::ContentAgeClock d;
  d.configure(1000.0, 5 * kMs);
  feed(d, 0, 500);
  d.sample(4'000'000'000u, 502 * kMs);  // a board restart: start over (invalid until enough samples)
  EXPECT_FALSE(d.valid());
}

TEST(ContentAgeClock, AnOlderSampleStepsBackAndReadsOld) {
  r1rt::ContentAgeClock c;
  c.configure(1000.0, 5 * kMs);
  feed(c, 0, 1000);
  c.sample(970, 1002 * kMs);  // the newest sample is from 32 ms ago (a late / reordered sample)
  ASSERT_TRUE(c.valid());
  EXPECT_NEAR(static_cast<double>(c.age_ns(1002 * kMs)), 32.0 * kMs, 1.0 * kMs);
}

TEST(CoreStale, RungOffByDefaultAlarmsAndCountsButKeepsBalancing) {
  Rig r;
  NEED_RIG(r);
  ASSERT_TRUE(r.enter());
  for (int i = 0; i < 100; ++i) r.tick();
  EXPECT_EQ(r.status().faults & R1RT_FAULT_STATE_OLD, 0u);
  EXPECT_EQ(r.status().content_age_valid, 1);
  for (int i = 0; i < 100; ++i) tick_late(r, 30);  // 200 ms of 30 ms-old content
  const r1rt_core_status s = r.status();
  EXPECT_NE(s.faults & R1RT_FAULT_STATE_OLD, 0u);
  EXPECT_EQ(s.state_old_episodes, 1u);
  EXPECT_NEAR(static_cast<double>(s.state_content_age_ns), 30.0 * kMs, 2.0 * kMs);
  EXPECT_EQ(s.mode, R1RT_MODE_BALANCE);  // the rung is OFF: an alarm only
  EXPECT_EQ(s.stale_rung_entries, 0u);
  EXPECT_TRUE(legs_stiff(r.out));
}

TEST(CoreStale, HoldTheStandFreezesAfterTheSustainTimeNeverLimp) {
  Rig r(true, stale_on);
  NEED_RIG(r);
  ASSERT_TRUE(r.enter());
  for (int i = 0; i < 100; ++i) r.tick();
  const std::int64_t t0 = r.t;  // the last sample
  std::int64_t t_freeze = 0;
  for (int i = 0; i < 60; ++i) {  // 120 ms without a sample
    r.tick(true);
    EXPECT_TRUE(legs_stiff(r.out));  // never limp
    if (t_freeze == 0 && r.status().mode == R1RT_MODE_STAND_FREEZE) t_freeze = r.t;
  }
  ASSERT_NE(t_freeze, 0);
  // age > 10 ms from t0 + 12 ms (the first tick over the limit), sustained 60 ms
  EXPECT_GE(t_freeze - t0, 70 * kMs);
  EXPECT_LE(t_freeze - t0, 74 * kMs);
  r1rt_core_status s = r.status();
  EXPECT_EQ(s.stale_freeze, 1);
  EXPECT_EQ(s.stale_rung_entries, 1u);
  EXPECT_EQ(s.faults & R1RT_FAULT_POLICY_INVALID, 0u);  // the reason is the state, not the policy
  for (int i = 0; i < 500; ++i) r.tick();  // the state is back: rung 1 stays frozen (the operator decides)
  s = r.status();
  EXPECT_EQ(s.mode, R1RT_MODE_STAND_FREEZE);
  EXPECT_EQ(s.stale_rung_returns, 0u);
  for (int j = 0; j < 12; ++j) EXPECT_NEAR(r.out.q[j], r.cfg.stand_q[j], 1e-9);  // at the stand after the ramp
  EXPECT_EQ(r1rt_core_request(r.core, R1RT_REQ_ENTER_BALANCE, r.t), R1RT_OK);  // the operator's way back
  r.tick();
  EXPECT_EQ(r.status().mode, R1RT_MODE_BALANCE);
  EXPECT_EQ(r.status().stale_freeze, 0);
}

TEST(CoreStale, AShortGapDoesNotFreeze) {
  Rig r(true, stale_on);
  NEED_RIG(r);
  ASSERT_TRUE(r.enter());
  for (int i = 0; i < 100; ++i) r.tick();
  for (int i = 0; i < 30; ++i) r.tick(true);  // 60 ms: over the limit for ~50 ms < the 60 ms sustain
  for (int i = 0; i < 50; ++i) r.tick();
  const r1rt_core_status s = r.status();
  EXPECT_EQ(s.mode, R1RT_MODE_BALANCE);
  EXPECT_EQ(s.stale_rung_entries, 0u);
  EXPECT_EQ(s.state_old_episodes, 0u);
  EXPECT_GT(s.state_old_ticks, 0u);
}

TEST(CoreStale, LateContentWithFreshStampsTriggersTheRung) {
  Rig r(true, stale_on);
  NEED_RIG(r);
  ASSERT_TRUE(r.enter());
  for (int i = 0; i < 100; ++i) r.tick();
  int n = 0;
  while (r.status().mode == R1RT_MODE_BALANCE && n < 100) {
    tick_late(r, 30);  // the receive age stays 0: only the board tick shows it
    ++n;
  }
  EXPECT_EQ(r.status().mode, R1RT_MODE_STAND_FREEZE);
  EXPECT_GE(n, 30);  // 60 ms sustained at 2 ms per tick
  EXPECT_LE(n, 32);
  EXPECT_LT(r.status().lowstate_age_ns, 3 * kMs);
}

TEST(CoreStale, ReturnModeComesBackToBalanceOnceTheStateIsLive) {
  Rig r(true, stale_return);
  NEED_RIG(r);
  ASSERT_TRUE(r.enter());
  for (int i = 0; i < 100; ++i) r.tick();
  for (int i = 0; i < 60; ++i) r.tick(true);
  ASSERT_EQ(r.status().mode, R1RT_MODE_STAND_FREEZE);
  int back = -1;
  for (int i = 0; i < 600 && back < 0; ++i) {
    r.tick();
    if (r.status().mode == R1RT_MODE_BALANCE) back = i;
  }
  ASSERT_GE(back, 0);
  // live from the first tick; BALANCE after max(the ramp's rest, 500 ms live)
  EXPECT_GE(back, 249);
  EXPECT_LE(back, 252);
  EXPECT_EQ(r.status().stale_rung_returns, 1u);
  EXPECT_EQ(r.status().stale_freeze, 0);
}

TEST(CoreStale, OnlyABalancingRobotIsFrozen) {
  Rig r(true, stale_on);
  NEED_RIG(r);
  // MONITOR: nothing published, nothing frozen, the alarm still counts
  for (int i = 0; i < 100; ++i) r.tick();
  for (int i = 0; i < 60; ++i) r.tick(true);
  const r1rt_core_status s = r.status();
  EXPECT_EQ(s.mode, R1RT_MODE_MONITOR);
  EXPECT_EQ(s.stale_rung_entries, 0u);
  EXPECT_EQ(s.state_old_episodes, 1u);
}

TEST(CoreStale, ConfigRefusesAnUnknownRung) {
  Rig r(true, [](r1rt_config& c) { c.stale_rung = 3; });
  EXPECT_FALSE(r.ok);
  EXPECT_EQ(r.why.rfind("create: ", 0), 0u);
}

// ---- session 10 (stale): the freeze's stand pose for the CURRENT arms (r1rt/stand_pose.hpp; STATE_AGE.md) ----
// Goldens from MuJoCo on the real-profile model (r1sim.model.build_model("real"), the model the arm model is
// generated from): the ankle-pitch shift that keeps the whole-body CoM over the feet where the rig's stand (the policy
// default + ankles -0.066) has it, with the arms at the 9 Oct table_ready posture (sim/r1sim/dds_bridge.py
// ARMS_TABLE_READY) = +0.010523 rad; both shoulder pitches 0.6 rad forward = +0.046558 rad. Session 10's built-in tree
// lacked two bodies of the full model - the backpack (0.454 kg) and the rear mass (1.6 kg, session 5's CoM fix), 2.05 kg
// behind the pelvis - and read ~10 % larger (+0.01154 / +0.05105 rad, kTreeTol 0.12); session 11 exports them
// (tools/gen_arm_model.py: the user kept the arm-aware freeze pose for HW2) -> within 2 % (fails at 0.02 on 1e55bdd).

namespace {

constexpr double kTableReadyArms[10] = {0.0689, 0.685, 0.0192, -0.5269, -1.5534,
                                        0.2079, -0.1853, 0.0042, 0.8141, 0.0039};
constexpr double kShiftTableReady = 0.010523;
constexpr double kShiftArmsForward = 0.046558;
constexpr double kTreeTol = 0.02;  // session 11: the tree has the backpack + the rear mass (above; was 0.12)

void table_ready_arms(Rig& r) {
  for (int k = 0; k < 10; ++k) r.st.q[14 + k] = kTableReadyArms[k];
}

}  // namespace

TEST(FreezeStandPose, TheAnkleShiftForTheArmsMatchesMuJoCo) {
  Rig r(true, stale_on);  // its config's stand_q only
  NEED_RIG(r);
  double upper[12];
  for (int k = 0; k < 12; ++k) upper[k] = r.cfg.stand_q[14 + k];
  EXPECT_EQ(r1rt::stand::ankle_shift_for_arms(r.cfg.stand_q, upper, 0.08), 0.0);  // the stand's own arms: exactly 0
  for (int k = 0; k < 10; ++k) upper[k] = kTableReadyArms[k];
  EXPECT_NEAR(r1rt::stand::ankle_shift_for_arms(r.cfg.stand_q, upper, 0.08), kShiftTableReady,
              kTreeTol * kShiftTableReady);
  for (int k = 0; k < 12; ++k) upper[k] = r.cfg.stand_q[14 + k];
  upper[0] -= 0.6;  // both shoulder pitches forward (positive pitch = the limb BACK, JOINT_DIRECTIONS.md)
  upper[5] -= 0.6;
  EXPECT_NEAR(r1rt::stand::ankle_shift_for_arms(r.cfg.stand_q, upper, 0.08), kShiftArmsForward,
              kTreeTol * kShiftArmsForward);
  EXPECT_EQ(r1rt::stand::ankle_shift_for_arms(r.cfg.stand_q, upper, 0.02), 0.0);  // beyond the bound: no shift
  upper[3] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(r1rt::stand::ankle_shift_for_arms(r.cfg.stand_q, upper, 0.08), 0.0);  // bad input: no shift
}

TEST(FreezeStandPose, TheStaleFreezeStandsTheLegsForTheArmsItHolds) {
  Rig r(true, stale_on);
  NEED_RIG(r);
  table_ready_arms(r);  // measured at the entry -> the arms HOLD there
  ASSERT_TRUE(r.enter());
  for (int i = 0; i < 100; ++i) r.tick();
  for (int i = 0; i < 60; ++i) {  // 120 ms without a sample -> the rung's freeze
    r.tick(true);
    EXPECT_TRUE(legs_stiff(r.out));  // never limp
  }
  ASSERT_EQ(r.status().mode, R1RT_MODE_STAND_FREEZE);
  for (int i = 0; i < 300; ++i) r.tick();  // the ramp (0.4 s) is over
  EXPECT_NEAR(r.out.q[4] - r.cfg.stand_q[4], kShiftTableReady, kTreeTol * kShiftTableReady);  // base: 0
  EXPECT_NEAR(r.out.q[10] - r.cfg.stand_q[10], kShiftTableReady, kTreeTol * kShiftTableReady);
  for (int j = 0; j < 14; ++j) {
    if (j != 4 && j != 10) {
      EXPECT_NEAR(r.out.q[j], r.cfg.stand_q[j], 1e-9);  // every other leg / waist joint: stand_q
    }
  }
  for (int k = 0; k < 10; ++k) EXPECT_NEAR(r.out.q[14 + k], kTableReadyArms[k], 0.02);  // the arms held where they were
  EXPECT_TRUE(legs_stiff(r.out));
}

TEST(FreezeStandPose, TheOperatorsFreezeOnTheStandKeepsTheConfiguredStand) {
  // a HANGING robot (ON_STAND): the freeze HW1 ran (S10) - the legs end at stand_q exactly, whatever the arms
  Rig r(true, [](r1rt_config& c) {
    for (int j = 0; j < 14; ++j) {
      c.on_stand_kp[j] = 150.0;
      c.on_stand_kd[j] = 4.0;
    }
  });
  NEED_RIG(r);
  table_ready_arms(r);
  auto builtin = [&r]() {
    r1rt_lowcmd_out b{};
    b.publish = 1;
    for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
      b.enable[j] = 1;
      b.q[j] = r.st.q[j] + 0.01;
      b.kd[j] = 3.0;
    }
    ASSERT_EQ(r1rt_core_set_builtin_cmd(r.core, &b, r.t), R1RT_OK);
  };
  r.tick();
  builtin();
  ASSERT_EQ(r1rt_core_request(r.core, R1RT_REQ_ON_STAND, r.t), R1RT_OK);
  for (int i = 0; i < 50; ++i) {
    r.tick();
    builtin();
  }
  ASSERT_EQ(r1rt_core_request(r.core, R1RT_REQ_TAKEOVER_RELEASED, r.t), R1RT_OK);
  for (int i = 0; i < 520; ++i) r.tick();
  ASSERT_EQ(r1rt_core_mode(r.core), R1RT_MODE_ON_STAND);
  ASSERT_EQ(r1rt_core_request(r.core, R1RT_REQ_STAND_FREEZE, r.t), R1RT_OK);
  for (int i = 0; i < 300; ++i) r.tick();
  ASSERT_EQ(r1rt_core_mode(r.core), R1RT_MODE_STAND_FREEZE);
  for (int j = 0; j < 14; ++j) EXPECT_NEAR(r.out.q[j], r.cfg.stand_q[j], 1e-9);
}

// ---- session 11 (core): the operator's way back from a freeze (`r1 balance`) judged on the TRUE state age ----
// S10's not-solid item (STATE_AGE.md "Session 10"): the board-tick clock re-based after 2 s of a steady lateness, so a
// state that stayed late through a freeze read "fresh" again and an operator's ENTER_BALANCE was accepted on it. Now a
// freeze holds the clock's relearn (state_age.hpp hold_relearn) and ENTER_BALANCE is refused while the state is older
// than the limit (the reason in the mailbox reply: R1RT_REFUSE_STATE). Each test below fails on 1e55bdd.

namespace {

// The operator's request through the mailbox, answered at the next writer tick - whose state is fresh (late_ms 0),
// `late_ms` old by its board tick (> 0) or withheld (< 0).
std::int32_t mailbox(Rig& r, std::int32_t req, std::int32_t* refusal, int late_ms = 0) {
  std::uint64_t seq = 0;
  if (r1rt_segment_request_mode(r.seg, req, R1RT_REQUESTER_OPERATOR, r.t, &seq) != R1RT_OK) return -1000;
  if (late_ms > 0) {
    tick_late(r, late_ms);
  } else {
    r.tick(late_ms < 0);
  }
  std::int32_t st = -1000, why = -1000;
  std::uint32_t mode = 0;
  if (r1rt_segment_mode_reply(r.seg, seq, &st, &why, &mode) != R1RT_OK) return -1001;
  *refusal = why;
  return st;
}

void tick_late_n(Rig& r, int n, int late_ms) {
  for (int i = 0; i < n; ++i) tick_late(r, late_ms);
}

}  // namespace

TEST(ContentAgeClock, AHeldClockNeverRelearnsSoAFrozenLateStateKeepsReadingLate) {
  r1rt::ContentAgeClock c;
  c.configure(1000.0, 5 * kMs);
  feed(c, 0, 1000);
  c.hold_relearn(true);           // the core: STAND_FREEZE
  feed(c, 1002 * kMs, 3000, 30);  // 3 s of a constant 30 ms: past the 2 s that re-based it on the base
  EXPECT_NEAR(static_cast<double>(c.age_ns(4002 * kMs)), 30.0 * kMs, 1.0 * kMs);
  EXPECT_EQ(c.relearns(), 0u);
  EXPECT_GE(c.relearns_held(), 1u);
  c.hold_relearn(false);  // out of the freeze: a new time base needs a FULL 2 s again
  feed(c, 4004 * kMs, 1900, 30);
  EXPECT_NEAR(static_cast<double>(c.age_ns(5904 * kMs)), 30.0 * kMs, 1.0 * kMs);
  EXPECT_EQ(c.relearns(), 0u);
  feed(c, 5906 * kMs, 200, 30);
  EXPECT_LT(c.age_ns(6106 * kMs), 2 * kMs);
  EXPECT_EQ(c.relearns(), 1u);
}

TEST(CoreStale, AFreezeNeverRebasesTheClockAndBalanceIsRefusedOnTheLateState) {
  Rig r(true, stale_on);
  NEED_RIG(r);
  ASSERT_TRUE(r.enter());
  for (int i = 0; i < 100; ++i) r.tick();
  int n = 0;
  while (r.status().mode == R1RT_MODE_BALANCE && n < 100) {  // 30 ms-old content: the rung freezes
    tick_late(r, 30);
    ++n;
  }
  ASSERT_EQ(r.status().mode, R1RT_MODE_STAND_FREEZE);
  tick_late_n(r, 1500, 30);  // 3 s more of the same lateness, frozen (the base re-based after 2 s: "fresh")
  r1rt_core_status s = r.status();
  EXPECT_EQ(s.content_age_relearns, 0u);
  EXPECT_NEAR(static_cast<double>(s.state_content_age_ns), 30.0 * kMs, 2.0 * kMs);
  EXPECT_NE(s.faults & R1RT_FAULT_STATE_OLD, 0u);
  std::int32_t why = 0;
  EXPECT_EQ(mailbox(r, R1RT_REQ_ENTER_BALANCE, &why, 30), R1RT_E_REFUSED);  // the operator's `r1 balance`: not on it
  EXPECT_EQ(why, R1RT_REFUSE_STATE);
  EXPECT_EQ(r.status().mode, R1RT_MODE_STAND_FREEZE);
  EXPECT_TRUE(legs_stiff(r.out));  // never limp
  for (int i = 0; i < 50; ++i) r.tick();  // the state is live again (fresh content)
  EXPECT_EQ(mailbox(r, R1RT_REQ_ENTER_BALANCE, &why), R1RT_OK);
  EXPECT_EQ(why, R1RT_REFUSE_NONE);
  r.tick();
  EXPECT_EQ(r.status().mode, R1RT_MODE_BALANCE);
  EXPECT_EQ(r.status().stale_freeze, 0);
}

TEST(CoreStale, TheOperatorsFreezeFromBalanceAndBackWithTheReasonOfARefusal) {
  Rig r;  // the shipped default: stale_rung 0 (HW2)
  NEED_RIG(r);
  ASSERT_TRUE(r.enter());
  for (int i = 0; i < 100; ++i) r.tick();
  std::int32_t why = -1;
  ASSERT_EQ(mailbox(r, R1RT_REQ_STAND_FREEZE, &why), R1RT_OK);  // `r1 freeze` (F5)
  EXPECT_EQ(r.status().mode, R1RT_MODE_STAND_FREEZE);
  for (int i = 0; i < 250; ++i) r.tick();  // the 0.4 s ramp and a little hold
  // no fresh state: refused with the STATE reason (the base said MODE for every refusal)
  for (int i = 0; i < 60; ++i) r.tick(true);
  EXPECT_EQ(mailbox(r, R1RT_REQ_ENTER_BALANCE, &why, -1), R1RT_E_REFUSED);
  EXPECT_EQ(why, R1RT_REFUSE_STATE);
  EXPECT_EQ(r.status().mode, R1RT_MODE_STAND_FREEZE);
  for (int i = 0; i < 50; ++i) r.tick();
  ASSERT_EQ(mailbox(r, R1RT_REQ_ENTER_BALANCE, &why), R1RT_OK);  // `r1 balance`
  r.tick();
  EXPECT_EQ(r.status().mode, R1RT_MODE_BALANCE);
  ASSERT_EQ(mailbox(r, R1RT_REQ_DAMP, &why), R1RT_OK);  // from DAMP: only through a takeover (MODE)
  EXPECT_EQ(mailbox(r, R1RT_REQ_ENTER_BALANCE, &why), R1RT_E_REFUSED);
  EXPECT_EQ(why, R1RT_REFUSE_MODE);
}
