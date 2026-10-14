// Unit tests of the built-in arm model (DESIGN §3.4; arm_model.hpp). The comparison with MuJoCo itself (mj_rne /
// mj_kinematics on 100,000 random poses) is tests/unit/test_arm_model.py through the binding; these tests check the
// model on its own: the gravity torque against the fingertip FK (virtual work), symmetries, bad input, determinism and
// that nothing allocates.
#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>
#include <random>

#include "r1rt/arm_model.hpp"

// ---------------------------------------------------------------- a counting global operator new (this test binary)
static std::atomic<long> g_news{0};
void* operator new(std::size_t n) {
  g_news.fetch_add(1);
  if (void* p = std::malloc(n == 0 ? 1 : n)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
// The nothrow / array forms as well, so EVERY new / delete pair of this binary goes through malloc / free: libstdc++'s
// std::stable_sort (inside GoogleTest) uses the nothrow form, and with only the plain forms replaced ASan reports
// alloc-dealloc-mismatch (its own nothrow new, our free) - session 6, the Linux build check (GCC 9 / libstdc++).
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
  g_news.fetch_add(1, std::memory_order_relaxed);
  return std::malloc(n ? n : 1);
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept { return ::operator new(n, std::nothrow); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }

namespace {

using namespace r1rt::arm;

constexpr double kG[3] = {0.0, 0.0, -9.81};

void random_arm(std::mt19937_64& rng, double q5[5], double h7[7]) {
  std::uniform_real_distribution<double> u(-1.5, 1.5), uh(-1.0, 1.0);
  for (int k = 0; k < 5; ++k) q5[k] = u(rng);
  for (int k = 0; k < 7; ++k) h7[k] = uh(rng);
}

TEST(ArmModel, GeneratedTableIsConsistent) {
  EXPECT_EQ(std::strlen(kModelDigest), 64u);
  ASSERT_GT(kNumNodes, 0);
  ASSERT_LE(kNumNodes, kMaxNodes);
  EXPECT_EQ(kNodes[0].parent, -1);
  EXPECT_EQ(kNodes[0].end, kNumNodes);
  for (int i = 1; i < kNumNodes; ++i) {
    EXPECT_LT(kNodes[i].parent, i) << "parents first";
    EXPECT_GT(kNodes[i].end, i);
    EXPECT_LE(kNodes[i].end, kNodes[kNodes[i].parent].end) << "a subtree inside its parent's";
    if (kNodes[i].input >= 0) {
      EXPECT_NEAR(std::sqrt(kNodes[i].axis[0] * kNodes[i].axis[0] + kNodes[i].axis[1] * kNodes[i].axis[1] +
                            kNodes[i].axis[2] * kNodes[i].axis[2]),
                  1.0, 1e-12);
    }
  }
  for (int s = 0; s < 2; ++s)
    for (int k = 0; k < 5; ++k) EXPECT_EQ(kNodes[kArmJointNode[s][k]].input, kArmInput0[s] + k);
  EXPECT_EQ(kNodes[kHeadJointNode[0]].input, kHeadInput0);
  EXPECT_EQ(kNodes[kHeadJointNode[1]].input, kHeadInput0 + 1);
}

// The payload's share of the holding torque equals -m g . d(grasp centre)/dq (virtual work), the grasp centre taken
// from the fingertip FK by finite differences: the gravity code and the FK code agree with each other.
TEST(ArmModel, PayloadTorqueMatchesVirtualWorkOfTheGraspCentre) {
  std::mt19937_64 rng(7);
  for (int trial = 0; trial < 50; ++trial) {
    for (int s = 0; s < 2; ++s) {
      const Side side = static_cast<Side>(s);
      double q[5], h[7], t0[5], t1[5];
      random_arm(rng, q, h);
      ASSERT_TRUE(arm_gravity(side, kG, q, h, 0.0, t0));
      ASSERT_TRUE(arm_gravity(side, kG, q, h, 1.0, t1));
      for (int j = 0; j < 5; ++j) {
        const double e = 1e-6;
        double qp[5], qm[5];
        std::memcpy(qp, q, sizeof q);
        std::memcpy(qm, q, sizeof q);
        qp[j] += e;
        qm[j] -= e;
        HandPoints a, b;
        ASSERT_TRUE(hand_points_torso(side, qp, h, a));
        ASSERT_TRUE(hand_points_torso(side, qm, h, b));
        double w = 0.0;
        for (int k = 0; k < 3; ++k) w += kG[k] * (a.grasp_centre[k] - b.grasp_centre[k]) / (2 * e);
        EXPECT_NEAR(t1[j] - t0[j], -w, 1e-7) << "side " << s << " joint " << j;
      }
    }
  }
}

TEST(ArmModel, NoGravityNoTorqueAndLinearInGravity) {
  std::mt19937_64 rng(3);
  double q[5], h[7], t[5], t2[5];
  random_arm(rng, q, h);
  const double g0[3] = {0.0, 0.0, 0.0};
  ASSERT_TRUE(arm_gravity(kLeft, g0, q, h, 0.5, t));
  for (double v : t) EXPECT_EQ(v, 0.0);
  const double g2[3] = {0.0, 0.0, -19.62};
  ASSERT_TRUE(arm_gravity(kLeft, kG, q, h, 0.5, t));
  ASSERT_TRUE(arm_gravity(kLeft, g2, q, h, 0.5, t2));
  for (int k = 0; k < 5; ++k) EXPECT_NEAR(t2[k], 2.0 * t[k], 1e-12);
}

// The two arms are mirror images (the URDF's y -> -y symmetry; the hands' presets mirrored): the left arm's
// torques at q equal the right arm's at the mirrored pose up to the joints' axis signs - checked as |tau| equality
// at a pose with every roll / yaw joint at zero (the mirror leaves pitch joints alone).
TEST(ArmModel, ArmsAreMirrorImagesInTheSagittalPlane) {
  const double q[5] = {-0.7, 0.0, 0.0, 0.9, 0.0};
  double tl[5], tr[5];
  ASSERT_TRUE(arm_gravity(kLeft, kG, q, nullptr, 0.0, tl));
  ASSERT_TRUE(arm_gravity(kRight, kG, q, nullptr, 0.0, tr));
  EXPECT_NEAR(tl[0], tr[0], 1e-3) << "shoulder pitch";
  EXPECT_NEAR(tl[3], tr[3], 1e-3) << "elbow";
  EXPECT_GT(std::fabs(tl[0]), 0.5) << "a forward-held arm loads the shoulder pitch";
}

// Upright torso, head yaw axis along gravity: the yaw joint carries no gravity torque in any pose.
TEST(ArmModel, HeadYawCarriesNoGravityWhenUpright) {
  for (double yaw : {-1.0, 0.0, 0.7}) {
    const double qh[2] = {0.0, yaw};
    double t[2];
    ASSERT_TRUE(head_gravity(kG, qh, t));
    EXPECT_NEAR(t[1], 0.0, 1e-12);
  }
}

TEST(ArmModel, GravityInTorsoUprightAndTilted) {
  const double up[4] = {1.0, 0.0, 0.0, 0.0};
  double g[3];
  ASSERT_TRUE(gravity_in_torso(up, 0.0, 0.0, g));
  EXPECT_NEAR(std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]), 9.81, 1e-12);
  // the IMU reads a 10 deg forward pitch (about +y): the torso's gravity is the world's gravity through the pelvis
  // orientation R_world_imu * R_pelvis_imu^T - the IMU is mounted with the model's (estimated) offset, session 4
  // (~0.96 deg pitch / 0.50 roll), so the torso itself is pitched ~9.04 deg: g_torso = R_pelvis_imu R_world_imu^T g
  const double a = 10.0 * M_PI / 180.0;
  const double qp[4] = {std::cos(a / 2), 0.0, std::sin(a / 2), 0.0};
  ASSERT_TRUE(gravity_in_torso(qp, 0.0, 0.0, g));
  const double* R = kSites[kSiteImu].rot;  // the IMU in the pelvis frame (row-major)
  const double gi[3] = {9.81 * std::sin(a), 0.0, -9.81 * std::cos(a)};  // R_world_imu^T (0, 0, -9.81)
  for (int r = 0; r < 3; ++r) EXPECT_NEAR(g[r], R[3 * r] * gi[0] + R[3 * r + 1] * gi[1] + R[3 * r + 2] * gi[2], 1e-9);
  EXPECT_GT(std::atan2(g[0], -g[2]), 0.0);  // gravity gains a +x component: the torso leans forward
  // yaw does not change gravity in the body
  const double qy[4] = {std::cos(0.4), 0.0, 0.0, std::sin(0.4)};
  double gy[3];
  ASSERT_TRUE(gravity_in_torso(up, 0.0, 0.0, g));
  ASSERT_TRUE(gravity_in_torso(qy, 0.0, 0.0, gy));
  for (int k = 0; k < 3; ++k) EXPECT_NEAR(g[k], gy[k], 1e-12);
}

TEST(ArmModel, BadInputIsRefusedWithZeros) {
  double q[5] = {0.1, 0.2, 0.3, 0.4, 0.5}, t[5];
  const double nan = std::nan("");
  double qn[5] = {0.1, nan, 0.3, 0.4, 0.5};
  EXPECT_FALSE(arm_gravity(kLeft, kG, qn, nullptr, 0.0, t));
  for (double v : t) EXPECT_EQ(v, 0.0);
  EXPECT_FALSE(arm_gravity(kLeft, kG, q, nullptr, -0.1, t)) << "negative payload";
  EXPECT_FALSE(arm_gravity(static_cast<Side>(2), kG, q, nullptr, 0.0, t)) << "no such side";
  const double gn[3] = {0.0, nan, -9.81};
  EXPECT_FALSE(arm_gravity(kRight, gn, q, nullptr, 0.0, t));
  double g[3];
  const double q0[4] = {0.0, 0.0, 0.0, 0.0};
  EXPECT_FALSE(gravity_in_torso(q0, 0.0, 0.0, g)) << "zero quaternion";
  double q26[26] = {}, tau[26];
  const double up[4] = {1.0, 0.0, 0.0, 0.0};
  q26[3] = nan;
  EXPECT_FALSE(gravity_ff26(q26, up, nullptr, nullptr, nullptr, tau));
  for (double v : tau) EXPECT_EQ(v, 0.0);
  BasePoints bp;
  EXPECT_FALSE(fingertips_base(q26, up, nullptr, nullptr, bp));
  EXPECT_EQ(bp.lowest_tip_z[0], 0.0);
}

TEST(ArmModel, GravityFf26WritesOnlyTheUpperBody) {
  double q26[26] = {}, tau[26];
  q26[14] = -0.5;  // left shoulder pitch forward
  q26[24] = 0.3;   // head pitched
  const double up[4] = {1.0, 0.0, 0.0, 0.0};
  const double pay[2] = {0.2, 0.0};
  ASSERT_TRUE(gravity_ff26(q26, up, nullptr, nullptr, pay, tau));
  for (int j = 0; j < 14; ++j) EXPECT_EQ(tau[j], 0.0) << j;
  double t5[5], g[3];
  ASSERT_TRUE(gravity_in_torso(up, 0.0, 0.0, g));
  ASSERT_TRUE(arm_gravity(kLeft, g, q26 + 14, nullptr, 0.2, t5));
  for (int k = 0; k < 5; ++k) EXPECT_EQ(tau[14 + k], t5[k]);
  EXPECT_NE(tau[24], 0.0);
}

TEST(ArmModel, FingertipsBaseStandingPose) {
  // feet flat, legs straight, arms down: the lowest foot site defines z = 0, the fingertips hang well above it
  double q26[26] = {};
  const double up[4] = {1.0, 0.0, 0.0, 0.0};
  BasePoints bp;
  ASSERT_TRUE(fingertips_base(q26, up, nullptr, nullptr, bp));
  EXPECT_NEAR(std::fmin(bp.feet[0][2], bp.feet[1][2]), 0.0, 1e-12);
  for (int s = 0; s < 2; ++s) {
    EXPECT_GT(bp.lowest_tip_z[s], 0.2);
    EXPECT_LT(bp.lowest_tip_z[s], 1.0);
  }
  EXPECT_GT(bp.hand[0].palm[1], 0.0) << "the left hand on the left (+y)";
  EXPECT_LT(bp.hand[1].palm[1], 0.0);
}

TEST(ArmModel, DeterministicAndAllocationFree) {
  std::mt19937_64 rng(11);
  double q26[26], hl[7], hr[7], tau_a[26], tau_b[26];
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  for (double& v : q26) v = u(rng);
  for (double& v : hl) v = u(rng);
  for (double& v : hr) v = u(rng);
  const double quat[4] = {0.98, 0.05, -0.1, 0.12};
  const double pay[2] = {0.1, 0.3};
  BasePoints a, b;
  double s40[kNumInputs] = {}, sites[kNumSiteDefs][3];
  const long n0 = g_news.load();
  for (int i = 0; i < 1000; ++i) {
    ASSERT_TRUE(gravity_ff26(q26, quat, hl, hr, pay, tau_a));
    ASSERT_TRUE(fingertips_base(q26, quat, hl, hr, a));
    ASSERT_TRUE(sites_pelvis(s40, sites));
  }
  EXPECT_EQ(g_news.load(), n0) << "the arm model called operator new";
  ASSERT_TRUE(gravity_ff26(q26, quat, hl, hr, pay, tau_b));
  ASSERT_TRUE(fingertips_base(q26, quat, hl, hr, b));
  for (int j = 0; j < 26; ++j) EXPECT_EQ(tau_a[j], tau_b[j]);
  for (int s = 0; s < 2; ++s) EXPECT_EQ(a.lowest_tip_z[s], b.lowest_tip_z[s]);
}

}  // namespace
