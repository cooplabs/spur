// The built-in arm model's evaluator (DESIGN.md §3.4; arm_model.hpp; docs/architecture/ARM_MODEL.md). The data
// (rt/src/arm_model_gen.cpp) is generated from the simulator's MJCF; this file walks it exactly as MuJoCo's
// mj_kinematics does (body frame = parent * (pos, rot); a hinge rotates about its anchor by q - qref), so the
// positions and the gravity torques equal mj_kinematics / mj_rne to rounding (tests/unit/test_arm_model.py:
// 100,000 random poses). Constant time, no allocation, no globals written.
#include "r1rt/arm_model.hpp"

#include <cmath>

namespace r1rt::arm {
namespace {

struct Frames {
  double R[kMaxNodes][9];  // node frame in the evaluation's root frame, row-major
  double p[kMaxNodes][3];
};

inline void matmul(const double* A, const double* B, double* C) noexcept {  // C = A B (3x3, row-major)
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) C[3 * i + j] = A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j];
}
inline void matvec(const double* A, const double* x, double* y) noexcept {  // y = A x
  for (int i = 0; i < 3; ++i) y[i] = A[3 * i] * x[0] + A[3 * i + 1] * x[1] + A[3 * i + 2] * x[2];
}
inline void mattvec(const double* A, const double* x, double* y) noexcept {  // y = A^T x
  for (int i = 0; i < 3; ++i) y[i] = A[i] * x[0] + A[3 + i] * x[1] + A[6 + i] * x[2];
}
inline void cross(const double* a, const double* b, double* c) noexcept {
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}
inline double dot(const double* a, const double* b) noexcept { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// Rotation by angle t about the unit axis a (Rodrigues), row-major.
inline void axis_angle(const double* a, double t, double* R) noexcept {
  const double c = std::cos(t), s = std::sin(t), v = 1.0 - c;
  const double x = a[0], y = a[1], z = a[2];
  R[0] = c + x * x * v;
  R[1] = x * y * v - z * s;
  R[2] = x * z * v + y * s;
  R[3] = y * x * v + z * s;
  R[4] = c + y * y * v;
  R[5] = y * z * v - x * s;
  R[6] = z * x * v - y * s;
  R[7] = z * y * v + x * s;
  R[8] = c + z * z * v;
}

bool finite(const double* x, int n) noexcept {
  for (int i = 0; i < n; ++i)
    if (!std::isfinite(x[i])) return false;
  return true;
}

// Frames of nodes [first, end) given their parents' frames already in f (MuJoCo mj_kinematics order).
void eval(const double q[kNumInputs], int first, int end, Frames& f) noexcept {
  for (int i = first; i < end; ++i) {
    const Node& n = kNodes[i];
    const int pa = n.parent;
    double t[3];
    matvec(f.R[pa], n.pos, t);
    double R0[9];
    matmul(f.R[pa], n.rot, R0);
    double p0[3] = {f.p[pa][0] + t[0], f.p[pa][1] + t[1], f.p[pa][2] + t[2]};
    if (n.input < 0) {
      for (int k = 0; k < 9; ++k) f.R[i][k] = R0[k];
      for (int k = 0; k < 3; ++k) f.p[i][k] = p0[k];
      continue;
    }
    double anchor[3];
    matvec(R0, n.jpos, anchor);
    for (int k = 0; k < 3; ++k) anchor[k] += p0[k];
    double J[9];
    axis_angle(n.axis, q[n.input] - n.qref, J);
    matmul(R0, J, f.R[i]);
    double r[3];
    matvec(f.R[i], n.jpos, r);
    for (int k = 0; k < 3; ++k) f.p[i][k] = anchor[k] - r[k];
  }
}

void set_identity(Frames& f, int node) noexcept {
  for (int k = 0; k < 9; ++k) f.R[node][k] = (k % 4 == 0) ? 1.0 : 0.0;
  for (int k = 0; k < 3; ++k) f.p[node][k] = 0.0;
}

void site_pos(const Frames& f, int site, double* out) noexcept {
  const SiteDef& s = kSites[site];
  matvec(f.R[s.node], s.pos, out);
  for (int k = 0; k < 3; ++k) out[k] += f.p[s.node][k];
}

void hand_points(const Frames& f, Side side, HandPoints& h) noexcept {
  const int s0 = kSitePalm[side];
  site_pos(f, s0, h.palm);
  for (int t = 0; t < 3; ++t) site_pos(f, s0 + 1 + t, h.tip[t]);
  for (int k = 0; k < 3; ++k)  // v2 api.py _grasp_centre: 0.5 thumb + 0.25 (index + middle)
    h.grasp_centre[k] = 0.5 * h.tip[0][k] + 0.25 * (h.tip[2][k] + h.tip[1][k]);
}

// Holding torque of the hinge at node j: tau = -a . ((S - M o) x g) with S / M the subtree's first moment / mass
// (+ the payload point), o the anchor and a the axis in the evaluation frame (MuJoCo's qfrc_bias at zero velocity).
double joint_gravity(const Frames& f, int j, const double g[3], double payload_kg, const double* payload_at) noexcept {
  const Node& n = kNodes[j];
  double S[3] = {0.0, 0.0, 0.0};
  double M = 0.0;
  for (int k = j; k < n.end; ++k) {
    double c[3];
    matvec(f.R[k], kNodes[k].ipos, c);
    const double m = kNodes[k].mass;
    for (int i = 0; i < 3; ++i) S[i] += m * (c[i] + f.p[k][i]);
    M += m;
  }
  if (payload_kg > 0.0 && payload_at != nullptr) {
    for (int i = 0; i < 3; ++i) S[i] += payload_kg * payload_at[i];
    M += payload_kg;
  }
  double o[3], a[3];
  matvec(f.R[j], n.jpos, o);
  for (int i = 0; i < 3; ++i) o[i] += f.p[j][i];
  matvec(f.R[j], n.axis, a);
  double r[3] = {S[0] - M * o[0], S[1] - M * o[1], S[2] - M * o[2]};
  double rg[3];
  cross(r, g, rg);
  return -dot(a, rg);
}

void zero(double* x, int n) noexcept {
  for (int i = 0; i < n; ++i) x[i] = 0.0;
}

// The 40-input vector for one arm / the head with everything else at zero (the evaluation starts at the torso).
void fill_hand(double q[kNumInputs], Side side, const double* q_hand7) noexcept {
  const double* h = q_hand7 != nullptr ? q_hand7 : kHandOpen[side];
  for (int k = 0; k < 7; ++k) q[kHandInput0[side] + k] = h[k];
}

bool quat_to_rot(const double qw[4], double R[9]) noexcept {
  if (!finite(qw, 4)) return false;
  const double nn = std::sqrt(dot(qw + 1, qw + 1) + qw[0] * qw[0]);
  if (!(nn > 1e-9)) return false;
  const double w = qw[0] / nn, x = qw[1] / nn, y = qw[2] / nn, z = qw[3] / nn;
  R[0] = 1 - 2 * (y * y + z * z);
  R[1] = 2 * (x * y - w * z);
  R[2] = 2 * (x * z + w * y);
  R[3] = 2 * (x * y + w * z);
  R[4] = 1 - 2 * (x * x + z * z);
  R[5] = 2 * (y * z - w * x);
  R[6] = 2 * (x * z - w * y);
  R[7] = 2 * (y * z + w * x);
  R[8] = 1 - 2 * (x * x + y * y);
  return true;
}

// The pelvis orientation in the world from the IMU reading: R_world_pelvis = R_world_imu * R_pelvis_imu^T.
bool pelvis_rot(const double quat_imu_wxyz[4], double R_wp[9]) noexcept {
  double R_wi[9];
  if (!quat_to_rot(quat_imu_wxyz, R_wi)) return false;
  const double* R_pi = kSites[kSiteImu].rot;  // the IMU in the pelvis frame
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      R_wp[3 * i + j] = R_wi[3 * i] * R_pi[3 * j] + R_wi[3 * i + 1] * R_pi[3 * j + 1] + R_wi[3 * i + 2] * R_pi[3 * j + 2];
  return true;
}

double yaw_of(const double R[9]) noexcept { return std::atan2(R[3], R[0]); }

}  // namespace

bool arm_gravity(Side side, const double g_torso[3], const double q_arm5[5], const double* q_hand7, double payload_kg,
                 double tau_out5[5]) noexcept {
  zero(tau_out5, 5);
  if ((side != kLeft && side != kRight) || !finite(g_torso, 3) || !finite(q_arm5, 5) ||
      (q_hand7 != nullptr && !finite(q_hand7, 7)) || !std::isfinite(payload_kg) || payload_kg < 0.0)
    return false;
  double q[kNumInputs];
  zero(q, kNumInputs);
  for (int k = 0; k < 5; ++k) q[kArmInput0[side] + k] = q_arm5[k];
  fill_hand(q, side, q_hand7);
  Frames f;
  set_identity(f, kTorsoNode);
  const int first = kArmJointNode[side][0];
  eval(q, first, kNodes[first].end, f);
  HandPoints h;
  hand_points(f, side, h);
  for (int k = 0; k < 5; ++k) tau_out5[k] = joint_gravity(f, kArmJointNode[side][k], g_torso, payload_kg, h.grasp_centre);
  return true;
}

bool head_gravity(const double g_torso[3], const double q_head2[2], double tau_out2[2]) noexcept {
  zero(tau_out2, 2);
  if (!finite(g_torso, 3) || !finite(q_head2, 2)) return false;
  double q[kNumInputs];
  zero(q, kNumInputs);
  q[kHeadInput0] = q_head2[0];
  q[kHeadInput0 + 1] = q_head2[1];
  Frames f;
  set_identity(f, kTorsoNode);
  const int first = kHeadJointNode[0];
  eval(q, first, kNodes[first].end, f);
  for (int k = 0; k < 2; ++k) tau_out2[k] = joint_gravity(f, kHeadJointNode[k], g_torso, 0.0, nullptr);
  return true;
}

bool gravity_in_torso(const double quat_imu_wxyz[4], double waist_roll, double waist_yaw,
                      double g_torso_out[3]) noexcept {
  zero(g_torso_out, 3);
  double R_wp[9];
  if (!std::isfinite(waist_roll) || !std::isfinite(waist_yaw) || !pelvis_rot(quat_imu_wxyz, R_wp)) return false;
  double q[kNumInputs];
  zero(q, kNumInputs);
  q[12] = waist_roll;
  q[13] = waist_yaw;
  Frames f;
  set_identity(f, 0);
  const int torso = kTorsoNode, waist = kNodes[torso].parent;
  eval(q, waist, waist + 1, f);
  eval(q, torso, torso + 1, f);
  double R_wt[9];
  matmul(R_wp, f.R[torso], R_wt);
  mattvec(R_wt, kGravityWorld, g_torso_out);
  return true;
}

bool gravity_ff26(const double q26[26], const double quat_imu_wxyz[4], const double* q_hand_left7,
                  const double* q_hand_right7, const double* payload_kg2, double tau26_out[26]) noexcept {
  zero(tau26_out, 26);
  if (!finite(q26, 26)) return false;
  double g[3];
  if (!gravity_in_torso(quat_imu_wxyz, q26[12], q26[13], g)) return false;
  const double* hands[2] = {q_hand_left7, q_hand_right7};
  for (int s = 0; s < 2; ++s) {
    const double pay = payload_kg2 != nullptr ? payload_kg2[s] : 0.0;
    if (!arm_gravity(static_cast<Side>(s), g, q26 + kArmInput0[s], hands[s], pay, tau26_out + kArmInput0[s])) {
      zero(tau26_out, 26);
      return false;
    }
  }
  if (!head_gravity(g, q26 + kHeadInput0, tau26_out + kHeadInput0)) {
    zero(tau26_out, 26);
    return false;
  }
  return true;
}

bool hand_points_torso(Side side, const double q_arm5[5], const double* q_hand7, HandPoints& out) noexcept {
  zero(&out.palm[0], 3);
  zero(&out.tip[0][0], 9);
  zero(&out.grasp_centre[0], 3);
  if ((side != kLeft && side != kRight) || !finite(q_arm5, 5) || (q_hand7 != nullptr && !finite(q_hand7, 7)))
    return false;
  double q[kNumInputs];
  zero(q, kNumInputs);
  for (int k = 0; k < 5; ++k) q[kArmInput0[side] + k] = q_arm5[k];
  fill_hand(q, side, q_hand7);
  Frames f;
  set_identity(f, kTorsoNode);
  const int first = kArmJointNode[side][0];
  eval(q, first, kNodes[first].end, f);
  hand_points(f, side, out);
  return true;
}

bool sites_pelvis(const double q40[kNumInputs], double pos_out[kNumSiteDefs][3]) noexcept {
  for (int s = 0; s < kNumSiteDefs; ++s) zero(pos_out[s], 3);
  if (!finite(q40, kNumInputs)) return false;
  Frames f;
  set_identity(f, 0);
  eval(q40, 1, kNumNodes, f);
  for (int s = 0; s < kNumSiteDefs; ++s) site_pos(f, s, pos_out[s]);
  return true;
}

bool fingertips_base(const double q26[26], const double quat_imu_wxyz[4], const double* q_hand_left7,
                     const double* q_hand_right7, BasePoints& out) noexcept {
  for (int s = 0; s < 2; ++s) {
    zero(&out.hand[s].palm[0], 3);
    zero(&out.hand[s].tip[0][0], 9);
    zero(&out.hand[s].grasp_centre[0], 3);
    zero(out.feet[s], 3);
    out.lowest_tip_z[s] = 0.0;
  }
  double R_wp[9];
  if (!finite(q26, 26) || (q_hand_left7 != nullptr && !finite(q_hand_left7, 7)) ||
      (q_hand_right7 != nullptr && !finite(q_hand_right7, 7)) || !pelvis_rot(quat_imu_wxyz, R_wp))
    return false;
  // v2 controller.rp_matrix: the IMU's roll and pitch only (yaw removed about the world z)
  const double yaw = yaw_of(R_wp), c = std::cos(-yaw), s = std::sin(-yaw);
  const double Rz[9] = {c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0};
  Frames f;
  matmul(Rz, R_wp, f.R[0]);
  zero(f.p[0], 3);
  double q[kNumInputs];
  for (int k = 0; k < 26; ++k) q[k] = q26[k];
  fill_hand(q, kLeft, q_hand_left7);
  fill_hand(q, kRight, q_hand_right7);
  eval(q, 1, kNumNodes, f);
  // v2 arms.py ArmKin.sync: origin at the foot sites' midpoint on the lower sole, x along the feet's mean yaw
  double pl[3], pr[3], Rl[9], Rr[9];
  site_pos(f, kSiteFootLeft, pl);
  site_pos(f, kSiteFootRight, pr);
  matmul(f.R[kSites[kSiteFootLeft].node], kSites[kSiteFootLeft].rot, Rl);
  matmul(f.R[kSites[kSiteFootRight].node], kSites[kSiteFootRight].rot, Rr);
  const double by = 0.5 * (yaw_of(Rl) + yaw_of(Rr)), cb = std::cos(by), sb = std::sin(by);
  const double Rb[9] = {cb, -sb, 0.0, sb, cb, 0.0, 0.0, 0.0, 1.0};
  const double o[3] = {0.5 * (pl[0] + pr[0]), 0.5 * (pl[1] + pr[1]), pl[2] < pr[2] ? pl[2] : pr[2]};
  auto to_base = [&](const double* x, double* y) {
    const double d[3] = {x[0] - o[0], x[1] - o[1], x[2] - o[2]};
    mattvec(Rb, d, y);
  };
  for (int sd = 0; sd < 2; ++sd) {
    HandPoints h;
    hand_points(f, static_cast<Side>(sd), h);
    to_base(h.palm, out.hand[sd].palm);
    for (int t = 0; t < 3; ++t) to_base(h.tip[t], out.hand[sd].tip[t]);
    to_base(h.grasp_centre, out.hand[sd].grasp_centre);
    double zmin = out.hand[sd].tip[0][2];
    for (int t = 1; t < 3; ++t) zmin = out.hand[sd].tip[t][2] < zmin ? out.hand[sd].tip[t][2] : zmin;
    out.lowest_tip_z[sd] = zmin;
  }
  to_base(pl, out.feet[0]);
  to_base(pr, out.feet[1]);
  return true;
}

}  // namespace r1rt::arm
