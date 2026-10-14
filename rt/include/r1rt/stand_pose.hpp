// The STAND_FREEZE stand pose for the CURRENT arms (session 10, stale; docs/sim/STATE_AGE.md "Session 10: the freeze
// pose"). STAND_FREEZE ramps the legs + waist to the configured stand (`stand_q`: the real-profile model's stand with
// its CoM centred over the feet FOR THE ARMS IN stand_q, r1sim.model.centred_stand_pose), while the arms and the head
// only decelerate to HOLD wherever they are. With the arms forward (the table_ready posture) the whole-body CoM then
// sits ~6 mm ahead of where the stand was centred and the frozen robot tilts 5.7-6.5 deg instead of 3.6 (the session-9 /
// 10 sweeps; above B7's 5 deg line); one ankle-pitch shift for both ankles that puts the CoM back where the configured
// stand had it reads 2.4-2.5 deg in the same runs.
//
// The geometry is the core's built-in model (arm_model.hpp's generated tree: the real-profile MJCF's bodies with their
// masses and CoMs, the foot sites), walked as MuJoCo's mj_kinematics does (the same evaluation as arm_model.cpp's
// `eval`; kept local here so the arm model's API stays as it is). Session 11: the tree carries the backpack (0.454 kg)
// and the rear mass (1.6 kg) too (tools/gen_arm_model.py), so the shift matches MuJoCo's on the full model within 2 %
// (session 10 without them: ~10 % larger, table_ready +0.0115 vs +0.0105 rad). Header-only, no allocation, nothing
// thrown, constant time (~8 tree walks of 45 bodies, run once when a freeze starts - not every tick).
#pragma once

#include <cmath>
#include <initializer_list>

#include "r1rt/arm_model.hpp"

namespace r1rt::stand {
namespace detail {

struct Tree {
  double R[arm::kMaxNodes][9];
  double p[arm::kMaxNodes][3];
};

inline void mm(const double* A, const double* B, double* C) noexcept {  // C = A B (3x3 row-major)
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) C[3 * i + j] = A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j];
}
inline void mv(const double* A, const double* x, double* y) noexcept {  // y = A x
  for (int i = 0; i < 3; ++i) y[i] = A[3 * i] * x[0] + A[3 * i + 1] * x[1] + A[3 * i + 2] * x[2];
}
inline void rodrigues(const double* a, double t, double* R) noexcept {
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

// Every body frame in the PELVIS frame (node 0 = the pelvis) for the 40 inputs (q26 + both hands).
inline void walk(const double q40[arm::kNumInputs], Tree& f) noexcept {
  for (int k = 0; k < 9; ++k) f.R[0][k] = (k % 4 == 0) ? 1.0 : 0.0;
  for (int k = 0; k < 3; ++k) f.p[0][k] = 0.0;
  for (int i = 1; i < arm::kNumNodes; ++i) {
    const arm::Node& n = arm::kNodes[i];
    const int pa = n.parent;
    double t[3];
    mv(f.R[pa], n.pos, t);
    double R0[9];
    mm(f.R[pa], n.rot, R0);
    const double p0[3] = {f.p[pa][0] + t[0], f.p[pa][1] + t[1], f.p[pa][2] + t[2]};
    if (n.input < 0) {
      for (int k = 0; k < 9; ++k) f.R[i][k] = R0[k];
      for (int k = 0; k < 3; ++k) f.p[i][k] = p0[k];
      continue;
    }
    double anchor[3];
    mv(R0, n.jpos, anchor);
    for (int k = 0; k < 3; ++k) anchor[k] += p0[k];
    double J[9];
    rodrigues(n.axis, q40[n.input] - n.qref, J);
    mm(R0, J, f.R[i]);
    double r[3];
    mv(f.R[i], n.jpos, r);
    for (int k = 0; k < 3; ++k) f.p[i][k] = anchor[k] - r[k];
  }
}

}  // namespace detail

// The whole robot's CoM minus the two foot sites' midpoint, x in the LEFT foot site's frame (forward along the sole):
// independent of the pelvis pose, and = the world x over the feet when the soles are level (what
// r1sim.model.centred_stand_pose drives to 0). The hands at the model's open preset. False (x = 0) on bad input.
inline bool com_over_feet_x(const double q26[26], double* x_out) noexcept {
  *x_out = 0.0;
  double q40[arm::kNumInputs];
  for (int j = 0; j < 26; ++j) {
    if (!std::isfinite(q26[j])) return false;
    q40[j] = q26[j];
  }
  for (int s = 0; s < 2; ++s)
    for (int k = 0; k < 7; ++k) q40[arm::kHandInput0[s] + k] = arm::kHandOpen[s][k];
  detail::Tree f;
  detail::walk(q40, f);
  double m = 0.0, c[3] = {0.0, 0.0, 0.0};
  for (int i = 0; i < arm::kNumNodes; ++i) {
    const arm::Node& n = arm::kNodes[i];
    double w[3];
    detail::mv(f.R[i], n.ipos, w);
    for (int k = 0; k < 3; ++k) c[k] += n.mass * (f.p[i][k] + w[k]);
    m += n.mass;
  }
  if (!(m > 0.0)) return false;
  double mid[3] = {0.0, 0.0, 0.0};
  for (int s : {arm::kSiteFootLeft, arm::kSiteFootRight}) {
    const arm::SiteDef& sd = arm::kSites[s];
    double v[3];
    detail::mv(f.R[sd.node], sd.pos, v);
    for (int k = 0; k < 3; ++k) mid[k] += 0.5 * (f.p[sd.node][k] + v[k]);
  }
  const arm::SiteDef& sl = arm::kSites[arm::kSiteFootLeft];
  double Rs[9];
  detail::mm(f.R[sl.node], sl.rot, Rs);
  const double d[3] = {c[0] / m - mid[0], c[1] / m - mid[1], c[2] / m - mid[2]};
  *x_out = Rs[0] * d[0] + Rs[3] * d[1] + Rs[6] * d[2];  // (Rs^T d).x
  return std::isfinite(*x_out);
}

inline constexpr int kAnkleLeft = 4;   // SDK order (r1sim.joints): left_ankle_pitch
inline constexpr int kAnkleRight = 10;  // right_ankle_pitch
inline constexpr int kUpper0 = 14;      // arms + head: joints 14..25

// The ankle-pitch shift (rad, added to BOTH ankle pitches of stand_q26) that keeps the whole-body CoM's offset over the
// feet what the configured stand has with its own arms, when the arms + head are at `upper_now12` (joints 14..25)
// instead. Exactly 0 when upper_now12 equals stand_q26's arms + head. 0 on non-finite input, no convergence (10 um),
// or a shift beyond +-max_shift (the configured stand is used then - the old behaviour).
inline double ankle_shift_for_arms(const double stand_q26[26], const double upper_now12[12],
                                   double max_shift) noexcept {
  double q[26];
  for (int j = 0; j < 26; ++j) q[j] = stand_q26[j];
  double x_ref = 0.0;
  if (!com_over_feet_x(q, &x_ref)) return 0.0;
  bool same = true;
  for (int k = 0; k < 12; ++k) {
    if (!std::isfinite(upper_now12[k])) return 0.0;
    same = same && upper_now12[k] == stand_q26[kUpper0 + k];
    q[kUpper0 + k] = upper_now12[k];
  }
  if (same) return 0.0;
  auto f = [&](double d, double* out) noexcept {
    q[kAnkleLeft] = stand_q26[kAnkleLeft] + d;
    q[kAnkleRight] = stand_q26[kAnkleRight] + d;
    double x = 0.0;
    const bool ok = com_over_feet_x(q, &x);
    *out = x - x_ref;
    return ok;
  };
  double d0 = 0.0, d1 = 0.01, f0 = 0.0, f1 = 0.0;  // the secant method (f is nearly linear in d: ~ -0.6 m/rad)
  if (!f(d0, &f0) || !f(d1, &f1)) return 0.0;
  for (int it = 0; it < 6; ++it) {
    if (std::fabs(f1) < 1e-5) break;
    if (f1 == f0) return 0.0;
    const double d2 = d1 - f1 * (d1 - d0) / (f1 - f0);
    d0 = d1;
    f0 = f1;
    d1 = d2;
    if (!std::isfinite(d1) || std::fabs(d1) > 2.0 * max_shift || !f(d1, &f1)) return 0.0;
  }
  if (!(std::fabs(f1) < 1e-5) || !(std::fabs(d1) <= max_shift)) return 0.0;
  return d1;
}

}  // namespace r1rt::stand
