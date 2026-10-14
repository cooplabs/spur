// The RT core's built-in arm model (DESIGN.md §3.4; docs/architecture/ARM_MODEL.md): the arms' and the head's
// gravity torque (the executor's feed-forward: HOLD / TRAJECTORY without sag, FREEDRIVE) and the fingertip forward
// kinematics (the surface guard), evaluated from a data table that tools/gen_arm_model.py exports from the SAME
// compiled MJCF the simulator runs (r1sim.model.build_model("real")) into rt/src/arm_model_gen.cpp. No general
// dynamics library: a fixed tree of hinge joints walked in a few hundred flops.
//
// Every function here allocates nothing, throws nothing, runs in constant time and is safe to call from the writer
// thread (no globals are written). Inputs are not trusted: a non-finite or unit-less input makes the function return
// false and write zeros (never NaN) - the executor then treats the feed-forward as absent.
//
// Frames (all right-handed, metres, radians):
//   torso   the torso_link body frame (after waist roll + yaw): where the arm chains start. Gravity torques depend on
//           the direction of gravity in this frame only - the caller passes it (g_torso), or gravity_in_torso()
//           derives it from the IMU and the waist joints.
//   pelvis  the pelvis body frame; the IMU (site "imu") is mounted in it with the exported rotation = the model's
//           IMU-to-pelvis mounting offset (session 4: r1sim.model.IMU_MOUNT_OFFSET_DEG, nominal 0.50 roll / 0.96 deg
//           pitch, an ESTIMATE from six sessions' logs until the HW1 IMU-level step); the functions below undo it, so
//           they return the pelvis's true orientation. When HW1 calibrates it, set the model's offset and regenerate.
//   base    v2's base frame (arms.py ArmKin.sync; r1sim.tier_a.a9_short_grasps.base_frame): the IMU's roll / pitch on
//           the pelvis (yaw removed, v2 controller.rp_matrix), origin at the two foot sites' midpoint on the LOWER
//           sole, x along the feet's mean yaw, z up. A table's / board's heights (r1rt_traj_guards.surface_z_m) are
//           declared in this frame.
// Joint inputs: q26 in the SDK order of r1sim.joints.JOINTS; hands (7,) in Dex3 DDS order (thumb_0, thumb_1,
// thumb_2, middle_0, middle_1, index_0, index_1). A null hand pointer = the open preset (kHandOpen).
#pragma once

#include <cstdint>

namespace r1rt::arm {

enum Side : int { kLeft = 0, kRight = 1 };

inline constexpr int kMaxNodes = 64;
inline constexpr int kNumInputs = 40;  // q26 + the left hand's 7 + the right hand's 7
inline constexpr int kHandInput0[2] = {26, 33};
inline constexpr int kArmInput0[2] = {14, 19};
inline constexpr int kHeadInput0 = 24;

// The exported sites, in this fixed order (tools/gen_arm_model.py SITES; checked by the test).
enum SiteId : int {
  kSiteFootLeft = 0,
  kSiteFootRight = 1,
  kSitePalmLeft = 2,   // then thumb, middle, index tips of the same hand at +1, +2, +3
  kSitePalmRight = 6,
  kSiteImu = 10,
  kNumSiteDefs = 11,
};
inline constexpr int kSitePalm[2] = {kSitePalmLeft, kSitePalmRight};

// One body of the exported tree (MuJoCo's order: parents first, subtrees contiguous).
struct Node {
  int parent;       // -1 = the pelvis (the root)
  int end;          // one past the last node of this node's subtree
  double pos[3];    // body frame origin in the parent frame
  double rot[9];    // body frame in the parent frame, row-major
  int input;        // the joint's index into the 40-input vector, -1 = welded to the parent
  double axis[3];   // hinge axis (unit) in the body frame
  double jpos[3];   // hinge anchor in the body frame
  double qref;      // MuJoCo's qpos0: the joint rotates by (q - qref)
  double mass;      // kg
  double ipos[3];   // centre of mass in the body frame
};
struct SiteDef {
  int node;
  double pos[3];
  double rot[9];
};

// ---- the generated data (rt/src/arm_model_gen.cpp) ----
extern const char kModelDigest[];  // sha256 of the exported numbers (r1sim model identity)
extern const double kGravityWorld[3];
extern const int kNumNodes;
extern const int kTorsoNode;
extern const int kArmJointNode[2][5];
extern const int kHeadJointNode[2];
extern const double kHandOpen[2][7];
extern const Node kNodes[kMaxNodes];
extern const SiteDef kSites[kNumSiteDefs];

// Points of one hand: the palm site, the three fingertip sites and v2's grasp centre (api.py `_grasp_centre`:
// 0.5 thumb tip + 0.25 (index tip + middle tip), the open hand's aim point), in the frame the function names.
struct HandPoints {
  double palm[3];
  double tip[3][3];  // thumb, middle, index
  double grasp_centre[3];
};
struct BasePoints {
  HandPoints hand[2];       // base frame
  double lowest_tip_z[2];   // per hand: the lowest of its three fingertips (the surface guard's quantity)
  double feet[2][3];        // the foot sites in the base frame (z of the lower one = 0 by construction)
};

// Gravity torque on the 5 joints of one arm (shoulder pitch, roll, yaw, elbow, wrist roll) = the torque the motors
// must add to HOLD the pose (MuJoCo's qfrc_bias with zero velocity): the arm, the hand (posed by q_hand7 or the open
// preset) and a declared payload (kg, a point mass at the grasp centre). g_torso = gravity acceleration (m/s^2) in
// the torso frame. Returns false (tau_out = 0) on non-finite input or a payload < 0.
bool arm_gravity(Side side, const double g_torso[3], const double q_arm5[5], const double* q_hand7, double payload_kg,
                 double tau_out5[5]) noexcept;

// Gravity torque on the head's 2 joints (pitch, yaw).
bool head_gravity(const double g_torso[3], const double q_head2[2], double tau_out2[2]) noexcept;

// Gravity (kGravityWorld) in the torso frame from the IMU orientation (w, x, y, z; normalised here; yaw is
// irrelevant) and the waist joints. Returns false (g = 0) for a non-finite / zero quaternion.
bool gravity_in_torso(const double quat_imu_wxyz[4], double waist_roll, double waist_yaw,
                      double g_torso_out[3]) noexcept;

// What the core passes to UpperBodyExecutor::set_gravity_ff every tick: tau26_out[14..25] = both arms (payload_kg[2]
// per hand, may be null = none) and the head; [0..13] = 0. Returns false (all zeros) on bad input.
bool gravity_ff26(const double q26[26], const double quat_imu_wxyz[4], const double* q_hand_left7,
                  const double* q_hand_right7, const double* payload_kg2, double tau26_out[26]) noexcept;

// One hand's points in the TORSO frame (independent of the waist, legs and IMU).
bool hand_points_torso(Side side, const double q_arm5[5], const double* q_hand7, HandPoints& out) noexcept;

// Both hands' points in v2's BASE frame from the whole measured state: q26 (legs, waist, arms, head), the IMU and the
// hands. This is the surface guard's input: lowest_tip_z[hand] < surface_z_m (+ margin) = the guard trips.
bool fingertips_base(const double q26[26], const double quat_imu_wxyz[4], const double* q_hand_left7,
                     const double* q_hand_right7, BasePoints& out) noexcept;

// Every exported site in the PELVIS frame for a full 40-input vector (validation: compared with mj_kinematics).
bool sites_pelvis(const double q40[kNumInputs], double pos_out[kNumSiteDefs][3]) noexcept;

}  // namespace r1rt::arm
