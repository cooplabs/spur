// r1rt_py: the built-in arm model (rt/include/r1rt/arm_model.hpp) for the simulator and the validation tests
// (tests/unit/test_arm_model.py compares it with MuJoCo's mj_rne / mj_kinematics). Batched: every function takes
// (N, k) float64 arrays and loops in C++, so 100,000 poses cost milliseconds. Simulation / validation only.
// Registered from r1rt_py.cpp's module init (one line: r1rt_bind_arm_model(m)).
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/tuple.h>

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>

#include "r1rt/arm_model.hpp"

namespace nb = nanobind;

void r1rt_bind_arm_model(nb::module_& m);

namespace {

using namespace r1rt::arm;
using In2 = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using In1 = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Out = nb::ndarray<nb::numpy, double>;

std::size_t rows(const In2& a, std::size_t cols, const char* what) {
  if (a.shape(1) != cols)
    throw nb::value_error((std::string(what) + ": expected shape (N, " + std::to_string(cols) + ")").c_str());
  return a.shape(0);
}
void same_rows(std::size_t n, std::size_t k, const char* what) {
  if (n != k) throw nb::value_error((std::string(what) + ": row count differs from the first argument").c_str());
}
const double* row_or_null(const std::optional<In2>& a, std::size_t i, std::size_t cols) {
  return a ? a->data() + i * cols : nullptr;
}
void check_opt(const std::optional<In2>& a, std::size_t n, std::size_t cols, const char* what) {
  if (a) same_rows(n, rows(*a, cols, what), what);
}
Out make(std::size_t n, std::initializer_list<std::size_t> tail) {
  std::size_t shape[4] = {n, 0, 0, 0};
  std::size_t total = n, nd = 1;
  for (std::size_t t : tail) {
    shape[nd++] = t;
    total *= t;
  }
  double* buf = new double[total == 0 ? 1 : total]();
  nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
  return Out(buf, nd, shape, owner);
}
[[noreturn]] void refused(const char* fn, std::size_t i) {
  throw nb::value_error((std::string(fn) + ": refused row " + std::to_string(i) + " (non-finite / invalid input)").c_str());
}

}  // namespace

void r1rt_bind_arm_model(nb::module_& m) {
  m.attr("ARM_MODEL_DIGEST") = nb::str(kModelDigest);
  m.attr("ARM_NUM_NODES") = nb::int_(kNumNodes);
  m.attr("ARM_SITES") = nb::make_tuple("left_foot", "right_foot", "left_hand_palm", "left_hand_thumb_tip",
                                       "left_hand_middle_tip", "left_hand_index_tip", "right_hand_palm",
                                       "right_hand_thumb_tip", "right_hand_middle_tip", "right_hand_index_tip", "imu");
  m.def(
      "arm_gravity",
      [](int side, const In2& g, const In2& q_arm, std::optional<In2> q_hand, const In1& payload) {
        const std::size_t n = rows(g, 3, "g_torso");
        same_rows(n, rows(q_arm, 5, "q_arm"), "q_arm");
        check_opt(q_hand, n, 7, "q_hand");
        same_rows(n, payload.shape(0), "payload");
        Out out = make(n, {5});
        double* o = out.data();
        for (std::size_t i = 0; i < n; ++i)
          if (!arm_gravity(static_cast<Side>(side), g.data() + 3 * i, q_arm.data() + 5 * i, row_or_null(q_hand, i, 7),
                           payload.data()[i], o + 5 * i))
            refused("arm_gravity", i);
        return out;
      },
      nb::arg("side"), nb::arg("g_torso"), nb::arg("q_arm"), nb::arg("q_hand").none(), nb::arg("payload"),
      "Holding torque (N, 5) of one arm (side 0 left / 1 right) for gravity g_torso (N, 3) in the torso frame");
  m.def(
      "head_gravity",
      [](const In2& g, const In2& q_head) {
        const std::size_t n = rows(g, 3, "g_torso");
        same_rows(n, rows(q_head, 2, "q_head"), "q_head");
        Out out = make(n, {2});
        for (std::size_t i = 0; i < n; ++i)
          if (!head_gravity(g.data() + 3 * i, q_head.data() + 2 * i, out.data() + 2 * i)) refused("head_gravity", i);
        return out;
      },
      nb::arg("g_torso"), nb::arg("q_head"), "Holding torque (N, 2) of the head (pitch, yaw)");
  m.def(
      "gravity_in_torso",
      [](const In2& quat, const In2& waist) {
        const std::size_t n = rows(quat, 4, "quat");
        same_rows(n, rows(waist, 2, "waist"), "waist");
        Out out = make(n, {3});
        for (std::size_t i = 0; i < n; ++i)
          if (!gravity_in_torso(quat.data() + 4 * i, waist.data()[2 * i], waist.data()[2 * i + 1], out.data() + 3 * i))
            refused("gravity_in_torso", i);
        return out;
      },
      nb::arg("quat_imu"), nb::arg("waist"), "Gravity (N, 3) in the torso frame from the IMU quaternion (w, x, y, z)");
  m.def(
      "gravity_ff26",
      [](const In2& q26, const In2& quat, std::optional<In2> hl, std::optional<In2> hr, std::optional<In2> pay) {
        const std::size_t n = rows(q26, 26, "q26");
        same_rows(n, rows(quat, 4, "quat"), "quat");
        check_opt(hl, n, 7, "q_hand_left");
        check_opt(hr, n, 7, "q_hand_right");
        check_opt(pay, n, 2, "payload");
        Out out = make(n, {26});
        for (std::size_t i = 0; i < n; ++i)
          if (!gravity_ff26(q26.data() + 26 * i, quat.data() + 4 * i, row_or_null(hl, i, 7), row_or_null(hr, i, 7),
                            row_or_null(pay, i, 2), out.data() + 26 * i))
            refused("gravity_ff26", i);
        return out;
      },
      nb::arg("q26"), nb::arg("quat_imu"), nb::arg("q_hand_left").none() = nb::none(),
      nb::arg("q_hand_right").none() = nb::none(), nb::arg("payload").none() = nb::none(),
      "The executor's feed-forward (N, 26): arms 14-23 and head 24-25, zeros elsewhere");
  m.def(
      "hand_points_torso",
      [](int side, const In2& q_arm, std::optional<In2> q_hand) {
        const std::size_t n = rows(q_arm, 5, "q_arm");
        check_opt(q_hand, n, 7, "q_hand");
        Out out = make(n, {5, 3});
        for (std::size_t i = 0; i < n; ++i) {
          HandPoints h;
          if (!hand_points_torso(static_cast<Side>(side), q_arm.data() + 5 * i, row_or_null(q_hand, i, 7), h))
            refused("hand_points_torso", i);
          double* o = out.data() + 15 * i;
          for (int k = 0; k < 3; ++k) {
            o[k] = h.palm[k];
            for (int t = 0; t < 3; ++t) o[3 * (1 + t) + k] = h.tip[t][k];
            o[12 + k] = h.grasp_centre[k];
          }
        }
        return out;
      },
      nb::arg("side"), nb::arg("q_arm"), nb::arg("q_hand").none() = nb::none(),
      "(N, 5, 3): palm, thumb / middle / index tips, grasp centre in the torso frame");
  m.def(
      "fingertips_base",
      [](const In2& q26, const In2& quat, std::optional<In2> hl, std::optional<In2> hr) {
        const std::size_t n = rows(q26, 26, "q26");
        same_rows(n, rows(quat, 4, "quat"), "quat");
        check_opt(hl, n, 7, "q_hand_left");
        check_opt(hr, n, 7, "q_hand_right");
        Out pts = make(n, {2, 5, 3}), low = make(n, {2}), feet = make(n, {2, 3});
        for (std::size_t i = 0; i < n; ++i) {
          BasePoints b;
          if (!fingertips_base(q26.data() + 26 * i, quat.data() + 4 * i, row_or_null(hl, i, 7), row_or_null(hr, i, 7),
                               b))
            refused("fingertips_base", i);
          for (int s = 0; s < 2; ++s) {
            double* o = pts.data() + 30 * i + 15 * static_cast<std::size_t>(s);
            for (int k = 0; k < 3; ++k) {
              o[k] = b.hand[s].palm[k];
              for (int t = 0; t < 3; ++t) o[3 * (1 + t) + k] = b.hand[s].tip[t][k];
              o[12 + k] = b.hand[s].grasp_centre[k];
              feet.data()[6 * i + 3 * static_cast<std::size_t>(s) + static_cast<std::size_t>(k)] = b.feet[s][k];
            }
            low.data()[2 * i + static_cast<std::size_t>(s)] = b.lowest_tip_z[s];
          }
        }
        return std::make_tuple(pts, low, feet);
      },
      nb::arg("q26"), nb::arg("quat_imu"), nb::arg("q_hand_left").none() = nb::none(),
      nb::arg("q_hand_right").none() = nb::none(),
      "v2's base frame: points (N, 2, 5, 3) [palm, thumb, middle, index, grasp centre], lowest tip z (N, 2), feet "
      "(N, 2, 3)");
  m.def(
      "sites_pelvis",
      [](const In2& q40) {
        const std::size_t n = rows(q40, kNumInputs, "q40");
        Out out = make(n, {kNumSiteDefs, 3});
        for (std::size_t i = 0; i < n; ++i) {
          double (*o)[3] = reinterpret_cast<double (*)[3]>(out.data() + 3 * kNumSiteDefs * i);
          if (!sites_pelvis(q40.data() + kNumInputs * i, o)) refused("sites_pelvis", i);
        }
        return out;
      },
      nb::arg("q40"), "Every exported site (N, 11, 3) in the pelvis frame for q26 + left hand 7 + right hand 7");
}
