// r1rt::PolicyMlp - the balance policy's network as fixed-size Eigen code (DESIGN.md §3.2).
//
// The live policy (R1MC-Manip2-Lean) is an observation normaliser + an MLP 74-512-256-128-14 with ELU, exported by
// mjlab / rsl_rl as ONNX. tools/policy_convert.py turns the ONNX into an "R1W1" weights file (format in that tool's
// docstring); this class loads it once and evaluates
//
//     x  = (obs - sub) / div                 (the graph's Sub / Div, not folded)
//     h0 = elu(W0 x  + b0)    512
//     h1 = elu(W1 h0 + b1)    256
//     h2 = elu(W2 h1 + b2)    128
//     act = W3 h2 + b3         14
//
// in float64, the same arithmetic as v2's NumPy executor (r1mc2/policy.py NumpyOnnx). Golden test: on the observation
// vectors of tests/assets/policy_golden (and 10,000 when R1_V2_DIR is set) the actions match v2's executor within
// 1e-5 absolute (rt/tests/test_policy_golden.cpp, tests/unit/test_policy_golden.py).
//
// Real-time rules: load() is the only function that allocates (one 64-byte aligned block holding every array);
// run() allocates nothing (no operator new, no Eigen malloc - checked by rt/tests/test_policy_mlp.cpp with a counting
// operator new and EIGEN_RUNTIME_NO_MALLOC), throws nothing and touches only the block and its stack (~7 KB). It keeps
// no mutable state, so concurrent calls on one loaded instance are safe.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace r1rt {

class PolicyMlp {
 public:
  static constexpr int kObs = 74;
  static constexpr int kH0 = 512;
  static constexpr int kH1 = 256;
  static constexpr int kH2 = 128;
  static constexpr int kAct = 14;
  static constexpr std::uint32_t kVersion = 1;

  PolicyMlp() noexcept = default;
  ~PolicyMlp();
  PolicyMlp(const PolicyMlp&) = delete;
  PolicyMlp& operator=(const PolicyMlp&) = delete;

  // Read an R1W1 weights file (tools/policy_convert.py). On failure returns false, writes the reason into err
  // (NUL-terminated, truncated to errlen) and keeps any previously loaded weights. Allocates.
  bool load(const char* path, char* err, std::size_t errlen) noexcept;

  bool loaded() const noexcept { return block_ != nullptr; }

  // One policy step: obs[kObs] -> act[kAct] (raw network output; the caller applies scale and default pose).
  // Precondition: loaded(). No allocation, no exceptions.
  void run(const double* obs, double* act) const noexcept;

  // sha256 of the ONNX file the weights were converted from (as recorded by the converter).
  const std::array<std::uint8_t, 32>& onnx_sha256() const noexcept { return sha256_; }
  double elu_alpha() const noexcept { return alpha_; }

 private:
  void release() noexcept;

  void* block_ = nullptr;  // one aligned allocation; the pointers below point into it
  const double* sub_ = nullptr;
  const double* div_ = nullptr;
  const double* w0_ = nullptr;
  const double* b0_ = nullptr;
  const double* w1_ = nullptr;
  const double* b1_ = nullptr;
  const double* w2_ = nullptr;
  const double* b2_ = nullptr;
  const double* w3_ = nullptr;
  const double* b3_ = nullptr;
  double alpha_ = 1.0;
  std::array<std::uint8_t, 32> sha256_{};
};

}  // namespace r1rt
