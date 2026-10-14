// r1rt_policy_eval: run the C++ policy runner over a file of observations (validation tool, not real-time code).
//
//   r1rt_policy_eval <weights.r1w> <observations.r1g> [<actions_out.f64>]
//
// The observation file is the golden format "R1G1" written by tools/policy_golden.py (little-endian):
//   magic "R1G1", uint32 version 1, uint32 n, uint32 n_obs (74), uint32 n_act (14), uint32 n_sim (rows from the sim),
//   8 reserved bytes, then float64 obs[n][n_obs], then float64 expected_act[n][n_act] (v2's NumPy executor).
// Prints n, the max / mean absolute error against the expected actions and the time per inference; writes the C++
// actions (float64 [n][n_act], raw) when an output path is given. Exit 0 when max error <= 1e-5, 1 otherwise,
// 2 on a usage or file error.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "r1rt/policy_mlp.hpp"

namespace {

constexpr double kTol = 1e-5;

std::uint32_t rd_u32(const unsigned char* p) {
  return std::uint32_t{p[0]} | (std::uint32_t{p[1]} << 8) | (std::uint32_t{p[2]} << 16) | (std::uint32_t{p[3]} << 24);
}

bool read_golden(const char* path, std::vector<double>& obs, std::vector<double>& act, std::uint32_t& n,
                 std::uint32_t& n_sim) {
  std::FILE* f = std::fopen(path, "rb");
  if (f == nullptr) return false;
  unsigned char h[32];
  bool ok = std::fread(h, 1, 32, f) == 32 && std::memcmp(h, "R1G1", 4) == 0 && rd_u32(h + 4) == 1 &&
            rd_u32(h + 12) == r1rt::PolicyMlp::kObs && rd_u32(h + 16) == r1rt::PolicyMlp::kAct;
  if (ok) {
    n = rd_u32(h + 8);
    n_sim = rd_u32(h + 20);
    obs.resize(std::size_t{n} * r1rt::PolicyMlp::kObs);
    act.resize(std::size_t{n} * r1rt::PolicyMlp::kAct);
    ok = std::fread(obs.data(), sizeof(double), obs.size(), f) == obs.size() &&
         std::fread(act.data(), sizeof(double), act.size(), f) == act.size();
  }
  std::fclose(f);
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::fprintf(stderr, "usage: %s <weights.r1w> <observations.r1g> [<actions_out.f64>]\n", argv[0]);
    return 2;
  }
  r1rt::PolicyMlp mlp;
  char err[256];
  if (!mlp.load(argv[1], err, sizeof err)) {
    std::fprintf(stderr, "r1rt_policy_eval: %s\n", err);
    return 2;
  }
  std::vector<double> obs, want;
  std::uint32_t n = 0, n_sim = 0;
  if (!read_golden(argv[2], obs, want, n, n_sim)) {
    std::fprintf(stderr, "r1rt_policy_eval: cannot read R1G1 file %s\n", argv[2]);
    return 2;
  }
  std::vector<double> got(want.size());
  const auto t0 = std::chrono::steady_clock::now();
  for (std::uint32_t i = 0; i < n; ++i) {
    mlp.run(&obs[std::size_t{i} * r1rt::PolicyMlp::kObs], &got[std::size_t{i} * r1rt::PolicyMlp::kAct]);
  }
  const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
  double max_err = 0.0, sum_err = 0.0;
  for (std::size_t k = 0; k < got.size(); ++k) {
    const double e = std::fabs(got[k] - want[k]);
    if (!(e <= max_err)) max_err = e;  // NaN propagates
    sum_err += e;
  }
  std::printf("r1rt_policy_eval: n=%u (sim %u) max_abs_err=%.3e mean_abs_err=%.3e us_per_inference=%.2f\n", n, n_sim,
              max_err, got.empty() ? 0.0 : sum_err / static_cast<double>(got.size()),
              n ? us / static_cast<double>(n) : 0.0);
  if (argc == 4) {
    std::FILE* f = std::fopen(argv[3], "wb");
    if (f == nullptr || std::fwrite(got.data(), sizeof(double), got.size(), f) != got.size()) {
      std::fprintf(stderr, "r1rt_policy_eval: cannot write %s\n", argv[3]);
      if (f != nullptr) std::fclose(f);
      return 2;
    }
    std::fclose(f);
  }
  return max_err <= kTol ? 0 : 1;
}
