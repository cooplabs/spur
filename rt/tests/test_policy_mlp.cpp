// r1rt::PolicyMlp: loading (and refusing bad files), the no-allocation rule of run(), time per inference.
// The numerical golden test against v2's NumPy executor is in test_policy_golden.cpp.
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <new>
#include <string>
#include <vector>

#include "r1rt/policy_mlp.hpp"

// ---------------------------------------------------------------- a counting global operator new (this test binary)
namespace {
std::atomic<long> g_news{0};
}  // namespace

void* operator new(std::size_t n) {
  g_news.fetch_add(1, std::memory_order_relaxed);
  if (void* p = std::malloc(n ? n : 1)) return p;
  throw std::bad_alloc();
}
void* operator new(std::size_t n, std::align_val_t al) {
  g_news.fetch_add(1, std::memory_order_relaxed);
  void* p = nullptr;
  const std::size_t a = static_cast<std::size_t>(al) < sizeof(void*) ? sizeof(void*) : static_cast<std::size_t>(al);
  if (posix_memalign(&p, a, n ? n : 1) == 0) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
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
void* operator new(std::size_t n, std::align_val_t al, const std::nothrow_t&) noexcept {
  g_news.fetch_add(1, std::memory_order_relaxed);
  void* p = nullptr;
  const std::size_t a = static_cast<std::size_t>(al) < sizeof(void*) ? sizeof(void*) : static_cast<std::size_t>(al);
  return posix_memalign(&p, a, n ? n : 1) == 0 ? p : nullptr;
}
void* operator new[](std::size_t n, std::align_val_t al) { return ::operator new(n, al); }
void operator delete(void* p, std::align_val_t, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }

namespace {

const char* kWeights = R1RT_POLICY_WEIGHTS;

bool have_weights() {
  std::ifstream f(kWeights, std::ios::binary);
  return f.good();
}

#define REQUIRE_WEIGHTS()                                                                               \
  if (!have_weights()) GTEST_SKIP() << "no converted weights at " << kWeights                           \
                                    << " (CMake found no Python with `onnx`; run it under `uv run`)"

std::vector<char> read_all(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

std::string write_tmp(const std::string& name, const std::vector<char>& bytes) {
  std::string path = testing::TempDir() + "r1rt_policy_" + name + ".r1w";
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  return path;
}

void patch_u32(std::vector<char>& b, std::size_t off, std::uint32_t v) { std::memcpy(&b[off], &v, 4); }

std::vector<double> some_obs(unsigned seed) {
  std::vector<double> x(r1rt::PolicyMlp::kObs);
  unsigned s = seed * 2654435761u + 1u;
  for (auto& v : x) {
    s = s * 1664525u + 1013904223u;
    v = (static_cast<double>(s >> 8) / 16777216.0 - 0.5) * 2.0;
  }
  return x;
}

}  // namespace

TEST(PolicyMlp, LoadsTheConvertedPolicy) {
  REQUIRE_WEIGHTS();
  r1rt::PolicyMlp mlp;
  char err[256] = {0};
  ASSERT_TRUE(mlp.load(kWeights, err, sizeof err)) << err;
  EXPECT_TRUE(mlp.loaded());
  EXPECT_EQ(mlp.elu_alpha(), 1.0);
  // The converter embeds the ONNX file's sha256 (policies/r1mc2_manip2_lean_20261006T0421Z/PROVENANCE.md).
  const unsigned char want[4] = {0x09, 0xbf, 0x2f, 0xd8};
  EXPECT_EQ(0, std::memcmp(mlp.onnx_sha256().data(), want, 4));
  std::vector<double> act(r1rt::PolicyMlp::kAct, std::nan(""));
  const auto obs = some_obs(1);
  mlp.run(obs.data(), act.data());
  for (double a : act) EXPECT_TRUE(std::isfinite(a));
}

TEST(PolicyMlp, RefusesBadFilesAndKeepsTheLoadedWeights) {
  REQUIRE_WEIGHTS();
  const auto good = read_all(kWeights);
  ASSERT_EQ(good.size(), 80u + 204578u * 8u);
  r1rt::PolicyMlp mlp;
  char err[256] = {0};
  ASSERT_TRUE(mlp.load(kWeights, err, sizeof err)) << err;
  std::vector<double> before(r1rt::PolicyMlp::kAct), after(r1rt::PolicyMlp::kAct);
  const auto obs = some_obs(2);
  mlp.run(obs.data(), before.data());

  struct Case {
    const char* name;
    std::vector<char> bytes;
    const char* reason;
  };
  std::vector<Case> cases;
  {
    auto b = good;
    b[0] = 'X';
    cases.push_back({"magic", b, "bad magic"});
  }
  {
    auto b = good;
    patch_u32(b, 4, 2);
    cases.push_back({"version", b, "version"});
  }
  {
    auto b = good;
    patch_u32(b, 12, 256);  // n_h0
    cases.push_back({"dims", b, "dims[1]"});
  }
  {
    auto b = good;
    patch_u32(b, 28, 2);
    cases.push_back({"activation", b, "not ELU"});
  }
  {
    auto b = good;
    b.resize(b.size() - 8);
    cases.push_back({"truncated", b, "bytes, expected"});
  }
  {
    auto b = good;
    b.push_back(0);
    cases.push_back({"trailing", b, "bytes, expected"});
  }
  {
    auto b = good;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::memcpy(&b[80 + 8 * (74 + 74 + 100)], &nan, 8);  // inside W0
    cases.push_back({"nan", b, "not finite"});
  }
  {
    auto b = good;
    const double zero = 0.0;
    std::memcpy(&b[80 + 8 * (74 + 3)], &zero, 8);  // div[3]
    cases.push_back({"zero_div", b, "divisor 3 is zero"});
  }
  for (const auto& c : cases) {
    const std::string p = write_tmp(c.name, c.bytes);
    err[0] = 0;
    EXPECT_FALSE(mlp.load(p.c_str(), err, sizeof err)) << c.name;
    EXPECT_NE(std::string(err).find(c.reason), std::string::npos) << c.name << ": " << err;
    EXPECT_TRUE(mlp.loaded());
    std::remove(p.c_str());
  }
  EXPECT_FALSE(mlp.load("/nonexistent/r1rt/policy.r1w", err, sizeof err));
  mlp.run(obs.data(), after.data());
  EXPECT_EQ(before, after);  // the refused files did not touch the loaded weights
}

TEST(PolicyMlp, RunAllocatesNothing) {
  REQUIRE_WEIGHTS();
  r1rt::PolicyMlp mlp;
  char err[256] = {0};
  const long n0 = g_news.load();
  ASSERT_TRUE(mlp.load(kWeights, err, sizeof err)) << err;
  EXPECT_GE(g_news.load() - n0, 1) << "load() allocates its block through operator new";
  const auto obs = some_obs(3);
  std::vector<double> act(r1rt::PolicyMlp::kAct);
  mlp.run(obs.data(), act.data());  // warm-up
  const long before = g_news.load();
  Eigen::internal::set_is_malloc_allowed(false);  // Eigen asserts (aborts) if anything inside run() mallocs
  for (int i = 0; i < 1000; ++i) mlp.run(obs.data(), act.data());
  Eigen::internal::set_is_malloc_allowed(true);
  EXPECT_EQ(g_news.load(), before) << "run() called operator new";
}

TEST(PolicyMlp, TimePerInference) {
  REQUIRE_WEIGHTS();
  r1rt::PolicyMlp mlp;
  char err[256] = {0};
  ASSERT_TRUE(mlp.load(kWeights, err, sizeof err)) << err;
  auto obs = some_obs(4);
  std::vector<double> act(r1rt::PolicyMlp::kAct);
  const int n = 2000;
  double sink = 0.0;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < n; ++i) {
    obs[0] = 0.001 * i;
    mlp.run(obs.data(), act.data());
    sink += act[0];
  }
  const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / n;
  std::printf("[ policy ] %.2f us per inference (this build's flags; sink %.3f)\n", us, sink);
  RecordProperty("us_per_inference", std::to_string(us));
  EXPECT_LT(us, 5000.0);  // sanity only: a 50 Hz policy has 20 ms; the real budget is measured on the Orin (HW1)
}
