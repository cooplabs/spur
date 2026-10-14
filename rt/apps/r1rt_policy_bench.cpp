// r1rt_policy_bench - T7 (RESEARCH.md §8, Q7): what one balance-policy step costs on this machine, called exactly as
// r1rt's policy thread calls it (DESIGN.md §3.2; docs/hardware/T7_BENCH.md). NOTHING MOVES: no DDS, no shared memory,
// no robot - the converted weights and an observation file in, one JSON object out.
//
//   r1rt_policy_bench --weights PATH --input FILE.r1pb [--mode periodic|burst] [--rate-hz 50] [--duration-s 60]
//                     [--iters 20000] [--warmup 50] [--cpu C] [--fifo PRIO] [--mlock] [--abs-max 50] [--out FILE]
//
// The input ("R1PB", tools/t7/make_input.py) is a recorded stance of the live policy in the simulator: per policy step
// the policy-order q / dq, the IMU quaternion + gyro, the command, the step's time and the lean target, plus the 14
// targets the Python reference computed from them. One call = what Core::policy_step does around the network
// (core.cpp ~205-245): restore the runner state, set the lean target, advance the phase clock, act() (observation +
// network + targets, with the profile's action_abs_max), save the state. The few shared-memory reads / writes around
// it (the committed seqlock, a CAS) are not included (tens of ns).
//
// Modes: `periodic` (default) = the policy thread's own pattern - absolute 20 ms deadlines (clock_nanosleep), one
// call per wake, so every call starts with whatever the sleep left in the caches; reports the wake lateness, the
// compute time and their sum (the age of the result at the end of the call). `burst` = back-to-back calls (warm caches:
// the best case; also what the writer's inline rung pays when it runs the policy every tick).
//
// Pass rule (the numbers in docs/hardware/T7_BENCH.md): the result valid (the C++ targets equal the file's within
// 2e-6 rad, no invalid output); the THREAD budget - max compute <= 3 ms (RESEARCH.md §7 budget table: "policy <= 3 ms of
// each 20 ms") and max (lateness + compute) <= the 20 ms period; the INLINE budget - max compute <= 1.0 ms (rung 1,
// DESIGN §4.3: the writer runs the policy inside its 2 ms tick; HW1 measured the writer's own worst wake lateness 336 us
// and worst iteration 329 us, leaving ~1.3 ms). A run counts as the T7 answer only with SCHED_FIFO, the CPU pinning and
// mlock granted (`conditions_ok`); on macOS none exist (reported, not failed).
#include <sys/mman.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "app_common.hpp"
#include "r1rt/policy_mlp.hpp"
#include "r1rt/policy_runner.hpp"
#include "r1rt/rt_thread.hpp"

namespace {

using r1rt::PolicyMlp;
using r1rt::PolicyRunner;
namespace proc = r1rt::proc;
namespace app = r1rt::app;

constexpr double kCheckTolRad = 2e-6;    // float32 storage of the expected targets (~1e-7 at 1-2 rad)
constexpr double kThreadBudgetUs = 3000.0;
constexpr double kInlineBudgetUs = 1000.0;

struct Step {
  double t;
  double lean_req;  // NaN = none
  double q[R1RT_POLICY_MAX_JOINTS], dq[R1RT_POLICY_MAX_JOINTS], quat[4], gyro[3], cmd[3];
};

struct Input {
  r1rt_policy_params params{};
  std::vector<Step> steps;
  std::vector<double> expected;  // n_steps x n_act
  std::uint32_t n_act = 0;
};

// The params block = r1sim.v3_core.PARAMS_LAYOUT (the same unpacking as rt/tests/test_core_util.hpp's R1PR reader).
bool unpack_params(const std::vector<double>& p, r1rt_policy_params& q, std::string& err) {
  std::size_t i = 0;
  auto take = [&](std::size_t k) -> const double* {
    if (i + k > p.size()) return nullptr;
    const double* v = p.data() + i;
    i += k;
    return v;
  };
  const std::size_t need = 5 + R1RT_POLICY_MAX_TERMS + R1RT_POLICY_MAX_JOINTS + R1RT_NUM_POLICY_ACTIONS +
                           3 * R1RT_POLICY_MAX_JOINTS + R1RT_NUM_POLICY_ACTIONS + 2 * R1RT_NUM_JOINTS + 2 + 6 + 3 + 6;
  if (p.size() != need) {
    err = "params layout mismatch (" + std::to_string(p.size()) + " values, want " + std::to_string(need) + ")";
    return false;
  }
  q.n_joints = static_cast<std::uint32_t>(*take(1));
  q.n_act = static_cast<std::uint32_t>(*take(1));
  q.n_terms = static_cast<std::uint32_t>(*take(1));
  q.has_lean = static_cast<std::uint32_t>(*take(1));
  q.has_waist = static_cast<std::uint32_t>(*take(1));
  const double* v = take(R1RT_POLICY_MAX_TERMS);
  for (int k = 0; k < R1RT_POLICY_MAX_TERMS; ++k) q.obs_terms[k] = static_cast<std::int32_t>(v[k]);
  v = take(R1RT_POLICY_MAX_JOINTS);
  for (int k = 0; k < R1RT_POLICY_MAX_JOINTS; ++k) q.joint_index[k] = static_cast<std::int32_t>(v[k]);
  v = take(R1RT_NUM_POLICY_ACTIONS);
  for (int k = 0; k < R1RT_NUM_POLICY_ACTIONS; ++k) q.act_pos[k] = static_cast<std::int32_t>(v[k]);
  std::memcpy(q.default_pos, take(R1RT_POLICY_MAX_JOINTS), sizeof q.default_pos);
  std::memcpy(q.kp, take(R1RT_POLICY_MAX_JOINTS), sizeof q.kp);
  std::memcpy(q.kd, take(R1RT_POLICY_MAX_JOINTS), sizeof q.kd);
  std::memcpy(q.action_scale, take(R1RT_NUM_POLICY_ACTIONS), sizeof q.action_scale);
  std::memcpy(q.kp_other, take(R1RT_NUM_JOINTS), sizeof q.kp_other);
  std::memcpy(q.kd_other, take(R1RT_NUM_JOINTS), sizeof q.kd_other);
  q.period_s = *take(1);
  q.step_dt_s = *take(1);
  std::memcpy(q.cmd_lo, take(3), sizeof q.cmd_lo);
  std::memcpy(q.cmd_hi, take(3), sizeof q.cmd_hi);
  q.lean_lo = *take(1);
  q.lean_hi = *take(1);
  q.lean_rate = *take(1);
  std::memcpy(q.waist_lo, take(2), sizeof q.waist_lo);
  std::memcpy(q.waist_hi, take(2), sizeof q.waist_hi);
  std::memcpy(q.waist_rate, take(2), sizeof q.waist_rate);
  return true;
}

bool read_input(const char* path, Input& in, std::string& err) {
  std::FILE* f = std::fopen(path, "rb");
  if (f == nullptr) {
    err = std::string("cannot open ") + path;
    return false;
  }
  std::vector<unsigned char> b;
  unsigned char buf[65536];
  std::size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) b.insert(b.end(), buf, buf + n);
  std::fclose(f);
  if (b.size() < 24 || std::memcmp(b.data(), "R1PB", 4) != 0) {
    err = "not an R1PB file";
    return false;
  }
  std::uint32_t h[5];
  std::memcpy(h, b.data() + 4, sizeof h);
  const std::uint32_t version = h[0], n_steps = h[1], n_params = h[2], nj = h[3], na = h[4];
  if (version != 1) {
    err = "R1PB version " + std::to_string(version);
    return false;
  }
  if (nj > R1RT_POLICY_MAX_JOINTS || na > R1RT_NUM_POLICY_ACTIONS || n_steps == 0) {
    err = "R1PB sizes out of range";
    return false;
  }
  const std::size_t per = 1 + 2 * nj + 4 + 3 + 3;
  const std::size_t want = 24 + 8 * static_cast<std::size_t>(n_params) + n_steps * (8 + 4 * per) +
                           4 * static_cast<std::size_t>(n_steps) * na;
  if (b.size() != want) {
    err = "R1PB size " + std::to_string(b.size()) + " != " + std::to_string(want);
    return false;
  }
  std::size_t off = 24;
  std::vector<double> p(n_params);
  std::memcpy(p.data(), b.data() + off, 8 * static_cast<std::size_t>(n_params));
  off += 8 * static_cast<std::size_t>(n_params);
  if (!unpack_params(p, in.params, err)) return false;
  if (in.params.n_joints != nj || in.params.n_act != na) {
    err = "R1PB header sizes differ from its params";
    return false;
  }
  in.steps.resize(n_steps);
  std::vector<float> fl(per);
  for (auto& s : in.steps) {
    std::memcpy(&s.t, b.data() + off, 8);
    off += 8;
    std::memcpy(fl.data(), b.data() + off, 4 * per);
    off += 4 * per;
    std::size_t k = 0;
    s.lean_req = fl[k++];
    for (std::uint32_t i = 0; i < nj; ++i) s.q[i] = fl[k++];
    for (std::uint32_t i = 0; i < nj; ++i) s.dq[i] = fl[k++];
    for (double& x : s.quat) x = fl[k++];
    for (double& x : s.gyro) x = fl[k++];
    for (double& x : s.cmd) x = fl[k++];
  }
  in.n_act = na;
  in.expected.resize(static_cast<std::size_t>(n_steps) * na);
  std::vector<float> e(in.expected.size());
  std::memcpy(e.data(), b.data() + off, 4 * e.size());
  for (std::size_t i = 0; i < e.size(); ++i) in.expected[i] = e[i];
  return true;
}

// One policy step as Core::policy_step makes it (core.cpp ~212-232): the runner state in and out of the hand-over
// slot, the lean target, the phase clock, act().
inline bool policy_call(PolicyRunner& r, PolicyRunner::State& slot, const Step& s, double abs_max) noexcept {
  r.restore(slot);
  if (!std::isnan(s.lean_req)) r.set_lean_target(s.lean_req);
  r.advance_clock(s.t);
  const bool ok = r.act(s.q, s.dq, s.quat, s.gyro, s.cmd, nullptr, abs_max);
  r.save(slot);
  return ok;
}

struct Stats {
  double p50 = 0, p90 = 0, p99 = 0, p999 = 0, max = 0, mean = 0;
  std::size_t n = 0;
};

Stats stats_us(std::vector<std::int64_t> v) {
  Stats s;
  s.n = v.size();
  if (v.empty()) return s;
  std::sort(v.begin(), v.end());
  auto pct = [&](double p) {  // nearest rank
    const auto rank = static_cast<std::size_t>(std::ceil(p * static_cast<double>(v.size())));
    return static_cast<double>(v[std::min(v.size(), std::max<std::size_t>(rank, 1)) - 1]) * 1e-3;
  };
  double sum = 0;
  for (auto x : v) sum += static_cast<double>(x);
  s.p50 = pct(0.50);
  s.p90 = pct(0.90);
  s.p99 = pct(0.99);
  s.p999 = pct(0.999);
  s.max = static_cast<double>(v.back()) * 1e-3;
  s.mean = sum / static_cast<double>(v.size()) * 1e-3;
  return s;
}

std::string stats_json(const Stats& s) {
  char b[256];
  std::snprintf(b, sizeof b, "{\"n\":%zu,\"p50\":%.3f,\"p90\":%.3f,\"p99\":%.3f,\"p999\":%.3f,\"max\":%.3f,\"mean\":%.3f}",
                s.n, s.p50, s.p90, s.p99, s.p999, s.max, s.mean);
  return b;
}

std::string read_first_line(const char* path) {
  std::FILE* f = std::fopen(path, "r");
  if (f == nullptr) return "";
  char b[256] = {0};
  const char* got = std::fgets(b, sizeof b, f);
  std::fclose(f);
  if (got == nullptr) return "";
  std::string s(b);
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
  return s;
}

const char* build_kind() {
#if defined(NDEBUG)
  return "release";
#else
  return "debug";
#endif
}

bool sanitized() {
#if defined(__SANITIZE_ADDRESS__)
  return true;
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(undefined_behavior_sanitizer)
  return true;
#else
  return false;
#endif
#else
  return false;
#endif
}

struct Args {
  std::string weights, input, out, mode = "periodic";
  double rate_hz = 50.0, duration_s = 60.0, abs_max = 50.0;
  long iters = 20000, warmup = 50;
  int cpu = -1, fifo = 0;
  bool mlock = false;
};

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) app::die("missing value for " + k);
      return argv[++i];
    };
    if (k == "--weights") {
      a.weights = val();
    } else if (k == "--input") {
      a.input = val();
    } else if (k == "--out") {
      a.out = val();
    } else if (k == "--mode") {
      a.mode = val();
    } else if (k == "--rate-hz") {
      a.rate_hz = std::stod(val());
    } else if (k == "--duration-s") {
      a.duration_s = std::stod(val());
    } else if (k == "--iters") {
      a.iters = std::stol(val());
    } else if (k == "--warmup") {
      a.warmup = std::stol(val());
    } else if (k == "--cpu") {
      a.cpu = std::stoi(val());
    } else if (k == "--fifo") {
      a.fifo = std::stoi(val());
    } else if (k == "--abs-max") {
      a.abs_max = std::stod(val());
    } else if (k == "--mlock") {
      a.mlock = true;
    } else if (k == "--help" || k == "-h") {
      std::puts(
          "r1rt_policy_bench --weights PATH --input FILE.r1pb [--mode periodic|burst] [--rate-hz 50] [--duration-s 60]\n"
          "                  [--iters 20000] [--warmup 50] [--cpu C] [--fifo PRIO] [--mlock] [--abs-max 50] [--out FILE]\n"
          "T7: one balance-policy step's cost as r1rt's policy thread pays it. Nothing moves (no DDS, no robot).");
      std::exit(0);
    } else {
      app::die("unknown argument " + k);
    }
  }
  if (a.weights.empty() || a.input.empty()) app::die("--weights and --input are required (--help)");
  if (a.mode != "periodic" && a.mode != "burst") app::die("--mode must be periodic or burst");
  if (!(a.rate_hz > 0.0) || !(a.duration_s > 0.0) || a.iters <= 0 || a.warmup < 0) app::die("bad numbers");
  return a;
}

}  // namespace

int main(int argc, char** argv) {
  const Args a = parse(argc, argv);
  app::install_signal_handlers();  // Ctrl-C ends a periodic run early and still prints the result
  Input in;
  std::string err;
  if (!read_input(a.input.c_str(), in, err)) app::die("input: " + err);
  PolicyMlp net;
  char ebuf[256] = {0};
  if (!net.load(a.weights.c_str(), ebuf, sizeof ebuf)) app::die(std::string("weights: ") + ebuf);
  PolicyRunner r;
  if (!r.init(in.params, &net, ebuf, sizeof ebuf)) app::die(std::string("policy params: ") + ebuf);
  const std::size_t n = in.steps.size();
  const int na = static_cast<int>(in.n_act);

  // 1. The check (not timed): the whole sequence once from a fresh runner, the targets vs the file's.
  PolicyRunner::State slot{};
  r.reset();
  r.save(slot);
  double max_err = 0.0;
  std::size_t invalid = 0;
  for (std::size_t k = 0; k < n; ++k) {
    if (!policy_call(r, slot, in.steps[k], a.abs_max)) ++invalid;
    for (int i = 0; i < na; ++i)
      max_err = std::fmax(max_err, std::fabs(r.target()[i] - in.expected[k * in.n_act + static_cast<std::size_t>(i)]));
  }
  const bool check_ok = invalid == 0 && max_err <= kCheckTolRad;

  // 2. The thread as r1rt's policy thread asks for it (DESIGN §2.1: FIFO 80 on CPU 6 on the Orin).
  const auto period_ns = static_cast<std::int64_t>(std::llround(1e9 / a.rate_hz));
  proc::RtGrant grant;
  grant.cpu = a.cpu;
  grant.priority = a.fifo;
  if (a.fifo > 0 || a.cpu >= 0) grant = proc::request_rt_thread(a.fifo, a.cpu, period_ns, period_ns / 4);
  int mlock_err = 0;
  const bool mlocked = a.mlock && proc::lock_memory(&mlock_err);
  if (!a.mlock) mlock_err = -1;

  const bool periodic = a.mode == "periodic";
  const std::size_t calls =
      periodic ? static_cast<std::size_t>(std::llround(a.duration_s * a.rate_hz)) : static_cast<std::size_t>(a.iters);
  std::vector<std::int64_t> compute, lateness, total;
  compute.reserve(calls);
  lateness.reserve(periodic ? calls : 0);
  total.reserve(periodic ? calls : 0);
  r.reset();
  r.save(slot);
  std::size_t k = 0;
  std::size_t invalid_timed = 0;
  std::uint64_t skipped = 0;
  auto next_step = [&]() -> const Step& {
    if (k == n) {  // the sequence wraps: a fresh runner, so the state follows the recorded stance again
      k = 0;
      r.reset();
      r.save(slot);
    }
    return in.steps[k++];
  };
  for (long w = 0; w < a.warmup; ++w) policy_call(r, slot, next_step(), a.abs_max);
  if (periodic) {
    proc::Periodic per(period_ns, proc::now_ns() + period_ns);
    for (std::size_t c = 0; c < calls; ++c) {
      std::int64_t late = 0;
      const std::int64_t deadline = per.wait(&late) - late;  // wait() returns the wake time
      const Step& s = next_step();
      const std::int64_t t0 = proc::now_ns();
      if (!policy_call(r, slot, s, a.abs_max)) ++invalid_timed;
      const std::int64_t t1 = proc::now_ns();
      compute.push_back(t1 - t0);
      lateness.push_back(late);
      total.push_back(t1 - deadline);
      if (app::g_signal.load(std::memory_order_relaxed) != 0) break;
    }
    skipped = per.skipped();
  } else {
    for (std::size_t c = 0; c < calls; ++c) {
      const Step& s = next_step();
      const std::int64_t t0 = proc::now_ns();
      if (!policy_call(r, slot, s, a.abs_max)) ++invalid_timed;
      compute.push_back(proc::now_ns() - t0);
    }
  }
  if (mlocked) munlockall();

  const Stats sc = stats_us(compute), sl = stats_us(lateness), st = stats_us(periodic ? total : compute);
#if defined(__linux__)
  const bool conditions_ok = grant.fifo && grant.affinity && mlocked;
#else
  const bool conditions_ok = false;  // no SCHED_FIFO / affinity / the robot's kernel: a reference, not T7
#endif
  const bool pass_thread = sc.max <= kThreadBudgetUs && st.max <= static_cast<double>(period_ns) * 1e-3;
  const bool pass_inline = sc.max <= kInlineBudgetUs;
  const bool pass = check_ok && invalid_timed == 0 && pass_thread && pass_inline;
  const std::string cpu_freq =
      a.cpu >= 0 ? read_first_line(("/sys/devices/system/cpu/cpu" + std::to_string(a.cpu) + "/cpufreq/scaling_cur_freq").c_str())
                 : "";
  const std::string governor =
      a.cpu >= 0 ? read_first_line(("/sys/devices/system/cpu/cpu" + std::to_string(a.cpu) + "/cpufreq/scaling_governor").c_str())
                 : "";
  char head[1024];
  std::snprintf(head, sizeof head,
                "{\"t7\":1,\"tool\":\"r1rt_policy_bench\",\"os\":\"%s\",\"build\":\"%s\",\"sanitized\":%s,\"mode\":\"%s\","
                "\"rate_hz\":%.3f,\"calls\":%zu,\"warmup\":%ld,\"input\":%s,\"input_steps\":%zu,\"n_obs\":%d,"
                "\"skipped_periods\":%llu,\"abs_max\":%.3f,\"mlock\":%s,\"mlock_errno\":%d,\"cpu_freq_khz\":%s,\"governor\":%s,",
                proc::os_name(), build_kind(), sanitized() ? "true" : "false", a.mode.c_str(), a.rate_hz, sc.n, a.warmup,
                app::jstr(a.input.c_str()).c_str(), n, r.n_obs(), static_cast<unsigned long long>(skipped), a.abs_max, mlocked ? "true" : "false", mlock_err,
                cpu_freq.empty() ? "null" : cpu_freq.c_str(), governor.empty() ? "null" : app::jstr(governor.c_str()).c_str());
  char check[256];
  std::snprintf(check, sizeof check,
                "\"check\":{\"steps\":%zu,\"max_abs_err_rad\":%.3e,\"tol_rad\":%.1e,\"invalid\":%zu,\"invalid_timed\":%zu,"
                "\"ok\":%s},",
                n, max_err, kCheckTolRad, invalid, invalid_timed, check_ok ? "true" : "false");
  char tail[512];
  std::snprintf(tail, sizeof tail,
                "\"budget_us\":{\"thread\":%.0f,\"inline\":%.0f,\"period\":%.0f,\"inline_rung_after_ms\":40},"
                "\"conditions_ok\":%s,\"pass_thread\":%s,\"pass_inline\":%s,\"pass\":%s}",
                kThreadBudgetUs, kInlineBudgetUs, static_cast<double>(period_ns) * 1e-3, conditions_ok ? "true" : "false",
                pass_thread ? "true" : "false", pass_inline ? "true" : "false", pass ? "true" : "false");
  const std::string json = std::string(head) + "\"grant\":" + app::grant_json(grant) + "," + check +
                           "\"compute_us\":" + stats_json(sc) + "," +
                           "\"lateness_us\":" + (periodic ? stats_json(sl) : std::string("null")) + "," +
                           "\"total_us\":" + stats_json(st) + "," + tail;
  app::emit(json);
  if (!a.out.empty()) {
    std::FILE* f = std::fopen(a.out.c_str(), "w");
    if (f == nullptr) app::die("cannot write " + a.out);
    std::fputs(json.c_str(), f);
    std::fputc('\n', f);
    std::fclose(f);
  }
  return check_ok ? 0 : 1;
}
