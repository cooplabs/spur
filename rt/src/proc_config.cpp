// The r1rt / r1guard configuration file parser (proc_config.hpp). Set-up code: allocation and strings are fine here
// (it runs before any real-time thread starts).
#include "r1rt/proc_config.hpp"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <vector>

namespace r1rt::proc {
namespace {

enum class Kind { kU32, kI64, kF64, kU8, kF64Vec, kI32Vec, kStr };

struct Entry {
  const char* key;
  Kind kind;
  void* ptr;
  std::size_t cap;          // array capacity (vectors)
  const std::uint32_t* n;   // vectors whose length is another key's value (policy arrays); nullptr = exactly cap
  std::size_t got = 0;      // values read (vectors)
  bool seen = false;
  bool optional = false;    // session 8: may be absent (the field keeps 0 = the core's default)
};

std::vector<Entry> table(ProcConfig& c) {
  r1rt_config& k = c.core;
  r1rt_policy_params& p = k.policy;
  constexpr std::size_t N = R1RT_NUM_JOINTS;
  return {
      {"writer_hz", Kind::kU32, &k.writer_rate_hz, 1, nullptr},
      {"policy_hz", Kind::kU32, &k.policy_rate_hz, 1, nullptr},
      {"weights", Kind::kStr, &c.weights, 1, nullptr},
      {"q_min", Kind::kF64Vec, k.q_min, N, nullptr},
      {"q_max", Kind::kF64Vec, k.q_max, N, nullptr},
      {"dq_max", Kind::kF64Vec, k.dq_max, N, nullptr},
      {"tau_max", Kind::kF64Vec, k.tau_max, N, nullptr},
      {"kp_max", Kind::kF64Vec, k.kp_max, N, nullptr},
      {"kd_max", Kind::kF64Vec, k.kd_max, N, nullptr},
      {"stop_time_s", Kind::kF64, &k.stop_time_s, 1, nullptr},
      {"arm_tilt_stop_rad", Kind::kF64, &k.arm_tilt_stop_rad, 1, nullptr},
      {"fall_tilt_rad", Kind::kF64, &k.fall_tilt_rad, 1, nullptr},
      {"lowstate_late_ns", Kind::kI64, &k.lowstate_late_ns, 1, nullptr},
      {"lowstate_stale_ns", Kind::kI64, &k.lowstate_stale_ns, 1, nullptr},
      {"lowstate_lost_ns", Kind::kI64, &k.lowstate_lost_ns, 1, nullptr},
      {"policy_max_age_ns", Kind::kI64, &k.policy_max_age_ns, 1, nullptr},
      {"policy.n_joints", Kind::kU32, &p.n_joints, 1, nullptr},
      {"policy.n_act", Kind::kU32, &p.n_act, 1, nullptr},
      {"policy.n_terms", Kind::kU32, &p.n_terms, 1, nullptr},
      {"policy.has_lean", Kind::kU32, &p.has_lean, 1, nullptr},
      {"policy.has_waist", Kind::kU32, &p.has_waist, 1, nullptr},
      {"policy.obs_terms", Kind::kI32Vec, p.obs_terms, R1RT_POLICY_MAX_TERMS, &p.n_terms},
      {"policy.joint_index", Kind::kI32Vec, p.joint_index, R1RT_POLICY_MAX_JOINTS, &p.n_joints},
      {"policy.act_pos", Kind::kI32Vec, p.act_pos, R1RT_NUM_POLICY_ACTIONS, &p.n_act},
      {"policy.default_pos", Kind::kF64Vec, p.default_pos, R1RT_POLICY_MAX_JOINTS, &p.n_joints},
      {"policy.kp", Kind::kF64Vec, p.kp, R1RT_POLICY_MAX_JOINTS, &p.n_joints},
      {"policy.kd", Kind::kF64Vec, p.kd, R1RT_POLICY_MAX_JOINTS, &p.n_joints},
      {"policy.action_scale", Kind::kF64Vec, p.action_scale, R1RT_NUM_POLICY_ACTIONS, &p.n_act},
      {"policy.kp_other", Kind::kF64Vec, p.kp_other, N, nullptr},
      {"policy.kd_other", Kind::kF64Vec, p.kd_other, N, nullptr},
      {"policy.period_s", Kind::kF64, &p.period_s, 1, nullptr},
      {"policy.step_dt_s", Kind::kF64, &p.step_dt_s, 1, nullptr},
      {"policy.cmd_lo", Kind::kF64Vec, p.cmd_lo, 3, nullptr},
      {"policy.cmd_hi", Kind::kF64Vec, p.cmd_hi, 3, nullptr},
      {"policy.lean_lo", Kind::kF64, &p.lean_lo, 1, nullptr},
      {"policy.lean_hi", Kind::kF64, &p.lean_hi, 1, nullptr},
      {"policy.lean_rate", Kind::kF64, &p.lean_rate, 1, nullptr},
      {"policy.waist_lo", Kind::kF64Vec, p.waist_lo, 2, nullptr},
      {"policy.waist_hi", Kind::kF64Vec, p.waist_hi, 2, nullptr},
      {"policy.waist_rate", Kind::kF64Vec, p.waist_rate, 2, nullptr},
      {"stand_q", Kind::kF64Vec, k.stand_q, N, nullptr},
      {"stand_kp", Kind::kF64Vec, k.stand_kp, N, nullptr},
      {"stand_kd", Kind::kF64Vec, k.stand_kd, N, nullptr},
      {"stand_ramp_s", Kind::kF64, &k.stand_ramp_s, 1, nullptr},
      {"upper_kp", Kind::kF64Vec, k.upper_kp, N, nullptr},
      {"upper_kd", Kind::kF64Vec, k.upper_kd, N, nullptr},
      {"damp_kd", Kind::kF64Vec, k.damp_kd, N, nullptr},
      {"action_abs_max", Kind::kF64, &k.action_abs_max, 1, nullptr},
      {"loop_gap_ns", Kind::kI64, &k.loop_gap_ns, 1, nullptr},
      // session 8 (HW1 fix #29): optional - absent / 0 = R1RT_DEFAULT_RESUME_GAP_NS (100 ms); older files stay valid
      {"resume_gap_ns", Kind::kI64, &k.resume_gap_ns, 1, nullptr, 0, false, true},
      {"takeover_blend_s", Kind::kF64, &k.takeover_blend_s, 1, nullptr},
      {"takeover_tilt_max_rad", Kind::kF64, &k.takeover_tilt_max_rad, 1, nullptr},
      {"takeover_dq_max", Kind::kF64, &k.takeover_dq_max, 1, nullptr},
      {"takeover_temp_max_c", Kind::kF64, &k.takeover_temp_max_c, 1, nullptr},
      {"takeover_state_max_age_ns", Kind::kI64, &k.takeover_state_max_age_ns, 1, nullptr},
      {"builtin_max_age_ns", Kind::kI64, &k.builtin_max_age_ns, 1, nullptr},
      {"builtin_min_leg_kp", Kind::kF64, &k.builtin_min_leg_kp, 1, nullptr},
      {"mirror_max_dev_rad", Kind::kF64, &k.mirror_max_dev_rad, 1, nullptr},
      {"takeover_mirror_alarm_ns", Kind::kI64, &k.takeover_mirror_alarm_ns, 1, nullptr},
      {"takeover_require_guard", Kind::kU32, &k.takeover_require_guard, 1, nullptr},
      {"guard_max_age_ns", Kind::kI64, &k.guard_max_age_ns, 1, nullptr},
      // ABI 5 (session 5): 1 = TAKEOVER / ON_STAND refused while the process's real-time grants are missing (robot
      // profiles); 0 = the simulator. rt_granted itself is not a file key: the process sets it (r1rt_core_set_rt_granted)
      {"takeover_require_rt", Kind::kU32, &k.takeover_require_rt, 1, nullptr},
      // ABI 5 (session 5): ON_STAND's hold gains for legs + waist (all 0 = the stand gains)
      {"on_stand_kp", Kind::kF64Vec, k.on_stand_kp, N, nullptr},
      {"on_stand_kd", Kind::kF64Vec, k.on_stand_kd, N, nullptr},
      // session 9 (U19 / D25): the state-age limit + the stale rung; optional (absent / 0 = the defaults, the rung OFF)
      {"state_age_limit_ns", Kind::kI64, &k.state_age_limit_ns, 1, nullptr, 0, false, true},
      {"state_age_sustain_ns", Kind::kI64, &k.state_age_sustain_ns, 1, nullptr, 0, false, true},
      {"stale_return_live_ns", Kind::kI64, &k.stale_return_live_ns, 1, nullptr, 0, false, true},
      {"board_tick_hz", Kind::kF64, &k.board_tick_hz, 1, nullptr, 0, false, true},
      {"stale_rung", Kind::kU32, &k.stale_rung, 1, nullptr, 0, false, true},
      {"guard.cycle_rate_hz", Kind::kU32, &c.guard.cycle_rate_hz, 1, nullptr},
      {"guard.stale_periods", Kind::kU32, &c.guard.stale_periods, 1, nullptr},
      {"guard.takeover_blend_s", Kind::kF64, &c.guard.takeover_blend_s, 1, nullptr},
      {"guard.takeover_enabled", Kind::kU8, &c.guard.takeover_enabled, 1, nullptr},
  };
}

std::string trim(const std::string& s) {
  const char* ws = " \t\r\n";
  const auto a = s.find_first_not_of(ws);
  if (a == std::string::npos) return {};
  const auto b = s.find_last_not_of(ws);
  return s.substr(a, b - a + 1);
}

bool parse_f64(const std::string& tok, double& v) {
  errno = 0;
  char* end = nullptr;
  v = std::strtod(tok.c_str(), &end);
  return errno == 0 && end != tok.c_str() && *end == '\0' && std::isfinite(v);
}

bool parse_i64(const std::string& tok, std::int64_t& v) {
  errno = 0;
  char* end = nullptr;
  const long long x = std::strtoll(tok.c_str(), &end, 10);
  v = static_cast<std::int64_t>(x);
  return errno == 0 && end != tok.c_str() && *end == '\0';
}

bool read_value(Entry& e, const std::string& val, std::string& err) {
  std::istringstream ss(val);
  std::vector<std::string> toks;
  for (std::string t; ss >> t;) toks.push_back(t);
  auto bad = [&](const char* why) {
    err = std::string(e.key) + ": " + why + " (\"" + val + "\")";
    return false;
  };
  switch (e.kind) {
    case Kind::kStr:
      if (val.empty()) return bad("empty");
      *static_cast<std::string*>(e.ptr) = (val == "none") ? std::string() : val;
      return true;
    case Kind::kU32:
    case Kind::kU8:
    case Kind::kI64: {
      std::int64_t v = 0;
      if (toks.size() != 1 || !parse_i64(toks[0], v)) return bad("not one integer");
      if (e.kind == Kind::kI64) {
        *static_cast<std::int64_t*>(e.ptr) = v;
      } else if (e.kind == Kind::kU32) {
        if (v < 0 || v > std::numeric_limits<std::uint32_t>::max()) return bad("out of range");
        *static_cast<std::uint32_t*>(e.ptr) = static_cast<std::uint32_t>(v);
      } else {
        if (v < 0 || v > 255) return bad("out of range");
        *static_cast<std::uint8_t*>(e.ptr) = static_cast<std::uint8_t>(v);
      }
      return true;
    }
    case Kind::kF64: {
      double v = 0.0;
      if (toks.size() != 1 || !parse_f64(toks[0], v)) return bad("not one finite number");
      *static_cast<double*>(e.ptr) = v;
      return true;
    }
    case Kind::kF64Vec:
    case Kind::kI32Vec: {
      if (toks.empty() || toks.size() > e.cap) return bad("wrong number of values");
      for (std::size_t i = 0; i < toks.size(); ++i) {
        if (e.kind == Kind::kF64Vec) {
          double v = 0.0;
          if (!parse_f64(toks[i], v)) return bad("not a finite number");
          static_cast<double*>(e.ptr)[i] = v;
        } else {
          std::int64_t v = 0;
          if (!parse_i64(toks[i], v) || v < std::numeric_limits<std::int32_t>::min() ||
              v > std::numeric_limits<std::int32_t>::max())
            return bad("not an int32");
          static_cast<std::int32_t*>(e.ptr)[i] = static_cast<std::int32_t>(v);
        }
      }
      e.got = toks.size();
      return true;
    }
  }
  return bad("unknown kind");
}

}  // namespace

bool parse_config(const std::string& text, ProcConfig& out, std::string& err) {
  ProcConfig c;
  c.core.abi_version = R1RT_ABI_VERSION;
  c.core.struct_size = sizeof(r1rt_config);
  std::vector<Entry> tab = table(c);
  std::istringstream in(text);
  std::string line;
  int lineno = 0;
  bool have_format = false;
  while (std::getline(in, line)) {
    ++lineno;
    const std::string t = trim(line);
    if (t.empty() || t[0] == '#') continue;
    const auto eq = t.find('=');
    if (eq == std::string::npos) {
      err = "line " + std::to_string(lineno) + ": no '='";
      return false;
    }
    const std::string key = trim(t.substr(0, eq));
    const std::string val = trim(t.substr(eq + 1));
    if (!have_format) {
      if (key != "format" || val != kConfigFormat) {
        err = std::string("the first key must be 'format = ") + kConfigFormat + "' (got '" + t + "')";
        return false;
      }
      have_format = true;
      continue;
    }
    Entry* e = nullptr;
    for (auto& x : tab)
      if (key == x.key) e = &x;
    if (e == nullptr) {
      err = "line " + std::to_string(lineno) + ": unknown key '" + key + "'";
      return false;
    }
    if (e->seen) {
      err = "line " + std::to_string(lineno) + ": duplicate key '" + key + "'";
      return false;
    }
    e->seen = true;
    if (!read_value(*e, val, err)) {
      err = "line " + std::to_string(lineno) + ": " + err;
      return false;
    }
  }
  if (!have_format) {
    err = "empty config (no format line)";
    return false;
  }
  for (const auto& e : tab) {
    if (!e.seen && e.optional) continue;
    if (!e.seen) {
      err = std::string("missing key '") + e.key + "'";
      return false;
    }
    if (e.kind == Kind::kF64Vec || e.kind == Kind::kI32Vec) {
      const std::size_t want = e.n != nullptr ? *e.n : e.cap;
      if (e.got != want) {
        err = std::string(e.key) + ": " + std::to_string(e.got) + " values, expected " + std::to_string(want);
        return false;
      }
    }
  }
  const auto& p = c.core.policy;
  if (p.n_joints > R1RT_POLICY_MAX_JOINTS || p.n_act > R1RT_NUM_POLICY_ACTIONS || p.n_terms > R1RT_POLICY_MAX_TERMS) {
    err = "policy sizes out of range";
    return false;
  }
  if (c.core.writer_rate_hz < 200 || c.core.writer_rate_hz > 1000 || c.core.policy_rate_hz == 0 ||
      c.guard.cycle_rate_hz == 0) {
    err = "rates out of range (writer 200-1000 Hz, policy / guard > 0)";
    return false;
  }
  out = std::move(c);
  return true;
}

bool load_config(const char* path, ProcConfig& out, std::string& err) {
  std::ifstream f(path);
  if (!f) {
    err = std::string("cannot open ") + path + ": " + std::strerror(errno);
    return false;
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  return parse_config(ss.str(), out, err);
}

std::size_t config_keys(const char* const** keys_out) {
  static std::vector<const char*> keys = [] {
    static ProcConfig dummy;
    std::vector<const char*> k{"format"};
    for (const auto& e : table(dummy)) k.push_back(e.key);
    return k;
  }();
  if (keys_out) *keys_out = keys.data();
  return keys.size();
}

}  // namespace r1rt::proc
