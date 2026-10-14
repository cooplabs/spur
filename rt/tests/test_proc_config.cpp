// The r1rt / r1guard config file (proc_config.hpp): a complete file parses into the right fields; anything not
// exactly right (missing / unknown / duplicate keys, wrong lengths, garbage numbers, no format line) is refused.
#include <gtest/gtest.h>

#include <map>
#include <sstream>
#include <string>

#include "r1rt/proc_config.hpp"

namespace {

using r1rt::proc::ProcConfig;

// A complete config with recognisable values (policy: 24 joints, 14 actions, 9 terms).
std::map<std::string, std::string> full() {
  std::map<std::string, std::string> m;
  auto vec = [](int n, double base) {
    std::ostringstream s;
    for (int i = 0; i < n; ++i) s << (i ? " " : "") << base + i;
    return s.str();
  };
  const char* const* keys = nullptr;
  const std::size_t n = r1rt::proc::config_keys(&keys);
  for (std::size_t i = 0; i < n; ++i) m[keys[i]] = "1";
  m["format"] = r1rt::proc::kConfigFormat;
  m["writer_hz"] = "500";
  m["policy_hz"] = "50";
  m["weights"] = "/tmp/w.r1w";
  for (const char* k : {"q_min", "q_max", "dq_max", "tau_max", "kp_max", "kd_max", "stand_q", "stand_kp", "stand_kd",
                        "upper_kp", "upper_kd", "damp_kd", "policy.kp_other", "policy.kd_other", "on_stand_kp",
                        "on_stand_kd"})
    m[k] = vec(26, 0.5);
  m["q_min"] = vec(26, -30.0);
  m["policy.n_joints"] = "24";
  m["policy.n_act"] = "14";
  m["policy.n_terms"] = "9";
  m["policy.obs_terms"] = "1 2 3 4 5 6 7 8 9";
  m["policy.joint_index"] = vec(24, 0);
  m["policy.act_pos"] = vec(14, 0);
  m["policy.default_pos"] = vec(24, 0.1);
  m["policy.kp"] = vec(24, 10);
  m["policy.kd"] = vec(24, 1);
  m["policy.action_scale"] = vec(14, 0.25);
  m["policy.cmd_lo"] = "-1 -1 -1";
  m["policy.cmd_hi"] = "1 1 1";
  m["policy.waist_lo"] = "-0.1 -0.2";
  m["policy.waist_hi"] = "0.1 0.2";
  m["policy.waist_rate"] = "0.5 0.5";
  m["guard.cycle_rate_hz"] = "1000";
  m["guard.stale_periods"] = "5";
  m["guard.takeover_blend_s"] = "0.3";
  m["guard.takeover_enabled"] = "1";
  m["lowstate_late_ns"] = "10000000";
  m["stand_ramp_s"] = "0.4";
  return m;
}

std::string text(const std::map<std::string, std::string>& m) {
  std::ostringstream s;
  s << "# test config\nformat = " << m.at("format") << "\n";
  for (const auto& [k, v] : m)
    if (k != "format") s << k << " = " << v << "\n";
  return s.str();
}

TEST(ProcConfig, ACompleteFileParsesIntoTheRightFields) {
  ProcConfig c;
  std::string err;
  ASSERT_TRUE(r1rt::proc::parse_config(text(full()), c, err)) << err;
  EXPECT_EQ(c.core.abi_version, R1RT_ABI_VERSION);
  EXPECT_EQ(c.core.struct_size, sizeof(r1rt_config));
  EXPECT_EQ(c.core.writer_rate_hz, 500u);
  EXPECT_EQ(c.core.policy_rate_hz, 50u);
  EXPECT_DOUBLE_EQ(c.core.q_min[0], -30.0);
  EXPECT_DOUBLE_EQ(c.core.q_min[25], -5.0);
  EXPECT_DOUBLE_EQ(c.core.damp_kd[25], 25.5);
  EXPECT_EQ(c.core.lowstate_late_ns, 10'000'000);
  EXPECT_DOUBLE_EQ(c.core.stand_ramp_s, 0.4);
  EXPECT_EQ(c.core.policy.n_joints, 24u);
  EXPECT_EQ(c.core.policy.act_pos[13], 13);
  EXPECT_EQ(c.core.policy.obs_terms[8], 9);
  EXPECT_DOUBLE_EQ(c.core.policy.action_scale[13], 13.25);
  EXPECT_DOUBLE_EQ(c.core.policy.waist_hi[1], 0.2);
  EXPECT_EQ(c.guard.stale_periods, 5u);
  EXPECT_EQ(c.guard.cycle_rate_hz, 1000u);
  EXPECT_EQ(c.guard.takeover_enabled, 1);
  EXPECT_EQ(c.weights, "/tmp/w.r1w");
  EXPECT_EQ(c.core.segment, nullptr);  // the process maps it
}

TEST(ProcConfig, NumbersRoundTripExactly) {
  auto m = full();
  m["stand_ramp_s"] = "0.40000000000000002";  // Python repr of 0.4 is "0.4"; any 17-digit form parses to the same double
  m["fall_tilt_rad"] = "0.52359877559829882";
  ProcConfig c;
  std::string err;
  ASSERT_TRUE(r1rt::proc::parse_config(text(m), c, err)) << err;
  EXPECT_EQ(c.core.stand_ramp_s, 0.4);
  EXPECT_EQ(c.core.fall_tilt_rad, 0.52359877559829882);
}

TEST(ProcConfig, WeightsNoneMeansNoPolicy) {
  auto m = full();
  m["weights"] = "none";
  ProcConfig c;
  std::string err;
  ASSERT_TRUE(r1rt::proc::parse_config(text(m), c, err)) << err;
  EXPECT_TRUE(c.weights.empty());
}

TEST(ProcConfig, RefusesWhatIsNotExactlyRight) {
  struct Case {
    const char* what;
    std::string txt;
    const char* in_err;
  };
  std::vector<Case> cases;
  {
    auto m = full();
    m.erase("guard_max_age_ns");
    cases.push_back({"missing key", text(m), "missing key 'guard_max_age_ns'"});
  }
  {
    auto m = full();
    m["takeover_blend"] = "1.0";
    cases.push_back({"unknown key", text(m), "unknown key 'takeover_blend'"});
  }
  {
    auto t = text(full()) + "writer_hz = 500\n";
    cases.push_back({"duplicate", t, "duplicate key 'writer_hz'"});
  }
  {
    auto m = full();
    m["q_max"] = "1 2 3";
    cases.push_back({"short array", text(m), "q_max: 3 values, expected 26"});
  }
  {
    auto m = full();
    m["policy.act_pos"] = "0 1 2";
    cases.push_back({"policy array vs n_act", text(m), "policy.act_pos: 3 values, expected 14"});
  }
  {
    auto m = full();
    m["stand_ramp_s"] = "0.4s";
    cases.push_back({"garbage number", text(m), "stand_ramp_s"});
  }
  {
    auto m = full();
    m["action_abs_max"] = "nan";
    cases.push_back({"non-finite", text(m), "action_abs_max"});
  }
  {
    auto m = full();
    m["writer_hz"] = "5000";
    cases.push_back({"rate out of range", text(m), "rates out of range"});
  }
  {
    std::string t = text(full());
    t.replace(t.find("format = "), 9, "formats = ");
    cases.push_back({"no format line", t, "the first key must be"});
  }
  {
    auto m = full();
    m["format"] = "r1rt-config/0";
    cases.push_back({"old format", text(m), "the first key must be"});
  }
  cases.push_back({"empty", "# nothing\n", "empty config"});
  for (const auto& c : cases) {
    ProcConfig out;
    std::string err;
    EXPECT_FALSE(r1rt::proc::parse_config(c.txt, out, err)) << c.what;
    EXPECT_NE(err.find(c.in_err), std::string::npos) << c.what << ": " << err;
  }
}

}  // namespace
