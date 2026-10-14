// r1rt/proc_config.hpp - the r1rt / r1guard processes' configuration file (session 5).
//
// One source of numbers: `python -m r1mc3.rtconfig --out FILE` writes the SAME dict that r1sim.v3_core.core_config
// builds for the lockstep simulator (the r1rt_py.Core constructor's keys), flattened, so the processes and the
// lockstep tests run identical values. The robot-profile path (config/robot_profile.r1_dex3.yaml -> this file) is a
// later step (session 6+): today the file comes from the simulator's model (said in PROCESSES.md).
//
// Format (versioned, line based, no dependency):
//     # comment
//     format = r1rt-config/1          <- first key, required
//     writer_hz = 500
//     q_min = -2.5 -2.5 ...           <- arrays: whitespace-separated numbers
//     weights = /abs/path/policy_weights.r1w   (or "none": BALANCE / takeover refused by the core)
//     policy.n_joints = 24            <- r1rt_policy_params, "policy." prefix
//     guard.stale_periods = 5         <- r1rt::GuardParams (r1guard), "guard." prefix
// Every key is required, every unknown key is refused, every array has its exact length (the policy arrays: the
// length their n_* says), every number parses completely - a config that is not exactly right never starts a process.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "r1rt/guard.hpp"
#include "r1rt/step_api.h"

namespace r1rt::proc {

inline constexpr const char* kConfigFormat = "r1rt-config/1";

struct ProcConfig {
  r1rt_config core{};      // abi_version / struct_size set; segment / segment_size / policy_weights_path left to the
                           // caller (the process maps the segment; weights points into `weights` below)
  GuardParams guard{};
  std::string weights;     // empty = "none"
};

// Parse `text` (the file's content). Returns true and fills `out`, or false with a one-line reason in `err`.
bool parse_config(const std::string& text, ProcConfig& out, std::string& err);

// Read + parse a file.
bool load_config(const char* path, ProcConfig& out, std::string& err);

// Every key the format has, in file order (rtconfig writes exactly these; a test pins both sides).
std::size_t config_keys(const char* const** keys_out);

}  // namespace r1rt::proc
