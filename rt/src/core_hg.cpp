// r1rt_fill_hg_lowcmd (step_api.h): the core's 26-joint command -> the hg LowCmd_ image the DDS layer writes
// (35 motor slots through R1RT_HG_SLOT_MAP_INIT, unused slots zero, mode_pr / mode_machine, Unitree's CRC32 last).
// In its own library (r1core_hg) because the CRC lives in r1io, which links the step ABI.
#include <cstring>

#include "r1rt/crc32.hpp"
#include "r1rt/hg_msgs.hpp"
#include "r1rt/step_api.h"

extern "C" uint32_t r1rt_fill_hg_lowcmd(const r1rt_lowcmd_out* cmd, void* out, size_t out_size) {
  if (cmd == nullptr || out == nullptr || out_size < sizeof(r1rt::hg::LowCmd)) return 0;
  constexpr std::uint32_t kSlot[R1RT_NUM_JOINTS] = R1RT_HG_SLOT_MAP_INIT;
  r1rt::hg::LowCmd m;
  std::memset(&m, 0, sizeof m);  // padding and unused slots 0: the CRC image is padding-independent anyway
  m.mode_pr = cmd->mode_pr;
  m.mode_machine = cmd->mode_machine;
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    r1rt::hg::MotorCmd& mc = m.motor_cmd[kSlot[j]];
    mc.mode = cmd->publish ? cmd->enable[j] : 0;
    mc.q = static_cast<float>(cmd->q[j]);
    mc.dq = static_cast<float>(cmd->dq[j]);
    mc.tau = static_cast<float>(cmd->tau_ff[j]);
    mc.kp = static_cast<float>(cmd->kp[j]);
    mc.kd = static_cast<float>(cmd->kd[j]);
  }
  m.crc = r1rt::hg_crc(m);
  std::memcpy(out, &m, sizeof m);
  return m.crc;
}
