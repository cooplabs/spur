/* Compiled as C11: the public C headers must stay valid C (they are the cross-language boundary). */
#include "r1rt/step_api.h"
#include "r1rt/types.h"

int r1rt_c_abi_check(void);

int r1rt_c_abi_check(void) {
  r1rt_lowcmd_out cmd;
  r1rt_lowstate_in st;
  int32_t status = 0;
  r1rt_core* core = r1rt_core_create(0, &status); /* NULL config: refused */
  if (core != 0 || status != R1RT_E_INVALID_ARG) return 1;
  st.board_tick = 1;
  if (r1rt_core_step(0, &st, 0, &cmd, 0) != R1RT_E_INVALID_ARG) return 2;
  if (cmd.publish != 0) return 3;
  if (r1rt_abi_version() != R1RT_ABI_VERSION) return 4;
  if (sizeof(r1rt_lowcmd_out) != 1080 || sizeof(r1rt_lowstate_in) != 1304) return 5;
  return 0;
}
