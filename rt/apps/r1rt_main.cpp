// r1rt - the real-time core as a PROCESS (session 5; DESIGN.md §2.1, §3.1, §4.2-§4.4; docs/architecture/PROCESSES.md).
//
//   r1rt --config FILE --shm-name NAME --domain N [--rate-hz 500] [--policy-weights PATH]
//        [--fault-writer-stall AT_S:DUR_S] [--fault-policy-nan AT_S:DUR_S] [--replace-stale]
//
// Creates the shared segment (shm_open), the core (r1rt::Core behind the step ABI) from the config file, and four
// threads: the WRITER (500 Hz, absolute deadlines: state -> r1rt_core_step -> the token re-checked -> rt/lowcmd), the
// POLICY thread (50 Hz, r1rt_core_policy_step), the DDS RECEIVE thread (rt/lowstate -> a seqlock; the built-in's
// rt/lowcmd from foreign writers -> r1rt_core_set_builtin_cmd), and main (signals, status lines). It starts in
// MONITOR and publishes nothing until a takeover / ON_STAND request arrives through the segment's mode mailbox
// (read inside r1rt_core_step: nothing here). NEVER damps: SIGTERM / SIGINT in a publishing mode are refused and
// logged (the operator stops through `damp` first, DESIGN §2.1); in MONITOR / DAMP they exit cleanly.
//
// Test hook (session 9, #29's open race): env R1RT_TEST_STALL_BEFORE_WRITE=<path> - when the file appears (main polls it
// every 20 ms and removes it), the writer stops the whole process (raise(SIGSTOP)) ONCE, at the next publish, between
// its stale / token checks and the DDS write: the window a real whole-machine stall can hit (inside the write path).
// After SIGCONT that packet - computed before the stop - goes out: the deterministic version of the race (event
// "test_stall_before_write"). Unset = nothing in the writer but one relaxed atomic load per publish.
//
// stdout: one JSON object per line - {"ready": true, ...} once, then {"status": ...} every second and events
// ({"event": "signal_refused"|"fault_writer_stall"|"exit", ...}). stderr: human-readable errors.
#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#include <unistd.h>

#include "app_common.hpp"
#include "r1rt/dds_io.hpp"
#include "r1rt/proc_config.hpp"
#include "r1rt/rt_thread.hpp"
#include "r1rt/seqlock.hpp"
#include "r1rt/shm.hpp"
#include "r1rt/shm_layout.h"
#include "r1rt/step_api.h"

namespace {

using namespace r1rt;

struct Args {
  std::string config, shm = "/r1mc3.rt", weights;
  int domain = 0;
  unsigned rate_hz = 0;
  double stall_at_s = -1.0, stall_dur_s = 0.0;
  double nan_at_s = -1.0, nan_dur_s = 0.0;  // session 6: --fault-policy-nan (the fault table's "policy NaN" row)
  bool replace_stale = false;
  bool check_config = false;
  bool stand_session = false;  // session 6 (DESIGN §3.5 start flag): no BALANCE for the process's lifetime
  double test_takeover_s = -1.0, test_released_s = -1.0;  // test hook: requests from the writer thread itself
};

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) app::die("missing value for " + k);
      return argv[++i];
    };
    if (k == "--config") {
      a.config = val();
    } else if (k == "--shm-name") {
      a.shm = val();
    } else if (k == "--domain") {
      a.domain = std::atoi(val().c_str());
    } else if (k == "--rate-hz") {
      a.rate_hz = static_cast<unsigned>(std::atoi(val().c_str()));
    } else if (k == "--policy-weights") {
      a.weights = val();
    } else if (k == "--fault-writer-stall") {
      const std::string v = val();
      if (std::sscanf(v.c_str(), "%lf:%lf", &a.stall_at_s, &a.stall_dur_s) != 2 || a.stall_at_s < 0 ||
          a.stall_dur_s <= 0)
        app::die("--fault-writer-stall wants AT_S:DUR_S (seconds after BALANCE entry : stall length)");
    } else if (k == "--fault-policy-nan") {
      const std::string v = val();
      if (std::sscanf(v.c_str(), "%lf:%lf", &a.nan_at_s, &a.nan_dur_s) != 2 || a.nan_at_s < 0 || a.nan_dur_s <= 0)
        app::die("--fault-policy-nan wants AT_S:DUR_S (seconds after BALANCE entry : NaN window length)");
    } else if (k == "--stand-session") {
      a.stand_session = true;
    } else if (k == "--replace-stale") {
      a.replace_stale = true;
    } else if (k == "--check-config") {
      a.check_config = true;
    } else if (k == "--test-takeover") {
      const std::string v = val();
      if (std::sscanf(v.c_str(), "%lf:%lf", &a.test_takeover_s, &a.test_released_s) != 2 || a.test_takeover_s < 0 ||
          a.test_released_s <= a.test_takeover_s)
        app::die("--test-takeover wants T_TAKEOVER_S:T_RELEASED_S (seconds after start, released later)");
    } else if (k == "-h" || k == "--help") {
      std::printf(
          "r1rt --config FILE --shm-name NAME --domain N [--rate-hz 500] [--policy-weights PATH]\n"
          "     [--fault-writer-stall AT_S:DUR_S] [--fault-policy-nan AT_S:DUR_S] [--replace-stale] [--check-config]\n"
          "     [--stand-session]\n"
          "     [--test-takeover T_TAKEOVER_S:T_RELEASED_S]\n");
      std::exit(0);
    } else {
      app::die("unknown argument " + k);
    }
  }
  if (a.config.empty()) app::die("--config FILE is required");
  return a;
}

// What the writer thread publishes for the status line (a seqlock: the writer never waits on main).
struct WriterStats {
  proc::TimingStats lateness;
  proc::TimingStats iteration;
  std::uint64_t ticks = 0;
  std::uint64_t publishes = 0;
  std::uint64_t not_owner_skips = 0;  // the core said publish but the token was gone at the write (the residual race)
  std::uint64_t stale_skips = 0;      // session 8: a command older than kStaleWriteNs at the write (a stop in between)
  std::uint64_t skipped_deadlines = 0;
  // session 8 (HW1 fix #31): a wake-up gap longer than resume_gap_ns is a STOP (a stand test's SIGSTOP, a process
  // stall), not an overrun: the deadlines skipped inside it are counted apart, so a real overrun stays visible
  std::uint64_t stops = 0;
  std::uint64_t stop_skipped = 0;     // skipped deadlines inside stops (overruns = skipped_deadlines - stop_skipped)
  std::int64_t stop_ns_max = 0;
  std::uint64_t resume_blends = 0;    // the core's resume blends from the measured pose (fix #29)
  std::uint64_t resume_waits_expired = 0;
  std::uint64_t resume_uncaught = 0;     // session 9: blends from a fresh-but-uncaught sample after the drain limit
  std::uint64_t resume_drain_waits = 0;  // session 9 (#29 hazard ii): waits extended over a backlog still draining
  std::int64_t t_balance_ns = -1;     // first writer tick in BALANCE
  std::uint16_t mode = R1RT_MODE_MONITOR;
};

// A command older than this at its write is dropped: half of r1guard's takeover threshold (5 writer periods, 10 ms) - any
// stop long enough for a takeover is caught, a normal iteration never (robot max 0.34 ms; a loaded Mac's Debug build a
// few ms at worst).
constexpr std::int64_t kStaleWriteNs = 5'000'000;

struct Shared {
  r1rt_core* core = nullptr;
  shm::Segment* seg = nullptr;
  dds_io::Bus* bus = nullptr;
  std::int64_t writer_period_ns = 2'000'000;
  std::int64_t resume_gap_ns = R1RT_DEFAULT_RESUME_GAP_NS;  // session 8: the core's (config) value, for the stop count
  std::int64_t policy_period_ns = 20'000'000;
  double stall_at_s = -1.0, stall_dur_s = 0.0;
  double nan_at_s = -1.0, nan_dur_s = 0.0;
  std::atomic<std::int64_t> nan_t0_ns{0};  // the NaN window as injected (main reports it)
  double test_takeover_s = -1.0, test_released_s = -1.0;
  std::atomic<std::int32_t> test_takeover_status{1};   // 1 = not yet requested
  std::atomic<std::int32_t> test_released_status{1};
  std::atomic<std::int32_t> test_refusal{0};
  std::atomic<bool> stop{false};
  std::atomic<int> grants_ready{0};
  proc::RtGrant writer_grant, policy_grant, rx_grant;
  Seqlock<WriterStats> wstats;
  std::atomic<std::int64_t> stall_started_ns{0};
  std::atomic<std::int64_t> stall_ended_ns{0};
  std::atomic<std::int32_t> rt_granted{-1};  // -1 = not yet known; main sets it once the threads report their grants
  std::atomic<bool> test_stall_armed{false};      // session 9 test hook (R1RT_TEST_STALL_BEFORE_WRITE): armed by main
  std::atomic<std::int64_t> test_stall_wake_ns{0};  // the stopped tick's wake-up (set before the SIGSTOP)
  std::atomic<std::int64_t> test_stall_cont_ns{0};  // when the write resumed after SIGCONT
};

std::int32_t clamp_i32(std::int64_t v) {
  return static_cast<std::int32_t>(std::max<std::int64_t>(INT32_MIN, std::min<std::int64_t>(INT32_MAX, v)));
}

void builtin_cb(void* ctx, const r1rt_lowcmd_out& cmd, std::int64_t t_rx_ns) {
  r1rt_core_set_builtin_cmd(static_cast<r1rt_core*>(ctx), &cmd, t_rx_ns);
}

void writer_thread(Shared* s) {
  // DESIGN §2.1: the writer FIFO 90, alone on CPU 7.
  s->writer_grant = proc::request_rt_thread(90, 7, s->writer_period_ns, s->writer_period_ns / 2);
  s->grants_ready.fetch_add(1, std::memory_order_release);
  WriterStats ws;
  proc::Periodic per(s->writer_period_ns, proc::now_ns() + s->writer_period_ns);
  r1rt_lowstate_in st;
  std::uint64_t seen = 0;
  r1rt_lowcmd_out cmd;
  bool stalled = false;
  const std::int64_t stall_at = s->stall_at_s >= 0 ? static_cast<std::int64_t>(s->stall_at_s * 1e9) : -1;
  const std::int64_t t_start = proc::now_ns();
  bool tk_done = s->test_takeover_s < 0, rel_done = s->test_takeover_s < 0;
  std::int32_t rt_applied = -1;  // the grant last handed to the core (set from THIS thread: the core is single-writer)
  std::int64_t prev_iteration = 0;
  std::int64_t prev_wake = -1;
  while (!s->stop.load(std::memory_order_acquire)) {
    std::int64_t late = 0;
    const std::uint64_t skipped_before = per.skipped();
    std::int64_t wake = per.wait(&late);
    if (prev_wake >= 0 && wake - prev_wake > s->resume_gap_ns) {  // session 8 (#31): a stop, not an overrun
      ws.stops += 1;
      ws.stop_skipped += per.skipped() - skipped_before;
      if (wake - prev_wake > ws.stop_ns_max) ws.stop_ns_max = wake - prev_wake;
    }
    prev_wake = wake;
    // Test hook (--test-takeover): the takeover requests issued by the writer thread itself, for the process tests
    // that run without r1d; the operator path is the segment's mode mailbox (read inside r1rt_core_step).
    if (!tk_done && wake - t_start >= static_cast<std::int64_t>(s->test_takeover_s * 1e9)) {
      tk_done = true;
      const std::int32_t rc = r1rt_core_request(s->core, R1RT_REQ_TAKEOVER, wake);
      r1rt_core_status cs{};
      r1rt_core_get_status(s->core, &cs);  // a writer-thread call
      s->test_refusal.store(cs.takeover_refusal, std::memory_order_relaxed);
      s->test_takeover_status.store(rc, std::memory_order_release);
    }
    if (tk_done && !rel_done && wake - t_start >= static_cast<std::int64_t>(s->test_released_s * 1e9)) {
      rel_done = true;
      s->test_released_status.store(r1rt_core_request(s->core, R1RT_REQ_TAKEOVER_RELEASED, wake),
                                    std::memory_order_release);
    }
    // The fault hook (DESIGN §12.6 "a sleep at a code point"): a real sleep at the top of the writer loop.
    if (stall_at >= 0 && !stalled && ws.t_balance_ns >= 0 && wake - ws.t_balance_ns >= stall_at) {
      stalled = true;
      s->stall_started_ns.store(proc::now_ns(), std::memory_order_release);
      proc::sleep_for(static_cast<std::int64_t>(s->stall_dur_s * 1e9));
      wake = proc::now_ns();
      s->stall_ended_ns.store(wake, std::memory_order_release);
      per = proc::Periodic(s->writer_period_ns, wake + s->writer_period_ns);
    }
    const r1rt_lowstate_in* sp = s->bus->latest(&st, &seen) ? &st : nullptr;
    // The process's facts into the core (session 5): its real-time grants (heartbeat; the takeover_require_rt gate)
    // and this tick's wake-up lateness + the previous iteration's duration (the telemetry record, DESIGN §9).
    const std::int32_t g = s->rt_granted.load(std::memory_order_acquire);
    if (g >= 0 && g != rt_applied) {
      r1rt_core_set_rt_granted(s->core, g);
      rt_applied = g;
    }
    r1rt_core_set_loop_timing(s->core, clamp_i32(late), clamp_i32(prev_iteration));
    r1rt_core_step(s->core, sp, wake, &cmd, nullptr);
    if (cmd.publish) {
      // The token re-checked immediately before the write (the core checked it inside the step; a guard takeover in
      // between must not be followed by our packet: the residual race is bounded to this window). Session 8 (DESIGN
      // §4.3's residual race, seen once in 4 in the defaults rehearsal: ONE packet after SIGCONT while r1guard owned the
      // token): a process stopped between the step and the write wakes with a command computed before the stop - its
      // token check and its targets are stale. If the clock moved more than kStaleWriteNs since the tick's wake-up
      // (the check AFTER the token load: a stop between the load and this point shows), the command is dropped; the
      // next tick computes afresh (the core sees the stop: the resume rule).
      const std::uint64_t w = s->seg->token.load();
      const bool stale = proc::now_ns() - wake > kStaleWriteNs;
      if (stale) {
        ws.stale_skips += 1;
      } else if (shm::OwnershipToken::owner_of(w) == R1RT_OWNER_R1RT) {
        if (s->test_stall_armed.load(std::memory_order_relaxed) && s->test_stall_armed.exchange(false)) {
          // test hook (session 9): the whole process stopped HERE - after both checks, before the write
          s->test_stall_wake_ns.store(wake, std::memory_order_relaxed);
          std::raise(SIGSTOP);
          s->test_stall_cont_ns.store(proc::now_ns(), std::memory_order_release);
        }
        s->bus->write(cmd);
        ws.publishes += 1;
      } else {
        ws.not_owner_skips += 1;
      }
    }
    const r1rt_mode mode = r1rt_core_mode(s->core);
    if (mode == R1RT_MODE_BALANCE && ws.t_balance_ns < 0) {
      ws.t_balance_ns = wake;
      // The fault hook (DESIGN §12.6 "policy NaN"; session 6): the policy's output NaN in [BALANCE + AT, + DUR) - the
      // core's own injection (both the thread and the inline rung), registered from this thread (release-published)
      if (s->nan_at_s >= 0) {
        const std::int64_t t0 = wake + static_cast<std::int64_t>(s->nan_at_s * 1e9);
        if (r1rt_core_inject(s->core, R1RT_INJECT_POLICY_NAN, t0, t0 + static_cast<std::int64_t>(s->nan_dur_s * 1e9),
                             0.0) == R1RT_OK)
          s->nan_t0_ns.store(t0, std::memory_order_release);
      }
    }
    ws.mode = mode;
    ws.ticks += 1;
    ws.lateness.add(late);
    prev_iteration = proc::now_ns() - wake;
    ws.iteration.add(prev_iteration);
    ws.skipped_deadlines = per.skipped();
    if (ws.ticks % 50 == 0) {
      r1rt_core_status cst{};
      r1rt_core_get_status(s->core, &cst);  // a writer-thread call
      ws.resume_blends = cst.resume_blends;
      ws.resume_waits_expired = cst.resume_waits_expired;
      ws.resume_uncaught = cst.resume_uncaught;
      ws.resume_drain_waits = cst.resume_drain_waits;
      s->wstats.write(ws);
    }
  }
  s->wstats.write(ws);
}

void policy_thread(Shared* s) {
  // DESIGN §2.1: the policy thread FIFO 80 on CPU 6 (with the DDS receive threads and the motor NIC's IRQ).
  s->policy_grant = proc::request_rt_thread(80, 6, s->policy_period_ns, s->policy_period_ns / 4);
  s->grants_ready.fetch_add(1, std::memory_order_release);
  proc::Periodic per(s->policy_period_ns, proc::now_ns() + s->policy_period_ns);
  while (!s->stop.load(std::memory_order_acquire)) {
    std::int64_t late = 0;
    const std::int64_t wake = per.wait(&late);
    r1rt_core_policy_step(s->core, wake);
  }
}

void rx_thread(Shared* s) {
  // DESIGN §2.1: the DDS receive thread FIFO 85 on CPU 6.
  s->rx_grant = proc::request_rt_thread(85, 6, 1'000'000, 200'000);
  s->grants_ready.fetch_add(1, std::memory_order_release);
  s->bus->run_receive(s->stop);
}

const char* mode_name(int m) {
  static const char* const kNames[] = {"MONITOR",      "TAKEOVER_MIRROR", "TAKEOVER_BLEND",   "BALANCE", "GUARDED",
                                       "STAND_FREEZE", "ON_STAND",        "OVERLAY_ARM_SDK", "DAMP"};
  return (m >= 0 && m <= 8) ? kNames[m] : "?";
}

}  // namespace

int main(int argc, char** argv) {
  const Args a = parse_args(argc, argv);
  app::install_signal_handlers();

  proc::ProcConfig pc;
  std::string err;
  if (!proc::load_config(a.config.c_str(), pc, err)) app::die("config: " + err);
  if (a.rate_hz != 0) {
    pc.core.writer_rate_hz = a.rate_hz;
    pc.core.loop_gap_ns = 2 * (1'000'000'000 / static_cast<std::int64_t>(a.rate_hz));
  }
  if (!a.weights.empty()) pc.weights = a.weights;
  if (a.check_config) {  // parse only (rtconfig's test: the two sides agree on every key)
    app::emit(std::string("{\"config_ok\":true,\"keys\":") + std::to_string(proc::config_keys(nullptr)) +
              ",\"writer_hz\":" + std::to_string(pc.core.writer_rate_hz) + ",\"takeover_require_guard\":" +
              std::to_string(pc.core.takeover_require_guard) + ",\"weights\":" + app::jstr(pc.weights.c_str()) + "}");
    return 0;
  }

  int mlock_err = 0;
  const bool mlocked = proc::lock_memory(&mlock_err);

  ShmSegment seg;
  const ShmError se = ShmSegment::create(a.shm.c_str(), seg, a.replace_stale ? kReplaceStale : kShmNone);
  if (se != ShmError::kOk)
    app::die(std::string("segment ") + a.shm + ": " + shm_error_str(se) +
             (se == ShmError::kExists ? " (a stale segment? --replace-stale is the operator's decision)" : ""));

  pc.core.segment = seg.data();
  pc.core.segment_size = seg.size();
  pc.core.policy_weights_path = pc.weights.empty() ? nullptr : pc.weights.c_str();
  std::int32_t st = 0;
  r1rt_core* core = nullptr;
  // Session 8 (HW1 fix #2): every start-up failure after the segment exists closes it first (the creator unlinks the
  // name) - app::die's std::exit skips this frame's destructors, and at HW1 a DDS start failure (rmem_max) left
  // /dev/shm/r1mc3.rt behind: the next start refused "segment exists" until it was removed by hand.
  auto die_closing = [&](const std::string& msg) {
    r1rt_core_destroy(core);
    seg.close();
    app::die(msg);
  };
  core = r1rt_core_create(&pc.core, &st);
  if (core == nullptr) die_closing("r1rt_core_create failed: status " + std::to_string(st));
  // session 6 (DESIGN §3.5 start flag): a stand session - set before any thread runs, never cleared (restart r1rt
  // without the flag to balance)
  if (a.stand_session && r1rt_core_set_stand_session(core, 1) != R1RT_OK) die_closing("set_stand_session failed");

  dds_io::BusOptions bo;
  bo.domain = a.domain;
  bo.role = "r1rt";
  bo.write_lowcmd = true;
  bo.read_builtin = true;
  bo.on_builtin = builtin_cb;
  bo.builtin_ctx = core;
  dds_io::Bus bus(bo);
  if (!bus.ok()) die_closing("DDS: " + bus.error());

  Shared sh;
  sh.core = core;
  sh.seg = seg.segment();
  sh.bus = &bus;
  sh.writer_period_ns = 1'000'000'000 / static_cast<std::int64_t>(pc.core.writer_rate_hz);
  sh.resume_gap_ns = pc.core.resume_gap_ns > 0 ? pc.core.resume_gap_ns : R1RT_DEFAULT_RESUME_GAP_NS;
  sh.policy_period_ns = 1'000'000'000 / static_cast<std::int64_t>(pc.core.policy_rate_hz);
  sh.stall_at_s = a.stall_at_s;
  sh.stall_dur_s = a.stall_dur_s;
  sh.nan_at_s = a.nan_at_s;
  sh.nan_dur_s = a.nan_dur_s;
  sh.test_takeover_s = a.test_takeover_s;
  sh.test_released_s = a.test_released_s;

  std::thread trx(rx_thread, &sh);
  std::thread tpol(policy_thread, &sh);
  std::thread twr(writer_thread, &sh);
  const std::int64_t t_wait = proc::now_ns() + 2'000'000'000;
  while (sh.grants_ready.load(std::memory_order_acquire) < 3 && proc::now_ns() < t_wait) proc::sleep_for(1'000'000);

  // rt_granted = SCHED_FIFO on the writer AND the memory locked (DESIGN §2.1); never on macOS (stated, not an error).
  const bool rt_granted = sh.writer_grant.fifo && mlocked;
  sh.rt_granted.store(rt_granted ? 1 : 0, std::memory_order_release);  // the writer hands it to the core
  {
    char b[768];
    std::snprintf(b, sizeof b,
                  "{\"ready\":true,\"process\":\"r1rt\",\"pid\":%d,\"rt_granted\":%s,\"rate_hz\":%u,\"policy_hz\":%u,"
                  "\"domain\":%d,\"shm\":%s,\"os\":\"%s\",\"abi\":%u,\"mlock\":%s,\"mlock_errno\":%d,"
                  "\"policy_loaded\":%s,\"fault_writer_stall\":[%.3f,%.3f],\"fault_policy_nan\":[%.3f,%.3f],"
                  "\"stand_session\":%s,",
                  static_cast<int>(getpid()), rt_granted ? "true" : "false", pc.core.writer_rate_hz,
                  pc.core.policy_rate_hz, a.domain, app::jstr(a.shm.c_str()).c_str(), proc::os_name(),
                  r1rt_abi_version(), mlocked ? "true" : "false", mlock_err, pc.weights.empty() ? "false" : "true",
                  a.stall_at_s, a.stall_dur_s, a.nan_at_s, a.nan_dur_s, a.stand_session ? "true" : "false");
    app::emit(std::string(b) + "\"sched\":{\"writer\":" + app::grant_json(sh.writer_grant) +
              ",\"policy\":" + app::grant_json(sh.policy_grant) + ",\"dds_rx\":" + app::grant_json(sh.rx_grant) +
              "}}");
  }

  const char* stall_path = std::getenv("R1RT_TEST_STALL_BEFORE_WRITE");  // session 9 test hook (header comment)
  bool stall_hook_reported = false;
  std::uint64_t handled_signals = 0;
  bool stall_start_reported = false, stall_end_reported = false;
  bool tk_reported = false, rel_reported = false;
  bool nan_reported = false;
  std::int64_t next_status = proc::now_ns() + 1'000'000'000;
  int exit_code = 0;
  for (;;) {
    proc::sleep_for(20'000'000);
    r1rt_heartbeat hb{};
    const bool have_hb = sh.seg->rt_heartbeat.try_read(hb) && hb.seq != 0;
    const int mode = have_hb ? static_cast<int>(hb.mode) : static_cast<int>(R1RT_MODE_MONITOR);  // GCC: -Werror=extra
    if (stall_path != nullptr && *stall_path != '\0' && !sh.test_stall_armed.load(std::memory_order_relaxed) &&
        ::access(stall_path, F_OK) == 0 && ::unlink(stall_path) == 0) {
      sh.test_stall_armed.store(true, std::memory_order_release);
      app::emit("{\"event\":\"test_stall_armed\"}");
      stall_hook_reported = false;
    }
    if (!stall_hook_reported && sh.test_stall_cont_ns.load(std::memory_order_acquire) != 0) {
      stall_hook_reported = true;
      app::emit("{\"event\":\"test_stall_before_write\",\"wake_ns\":" +
                std::to_string(sh.test_stall_wake_ns.load()) + ",\"cont_ns\":" +
                std::to_string(sh.test_stall_cont_ns.load()) + "}");
      sh.test_stall_cont_ns.store(0, std::memory_order_relaxed);
    }
    if (!stall_start_reported && sh.stall_started_ns.load(std::memory_order_acquire) != 0) {
      stall_start_reported = true;
      app::emit("{\"event\":\"fault_writer_stall\",\"t_start_ns\":" +
                std::to_string(sh.stall_started_ns.load()) + ",\"dur_s\":" + std::to_string(a.stall_dur_s) + "}");
    }
    if (!stall_end_reported && sh.stall_ended_ns.load(std::memory_order_acquire) != 0) {
      stall_end_reported = true;
      app::emit("{\"event\":\"fault_writer_stall_end\",\"t_end_ns\":" + std::to_string(sh.stall_ended_ns.load()) +
                "}");
    }
    if (!nan_reported && sh.nan_t0_ns.load(std::memory_order_acquire) != 0) {
      nan_reported = true;
      const std::int64_t t0 = sh.nan_t0_ns.load();
      app::emit("{\"event\":\"fault_policy_nan\",\"t0_ns\":" + std::to_string(t0) + ",\"t1_ns\":" +
                std::to_string(t0 + static_cast<std::int64_t>(a.nan_dur_s * 1e9)) + "}");
    }
    if (!tk_reported && sh.test_takeover_status.load(std::memory_order_acquire) != 1) {
      tk_reported = true;
      app::emit("{\"event\":\"test_takeover\",\"status\":" + std::to_string(sh.test_takeover_status.load()) +
                ",\"refusal\":" + std::to_string(sh.test_refusal.load()) + "}");
    }
    if (!rel_reported && sh.test_released_status.load(std::memory_order_acquire) != 1) {
      rel_reported = true;
      app::emit("{\"event\":\"test_released\",\"status\":" + std::to_string(sh.test_released_status.load()) + "}");
    }
    const std::uint64_t nsig = app::g_signal_count.load(std::memory_order_relaxed);
    if (nsig != handled_signals) {
      handled_signals = nsig;
      const int sig = app::g_signal.load(std::memory_order_relaxed);
      if (mode == R1RT_MODE_MONITOR || mode == R1RT_MODE_DAMP) {
        app::emit(std::string("{\"event\":\"exit\",\"signal\":") + std::to_string(sig) + ",\"mode\":\"" +
                  mode_name(mode) + "\"}");
        break;
      }
      // Never stop publishing on a signal (a stopped writer = a limp robot or a guard takeover): refuse, say so.
      app::emit(std::string("{\"event\":\"signal_refused\",\"signal\":") + std::to_string(sig) + ",\"mode\":\"" +
                mode_name(mode) + "\",\"why\":\"publishing: damp first (operator), then stop\"}");
      std::fprintf(stderr, "r1rt: signal %d refused in %s (publishing; never damp on a stop - operator: damp first)\n",
                   sig, mode_name(mode));
    }
    if (proc::now_ns() >= next_status) {
      next_status += 1'000'000'000;
      WriterStats w;
      if (!sh.wstats.try_read(w)) continue;
      const auto& c = bus.counters();
      char lb[512], ib[512];
      // session 10: the state-age counters (session 9's STATE_OLD alarm, docs/sim/STATE_AGE.md) from the segment's
      // executor status (layout 4) - the numbers r1d reads; HW2's log check (D34) reads them from this line
      char sa[512];
      r1rt_exec_status es{};
      if (sh.seg->exec_status.try_read(es) && es.seq != 0) {
        std::snprintf(sa, sizeof sa,
                      "{\"valid\":%s,\"content_ms\":%.3f,\"content_ms_max\":%.3f,\"old_ticks\":%" PRIu64
                      ",\"old_episodes\":%" PRIu64 ",\"old_ms_max\":%.3f,\"rung_entries\":%" PRIu64
                      ",\"relearns\":%" PRIu64 ",\"stale_freeze\":%s,\"stale_rung\":%u}",
                      es.content_age_valid ? "true" : "false",
                      es.state_content_age_ns < 0 ? -1.0 : static_cast<double>(es.state_content_age_ns) * 1e-6,
                      static_cast<double>(es.state_content_age_max_ns) * 1e-6, es.state_old_ticks,
                      es.state_old_episodes, static_cast<double>(es.state_old_ns_max) * 1e-6, es.stale_rung_entries,
                      es.content_age_relearns, es.stale_freeze ? "true" : "false",
                      static_cast<unsigned>(es.stale_rung));
      } else {
        std::snprintf(sa, sizeof sa, "null");
      }
      char b[2560];
      std::snprintf(b, sizeof b,
                    "{\"status\":\"r1rt\",\"t_ns\":%" PRId64 ",\"mode\":\"%s\",\"ticks\":%" PRIu64
                    ",\"publishes\":%" PRIu64 ",\"not_owner_skips\":%" PRIu64 ",\"skipped_deadlines\":%" PRIu64
                    ",\"faults\":%u,\"token_owner\":%u,\"lowstate_rx\":%" PRIu64 ",\"lowstate_crc_bad\":%" PRIu64
                    ",\"builtin_rx\":%" PRIu64 ",\"own_lowcmd_rx\":%" PRIu64 ",\"foreign_writers\":%" PRIu64
                    ",\"dds_writes\":%" PRIu64 ",\"dds_write_errors\":%" PRIu64 ",\"stale_skips\":%" PRIu64 ",\"stops\":%" PRIu64
                    ",\"stop_skipped\":%" PRIu64 ",\"overruns\":%" PRIu64 ",\"stop_ms_max\":%.3f"
                    ",\"resume_blends\":%" PRIu64 ",\"resume_waits_expired\":%" PRIu64 ",\"resume_uncaught\":%" PRIu64
                    ",\"resume_drain_waits\":%" PRIu64 ",\"lateness\":%s,\"iteration\":%s,\"state_age\":%s}",
                    proc::now_ns(), mode_name(mode), w.ticks, w.publishes, w.not_owner_skips, w.skipped_deadlines,
                    have_hb ? hb.faults : 0u,
                    static_cast<unsigned>(shm::OwnershipToken::owner_of(sh.seg->token.load())),
                    c.lowstate_rx.load(), c.lowstate_crc_bad.load(), c.builtin_rx.load(), c.own_lowcmd_rx.load(),
                    c.foreign_writers.load(), c.writes.load(), c.write_errors.load(), w.stale_skips, w.stops, w.stop_skipped,
                    w.skipped_deadlines - w.stop_skipped, static_cast<double>(w.stop_ns_max) * 1e-6, w.resume_blends,
                    w.resume_waits_expired, w.resume_uncaught, w.resume_drain_waits, w.lateness.json(lb, sizeof lb),
                    w.iteration.json(ib, sizeof ib), sa);
      app::emit(b);
    }
  }

  sh.stop.store(true, std::memory_order_release);
  twr.join();
  tpol.join();
  trx.join();
  r1rt_core_destroy(core);
  seg.close();  // the creator unlinks the name
  return exit_code;
}
