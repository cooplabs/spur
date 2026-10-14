// r1guard - the hot standby as a PROCESS (session 5; DESIGN.md §2.1, §4.3 rung 2; GUARD.md, PROCESSES.md).
//
//   r1guard --config FILE --shm-name NAME --domain N [--observe-only] [--attach-timeout-s 10]
//
// Attaches to r1rt's segment (a bounded wait for its creation), runs r1rt::Guard::cycle() at 1 kHz on its OWN rt/lowstate
// reader (its own DDS receive thread), and writes rt/lowcmd only while it owns the token (checked right before each
// write). After the first write following a takeover, a non-real-time thread captures the stalled core's /proc
// forensics (Linux; ENOSYS on macOS) and prints them as one JSON event. It never damps and never hands the token back
// (an operator decision). SIGTERM / SIGINT while it publishes are refused (logged); on standby they exit cleanly.
//
// stdout: JSON lines - {"ready": true, ...}, {"event": "takeover", ...}, {"status": "r1guard", ...} every second.
#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#include "app_common.hpp"
#include "r1rt/dds_io.hpp"
#include "r1rt/guard.hpp"
#include "r1rt/proc_config.hpp"
#include "r1rt/rt_thread.hpp"
#include "r1rt/seqlock.hpp"
#include "r1rt/shm.hpp"
#include "r1rt/shm_layout.h"

namespace {

using namespace r1rt;

struct Args {
  std::string config, shm = "/r1mc3.rt";
  int domain = 0;
  bool observe_only = false;
  double attach_timeout_s = 10.0;
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
    } else if (k == "--observe-only") {
      a.observe_only = true;
    } else if (k == "--attach-timeout-s") {
      a.attach_timeout_s = std::atof(val().c_str());
    } else if (k == "-h" || k == "--help") {
      std::printf("r1guard --config FILE --shm-name NAME --domain N [--observe-only] [--attach-timeout-s 10]\n");
      std::exit(0);
    } else {
      app::die("unknown argument " + k);
    }
  }
  if (a.config.empty()) app::die("--config FILE is required");
  return a;
}

struct CycleStats {
  proc::TimingStats lateness;
  proc::TimingStats iteration;
  std::uint64_t writes = 0;
  std::uint64_t not_owner_skips = 0;
  std::int64_t t_first_write_ns = 0;  // the first write after the takeover
};

struct Shared {
  Guard* guard = nullptr;
  shm::Segment* seg = nullptr;
  dds_io::Bus* bus = nullptr;
  std::int64_t period_ns = 1'000'000;
  std::uint64_t write_every = 2;  // cycle rate / the core's writer rate (1 kHz / 500 Hz)
  std::atomic<bool> stop{false};
  std::atomic<int> grants_ready{0};
  proc::RtGrant cycle_grant, rx_grant;
  Seqlock<CycleStats> cstats;
  Seqlock<GuardStatus> gstatus;
  std::atomic<std::int64_t> first_write_ns{0};  // set by the cycle thread at each takeover's first write
  std::atomic<std::uint64_t> first_write_takeover{0};  // session 6: which takeover (re-armed guards take over again)
  std::atomic<std::int32_t> rt_granted{-1};     // -1 = not yet known; main sets it once the threads report
};

void cycle_thread(Shared* s) {
  // DESIGN §2.1: r1guard FIFO 88 on CPU 5.
  s->cycle_grant = proc::request_rt_thread(88, 5, s->period_ns, s->period_ns / 2);
  s->grants_ready.fetch_add(1, std::memory_order_release);
  CycleStats cs;
  proc::Periodic per(s->period_ns, proc::now_ns() + s->period_ns);
  r1rt_lowstate_in st;
  std::uint64_t seen = 0;
  r1rt_lowcmd_out out;
  GuardStatus gst{};
  std::uint64_t n = 0;
  std::uint64_t active_cycles = 0;
  std::uint64_t written_takeover = 0;  // session 6: the takeover whose first write was reported
  std::int32_t rt_applied = -1;
  while (!s->stop.load(std::memory_order_acquire)) {
    std::int64_t late = 0;
    const std::int64_t wake = per.wait(&late);
    const std::int32_t g = s->rt_granted.load(std::memory_order_acquire);
    if (g >= 0 && g != rt_applied) {  // the guard's grants into its heartbeat (set from the cycle thread only)
      s->guard->set_rt_granted(g != 0);
      rt_applied = g;
    }
    const r1rt_lowstate_in* sp = s->bus->latest(&st, &seen) ? &st : nullptr;
    s->guard->cycle(sp, wake, &out);
    // Active: the guard cycles at 1 kHz but writes at the core's writer rate (DESIGN §2.1 "writer rate when active";
    // the first write in the takeover's own cycle, then every `write_every`-th cycle).
    if (!out.publish) active_cycles = 0;  // session 6: a re-armed guard's next takeover writes in its own cycle again
    if (out.publish && (active_cycles++ % s->write_every) == 0) {
      const std::uint64_t w = s->seg->token.load();
      if (shm::OwnershipToken::owner_of(w) == R1RT_OWNER_R1GUARD) {
        s->bus->write(out);
        cs.writes += 1;
        const std::uint64_t tk = s->guard->takeovers();
        if (tk != written_takeover) {  // each takeover's first write (session 6: the guard re-arms after a hand-back)
          written_takeover = tk;
          cs.t_first_write_ns = proc::now_ns();
          s->first_write_takeover.store(tk, std::memory_order_relaxed);
          s->first_write_ns.store(cs.t_first_write_ns, std::memory_order_release);
        }
      } else {
        cs.not_owner_skips += 1;
      }
    }
    cs.lateness.add(late);
    cs.iteration.add(proc::now_ns() - wake);
    if (++n % 100 == 0) {
      s->guard->status(&gst);
      s->gstatus.write(gst);
      s->cstats.write(cs);
    }
  }
  s->guard->status(&gst);
  s->gstatus.write(gst);
  s->cstats.write(cs);
}

void rx_thread(Shared* s) {
  s->rx_grant = proc::request_rt_thread(85, 5, 1'000'000, 200'000);
  s->grants_ready.fetch_add(1, std::memory_order_release);
  s->bus->run_receive(s->stop);
}

const char* mode_name(int m) {
  static const char* const kNames[] = {"MONITOR",      "TAKEOVER_MIRROR", "TAKEOVER_BLEND",   "BALANCE", "GUARDED",
                                       "STAND_FREEZE", "ON_STAND",        "OVERLAY_ARM_SDK", "DAMP"};
  return (m >= 0 && m <= 8) ? kNames[m] : "?";
}

std::string forensics_json(const GuardForensics& f, std::int64_t t_first_write_ns) {
  char b[1024];
  std::snprintf(b, sizeof b,
                "{\"event\":\"takeover\",\"process\":\"r1guard\",\"t_takeover_ns\":%" PRId64
                ",\"t_first_write_ns\":%" PRId64 ",\"rt_beat_t_ns\":%" PRId64 ",\"rt_silence_ns\":%" PRId64
                ",\"mode_from\":\"%s\",\"rt_faults\":%u,\"rt_pid\":%d,\"token_before\":%" PRIu64
                ",\"token_after\":%" PRIu64 ",\"guard_lowstate_age_ns\":%" PRId64
                ",\"tilt_deg\":%.3f,\"shadow_valid\":%u,\"shadow_dev_rad\":%.6f,\"proc_err\":%d,\"n_threads\":%d,"
                "\"threads\":[",
                f.t_takeover_ns, t_first_write_ns, f.rt_beat_t_ns, f.rt_silence_ns, mode_name(f.mode_from), f.rt_faults,
                f.rt_pid, f.token_before, f.token_after, f.guard_lowstate_age_ns, f.tilt_rad * 57.29577951308232,
                f.shadow_valid, f.shadow_dev_rad, f.proc_err, f.n_threads);
  std::string o = b;
  for (int i = 0; i < f.n_threads && i < 8; ++i) {
    const auto& t = f.threads[i];
    o += (i ? "," : "");
    o += "{\"tid\":" + std::to_string(t.tid) + ",\"err\":" + std::to_string(t.err) + ",\"comm\":" + app::jstr(t.comm) +
         ",\"wchan\":" + app::jstr(t.wchan) + ",\"syscall\":" + app::jstr(t.syscall) + "}";
  }
  return o + "]}";
}

}  // namespace

int main(int argc, char** argv) {
  const Args a = parse_args(argc, argv);
  app::install_signal_handlers();

  proc::ProcConfig pc;
  std::string err;
  if (!proc::load_config(a.config.c_str(), pc, err)) app::die("config: " + err);

  int mlock_err = 0;
  const bool mlocked = proc::lock_memory(&mlock_err);

  // r1rt creates the segment; wait for it (bounded) - a guard that never attaches is reported, never silent.
  // Session 13 (HW2 day 2): also while r1rt is between shm_open and ftruncate (the 0-byte object: "segment smaller than
  // expected" 8 ms after r1rt's start, once in four starts) - for at most 2 s of that state (ShmSegment::attach_wait).
  ShmSegment seg;
  ShmSegment::AttachWait aw{};
  const ShmError se =
      ShmSegment::attach_wait(a.shm.c_str(), seg, static_cast<std::int64_t>(a.attach_timeout_s * 1e9), &aw);
  if (se != ShmError::kOk) {
    char why[200];
    std::snprintf(why, sizeof why, " (after %d tries over %lld ms, %d of them a segment still being sized; "
                  "--attach-timeout-s %.1f, a too-small segment at most 2 s)", aw.tries,
                  static_cast<long long>(aw.waited_ns / 1000000), aw.too_small, a.attach_timeout_s);
    app::die(std::string("attach ") + a.shm + ": " + shm_error_str(se) + why);
  }
  if (aw.too_small > 0)
    std::fprintf(stderr, "r1guard: attached %s after %d tries over %lld ms (%d while r1rt was still sizing it)\n",
                 a.shm.c_str(), aw.tries, static_cast<long long>(aw.waited_ns / 1000000), aw.too_small);

  pc.core.segment = seg.data();
  pc.core.segment_size = seg.size();
  pc.core.policy_weights_path = pc.weights.empty() ? nullptr : pc.weights.c_str();
  GuardParams gp = pc.guard;
  if (a.observe_only) gp.takeover_enabled = 0;
  static Guard guard;  // large (policy runner + forensics): not on main's stack
  char gerr[256] = {};
  const std::int32_t gs = guard.init(pc.core, gp, gerr, sizeof gerr);
  if (gs != R1RT_OK) app::die(std::string("guard init: ") + gerr + " (status " + std::to_string(gs) + ")");

  dds_io::BusOptions bo;
  bo.domain = a.domain;
  bo.role = "r1guard";
  bo.write_lowcmd = true;
  bo.read_builtin = false;
  dds_io::Bus bus(bo);
  if (!bus.ok()) app::die("DDS: " + bus.error());

  Shared sh;
  sh.guard = &guard;
  sh.seg = seg.segment();
  sh.bus = &bus;
  sh.period_ns = 1'000'000'000 / static_cast<std::int64_t>(gp.cycle_rate_hz);
  sh.write_every = std::max<std::uint64_t>(1, gp.cycle_rate_hz / std::max<std::uint32_t>(1, pc.core.writer_rate_hz));

  std::thread trx(rx_thread, &sh);
  std::thread tcy(cycle_thread, &sh);
  const std::int64_t t_wait = proc::now_ns() + 2'000'000'000;
  while (sh.grants_ready.load(std::memory_order_acquire) < 2 && proc::now_ns() < t_wait) proc::sleep_for(1'000'000);
  const bool rt_granted = sh.cycle_grant.fifo && mlocked;
  sh.rt_granted.store(rt_granted ? 1 : 0, std::memory_order_release);
  {
    char b[512];
    std::snprintf(b, sizeof b,
                  "{\"ready\":true,\"process\":\"r1guard\",\"pid\":%d,\"rt_granted\":%s,\"cycle_hz\":%u,"
                  "\"stale_periods\":%u,\"write_every\":%llu,\"takeover_enabled\":%s,\"domain\":%d,\"shm\":%s,\"os\":\"%s\",\"mlock\":%s,"
                  "\"mlock_errno\":%d,\"rt_pid\":%d,",
                  static_cast<int>(getpid()), rt_granted ? "true" : "false", gp.cycle_rate_hz, gp.stale_periods,
                  static_cast<unsigned long long>(sh.write_every), gp.takeover_enabled ? "true" : "false", a.domain, app::jstr(a.shm.c_str()).c_str(), proc::os_name(),
                  mlocked ? "true" : "false", mlock_err, static_cast<int>(sh.seg->header.creator_pid));
    app::emit(std::string(b) + "\"sched\":{\"cycle\":" + app::grant_json(sh.cycle_grant) +
              ",\"dds_rx\":" + app::grant_json(sh.rx_grant) + "}}");
  }

  std::uint64_t handled_signals = 0;
  std::int64_t forensics_for = 0;  // the first-write time already reported (one forensics line per takeover)
  std::uint64_t seen_rearms = 0, seen_damps = 0, seen_ignored = 0;
  std::int64_t next_status = proc::now_ns() + 1'000'000'000;
  for (;;) {
    proc::sleep_for(5'000'000);
    const std::int64_t fw = sh.first_write_ns.load(std::memory_order_acquire);
    if (fw != 0 && fw != forensics_for) {
      // File I/O on the main thread (never in cycle()): the stalled core's threads from /proc (Linux).
      forensics_for = fw;
      GuardForensics& f = guard.forensics_for_capture();
      f.proc_err = capture_proc_forensics(f.rt_pid, &f);
      app::emit(forensics_json(f, fw));
    }
    GuardStatus gst{};
    const bool have = sh.gstatus.try_read(gst);
    if (have) {  // session 6: the hand-back and the operator's damp as event lines (the status poll's resolution)
      if (gst.rearms != seen_rearms) {
        seen_rearms = gst.rearms;
        app::emit("{\"event\":\"rearmed\",\"process\":\"r1guard\",\"rearms\":" + std::to_string(gst.rearms) +
                  ",\"t_ns\":" + std::to_string(proc::now_ns()) + "}");
      }
      if (gst.operator_damps != seen_damps) {
        seen_damps = gst.operator_damps;
        app::emit("{\"event\":\"operator_damp\",\"process\":\"r1guard\",\"t_ns\":" + std::to_string(proc::now_ns()) +
                  "}");
      }
      if (gst.damp_ignored != seen_ignored) {
        seen_ignored = gst.damp_ignored;
        app::emit("{\"event\":\"damp_ignored\",\"process\":\"r1guard\",\"count\":" +
                  std::to_string(gst.damp_ignored) + ",\"why\":\"not from the operator\"}");
      }
    }
    const std::uint64_t nsig = app::g_signal_count.load(std::memory_order_relaxed);
    if (nsig != handled_signals) {
      handled_signals = nsig;
      const int sig = app::g_signal.load(std::memory_order_relaxed);
      // standby: exits; active in the OPERATOR's damp too (session 6, as r1rt exits in DAMP: the board keeps the last
      // damp command - nothing of ours holds anything); active and holding: refused
      if (!(have && gst.active) || gst.act_mode == R1RT_MODE_DAMP) {
        app::emit("{\"event\":\"exit\",\"process\":\"r1guard\",\"signal\":" + std::to_string(sig) + "}");
        break;
      }
      app::emit("{\"event\":\"signal_refused\",\"process\":\"r1guard\",\"signal\":" + std::to_string(sig) +
                ",\"why\":\"the guard owns rt/lowcmd: damp first (operator), then stop\"}");
      std::fprintf(stderr, "r1guard: signal %d refused (active: publishing)\n", sig);
    }
    if (proc::now_ns() >= next_status && have) {
      next_status += 1'000'000'000;
      CycleStats cs;
      if (!sh.cstats.try_read(cs)) continue;
      const auto& c = bus.counters();
      char lb[512], ib[512], b[2048];
      std::snprintf(b, sizeof b,
                    "{\"status\":\"r1guard\",\"t_ns\":%" PRId64 ",\"active\":%s,\"armed\":%s,\"rearms\":%" PRIu64
                    ",\"operator_damps\":%" PRIu64 ",\"act_mode\":\"%s\","
                    "\"mode_from\":\"%s\",\"cycles\":%" PRIu64 ",\"publishes\":%" PRIu64 ",\"writes\":%" PRIu64
                    ",\"not_owner_skips\":%" PRIu64 ",\"takeovers\":%" PRIu64 ",\"shadow_steps\":%" PRIu64
                    ",\"shadow_invalid\":%" PRIu64 ",\"rt_beat_age_ns\":%" PRId64 ",\"lowstate_age_ns\":%" PRId64
                    ",\"faults\":%u,\"tilt_deg\":%.3f,\"token_owner\":%u,\"lowstate_rx\":%" PRIu64
                    ",\"stops\":%" PRIu64 ",\"resume_takeovers\":%" PRIu64 ",\"resume_blends\":%" PRIu64
                    ",\"resume_waits_expired\":%" PRIu64 ",\"resume_wait_ms_last\":%.3f,\"resume_wait_samples_last\":%" PRIu64
                    ",\"resume_wait_old_samples_last\":%" PRIu64 ",\"resume_uncaught\":%" PRIu64
                    ",\"resume_drain_waits\":%" PRIu64 ",\"lateness\":%s,\"iteration\":%s}",
                    proc::now_ns(), gst.active ? "true" : "false", gst.armed ? "true" : "false", gst.rearms,
                    gst.operator_damps, mode_name(gst.act_mode), mode_name(gst.mode_from),
                    gst.cycles, gst.publishes, cs.writes, cs.not_owner_skips, gst.takeovers, gst.shadow_steps,
                    gst.shadow_invalid, gst.rt_beat_age_ns, gst.lowstate_age_ns, gst.faults,
                    gst.tilt_rad * 57.29577951308232,
                    static_cast<unsigned>(shm::OwnershipToken::owner_of(sh.seg->token.load())),
                    c.lowstate_rx.load(), gst.stops, gst.resume_takeovers, gst.resume_blends,
                    gst.resume_waits_expired, static_cast<double>(gst.resume_wait_ns_last) * 1e-6,
                    gst.resume_wait_samples_last, gst.resume_wait_old_samples_last, gst.resume_uncaught,
                    gst.resume_drain_waits, cs.lateness.json(lb, sizeof lb),
                    cs.iteration.json(ib, sizeof ib));
      app::emit(b);
    }
  }
  sh.stop.store(true, std::memory_order_release);
  tcy.join();
  trx.join();
  seg.close();
  return 0;
}
