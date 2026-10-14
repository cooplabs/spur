// Shared by the r1rt and r1guard process wrappers (rt/apps): argument helpers, the signal flag, JSON-line output.
// Process plumbing only - the real-time behaviour lives in the libraries (Core, Guard), DESIGN §2 principle 2.
#pragma once

#include <signal.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#include "r1rt/rt_thread.hpp"

namespace r1rt::app {

// The last SIGTERM / SIGINT received (0 = none). The handler only stores it; the main thread decides.
inline std::atomic<int> g_signal{0};
inline std::atomic<std::uint64_t> g_signal_count{0};

inline void on_signal(int sig) {
  g_signal.store(sig, std::memory_order_relaxed);
  g_signal_count.fetch_add(1, std::memory_order_relaxed);
}

inline void install_signal_handlers() {
  struct sigaction sa;
  std::memset(&sa, 0, sizeof sa);
  sa.sa_handler = on_signal;
  sigemptyset(&sa.sa_mask);
  sigaction(SIGTERM, &sa, nullptr);
  sigaction(SIGINT, &sa, nullptr);
  signal(SIGPIPE, SIG_IGN);  // a closed stdout must never kill a publishing process
}

// One JSON object per line on stdout (the ready line, events, status); serialised, flushed.
inline std::mutex g_out_mu;
inline void emit(const std::string& line) {
  std::lock_guard<std::mutex> lk(g_out_mu);
  std::fputs(line.c_str(), stdout);
  std::fputc('\n', stdout);
  std::fflush(stdout);
}

inline std::string jstr(const char* s) {
  std::string o = "\"";
  for (const char* p = s; *p; ++p) {
    const unsigned char c = static_cast<unsigned char>(*p);
    if (c == '"' || c == '\\') {
      o += '\\';
      o += static_cast<char>(c);
    } else if (c < 0x20) {
      char b[8];
      std::snprintf(b, sizeof b, "\\u%04x", c);
      o += b;
    } else {
      o += static_cast<char>(c);
    }
  }
  return o + "\"";
}

inline std::string grant_json(const proc::RtGrant& g) {
  char b[256];
  std::snprintf(b, sizeof b,
                "{\"fifo\":%s,\"priority\":%d,\"fifo_errno\":%d,\"cpu\":%d,\"affinity\":%s,\"affinity_errno\":%d,"
                "\"time_constraint\":%s}",
                g.fifo ? "true" : "false", g.priority, g.fifo_errno, g.cpu, g.affinity ? "true" : "false",
                g.affinity_errno, g.time_constraint ? "true" : "false");
  return b;
}

[[noreturn]] inline void die(const std::string& msg, int code = 2) {
  emit(std::string("{\"error\":") + jstr(msg.c_str()) + "}");
  std::fprintf(stderr, "%s\n", msg.c_str());
  std::exit(code);
}

}  // namespace r1rt::app
