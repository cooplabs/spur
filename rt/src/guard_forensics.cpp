// capture_proc_forensics (guard.hpp): what the stalled core's threads were doing, from /proc (Linux). DESIGN.md §4.3
// rung 2 / §9: "forensics (/proc/<tid>/stack, wchan, syscall of the stalled thread) to the flight recorder". File I/O:
// the guard's process wrapper calls it AFTER the takeover's first write, never inside Guard::cycle().
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "r1rt/guard.hpp"

#if defined(__linux__)
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace r1rt {

#if defined(__linux__)
namespace {

// One small /proc file into buf (NUL-terminated, trailing newline removed). 0 or errno.
int read_small(const char* path, char* buf, std::size_t n) noexcept {
  buf[0] = '\0';
  const int fd = ::open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) return errno;
  ssize_t k = ::read(fd, buf, n - 1);
  const int e = k < 0 ? errno : 0;
  ::close(fd);
  if (k < 0) return e;
  buf[k] = '\0';
  while (k > 0 && (buf[k - 1] == '\n' || buf[k - 1] == ' ')) buf[--k] = '\0';
  return 0;
}

}  // namespace

std::int32_t capture_proc_forensics(std::int32_t pid, GuardForensics* f) noexcept {
  f->n_threads = 0;
  char dir[64];
  std::snprintf(dir, sizeof dir, "/proc/%d/task", static_cast<int>(pid));
  DIR* d = ::opendir(dir);
  if (d == nullptr) {
    f->proc_err = errno;
    return f->proc_err;
  }
  constexpr int kMax = static_cast<int>(sizeof(f->threads) / sizeof(f->threads[0]));
  while (f->n_threads < kMax) {
    const dirent* e = ::readdir(d);
    if (e == nullptr) break;
    if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
    GuardProcThread& t = f->threads[f->n_threads++];
    t.tid = static_cast<std::int32_t>(std::atoi(e->d_name));
    t.err = 0;
    char path[384];  // dir + d_name (<= 255) + the item name: GCC's -Wformat-truncation needs the room
    struct Item {
      const char* name;
      char* buf;
      std::size_t n;
    } items[] = {{"comm", t.comm, sizeof t.comm},
                 {"wchan", t.wchan, sizeof t.wchan},
                 {"syscall", t.syscall, sizeof t.syscall},
                 {"stack", t.stack, sizeof t.stack}};
    for (const Item& it : items) {
      const int len = std::snprintf(path, sizeof path, "%s/%s/%s", dir, e->d_name, it.name);
      if (len < 0 || static_cast<std::size_t>(len) >= sizeof path) continue;  // never read a truncated path
      const int rc = read_small(path, it.buf, it.n);
      if (rc != 0 && t.err == 0) t.err = rc;  // stack needs CAP_SYS_ADMIN: EACCES is the normal case
    }
  }
  ::closedir(d);
  f->proc_err = 0;
  return 0;
}
#else
std::int32_t capture_proc_forensics(std::int32_t pid, GuardForensics* f) noexcept {
  (void)pid;
  f->n_threads = 0;
  f->proc_err = ENOSYS;  // no /proc task files (macOS): the lockstep record (heartbeat, mirror, token) stays
  return ENOSYS;
}
#endif

}  // namespace r1rt
