// The DDS layer of the r1rt / r1guard processes (dds_io.hpp). CycloneDDS-CXX 0.10.2 (the ISO C++ PSM) for the
// entities and samples; the C API for the one thing the PSM does not expose simply: a matched publication's QoS
// (its user_data tells our own writers from the built-in controller's).
#include "r1rt/dds_io.hpp"

#include <cstring>
#include <vector>

#include "dds/dds.h"
#include "dds/dds.hpp"
#include "r1rt/crc32.hpp"
#include "r1rt/hg_msgs.hpp"
#include "unitree/idl/hg/LowCmd_.hpp"
#include "unitree/idl/hg/LowState_.hpp"

// LeakSanitizer (on by default with ASan on Linux; off on macOS): cyclonedds-cxx 0.10.2 leaks the user-data buffer it
// converts for every DataWriter QoS with a UserData policy (core::convertByteSeq -> dds_alloc, never freed; ~13 B once
// per writer - our writers tag themselves "r1mc3:<role>"). A third-party leak at start-up, not ours: suppressed for
// every sanitizer build that links this file (r1rt, r1guard, the DDS tests), so LSan neither fails the tests nor turns
// the processes' clean exit codes into 23 (session 6, the Linux build check).
#if defined(__SANITIZE_ADDRESS__)
#define R1RT_LSAN_ON 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define R1RT_LSAN_ON 1
#endif
#endif
#if defined(R1RT_LSAN_ON)
extern "C" const char* __lsan_default_suppressions();  // NOLINT(bugprone-reserved-identifier): LSan's hook
extern "C" const char* __lsan_default_suppressions() { return "leak:org::eclipse::cyclonedds::core::convertByteSeq\n"; }
#endif

namespace r1rt::dds_io {
namespace {
constexpr std::uint32_t kSlot[R1RT_NUM_JOINTS] = R1RT_HG_SLOT_MAP_INIT;

hg::LowState image_of(const uhg::LowState_& m) noexcept {
  hg::LowState s;
  hg::clear(s);
  s.version[0] = m.version()[0];
  s.version[1] = m.version()[1];
  s.mode_pr = m.mode_pr();
  s.mode_machine = m.mode_machine();
  s.tick = m.tick();
  const auto& imu = m.imu_state();
  for (std::size_t i = 0; i < 4; ++i) s.imu_state.quaternion[i] = imu.quaternion()[i];
  for (std::size_t i = 0; i < 3; ++i) {
    s.imu_state.gyroscope[i] = imu.gyroscope()[i];
    s.imu_state.accelerometer[i] = imu.accelerometer()[i];
    s.imu_state.rpy[i] = imu.rpy()[i];
  }
  s.imu_state.temperature = imu.temperature();
  for (std::size_t k = 0; k < static_cast<std::size_t>(hg::kNumMotorSlots); ++k) {
    const auto& d = m.motor_state()[k];
    hg::MotorState& o = s.motor_state[k];
    o.mode = d.mode();
    o.q = d.q();
    o.dq = d.dq();
    o.ddq = d.ddq();
    o.tau_est = d.tau_est();
    o.temperature[0] = d.temperature()[0];
    o.temperature[1] = d.temperature()[1];
    o.vol = d.vol();
    o.sensor[0] = d.sensor()[0];
    o.sensor[1] = d.sensor()[1];
    o.motorstate = d.motorstate();
    for (std::size_t i = 0; i < 4; ++i) o.reserve[i] = d.reserve()[i];
  }
  for (std::size_t i = 0; i < static_cast<std::size_t>(hg::kRemoteBytes); ++i)
    s.wireless_remote[i] = m.wireless_remote()[i];
  for (std::size_t i = 0; i < 4; ++i) s.reserve[i] = m.reserve()[i];
  s.crc = m.crc();
  return s;
}

hg::LowCmd image_of(const uhg::LowCmd_& m) noexcept {
  hg::LowCmd c;
  hg::clear(c);
  c.mode_pr = m.mode_pr();
  c.mode_machine = m.mode_machine();
  for (std::size_t i = 0; i < static_cast<std::size_t>(hg::kNumMotorSlots); ++i) {
    const auto& d = m.motor_cmd()[i];
    c.motor_cmd[i] = {d.mode(), d.q(), d.dq(), d.tau(), d.kp(), d.kd(), d.reserve()};
  }
  for (std::size_t i = 0; i < 4; ++i) c.reserve[i] = m.reserve()[i];
  c.crc = m.crc();
  return c;
}
}  // namespace

std::uint32_t lowstate_crc(const uhg::LowState_& m) noexcept { return hg_crc(image_of(m)); }

void to_lowstate_in(const uhg::LowState_& m, std::int64_t t_rx_ns, r1rt_lowstate_in* out) noexcept {
  std::memset(out, 0, sizeof(*out));
  out->t_rx_ns = t_rx_ns;
  out->board_tick = m.tick();
  out->mode_pr = m.mode_pr();
  out->mode_machine = m.mode_machine();
  out->crc_ok = (hg_crc(image_of(m)) == m.crc()) ? 1 : 0;
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    const auto& d = m.motor_state()[kSlot[j]];
    out->q[j] = d.q();
    out->dq[j] = d.dq();
    out->ddq[j] = d.ddq();
    out->tau_est[j] = d.tau_est();
    out->temperature[j][0] = d.temperature()[0];
    out->temperature[j][1] = d.temperature()[1];
    out->voltage[j] = d.vol();
    out->motor_state[j] = d.motorstate();
  }
  const auto& imu = m.imu_state();
  for (std::size_t i = 0; i < 4; ++i) out->imu_quat[i] = imu.quaternion()[i];
  for (std::size_t i = 0; i < 3; ++i) {
    out->imu_gyro[i] = imu.gyroscope()[i];
    out->imu_acc[i] = imu.accelerometer()[i];
    out->imu_rpy[i] = imu.rpy()[i];
  }
  for (std::size_t i = 0; i < R1RT_REMOTE_BYTES; ++i) out->wireless_remote[i] = m.wireless_remote()[i];
}

bool to_lowcmd_out(const uhg::LowCmd_& m, r1rt_lowcmd_out* out) noexcept {
  std::memset(out, 0, sizeof(*out));
  out->publish = 1;
  out->mode_pr = m.mode_pr();
  out->mode_machine = m.mode_machine();
  for (int j = 0; j < R1RT_NUM_JOINTS; ++j) {
    const auto& d = m.motor_cmd()[kSlot[j]];
    out->enable[j] = d.mode();
    out->q[j] = d.q();
    out->dq[j] = d.dq();
    out->kp[j] = d.kp();
    out->kd[j] = d.kd();
    out->tau_ff[j] = d.tau();
  }
  return hg_crc(image_of(m)) == m.crc();
}

std::uint32_t to_dds(const r1rt_lowcmd_out& cmd, uhg::LowCmd_* out) noexcept {
  hg::LowCmd img;
  const std::uint32_t crc = r1rt_fill_hg_lowcmd(&cmd, &img, sizeof img);
  out->mode_pr(img.mode_pr);
  out->mode_machine(img.mode_machine);
  for (std::size_t i = 0; i < static_cast<std::size_t>(hg::kNumMotorSlots); ++i) {
    auto& c = out->motor_cmd()[i];
    const hg::MotorCmd& s = img.motor_cmd[i];
    c.mode(s.mode);
    c.q(s.q);
    c.dq(s.dq);
    c.tau(s.tau);
    c.kp(s.kp);
    c.kd(s.kd);
    c.reserve(s.reserve);
  }
  for (std::size_t i = 0; i < 4; ++i) out->reserve()[i] = img.reserve[i];
  out->crc(img.crc);
  return crc;
}

// ---------------------------------------------------------------------------------------------------------------------

struct Bus::Impl {
  dds::domain::DomainParticipant dp;
  dds::topic::Topic<uhg::LowState_> t_state;
  dds::topic::Topic<uhg::LowCmd_> t_cmd;
  dds::sub::Subscriber sub;
  dds::pub::Publisher pub;
  dds::sub::DataReader<uhg::LowState_> r_state;
  dds::sub::DataReader<uhg::LowCmd_> r_cmd;
  dds::pub::DataWriter<uhg::LowCmd_> w_cmd;
  uhg::LowCmd_ out;  // the writer's message (reused; written by the writer thread only)

  explicit Impl(int domain)
      : dp(static_cast<uint32_t>(domain)),
        t_state(dp, kTopicLowState),
        t_cmd(dp, kTopicLowCmd),
        sub(dp),
        pub(dp),
        r_state(dds::core::null),
        r_cmd(dds::core::null),
        w_cmd(dds::core::null) {}
};

namespace {
dds::sub::qos::DataReaderQos reader_qos(const dds::sub::Subscriber& sub) {
  // The latest sample is all a control loop wants: best effort, keep last 1 (matches a reliable or best-effort writer).
  dds::sub::qos::DataReaderQos q = sub.default_datareader_qos();
  q << dds::core::policy::Reliability::BestEffort() << dds::core::policy::History::KeepLast(1);
  return q;
}
}  // namespace

Bus::Bus(const BusOptions& opt) noexcept : opt_(opt) {
  try {
    impl_ = std::make_unique<Impl>(opt.domain);
    impl_->r_state = dds::sub::DataReader<uhg::LowState_>(impl_->sub, impl_->t_state, reader_qos(impl_->sub));
    if (opt.read_builtin)
      impl_->r_cmd = dds::sub::DataReader<uhg::LowCmd_>(impl_->sub, impl_->t_cmd, reader_qos(impl_->sub));
    if (opt.write_lowcmd) {
      dds::pub::qos::DataWriterQos wq = impl_->pub.default_datawriter_qos();
      const std::string tag = std::string(kUserDataPrefix) + (opt.role ? opt.role : "?");
      wq << dds::core::policy::UserData(dds::core::ByteSeq(tag.begin(), tag.end()))
         << dds::core::policy::History::KeepLast(1);
      impl_->w_cmd = dds::pub::DataWriter<uhg::LowCmd_>(impl_->pub, impl_->t_cmd, wq);
      writer_handle_ = impl_->w_cmd.instance_handle()->handle();
    }
    ok_ = true;
  } catch (const std::exception& e) {
    err_ = e.what();
    ok_ = false;
  }
}

Bus::~Bus() = default;

bool Bus::publication_is_ours(std::uint64_t h) noexcept {
  for (int i = 0; i < n_pubs_; ++i)
    if (pub_handle_[i] == h) return pub_ours_[i] != 0;
  bool ours = false;
  if (impl_ && !impl_->r_cmd.is_nil()) {
    const dds_entity_t rd = impl_->r_cmd->get_ddsc_entity();
    dds_builtintopic_endpoint_t* ep = dds_get_matched_publication_data(rd, h);
    if (ep == nullptr) {  // not (yet) known as matched: foreign for this sample, decided again on the next one
      cnt_.lookup_failures.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
    {
      void* ud = nullptr;
      size_t sz = 0;
      if (ep->qos != nullptr && dds_qget_userdata(ep->qos, &ud, &sz) && ud != nullptr) {
        const std::size_t n = std::strlen(kUserDataPrefix);
        ours = sz >= n && std::memcmp(ud, kUserDataPrefix, n) == 0;
        dds_free(ud);
      }
      dds_builtintopic_free_endpoint(ep);
    }
  }
  if (!ours) cnt_.foreign_writers.fetch_add(1, std::memory_order_relaxed);
  if (n_pubs_ < kMaxPubs) {
    pub_handle_[n_pubs_] = h;
    pub_ours_[n_pubs_] = ours ? 1 : 0;
    ++n_pubs_;
  }
  return ours;
}

void Bus::run_receive(const std::atomic<bool>& stop) noexcept {
  if (!ok_) return;
  try {
    dds::core::cond::WaitSet ws;
    dds::sub::cond::ReadCondition rc_state(impl_->r_state, dds::sub::status::DataState::any());
    ws += rc_state;
    dds::sub::cond::ReadCondition rc_cmd(dds::core::null);
    if (!impl_->r_cmd.is_nil()) {
      rc_cmd = dds::sub::cond::ReadCondition(impl_->r_cmd, dds::sub::status::DataState::any());
      ws += rc_cmd;
    }
    const auto timeout = dds::core::Duration::from_millisecs(50);
    while (!stop.load(std::memory_order_acquire)) {
      try {
        ws.wait(timeout);
      } catch (const dds::core::TimeoutError&) {
        continue;
      }
      timespec ts{};
      clock_gettime(CLOCK_MONOTONIC, &ts);
      const std::int64_t t_rx = static_cast<std::int64_t>(ts.tv_sec) * 1'000'000'000 + ts.tv_nsec;
      {
        auto samples = impl_->r_state.take();
        const uhg::LowState_* last = nullptr;
        for (const auto& s : samples)
          if (s.info().valid()) last = &s.data();
        if (last != nullptr) {
          r1rt_lowstate_in in;
          to_lowstate_in(*last, t_rx, &in);
          cnt_.lowstate_rx.fetch_add(1, std::memory_order_relaxed);
          if (!in.crc_ok) cnt_.lowstate_crc_bad.fetch_add(1, std::memory_order_relaxed);
          state_.write(in);
          state_count_.fetch_add(1, std::memory_order_release);
        }
      }
      if (!impl_->r_cmd.is_nil()) {
        auto samples = impl_->r_cmd.take();
        for (const auto& s : samples) {
          if (!s.info().valid()) continue;
          const std::uint64_t h = s.info().publication_handle()->handle();
          if (h == writer_handle_ || publication_is_ours(h)) {
            cnt_.own_lowcmd_rx.fetch_add(1, std::memory_order_relaxed);
            continue;
          }
          r1rt_lowcmd_out cmd;
          if (!to_lowcmd_out(s.data(), &cmd)) {
            cnt_.builtin_crc_bad.fetch_add(1, std::memory_order_relaxed);
            continue;
          }
          cnt_.builtin_rx.fetch_add(1, std::memory_order_relaxed);
          if (opt_.on_builtin != nullptr) opt_.on_builtin(opt_.builtin_ctx, cmd, t_rx);
        }
      }
    }
  } catch (const std::exception& e) {
    err_ = std::string("receive loop: ") + e.what();
  }
}

bool Bus::latest(r1rt_lowstate_in* out, std::uint64_t* seen) const noexcept {
  const std::uint64_t n = state_count_.load(std::memory_order_acquire);
  if (n == *seen) return false;
  if (!state_.try_read(*out)) return false;
  *seen = n;
  return true;
}

bool Bus::write(const r1rt_lowcmd_out& cmd) noexcept {
  if (!ok_ || impl_->w_cmd.is_nil()) return false;
  to_dds(cmd, &impl_->out);
  try {
    impl_->w_cmd.write(impl_->out);
    cnt_.writes.fetch_add(1, std::memory_order_relaxed);
    return true;
  } catch (const std::exception&) {
    cnt_.write_errors.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
}

}  // namespace r1rt::dds_io
