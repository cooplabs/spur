// r1rt::PolicyMlp (see the header). The weights file format is documented in tools/policy_convert.py.
#include "r1rt/policy_mlp.hpp"

#include <Eigen/Core>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <new>

namespace r1rt {
namespace {

constexpr std::size_t kHeaderBytes = 80;
constexpr std::uint32_t kActElu = 1;
constexpr std::size_t kAlignDoubles = 8;  // every array starts on a 64-byte boundary inside the block

// Array sizes in file order: sub, div, W0, b0, W1, b1, W2, b2, W3, b3.
constexpr std::size_t kCounts[10] = {
    PolicyMlp::kObs,
    PolicyMlp::kObs,
    std::size_t{PolicyMlp::kH0} * PolicyMlp::kObs,
    PolicyMlp::kH0,
    std::size_t{PolicyMlp::kH1} * PolicyMlp::kH0,
    PolicyMlp::kH1,
    std::size_t{PolicyMlp::kH2} * PolicyMlp::kH1,
    PolicyMlp::kH2,
    std::size_t{PolicyMlp::kAct} * PolicyMlp::kH2,
    PolicyMlp::kAct,
};

constexpr std::size_t padded(std::size_t n) { return (n + kAlignDoubles - 1) / kAlignDoubles * kAlignDoubles; }

constexpr std::size_t file_doubles() {
  std::size_t n = 0;
  for (std::size_t c : kCounts) n += c;
  return n;
}

constexpr std::size_t block_doubles() {
  std::size_t n = 0;
  for (std::size_t c : kCounts) n += padded(c);
  return n;
}

void set_err(char* err, std::size_t errlen, const char* fmt, ...) {
  if (err == nullptr || errlen == 0) return;
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(err, errlen, fmt, ap);
  va_end(ap);
}

std::uint32_t rd_u32(const unsigned char* p) {
  return std::uint32_t{p[0]} | (std::uint32_t{p[1]} << 8) | (std::uint32_t{p[2]} << 16) | (std::uint32_t{p[3]} << 24);
}

bool host_little_endian() {
  const std::uint32_t one = 1;
  unsigned char b = 0;
  std::memcpy(&b, &one, 1);
  return b == 1;
}

struct FileCloser {
  std::FILE* f;
  ~FileCloser() {
    if (f != nullptr) std::fclose(f);
  }
};

// ELU exactly as v2's NumPy executor: where(x > 0, x, alpha * (exp(min(x, 0)) - 1)).
template <int N>
void elu_inplace(Eigen::Matrix<double, N, 1>& h, double alpha) noexcept {
  double* v = h.data();
  for (int i = 0; i < N; ++i) {
    const double x = v[i];
    v[i] = x > 0.0 ? x : alpha * (std::exp(x < 0.0 ? x : 0.0) - 1.0);
  }
}

}  // namespace

PolicyMlp::~PolicyMlp() { release(); }

void PolicyMlp::release() noexcept {
  if (block_ != nullptr) {
    ::operator delete(block_, std::align_val_t{64});
    block_ = nullptr;
  }
  sub_ = div_ = w0_ = b0_ = w1_ = b1_ = w2_ = b2_ = w3_ = b3_ = nullptr;
}

bool PolicyMlp::load(const char* path, char* err, std::size_t errlen) noexcept {
  if (!host_little_endian()) {
    set_err(err, errlen, "big-endian hosts are not supported (the R1W1 file is little-endian)");
    return false;
  }
  std::FILE* f = std::fopen(path, "rb");
  if (f == nullptr) {
    set_err(err, errlen, "cannot open %s", path);
    return false;
  }
  FileCloser closer{f};
  unsigned char h[kHeaderBytes];
  if (std::fread(h, 1, kHeaderBytes, f) != kHeaderBytes) {
    set_err(err, errlen, "%s: truncated header", path);
    return false;
  }
  if (std::memcmp(h, "R1W1", 4) != 0) {
    set_err(err, errlen, "%s: not an R1W1 weights file (bad magic)", path);
    return false;
  }
  const std::uint32_t version = rd_u32(h + 4);
  if (version != kVersion) {
    set_err(err, errlen, "%s: version %u, this runner reads version %u", path, version, kVersion);
    return false;
  }
  const std::uint32_t want[5] = {kObs, kH0, kH1, kH2, kAct};
  for (int i = 0; i < 5; ++i) {
    const std::uint32_t d = rd_u32(h + 8 + 4 * i);
    if (d != want[i]) {
      set_err(err, errlen, "%s: dims[%d] = %u, this runner is built for %u (74-512-256-128-14)", path, i, d, want[i]);
      return false;
    }
  }
  if (rd_u32(h + 28) != kActElu) {
    set_err(err, errlen, "%s: activation %u is not ELU", path, rd_u32(h + 28));
    return false;
  }
  double alpha = 0.0;
  std::memcpy(&alpha, h + 32, sizeof alpha);
  if (!std::isfinite(alpha)) {
    set_err(err, errlen, "%s: ELU alpha is not finite", path);
    return false;
  }
  // Exact size: header + every array, nothing after.
  if (std::fseek(f, 0, SEEK_END) != 0) {
    set_err(err, errlen, "%s: cannot seek", path);
    return false;
  }
  const long size = std::ftell(f);
  const std::size_t want_size = kHeaderBytes + file_doubles() * sizeof(double);
  if (size < 0 || static_cast<std::size_t>(size) != want_size) {
    set_err(err, errlen, "%s: %ld bytes, expected %zu", path, size, want_size);
    return false;
  }
  if (std::fseek(f, static_cast<long>(kHeaderBytes), SEEK_SET) != 0) {
    set_err(err, errlen, "%s: cannot seek", path);
    return false;
  }

  void* block = ::operator new(block_doubles() * sizeof(double), std::align_val_t{64}, std::nothrow);
  if (block == nullptr) {
    set_err(err, errlen, "out of memory");
    return false;
  }
  double* base = static_cast<double*>(block);
  const double* ptr[10];
  std::size_t off = 0;
  for (int i = 0; i < 10; ++i) {
    double* dst = base + off;
    if (std::fread(dst, sizeof(double), kCounts[i], f) != kCounts[i]) {
      ::operator delete(block, std::align_val_t{64});
      set_err(err, errlen, "%s: short read in array %d", path, i);
      return false;
    }
    for (std::size_t k = 0; k < kCounts[i]; ++k) {
      if (!std::isfinite(dst[k])) {
        ::operator delete(block, std::align_val_t{64});
        set_err(err, errlen, "%s: array %d element %zu is not finite", path, i, k);
        return false;
      }
    }
    for (std::size_t k = kCounts[i]; k < padded(kCounts[i]); ++k) dst[k] = 0.0;
    ptr[i] = dst;
    off += padded(kCounts[i]);
  }
  for (int i = 0; i < kObs; ++i) {
    if (ptr[1][i] == 0.0) {
      ::operator delete(block, std::align_val_t{64});
      set_err(err, errlen, "%s: normaliser divisor %d is zero", path, i);
      return false;
    }
  }

  release();
  block_ = block;
  sub_ = ptr[0];
  div_ = ptr[1];
  w0_ = ptr[2];
  b0_ = ptr[3];
  w1_ = ptr[4];
  b1_ = ptr[5];
  w2_ = ptr[6];
  b2_ = ptr[7];
  w3_ = ptr[8];
  b3_ = ptr[9];
  alpha_ = alpha;
  std::memcpy(sha256_.data(), h + 40, sha256_.size());
  return true;
}

void PolicyMlp::run(const double* obs, double* act) const noexcept {
  using Eigen::Map;
  using Eigen::Matrix;
  using Eigen::RowMajor;
  constexpr int A = Eigen::Aligned64;
  using VecObs = Matrix<double, kObs, 1>;
  using Vec0 = Matrix<double, kH0, 1>;
  using Vec1 = Matrix<double, kH1, 1>;
  using Vec2 = Matrix<double, kH2, 1>;
  using VecAct = Matrix<double, kAct, 1>;

  const Map<const Matrix<double, kH0, kObs, RowMajor>, A> W0(w0_);
  const Map<const Matrix<double, kH1, kH0, RowMajor>, A> W1(w1_);
  const Map<const Matrix<double, kH2, kH1, RowMajor>, A> W2(w2_);
  const Map<const Matrix<double, kAct, kH2, RowMajor>, A> W3(w3_);

  VecObs x;
  for (int i = 0; i < kObs; ++i) x[i] = (obs[i] - sub_[i]) / div_[i];  // the graph's Sub then Div

  Vec0 h0;
  h0.noalias() = W0 * x;
  h0 += Map<const Vec0, A>(b0_);
  elu_inplace(h0, alpha_);

  Vec1 h1;
  h1.noalias() = W1 * h0;
  h1 += Map<const Vec1, A>(b1_);
  elu_inplace(h1, alpha_);

  Vec2 h2;
  h2.noalias() = W2 * h1;
  h2 += Map<const Vec2, A>(b2_);
  elu_inplace(h2, alpha_);

  VecAct y;
  y.noalias() = W3 * h2;
  y += Map<const VecAct, A>(b3_);
  for (int i = 0; i < kAct; ++i) act[i] = y[i];
}

}  // namespace r1rt
