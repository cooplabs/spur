// r1rt/crc32.hpp - Unitree's CRC32 for the hg LowCmd_ / LowState_ messages (C++17).
//
// The algorithm (unitree_sdk2 @63096d0 include/unitree/dds_wrapper/common/crc.h crc32_core; the same as
// unitree_sdk2_python utils/crc.py _crc_py): init 0xFFFFFFFF, polynomial 0x04C11DB7, no reflection, no final XOR,
// over 32-bit words, each word MSB first. That equals CRC-32/MPEG-2 over the big-endian bytes of every word, which
// is what the table-driven crc32_words() computes (tested equal to the bit-wise port on every golden message).
// Both are allocation-free and noexcept; crc32_words() is the one for the 500 Hz writer.
#pragma once

#include <cstddef>
#include <cstdint>

#include "r1rt/hg_msgs.hpp"

namespace r1rt {

// Bit-wise reference: a line-by-line port of Unitree's crc32_core (attributed above).
std::uint32_t crc32_core_ref(const std::uint32_t* words, std::size_t n) noexcept;

// Table-driven (one 256-entry table, byte-wise MSB first over each word's big-endian bytes).
std::uint32_t crc32_words(const std::uint32_t* words, std::size_t n) noexcept;

// CRC-32/MPEG-2 over plain bytes (the standard catalogue algorithm; used to anchor the word CRC to a published
// check value: "123456789" -> 0x0376E6E7).
std::uint32_t crc32_mpeg2_bytes(const std::uint8_t* data, std::size_t n) noexcept;

// The message CRC exactly as the robot checks it: over the canonical CRC image (hg::crc_image), all words but the
// last. Set `m.crc = hg_crc(m)` before writing a LowCmd_; compare for a received LowState_.
std::uint32_t hg_crc(const hg::LowCmd& m) noexcept;
std::uint32_t hg_crc(const hg::LowState& m) noexcept;

}  // namespace r1rt
