/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"

// System include(s).
#include <bit>
#include <cstdint>

namespace traccc::device::details {

/// FNV-1a 64-bit offset basis (seed value for a @c hash_u32 chain).
inline constexpr std::uint64_t fnv_offset_basis = 14695981039346656037ull;
/// FNV-1a 64-bit prime.
inline constexpr std::uint64_t fnv_prime = 1099511628211ull;

/// @brief Map a float to an unsigned int whose ascending integer order matches
/// the float's IEEE-754 ascending order.
///
/// This lets a float be packed into an integer radix / comparison sort key.
/// Order-preserving "Herf flip": flip every bit of a negative float, and only
/// the sign bit of a non-negative float. Uses @c std::bit_cast (C++20,
/// well-defined) rather than a type-punning union.
///
/// @param[in] f The float to encode
/// @return an order-preserving unsigned representation of @p f
///
TRACCC_HOST_DEVICE inline unsigned int orderable_float(float f) {
    const std::uint32_t b = std::bit_cast<std::uint32_t>(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

/// @brief FNV-1a mix of a 32-bit value into a running 64-bit hash.
///
/// Used to build a deterministic per-object key from a sequence of indices
/// (e.g. the spacepoint indices along a seed-candidate path). Seed a chain
/// from @c fnv_offset_basis.
///
/// @param[in] h The running hash (seed with @c fnv_offset_basis)
/// @param[in] v The value folded into the hash
/// @return the updated hash
///
TRACCC_HOST_DEVICE inline std::uint64_t hash_u32(std::uint64_t h,
                                                 std::uint32_t v) {
    return (h ^ static_cast<std::uint64_t>(v)) * fnv_prime;
}

}  // namespace traccc::device::details
