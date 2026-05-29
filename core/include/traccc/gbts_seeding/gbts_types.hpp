/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/definitions/qualifiers.hpp"

namespace traccc {

struct TRACCC_ALIGN(8) uint2 {
    unsigned int x, y;
};

struct TRACCC_ALIGN(8) int2 {
    int x, y;
};

struct TRACCC_ALIGN(16) float4 {
    float x, y, z, w;
};

struct TRACCC_ALIGN(8) float2 {
    float x, y;
};

struct TRACCC_ALIGN(16) uint4 {
    unsigned int x, y, z, w;
};

struct TRACCC_ALIGN(4) short2 {
    short x, y;
};

inline TRACCC_HOST_DEVICE traccc::int2 make_int2(const int x, const int y) {
    return traccc::int2{x, y};
}

inline TRACCC_HOST_DEVICE traccc::float4 make_float4(const float x,
                                                     const float y,
                                                     const float z,
                                                     const float w) {
    return traccc::float4{x, y, z, w};
}

inline TRACCC_HOST_DEVICE traccc::float2 make_float2(const float x,
                                                     const float y) {
    return traccc::float2{x, y};
}

inline TRACCC_HOST_DEVICE traccc::uint2 make_uint2(const unsigned int x,
                                                   const unsigned int y) {
    return traccc::uint2{x, y};
}

inline TRACCC_HOST_DEVICE traccc::uint4 make_uint4(const unsigned int x,
                                                   const unsigned int y,
                                                   const unsigned int z,
                                                   const unsigned int w) {
    return traccc::uint4{x, y, z, w};
}

inline TRACCC_HOST_DEVICE traccc::short2 make_short2(const short x,
                                                     const short y) {
    return traccc::short2{x, y};
}

namespace device {

inline constexpr float gbts_pi_f = 3.1415927410125732422f;
inline constexpr float gbts_two_pi_f = 2.0f * gbts_pi_f;

}  // namespace device

}  // namespace traccc
