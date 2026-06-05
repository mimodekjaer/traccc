/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/utils/trigonometric_helpers.hpp"

// System include(s).
#include <cmath>

// CUDA HIP
#if defined(__CUDACC__) || defined(__HIP__)
#define TRACCC_GBTS_USE_BUILTIN_VECTORS 1
#include <vector_functions.h>
#include <vector_types.h>
#else
// SYCL / Alpaka / host fall through to the portable struct.
#define TRACCC_GBTS_USE_BUILTIN_VECTORS 0
#endif


#if defined(TRACCC_GBTS_EDGE_HALF) && defined(__CUDACC__)
#include <cuda_fp16.h>
#endif

namespace traccc {

#if TRACCC_GBTS_USE_BUILTIN_VECTORS

// Built-in vector types and constructors (CUDA / HIP).
using ::float2;
using ::float4;
using ::int2;
using ::short2;
using ::uint2;
using ::uint4;

using ::make_float2;
using ::make_float4;
using ::make_int2;
using ::make_short2;
using ::make_uint2;
using ::make_uint4;

#else

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

#endif  // TRACCC_GBTS_USE_BUILTIN_VECTORS

#if defined(TRACCC_GBTS_EDGE_HALF) && defined(__CUDACC__)

using gbts_edge_t = __half;

struct TRACCC_ALIGN(8) gbts_edge4 {
    gbts_edge_t x, y, z, w;
};

inline TRACCC_HOST_DEVICE gbts_edge_t gbts_edge_from_float(const float f) {
    return __float2half(f);
}
inline TRACCC_HOST_DEVICE float gbts_edge_to_float(const gbts_edge_t r) {
    return __half2float(r);
}

#else

using gbts_edge_t = float;
using gbts_edge4 = float4;

inline TRACCC_HOST_DEVICE gbts_edge_t gbts_edge_from_float(const float f) {
    return f;
}
inline TRACCC_HOST_DEVICE float gbts_edge_to_float(const gbts_edge_t r) {
    return r;
}

#endif

inline TRACCC_HOST_DEVICE gbts_edge4 gbts_make_edge4(const float x,
                                                     const float y,
                                                     const float z,
                                                     const float w) {
    return gbts_edge4{gbts_edge_from_float(x), gbts_edge_from_float(y),
                      gbts_edge_from_float(z), gbts_edge_from_float(w)};
}

namespace device {

inline constexpr float PI_F = traccc::constant<float>::pi;
inline constexpr float TWO_PI_F = 2.0f * traccc::constant<float>::pi;

/// Wrap an angle into (-pi, pi], matching the reference (round-to-nearest).
TRACCC_HOST_DEVICE inline float phi_wrap(const float phi) {
    return phi - TWO_PI_F * rintf(phi * (1.0f / TWO_PI_F));
}

#if defined(TRACCC_GBTS_EDGE_HALF) && defined(__CUDACC__)
/// Half-precision phi wrap (device-only); kept so half edge math still works.
__device__ inline __half phi_wrap(const __half phi) {
    const __half two_pi_h = __float2half(2.0f * traccc::constant<float>::pi);
    const __half one_h = __float2half(1.0f);
    return phi - two_pi_h * hrint(phi * (one_h / two_pi_h));
}
#endif

}  // namespace device

}  // namespace traccc
