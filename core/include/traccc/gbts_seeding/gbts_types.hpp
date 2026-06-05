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

// covfie include(s).
#include <covfie/core/array.hpp>

// System include(s).
#include <cmath>
#include <cstddef>

// ---------------------------------------------------------------------------
// Half-precision edge representation
// ---------------------------------------------------------------------------
// fp16 edges are the default on every GPU backend that ships a native half
// type: CUDA, HIP and SYCL (which also covers alpaka's CUDA/HIP/SYCL
// accelerators). A pure host / CPU build, where no half type is available,
// resolves to float -- define TRACCC_GBTS_EDGE_FORCE_FLOAT to force that too.
#if !defined(TRACCC_GBTS_EDGE_FORCE_FLOAT)
#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP__)
#define TRACCC_GBTS_EDGE_HALF 1
#if defined(__HIPCC__) || defined(__HIP__)
#include <hip/hip_fp16.h>
#else
#include <cuda_fp16.h>
#endif
#elif defined(SYCL_LANGUAGE_VERSION) || defined(__SYCL_DEVICE_ONLY__)
#define TRACCC_GBTS_EDGE_HALF 1
#define TRACCC_GBTS_EDGE_HALF_SYCL 1
#include <sycl/sycl.hpp>
#endif
#endif

namespace traccc {

// ---------------------------------------------------------------------------
// Portable fixed-size vector types
// ---------------------------------------------------------------------------
// Backed by covfie::array and indexed with operator[]. The alignment is forced
// to the packed byte size so the GPU can issue wide (vectorised) loads, keeping
// the layout identical to the native CUDA float4 / int2 / ... types.
template <typename T, std::size_t N>
struct TRACCC_ALIGN(sizeof(T) * N) gbts_array
    : public covfie::array::array<T, N> {
    using base_type = covfie::array::array<T, N>;
    using base_type::base_type;  // inherit covfie's (variadic) constructors
};

using float2 = gbts_array<float, 2>;
using float4 = gbts_array<float, 4>;
using int2 = gbts_array<int, 2>;
using uint2 = gbts_array<unsigned int, 2>;
using uint4 = gbts_array<unsigned int, 4>;
using short2 = gbts_array<short, 2>;

inline TRACCC_HOST_DEVICE float2 make_float2(const float x, const float y) {
    return float2{x, y};
}

inline TRACCC_HOST_DEVICE float4 make_float4(const float x, const float y,
                                             const float z, const float w) {
    return float4{x, y, z, w};
}

inline TRACCC_HOST_DEVICE int2 make_int2(const int x, const int y) {
    return int2{x, y};
}

inline TRACCC_HOST_DEVICE uint2 make_uint2(const unsigned int x,
                                           const unsigned int y) {
    return uint2{x, y};
}

inline TRACCC_HOST_DEVICE uint4 make_uint4(const unsigned int x,
                                           const unsigned int y,
                                           const unsigned int z,
                                           const unsigned int w) {
    return uint4{x, y, z, w};
}

inline TRACCC_HOST_DEVICE short2 make_short2(const short x, const short y) {
    return short2{x, y};
}

// ---------------------------------------------------------------------------
// Edge type (fp16 where available, float on CPU)
// ---------------------------------------------------------------------------
#if defined(TRACCC_GBTS_EDGE_HALF)

#if defined(TRACCC_GBTS_EDGE_HALF_SYCL)
using gbts_edge_t = ::sycl::half;
#else
using gbts_edge_t = __half;
#endif

// 8-byte aligned half4 (matches the reference half4 layout).
using gbts_edge4 = gbts_array<gbts_edge_t, 4>;

inline TRACCC_HOST_DEVICE gbts_edge_t gbts_edge_from_float(const float f) {
#if defined(TRACCC_GBTS_EDGE_HALF_SYCL)
    return static_cast<::sycl::half>(f);
#else
    return __float2half(f);
#endif
}
inline TRACCC_HOST_DEVICE float gbts_edge_to_float(const gbts_edge_t r) {
#if defined(TRACCC_GBTS_EDGE_HALF_SYCL)
    return static_cast<float>(r);
#else
    return __half2float(r);
#endif
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

#if defined(TRACCC_GBTS_EDGE_HALF)
/// Half-precision phi wrap; kept so half edge math still works on every backend.
#if defined(TRACCC_GBTS_EDGE_HALF_SYCL)
inline ::sycl::half phi_wrap(const ::sycl::half phi) {
    const ::sycl::half two_pi_h = static_cast<::sycl::half>(TWO_PI_F);
    return phi - two_pi_h * ::sycl::rint(phi / two_pi_h);
}
#else
__device__ inline gbts_edge_t phi_wrap(const gbts_edge_t phi) {
    const __half two_pi_h = __float2half(TWO_PI_F);
    const __half one_h = __float2half(1.0f);
    return phi - two_pi_h * hrint(phi * (one_h / two_pi_h));
}
#endif
#endif

}  // namespace device

}  // namespace traccc
