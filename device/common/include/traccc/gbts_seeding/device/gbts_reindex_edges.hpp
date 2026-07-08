/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/concepts/thread_id.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device {

/// (Global Event Data) Payload for the @c traccc::device::gbts_reindex_edges
/// function
struct gbts_reindex_edges_payload {
    /// Number of original edges
    unsigned int nEdges;
    /// In/out: 0/1 per-edge "kept" flags from gbts_match_graph_edges in,
    /// inclusive kept-count out (in place): inclusive_kept[e] = number of
    /// kept edges in [0, e], so a kept edge's compact index is
    /// inclusive_kept[e] - 1. Consumed directly by gbts_compress_graph.
    vecmem::data::vector_view<unsigned int> inclusive_kept;
    /// In/out: global atomic counter of edges that survived re-indexing
    unsigned int* nConnectedEdgesCounter;
};

/// @brief Turn the 0/1 kept flags into the inclusive kept-count, in place.
///
/// The CUDA backend implements this as a deterministic in-place
/// thrust::inclusive_scan. The fallback below only claims compact indices
/// atomically and does NOT produce the monotone counts that
/// gbts_compress_graph's kept-test requires -- it must be replaced by a scan
/// when the SYCL/alpaka backends are ported.
///
/// @param[in] thread_id Thread identifier for the kernel launch
/// @param[in] payload   The global memory payload
///
template <concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void gbts_reindex_edges(
    const thread_id_t& thread_id, const gbts_reindex_edges_payload& payload);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/gbts_reindex_edges.ipp"
