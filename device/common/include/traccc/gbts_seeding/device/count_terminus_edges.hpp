/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/concepts/barrier.hpp"
#include "traccc/edm/container.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

namespace traccc::device {

template <concepts::barrier barrier_t>
TRACCC_HOST_DEVICE inline void count_terminus_edges(
    unsigned int blockIndex, unsigned int threadIndex, unsigned int blockSize,
    const barrier_t& barrier, int& outgoingCount,
    collection_types<short2>::view d_outgoing_paths_view,
    unsigned int& nPathsCounter, unsigned int& nPathStoreSizeCounter,
    unsigned int nEdges);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/count_terminus_edges.ipp"
