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

// VecMem include(s).
#include <vecmem/memory/device_atomic_ref.hpp>

namespace traccc::device {

template <concepts::barrier barrier_t>
TRACCC_HOST_DEVICE inline void count_terminus_edges(
    const unsigned int blockIndex, const unsigned int threadIndex,
    const unsigned int blockSize, const barrier_t& barrier, int& outgoingCount,
    collection_types<short2>::view d_outgoing_paths_view,
    unsigned int& nPathsCounter, unsigned int& nPathStoreSizeCounter,
    const unsigned int nEdges) {

    collection_types<short2>::device d_outgoing_paths(d_outgoing_paths_view);

    if (threadIndex == 0) {
        outgoingCount = 0;
    }
    barrier.blockBarrier();

    const unsigned int edge_idx = threadIndex + blockIndex * blockSize;
    if (edge_idx < nEdges) {
        const short2 out_paths = d_outgoing_paths[edge_idx];
        if (out_paths.y != -1) {
            d_outgoing_paths[edge_idx].y = static_cast<short>(
                vecmem::device_atomic_ref<unsigned int>(nPathStoreSizeCounter)
                    .fetch_add(1u));
            vecmem::device_atomic_ref<int>(outgoingCount)
                .fetch_add(out_paths.x);
        }
    }
    barrier.blockBarrier();
    if (threadIndex == 0) {
        vecmem::device_atomic_ref<unsigned int>(nPathsCounter)
            .fetch_add(static_cast<unsigned int>(outgoingCount));
    }
}

}  // namespace traccc::device
