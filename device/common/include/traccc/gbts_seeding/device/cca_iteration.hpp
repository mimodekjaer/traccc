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
TRACCC_HOST_DEVICE inline void cca_iteration(
    unsigned int blockIndex, unsigned int threadIndex, unsigned int blockSize,
    unsigned int gridSize, const barrier_t& barrier,
    const collection_types<int>::const_view& d_output_graph_view,
    collection_types<char>::view d_levels_view,
    collection_types<int>::view d_active_edges_view,
    collection_types<short2>::view d_outgoing_paths_view,
    unsigned int& nActiveEdgesA, unsigned int& nActiveEdgesB,
    unsigned int& nCCABlockCounter, int iter, unsigned int nEdges,
    unsigned int max_num_neighbours, int minLevel);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/cca_iteration.ipp"
