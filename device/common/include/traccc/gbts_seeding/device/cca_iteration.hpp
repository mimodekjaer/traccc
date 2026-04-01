/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Local include(s).
#include "traccc/device/global_index.hpp"

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/container.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void cca_iteration_kernel(
    const global_index_t global_index,
    const collection_types<int>::const_view d_output_graph_view,
    const collection_types<char>::view d_levels_view,
    const collection_types<int>::view d_active_edges_view,
    const collection_types<short2>::view d_outgoing_paths_view,
    const collection_types<unsigned int>::view d_counters_view,
    const int iter, const unsigned int nEdges, const int toggle,
    const unsigned int max_num_neighbours, const int minLevel);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/gbts_seeding/device/impl/cca_iteration.ipp"
