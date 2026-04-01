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
inline void graph_edge_matching_kernel(
    const gbts_graph_building_params d_graph_building_params,
    const collection_types<half4>::const_view d_edge_params_view,
    const collection_types<int2>::const_view d_edge_nodes_view,
    const collection_types<int>::const_view d_num_outgoing_edges_view,
    const collection_types<int>::const_view d_edge_links_view,
    const collection_types<unsigned char>::view d_num_neighbours_view,
    const collection_types<int>::view d_neighbours_view,
    const collection_types<int>::view d_reIndexer_view,
    const collection_types<unsigned int>::view d_counters_view,
    const unsigned int nEdges, const unsigned int nMaxNei);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/gbts_seeding/device/impl/graph_edge_matching.ipp"
