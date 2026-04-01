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
inline void graph_edge_making_kernel(
    const collection_types<int>::const_view d_bin_pair_views_view,
    const collection_types<float>::const_view d_bin_pair_dphi_view,
    const collection_types<float>::const_view d_node_params_view,
    const gbts_graph_building_params d_graph_building_params,
    const collection_types<unsigned int>::view d_counters_view,
    const collection_types<int2>::view d_edge_nodes_view,
    const collection_types<half4>::view d_edge_params_view,
    const collection_types<int>::view d_num_outgoing_edges_view,
    const unsigned int nMaxEdges, const unsigned int nPhiBins);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/gbts_seeding/device/impl/graph_edge_making.ipp"
