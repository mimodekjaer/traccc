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
#include "traccc/gbts_seeding/gbts_seeding_config.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

namespace traccc::device {

template <concepts::barrier barrier_t>
TRACCC_HOST_DEVICE inline void graph_edge_making(
    unsigned int blockIndex, unsigned int threadIndex, unsigned int blockSize,
    const barrier_t& barrier, float* tau_min, float* tau_max, float* phi,
    float* r, float* z,
    const collection_types<int>::const_view& d_bin_pair_views_view,
    const collection_types<float>::const_view& d_bin_pair_dphi_view,
    const collection_types<float>::const_view& d_node_params_view,
    const gbts_graph_building_params& d_graph_building_params,
    unsigned int& nEdgesCounter,
    collection_types<int2>::view d_edge_nodes_view,
    collection_types<half4>::view d_edge_params_view,
    collection_types<int>::view d_num_outgoing_edges_view,
    unsigned int nMaxEdges, unsigned int nPhiBins);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/graph_edge_making.ipp"
