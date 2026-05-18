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
inline void graph_compression(
    global_index_t globalIndex,
    const collection_types<int>::const_view& d_orig_node_index_view,
    const collection_types<int2>::const_view& d_edge_nodes_view,
    const collection_types<unsigned char>::const_view& d_num_neighbours_view,
    const collection_types<int>::const_view& d_neighbours_view,
    const collection_types<int>::const_view& d_reIndexer_view,
    collection_types<int>::view d_output_graph_view, unsigned int nEdges,
    unsigned int nMaxNei);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/graph_compression.ipp"
