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
#include "traccc/gbts_seeding/gbts_types.hpp"
#include "traccc/edm/container.hpp"


namespace traccc::device {

TRACCC_HOST_DEVICE
inline void graph_compression_kernel(
    const int idx,
    const collection_types<int>::const_view d_orig_node_index_view, const collection_types<int2>::const_view d_edge_nodes_view,
    const collection_types<unsigned char>::const_view d_num_neighbours_view, const collection_types<int>::const_view d_neighbours_view,
    const collection_types<int>::const_view d_reIndexer_view, const collection_types<int>::view d_output_graph_view,
    const unsigned int nMaxNei,
    const unsigned int nEdges) {

    if (idx >= nEdges) {
        return;
    }

    const collection_types<int>::const_device d_reIndexer(d_reIndexer_view);

    const int edge_size = static_cast<int>(2 + 1 + nMaxNei);

    int newIdx = d_reIndexer[static_cast<unsigned int>(idx)];
    if (newIdx == -1) {
        return;
    }


    collection_types<int>::device d_output_graph(d_output_graph_view);
    const collection_types<int>::const_device d_orig_node_index(d_orig_node_index_view);
    const collection_types<int2>::const_device d_edge_nodes(d_edge_nodes_view);
    const collection_types<unsigned char>::const_device d_num_neighbours(d_num_neighbours_view);
    const collection_types<int>::const_device d_neighbours(d_neighbours_view);


    int pos = edge_size * newIdx;
    int2 edge_nodes = d_edge_nodes[static_cast<unsigned int>(idx)];
    int node1_idx = d_orig_node_index[static_cast<unsigned int>(edge_nodes.x)];
    d_output_graph[static_cast<unsigned int>(pos + traccc::device::gbts_consts::node1)] = node1_idx;
    int node2_idx = d_orig_node_index[static_cast<unsigned int>(edge_nodes.y)];
    d_output_graph[static_cast<unsigned int>(pos + traccc::device::gbts_consts::node2)] = node2_idx;

    unsigned char nNei = d_num_neighbours[static_cast<unsigned int>(idx)];
    d_output_graph[static_cast<unsigned int>(pos + traccc::device::gbts_consts::nNei)] = nNei;
    int nei_pos = static_cast<int>(nMaxNei) * idx;
    for (int k = 0; k < nNei; k++) {
        d_output_graph[static_cast<unsigned int>(pos + traccc::device::gbts_consts::nei_start + k)] =
            d_reIndexer[static_cast<unsigned int>(d_neighbours[static_cast<unsigned int>(nei_pos + k)])];
    }

}

} // namespace traccc::device