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
inline void graph_edge_linking_kernel(
    const global_index_t global_index,
    const collection_types<int2>::const_view d_edge_nodes_view,
    const collection_types<int>::view d_edge_links_view,
    const collection_types<int>::view d_num_outgoing_edges_view,
    const unsigned int nEdges) {


if (global_index >= nEdges) {
    return;
}

const collection_types<int2>::const_device d_edge_nodes(d_edge_nodes_view);
collection_types<int>::device d_edge_links(d_edge_links_view);
collection_types<int>::device d_num_outgoing_edges(d_num_outgoing_edges_view);


int sharedNode = d_edge_nodes[global_index].y;

// this converts num_outgoing_edges to the start postion for each node in
// d_edge_links
vecmem::device_atomic_ref<int> d_num_outgoing_edges_ref(d_num_outgoing_edges[static_cast<unsigned int>(sharedNode)]);
int pos = d_num_outgoing_edges_ref.fetch_add(-1);
// provides views of edges leaving the sharedNode for linking
    d_edge_links[static_cast<unsigned int>(pos - 1)] = static_cast<int>(global_index);
}


} // namespace traccc::device