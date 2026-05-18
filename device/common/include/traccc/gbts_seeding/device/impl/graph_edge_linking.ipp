/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/global_index.hpp"
#include "traccc/edm/container.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/memory/device_atomic_ref.hpp>

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void graph_edge_linking(
    const global_index_t globalIndex,
    const collection_types<int2>::const_view& d_edge_nodes_view,
    collection_types<int>::view d_edge_links_view,
    collection_types<int>::view d_num_outgoing_edges_view,
    const unsigned int nEdges) {

    if (globalIndex >= nEdges) {
        return;
    }

    const collection_types<int2>::const_device d_edge_nodes(d_edge_nodes_view);
    collection_types<int>::device d_edge_links(d_edge_links_view);
    collection_types<int>::device d_num_outgoing_edges(
        d_num_outgoing_edges_view);

    const int sharedNode = d_edge_nodes[globalIndex].y;
    const int pos =
        vecmem::device_atomic_ref<int>(
            d_num_outgoing_edges[static_cast<unsigned int>(sharedNode)])
            .fetch_add(-1);
    d_edge_links[static_cast<unsigned int>(pos - 1)] =
        static_cast<int>(globalIndex);
}

}  // namespace traccc::device
