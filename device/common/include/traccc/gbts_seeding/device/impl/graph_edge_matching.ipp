/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/container.hpp"
#include "traccc/gbts_seeding/gbts_seeding_config.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/memory/device_atomic_ref.hpp>

// System include(s).
#include <cmath>

namespace traccc::device {

TRACCC_HOST_DEVICE inline void graph_edge_matching(
    const unsigned int blockIndex, const unsigned int threadIndex,
    const unsigned int blockSize,
    const gbts_graph_building_params& d_graph_building_params,
    const collection_types<half4>::const_view& d_edge_params_view,
    const collection_types<int2>::const_view& d_edge_nodes_view,
    const collection_types<int>::const_view& d_num_outgoing_edges_view,
    const collection_types<int>::const_view& d_edge_links_view,
    collection_types<unsigned char>::view d_num_neighbours_view,
    collection_types<int>::view d_neighbours_view,
    collection_types<int>::view d_reIndexer_view,
    unsigned int& nConnectionsCounter,
    const unsigned int nEdges, const unsigned int nMaxNei) {

    const collection_types<half4>::const_device d_edge_params(
        d_edge_params_view);
    const collection_types<int2>::const_device d_edge_nodes(d_edge_nodes_view);
    const collection_types<int>::const_device d_num_outgoing_edges(
        d_num_outgoing_edges_view);
    const collection_types<int>::const_device d_edge_links(d_edge_links_view);
    collection_types<unsigned char>::device d_num_neighbours(
        d_num_neighbours_view);
    collection_types<int>::device d_neighbours(d_neighbours_view);
    collection_types<int>::device d_reIndexer(d_reIndexer_view);

    const float cut_dphi_max = d_graph_building_params.cut_dphi_max;
    const float cut_dcurv_max = d_graph_building_params.cut_dcurv_max;
    const float cut_tau_ratio_max = d_graph_building_params.cut_tau_ratio_max;
    constexpr float PI_h = 3.141592654f;
    constexpr float PI_2_h = 6.283185307f;
    constexpr float ONE_h = 1.0f;

    const unsigned int edge1_idx = blockIndex * blockSize + threadIndex;
    if (edge1_idx >= nEdges) {
        return;
    }

    const int sharedNode = d_edge_nodes[edge1_idx].x;
    const int link_begin =
        d_num_outgoing_edges[static_cast<unsigned int>(sharedNode)];
    const int nLinks =
        d_num_outgoing_edges[static_cast<unsigned int>(sharedNode + 1)] -
        link_begin;
    if (nLinks == 0) {
        return;
    }

    const half4 params1 = d_edge_params[edge1_idx];
    const float uat_2 = ONE_h / params1.x;
    const float Phi2 = params1.z;
    const float curv2 = params1.y;

    const unsigned int nei_pos = nMaxNei * edge1_idx;

    unsigned char num_nei = 0;
    for (int k = 0; k < nLinks; k++) {
        if (num_nei >= nMaxNei) {
            break;
        }
        const int edge2_idx =
            d_edge_links[static_cast<unsigned int>(link_begin + k)];
        const half4 params2 =
            d_edge_params[static_cast<unsigned int>(edge2_idx)];

        const float tau_ratio = params2.x * uat_2 - ONE_h;
        if (fabsf(tau_ratio) > cut_tau_ratio_max) {
            continue;
        }

        float dPhi = Phi2 - params2.w;
        if (dPhi < -PI_h) {
            dPhi += PI_2_h;
        } else if (dPhi > PI_h) {
            dPhi -= PI_2_h;
        }
        if (fabsf(dPhi) > cut_dphi_max) {
            continue;
        }

        const float dcurv = curv2 - params2.y;
        if (fabsf(dcurv) > cut_dcurv_max) {
            continue;
        }

        d_neighbours[nei_pos + static_cast<unsigned int>(num_nei)] = edge2_idx;
        d_reIndexer[static_cast<unsigned int>(edge2_idx)] = 1;
        ++num_nei;
    }

    d_num_neighbours[edge1_idx] = num_nei;
    if (num_nei != 0) {
        d_reIndexer[edge1_idx] = 1;
        vecmem::device_atomic_ref<unsigned int>(nConnectionsCounter)
            .fetch_add(static_cast<unsigned int>(num_nei));
    }
}

}  // namespace traccc::device
