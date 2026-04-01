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

#include <cmath>

// TODO not started

namespace traccc::device {
    
    TRACCC_HOST_DEVICE
    inline void graph_edge_matching_kernel(
        const global_index_t global_index,
        const gbts_edge_matching_params d_edge_matching_params,
        const collection_types<float4>::const_view d_edge_params_view, 
        const collection_types<int2>::const_view d_edge_nodes_view,
        const collection_types<int>::const_view d_num_outgoing_edges_view, 
        const collection_types<int>::const_view d_edge_links_view,
        const collection_types<unsigned char>::view d_num_neighbours_view, 
        const collection_types<int>::view d_neighbours_view, 
        const collection_types<int>::view d_reIndexer_view,
        const collection_types<unsigned int>::view d_counters_view, 
        const unsigned int nEdges,
        const unsigned int nMaxNei) {

        if (global_index >= nEdges) {
            return;
        }
        const collection_types<float4>::const_device d_edge_params(d_edge_params_view);
        collection_types<int2>::const_device d_edge_nodes(d_edge_nodes_view);
        const collection_types<int>::const_device d_num_outgoing_edges(d_num_outgoing_edges_view);
        const collection_types<int>::const_device d_edge_links(d_edge_links_view);
        collection_types<unsigned char>::device d_num_neighbours(d_num_neighbours_view);
        collection_types<int>::device d_neighbours(d_neighbours_view);
        collection_types<int>::device d_reIndexer(d_reIndexer_view);
        collection_types<unsigned int>::device d_counters(d_counters_view);
        
        float PI_F = 3.141592654f;
        float PI_2_F = 6.283185307f;
        
        int sharedNode = d_edge_nodes[global_index].x;

        int link_begin = d_num_outgoing_edges[static_cast<unsigned int>(sharedNode)];
        // the number of edges leaving the sharedNode
        int nLinks = d_num_outgoing_edges[static_cast<unsigned int>(sharedNode + 1)] - link_begin;
        if (nLinks == 0) {
            return;
        }
        float4 params1 = d_edge_params[global_index];  // [exp_eta, curv, Phi1, Phi2]

        float uat_2 = 1.0f / params1.x;
        float Phi2 = params1.z;
        float curv2 = params1.y;

        int nei_pos = static_cast<int>(nMaxNei * global_index);
    
        unsigned char num_nei = 0;
    
        for (int k = 0; k < nLinks; k++) {  // loop over potential neighbours
    
            if (num_nei >= nMaxNei) {
                break;
            }
            int edge2_idx = d_edge_links[static_cast<unsigned int>(link_begin + k)];

            float4 params2 = d_edge_params[static_cast<unsigned int>(edge2_idx)];
    
            float tau_ratio = params2.x * uat_2 - 1.0f;
    
            if (fabsf(tau_ratio) > d_edge_matching_params.cut_tau_ratio_max) {  // bad match
                continue;
            }
    
            float dPhi = Phi2 - params2.w;  // Phi2
    
            if (dPhi < -PI_F) {
                dPhi += PI_2_F;
            } else if (dPhi > PI_F) {
                dPhi -= PI_2_F;
            }
            if (fabsf(dPhi) > d_edge_matching_params.cut_dphi_max) {
                continue;
            }
    
            float dcurv = curv2 - params2.y;
    
            if (fabsf(dcurv) > d_edge_matching_params.cut_dcurv_max) {
                continue;
            }
    
            d_neighbours[static_cast<unsigned int>(nei_pos + num_nei)] = edge2_idx;
            d_reIndexer[static_cast<unsigned int>(edge2_idx)] = 1;
    
            ++num_nei;
        }
    
        d_num_neighbours[global_index] = num_nei;
    
        if (num_nei != 0) {
            d_reIndexer[global_index] = 1;
            vecmem::device_atomic_ref<unsigned int> d_counters_1_ref(d_counters[1]);
            d_counters_1_ref.fetch_add(num_nei);
        }
    }

} // namespace traccc::device