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


// TODO not started

namespace traccc::device {

    TRACCC_HOST_DEVICE
    inline void fit_segments_kernel(
        const global_index_t global_index,
        const collection_types<float4>::const_view d_sp_reduced_view, 
        const collection_types<int>::const_view d_output_graph_view, 
        const collection_types<int2>::const_view d_path_store_view,
        const collection_types<int2>::view d_seed_proposals_view, 
        const collection_types<unsigned long long int>::view d_edge_bids_view,
        const collection_types<char>::view d_seed_ambiguity_view, 
        const collection_types<unsigned int>::view d_counters_view,
        const unsigned int nTerminusEdges, 
        const int minLevel, 
        const unsigned int max_num_neighbours,
        const gbts_seed_extraction_params seed_extraction_params) {
            
        const collection_types<float4>::const_device d_sp_reduced(d_sp_reduced_view);
        const collection_types<int>::const_device d_output_graph(d_output_graph_view);
        const collection_types<int2>::const_device d_path_store(d_path_store_view);
        collection_types<int2>::device d_seed_proposals(d_seed_proposals_view);
        collection_types<unsigned long long int>::device d_edge_bids(d_edge_bids_view);
        collection_types<char>::device d_seed_ambiguity(d_seed_ambiguity_view);
        collection_types<unsigned int>::device d_counters(d_counters_view);
    
        // take an extracted path and fit it to produce a quality score
        unsigned int path_idx = global_index + nTerminusEdges;
        if (path_idx >= d_counters[7]) {
            return;
        }
        int edge_size = static_cast<int>(2 + 1 + max_num_neighbours);

        char length = 1;

        bool toggle = false;
        edgeState state1;
        edgeState state2;

        int2 path = d_path_store[path_idx];

        const int nodeidx1 =
            d_output_graph[static_cast<unsigned int>(traccc::device::gbts_consts::node1 + edge_size * path.x)];
        float4 node1 = d_sp_reduced[static_cast<unsigned int>(nodeidx1)];
        const int nodeidx2 =
            d_output_graph[static_cast<unsigned int>(traccc::device::gbts_consts::node2 + edge_size * path.x)];
        float4 node2 = d_sp_reduced[static_cast<unsigned int>(nodeidx2)];

        state1.initialize(node2, node1);
        while (path.y >= 0) {
            path = d_path_store[static_cast<unsigned int>(path.y)];

            node2 = d_sp_reduced[static_cast<unsigned int>(d_output_graph[static_cast<unsigned int>(traccc::device::gbts_consts::node2 +
                                               edge_size * path.x)])];
            if (toggle) {
                if (!update(&state1, &state2, node2, seed_extraction_params)) {
                    state1 = state2;
                    break;
                }
            } else if (!update(&state2, &state1, node2, seed_extraction_params)) {
                break;
            }
            toggle = !toggle;
            length++;
        }
        // only keep long enough seeds
        if (length < minLevel) {
            return;
        }
        int qual = 0;
        if (toggle) {
            qual = static_cast<int>(seed_extraction_params.qual_scale * state2.m_J);
        } else {
            qual = static_cast<int>(seed_extraction_params.qual_scale * state1.m_J);
        }
        vecmem::device_atomic_ref<unsigned int> d_counters_8_ref(d_counters[8]);
        int prop_idx = static_cast<int>(d_counters_8_ref.fetch_add(1));
        // perform first round of bidding for disambiguation
        // only on the outermost edge
        add_seed_proposal(qual, path_idx, prop_idx, d_seed_ambiguity_view,
                          d_seed_proposals_view, d_edge_bids_view, d_path_store_view, 1);
    }



} // namespace traccc::device