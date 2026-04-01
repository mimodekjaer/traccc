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


// TODO not finished

namespace traccc::device {

    TRACCC_HOST_DEVICE
    inline void cca_iteration_kernel(
        const global_index_t global_index,
        const collection_types<int>::const_view d_output_graph_view, 
        const collection_types<char>::view d_levels_view, 
        const collection_types<int>::view d_active_edges_view,
        const collection_types<short2>::view d_outgoing_paths_view, 
        const collection_types<unsigned int>::view d_counters_view, 
        const int iter,
        const unsigned int nEdges,
        const unsigned int max_num_neighbours, 
        const int minLevel) {

        const unsigned int toggle = static_cast<unsigned int>(iter % 2);

        collection_types<unsigned int>::device d_counters(d_counters_view);
        unsigned int nEdgesLeft = d_counters[3 + toggle];
        if(global_index >= nEdgesLeft) {
            return;
        }
    
        const collection_types<int>::const_device d_output_graph(d_output_graph_view);
        collection_types<char>::device d_levels(d_levels_view);
        collection_types<int>::device d_active_edges(d_active_edges_view);
        collection_types<short2>::device d_outgoing_paths(d_outgoing_paths_view);
        unsigned int edge_size = 2 + 1 + max_num_neighbours;

        int levelLoad = static_cast<int>(static_cast<unsigned int>(toggle) * nEdges);
        int levelStore = static_cast<int>(static_cast<unsigned int>(1 - toggle) * nEdges);


        int edgeIdx = iter == 0 ? static_cast<int>(global_index) : d_active_edges[global_index];

        int edge_pos = static_cast<int>(edge_size) * edgeIdx;

        int nNei = d_output_graph[static_cast<unsigned int>(edge_pos + traccc::device::gbts_consts::nNei)];

        char next_level = d_levels[static_cast<unsigned int>(levelLoad + edgeIdx)];

        bool localChange = false;
        for (int nIdx = 0; nIdx < nNei;
                nIdx++) {  // loop over neighbouring edges

            int nextEdgeIdx =
                d_output_graph[static_cast<unsigned int>(edge_pos +
                                traccc::device::gbts_consts::nei_start + nIdx)];
            char forward_level = d_levels[static_cast<unsigned int>(levelLoad + nextEdgeIdx)];

            if (next_level == forward_level) {
                next_level = static_cast<char>(forward_level + 1);
                localChange = true;
                break;
            }
        }
        // add all remianing edges to level_views on the last iteration
        if (localChange) {
            if (iter == traccc::device::gbts_consts::max_cca_iter - 1) {
                // shorten paths longer than max_cca_iter
                d_outgoing_paths[static_cast<unsigned int>(edgeIdx)].y = -1;
            } else {

                vecmem::device_atomic_ref<unsigned int> d_counters_4_toggle_ref(d_counters[static_cast<unsigned int>(4 - toggle)]);
                unsigned int edgesLeftPlace = d_counters_4_toggle_ref.fetch_add(1);
                d_active_edges[edgesLeftPlace] =
                    edgeIdx;  // for the next iteration
            }
        } else {
            short out_paths = 0;
            for (int nIdx = 0; nIdx < nNei; ++nIdx) {
                int nextEdgeIdx =
                    d_output_graph[static_cast<unsigned int>(edge_pos +
                                    traccc::device::gbts_consts::nei_start +
                                    nIdx)];
                if (next_level == 1 + d_levels[static_cast<unsigned int>(nextEdgeIdx)]) {
                    // calculate the #d_state_store nodes for segment extraction
                    // starting at this edge
                    out_paths = static_cast<short>(out_paths + 1 + d_outgoing_paths[static_cast<unsigned int>(nextEdgeIdx)].x);
                }
                // flag as not terminus edge
                d_outgoing_paths[static_cast<unsigned int>(nextEdgeIdx)].y = -1;
            }
            // flag as long enough segement to become a seed
            d_outgoing_paths[static_cast<unsigned int>(edgeIdx)] =
                make_short2(out_paths, static_cast<short>((next_level >= minLevel) - 1));
        }
        // store new level and ensure all final
        // levels are on both sides of the array
        d_levels[static_cast<unsigned int>(levelStore + edgeIdx)] = next_level;
    }
    barrier().blockBarrier();

    if (threadIdx.x == 0) {
        if (vecmem::device_atomic_ref<unsigned int>(d_counters[5]).fetch_add(1) == gridDim.x - 1) {
            // this is the last block
            d_counters[3 + toggle] = 0;
            d_counters[5] = 0;
        }
    }











} // namespace traccc::device