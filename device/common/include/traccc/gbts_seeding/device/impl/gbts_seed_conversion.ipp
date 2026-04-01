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
#include "traccc/edm/seed_collection.hpp"


// TODO not started  

namespace traccc::device {


    TRACCC_HOST_DEVICE
    inline void gbts_seed_conversion_kernel(
        const global_index_t global_index,
        const collection_types<int2>::view d_seed_proposals_view, 
        const collection_types<char>::view d_seed_ambiguity_view, 
        const collection_types<int2>::const_view d_path_store_view,
        const collection_types<int>::const_view d_output_graph_view, 
        const edm::seed_collection::view output_seeds,
        const unsigned int nProps, 
        const unsigned int max_num_neighbours) {

        if (global_index >= nProps) {
            return;
        }

        const collection_types<int2>::const_device d_seed_proposals(d_seed_proposals_view);
        const collection_types<char>::const_device d_seed_ambiguity(d_seed_ambiguity_view);
        const collection_types<int2>::const_device d_path_store(d_path_store_view);
        const collection_types<int>::const_device d_output_graph(d_output_graph_view);
    
        int edge_size = static_cast<int>(2 + 1 + max_num_neighbours);
        edm::seed_collection::device seeds_device(output_seeds);

        if (d_seed_ambiguity[global_index] == -2) {
                // drop seeds that lost the bidding
                return;
            }
            Tracklet seed;
            seed.size = 0;
            // dummy path to start the loop
            int2 path = make_int2(0, d_seed_proposals[global_index].y);
            while (path.y >= 0) {
                path = d_path_store[static_cast<unsigned int>(path.y)];
                seed.nodes[seed.size++] =
                    d_output_graph[static_cast<unsigned int>(traccc::device::gbts_consts::node1 +
                                   edge_size * path.x)];
            }
            seed.nodes[seed.size++] =
                d_output_graph[static_cast<unsigned int>(traccc::device::gbts_consts::node2 +
                               edge_size * path.x)];
            // sample begining, middle, end sp from tracklet for now
            seeds_device.push_back({seed.nodes[seed.size - 1],
                                    seed.nodes[(1 + seed.size) / 2 - 1],
                                    seed.nodes[0]});
        }
    

} // namespace traccc::device