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




// TO-DO?: reset prop count each iter and make new props like CCA active_edges
TRACCC_HOST_DEVICE
inline void seeds_rebid_for_edges_kernel(
    const global_index_t global_index,
    const collection_types<int2>::const_view d_path_store_view,
                                      const collection_types<int2>::view d_seed_proposals_view,
                                      const collection_types<unsigned long long int>::view d_edge_bids_view,
                                      const collection_types<char>::view d_seed_ambiguity_view,
                                      const unsigned int nProps) {


    if (global_index >= nProps) {
        return;
    }

    const collection_types<int2>::const_device d_path_store(d_path_store_view);
    collection_types<int2>::device d_seed_proposals(d_seed_proposals_view);
    collection_types<unsigned long long int>::device d_edge_bids(d_edge_bids_view);
    collection_types<char>::device d_seed_ambiguity(d_seed_ambiguity_view);

        char ambi = d_seed_ambiguity[global_index];
        if (ambi == -2 | ambi == 0) {
            // only rebid for maybes
            return;
        }
        int2 prop = d_seed_proposals[global_index];

        add_seed_proposal(prop.x, prop.y, global_index, d_seed_ambiguity_view,
                          d_seed_proposals_view, d_edge_bids_view, d_path_store_view, -1);
    
}



} // namespace traccc::device