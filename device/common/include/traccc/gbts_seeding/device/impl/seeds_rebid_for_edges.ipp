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
#include "traccc/gbts_seeding/device/add_seed_proposal.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void seeds_rebid_for_edges(
    const global_index_t globalIndex, const unsigned int gridSize,
    const collection_types<int2>::const_view& d_path_store_view,
    const collection_types<int2>::view d_seed_proposals_view,
    const collection_types<unsigned long long int>::view d_edge_bids_view,
    const collection_types<char>::view d_seed_ambiguity_view,
    const unsigned int nProps) {

    collection_types<char>::device d_seed_ambiguity(d_seed_ambiguity_view);
    collection_types<int2>::device d_seed_proposals(d_seed_proposals_view);

    for (unsigned int prop_idx = globalIndex; prop_idx < nProps;
         prop_idx += gridSize) {
        const char ambi = d_seed_ambiguity[prop_idx];
        if ((ambi == -2) | (ambi == 0)) {
            continue;
        }
        const int2 prop = d_seed_proposals[prop_idx];

        add_seed_proposal(prop.x, prop.y, prop_idx, d_seed_ambiguity_view,
                          d_seed_proposals_view, d_edge_bids_view,
                          d_path_store_view, -1);
    }
}

}  // namespace traccc::device
