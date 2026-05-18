/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/edm/container.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/memory/device_atomic_ref.hpp>

namespace traccc::device {

/// Performs seed disambiguation through seeds bidding to use edges with seed
/// quality.
TRACCC_HOST_DEVICE
inline void add_seed_proposal(
    const int qual, const int path_idx, const unsigned int prop_idx,
    collection_types<char>::view d_seed_ambiguity_view,
    collection_types<int2>::view d_seed_proposals_view,
    collection_types<unsigned long long int>::view d_edge_bids_view,
    const collection_types<int2>::const_view& d_path_store_view,
    char depth = -1) {

    collection_types<char>::device d_seed_ambiguity(d_seed_ambiguity_view);
    collection_types<int2>::device d_seed_proposals(d_seed_proposals_view);
    collection_types<unsigned long long int>::device d_edge_bids(
        d_edge_bids_view);
    const collection_types<int2>::const_device d_path_store(d_path_store_view);

    d_seed_proposals[prop_idx] = make_int2(qual, path_idx);
    d_seed_ambiguity[prop_idx] = 0;

    const unsigned long long int seed_bid =
        (static_cast<unsigned long long int>(qual) << 32) |
        (static_cast<unsigned long long int>(prop_idx));

    int2 path = make_int2(0, d_seed_proposals[prop_idx].y);
    while (path.y >= 0 && depth != 0) {
        path = d_path_store[static_cast<unsigned int>(path.y)];
        depth--;

        vecmem::device_atomic_ref<unsigned long long int> atomic_bid(
            d_edge_bids[static_cast<unsigned int>(path.x)]);
        unsigned long long int old_val = atomic_bid.load();
        while (seed_bid > old_val &&
               !atomic_bid.compare_exchange_strong(old_val, seed_bid)) {
            // old_val updated by compare_exchange_strong on failure
        }
        const unsigned long long int competing_offer = old_val;

        if (competing_offer > seed_bid) {
            d_seed_ambiguity[prop_idx] = -1;
        } else if (competing_offer != 0) {
            d_seed_ambiguity[competing_offer & 0xFFFFFFFFLL] = -1;
        }
    }
}

}  // namespace traccc::device
