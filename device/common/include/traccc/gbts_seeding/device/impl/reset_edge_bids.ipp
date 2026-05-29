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
inline void reset_edge_bids(
    const global_index_t globalIndex, const unsigned int gridSize,
    const collection_types<int2>::const_view& d_path_store_view,
    const collection_types<int2>::view d_seed_proposals_view,
    const collection_types<unsigned long long int>::view d_edge_bids_view,
    const collection_types<char>::view d_seed_ambiguity_view,
    const unsigned int nProps, unsigned int& nRejectedPropsCounter,
    const int round) {

    const collection_types<int2>::const_device d_path_store(d_path_store_view);
    collection_types<int2>::device d_seed_proposals(d_seed_proposals_view);
    const collection_types<unsigned long long int>::const_device d_edge_bids(
        d_edge_bids_view);
    collection_types<char>::device d_seed_ambiguity(d_seed_ambiguity_view);

    for (unsigned int prop_idx = globalIndex; prop_idx < nProps;
         prop_idx += gridSize) {

        const char ambi = d_seed_ambiguity[prop_idx];
        if (round == -1) {
            if (ambi == 0) {
                d_seed_ambiguity[prop_idx] = 1;
                continue;
            } else {
                d_seed_ambiguity[prop_idx] = -2;
                vecmem::device_atomic_ref<unsigned int>(nRejectedPropsCounter)
                    .fetch_add(1u);
                continue;
            }
        } else if ((ambi == -2) | (ambi == 0)) {
            continue;
        }
        const int2 prop = d_seed_proposals[prop_idx];

        bool isgood = true;
        int2 path = make_int2(0, prop.y);
        while (path.y >= 0) {
            path = d_path_store[static_cast<unsigned int>(path.y)];
            const unsigned long long int best_bid =
                d_edge_bids[static_cast<unsigned int>(path.x)];
            if (d_seed_ambiguity[static_cast<unsigned int>(
                    best_bid & 0xFFFFFFFFLL)] == 0) {
                isgood = false;
                break;
            }
        }
        if (isgood) {
            d_seed_ambiguity[prop_idx] = 1;
        } else {
            d_seed_ambiguity[prop_idx] = -2;
            vecmem::device_atomic_ref<unsigned int>(nRejectedPropsCounter)
                .fetch_add(1u);
        }
    }
}

}  // namespace traccc::device