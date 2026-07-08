/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/concepts/thread_id.hpp"
#include "traccc/gbts_seeding/device/details/gbts_create_seed_candidate.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/containers/device_vector.hpp>
#include <vecmem/memory/device_atomic_ref.hpp>

namespace traccc::device {

// Promote/reject pass (pass A): decide which proposals bid this round and
// initialise their ambiguity to 0, BEFORE any bidding. Each thread only writes
// its own proposal's ambiguity / bid_active here, so there is no cross-proposal
// race. The 0-init is therefore separated from the -1 loser marks done in the
// bid pass below.
template <concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void gbts_rebid_promote_seeds(
    const thread_id_t& thread_id,
    const gbts_rebid_seeds_for_edges_payload& payload) {

    vecmem::device_vector<char> d_seed_ambiguity(payload.seed_ambiguity);
    vecmem::device_vector<char> d_bid_active(payload.bid_active);

    const unsigned int globalIdx = thread_id.getGlobalThreadIdX();
    const unsigned int blockDimX = thread_id.getBlockDimX();
    const unsigned int gridDimX = thread_id.getGridDimX();

    for (unsigned int prop_idx = globalIdx; prop_idx < payload.nProps;
         prop_idx += blockDimX * gridDimX) {

        const char ambi = d_seed_ambiguity[prop_idx];

        if (payload.first_round) {
            if (ambi == 0) {
                // won its depth-1 bid -> rebids this round (init to 0)
                d_seed_ambiguity[prop_idx] = 0;
                d_bid_active[prop_idx] = 1;
            } else {
                // lost its depth-1 bid -> reject
                d_seed_ambiguity[prop_idx] = -2;
                d_bid_active[prop_idx] = 0;
                // count rejected props to calculate nSeeds
                vecmem::device_atomic_ref<unsigned int>(
                    *payload.nRejectedPropsCounter)
                    .fetch_add(1u);
            }
        } else if ((ambi == -2) | (ambi == 0)) {
            // rejected or already-confirmed winner -> does not rebid
            d_bid_active[prop_idx] = 0;
        } else {
            // 'maybe' -> rebids this round (init to 0)
            d_seed_ambiguity[prop_idx] = 0;
            d_bid_active[prop_idx] = 1;
        }
    }
}

// Bid pass (pass B1): every proposal flagged active in the promote pass
// re-bids for all the edges on its path. Bids only -- no loser marking: the
// won/lost state is derived in the mark pass below from the settled per-edge
// maxima, so the greedy resolution is a pure function of the bid set,
// independent of atomic arrival order.
template <concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void gbts_rebid_seeds_for_edges(
    const thread_id_t& thread_id,
    const gbts_rebid_seeds_for_edges_payload& payload) {

    const vecmem::device_vector<const char> d_bid_active(payload.bid_active);
    const vecmem::device_vector<const int2> d_seed_proposals(
        payload.seed_proposals);

    const unsigned int globalIdx = thread_id.getGlobalThreadIdX();
    const unsigned int blockDimX = thread_id.getBlockDimX();
    const unsigned int gridDimX = thread_id.getGridDimX();

    for (unsigned int prop_idx = globalIdx; prop_idx < payload.nProps;
         prop_idx += blockDimX * gridDimX) {

        if (d_bid_active[prop_idx] == 0) {
            continue;
        }
        const int2 prop = d_seed_proposals[prop_idx];

        details::gbts_place_seed_bid(prop.x, prop.y, prop_idx,
                                     payload.edge_bids, payload.path_store, -1);
    }
}

// Mark pass (pass B2): runs after all of pass B1's bids have settled (kernel
// boundary). Each active proposal re-walks its chain and checks whether its
// own bid is the final maximum on every edge; if not, it lost somewhere and
// marks ITSELF -1. Own-slot writes only.
template <concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void gbts_mark_rebid_losers(
    const thread_id_t& thread_id,
    const gbts_rebid_seeds_for_edges_payload& payload) {

    const vecmem::device_vector<const char> d_bid_active(payload.bid_active);
    const vecmem::device_vector<const int2> d_seed_proposals(
        payload.seed_proposals);
    vecmem::device_vector<char> d_seed_ambiguity(payload.seed_ambiguity);

    const unsigned int globalIdx = thread_id.getGlobalThreadIdX();
    const unsigned int blockDimX = thread_id.getBlockDimX();
    const unsigned int gridDimX = thread_id.getGridDimX();

    for (unsigned int prop_idx = globalIdx; prop_idx < payload.nProps;
         prop_idx += blockDimX * gridDimX) {

        if (d_bid_active[prop_idx] == 0) {
            continue;
        }
        const int2 prop = d_seed_proposals[prop_idx];

        if (!details::gbts_seed_bid_won(prop.x, prop.y, prop_idx,
                                        payload.edge_bids, payload.path_store,
                                        -1)) {
            d_seed_ambiguity[prop_idx] = -1;
        }
    }
}

}  // namespace traccc::device
