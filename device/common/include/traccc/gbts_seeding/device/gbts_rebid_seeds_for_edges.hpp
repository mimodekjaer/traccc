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
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

// System include(s).
#include <cstdint>

namespace traccc::device {

/// (Global Event Data) Payload for the @c
/// traccc::device::gbts_rebid_seeds_for_edges function
struct gbts_rebid_seeds_for_edges_payload {
    /// Number of seed proposals
    unsigned int nProps;
    /// Per-path (edge index, parent path-store index or -1) entries
    vecmem::data::vector_view<const int2> path_store;
    /// Per-seed-proposal (path_store index, level)
    vecmem::data::vector_view<int2> seed_proposals;
    /// In/out: per-edge highest-bidder seed proposal (cleared on entry)
    vecmem::data::vector_view<unsigned long long int> edge_bids;
    /// In/out: per-seed-proposal ambiguity tag
    vecmem::data::vector_view<char> seed_ambiguity;
    /// Out (promote pass) / in (bid pass): per-proposal flag marking the
    /// proposals that bid this round. Written by gbts_rebid_promote_seeds and
    /// read by gbts_rebid_seeds_for_edges, so the bid pass never reads the
    /// ambiguity it is concurrently marking.
    vecmem::data::vector_view<char> bid_active;
    /// In/out: global atomic counter of rejected proposals
    unsigned int* nRejectedPropsCounter;
    /// True on the first bidding round (folds the init pass)
    bool first_round;
};

/// @brief Promote/reject pass run before @c gbts_rebid_seeds_for_edges.
///
/// Decides, per proposal, whether it bids this round: it sets bid_active and
/// initialises that proposal's ambiguity to 0 (so the subsequent bid pass only
/// ever writes the -1 loser marks, never the 0 init -- removing the init/mark
/// race). On the first round it also rejects (-2) and counts the proposals that
/// lost their depth-1 bid. State transitions are identical to the original
/// fused kernel; only the init is separated into this pass.
///
/// @param[in] thread_id Thread identifier for the kernel launch
/// @param[in] payload   The global memory payload
///
template <concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void gbts_rebid_promote_seeds(
    const thread_id_t& thread_id,
    const gbts_rebid_seeds_for_edges_payload& payload);

/// @brief Have one surviving seed re-bid for every edge along its path.
///
/// Processes one proposal (the grid-stride loop lives in the kernel wrapper):
/// for every edge on its path-store chain it atomically compares its packed
/// bid against the edge bids and replaces the entry if the new bid wins.
///
/// @param[in] thread_id Thread identifier for the kernel launch
/// @param[in] payload   The global memory payload
///
template <concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void gbts_rebid_seeds_for_edges(
    const thread_id_t& thread_id,
    const gbts_rebid_seeds_for_edges_payload& payload);

/// @brief Mark the active proposals that lost their (settled) re-bid.
///
/// Runs after @c gbts_rebid_seeds_for_edges has completed (kernel boundary):
/// each active proposal checks whether its own bid is the final maximum on
/// every edge of its chain and sets its own ambiguity to -1 if it was outbid
/// anywhere. Own-slot writes only, so the round's outcome is a pure function
/// of the settled bids, independent of any atomic ordering.
///
/// @param[in] thread_id Thread identifier for the kernel launch
/// @param[in] payload   The global memory payload
///
template <concepts::thread_id1 thread_id_t>
TRACCC_HOST_DEVICE inline void gbts_mark_rebid_losers(
    const thread_id_t& thread_id,
    const gbts_rebid_seeds_for_edges_payload& payload);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/gbts_rebid_seeds_for_edges.ipp"
