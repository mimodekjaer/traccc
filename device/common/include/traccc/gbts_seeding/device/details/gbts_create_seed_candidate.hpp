/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/containers/data/vector_view.hpp>

namespace traccc::device::details {

/// @brief Let a seed proposal bid for every edge on its path (bid-only pass).
///
/// Packs the seed bid (quality in the high 32 bits, proposal id in the low 32
/// bits) and walks the path-store chain, atomically raising each touched
/// edge's best bid with a fetch_max. No loser marking happens here: the
/// won/lost state is derived afterwards by @c gbts_seed_bid_won on the settled
/// per-edge maxima (separated by a kernel boundary), so the round's outcome is
/// a pure function of the bid set, independent of atomic arrival order.
///
/// @param[in] qual              Fit quality (higher wins)
/// @param[in] path_idx          Leaf entry of the proposal's path-store chain
/// @param[in] prop_idx          Canonical proposal index (bid tie-break)
/// @param[in,out] d_edge_bids_view  Per-edge highest-bid words
/// @param[in] d_path_store_view Per-path (edge, parent) entries
/// @param[in] depth             Maximum walk depth (-1 == unlimited)
///
TRACCC_HOST_DEVICE
inline void gbts_place_seed_bid(
    const int qual, const int path_idx, const unsigned int prop_idx,
    const vecmem::data::vector_view<unsigned long long int> d_edge_bids_view,
    const vecmem::data::vector_view<const int2>& d_path_store_view,
    char depth = -1);

/// @brief Check whether a proposal's bid is the settled winner on every edge
/// of its path.
///
/// Must run after every @c gbts_place_seed_bid of the round has completed
/// (kernel boundary). Bids are unique (the low 32 bits carry the proposal id),
/// so "my bid equals the stored maximum on every walked edge" is exactly the
/// strict-winner condition.
///
/// @param[in] qual              Fit quality used when the bid was placed
/// @param[in] path_idx          Leaf entry of the proposal's path-store chain
/// @param[in] prop_idx          Canonical proposal index used when bidding
/// @param[in] d_edge_bids_view  Per-edge highest-bid words (settled)
/// @param[in] d_path_store_view Per-path (edge, parent) entries
/// @param[in] depth             Maximum walk depth (-1 == unlimited)
/// @return true if the proposal won every edge it bid on
///
TRACCC_HOST_DEVICE
inline bool gbts_seed_bid_won(
    const int qual, const int path_idx, const unsigned int prop_idx,
    const vecmem::data::vector_view<unsigned long long int> d_edge_bids_view,
    const vecmem::data::vector_view<const int2>& d_path_store_view,
    char depth = -1);

/// @brief Register a new seed proposal and let it bid for every edge on its
/// path.
///
/// Records the (quality, path index) tuple at prop_idx, packs the seed
/// bid (quality in high 32 bits, proposal id in low 32 bits), and walks the
/// path-store chain atomically claiming each edge it touches.  Whenever the
/// new bid loses, the proposal's own ambiguity tag is set to -1; whenever it
/// wins, the previous holder's ambiguity tag is set to -1.
///
/// @param[in]  qual                       Fit quality (higher wins)
/// @param[in]  path_idx                   Leaf entry in d_path_store_view
/// @param[in]  prop_idx                   Slot for the new proposal
/// @param[in,out] d_seed_ambiguity_view   Per-seed ambiguity tag
/// @param[in,out] d_seed_proposals_view   Per-seed (quality, path index)
/// @param[in,out] d_edge_bids_view        Per-edge highest-bidder seed
/// @param[in]  d_path_store_view          Per-path (parent, edge) entries
/// @param[in]  depth                      Optional maximum walk depth (-1 ==
/// unlimited)
///
TRACCC_HOST_DEVICE
inline void gbts_create_seed_candidate(
    const int qual, const int path_idx, const unsigned int prop_idx,
    const vecmem::data::vector_view<char> d_seed_ambiguity_view,
    const vecmem::data::vector_view<int2> d_seed_proposals_view,
    const vecmem::data::vector_view<unsigned long long int> d_edge_bids_view,
    const vecmem::data::vector_view<const int2>& d_path_store_view,
    char depth = -1);

}  // namespace traccc::device::details

#include "traccc/gbts_seeding/device/impl/gbts_create_seed_candidate.ipp"
