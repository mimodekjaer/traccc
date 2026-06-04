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
#include "traccc/edm/container.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// System include(s).
#include <cstdint>

namespace traccc::device {

/// Have each surviving seed re-bid for every edge along its path.
///
/// Each thread (strided by gridSize) walks one accepted proposal, then
/// for every edge on its path-store chain atomically compares its packed bid
/// against d_edge_bids_view and replaces the entry if the new bid wins.
///
/// @param[in]  globalIndex                 Current thread index
/// @param[in]  gridSize                    Total thread count (stride)
/// @param[in]  d_path_store_view           Per-path (parent, edge) entries
/// @param[in]  d_seed_proposals_view       Per-seed (path index, level)
/// @param[in,out] d_edge_bids_view         Per-edge highest-bidder seed
/// @param[in]  d_seed_ambiguity_view       Per-seed ambiguity tag
/// @param[in]  nProps                      Number of seed proposals
///
TRACCC_HOST_DEVICE
inline void seeds_rebid_for_edges(
    const global_index_t globalIndex,
    const unsigned int gridSize,
    const collection_types<int2>::const_view& d_path_store_view,
    const collection_types<int2>::view d_seed_proposals_view,
    const collection_types<unsigned long long int>::view d_edge_bids_view,
    const collection_types<char>::view d_seed_ambiguity_view,
    const unsigned int nProps, unsigned int& nRejectedPropsCounter,
    const bool first_round);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/seeds_rebid_for_edges.ipp"
