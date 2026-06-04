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

/// Each accepted seed bids on its constituent hits.
///
/// Each thread (strided by gridSize) takes one accepted proposal, walks
/// the seed's edges via the compact graph to enumerate hit indices, and
/// atomically updates d_hit_bids_view with its packed seed bid if it
/// outranks the current best for that hit.
///
/// @param[in]  globalIndex                 Current thread index
/// @param[in]  gridSize                    Total thread count (stride)
/// @param[in]  d_output_graph_view         Compact graph from graph_compression
/// @param[in]  d_seed_proposals_view       Per-seed (path index, level)
/// @param[in]  d_path_store_view           Per-path (parent, edge) entries
/// @param[in]  d_seed_ambiguity_view       Per-seed ambiguity tag
/// @param[in,out] d_hit_bids_view          Per-hit highest-bidder seed
/// @param[in]  nProps                      Number of seed proposals
/// @param[in]  edge_size                   Stride in d_output_graph_view
///                                         for one edge record
///
TRACCC_HOST_DEVICE
inline void seeds_bid_for_hits(
    const global_index_t globalIndex, 
    const unsigned int gridSize,
    const collection_types<unsigned int>::const_view& d_output_graph_view,
    const collection_types<int2>::const_view& d_seed_proposals_view,
    const collection_types<int2>::const_view& d_path_store_view,
    const collection_types<char>::const_view& d_seed_ambiguity_view,
    const collection_types<unsigned long long int>::view d_hit_bids_view,
    const unsigned int nProps,
    const unsigned int edge_size);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/seeds_bid_for_hits.ipp"
