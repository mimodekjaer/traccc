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

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void seeds_bid_for_hits(
    global_index_t globalIndex, unsigned int gridSize,
    const collection_types<int>::const_view& d_output_graph_view,
    const collection_types<int2>::const_view& d_seed_proposals_view,
    const collection_types<int2>::const_view& d_path_store_view,
    const collection_types<char>::const_view& d_seed_ambiguity_view,
    collection_types<unsigned long long int>::view d_hit_bids_view,
    unsigned int nProps, int edge_size);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/seeds_bid_for_hits.ipp"
