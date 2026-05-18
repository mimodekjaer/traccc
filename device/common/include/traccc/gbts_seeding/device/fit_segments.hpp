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
#include "traccc/gbts_seeding/gbts_seeding_config.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void fit_segments(
    global_index_t globalIndex,
    const collection_types<float4>::const_view& d_sp_reduced_view,
    const collection_types<int>::const_view& d_output_graph_view,
    const collection_types<int2>::const_view& d_path_store_view,
    collection_types<int2>::view d_seed_proposals_view,
    collection_types<unsigned long long int>::view d_edge_bids_view,
    collection_types<char>::view d_seed_ambiguity_view,
    unsigned int nPathStoreSize, unsigned int& nPropsCounter,
    unsigned int nTerminusEdges, int minLevel, unsigned int max_num_neighbours,
    const gbts_seed_extraction_params& seed_extraction_params);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/fit_segments.ipp"
