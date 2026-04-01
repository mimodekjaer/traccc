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
#include "traccc/edm/seed_collection.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void gbts_seed_conversion_kernel(
    const global_index_t global_index,
    const collection_types<int2>::view d_seed_proposals_view,
    const collection_types<char>::view d_seed_ambiguity_view,
    const collection_types<int2>::const_view d_path_store_view,
    const collection_types<int>::const_view d_output_graph_view,
    const edm::seed_collection::view output_seeds,
    const unsigned int nProps, const unsigned int max_num_neighbours);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/gbts_seeding/device/impl/gbts_seed_conversion.ipp"
