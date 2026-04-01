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
inline void count_terminus_edges_kernel(
    const global_index_t global_index,
    const collection_types<short2>::view d_outgoing_paths_view,
    const collection_types<unsigned int>::view d_counters_view);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/gbts_seeding/device/impl/count_terminus_edges.ipp"
