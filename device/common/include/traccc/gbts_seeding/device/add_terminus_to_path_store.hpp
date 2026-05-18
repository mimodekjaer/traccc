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
inline void add_terminus_to_path_store(
    global_index_t globalIndex, collection_types<int2>::view d_path_store_view,
    const collection_types<short2>::const_view& d_outgoing_paths_view,
    unsigned int nEdges);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/add_terminus_to_path_store.ipp"
