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

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void edge_re_indexing(
    global_index_t globalIndex, collection_types<int>::view d_reIndexer_view,
    unsigned int& nConnectedEdges, unsigned int nEdges);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/edge_re_indexing.ipp"
