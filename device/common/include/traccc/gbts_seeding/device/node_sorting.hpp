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
inline void node_sorting(
    global_index_t globalIndex,
    const collection_types<float4>::const_view& d_sp_params_view,
    const collection_types<int>::const_view& d_node_eta_index_view,
    const collection_types<int>::const_view& d_node_phi_index_view,
    collection_types<int>::view d_phi_cusums_view,
    collection_types<float>::view d_node_params_view,
    collection_types<int>::view d_node_index_view,
    const collection_types<int>::const_view& d_original_sp_idx_view,
    const gbts_graph_building_params& ap, unsigned int nNodes,
    unsigned int nPhiBins);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/node_sorting.ipp"
