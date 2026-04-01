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
inline void node_sorting_kernel(
    const int idx,
    const collection_types<float4>::const_view d_sp_params_view,
    const collection_types<int>::const_view d_node_eta_index_view,
    const collection_types<int>::const_view d_node_phi_index_view,
    const collection_types<int>::view d_phi_cusums_view,
    const collection_types<float>::view d_node_params_view,
    const collection_types<int>::view d_node_index_view,
    const collection_types<int>::const_view d_original_sp_idx_view,
    const gbts_graph_width_cuts width_cuts, const unsigned int nPhiBins);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/gbts_seeding/device/impl/node_sorting.ipp"
