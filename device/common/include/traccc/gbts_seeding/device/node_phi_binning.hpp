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

// Math include(s).
#include <cmath>
#include <numbers>

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void node_phi_binning_kernel(
    const global_index_t globalIndex,
    const collection_types<float4>::const_view d_sp_params_view,
    const collection_types<int>::view d_node_phi_index_view,
    const unsigned int nNodesPerBlock, const unsigned int nNodes,
    const unsigned int nPhiBins);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/gbts_seeding/device/impl/node_phi_binning.ipp"
