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
inline void bin_sp_by_layer(
    const global_index_t globalIndex,
    const collection_types<float4>::view sp_params_view,
    const collection_types<float4>::const_view reducedSP_view,
    const collection_types<int>::view layerCounts_view,
    const collection_types<short>::const_view spacepointsLayer_view,
    const collection_types<int>::view original_sp_idx_view);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/gbts_seeding/device/impl/bin_sp_by_layer.ipp"
