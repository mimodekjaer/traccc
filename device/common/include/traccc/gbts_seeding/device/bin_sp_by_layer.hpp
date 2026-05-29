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

/// Scatter reduced spacepoints into layer-ordered slots.
///
/// Each thread takes one spacepoint, atomically decrements the per-layer
/// write cursor in layerCounts_view, and copies the reduced (x, y, z, r)
/// tuple plus the original SP index into the layer-ordered output.
///
/// @param[in]  globalIndex            Current thread index
/// @param[out] sp_params_view         Layer-ordered (x, y, z, r) per SP
/// @param[in]  reducedSP_view         Reduced SP info from @c count_sp_by_layer
/// @param[in,out] layerCounts_view    Per-layer running write cursor
/// @param[in]  spacepointsLayer_view  GBTS layer assignment per SP
/// @param[out] original_sp_idx_view   Layer-ordered → original SP index map
/// @param[in]  nSp                    Number of spacepoints in the event
///
TRACCC_HOST_DEVICE
inline void bin_sp_by_layer(
    const global_index_t globalIndex,
    const collection_types<float4>::view sp_params_view,
    const collection_types<float4>::const_view& reducedSP_view,
    const collection_types<int>::view layerCounts_view,
    const collection_types<short>::const_view& spacepointsLayer_view,
    const collection_types<int>::view original_sp_idx_view, unsigned int nSp);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/bin_sp_by_layer.ipp"
