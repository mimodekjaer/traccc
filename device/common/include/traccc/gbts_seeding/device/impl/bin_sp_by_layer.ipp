/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2026 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

// Project include(s).
#include "traccc/definitions/qualifiers.hpp"
#include "traccc/device/global_index.hpp"
#include "traccc/edm/container.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

// VecMem include(s).
#include <vecmem/memory/device_atomic_ref.hpp>

// System include(s).
#include <climits>

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void bin_sp_by_layer(
    const global_index_t globalIndex,
    const collection_types<float4>::view sp_params_view,
    const collection_types<float4>::const_view& reducedSP_view,
    const collection_types<unsigned int>::view layerCounts_view,
    const collection_types<short>::const_view& spacepointsLayer_view,
    const collection_types<unsigned int>::view original_sp_idx_view,
    const unsigned int nSp) {

    if (globalIndex >= nSp) {
        return;
    }
    const collection_types<float4>::const_device reducedSP(reducedSP_view);
    const collection_types<short>::const_device spacepointsLayer(
        spacepointsLayer_view);
    collection_types<unsigned int>::device layerCounts(layerCounts_view);
    collection_types<float4>::device sp_params(sp_params_view);
    collection_types<unsigned int>::device original_sp_idx(original_sp_idx_view);

    const float4 sp = reducedSP[globalIndex];
    if (sp.w < -CHAR_MAX) {
        return;
    }
    const short layerIdx = spacepointsLayer[globalIndex];
    const unsigned int binedIdx = 
        vecmem::device_atomic_ref<unsigned int>(layerCounts[layerIdx])
        .fetch_sub(1) - 1;
    original_sp_idx[binedIdx] = globalIndex;
    sp_params[binedIdx] = sp;
}

}  // namespace traccc::device
