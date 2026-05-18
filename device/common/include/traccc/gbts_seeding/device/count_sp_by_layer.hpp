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
#include "traccc/edm/measurement_collection.hpp"
#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/gbts_seeding/gbts_types.hpp"

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void count_sp_by_layer(
    global_index_t globalIndex,
    const traccc::edm::spacepoint_collection::const_view& spacepoints_view,
    const edm::measurement_collection::const_view& measurements_view,
    const collection_types<short>::const_view& volumeToLayerMap_view,
    const collection_types<std::pair<unsigned int, unsigned int>>::const_view&
        surfaceToLayerMap_view,
    const collection_types<char>::const_view& layerType_view,
    collection_types<float4>::view reducedSP_view,
    collection_types<int>::view layerCounts_view,
    collection_types<short>::view spacepointsLayer_view,
    float type1_max_width, unsigned int nSp,
    long unsigned int volumeMapSize, long unsigned int surfaceMapSize,
    bool doTauCut);

}  // namespace traccc::device

#include "traccc/gbts_seeding/device/impl/count_sp_by_layer.ipp"
