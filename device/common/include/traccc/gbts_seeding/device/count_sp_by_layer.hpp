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

// VecMem include(s).
#include <vecmem/memory/device_atomic_ref.hpp>

namespace traccc::device {

TRACCC_HOST_DEVICE
inline void count_sp_by_layer(
    const global_index_t globalIndex,
    const traccc::edm::spacepoint_collection::const_view spacepoints_view,
    const edm::measurement_collection::const_view measurements_view,
    const collection_types<short>::const_view volumeToLayerMap_view,
    const collection_types<std::pair<unsigned int, unsigned int>>::const_view
        surfaceToLayerMap_view,
    const collection_types<char>::const_view layerType_view,
    const collection_types<float4>::view reducedSP_view,
    const collection_types<int>::view layerCounts_view,
    const collection_types<short>::view spacepointsLayer_view,
    const float type1_max_width, const long unsigned int volumeMapSize,
    const long unsigned int surfaceMapSize, const bool doTauCut = true);

}  // namespace traccc::device

// Include the implementation.
#include "traccc/gbts_seeding/device/impl/count_sp_by_layer.ipp"
